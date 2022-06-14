// Copyright (c) 2022, Compiler Explorer Authors
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright notice,
//       this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

import _ from 'underscore';
import * as monaco from 'monaco-editor';
import {Container} from 'golden-layout';

import {MonacoPane} from './pane';
import {IrState} from './ir-view.interfaces';
import {MonacoPaneState} from './pane.interfaces';

import {ga} from '../analytics';
import {extendConfig} from '../monaco-config';
import {applyColours} from '../colour';

import {Hub} from '../hub';

import * as utils from '../utils';

import {LLVMOptPipelineOutput, OutputLine, Pass} from '../../types/compilation/llvm-opt-pipeline-output.interfaces';
import TomSelect from 'tom-select';

import scrollIntoView from 'scroll-into-view-if-needed';

export class LLVMOptPipeline extends MonacoPane<monaco.editor.IStandaloneDiffEditor, IrState> {
    results: LLVMOptPipelineOutput = {};
    irCode: any[] = [];
    passesColumn: JQuery;
    passesList: JQuery;
    body: JQuery;
    clickCallback: (e: JQuery.ClickEvent) => void;
    keydownCallback: (e: JQuery.KeyDownEvent) => void;
    isPassListSelected = false;
    functionSelector: TomSelect;
    selectedFunction = '';
    originalModel: any;
    modifiedModel: any;

    constructor(hub: Hub, container: Container, state: IrState & MonacoPaneState) {
        super(hub, container, state);
        this.passesColumn = this.domRoot.find('.passes-column');
        this.passesList = this.domRoot.find('.passes-list');
        this.body = this.domRoot.find('.llvm-opt-pipeline-body');
        const selector = this.domRoot.get()[0].getElementsByClassName('function-selector')[0];
        if (!(selector instanceof HTMLSelectElement)) {
            throw new Error('.function-selector is not an HTMLSelectElement');
        }
        /*const instance = new TomSelect(selector, {
            sortField: 'name',
            valueField: 'id',
            labelField: 'name',
            searchField: ['name'],
            options: [
                {id: 1, name: 'Assembly'},
                {id: 2, name: 'Compiler stdout'},
                {id: 3, name: 'Compiler stderr'},
                {id: 4, name: 'Execution stdout'},
                {id: 5, name: 'Execution stderr'},
                {id: 6, name: 'GNAT Expanded Code'},
                {id: 7, name: 'GNAT Tree Code'},
            ],
            items: [],
            render: {
                option: (item, escape) => {
                    return `<div>${escape(item.name)}</div>`;
                },
            },
            dropdownParent: 'body',
            plugins: ['input_autogrow'],
            onChange: value => {
                //if (picker.classList.contains('lhsdifftype')) {
                //    this.lhs.difftype = parseInt(value as any as string);
                //    this.lhs.refresh();
                //} else {
                //    this.rhs.difftype = parseInt(value as any as string);
                //    this.rhs.refresh();
                //}
                //this.updateState();
            },
        });*/
        this.functionSelector = new TomSelect(selector, {
            valueField: 'value',
            labelField: 'title',
            searchField: ['title'],
            dropdownParent: 'body',
            plugins: ['input_autogrow'],
            sortField: 'title',
            onChange: e => this.onFnChange(e as any as string),
        });
        //this.functionSelector.addOption({
        //    title: "Test",
        //    value: "x"
        //});
        //this.functionSelector.addOption({
        //    title: "TestTestTestTestTestTestTestTestTestTestTestTestTestTestTestTestTestTestTestTestTestTestTestTest",
        //    value: "y"
        //});
        //this.functionSelector.addOption({
        //    title: "alpha",
        //    value: "a"
        //});
        this.clickCallback = this.onClickCallback.bind(this);
        this.keydownCallback = this.onKeydownCallback.bind(this);
        $(document).on('click', this.clickCallback);
        $(document).on('keydown', this.keydownCallback);
    }

    override getInitialHTML(): string {
        return $('#llvm-opt-pipeline').html();
    }

    override createEditor(editorRoot: HTMLElement): monaco.editor.IStandaloneDiffEditor {
        const editor = monaco.editor.createDiffEditor(
            editorRoot,
            extendConfig({
                language: 'llvm-ir',
                readOnly: true,
                glyphMargin: true,
                lineNumbersMinChars: 3,
            })
        );
        this.originalModel = monaco.editor.createModel('', 'llvm-ir');
        this.modifiedModel = monaco.editor.createModel('', 'llvm-ir');
        editor.setModel({original: this.originalModel, modified: this.modifiedModel});
        return editor;
    }

    override registerOpeningAnalyticsEvent(): void {
        ga.proxy('send', {
            hitType: 'event',
            eventCategory: 'OpenViewPane',
            eventAction: 'LLVMOptPipelineView',
        });
    }

    override getDefaultPaneName(): string {
        return 'LLVM Opt Pipeline Viewer';
    }

    override registerEditorActions(): void {
        this.editor.addAction({
            id: 'viewsource',
            label: 'Scroll to source',
            keybindings: [monaco.KeyMod.CtrlCmd | monaco.KeyCode.F10],
            contextMenuGroupId: 'navigation',
            contextMenuOrder: 1.5,
            run: editor => {
                const position = editor.getPosition();
                if (position != null) {
                    const desiredLine = position.lineNumber - 1;
                    const source = this.irCode[desiredLine].source;
                    if (source !== null && source.file !== null) {
                        this.eventHub.emit(
                            'editorLinkLine',
                            this.compilerInfo.editorId as number,
                            source.line,
                            -1,
                            -1,
                            true
                        );
                    }
                }
            },
        });
    }

    override registerCallbacks(): void {
        super.registerCallbacks();
        this.paneRenaming.on('renamePane', this.updateState.bind(this));

        this.eventHub.emit('llvmOptPipelineViewOpened', this.compilerInfo.compilerId);
        this.eventHub.emit('requestSettings');
    }

    override onCompileResult(compilerId: number, compiler: any, result: any): void {
        //console.log(compilerId, compiler, result);
        if (this.compilerInfo.compilerId !== compilerId) return;
        if (result.hasLLVMOptPipelineOutput) {
            this.updateResults(result.llvmOptPipelineOutput as LLVMOptPipelineOutput);
        } else if (compiler.supportsLLVMOptPipelineView) {
            //this.showIrResults([{text: '<No output>'}]); // TODO
        }
    }

    override onCompiler(compilerId: number, compiler: any, options: unknown, editorId: number, treeId: number): void {
        if (this.compilerInfo.compilerId !== compilerId) return;
        this.compilerInfo.compilerName = compiler ? compiler.name : '';
        this.compilerInfo.editorId = editorId;
        this.compilerInfo.treeId = treeId;
        this.updateTitle();
        if (compiler && !compiler.supportsLLVMOptPipelineView) {
            //this.editor.setValue('<LLVM IR output is not supported for this compiler>');
        }
    }

    updateResults(results: LLVMOptPipelineOutput): void {
        this.results = results;
        //const functions = Object.keys(result);
        this.functionSelector.clearOptions();
        this.functionSelector.clearActiveOption();
        const keys = Object.keys(results);
        if (keys.length === 0) {
            this.functionSelector.addOption({
                title: '<No functions available>',
                value: '<No functions available>',
            });
        }
        for (const fn of keys) {
            this.functionSelector.addOption({
                title: fn,
                value: fn,
            });
        }

        //this.irCode = result;
        /*this.editor.getModel()?.setValue(result.length ? _.pluck(result, 'text').join('\n') : '<No LLVM IR generated>');

        if (!this.isAwaitingInitialResults) {
            if (this.selection) {
                this.editor.setSelection(this.selection);
                this.editor.revealLinesInCenter(this.selection.startLineNumber, this.selection.endLineNumber);
            }
            this.isAwaitingInitialResults = true;
        }*/
        //const x = result[Object.keys(result)[0]][0];
        //this.editor.getModel()?.setValue(x.before.map(y => y.text).join("\n") + x.after.map(y => y.text).join("\n"));
    }

    onFnChange(name: string) {
        this.selectedFunction = name;
        const passes = this.results[name];
        this.passesList.empty();
        let isFirstMachinePass = true;
        for (const [i, pass] of passes.entries()) {
            let className = pass.irChanged ? 'changed' : '';
            if (pass.machine && isFirstMachinePass) {
                className += ' firstMachinePass';
                isFirstMachinePass = false;
            }
            this.passesList.append(`<div data-i="${i}" class="pass ${className}">${_.escape(pass.name)}</div>`);
        }
        const passDivs = this.passesList.find('.pass');
        passDivs.on('click', e => {
            const target = e.target;
            this.passesList.find('.active').removeClass('active');
            $(target).addClass('active');
            this.displayPass(parseInt(target.getAttribute('data-i') as string));
        });
        this.resize(); // pass list width may change
    }

    displayPass(i: number) {
        const pass = this.results[this.selectedFunction][i];
        console.log(pass);
        const before = pass.before.map(x => x.text).join('\n');
        const after = pass.after.map(x => x.text).join('\n');
        console.log(this.editor.getModel());
        console.log(this.editor.getModel()?.original);
        this.editor.getModel()?.original.setValue(before);
        this.editor.getModel()?.modified.setValue(after);
    }

    onClickCallback(e: JQuery.ClickEvent) {
        this.isPassListSelected = (this.passesList.get(0) as HTMLElement).contains(e.target);
    }

    onKeydownCallback(e: JQuery.KeyDownEvent) {
        if (this.isPassListSelected) {
            if (e.key === 'ArrowUp') {
                e.preventDefault();
                const active = this.passesList.find('.active');
                const prev = active.prev().get(0);
                if (prev) {
                    active.removeClass('active');
                    $(prev).addClass('active');
                    scrollIntoView(prev, {
                        scrollMode: 'if-needed',
                        block: 'nearest',
                    });
                    this.displayPass(parseInt(prev.getAttribute('data-i') as string));
                }
            }
            if (e.key === 'ArrowDown') {
                e.preventDefault();
                const active = this.passesList.find('.active');
                const next = active.next().get(0);
                if (next) {
                    active.removeClass('active');
                    $(next).addClass('active');
                    scrollIntoView(next, {
                        scrollMode: 'if-needed',
                        block: 'nearest',
                    });
                    this.displayPass(parseInt(next.getAttribute('data-i') as string));
                }
            }
        }
    }

    override resize() {
        _.defer(() => {
            const topBarHeight = utils.updateAndCalcTopBarHeight(this.domRoot, this.topBar, this.hideable);
            this.editor.layout({
                width: (this.domRoot.width() as number) - (this.passesColumn.width() as number),
                height: (this.domRoot.height() as number) - topBarHeight,
            });
            (this.body.get(0) as HTMLElement).style.height = (this.domRoot.height() as number) - topBarHeight + 'px';
        });
    }

    override close(): void {
        $(document).off('click', this.clickCallback);
        $(document).off('keydown', this.keydownCallback);
        this.eventHub.unsubscribe();
        this.eventHub.emit('llvmOptPipelineViewClosed', this.compilerInfo.compilerId);
        this.editor.dispose();
    }
}
