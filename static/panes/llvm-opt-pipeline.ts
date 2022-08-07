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

import $ from 'jquery';
import _ from 'underscore';
import * as monaco from 'monaco-editor';
import {Container} from 'golden-layout';
import TomSelect from 'tom-select';
import scrollIntoView from 'scroll-into-view-if-needed';

import {MonacoPane} from './pane';
import {LLVMOptPipelineViewState} from './llvm-opt-pipeline.interfaces';
import {MonacoPaneState} from './pane.interfaces';

import {ga} from '../analytics';
import {extendConfig} from '../monaco-config';
import {Hub} from '../hub';
import * as utils from '../utils';
import {Toggles} from '../widgets/toggles';

import {
    LLVMOptPipelineBackendOptions,
    LLVMOptPipelineOutput,
} from '../../types/compilation/llvm-opt-pipeline-output.interfaces';

const MIN_SIDEBAR_WIDTH = 100;

export class LLVMOptPipeline extends MonacoPane<monaco.editor.IStandaloneDiffEditor, LLVMOptPipelineViewState> {
    results: LLVMOptPipelineOutput = {};
    passesColumn: JQuery;
    passesList: JQuery;
    passesColumnResizer: JQuery;
    body: JQuery;
    clickCallback: (e: JQuery.ClickEvent) => void;
    keydownCallback: (e: JQuery.KeyDownEvent) => void;
    isPassListSelected = false;
    functionSelector: TomSelect;
    originalModel: any;
    modifiedModel: any;
    options: Toggles;
    state: LLVMOptPipelineViewState;
    lastOptions: LLVMOptPipelineBackendOptions = {
        fullModule: false,
        noDiscardValueNames: true,
        demangle: true,
        libraryFunctions: true,
    };
    resizeStartX: number;
    resizeStartWidth: number;
    resizeDragMoveBind: (e: MouseEvent) => void;
    resizeDragEndBind: (e: MouseEvent) => void;
    firstResults = true;

    constructor(hub: Hub, container: Container, state: LLVMOptPipelineViewState & MonacoPaneState) {
        super(hub, container, state);
        this.passesColumn = this.domRoot.find('.passes-column');
        this.passesList = this.domRoot.find('.passes-list');
        this.body = this.domRoot.find('.llvm-opt-pipeline-body');
        if (state.sidebarWidth === 0) {
            _.defer(() => {
                state.sidebarWidth = parseInt(
                    (document.defaultView as Window).getComputedStyle(this.passesColumn.get()[0]).width,
                    10
                );
                state.sidebarWidth = Math.max(state.sidebarWidth, MIN_SIDEBAR_WIDTH);
                this.resize();
                this.updateState();
            });
        } else {
            state.sidebarWidth = Math.max(state.sidebarWidth, MIN_SIDEBAR_WIDTH);
            this.passesColumn.get()[0].style.width = state.sidebarWidth + 'px';
        }
        this.state = state;
        const selector = this.domRoot.get()[0].getElementsByClassName('function-selector')[0];
        if (!(selector instanceof HTMLSelectElement)) {
            throw new Error('.function-selector is not an HTMLSelectElement');
        }
        this.functionSelector = new TomSelect(selector, {
            valueField: 'value',
            labelField: 'title',
            searchField: ['title'],
            dropdownParent: 'body',
            plugins: ['input_autogrow'],
            sortField: 'title',
            onChange: e => this.selectFunction(e as any as string),
        });
        this.clickCallback = this.onClickCallback.bind(this);
        this.keydownCallback = this.onKeydownCallback.bind(this);
        $(document).on('click', this.clickCallback);
        $(document).on('keydown', this.keydownCallback);
        this.eventHub.emit('llvmOptPipelineViewOpened', this.compilerInfo.compilerId);
        this.eventHub.emit('requestSettings');
        this.emitOptions(true);
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

    override registerButtons(state: LLVMOptPipelineViewState) {
        super.registerButtons(state);
        this.options = new Toggles(this.domRoot.find('.options'), state as unknown as Record<string, boolean>);
        this.options.on('change', this.onOptionsChange.bind(this));

        this.passesColumnResizer = this.domRoot.find('.passes-column-resizer');
        this.passesColumnResizer.get()[0].addEventListener('mousedown', this.initResizeDrag.bind(this), false);
    }

    initResizeDrag(e: MouseEvent) {
        // taken from SO
        this.resizeStartX = e.clientX;
        // (this.passesColumn.width() as number)
        // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
        this.resizeStartWidth = parseInt(document.defaultView!.getComputedStyle(this.passesColumn.get()[0]).width, 10);
        this.resizeDragMoveBind = this.resizeDragMove.bind(this);
        this.resizeDragEndBind = this.resizeDragEnd.bind(this);
        document.documentElement.addEventListener('mousemove', this.resizeDragMoveBind, false);
        document.documentElement.addEventListener('mouseup', this.resizeDragEndBind, false);
    }

    resizeDragMove(e: MouseEvent) {
        let width = this.resizeStartWidth + e.clientX - this.resizeStartX;
        if (width < MIN_SIDEBAR_WIDTH) {
            width = MIN_SIDEBAR_WIDTH;
        }
        this.passesColumn.get()[0].style.width = width + 'px';
        this.state.sidebarWidth = width;
        this.resize();
    }

    resizeDragEnd(e: MouseEvent) {
        document.documentElement.removeEventListener('mousemove', this.resizeDragMoveBind, false);
        document.documentElement.removeEventListener('mouseup', this.resizeDragEndBind, false);
    }

    emitOptions(force = false) {
        const options = this.options.get();
        // TODO: Make use of filter-inconsequential-passes on the back end? Maybe provide a specific function arg to
        // the backend? Would be a data transfer optimization.
        const newOptions: LLVMOptPipelineBackendOptions = {
            //'filter-inconsequential-passes': options['filter-inconsequential-passes'],
            fullModule: options['dump-full-module'],
            noDiscardValueNames: options['-fno-discard-value-names'],
            demangle: options['demangle-symbols'],
            libraryFunctions: options['library-functions'],
        };
        let changed = false;
        for (const k in newOptions) {
            if (newOptions[k] !== this.lastOptions[k]) {
                changed = true;
            }
        }
        this.lastOptions = newOptions;
        if (changed || force) {
            this.eventHub.emit('llvmOptPipelineViewOptionsUpdated', this.compilerInfo.compilerId, newOptions, true);
        }
    }

    onOptionsChange() {
        // Redo pass sidebar
        this.selectFunction(this.state.selectedFunction);
        // Inform compiler of the options
        this.emitOptions();
    }

    override onCompileResult(compilerId: number, compiler: any, result: any): void {
        if (this.compilerInfo.compilerId !== compilerId) return;
        if (result.hasLLVMOptPipelineOutput) {
            this.updateResults(result.llvmOptPipelineOutput as LLVMOptPipelineOutput);
        } else if (compiler.supportsLLVMOptPipelineView) {
            this.updateResults({});
            this.editor.getModel()?.original.setValue('<Error>');
            this.editor.getModel()?.modified.setValue('');
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
        let selectedFunction = this.state.selectedFunction; // one of the .clear calls below will end up resetting this
        this.functionSelector.clear();
        this.functionSelector.clearOptions();
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
        this.passesList.empty();
        if (keys.length > 0) {
            if (selectedFunction === '' || !(selectedFunction in results)) {
                selectedFunction = keys[0];
            }
            this.functionSelector.setValue(selectedFunction);
        } else {
            // restore this.selectedFunction, next time the compilation results aren't errors the selected function will
            // still be the same
            this.state.selectedFunction = selectedFunction;
        }
    }

    selectFunction(name: string) {
        this.state.selectedFunction = name;
        if (!(name in this.results)) {
            return;
        }
        const filterInconsequentialPasses = this.options.get()['filter-inconsequential-passes'];
        const passes = this.results[name];
        this.passesList.empty();
        let isFirstMachinePass = true;
        for (const [i, pass] of passes.entries()) {
            if (filterInconsequentialPasses && !pass.irChanged) {
                continue;
            }
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
        // try to select a pass
        if (this.state.selectedIndex >= passes.length) {
            this.state.selectedIndex = 0;
        }
        const selectedPassDiv = this.passesList.find(`[data-i=${this.state.selectedIndex}]`);
        selectedPassDiv.addClass('active');
        // displayPass updates state
        this.displayPass(this.state.selectedIndex);
        // if loading from a url center the active pass
        if (this.firstResults) {
            this.firstResults = false;
            const activePass = this.passesList.find('.active').get(0);
            if (activePass) {
                scrollIntoView(activePass, {
                    scrollMode: 'if-needed',
                    block: 'center',
                });
            }
        }
    }

    displayPass(i: number) {
        if (this.state.selectedFunction in this.results && i < this.results[this.state.selectedFunction].length) {
            this.state.selectedIndex = i;
            const pass = this.results[this.state.selectedFunction][i];
            const before = pass.before.map(x => x.text).join('\n');
            const after = pass.after.map(x => x.text).join('\n');
            this.editor.getModel()?.original.setValue(before);
            this.editor.getModel()?.modified.setValue(after);
            this.updateState();
        }
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

    override getCurrentState() {
        return {
            ...this.options.get(),
            ...super.getCurrentState(),
            selectedFunction: this.state.selectedFunction,
            selectedIndex: this.state.selectedIndex,
            sidebarWidth: this.state.sidebarWidth,
        };
    }

    override resize() {
        _.defer(() => {
            const topBarHeight = utils.updateAndCalcTopBarHeight(this.domRoot, this.topBar, this.hideable);
            const otherWidth = (this.passesColumn.width() as number) + (this.passesColumnResizer.width() as number);
            const domWidth = this.domRoot.width() as number;
            if (otherWidth > domWidth) {
                this.passesColumn.get()[0].style.width = domWidth + 'px';
            }
            this.editor.layout({
                width: domWidth - otherWidth,
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
