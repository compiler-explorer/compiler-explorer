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

export class LLVMOptPipeline extends MonacoPane<monaco.editor.IStandaloneCodeEditor, IrState> {
    irCode: any[] = [];
    passesColumn: JQuery;

    constructor(hub: Hub, container: Container, state: IrState & MonacoPaneState) {
        super(hub, container, state);
        this.passesColumn = this.domRoot.find('.passes-column');
    }

    override getInitialHTML(): string {
        return $('#llvm-opt-pipeline').html();
    }

    override createEditor(editorRoot: HTMLElement): monaco.editor.IStandaloneCodeEditor {
        return monaco.editor.create(
            editorRoot,
            extendConfig({
                language: 'llvm-ir',
                readOnly: true,
                glyphMargin: true,
                lineNumbersMinChars: 3,
                automaticLayout: true,
            })
        );
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
        if (result.hasIrOutput) {
            this.showIrResults(result.irOutput);
        } else if (compiler.supportsIrView) {
            this.showIrResults([{text: '<No output>'}]);
        }
    }

    override onCompiler(compilerId: number, compiler: any, options: unknown, editorId: number, treeId: number): void {
        if (this.compilerInfo.compilerId !== compilerId) return;
        this.compilerInfo.compilerName = compiler ? compiler.name : '';
        this.compilerInfo.editorId = editorId;
        this.compilerInfo.treeId = treeId;
        this.updateTitle();
        if (compiler && !compiler.supportsIrView) {
            this.editor.setValue('<LLVM IR output is not supported for this compiler>');
        }
    }

    showIrResults(result: any[]): void {
        this.irCode = result;
        this.editor.getModel()?.setValue(result.length ? _.pluck(result, 'text').join('\n') : '<No LLVM IR generated>');

        if (!this.isAwaitingInitialResults) {
            if (this.selection) {
                this.editor.setSelection(this.selection);
                this.editor.revealLinesInCenter(this.selection.startLineNumber, this.selection.endLineNumber);
            }
            this.isAwaitingInitialResults = true;
        }
    }

    override resize() {
        _.defer(() => {
            const topBarHeight = utils.updateAndCalcTopBarHeight(this.domRoot, this.topBar, this.hideable);
            this.editor.layout({
                width: (this.domRoot.width() as number) - (this.passesColumn.width() as number),
                height: (this.domRoot.height() as number) - topBarHeight,
            });
        });
    }

    override close(): void {
        this.eventHub.unsubscribe();
        this.eventHub.emit('llvmOptPipelineViewClosed', this.compilerInfo.compilerId);
        this.editor.dispose();
    }
}

/*import * as monaco from 'monaco-editor';
import _ from 'underscore';
import {MonacoPane} from './pane';
import {ga} from '../analytics';
import * as monacoConfig from '../monaco-config';
import {Container} from 'golden-layout';
import {MonacoPaneState} from './pane.interfaces';
import {Hub} from '../hub';

export class LLVMOptPipeline extends MonacoPane<monaco.editor.IStandaloneCodeEditor, {}> {
    constructor(hub: Hub, container: Container, state: {} & MonacoPaneState) {
        super(hub, container, state);
        this.eventHub.emit('llvmOptPipelineViewOpened', this.compilerInfo.compilerId);
        this.eventHub.emit('requestSettings');
    }

    override getInitialHTML(): string {
        return $('#llvm-opt-pipeline').html();
    }

    override createEditor(editorRoot: HTMLElement): monaco.editor.IStandaloneCodeEditor {
        return monaco.editor.create(
            editorRoot,
            monacoConfig.extendConfig({
                language: 'llvm-ir',
                readOnly: true,
                glyphMargin: true,
                lineNumbersMinChars: 3,
            })
        );
    }

    override registerOpeningAnalyticsEvent(): void {
        ga.proxy('send', {
            hitType: 'event',
            eventCategory: 'OpenViewPane',
            eventAction: 'LLVMOptPipeline',
        });
    }

    override registerButtons(state: {} & MonacoPaneState): void {
        super.registerButtons(state);
    }

    showCompilationLoadingMessage() {
        this.showPpResults('<Compiling...>');
    }

    override resize() {
        const topBarHeight = this.topBar.outerHeight(true) as number;
        this.editor.layout({
            width: this.domRoot.width() as number,
            height: (this.domRoot.height() as number) - topBarHeight,
        });
    }

    override onCompileResult(compilerId: number, compiler: any, result: any) {
        if (this.compilerInfo.compilerId !== compilerId) return;

        if (result.hasPpOutput) {
            this.showPpResults(result.ppOutput);
        } else if (compiler.supportsPpView) {
            this.showPpResults('<No output>');
        }
    }

    getCurrentEditorLanguage() {
        return this.editor.getModel()?.getLanguageId();
    }

    override getDefaultPaneName() {
        return 'LLVM Opt Pipeline';
    }

    showPpResults(results) {
        if (typeof results === 'object') {
            if (results.numberOfLinesFiltered > 0) {
                this.editor.setValue(
                    `/* <${results.numberOfLinesFiltered} lines filtered> * /\n\n` + results.output.trimStart()
                );
            } else {
                this.editor.setValue(results.output.trimStart());
            }
        } else {
            this.editor.setValue(results);
        }

        if (!this.isAwaitingInitialResults) {
            if (this.selection) {
                this.editor.setSelection(this.selection);
                this.editor.revealLinesInCenter(this.selection.startLineNumber, this.selection.endLineNumber);
            }
            this.isAwaitingInitialResults = true;
        }
    }

    override onCompiler(id, compiler, options, editorid, treeid) {
        if (id === this.compilerInfo.compilerId) {
            this.compilerInfo.compilerName = compiler ? compiler.name : '';
            this.compilerInfo.editorId = editorid;
            this.compilerInfo.treeId = treeid;
            this.updateTitle();
            if (compiler && !compiler.supportsPpView) {
                this.editor.setValue('<Preprocessor output is not supported for this compiler>');
            }
        }
    }

    override onCompilerClose(id) {
        if (id === this.compilerInfo.compilerId) {
            // We can't immediately close as an outer loop somewhere in GoldenLayout is iterating over
            // the hierarchy. We can't modify while it's being iterated over.
            this.close();
            _.defer(function (self) {
                self.container.close();
            }, this);
        }
    }

    override close() {
        this.eventHub.unsubscribe();
        this.eventHub.emit('llvmOptPipelineViewClosed', this.compilerInfo.compilerId);
        this.editor.dispose();
    }
}*/
