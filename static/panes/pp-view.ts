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
import {Toggles} from '../widgets/toggles.js';
import * as monaco from 'monaco-editor';
import _ from 'underscore';
import {MonacoPane} from './pane.js';
import {ga} from '../analytics.js';
import * as monacoConfig from '../monaco-config.js';
import {PPViewState} from './pp-view.interfaces.js';
import {Container} from 'golden-layout';
import {MonacoPaneState} from './pane.interfaces.js';
import {Hub} from '../hub.js';
import {unwrap} from '../assert.js';
import {CompilationResult} from '../compilation/compilation.interfaces.js';
import {CompilerInfo} from '../compiler.interfaces.js';

export class PP extends MonacoPane<monaco.editor.IStandaloneCodeEditor, PPViewState> {
    options: any;

    constructor(hub: Hub, container: Container, state: PPViewState & MonacoPaneState) {
        super(hub, container, state);
        this.eventHub.emit('ppViewOpened', this.compilerInfo.compilerId);
        this.eventHub.emit('requestSettings');
        this.onOptionsChange();
        if (state.ppOutput) {
            this.showPpResults(state.ppOutput);
        } else {
            this.showCompilationLoadingMessage();
        }
    }

    override getInitialHTML(): string {
        return $('#pp').html();
    }

    override createEditor(editorRoot: HTMLElement): monaco.editor.IStandaloneCodeEditor {
        return monaco.editor.create(
            editorRoot,
            monacoConfig.extendConfig({
                language: 'plaintext',
                readOnly: true,
                glyphMargin: true,
                lineNumbersMinChars: 3,
            }),
        );
    }

    override getPrintName() {
        return 'Preprocessor Output';
    }

    override getDefaultPaneName() {
        return 'Preprocessor Output';
    }

    override registerOpeningAnalyticsEvent(): void {
        ga.proxy('send', {
            hitType: 'event',
            eventCategory: 'OpenViewPane',
            eventAction: 'PP',
        });
    }

    override registerButtons(state: PPViewState & MonacoPaneState): void {
        super.registerButtons(state);
        this.options = new Toggles(this.domRoot.find('.options'), state as unknown as Record<string, boolean>);
        this.options.on('change', this.onOptionsChange.bind(this));
    }

    onOptionsChange() {
        const options = this.options.get();
        this.updateState();
        // update parameters for the compiler and recompile
        this.showCompilationLoadingMessage();
        this.eventHub.emit(
            'ppViewOptionsUpdated',
            this.compilerInfo.compilerId,
            {
                'filter-headers': options['filter-headers'],
                'clang-format': options['clang-format'],
            },
            true,
        );
    }

    showCompilationLoadingMessage() {
        this.showPpResults('<Compiling...>');
    }

    override resize() {
        const topBarHeight = unwrap(this.topBar.outerHeight(true));
        this.editor.layout({
            width: unwrap(this.domRoot.width()),
            height: unwrap(this.domRoot.height()) - topBarHeight,
        });
    }

    override onCompileResult(compilerId: number, compiler: CompilerInfo, result: CompilationResult) {
        if (this.compilerInfo.compilerId !== compilerId) return;

        if (result.hasPpOutput) {
            this.showPpResults(result.ppOutput);
        } else if (compiler.supportsPpView) {
            this.showPpResults('<No output>');
        }

        const lang = compiler.lang === 'c' ? 'c' : compiler.lang === 'c++' ? 'cpp' : 'plaintext';
        const model = this.editor.getModel();
        if (model != null && this.getCurrentEditorLanguage() !== lang) {
            monaco.editor.setModelLanguage(model, lang);
        }
    }

    getCurrentEditorLanguage() {
        return this.editor.getModel()?.getLanguageId();
    }

    showPpResults(results) {
        if (typeof results === 'object') {
            if (results.numberOfLinesFiltered > 0) {
                this.editor.setValue(
                    `/* <${results.numberOfLinesFiltered} lines filtered> */\n\n` + results.output.trimStart(),
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

    override onCompiler(id: number, compiler: CompilerInfo | null, options: string, editorId: number, treeId: number) {
        if (id === this.compilerInfo.compilerId) {
            this.compilerInfo.compilerName = compiler ? compiler.name : '';
            this.compilerInfo.editorId = editorId;
            this.compilerInfo.treeId = treeId;
            this.updateTitle();
            if (compiler && !compiler.supportsPpView) {
                this.editor.setValue('<Preprocessor output is not supported for this compiler>');
            }
        }
    }

    override updateState() {
        this.container.setState(this.currentState());
    }

    currentState() {
        const state = {
            id: this.compilerInfo.compilerId,
            editorid: this.compilerInfo.editorId,
            treeid: this.compilerInfo.treeId,
            selection: this.selection,
            ...this.options.get(),
        };
        this.paneRenaming.addState(state);
        this.fontScale.addState(state);
        return state;
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
        this.eventHub.emit('ppViewClosed', this.compilerInfo.compilerId);
        this.editor.dispose();
    }
}
