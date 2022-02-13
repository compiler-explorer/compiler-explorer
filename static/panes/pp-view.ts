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

'use strict';

import { Toggles } from '../toggles';
import * as monaco from 'monaco-editor';
import _ from 'underscore';
import $ from 'jquery';
import { Pane } from './pane';
import { ga } from '../analytics';
import * as monacoConfig from '../monaco-config';
import { PPViewState } from './pp-view.interfaces';
import { Container } from 'golden-layout';
import { BasePaneState } from './pane.interfaces';

export class PP extends Pane<monaco.editor.IStandaloneCodeEditor, PPViewState> {
    options: any;

    constructor(hub: any, container: Container, state: PPViewState & BasePaneState) {
        super(hub, container, state);
        this.eventHub.emit('ppViewOpened', this.compilerInfo.compilerId);
        this.eventHub.emit('requestSettings');
        this.onOptionsChange();
        if (state && state.ppOutput) {
            this.showPpResults(state.ppOutput);
        } else {
            this.showCompilationLoadingMessage();
        }
    }

    override getInitialHTML(): string {
        return $('#pp').html();
    }

    override createEditor(editorRoot: HTMLElement): monaco.editor.IStandaloneCodeEditor {
        return monaco.editor.create(editorRoot, monacoConfig.extendConfig({
            language: 'plaintext',
            readOnly: true,
            glyphMargin: true,
            lineNumbersMinChars: 3,
        }));
    }

    override registerOpeningAnalyticsEvent(): void {
        ga.proxy('send', {
            hitType: 'event',
            eventCategory: 'OpenViewPane',
            eventAction: 'PP',
        });
    }

    override registerButtons(state: PPViewState & BasePaneState): void {
        this.options = new Toggles(this.domRoot.find('.options'), ((state as unknown) as Record<string, boolean>));
        this.options.on('change', this.onOptionsChange.bind(this));
    }

    onOptionsChange() {
        const options = this.options.get();
        this.updateState();
        // update parameters for the compiler and recompile
        this.showCompilationLoadingMessage();
        this.eventHub.emit('ppViewOptionsUpdated', this.compilerInfo.compilerId, {
            'filter-headers': options['filter-headers'],
            'clang-format': options['clang-format'],
        }, true);
    }

    showCompilationLoadingMessage() {
        this.showPpResults('<Compiling...>');
    }

    override resize() {
        const topBarHeight = this.topBar.outerHeight(true);
        this.editor.layout({
            width: this.domRoot.width(),
            height: this.domRoot.height() - topBarHeight,
        });
    }

    override onCompileResult(compilerId: number, compiler: any, result: any) {
        if (this.compilerInfo.compilerId !== compilerId) return;
    
        if (result.hasPpOutput) {
            this.showPpResults(result.ppOutput);
        } else if (compiler.supportsPpView) {
            this.showPpResults('<No output>');
        }

        const lang = compiler.lang === 'c' ? 'c' : compiler.lang === 'c++' ? 'cpp' : 'plaintext';
        if (this.getCurrentEditorLanguage() !== lang) {
            monaco.editor.setModelLanguage(this.editor.getModel(), lang);
        }
    }

    getCurrentEditorLanguage() {
        return this.editor.getModel().getLanguageId();
    }

    override getPaneName() {
        return 'Preprocessor Output ' + this.compilerInfo.compilerName +
            ' (Editor #' + this.compilerInfo.editorId + ', Compiler #' + this.compilerInfo.compilerId + ')';
    }

    override updateTitle() {
        this.container.setTitle(this.getPaneName());
    }

    getDisplayablePp(ppResult) {
        return '**' + ppResult.ppType + '** - ' + ppResult.displayString;
    }

    showPpResults(results) {
        if (typeof results === 'object') {
            if (results.numberOfLinesFiltered > 0) {
                this.editor.setValue(`/* <${results.numberOfLinesFiltered} lines filtered> */\n\n`
                                     + results.output.trimStart());
            } else {
                this.editor.setValue(results.output.trimStart());
            }
        } else {
            this.editor.setValue(results);
        }
    
        if (!this.isAwaitingInitialResults) {
            if (this.selection) {
                this.editor.setSelection(this.selection);
                this.editor.revealLinesInCenter(this.selection.startLineNumber,
                    this.selection.endLineNumber);
            }
            this.isAwaitingInitialResults = true;
        }
    }

    override onCompiler(id, compiler, options, editorid) {
        if (id === this.compilerInfo.compilerId) {
            this.compilerInfo.compilerName = compiler ? compiler.name : '';
            this.compilerInfo.editorId = editorid;
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
        const options = this.options.get();
        const state = {
            id: this.compilerInfo.compilerId,
            editorid: this.compilerInfo.editorId,
            selection: this.selection,
            'filter-headers': options['filter-headers'],
            'clang-format': options['clang-format'],
        };
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
