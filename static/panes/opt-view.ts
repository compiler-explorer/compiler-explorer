// Copyright (c) 2017, Jared Wyles
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
import {OptState} from './opt-view.interfaces';
import {MonacoPaneState} from './pane.interfaces';

import {ga} from '../analytics';
import {extendConfig} from '../monaco-config';
import {Hub} from '../hub';

type SourceLocation = {
    File: string;
    Line: number;
    Column: number;
};

type OptCodeEntry = {
    // TODO: Not fully correct type yet, will do for now
    DebugLoc: SourceLocation;
    Function: string;
    Pass: string;
    Name: string;
    text: string;
    optType: string;
    displayString: string;
};

export class Opt extends MonacoPane<monaco.editor.IStandaloneCodeEditor, OptState> {
    decorations: any = {};
    currentDecorations: string[] = [];
    // Note: bool | undef here instead of just bool because of an issue with field initialization order
    isCompilerSupported?: boolean;

    constructor(hub: Hub, container: Container, state: OptState & MonacoPaneState) {
        super(hub, container, state);
        if (state.optOutput) {
            this.showOptResults(state.optOutput);
        }
        this.eventHub.emit('optViewOpened', this.compilerInfo.compilerId);
    }

    override getInitialHTML(): string {
        return $('#opt').html();
    }

    override createEditor(editorRoot: HTMLElement): monaco.editor.IStandaloneCodeEditor {
        return monaco.editor.create(
            editorRoot,
            extendConfig({
                language: 'plaintext',
                readOnly: true,
                glyphMargin: true,
            })
        );
    }

    override registerOpeningAnalyticsEvent() {
        ga.proxy('send', {
            hitType: 'event',
            eventCategory: 'OpenViewPane',
            eventAction: 'Opt',
        });
    }

    override registerCallbacks() {
        this.eventHub.emit('requestSettings');
        this.eventHub.emit('findCompilers');

        this.container.on('shown', this.resize, this);

        const cursorSelectionThrottledFunction = _.throttle(this.onDidChangeCursorSelection.bind(this), 500);
        this.editor.onDidChangeCursorSelection(e => {
            cursorSelectionThrottledFunction(e);
        });
    }

    override onCompileResult(id: number, compiler, result) {
        if (this.compilerInfo.compilerId !== id || !this.isCompilerSupported) return;
        this.editor.setValue(result.source);
        if (result.hasOptOutput) {
            this.showOptResults(result.optOutput);
        }
        // TODO: This is unelegant again. Previously took advantage of fourth argument for the compileResult event.
        const lang = compiler.lang === 'c++' ? 'cpp' : compiler.lang;
        const model = this.editor.getModel();
        if (model != null && this.getCurrentEditorLanguage() !== lang) {
            monaco.editor.setModelLanguage(model, lang);
        }

        if (!this.isAwaitingInitialResults) {
            if (this.selection) {
                this.editor.setSelection(this.selection);
                this.editor.revealLinesInCenter(this.selection.startLineNumber, this.selection.endLineNumber);
            }
            this.isAwaitingInitialResults = true;
        }
    }

    // Monaco language id of the current editor
    getCurrentEditorLanguage() {
        return this.editor.getModel()?.getLanguageId();
    }

    override getDefaultPaneName() {
        return 'Opt Viewer';
    }

    getDisplayableOpt(optResult: OptCodeEntry) {
        return {
            value: '**' + optResult.optType + '** - ' + optResult.displayString,
            isTrusted: false,
        };
    }

    showOptResults(results: OptCodeEntry[]) {
        const opt: monaco.editor.IModelDeltaDecoration[] = [];

        const groupedResults = _.groupBy(
            /* eslint-disable-next-line @typescript-eslint/no-unnecessary-condition */ // TODO
            results.filter(x => x.DebugLoc !== undefined),
            x => x.DebugLoc.Line
        );

        for (const [key, value] of Object.entries(groupedResults)) {
            const linenumber = Number(key);
            const className = value.reduce((acc, x) => {
                if (x.optType === 'Missed' || acc === 'Missed') {
                    return 'Missed';
                } else if (x.optType === 'Passed' || acc === 'Passed') {
                    return 'Passed';
                }
                return x.optType;
            }, '');
            const contents = value.map(this.getDisplayableOpt);
            opt.push({
                range: new monaco.Range(linenumber, 1, linenumber, Infinity),
                options: {
                    isWholeLine: true,
                    glyphMarginClassName: 'opt-decoration.' + className.toLowerCase(),
                    hoverMessage: contents,
                    glyphMarginHoverMessage: contents,
                },
            });
        }

        this.currentDecorations = this.editor.deltaDecorations(this.currentDecorations, opt);
    }

    override onCompiler(id: number, compiler) {
        if (id === this.compilerInfo.compilerId) {
            this.compilerInfo.compilerName = compiler ? compiler.name : '';
            this.updateTitle();
            this.isCompilerSupported = compiler ? compiler.supportsOptOutput : false;
            if (!this.isCompilerSupported) {
                this.editor.setValue('<OPT output is not supported for this compiler>');
            }
        }
    }

    override close() {
        this.eventHub.unsubscribe();
        this.eventHub.emit('optViewClosed', this.compilerInfo.compilerId);
        this.editor.dispose();
    }
}
