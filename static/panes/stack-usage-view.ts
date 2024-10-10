// Copyright (c) 2023, Compiler Explorer Authors
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

import {MonacoPane} from './pane.js';
import {StackUsageState, suCodeEntry} from './stack-usage-view.interfaces.js';
import {MonacoPaneState} from './pane.interfaces.js';

import {extendConfig} from '../monaco-config.js';
import {Hub} from '../hub.js';
import {CompilationResult} from '../compilation/compilation.interfaces.js';
import {CompilerInfo} from '../compiler.interfaces.js';
import {unwrap} from '../assert.js';
import {SentryCapture} from '../sentry.js';

type SuClass = 'None' | 'static' | 'dynamic' | 'dynamic,bounded';

type SuViewLine = {
    text: string;
    srcLine: number;
    suClass: SuClass;
};

export class StackUsage extends MonacoPane<monaco.editor.IStandaloneCodeEditor, StackUsageState> {
    // Note: bool | undef here instead of just bool because of an issue with field initialization order
    isCompilerSupported?: boolean;

    constructor(hub: Hub, container: Container, state: StackUsageState & MonacoPaneState) {
        super(hub, container, state);
        if (state.suOutput) {
            this.showStackUsageResults(state.suOutput, state.source);
        }
        this.eventHub.emit('stackUsageViewOpened', this.compilerInfo.compilerId);
    }

    override getInitialHTML(): string {
        return $('#stackusage').html();
    }

    override createEditor(editorRoot: HTMLElement): void {
        this.editor = monaco.editor.create(
            editorRoot,
            extendConfig({
                language: 'plaintext',
                readOnly: true,
                glyphMargin: true,
            }),
        );
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

    override onCompileResult(id: number, compiler: CompilerInfo, result: CompilationResult) {
        if (this.compilerInfo.compilerId !== id || !this.isCompilerSupported) return;
        this.editor.setValue(unwrap(result.source));
        if (result.stackUsageOutput) {
            this.showStackUsageResults(result.stackUsageOutput);
        }

        // TODO: This is inelegant again. Previously took advantage of fourth argument for the compileResult event.
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
        return 'Stack Usage Viewer';
    }

    override getPrintName() {
        return '<Unimplemented>';
    }

    showStackUsageResults(suEntries: suCodeEntry[], source?: string) {
        const splitLines = (text: string): string[] => {
            if (!text) return [];
            const result = text.split(/\r?\n/);
            if (result.length > 0 && result[result.length - 1] === '') return result.slice(0, -1);
            return result;
        };

        const srcLines: string[] = source ? splitLines(source) : [];
        const srcAsSuLines: SuViewLine[] = srcLines.map((line, i) => ({text: line, srcLine: i, suClass: 'None'}));

        const groupedResults = _.groupBy(suEntries, x => x.DebugLoc.Line);

        const resLines = [...srcAsSuLines];
        for (const [key, value] of Object.entries(groupedResults)) {
            const origLineNum = Number(key);
            const curLineNum = resLines.findIndex(line => line.srcLine === origLineNum);
            const contents = value.map(rem => ({
                text: rem.displayString,
                srcLine: -1,
                suClass: rem.Qualifier,
            }));
            resLines.splice(curLineNum, 0, ...contents);
        }

        const newText: string = resLines.reduce((accText, curSrcLine) => {
            return accText + (curSrcLine.suClass === 'None' ? curSrcLine.text : '  ') + '\n';
        }, '');
        this.editor.setValue(newText);

        const suDecorations: monaco.editor.IModelDeltaDecoration[] = [];
        resLines.forEach((line, lineNum) => {
            // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
            if (!line.suClass) {
                // Shouldn't be possible, temp SentryCapture here to investigate
                // https://compiler-explorer.sentry.io/issues/5374209222/
                SentryCapture(
                    {comp: this.compilerInfo.compilerId, code: srcLines},
                    'StackUsageView: line.suClass is undefined',
                );
                return;
            }
            if (line.suClass !== 'None') {
                suDecorations.push({
                    range: new monaco.Range(lineNum + 1, 1, lineNum + 1, Infinity),
                    options: {
                        isWholeLine: true,
                        after: {
                            content: line.text,
                        },
                        inlineClassName: 'stack-usage.' + line.suClass.replace(',', '_'),
                    },
                });
            }
        });
        this.editorDecorations.set(suDecorations);
    }

    override onCompiler(id: number, compiler: CompilerInfo | null) {
        if (id === this.compilerInfo.compilerId) {
            this.compilerInfo.compilerName = compiler ? compiler.name : '';
            this.updateTitle();
            this.isCompilerSupported = compiler ? compiler.supportsStackUsageOutput : false;
            if (!this.isCompilerSupported) {
                this.editor.setValue('<Stack usage output is not supported for this compiler>');
            }
        }
    }

    // Don't do anything for this pane
    override sendPrintData() {}

    override close() {
        this.eventHub.unsubscribe();
        this.eventHub.emit('stackUsageViewClosed', this.compilerInfo.compilerId);
        this.editor.dispose();
    }
}
