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

import $ from 'jquery';
import _ from 'underscore';
import * as monaco from 'monaco-editor';
import {Container} from 'golden-layout';

import {MonacoPane} from './pane.js';
import {OptState, OptRemark} from './opt-view.interfaces.js';
import {MonacoPaneState} from './pane.interfaces.js';

import {extendConfig} from '../monaco-config.js';
import {Hub} from '../hub.js';
import {CompilationResult} from '../compilation/compilation.interfaces.js';
import {CompilerInfo} from '../compiler.interfaces.js';
import {unwrap} from '../assert.js';
import {Toggles} from '../widgets/toggles.js';

type OptClass = 'None' | 'Missed' | 'Passed' | 'Analysis';

type OptviewLine = {
    text: string;
    srcLine: number;
    optClass: OptClass;
};

export class Opt extends MonacoPane<monaco.editor.IStandaloneCodeEditor, OptState> {
    // Note: bool | undef here instead of just bool because of an issue with field initialization order
    private isCompilerSupported?: boolean;
    private filters: Toggles;
    private toggleWrapButton: Toggles;
    private wrapButton: JQuery<HTMLElement>;
    private wrapTitle: JQuery<HTMLElement>;

    // Keep optRemarks as state, to avoid triggerring a recompile when options change
    private optRemarks: OptRemark[];
    private srcAsOptview: OptviewLine[];

    constructor(hub: Hub, container: Container, state: OptState & MonacoPaneState) {
        super(hub, container, state);
        this.optRemarks = state.optOutput ?? [];
        this.eventHub.emit('optViewOpened', this.compilerInfo.compilerId);
    }

    override getInitialHTML(): string {
        return $('#opt-view').html();
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

    override getPrintName() {
        return 'Opt Remarks';
    }

    override registerButtons(state: OptState) {
        super.registerButtons(state);
        this.filters = new Toggles(this.domRoot.find('.filters'), state as unknown as Record<string, boolean>);
        this.filters.on('change', this.showOptRemarks.bind(this));

        this.toggleWrapButton = new Toggles(this.domRoot.find('.options'), state as unknown as Record<string, boolean>);
        this.toggleWrapButton.on('change', this.onToggleWrapChange.bind(this));

        this.wrapButton = this.domRoot.find('.wrap-lines');
        this.wrapTitle = this.wrapButton.prop('title');

        if (state.wrap === true) {
            this.wrapButton.prop('title', '[ON] ' + this.wrapTitle);
        } else {
            this.wrapButton.prop('title', '[OFF] ' + this.wrapTitle);
        }
    }

    onToggleWrapChange(): void {
        const state = this.getCurrentState();
        if (state.wrap) {
            this.editor.updateOptions({wordWrap: 'on'});
            this.wrapButton.prop('title', '[ON] ' + this.wrapTitle);
        } else {
            this.editor.updateOptions({wordWrap: 'off'});
            this.wrapButton.prop('title', '[OFF] ' + this.wrapTitle);
        }

        this.updateState();
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

        const splitLines = (text: string): string[] => {
            if (!text) return [];
            const result = text.split(/\r?\n/);
            if (result.length > 0 && result[result.length - 1] === '') return result.slice(0, -1);
            return result;
        };
        const srcLines: string[] = result.source ? splitLines(result.source) : [];
        this.srcAsOptview = [];

        for (const i in srcLines) {
            this.srcAsOptview.push({text: srcLines[i], srcLine: Number(i), optClass: 'None'});
        }

        this.editor.setValue(unwrap(result.source));
        if (result.optOutput) {
            this.optRemarks = result.optOutput;
            this.showOptRemarks();
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
        return 'Opt Viewer';
    }

    showOptRemarks() {
        const filters = this.filters.get();
        const includeMissed: boolean = filters['filter-missed'];
        const includePassed: boolean = filters['filter-passed'];
        const includeAnalysis: boolean = filters['filter-analysis'];

        const remarksToDisplay = this.optRemarks.filter(rem => {
            return (
                /* eslint-disable-next-line @typescript-eslint/no-unnecessary-condition */ // TODO
                !!rem.DebugLoc &&
                ((rem.optType === 'Missed' && includeMissed) ||
                    (rem.optType === 'Passed' && includePassed) ||
                    (rem.optType === 'Analysis' && includeAnalysis))
            );
        });
        const groupedRemarks = _(remarksToDisplay).groupBy(x => x.DebugLoc.Line);

        const resLines = [...this.srcAsOptview];
        for (const [key, value] of Object.entries(groupedRemarks)) {
            const origLineNum = Number(key);
            const curLineNum = resLines.findIndex(line => line.srcLine === origLineNum);
            const contents = value.map(rem => ({
                text: rem.displayString,
                srcLine: -1,
                optClass: rem.optType,
            }));
            resLines.splice(curLineNum, 0, ...contents);
        }

        const newText: string = resLines.reduce((accText, curSrcLine) => {
            return accText + (curSrcLine.optClass === 'None' ? curSrcLine.text : '  ') + '\n';
        }, '');
        this.editor.setValue(newText);

        const optDecorations: monaco.editor.IModelDeltaDecoration[] = [];
        resLines.forEach((line, lineNum) => {
            if (line.optClass !== 'None') {
                optDecorations.push({
                    range: new monaco.Range(lineNum + 1, 1, lineNum + 1, Infinity),
                    options: {
                        isWholeLine: true,
                        after: {
                            content: line.text,
                        },
                        inlineClassName: 'opt-line.' + line.optClass.toLowerCase(),
                    },
                });
            }
        });
        this.editorDecorations.set(optDecorations);
    }

    override onCompiler(id: number, compiler: CompilerInfo | null, options: string, editorId: number, treeId: number) {
        if (id === this.compilerInfo.compilerId) {
            this.compilerInfo.compilerName = compiler ? compiler.name : '';
            this.updateTitle();
            this.isCompilerSupported = compiler ? compiler.supportsOptOutput : false;
            if (!this.isCompilerSupported) {
                this.editor.setValue('<OPT remarks are not supported for this compiler>');
            }
        }
    }

    override getCurrentState() {
        return {
            ...this.filters.get(),
            wrap: this.toggleWrapButton.get().wrap,
            ...super.getCurrentState(),
        };
    }

    // Don't do anything for this pane
    override sendPrintData() {}

    override close() {
        this.eventHub.unsubscribe();
        this.eventHub.emit('optViewClosed', this.compilerInfo.compilerId);
        this.editor.dispose();
    }
}
