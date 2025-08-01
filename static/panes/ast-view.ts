// Copyright (c) 2017, Sy Brand
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

import {Container} from 'golden-layout';
import $ from 'jquery';
import * as monaco from 'monaco-editor';
import _ from 'underscore';
import {CompilationResult} from '../../types/compilation/compilation.interfaces.js';
import {CompilerInfo} from '../../types/compiler.interfaces.js';
import {ResultLine} from '../../types/resultline/resultline.interfaces.js';
import {unwrap} from '../assert.js';
import * as colour from '../colour.js';
import {Hub} from '../hub.js';
import * as monacoConfig from '../monaco-config.js';
import {AstState} from './ast-view.interfaces.js';
import {MonacoPaneState} from './pane.interfaces.js';
import {MonacoPane} from './pane.js';

type DecorationEntry = {
    linkedCode: any[];
};

type SourceLocation = {
    line: number | null; // Null only for malformed strings
    col: number | null; // Ditto
};

type AstCodeEntry = {
    text: string;
    source?: {
        from: SourceLocation | null;
        to: SourceLocation | null;
    };
};

export class Ast extends MonacoPane<monaco.editor.IStandaloneCodeEditor, AstState> {
    // TODO: eliminate deprecated deltaDecorations monaco API
    decorations: DecorationEntry = {linkedCode: []};
    prevDecorations: any[] = [];
    colourScheme?: string = undefined;
    srcColours?: Record<number, number> = undefined;
    astCode?: AstCodeEntry[] = undefined;
    linkedFadeTimeoutId?: NodeJS.Timeout = undefined;

    constructor(hub: Hub, container: Container, state: AstState & MonacoPaneState) {
        super(hub, container, state);

        if (state.astOutput) {
            this.showAstResults(state.astOutput);
        }
    }

    override getInitialHTML(): string {
        return $('#ast').html();
    }

    override registerCallbacks(): void {
        const mouseMoveThrottledFunction = _.throttle(this.onMouseMove.bind(this), 50);
        this.editor.onMouseMove(e => mouseMoveThrottledFunction(e));

        this.fontScale.on('change', this.updateState.bind(this));
        this.eventHub.on('renamePane', this.updateState.bind(this));

        this.container.on('destroy', this.close, this);

        this.eventHub.on('compileResult', this.onCompileResult, this);
        this.eventHub.on('compiler', this.onCompiler, this);
        this.eventHub.on('colours', this.onColours, this);
        this.eventHub.on('panesLinkLine', this.onPanesLinkLine, this);
        this.eventHub.on('compilerClose', this.onCompilerClose, this);
        this.eventHub.on('settingsChange', this.onSettingsChange, this);
        this.eventHub.emit('astViewOpened', this.compilerInfo.compilerId);
        this.eventHub.emit('requestSettings');

        this.container.on('resize', this.resize, this);
        this.container.on('shown', this.resize, this);

        const cursorSelectionThrottledFunction = _.throttle(this.onDidChangeCursorSelection.bind(this), 500);
        this.editor.onDidChangeCursorSelection(e => cursorSelectionThrottledFunction(e));
    }

    override createEditor(editorRoot: HTMLElement): void {
        this.editor = monaco.editor.create(
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
        return 'Ast Output';
    }

    override getDefaultPaneName() {
        return 'Ast Viewer';
    }

    onMouseMove(e: monaco.editor.IEditorMouseEvent) {
        if (e.target.position === null) return;
        if (this.settings.hoverShowSource === true) {
            this.clearLinkedLines();
            if (this.astCode && e.target.position.lineNumber - 1 in this.astCode) {
                const hoverCode = this.astCode[e.target.position.lineNumber - 1];
                let sourceLine = -1;
                let colBegin = -1;
                let colEnd = -1;
                // We check that we actually have something to show at this point!
                if (hoverCode.source?.from?.line) {
                    sourceLine = hoverCode.source.from.line;
                    // Highlight part of a line corresponding to the node if it fits on one line
                    if (
                        hoverCode.source.to &&
                        hoverCode.source.from.line === hoverCode.source.to.line &&
                        hoverCode.source.from.col &&
                        hoverCode.source.to.col
                    ) {
                        colBegin = hoverCode.source.from.col;
                        colEnd = hoverCode.source.to.col;
                    }
                }
                this.eventHub.emit(
                    'editorLinkLine',
                    unwrap(this.compilerInfo.editorId),
                    sourceLine,
                    colBegin,
                    colEnd,
                    false,
                );
                this.eventHub.emit(
                    'panesLinkLine',
                    this.compilerInfo.compilerId,
                    sourceLine,
                    colBegin,
                    colEnd,
                    false,
                    this.getPaneName(),
                );
            }
        }
    }

    getCurrentEditorLanguage() {
        return this.editor.getModel()?.getLanguageId();
    }

    override onCompileResult(id: number, compiler: CompilerInfo, result: CompilationResult) {
        if (this.compilerInfo.compilerId !== id) return;

        if (result.astOutput) {
            this.showAstResults(result.astOutput);
            this.tryApplyAstColours();
        } else if (compiler.supportsAstView) {
            this.showAstResults([{text: '<No output>'}]);
        }

        // TODO: This is inelegant. Previously took advantage of fourth argument for the compileResult event.
        // I'm guessing it's not part of the TS rewrite because it's not always passed by the emitter.
        const lang = compiler.lang === 'c++' ? 'cpp' : compiler.lang;
        const model = this.editor.getModel();
        if (model != null && this.getCurrentEditorLanguage() !== lang) {
            monaco.editor.setModelLanguage(model, lang);
        }
    }

    showAstResults(results: any) {
        const fullText = typeof results === 'string' ? results : results.map((x: ResultLine) => x.text).join('\n');
        this.editor.setValue(fullText);
        if (results) {
            if (typeof results === 'string') {
                this.astCode = results.split('\n').map(x => {
                    return {
                        text: x,
                    };
                });
            } else if (Array.isArray(results)) {
                this.astCode = results;
            } else {
                this.astCode = [];
            }
        } else {
            this.astCode = [];
        }

        if (!this.isAwaitingInitialResults) {
            if (this.selection) {
                this.editor.setSelection(this.selection);
                this.editor.revealLinesInCenter(this.selection.startLineNumber, this.selection.endLineNumber);
            }
            this.isAwaitingInitialResults = true;
        }
    }

    override onCompiler(id: number, compiler: CompilerInfo | null, options: string, editorid: number, treeid: number) {
        if (id === this.compilerInfo.compilerId) {
            this.compilerInfo.compilerName = compiler ? compiler.name : '';
            this.compilerInfo.editorId = editorid;
            this.compilerInfo.treeId = treeid;
            this.updateTitle();
            if (compiler && !compiler.supportsAstView) {
                this.editor.setValue('<AST output is not supported for this compiler>');
            }
        }
    }

    tryApplyAstColours(): void {
        if (!this.srcColours || !this.colourScheme || !this.astCode || this.astCode.length === 0) return;
        const astColours: Record<number, number> = {};
        for (const [index, code] of this.astCode.entries()) {
            if (
                code.source?.from?.line &&
                code.source.to?.line &&
                code.source.from.line <= code.source.to.line &&
                code.source.to.line < code.source.from.line + 100
            ) {
                for (let i = code.source.from.line; i <= code.source.to.line; ++i) {
                    if (i - 1 in this.srcColours) {
                        astColours[index] = this.srcColours[i - 1];
                        break;
                    }
                }
            }
        }
        colour.applyColours(astColours, this.colourScheme, this.editorDecorations);
    }

    onColours(id: number, srcColours: Record<number, number>, colourScheme: string): void {
        if (id !== this.compilerInfo.editorId) return;

        this.srcColours = srcColours;
        this.colourScheme = colourScheme;

        this.tryApplyAstColours();
    }

    updateDecorations() {
        this.prevDecorations = this.editor.deltaDecorations(
            this.prevDecorations,
            _.flatten(_.values(this.decorations)),
        );
    }

    clearLinkedLines() {
        this.decorations.linkedCode = [];
        this.updateDecorations();
    }

    onPanesLinkLine(
        compilerId: number,
        lineNumber: number,
        colBegin: number,
        colEnd: number,
        revealLine: boolean,
        sender: string,
    ) {
        if (Number(compilerId) === this.compilerInfo.compilerId && this.astCode) {
            const lineNums: number[] = [];
            const singleNodeLines: number[] = [];
            const signalFromAnotherPane = sender !== this.getPaneName();
            for (const [i, astLine] of this.astCode.entries()) {
                if (
                    astLine.source?.from?.line &&
                    astLine.source.to?.line &&
                    astLine.source.from.line <= lineNumber &&
                    lineNumber <= astLine.source.to.line
                ) {
                    const line = i + 1;
                    lineNums.push(line);
                    if (
                        signalFromAnotherPane &&
                        astLine.source.from.line === lineNumber &&
                        astLine.source.to.line === lineNumber &&
                        astLine.source.from.col &&
                        astLine.source.to.col &&
                        astLine.source.from.col < colEnd &&
                        colBegin <= astLine.source.to.col
                    ) {
                        singleNodeLines.push(line);
                    }
                }
            }
            if (revealLine && lineNums[0]) this.editor.revealLineInCenter(lineNums[0]);
            const lineClass = signalFromAnotherPane ? 'linked-code-decoration-line' : '';
            const linkedLineDecorations = lineNums.map(line => ({
                range: new monaco.Range(line, 1, line, 1),
                options: {
                    isWholeLine: true,
                    linesDecorationsClassName: 'linked-code-decoration-margin',
                    className: lineClass,
                },
            }));
            const directlyLinkedLineDecorations = singleNodeLines.map(line => ({
                range: new monaco.Range(line, 1, line, 1),
                options: {
                    isWholeLine: true,
                    inlineClassName: 'linked-code-decoration-column',
                },
            }));
            this.decorations.linkedCode = [...linkedLineDecorations, ...directlyLinkedLineDecorations];

            if (!this.settings.indefiniteLineHighlight) {
                if (this.linkedFadeTimeoutId) {
                    clearTimeout(this.linkedFadeTimeoutId);
                }
                this.linkedFadeTimeoutId = setTimeout(() => {
                    this.clearLinkedLines();
                    this.linkedFadeTimeoutId = undefined;
                }, 5000);
            }
            this.updateDecorations();
        }
    }

    override close() {
        this.eventHub.unsubscribe();
        this.eventHub.emit('astViewClosed', this.compilerInfo.compilerId);
        this.editor.dispose();
    }
}
