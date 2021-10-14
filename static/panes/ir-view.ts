// Copyright (c) 2021, Compiler Explorer Authors
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

import {Pane} from './pane';
import {IrState} from './ir-view.interfaces';
import {BasePaneState} from './pane.interfaces';

import {ga} from '../analytics';
import {extendConfig} from '../monaco-config';
import {applyColours} from '../colour';

export class Ir extends Pane<monaco.editor.IStandaloneCodeEditor, IrState> {
    linkedFadeTimeoutId: number = -1;
    irCode: any[] = [];
    colours: any[] = [];
    decorations: any = {};
    previousDecorations: string[] = [];

    constructor(hub: any, container: Container, state: IrState & BasePaneState) {
        super(hub, container, state);
        if (state && state.irOutput) {
            this.showIrResults(state.irOutput);
        }
    }

    override getInitialHTML(): string {
        return $('#ir').html();
    }

    override createEditor(editorRoot: HTMLElement): monaco.editor.IStandaloneCodeEditor {
        return monaco.editor.create(editorRoot, extendConfig({
            language: 'llvm-ir',
            readOnly: true,
            glyphMargin: true,
            lineNumbersMinChars: 3,
        }));
    }

    override registerOpeningAnalyticsEvent(): void {
        ga.proxy('send', {
            hitType: 'event',
            eventCategory: 'OpenViewPane',
            eventAction: 'Ir',
        });
    }

    override getPaneName(): string {
        return `LLVM IR Viewer ${this.compilerInfo.compilerName}` +
            `(Editor #${this.compilerInfo.editorId}, ` +
            `Compiler #${this.compilerInfo.compilerId})`;
    }

    override registerEditorActions(): void {
        this.editor.addAction({
            id: 'viewsource',
            label: 'Scroll to source',
            keybindings: [monaco.KeyMod.CtrlCmd | monaco.KeyCode.F10],
            contextMenuGroupId: 'navigation',
            contextMenuOrder: 1.5,
            run: (editor) => {
                const desiredLine = editor.getPosition().lineNumber - 1;
                const source = this.irCode[desiredLine].source;
                if (source !== null && source.file !== null) {
                    this.eventHub.emit('editorLinkLine', this.compilerInfo.editorId, source.line, -1, -1, true);
                }
            },
        });
    }

    override registerCallbacks(): void {
        const onMouseMove = _.throttle(this.onMouseMove.bind(this), 50);
        const onDidChangeCursorSelection = _.throttle(this.onDidChangeCursorSelection.bind(this), 500);
        const onColoursOnCompile = this.eventHub.mediateDependentCalls(this.onColours.bind(this), this.onCompileResult.bind(this));

        this.eventHub.on('compileResult', onColoursOnCompile.dependencyProxy, this);
        this.eventHub.on('colours', onColoursOnCompile.dependentProxy, this);
        this.eventHub.on('panesLinkLine', this.onPanesLinkLine.bind(this));

        this.editor.onMouseMove((event) => onMouseMove(event));
        this.editor.onDidChangeCursorSelection((event) => onDidChangeCursorSelection(event));

        this.eventHub.emit('irViewOpened', this.compilerInfo.compilerId);
        this.eventHub.emit('requestSettings');
    }

    override onCompileResult(compilerId: number, compiler: any, result: any): void {
        if (this.compilerInfo.compilerId !== compilerId) return;
        if (result.hasIrOutput) {
            this.showIrResults(result.irOutput);
        } else if (compiler.supportsIrView) {
            this.showIrResults([{text: '<No output>'}]);
        }
    }

    override onCompiler(compilerId: number, compiler: any, options: unknown, editorId: number): void {
        if (this.compilerInfo.compilerId !== compilerId) return;
        this.compilerInfo.compilerName = compiler ? compiler.name : '';
        this.compilerInfo.editorId = editorId;
        this.setTitle();
        if (compiler && !compiler.supportsIrView) {
            this.editor.setValue('<LLVM IR output is not supported for this compiler>');
        }
    }

    showIrResults(result: any[]): void {
        if (!this.editor) return;
        this.irCode = result;
        this.editor.getModel().setValue(result.length
            ? _.pluck(result, 'text').join('\n')
            : '<No LLVM IR generated>');

        if (!this.isAwaitingInitialResults) {
            if (this.selection) {
                this.editor.setSelection(this.selection);
                this.editor.revealLinesInCenter(this.selection.startLineNumber,
                    this.selection.endLineNumber);
            }
            this.isAwaitingInitialResults = true;
        }
    }

    // TODO(supergrecko): refactor this function
    onColours(compilerId: number, colours: any, scheme: any): void {
        if (compilerId === this.compilerInfo.compilerId) {
            var irColours = {};
            _.each(this.irCode, function (x, index) {
                if (x.source && x.source.file === null && x.source.line > 0 && colours[x.source.line - 1] !== undefined) {
                    irColours[index] = colours[x.source.line - 1];
                }
            });
            // @ts-expect-error
            this.colours = applyColours(this.editor, irColours, scheme, this.colours);
        }
    }

    // TODO(supergrecko): refactor this function
    onMouseMove(e: monaco.editor.IEditorMouseEvent): void {
        if (e === null || e.target === null || e.target.position === null) return;
        // @ts-expect-error
        if (this.settings.hoverShowSource === true && this.irCode) {
            this.clearLinkedLines();
            var hoverCode = this.irCode[e.target.position.lineNumber - 1];
            if (hoverCode) {
                // We check that we actually have something to show at this point!
                var sourceLine = -1;
                var sourceColBegin = -1;
                var sourceColEnd = -1;
                if (hoverCode.source && !hoverCode.source.file) {
                    sourceLine = hoverCode.source.line;
                    if (hoverCode.source.column) {
                        sourceColBegin = hoverCode.source.column;
                        sourceColEnd = sourceColBegin;
                    }
                }
                this.eventHub.emit('editorLinkLine', this.compilerInfo.editorId, sourceLine, sourceColBegin, sourceColEnd, false);
                this.eventHub.emit('panesLinkLine', this.compilerInfo.compilerId,
                    sourceLine, sourceColBegin, sourceColEnd,
                    false, this.getPaneName());
            }
        }
    }

    // TODO(supergrecko): refactor this function
    onPanesLinkLine(compilerId: number, lineNumber: number, colBegin, colEnd, revealLine, sender): void {
        if (Number(compilerId) === this.compilerInfo.compilerId) {
            var lineNums = [];
            var directlyLinkedLineNums = [];
            var signalFromAnotherPane = sender !== this.getPaneName();
            _.each(this.irCode, function (irLine, i) {
                if (irLine.source && irLine.source.file === null && irLine.source.line === lineNumber) {
                    var line = i + 1;
                    lineNums.push(line);
                    var currentCol = irLine.source.column;
                    if (signalFromAnotherPane && currentCol && colBegin <= currentCol && currentCol <= colEnd) {
                        directlyLinkedLineNums.push(line);
                    }
                }
            });
            if (revealLine && lineNums[0]) this.editor.revealLineInCenter(lineNums[0]);
            var lineClass = sender !== this.getPaneName() ? 'linked-code-decoration-line' : '';
            var linkedLineDecorations = _.map(lineNums, function (line) {
                return {
                    range: new monaco.Range(line, 1, line, 1),
                    options: {
                        isWholeLine: true,
                        linesDecorationsClassName: 'linked-code-decoration-margin',
                        className: lineClass,
                    },
                };
            });
            var directlyLinkedLineDecorations = _.map(directlyLinkedLineNums, function (line) {
                return {
                    range: new monaco.Range(line, 1, line, 1),
                    options: {
                        isWholeLine: true,
                        inlineClassName: 'linked-code-decoration-column',
                    },
                };
            });
            // @ts-expect-error
            this.decorations.linkedCode = linkedLineDecorations.concat(directlyLinkedLineDecorations);
            if (this.linkedFadeTimeoutId !== -1) {
                clearTimeout(this.linkedFadeTimeoutId);
            }
            // @ts-expect-error
            this.linkedFadeTimeoutId = setTimeout(_.bind(function () {
                this.clearLinkedLines();
                this.linkedFadeTimeoutId = -1;
            }, this), 5000);
            this.updateDecorations();
        }
    }

    updateDecorations(): void {
        this.previousDecorations = this.editor.deltaDecorations(
            this.previousDecorations, _.flatten(_.values(this.decorations)));
    }

    clearLinkedLines(): void {
        this.decorations.linkedCode = [];
        this.updateDecorations();
    }

    /** LLVM IR View proxies some things in the standard callbacks */
    override registerStandardCallbacks(): void {
        this.fontScale.on('change', this.updateState.bind(this));
        this.container.on('destroy', this.close.bind(this));
        this.container.on('resize', this.resize.bind(this));
        this.eventHub.on('compiler', this.onCompiler.bind(this));
        this.eventHub.on('compilerClose', this.onCompilerClose.bind(this));
        this.eventHub.on('settingsChange', this.onSettingsChange.bind(this));
        this.eventHub.on('shown', this.resize.bind(this));
        this.eventHub.on('resize', this.resize.bind(this));
    }

    override close(): void {
        this.eventHub.unsubscribe();
        this.eventHub.emit('irViewClosed', this.compilerInfo.compilerId);
        this.editor.dispose();
    }
}
