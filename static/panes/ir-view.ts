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

import {MonacoPane} from './pane';
import {IrState} from './ir-view.interfaces';
import {MonacoPaneState} from './pane.interfaces';

import {ga} from '../analytics';
import {extendConfig} from '../monaco-config';
import {applyColours} from '../colour';

import {PaneRenaming} from '../widgets/pane-renaming';
import {Hub} from '../hub';

export class Ir extends MonacoPane<monaco.editor.IStandaloneCodeEditor, IrState> {
    linkedFadeTimeoutId = -1;
    irCode: any[] = [];
    colours: any[] = [];
    decorations: any = {};
    previousDecorations: string[] = [];

    constructor(hub: Hub, container: Container, state: IrState & MonacoPaneState) {
        super(hub, container, state);
        if (state.irOutput) {
            this.showIrResults(state.irOutput);
        }
    }

    override getInitialHTML(): string {
        return $('#ir').html();
    }

    override createEditor(editorRoot: HTMLElement): monaco.editor.IStandaloneCodeEditor {
        return monaco.editor.create(
            editorRoot,
            extendConfig({
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
            eventAction: 'Ir',
        });
    }

    override getDefaultPaneName(): string {
        return 'LLVM IR Viewer';
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
                        this.eventHub.emit('editorLinkLine', this.compilerInfo.editorId, source.line, -1, -1, true);
                    }
                }
            },
        });
    }

    override registerCallbacks(): void {
        const onMouseMove = _.throttle(this.onMouseMove.bind(this), 50);
        const onDidChangeCursorSelection = _.throttle(this.onDidChangeCursorSelection.bind(this), 500);
        const onColoursOnCompile = this.eventHub.mediateDependentCalls(
            this.onColours.bind(this),
            this.onCompileResult.bind(this)
        );

        this.paneRenaming.on('renamePane', this.updateState.bind(this));

        this.eventHub.on('compileResult', onColoursOnCompile.dependencyProxy, this);
        this.eventHub.on('colours', onColoursOnCompile.dependentProxy, this);
        this.eventHub.on('panesLinkLine', this.onPanesLinkLine.bind(this));

        this.editor.onMouseMove(event => onMouseMove(event));
        this.editor.onDidChangeCursorSelection(event => onDidChangeCursorSelection(event));

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

    onColours(compilerId: number, colours: any, scheme: any): void {
        if (compilerId !== this.compilerInfo.compilerId) return;
        const irColours: Record<number, number> = {};
        for (const [index, code] of this.irCode.entries()) {
            if (
                code.source &&
                code.source.file === null &&
                code.source.line > 0 &&
                colours[code.source.line - 1] !== undefined
            ) {
                irColours[index] = colours[code.source.line - 1];
            }
        }
        this.colours = applyColours(this.editor, irColours, scheme, this.colours);
    }

    onMouseMove(e: monaco.editor.IEditorMouseEvent): void {
        if (e.target.position === null) return;
        if (this.settings.hoverShowSource === true) {
            this.clearLinkedLines();
            if (e.target.position.lineNumber - 1 in this.irCode) {
                const hoverCode = this.irCode[e.target.position.lineNumber - 1];
                let sourceLine = -1;
                let sourceColumnBegin = -1;
                let sourceColumnEnd = -1;

                if (hoverCode.source && !hoverCode.source.file) {
                    sourceLine = hoverCode.source.line;
                    if (hoverCode.source.column) {
                        sourceColumnBegin = hoverCode.source.column;
                        sourceColumnEnd = sourceColumnBegin;
                    }
                }

                this.eventHub.emit(
                    'editorLinkLine',
                    this.compilerInfo.editorId,
                    sourceLine,
                    sourceColumnBegin,
                    sourceColumnEnd,
                    false
                );
                this.eventHub.emit(
                    'panesLinkLine',
                    this.compilerInfo.compilerId,
                    sourceLine,
                    sourceColumnBegin,
                    sourceColumnEnd,
                    false,
                    this.getPaneName()
                );
            }
        }
    }

    onPanesLinkLine(
        compilerId: number,
        lineNumber: number,
        columnBegin: number,
        columnEnd: number,
        revealLinesInEditor: boolean,
        sender: string
    ): void {
        if (compilerId !== this.compilerInfo.compilerId) return;
        const lineNumbers: number[] = [];
        const directlyLinkedLineNumbers: number[] = [];
        const isSignalFromAnotherPane = sender !== this.getPaneName();

        for (const [index, irLine] of this.irCode.entries()) {
            if (irLine.source && irLine.source.file === null && irLine.source.line === lineNumber) {
                const line = index + 1;
                const currentColumn = irLine.source.column;
                lineNumbers.push(line);
                if (
                    isSignalFromAnotherPane &&
                    currentColumn &&
                    columnBegin <= currentColumn &&
                    currentColumn <= columnEnd
                ) {
                    directlyLinkedLineNumbers.push(line);
                }
            }
        }

        if (revealLinesInEditor && lineNumbers[0]) {
            // Just make sure that the mapped line is in view!
            this.editor.revealLinesInCenter(lineNumbers[0], lineNumbers[0]);
        }

        const lineClassName = isSignalFromAnotherPane ? 'linked-code-decoration-line' : '';
        const linkedLineDecorations: monaco.editor.IModelDeltaDecoration[] = lineNumbers.map(line => ({
            range: new monaco.Range(line, 1, line, 1),
            options: {
                isWholeLine: true,
                linesDecorationsClassName: 'linked-code-decoration-margin',
                className: lineClassName,
            },
        }));
        const directlyLinkedLineDecorations: monaco.editor.IModelDeltaDecoration[] = directlyLinkedLineNumbers.map(
            line => ({
                range: new monaco.Range(line, 1, line, 1),
                options: {
                    isWholeLine: true,
                    inlineClassName: 'linked-code-decoration-column',
                },
            })
        );
        this.decorations.linkedCode = [...linkedLineDecorations, ...directlyLinkedLineDecorations];

        if (this.linkedFadeTimeoutId !== -1) {
            clearTimeout(this.linkedFadeTimeoutId);
        }
        // @ts-expect-error mismatched type on setTimeout, assumes NodeJS.Timeout
        this.linkedFadeTimeoutId = setTimeout(() => {
            this.clearLinkedLines();
            this.linkedFadeTimeoutId = -1;
        }, 5000);
        this.updateDecorations();
    }

    updateDecorations(): void {
        this.previousDecorations = this.editor.deltaDecorations(
            this.previousDecorations,
            _.flatten(_.values(this.decorations))
        );
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
