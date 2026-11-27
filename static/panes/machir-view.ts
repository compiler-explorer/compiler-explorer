// Copyright (c) 2026, Compiler Explorer Authors
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

import {unwrap} from '../../shared/assert.js';
import {ParsedAsmResultLine} from '../../types/asmresult/asmresult.interfaces.js';
import {CompilationResult} from '../../types/compilation/compilation.interfaces.js';
import {CompilerInfo} from '../../types/compiler.interfaces.js';
import {applyColours} from '../colour.js';
import {Hub} from '../hub.js';
import {extendConfig} from '../monaco-config.js';
import {MachIrState} from './machir-view.interfaces.js';
import {MonacoPaneState} from './pane.interfaces.js';
import {MonacoPane} from './pane.js';

/**
 * The pane shows `--emit-ir=listing` as the compiler wrote it, `unattached:` block included. Those instructions are
 * real IR that no block owns, they carry real positions, and the pane has no filter to hide them behind, so dropping
 * them would make the pane disagree with the file the compiler wrote.
 *
 * A line maps to the editor only when its position is in the user's own source, which the adapter marks with a null
 * file. A position in another file of the project, or in std, keeps its path and so links to nothing: the editor
 * holds one file, and a line number from a different one names a line the user never wrote.
 */
export class MachIr extends MonacoPane<monaco.editor.IStandaloneCodeEditor, MachIrState> {
    private listing: ParsedAsmResultLine[] = [];
    private srcColours?: Record<number, number | undefined>;
    private colourScheme?: string;
    private linkedFadeTimeoutId: NodeJS.Timeout | null = null;
    private decorations: Record<string, monaco.editor.IModelDeltaDecoration[]> = {};
    private previousDecorations: string[] = [];

    constructor(hub: Hub, container: Container, state: MachIrState & MonacoPaneState) {
        super(hub, container, state);
        if (state.machIrOutput) {
            this.showMachIrResults(state.machIrOutput);
        }
        // announced only once built: the source editor answers requestSettings with colours synchronously
        this.eventHub.emit('machIrViewOpened', this.compilerInfo.compilerId);
        this.eventHub.emit('requestSettings');
    }

    override getInitialHTML(): string {
        return $('#machIr').html();
    }

    override createEditor(editorRoot: HTMLElement): void {
        this.editor = monaco.editor.create(
            editorRoot,
            extendConfig({
                language: 'mach-ir',
                readOnly: true,
                glyphMargin: true,
                lineNumbersMinChars: 3,
            }),
        );
    }

    override getPrintName() {
        return 'Mach IR Output';
    }

    override getDefaultPaneName(): string {
        return 'Mach IR Viewer';
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
                if (position === null) return;
                const source = this.listing[position.lineNumber - 1]?.source;
                if (source && source.file === null && this.compilerInfo.editorId != null) {
                    this.eventHub.emit('editorLinkLine', this.compilerInfo.editorId, source.line ?? -1, -1, -1, true);
                }
            },
        });
    }

    override registerCallbacks(): void {
        const onMouseMove = _.throttle(this.onMouseMove.bind(this), 50);
        const onDidChangeCursorSelection = _.throttle(
            (event: monaco.editor.ICursorSelectionChangedEvent) => this.onDidChangeCursorSelection(event),
            500,
        );
        this.editor.onMouseMove(event => onMouseMove(event));
        this.editor.onDidChangeCursorSelection(event => onDidChangeCursorSelection(event));

        this.eventHub.on('colours', this.onColours, this);
        this.eventHub.on('panesLinkLine', this.onPanesLinkLine, this);
    }

    override onCompileResult(compilerId: number, compiler: CompilerInfo, result: CompilationResult): void {
        if (this.compilerInfo.compilerId !== compilerId) return;
        if (result.machIrOutput) {
            this.showMachIrResults(result.machIrOutput);
        } else if (compiler.supportsMachIrView) {
            this.showMachIrResults([{text: '<No output>'}]);
        }
    }

    override onCompiler(
        compilerId: number,
        compiler: CompilerInfo | null,
        options: string,
        editorId?: number,
        treeId?: number,
    ): void {
        if (this.compilerInfo.compilerId === compilerId) {
            this.compilerInfo.compilerName = compiler ? compiler.name : '';
            this.compilerInfo.editorId = editorId;
            this.compilerInfo.treeId = treeId;
            this.updateTitle();
            if (compiler && !compiler.supportsMachIrView) {
                this.showMachIrResults([{text: '<Mach IR output is not supported for this compiler>'}]);
            }
        }
    }

    showMachIrResults(result: ParsedAsmResultLine[]): void {
        this.listing = result;
        this.editor.getModel()?.setValue(result.length ? _.pluck(result, 'text').join('\n') : '<No Mach IR generated>');

        if (!this.isAwaitingInitialResults) {
            if (this.selection) {
                this.editor.setSelection(this.selection);
                this.editor.revealLinesInCenter(this.selection.selectionStartLineNumber, this.selection.endLineNumber);
            }
            this.isAwaitingInitialResults = true;
        }
        this.applyMachIrColours();
    }

    onColours(editorId: number, srcColours: Record<number, number>, scheme: string): void {
        if (editorId !== this.compilerInfo.editorId) return;
        this.srcColours = srcColours;
        this.colourScheme = scheme;
        this.applyMachIrColours();
    }

    /** The colours arrive before the listing does, so this runs from both sides rather than from the event alone. */
    private applyMachIrColours(): void {
        if (!this.srcColours || !this.colourScheme) return;

        const listingColours: Record<number, number> = {};
        for (const [index, line] of this.listing.entries()) {
            const source = line.source;
            if (source && source.file === null && source.line !== null && source.line > 0) {
                const colour = this.srcColours[source.line - 1];
                if (colour !== undefined) listingColours[index] = colour;
            }
        }
        applyColours(listingColours, this.colourScheme, this.editorDecorations);
    }

    onMouseMove(event: monaco.editor.IEditorMouseEvent): void {
        if (event.target.position === null || this.settings.hoverShowSource !== true) return;

        this.clearLinkedLines();
        const source = this.listing[event.target.position.lineNumber - 1]?.source;
        const isOwnSource = source != null && source.file === null;
        const line = isOwnSource ? (source.line ?? -1) : -1;
        const column = isOwnSource && source.column ? source.column : -1;

        this.eventHub.emit('editorLinkLine', unwrap(this.compilerInfo.editorId), line, column, column, false);
        this.eventHub.emit(
            'panesLinkLine',
            this.compilerInfo.compilerId,
            line,
            column,
            column,
            false,
            this.getPaneName(),
        );
    }

    onPanesLinkLine(
        compilerId: number,
        lineNumber: number,
        columnBegin: number,
        columnEnd: number,
        revealLine: boolean,
        sender: string,
    ): void {
        if (compilerId !== this.compilerInfo.compilerId) return;

        const fromAnotherPane = sender !== this.getPaneName();
        const lineNumbers: number[] = [];
        const columnLineNumbers: number[] = [];
        for (const [index, line] of this.listing.entries()) {
            const source = line.source;
            if (!source || source.file !== null || source.line !== lineNumber) continue;
            lineNumbers.push(index + 1);
            if (fromAnotherPane && source.column && columnBegin <= source.column && source.column <= columnEnd) {
                columnLineNumbers.push(index + 1);
            }
        }

        if (revealLine && lineNumbers[0]) this.editor.revealLinesInCenter(lineNumbers[0], lineNumbers[0]);

        const lineClass = fromAnotherPane ? 'linked-code-decoration-line' : '';
        this.decorations.linkedCode = [
            ...lineNumbers.map(line => ({
                range: new monaco.Range(line, 1, line, 1),
                options: {
                    isWholeLine: true,
                    linesDecorationsClassName: 'linked-code-decoration-margin',
                    className: lineClass,
                },
            })),
            ...columnLineNumbers.map(line => ({
                range: new monaco.Range(line, 1, line, 1),
                options: {isWholeLine: true, inlineClassName: 'linked-code-decoration-column'},
            })),
        ];

        if (!this.settings.indefiniteLineHighlight) {
            if (this.linkedFadeTimeoutId !== null) clearTimeout(this.linkedFadeTimeoutId);
            this.linkedFadeTimeoutId = setTimeout(() => {
                this.clearLinkedLines();
                this.linkedFadeTimeoutId = null;
            }, 5000);
        }
        this.updateDecorations();
    }

    private updateDecorations(): void {
        this.previousDecorations = this.editor.deltaDecorations(
            this.previousDecorations,
            Object.values(this.decorations).flat(),
        );
    }

    private clearLinkedLines(): void {
        this.decorations.linkedCode = [];
        this.updateDecorations();
    }

    override close(): void {
        this.eventHub.unsubscribe();
        this.eventHub.emit('machIrViewClosed', this.compilerInfo.compilerId);
        this.editor.dispose();
    }
}
