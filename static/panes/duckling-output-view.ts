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

import {CompilationResult} from '../../types/compilation/compilation.interfaces.js';
import {CompilerInfo, DucklingOutputKind, isDucklingOutputKind} from '../../types/compiler.interfaces.js';
import {Hub} from '../hub.js';
import {extendConfig} from '../monaco-config.js';
import {DucklingOutputViewState} from './duckling-output-view.interfaces.js';
import {MonacoPaneState} from './pane.interfaces.js';
import {MonacoPane} from './pane.js';

const outputNames: Record<DucklingOutputKind, string> = {
    lexer: 'Lexer',
    parser: 'Parser',
    hir: 'HIR',
    mir: 'MIR',
    lir: 'LIR',
    llvm: 'LLVM IR',
    dbc: 'DBC',
};

export function formatDucklingParserOutput(source: string): {output: string; languageId: string} {
    const tree = source.replace(/^Parsed tree:\s*/, '').trim();
    if (!tree) return {output: source, languageId: 'plaintext'};

    let inString = false;
    let escaped = false;
    let normalized = '';
    for (let index = 0; index < tree.length; index++) {
        const character = tree[index];
        if (inString) {
            normalized += character;
            if (escaped) {
                escaped = false;
            } else if (character === '\\') {
                escaped = true;
            } else if (character === '"') {
                inString = false;
            }
        } else if (character === '"') {
            inString = true;
            normalized += character;
        } else if (character === ',') {
            let nextIndex = index + 1;
            while (nextIndex < tree.length && /\s/.test(tree[nextIndex])) nextIndex++;
            if (tree[nextIndex] !== '}' && tree[nextIndex] !== ']') normalized += character;
        } else {
            normalized += character;
        }
    }

    try {
        return {output: JSON.stringify(JSON.parse(normalized), null, 2), languageId: 'json'};
    } catch {
        return {output: source, languageId: 'plaintext'};
    }
}

export class DucklingOutputView extends MonacoPane<monaco.editor.IStandaloneCodeEditor, DucklingOutputViewState> {
    private outputKind!: DucklingOutputKind;
    private outputViewOpenAnnounced!: boolean;
    private wrapLines!: boolean;
    private wrapButton!: JQuery<HTMLElement>;
    private wrapTitle!: string;

    constructor(hub: Hub, container: Container, state: DucklingOutputViewState & MonacoPaneState) {
        super(hub, container, state);
        if (state.ducklingOutput?.kind === this.outputKind) {
            this.showOutput(state.ducklingOutput);
        }
    }

    override initializeDefaults(): void {
        this.outputViewOpenAnnounced = false;
        this.wrapLines = false;
    }

    override initializeStateDependentProperties(state: DucklingOutputViewState): void {
        this.outputKind = isDucklingOutputKind(state.ducklingOutputKind) ? state.ducklingOutputKind : 'hir';
        this.wrapLines = state.wrap === true;
    }

    override getInitialHTML(): string {
        return $('#ducklingOutput').html() ?? '';
    }

    override createEditor(editorRoot: HTMLElement): void {
        this.editor = monaco.editor.create(
            editorRoot,
            extendConfig({
                language: 'plaintext',
                readOnly: true,
                glyphMargin: true,
                lineNumbersMinChars: 3,
                wordWrap: this.wrapLines ? 'on' : 'off',
            }),
        );
    }

    override getPrintName(): string {
        return `Duckling ${outputNames[this.outputKind]}`;
    }

    override getDefaultPaneName(): string {
        return `Duckling ${outputNames[this.outputKind]}`;
    }

    override getCurrentState(): MonacoPaneState & DucklingOutputViewState {
        return {...super.getCurrentState(), ducklingOutputKind: this.outputKind, wrap: this.wrapLines};
    }

    override registerButtons(state: DucklingOutputViewState): void {
        super.registerButtons(state);
        this.wrapButton = this.domRoot.find('.wrap-lines');
        this.wrapTitle = this.wrapButton.prop('title') ?? 'Wrap lines';
        this.updateWrapButton();
        this.wrapButton.on('click', () => this.toggleWrapLines());
    }

    override registerCallbacks(): void {
        const throttleFunction = _.throttle(
            (event: monaco.editor.ICursorSelectionChangedEvent) => this.onDidChangeCursorSelection(event),
            500,
        );
        this.editor.onDidChangeCursorSelection(event => throttleFunction(event));

        this.eventHub.emit('requestCompiler', this.compilerInfo.compilerId);
        this.eventHub.emit('requestSettings');
    }

    override onCompileResult(compilerId: number, compiler: CompilerInfo, result: CompilationResult): void {
        if (this.compilerInfo.compilerId !== compilerId) return;

        const output =
            result.ducklingOutputs?.[this.outputKind] ??
            (result.ducklingOutput?.kind === this.outputKind ? result.ducklingOutput : undefined);
        if (output) {
            this.showOutput(output);
        } else if (compiler.supportsDucklingOutputs?.includes(this.outputKind)) {
            this.showMessage('<No output>');
        }
    }

    override onCompiler(
        compilerId: number,
        compiler: CompilerInfo | null,
        _options: string,
        editorId: number,
        treeId: number,
    ): void {
        if (this.compilerInfo.compilerId !== compilerId) return;

        this.compilerInfo.compilerName = compiler?.name ?? '';
        this.compilerInfo.editorId = editorId;
        this.compilerInfo.treeId = treeId;
        if (!compiler?.supportsDucklingOutputs?.includes(this.outputKind)) {
            this.showMessage('<Duckling outputs are not supported for this compiler>');
        } else {
            if (!this.outputViewOpenAnnounced) {
                this.outputViewOpenAnnounced = true;
                this.eventHub.emit('ducklingOutputViewOpened', compilerId, this.outputKind);
            }
        }

        this.updateTitle();
    }

    private showOutput(result: NonNullable<CompilationResult['ducklingOutput']>): void {
        const output = result.output.length > 0 ? _.pluck(result.output, 'text').join('\n') : '<No output>';
        const formattedOutput = this.outputKind === 'parser' ? formatDucklingParserOutput(output) : undefined;
        this.setOutputText(formattedOutput?.output ?? output);
        const model = this.editor.getModel();
        if (model) monaco.editor.setModelLanguage(model, formattedOutput?.languageId ?? result.languageId);
    }

    private showMessage(message: string): void {
        this.setOutputText(message);
        const model = this.editor.getModel();
        if (model) monaco.editor.setModelLanguage(model, 'plaintext');
    }

    private updateWrapButton(): void {
        this.wrapButton
            .toggleClass('active', this.wrapLines)
            .attr('aria-pressed', String(this.wrapLines))
            .prop('title', `[${this.wrapLines ? 'ON' : 'OFF'}] ${this.wrapTitle}`)
            .siblings('input[type=checkbox]')
            .prop('checked', this.wrapLines);
    }

    private toggleWrapLines(): void {
        this.wrapLines = !this.wrapLines;
        this.editor.updateOptions({wordWrap: this.wrapLines ? 'on' : 'off'});
        this.updateWrapButton();
        this.updateState();
    }

    private setOutputText(text: string): void {
        const model = this.editor.getModel();
        if (!model) return;

        model.setValue(text);
        if (!this.isAwaitingInitialResults) {
            if (this.selection) {
                this.editor.setSelection(this.selection);
                this.editor.revealLinesInCenter(this.selection.selectionStartLineNumber, this.selection.endLineNumber);
            }
            this.isAwaitingInitialResults = true;
        }
    }

    override close(): void {
        this.eventHub.unsubscribe();
        this.eventHub.emit('ducklingOutputViewClosed', this.compilerInfo.compilerId, this.outputKind);
        this.editor.dispose();
    }
}
