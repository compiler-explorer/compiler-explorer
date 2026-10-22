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
import {
    CompilerInfo,
    DUCKLING_OUTPUT_KINDS,
    DucklingOutputKind,
    isDucklingOutputKind,
} from '../../types/compiler.interfaces.js';
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

export class DucklingOutputView extends MonacoPane<monaco.editor.IStandaloneCodeEditor, DucklingOutputViewState> {
    private outputKind!: DucklingOutputKind;
    private outputKindSelect!: JQuery<HTMLSelectElement>;
    private supportedOutputKinds!: DucklingOutputKind[];
    private outputViewOpenAnnounced!: boolean;

    constructor(hub: Hub, container: Container, state: DucklingOutputViewState & MonacoPaneState) {
        super(hub, container, state);
        if (state.ducklingOutput?.kind === this.outputKind) {
            this.showOutput(state.ducklingOutput);
        }
    }

    override initializeDefaults(): void {
        this.supportedOutputKinds = [];
        this.outputViewOpenAnnounced = false;
    }

    override initializeStateDependentProperties(state: DucklingOutputViewState): void {
        this.outputKind = isDucklingOutputKind(state.ducklingOutputKind) ? state.ducklingOutputKind : 'llvm';
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
            }),
        );
    }

    override getPrintName(): string {
        return `Duckling ${outputNames[this.outputKind]}`;
    }

    override getDefaultPaneName(): string {
        return 'Duckling Output';
    }

    override getCurrentState(): MonacoPaneState & DucklingOutputViewState {
        return {...super.getCurrentState(), ducklingOutputKind: this.outputKind};
    }

    override registerButtons(state: DucklingOutputViewState): void {
        super.registerButtons(state);
        this.outputKindSelect = this.domRoot.find('.duckling-output-kind');
        this.outputKindSelect.val(this.outputKind).prop('disabled', true);
        this.outputKindSelect.on('change', () => {
            const selectedKind = this.outputKindSelect.val();
            if (!isDucklingOutputKind(selectedKind)) return;
            if (!this.supportedOutputKinds.includes(selectedKind)) {
                this.outputKindSelect.val(this.outputKind);
                return;
            }
            if (selectedKind === this.outputKind) return;

            this.outputKind = selectedKind;
            this.updateTitle();
            this.updateState();
            this.showMessage('<Compiling...>');
            this.eventHub.emit('ducklingOutputKindChanged', this.compilerInfo.compilerId, selectedKind);
        });
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

        if (result.ducklingOutput?.kind === this.outputKind) {
            this.showOutput(result.ducklingOutput);
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
        this.supportedOutputKinds = DUCKLING_OUTPUT_KINDS.filter(kind =>
            compiler?.supportsDucklingOutputs?.includes(kind),
        );

        this.outputKindSelect.find('option').each((_index, option) => {
            const optionKind = $(option).val();
            $(option).prop(
                'hidden',
                !isDucklingOutputKind(optionKind) || !this.supportedOutputKinds.includes(optionKind),
            );
        });
        this.outputKindSelect.prop('disabled', this.supportedOutputKinds.length === 0);

        if (this.supportedOutputKinds.length === 0) {
            this.showMessage('<Duckling outputs are not supported for this compiler>');
        } else {
            const outputKindChanged = !this.supportedOutputKinds.includes(this.outputKind);
            if (outputKindChanged) {
                this.outputKind = this.supportedOutputKinds[0];
                this.outputKindSelect.val(this.outputKind);
                this.updateState();
                this.showMessage('<Compiling...>');
            }

            if (!this.outputViewOpenAnnounced) {
                this.outputViewOpenAnnounced = true;
                this.eventHub.emit('ducklingOutputViewOpened', compilerId, this.outputKind);
            } else if (outputKindChanged) {
                this.eventHub.emit('ducklingOutputKindChanged', compilerId, this.outputKind);
            }
        }

        this.updateTitle();
    }

    private showOutput(result: NonNullable<CompilationResult['ducklingOutput']>): void {
        this.setOutputText(result.output.length > 0 ? _.pluck(result.output, 'text').join('\n') : '<No output>');
        const model = this.editor.getModel();
        if (model) monaco.editor.setModelLanguage(model, result.languageId);
    }

    private showMessage(message: string): void {
        this.setOutputText(message);
        const model = this.editor.getModel();
        if (model) monaco.editor.setModelLanguage(model, 'plaintext');
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
        this.eventHub.emit('ducklingOutputViewClosed', this.compilerInfo.compilerId);
        this.editor.dispose();
    }
}
