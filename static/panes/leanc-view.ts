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
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
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
import {Hub} from '../hub.js';
import {extendConfig} from '../monaco-config.js';
import {Toggles} from '../widgets/toggles.js';
import {LeanCOptions, LeanCState} from './leanc-view.interfaces.js';
import {MonacoPaneState} from './pane.interfaces.js';
import {MonacoPane} from './pane.js';

export class LeanC extends MonacoPane<monaco.editor.IStandaloneCodeEditor, LeanCState> {
    private options: Toggles;

    constructor(hub: Hub, container: Container, state: LeanCState & MonacoPaneState) {
        super(hub, container, state);
        if (state.leanCOutput) {
            this.showLeanCResults(state.leanCOutput);
        }

        this.onOptionsChange();
    }

    override getInitialHTML(): string {
        return $('#leanC').html();
    }

    override createEditor(editorRoot: HTMLElement): void {
        this.editor = monaco.editor.create(
            editorRoot,
            extendConfig({
                language: 'c',
                readOnly: true,
                glyphMargin: true,
                lineNumbersMinChars: 3,
            }),
        );
    }

    override getPrintName() {
        return 'Lean C Output';
    }

    override getDefaultPaneName(): string {
        return 'Lean C Output';
    }

    override registerButtons(state: LeanCState & MonacoPaneState): void {
        super.registerButtons(state);
        this.options = new Toggles(this.domRoot.find('.options'), state as unknown as Record<string, boolean>);
        this.options.on('change', this.onOptionsChange.bind(this));
    }

    override registerCallbacks(): void {
        const throttleFunction = _.throttle(
            (event: monaco.editor.ICursorSelectionChangedEvent) => this.onDidChangeCursorSelection(event),
            500,
        );
        this.editor.onDidChangeCursorSelection(event => throttleFunction(event));
        this.eventHub.emit('leanCViewOpened', this.compilerInfo.compilerId);
        this.eventHub.emit('requestSettings');
    }

    override getCurrentState(): MonacoPaneState {
        return {
            ...super.getCurrentState(),
            ...this.options.get(),
        };
    }

    onOptionsChange(): void {
        const options = this.options.get() as LeanCOptions;
        this.updateState();
        this.showLeanCResults([{text: '<Compiling...>'}]);
        this.eventHub.emit('leanCViewOptionsUpdated', this.compilerInfo.compilerId, options, true);
    }

    override onCompileResult(compilerId: number, compiler: CompilerInfo, result: CompilationResult): void {
        if (this.compilerInfo.compilerId !== compilerId) return;
        if (result.leanCOutput) {
            this.showLeanCResults(result.leanCOutput);
        } else if (compiler.supportsLeanCView) {
            this.showLeanCResults([{text: '<No output>'}]);
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
            if (compiler && !compiler.supportsLeanCView) {
                this.showLeanCResults([{text: '<Lean C output is not supported for this compiler>'}]);
            }
        }
    }

    showLeanCResults(result: Record<'text', string>[]): void {
        this.editor
            .getModel()
            ?.setValue(result.length ? _.pluck(result, 'text').join('\n') : '<No Lean C output generated>');

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
        this.eventHub.emit('leanCViewClosed', this.compilerInfo.compilerId);
        this.editor.dispose();
    }
}
