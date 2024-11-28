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

import $ from 'jquery';
import _ from 'underscore';
import * as monaco from 'monaco-editor';
import {Container} from 'golden-layout';

import {MonacoPane} from './pane.js';
import {MonacoPaneState} from './pane.interfaces.js';
import {ClangirState} from './clangir-view.interfaces.js';

import {extendConfig} from '../monaco-config.js';
import {Hub} from '../hub.js';
import {CompilationResult} from '../compilation/compilation.interfaces.js';
import {CompilerInfo} from '../compiler.interfaces.js';
import {Toggles} from '../widgets/toggles.js';
import {ClangirBackendOptions} from '../../types/compilation/clangir.interfaces.js';

export class Clangir extends MonacoPane<monaco.editor.IStandaloneCodeEditor, ClangirState> {
    private options: Toggles;
    private lastOptions: ClangirBackendOptions = {
        flatCFG: false,
    };

    constructor(hub: Hub, container: Container, state: ClangirState & MonacoPaneState) {
        super(hub, container, state);
        if (state.clangirOutput) {
            this.showClangirResults(state.clangirOutput);
        }
        this.onOptionsChange(true);
    }

    override getInitialHTML(): string {
        return $('#clangir').html();
    }

    override createEditor(editorRoot: HTMLElement): void {
        this.editor = monaco.editor.create(
            editorRoot,
            extendConfig({
                language: 'mlir',
                readOnly: true,
                glyphMargin: true,
                lineNumbersMinChars: 3,
            }),
        );
    }

    override getPrintName() {
        return 'ClangIR Output';
    }

    override getDefaultPaneName(): string {
        return 'ClangIR Viewer';
    }

    override registerButtons(state: ClangirState): void {
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
        this.eventHub.emit('clangirViewOpened', this.compilerInfo.compilerId);
        this.eventHub.emit('requestSettings');
    }

    override getCurrentState(): MonacoPaneState {
        return {
            ...super.getCurrentState(),
            ...this.options.get(),
        };
    }

    onOptionsChange(force = false) {
        const options = this.options.get();
        const newOptions: ClangirBackendOptions = {
            flatCFG: options['flat-cfg'],
        };
        let changed = false;
        for (const k in newOptions) {
            if (newOptions[k as keyof ClangirBackendOptions] !== this.lastOptions[k as keyof ClangirBackendOptions]) {
                changed = true;
                break;
            }
        }
        this.lastOptions = newOptions;
        if (changed || force) {
            this.eventHub.emit('clangirViewOptionsUpdated', this.compilerInfo.compilerId, newOptions, true);
        }
    }

    override onCompileResult(compilerId: number, compiler: CompilerInfo, result: CompilationResult): void {
        if (this.compilerInfo.compilerId !== compilerId) return;
        if (result.clangirOutput) {
            this.showClangirResults(result.clangirOutput);
        } else if (compiler.supportsRustMirView) {
            this.showClangirResults([{text: '<No output>'}]);
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
            if (compiler && !compiler.supportsClangirView) {
                this.showClangirResults([{text: '<ClangIR output is not supported for this compiler>'}]);
            }
        }
    }

    showClangirResults(result: any[]): void {
        this.editor.getModel()?.setValue(result.length ? _.pluck(result, 'text').join('\n') : '<No ClangIR generated>');

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
        this.eventHub.emit('clangirViewClosed', this.compilerInfo.compilerId);
        this.editor.dispose();
    }
}
