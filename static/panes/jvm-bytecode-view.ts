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

import type {Container} from 'golden-layout';
import $ from 'jquery';
import * as monaco from 'monaco-editor';

import type {ParsedAsmResultLine} from '../../types/asmresult/asmresult.interfaces.js';
import type {CompilationResult} from '../../types/compilation/compilation.interfaces.js';
import type {CompilerInfo} from '../../types/compiler.interfaces.js';
import {applyColours} from '../colour.js';
import type {Hub} from '../hub.js';
import {extendConfig} from '../monaco-config.js';
import type {MonacoPaneState} from './pane.interfaces.js';
import {MonacoPane} from './pane.js';

export interface JvmBytecodeState {
    jvmBytecodeOutput?: ParsedAsmResultLine[];
}

export class JvmBytecode extends MonacoPane<monaco.editor.IStandaloneCodeEditor, JvmBytecodeState> {
    private output: ParsedAsmResultLine[] = [];
    private sourceColours: Record<number, number> = {};
    private colourScheme = '';

    constructor(hub: Hub, container: Container, state: JvmBytecodeState & MonacoPaneState) {
        super(hub, container, state);
        this.showResults(state.jvmBytecodeOutput ?? []);
        this.eventHub.emit('resendCompilation', this.compilerInfo.compilerId);
    }

    override getInitialHTML(): string {
        return $('#jvmBytecode').html();
    }

    override createEditor(root: HTMLElement): void {
        this.editor = monaco.editor.create(root, extendConfig({language: 'asm', readOnly: true, glyphMargin: true}));
    }

    override getPrintName(): string {
        return 'JVM Bytecode';
    }

    override getDefaultPaneName(): string {
        return 'JVM Bytecode';
    }

    override registerCallbacks(): void {
        this.eventHub.on('colours', (editorId, colours, scheme) => {
            if (editorId !== this.compilerInfo.editorId) return;
            this.sourceColours = colours;
            this.colourScheme = scheme;
            this.applySourceColours();
        });
        this.eventHub.emit('jvmBytecodeViewOpened', this.compilerInfo.compilerId);
        this.eventHub.emit('requestSettings');
    }

    override onCompileResult(id: number, compiler: CompilerInfo, result: CompilationResult): void {
        if (id === this.compilerInfo.compilerId) this.showResults(result.jvmBytecodeOutput ?? []);
    }

    override onCompiler(
        id: number,
        compiler: CompilerInfo | null,
        options: string,
        editorId?: number,
        treeId?: number,
    ): void {
        if (id !== this.compilerInfo.compilerId) return;
        this.compilerInfo.compilerName = compiler?.name ?? '';
        this.compilerInfo.editorId = editorId;
        this.compilerInfo.treeId = treeId;
        this.updateTitle();
        if (!compiler?.supportsJvmBytecodeView)
            this.showResults([{text: '<JVM bytecode is not supported for this compiler>'}]);
    }

    private showResults(output: ParsedAsmResultLine[]): void {
        this.output = output;
        this.editor
            .getModel()
            ?.setValue(output.length ? output.map(line => line.text).join('\n') : '<No JVM bytecode output>');
        this.applySourceColours();
    }

    private applySourceColours(): void {
        const colours: Record<number, number> = {};
        this.output?.forEach((line, index) => {
            const source = line.source?.line;
            if (source && this.sourceColours[source - 1] !== undefined) colours[index] = this.sourceColours[source - 1];
        });
        if (this.colourScheme) applyColours(colours, this.colourScheme, this.editorDecorations);
    }

    override close(): void {
        this.eventHub.unsubscribe();
        this.eventHub.emit('jvmBytecodeViewClosed', this.compilerInfo.compilerId);
        this.editor.dispose();
    }
}
