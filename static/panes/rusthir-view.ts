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
import {MonacoPaneState} from './pane.interfaces';
import {RustHirState} from './rusthir-view.interfaces';

import {ga} from '../analytics';
import {extendConfig} from '../monaco-config';
import {Hub} from '../hub';

export class RustHir extends MonacoPane<monaco.editor.IStandaloneCodeEditor, RustHirState> {
    constructor(hub: Hub, container: Container, state: RustHirState & MonacoPaneState) {
        super(hub, container, state);
        if (state.rustHirOutput) {
            this.showRustHirResults(state.rustHirOutput);
        }
    }

    override getInitialHTML(): string {
        return $('#rusthir').html();
    }

    override createEditor(editorRoot: HTMLElement): monaco.editor.IStandaloneCodeEditor {
        return monaco.editor.create(
            editorRoot,
            extendConfig({
                language: 'plainText',
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
            eventAction: 'RustHir',
        });
    }

    override getDefaultPaneName(): string {
        return 'Rust HIR Viewer';
    }

    override registerCallbacks(): void {
        const throttleFunction = _.throttle(event => this.onDidChangeCursorSelection(event), 500);
        this.editor.onDidChangeCursorSelection(event => throttleFunction(event));
        this.eventHub.emit('rustHirViewOpened', this.compilerInfo.compilerId);
        this.eventHub.emit('requestSettings');
    }

    override onCompileResult(compilerId: number, compiler: any, result: any): void {
        if (this.compilerInfo.compilerId !== compilerId) return;
        if (result.hasRustHirOutput) {
            this.showRustHirResults(result.rustHirOutput);
        } else if (compiler.supportsRustHirView) {
            this.showRustHirResults([{text: '<No output>'}]);
        }
    }

    override onCompiler(compilerId: number, compiler: any, options: any, editorId?: number, treeId?: number): void {
        if (this.compilerInfo.compilerId === compilerId) {
            this.compilerInfo.compilerName = compiler ? compiler.name : '';
            this.compilerInfo.editorId = editorId;
            this.compilerInfo.treeId = treeId;
            this.updateTitle();
            if (compiler && !compiler.supportsRustHirView) {
                this.showRustHirResults([
                    {
                        text: '<Rust HIR output is not supported for this compiler>',
                    },
                ]);
            }
        }
    }

    showRustHirResults(result: any[]): void {
        this.editor
            .getModel()
            ?.setValue(result.length ? _.pluck(result, 'text').join('\n') : '<No Rust HIR generated>');

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
        this.eventHub.emit('rustHirViewClosed', this.compilerInfo.compilerId);
        this.editor.dispose();
    }
}
