// Copyright (c) 2025, Compiler Explorer Authors
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
import {YulBackendOptions} from '../../types/compilation/yul.interfaces.js';
import {CompilerInfo} from '../../types/compiler.interfaces.js';
import {Hub} from '../hub.js';
import {extendConfig} from '../monaco-config.js';
import {Toggles} from '../widgets/toggles.js';
import {MonacoPaneState} from './pane.interfaces.js';
import {MonacoPane} from './pane.js';
import {YulState} from './yul-view.interfaces.js';

export class Yul extends MonacoPane<monaco.editor.IStandaloneCodeEditor, YulState> {
    private filters: Toggles;
    private lastOptions: YulBackendOptions = {
        filterDebugInfo: true,
    };

    constructor(hub: Hub, container: Container, state: YulState & MonacoPaneState) {
        super(hub, container, state);
        if (state.yulOutput) {
            this.showYulResults(state.yulOutput);
        }

        this.onOptionsChange(true);
    }

    override getInitialHTML(): string {
        return $('#yul').html();
    }

    override createEditor(editorRoot: HTMLElement): void {
        this.editor = monaco.editor.create(
            editorRoot,
            extendConfig({
                language: 'yul',
                readOnly: true,
                glyphMargin: true,
                lineNumbersMinChars: 3,
            }),
        );
    }

    override getPrintName() {
        return 'Yul Output';
    }

    override getDefaultPaneName(): string {
        return 'Yul (Solidity IR) Viewer';
    }

    override registerButtons(state: YulState): void {
        super.registerButtons(state);
        this.filters = new Toggles(this.domRoot.find('.filters'), state as unknown as Record<string, boolean>);
        this.filters.on('change', () => this.onOptionsChange());
    }

    updateButtons(compiler: CompilerInfo | null): void {
        this.filters.enableToggle('filter-debug-info', !!compiler?.supportsYulView);
    }

    override registerCallbacks(): void {
        const throttleFunction = _.throttle(
            (event: monaco.editor.ICursorSelectionChangedEvent) => this.onDidChangeCursorSelection(event),
            500,
        );
        this.editor.onDidChangeCursorSelection(event => throttleFunction(event));
        this.eventHub.emit('yulViewOpened', this.compilerInfo.compilerId);
        this.eventHub.emit('requestSettings');
    }

    override getCurrentState(): MonacoPaneState {
        return {
            ...super.getCurrentState(),
            ...this.filters.get(),
        };
    }

    onOptionsChange(force = false): void {
        const filters = this.filters.get();
        const newOptions: YulBackendOptions = {
            filterDebugInfo: filters['filter-debug-info'],
        };

        let changed = false;
        for (const key in newOptions) {
            if (newOptions[key as keyof YulBackendOptions] !== this.lastOptions[key as keyof Yul]) {
                changed = true;
                break;
            }
        }

        this.lastOptions = newOptions;
        if (changed || force) {
            this.eventHub.emit('yulViewOptionsUpdated', this.compilerInfo.compilerId, newOptions, true);
        }
    }

    override onCompileResult(compilerId: number, compiler: CompilerInfo, result: CompilationResult): void {
        if (this.compilerInfo.compilerId !== compilerId) return;

        if (result.yulOutput) {
            this.showYulResults(result.yulOutput);
        } else if (compiler.supportsYulView) {
            this.showYulResults([{text: '<No output>'}]);
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
            this.compilerInfo.compilerName = compiler?.name || '';
            this.compilerInfo.editorId = editorId;
            this.compilerInfo.treeId = treeId;
            this.updateTitle();
            this.updateButtons(compiler);
            if (!compiler?.supportsYulView) {
                const text = compiler?.name.toLowerCase().includes('resolc')
                    ? '<Yul output is only supported for this compiler when the input language is Solidity>'
                    : '<Yul output is not supported for this compiler>';
                this.showYulResults([{text}]);
            }
        }
    }

    showYulResults(result: any[]): void {
        const newValue = result.length ? _.pluck(result, 'text').join('\n') : '<No Yul generated>';
        this.editor.getModel()?.setValue(newValue);

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
        this.eventHub.emit('yulViewClosed', this.compilerInfo.compilerId);
        this.editor.dispose();
    }
}
