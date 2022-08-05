// Copyright (c) 2021, Tom Ritter
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
import * as monaco from 'monaco-editor';
import _ from 'underscore';
import {MonacoPane} from './pane';
import {ga} from '../analytics';
import * as monacoConfig from '../monaco-config';
import {Container} from 'golden-layout';
import {MonacoPaneState} from './pane.interfaces';
import {Hub} from '../hub';
import {ToolInputViewState} from './tool-input-view.interfaces';
import {Settings} from '../settings';

export class ToolInputView extends MonacoPane<monaco.editor.IStandaloneCodeEditor, ToolInputViewState> {
    _toolId: number;
    _toolName: string;
    shouldSetSelectionInitially: boolean;
    debouncedEmitChange: (() => void) & _.Cancelable;
    cursorSelectionThrottledFunction: ((e: any) => void) & _.Cancelable;
    lastChangeEmitted: string | undefined;
    constructor(hub: Hub, container: Container, state: ToolInputViewState & MonacoPaneState) {
        if ((state as any).compilerId) state.id = (state as any).compilerId;
        super(hub, container, state);

        this.settings = Settings.getStoredSettings();

        this._toolId = state.toolId;
        this._toolName = state.toolName;
        // TODO according to TS typing this should always be true
        this.shouldSetSelectionInitially = !!this.selection;

        this.updateTitle();
        this.onSettingsChange(this.settings);
        this.eventHub.emit('toolInputViewOpened', this._toolId);
    }

    override getInitialHTML() {
        return $('#tool-input').html();
    }

    override createEditor(editorRoot: HTMLElement) {
        return monaco.editor.create(
            editorRoot,
            monacoConfig.extendConfig({
                value: '',
                language: 'plaintext',
                readOnly: false,
                glyphMargin: true,
            })
        );
    }

    override registerOpeningAnalyticsEvent() {
        ga.proxy('send', {
            hitType: 'event',
            eventCategory: 'OpenViewPane',
            eventAction: 'toolInputView',
        });
    }

    override registerCallbacks() {
        this.eventHub.on('toolClosed', this.onToolClose, this);
        this.eventHub.on('toolInputViewCloseRequest', this.onToolInputViewCloseRequest, this);
        this.eventHub.on('setToolInput', this.onSetToolInput, this);

        this.container.layoutManager.on('initialised', () => {
            // Once initialized, let everyone know what text we have.
            this.maybeEmitChange(false);
        });
        this.eventHub.on('initialised', () => this.maybeEmitChange(false), this);

        this.editor.getModel()?.onDidChangeContent(() => {
            this.debouncedEmitChange();
            this.updateState();
        });

        this.cursorSelectionThrottledFunction = _.throttle(this.onDidChangeCursorSelection.bind(this), 500);
        this.editor.onDidChangeCursorSelection(e => {
            this.cursorSelectionThrottledFunction(e);
        });
    }

    override getDefaultPaneName() {
        return '';
    }

    override getPaneName() {
        return `Tool Input ${this._toolName} (Compiler #${this.compilerInfo.compilerId})`;
    }

    override getCurrentState() {
        const parent = super.getCurrentState();
        return {
            ...parent,
            toolId: this._toolId,
            toolName: this._toolName,
            compilerId: this.compilerInfo.compilerId,
            selection: this.selection,
        };
    }

    override onCompiler(compilerId: number, compiler: unknown, options: unknown, editorId: number, treeId: number) {}

    override onCompileResult(compilerId: number, compiler: unknown, result: unknown) {}

    override close() {
        this.eventHub.unsubscribe();
        this.eventHub.emit('toolInputViewClosed', this.compilerInfo.compilerId, this._toolId, this.getInput());
        this.editor.dispose();
    }

    onToolClose(compilerId, toolSettings) {
        if (this.compilerInfo.compilerId === compilerId && this._toolId === toolSettings.toolId) {
            // We can't immediately close as an outer loop somewhere in GoldenLayout is iterating over
            // the hierarchy. We can't modify while it's being iterated over.
            this.close();
            _.defer(function (self) {
                self.container.close();
            }, this);
        }
    }

    onToolInputViewCloseRequest(compilerId, toolId) {
        if (this.compilerInfo.compilerId === compilerId && this._toolId === toolId) {
            this.close();
            _.defer(function (self) {
                self.container.close();
            }, this);
        }
    }

    override onCompilerClose(id) {
        if (id === this.compilerInfo.compilerId) {
            // We can't immediately close as an outer loop somewhere in GoldenLayout is iterating over
            // the hierarchy. We can't modify while it's being iterated over.
            this.close();
            _.defer(function (self) {
                self.container.close();
            }, this);
        }
    }

    override onSettingsChange(newSettings) {
        super.onSettingsChange(newSettings);
        this.debouncedEmitChange = _.debounce(() => {
            this.maybeEmitChange(false);
        }, newSettings.delayAfterChange);
    }

    override onDidChangeCursorSelection(e) {
        // On initialization this callback fires with the default selection
        // overwriting any selection from state. If we are awaiting initial
        // selection setting then don't update our selection.
        if (!this.shouldSetSelectionInitially) {
            this.selection = e.selection;
            this.updateState();
        }
    }

    onSetToolInput(compilerId, toolId, value) {
        if (this.compilerInfo.compilerId === compilerId && this._toolId === toolId) {
            const ret = this.editor.getModel()?.setValue(value);
            if (this.shouldSetSelectionInitially && this.selection) {
                this.editor.setSelection(this.selection);
                this.editor.revealLinesInCenter(this.selection.startLineNumber, this.selection.endLineNumber);
                this.shouldSetSelectionInitially = false;
            }
            return ret;
        }
    }

    getInput() {
        if (!this.editor.getModel()) {
            return '';
        }
        return this.editor.getModel()?.getValue() ?? '';
    }

    maybeEmitChange(force) {
        const input = this.getInput();
        if (!force && input === this.lastChangeEmitted) return;

        this.lastChangeEmitted = input;
        this.eventHub.emit('toolInputChange', this.compilerInfo.compilerId, this._toolId, this.lastChangeEmitted);
    }
}
