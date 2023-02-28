// Copyright (c) 2022, Tom Ritter and Compiler Explorer Authors
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
import _, {Cancelable} from 'underscore';
import {MonacoPane} from './pane.js';
import {ga} from '../analytics.js';
import * as monacoConfig from '../monaco-config.js';
import {FlagsViewState} from './flags-view.interfaces.js';
import {Container} from 'golden-layout';
import {MonacoPaneState} from './pane.interfaces.js';
import {Settings, SiteSettings} from '../settings.js';
import {Hub} from '../hub.js';

export class Flags extends MonacoPane<monaco.editor.IStandaloneCodeEditor, FlagsViewState> {
    debouncedEmitChange: (e: boolean) => void = () => {};
    cursorSelectionThrottledFunction: ((e: any) => void) & Cancelable;
    lastChangeEmitted: string;
    constructor(hub: Hub, container: Container, state: FlagsViewState & MonacoPaneState) {
        super(hub, container, state);

        let value = '';
        if (state.compilerFlags) {
            value = state.compilerFlags.replace(/ /g, '\n');
        }
        this.editor.setValue(value);

        this.onSettingsChange(Settings.getStoredSettings());
        this.eventHub.emit('flagsViewOpened', this.compilerInfo.compilerId);
    }

    override getInitialHTML() {
        return $('#flags').html();
    }

    override createEditor(editorRoot: HTMLElement) {
        return monaco.editor.create(
            editorRoot,
            monacoConfig.extendConfig({
                language: 'plaintext',
                readOnly: false,
                glyphMargin: true,
            }),
        );
    }

    override registerOpeningAnalyticsEvent() {
        ga.proxy('send', {
            hitType: 'event',
            eventCategory: 'OpenViewPane',
            eventAction: 'detailedCompilerFlags',
        });
    }

    override registerCallbacks() {
        this.eventHub.emit('requestSettings');
        this.eventHub.emit('findCompilers');

        this.container.layoutManager.on('initialised', () => {
            // Once initialized, let everyone know what text we have.
            this.maybeEmitChange(false);
        });
        this.eventHub.on('initialised', () => this.maybeEmitChange(false), this);

        this.editor.getModel()?.onDidChangeContent(() => {
            this.debouncedEmitChange(false);
            this.updateState();
        });

        this.cursorSelectionThrottledFunction = _.throttle(this.onDidChangeCursorSelection.bind(this), 500);
        this.editor.onDidChangeCursorSelection(e => {
            this.cursorSelectionThrottledFunction(e);
        });
    }

    override getDefaultPaneName() {
        return 'Detailed Compiler Flags';
    }

    override onCompiler(compilerId: number, compiler: any, options: unknown, editorId: number, treeId: number) {
        if (compilerId === this.compilerInfo.compilerId) {
            this.compilerInfo.compilerName = compiler ? compiler.name : '';
            this.compilerInfo.editorId = editorId;
            this.compilerInfo.treeId = treeId;
            this.updateTitle();
        }
    }

    override onCompileResult(compilerId: number, compiler: unknown, result: unknown) {}

    override close() {
        this.eventHub.unsubscribe();
        this.eventHub.emit('flagsViewClosed', this.compilerInfo.compilerId, this.getOptions());
        this.editor.dispose();
    }

    override onCompilerClose(compilerId: number) {
        if (compilerId === this.compilerInfo.compilerId) {
            // We can't immediately close as an outer loop somewhere in GoldenLayout is iterating over
            // the hierarchy. We can't modify while it's being iterated over.
            this.close();
            _.defer(() => {
                this.container.close();
            });
        }
    }

    override onSettingsChange(newSettings: SiteSettings) {
        super.onSettingsChange(newSettings);
        this.debouncedEmitChange = _.debounce(this.maybeEmitChange.bind(this), newSettings.delayAfterChange);
    }

    getOptions() {
        const lines = this.editor.getModel()?.getValue();
        return lines ? lines.replace(/\n/g, ' ') : ''; // TODO
    }

    override getCurrentState() {
        const parent = super.getCurrentState();
        const state = {
            compilerFlags: this.getOptions(),
            ...parent,
        };
        // FIXME: need some way to have this return S & MonacoPaneState
        // or maybe have an abstract function in pane.ts returning S
        // that pane.ts can then merge. will require reworking pane.ts'
        // interface a bit either way
        return state as MonacoPaneState;
    }

    maybeEmitChange(force) {
        const options = this.getOptions();
        if (!force && options === this.lastChangeEmitted) return;

        this.lastChangeEmitted = options;
        this.eventHub.emit('compilerFlagsChange', this.compilerInfo.compilerId, this.lastChangeEmitted);
    }
}
