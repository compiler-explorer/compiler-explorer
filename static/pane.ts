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
import { Container } from 'golden-layout';
import monaco from 'monaco-editor';

import { FontScale } from './fontscale'
import {BasicPane, OpaqueState, PaneCompilerState} from './pane.interfaces';
import { SiteSettings } from './settings.interrfaces';

export abstract class Pane<E extends monaco.editor.ICodeEditor = monaco.editor.IStandaloneCodeEditor> implements BasicPane {
    compilerInfo: PaneCompilerState;
    container: Container;
    domRoot: JQuery;
    topBar: JQuery;
    eventHub: any /* typeof hub.createEventHub() */;
    selection: monaco.ISelection;
    editor: E;
    fontScale: typeof FontScale
    isAwaitingInitialResults: boolean = true;
    settings: SiteSettings | {} = {};

    /**
     *
     * @param hub
     * @param state
     * @param container
     * @protected
     */
    protected constructor(hub: any /* Hub */, state: OpaqueState, container: Container) {
        this.container = container;
        this.eventHub = hub.createEventHub();
        this.domRoot = container.getElement();
        this.initializeDomRoot();

        const editorRoot = this.domRoot.find('.monaco-placeholder')[0];
        this.initializeEditor(editorRoot);

        this.selection = state.selection
        this.compilerInfo = {
            compilerId: state.id,
            compilerName: state.compilerName,
            editorId: state.editorid,
        }
        this.fontScale = new FontScale(this.domRoot, state, this.editor);
        this.topBar = this.domRoot.find('.top-bar');

        this.initButtons(state);
        this.initStandardCallbacks();
        this.initCallbacks();
        this.setTitle();
    }

    /**
     * Initialize the DOM root node with the pane's default content. Typical
     * implementation looks like this:
     *
     * ```ts
     * this.domRoot.html($('#rustmir').html());
     * ```
     */
    abstract initializeDomRoot(): void;

    /**
     * Initialize the monaco editor instance. Typical implementation for looks
     * like this:
     *
     * ```ts
     * this.editor = monaco.editor.create(editorRoot, extendConfig({
     *     // goodies
     * }));
     * ```
     */
    abstract initializeEditor(editorRoot: HTMLElement): void;

    /**
     * Emit analytics event for opening the pane tab. Typical implementation
     * looks like this:
     *
     * ```ts
     * ga.proxy('send', {
     *   hitType: 'event',
     *   eventCategory: 'OpenViewPane',
     *   eventAction: 'RustMir',
     * });
     * ```
     */
    abstract emitOpeningAnalyticsEvent(): void

    abstract initButtons(state: unknown /* typeof state */);
    abstract initCallbacks();
    abstract getPaneName(): string;
    abstract onCompiler(id: unknown, compiler: unknown, options: unknown, editorId: unknown);
    abstract onCompileResult(id: unknown, compiler: unknown, result: unknown);
    abstract close();

    initStandardCallbacks() {
        this.fontScale.on('change', () => this.updateState());
        this.container.on('destroy', () => this.close());
        this.container.on('resize', () => this.resize());
        this.eventHub.on('compileResult', (id, compiler, result) => {
            this.onCompileResult(id, compiler, result);
        });
        this.eventHub.on('compiler', (id, compiler, options, editorId) => {
            this.onCompiler(id, compiler, options, editorId);
        });
        this.eventHub.on('compilerClose', (id) => this.onCompilerClose(id));
        this.eventHub.on('settingsChange', (settings) => this.onSettingsChange(settings));
        this.eventHub.on('shown', () => this.resize());
        this.eventHub.on('resize', () => this.resize());
    }

    setTitle() {
        this.container.setTitle(this.getPaneName());
    }

    resize() {
        const topBarHeight = this.topBar.outerHeight(true);
        this.editor.layout({
            width: this.domRoot.width(),
            height: this.domRoot.height() - topBarHeight,
        });
    }

    onCompilerClose(id: unknown) {
        if (this.compilerInfo.compilerId === id) {
            _.defer(() => this.container.close())
        }
    }

    onDidChangeCursorSelection(event: monaco.editor.ICursorSelectionChangedEvent) {
        if (this.isAwaitingInitialResults) {
            this.selection = event.selection
            this.updateState();
        }
    }

    onSettingsChange(settings: SiteSettings) {
        this.settings = settings;
        this.editor.updateOptions({
            contextmenu: settings.useCustomContextMenu,
            minimap: {
                enabled: settings.showMinimap,
            },
            fontFamily: settings.editorFFont,
            fontLigatures: settings.editorsFLigatures,
        })
    }

    getCurrentState() {
        const state = {
            id: this.compilerInfo.compilerId,
            editorId: this.compilerInfo.editorId,
            selection: this.selection,
        };
        this.fontScale.addState(state);
        return state;
    }

    updateState() {
        this.container.setState(this.getCurrentState());
    }
}
