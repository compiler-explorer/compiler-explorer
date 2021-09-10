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
import * as monaco from 'monaco-editor';

import { BasePaneState, PaneCompilerState } from './pane.interfaces';

import { FontScale } from '../fontscale';
import { SiteSettings } from '../settings.interfaces';
import * as utils from '../utils';

/**
 * Basic container for a tool pane in Compiler Explorer
 *
 * Type parameter E indicates which monaco editor kind this pane hosts. Common
 * values are monaco.editor.IDiffEditor and monaco.ICodeEditor
 */
export abstract class Pane<E extends monaco.editor.IEditor> {
    compilerInfo: PaneCompilerState;
    container: Container;
    domRoot: JQuery;
    topBar: JQuery;
    hideable: JQuery;
    eventHub: any /* typeof hub.createEventHub() */;
    selection: monaco.Selection;
    editor: E;
    fontScale: typeof FontScale;
    isAwaitingInitialResults: boolean = false;
    settings: SiteSettings | {} = {};

    /**
     * Base constructor for any pane. Performs common initialization tasks such
     * as registering standard event listeners and lifecycle handlers.
     *
     * Overridable for implementors
     */
    protected constructor(hub: any /* Hub */, container: Container, state: BasePaneState) {
        this.container = container;
        this.eventHub = hub.createEventHub();
        this.domRoot = container.getElement();
        this.hideable = this.domRoot.find('.hideable');
        this.initializeDOMRoot();

        const editorRoot = this.domRoot.find('.monaco-placeholder')[0];
        this.createEditor(editorRoot);

        this.selection = state.selection;
        this.compilerInfo = {
            compilerId: state.id,
            compilerName: state.compilerName,
            editorId: state.editorid,
        };
        this.fontScale = new FontScale(this.domRoot, state, this.editor);
        this.topBar = this.domRoot.find('.top-bar');

        this.registerButtons(state);
        this.registerStandardCallbacks();
        this.registerCallbacks();
        this.setTitle();
        this.registerOpeningAnalyticsEvent();
    }

    /**
     * Initialize the DOM root node with the pane's default content. Typical
     * implementation looks like this:
     *
     * ```ts
     * this.domRoot.html($('#rustmir').html());
     * ```
     */
    abstract initializeDOMRoot(): void;

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
    abstract createEditor(editorRoot: HTMLElement): void;

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
    abstract registerOpeningAnalyticsEvent(): void

    /** Optionally overridable code for initializing pane buttons */
    registerButtons(state: BasePaneState /* typeof state */): void {}

    /** Optionally overridable code for initializing event callbacks */
    registerCallbacks(): void {}

    abstract getPaneName(): string;
    abstract onCompiler(id: number, compiler: any, options: any, editorId: number): void;
    abstract onCompileResult(id: unknown, compiler: unknown, result: unknown): void;
    abstract close(): void;

    /** Initialize standard lifecycle hooks */
    protected registerStandardCallbacks(): void {
        this.fontScale.on('change', this.updateState.bind(this));
        this.container.on('destroy', this.close.bind(this));
        this.container.on('resize', this.resize.bind(this));
        this.eventHub.on('compileResult', this.onCompileResult.bind(this));
        this.eventHub.on('compiler', this.onCompiler.bind(this));
        this.eventHub.on('compilerClose', this.onCompilerClose.bind(this));
        this.eventHub.on('settingsChange', this.onSettingsChange.bind(this));
        this.eventHub.on('shown', this.resize.bind(this));
        this.eventHub.on('resize', this.resize.bind(this));
    }

    protected setTitle() {
        this.container.setTitle(this.getPaneName());
    }

    protected onCompilerClose(id: unknown) {
        if (this.compilerInfo.compilerId === id) {
            _.defer(() => this.container.close());
        }
    }

    protected onDidChangeCursorSelection(event: monaco.editor.ICursorSelectionChangedEvent) {
        if (this.isAwaitingInitialResults) {
            this.selection = event.selection;
            this.updateState();
        }
    }

    protected onSettingsChange(settings: SiteSettings) {
        this.settings = settings;
        this.editor.updateOptions({
            contextmenu: settings.useCustomContextMenu,
            minimap: {
                enabled: settings.showMinimap,
            },
            fontFamily: settings.editorsFFont,
            fontLigatures: settings.editorsFLigatures,
        });
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

    resize() {
        const topBarHeight = utils.updateAndCalcTopBarHeight(this.domRoot,
            this.topBar, this.hideable);
        if (!this.editor) return;
        this.editor.layout({
            width: this.domRoot.width(),
            height: this.domRoot.height() - topBarHeight,
        });
    }
}
