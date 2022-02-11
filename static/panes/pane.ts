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
import { SiteSettings } from '../settings';
import * as utils from '../utils';

import { PaneRenaming } from '../pane-renaming';

/**
 * Basic container for a tool pane in Compiler Explorer
 *
 * Type parameter E indicates which monaco editor kind this pane hosts. Common
 * values are monaco.editor.IDiffEditor and monaco.ICodeEditor
 *
 * Type parameter S refers to a state interface for the pane
 */
export abstract class Pane<E extends monaco.editor.IEditor, S> {
    compilerInfo: PaneCompilerState;
    container: Container;
    domRoot: JQuery;
    topBar: JQuery;
    hideable: JQuery;
    eventHub: any /* typeof hub.createEventHub() */;
    selection: monaco.Selection;
    editor: E;
    fontScale: FontScale;
    isAwaitingInitialResults = false;
    settings: SiteSettings | Record<string, never> = {};
    paneName: string;

    /**
     * Base constructor for any pane. Performs common initialization tasks such
     * as registering standard event listeners and lifecycle handlers.
     *
     * Overridable for implementors
     */
    protected constructor(hub: any /* Hub */, container: Container, state: S & BasePaneState) {
        this.container = container;
        this.eventHub = hub.createEventHub();
        this.domRoot = container.getElement();
        this.hideable = this.domRoot.find('.hideable');

        this.domRoot.html(this.getInitialHTML());
        const editorRoot = this.domRoot.find('.monaco-placeholder')[0];
        this.editor = this.createEditor(editorRoot);

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
        this.registerEditorActions();
        this.updateTitle();
        this.registerOpeningAnalyticsEvent();
    }

    /**
     * Get the initial HTML layout for the pane's default content. A typical
     * implementation looks like this:
     *
     * ```ts
     * override getInitialHTML(): string {
     *     return $('#rustmir').html();
     * }
     * ```
     */
    abstract getInitialHTML(): string;

    /**
     * Initialize the monaco editor instance. Typical implementation for looks
     * like this:
     *
     * ```ts
     * return monaco.editor.create(editorRoot, extendConfig({
     *     // goodies
     * }));
     * ```
     */
    abstract createEditor(editorRoot: HTMLElement): E;

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
    registerButtons(state: S): void {}

    /** Optionally overridable code for initializing event callbacks */
    registerCallbacks(): void {}

    /**
     * Optionally overridable code for initializing monaco actions on the
     * editor instance
     */
    registerEditorActions(): void {}

    /**
     * Produce a textual title for the pane
     *
     * Typical implementation uses the compiler and editor ids in combination
     * with a name.
     *
     * This title is attached to the pane in the UI.
     *
     * ```ts
     * return `Rust MIR Viewer ${this.compilerInfo.compilerName}` +
     *     `(Editor #${this.compilerInfo.editorId}, ` +
     *     `Compiler #${this.compilerInfo.compilerId})`;
     * ```
     */
    abstract getPaneName(): string;

    /**
     * Handle user selected compiler change.
     *
     * This event is triggered when the user selects a different compiler in the
     * compiler dropdown.
     *
     * Note that this event is also triggered when the changed compiler is not
     * the one this view is attached to. Therefore it is smart to check that
     * the updated compiler is the one the view is attached to. This can be done
     * with a simple check.
     *
     * ```ts
     * if (this.compilerInfo.compilerId === compilerId) { ... }
     * ```
     *
     * @param compilerId - Id of the compiler that had its version changed
     * @param compiler - The updated compiler object
     * @param options
     * @param editorId - The editor id the updated compiler is attached to
     */
    abstract onCompiler(compilerId: number, compiler: unknown, options: unknown, editorId: number): void;

    /**
     * Handle compilation result.
     *
     * This event is triggered when a code compilation was triggered.
     *
     * Note that this event is triggered for *any* compilation, even when the
     * compilation was done for a source/compiler this view is not attached to.
     * Therefore it is smart to check that the updated compiler is the one the
     * view is attached to. This can be done with a simple check.
     *
     * ```ts
     * if (this.compilerInfo.compilerId === compilerId) { ... }
     * ```
     *
     * @param compilerId - Id of the compiler that had a compilation
     * @param compiler - The compiler object
     * @param result - The entire HTTP request response
     */
    abstract onCompileResult(compilerId: number, compiler: unknown, result: unknown): void;

    /**
     * Perform any clean-up events when the pane is closed.
     *
     * This is typically used to emit an analytics event for closing the pane,
     * unsubscribing from the event hub and disposing the monaco editor.
     */
    abstract close(): void;

    /** Initialize standard lifecycle hooks */
    protected registerStandardCallbacks(): void {
        this.fontScale.on('change', this.updateState.bind(this));
        this.container.on('destroy', this.close.bind(this));
        this.container.on('resize', this.resize.bind(this));
        PaneRenaming.registerCallback(this);
        this.eventHub.on('compileResult', this.onCompileResult.bind(this));
        this.eventHub.on('compiler', this.onCompiler.bind(this));
        this.eventHub.on('compilerClose', this.onCompilerClose.bind(this));
        this.eventHub.on('settingsChange', this.onSettingsChange.bind(this));
        this.eventHub.on('shown', this.resize.bind(this));
        this.eventHub.on('resize', this.resize.bind(this));
    }

    protected updateTitle() {
        const name = this.paneName ? this.paneName : this.getPaneName();
        this.container.setTitle(_.escape(name));
    }

    /** Close the pane if the compiler this pane was attached to closes */
    protected onCompilerClose(compilerId: number) {
        if (this.compilerInfo.compilerId === compilerId) {
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
