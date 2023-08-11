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
import {Container} from 'golden-layout';
import * as monaco from 'monaco-editor';

import {MonacoPaneState, PaneCompilerState, PaneState} from './pane.interfaces.js';

import {FontScale} from '../widgets/fontscale.js';
import {Settings, SiteSettings} from '../settings.js';
import * as utils from '../utils.js';

import {PaneRenaming} from '../widgets/pane-renaming.js';
import {EventHub} from '../event-hub.js';
import {Hub} from '../hub.js';
import {unwrap} from '../assert.js';
import {CompilerInfo} from '../compiler.interfaces.js';
import {CompilationResult} from '../compilation/compilation.interfaces.js';
import {escapeHTML} from '../../shared/common-utils.js';

/**
 * Basic container for a tool pane in Compiler Explorer.
 *
 * Type parameter S refers to a state interface for the pane
 */
export abstract class Pane<S> {
    compilerInfo: PaneCompilerState;
    container: Container;
    domRoot: JQuery;
    topBar: JQuery;
    hideable: JQuery;
    protected hub: Hub;
    eventHub: EventHub;
    isAwaitingInitialResults = false;
    settings: SiteSettings;
    paneName: string | undefined = undefined;
    paneRenaming: PaneRenaming;

    /**
     * Base constructor for any pane. Performs common initialization tasks such
     * as registering standard event listeners and lifecycle handlers.
     *
     * Overridable for implementors
     */
    protected constructor(hub: Hub, container: Container, state: S & PaneState) {
        this.container = container;
        this.hub = hub;
        this.eventHub = hub.createEventHub();
        this.domRoot = container.getElement();
        this.domRoot.html(this.getInitialHTML());

        this.hideable = this.domRoot.find('.hideable');

        this.initializeCompilerInfo(state);
        this.topBar = this.domRoot.find('.top-bar');

        this.paneRenaming = new PaneRenaming(this, state);

        this.initializeDefaults();
        this.initializeGlobalDependentProperties();
        this.initializeStateDependentProperties(state);

        this.registerDynamicElements(state);

        this.registerButtons(state);
        this.registerStandardCallbacks();
        this.registerCallbacks();
        this.registerOpeningAnalyticsEvent();
    }

    protected initializeCompilerInfo(state: PaneState) {
        this.compilerInfo = {
            compilerId: state.id,
            compilerName: state.compilerName,
            editorId: state.editorid,
            treeId: state.treeid,
        };
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
    abstract registerOpeningAnalyticsEvent(): void;

    initializeDefaults(): void {}

    initializeGlobalDependentProperties(): void {
        this.settings = Settings.getStoredSettings();
    }

    initializeStateDependentProperties(state: S): void {}

    /** Optional overridable code for initializing necessary elements before rest of registers **/
    registerDynamicElements(state: S): void {}

    /** Optionally overridable code for initializing pane buttons */
    registerButtons(state: S): void {}

    /** Optionally overridable code for initializing event callbacks */
    registerCallbacks(): void {}

    /**
     * Handle user selected compiler change.
     *
     * This event is triggered when the user selects a different compiler in the
     * compiler dropdown.
     *
     * Note that this event is also triggered when the changed compiler is not
     * the one this view is attached to. Therefore, it is smart to check that
     * the updated compiler is the one the view is attached to. This can be done
     * with a simple check.
     *
     * ```ts
     * if (this.compilerInfo.compilerId === compilerId) { ... }
     * ```
     *
     * @param compilerId - Id of the compiler that had its version changed
     * @param compiler - The updated compiler object
     * @param options - User commandline args
     * @param editorId - The editor id the updated compiler is attached to
     */
    abstract onCompiler(
        compilerId: number,
        compiler: CompilerInfo | null,
        options: string,
        editorId: number,
        treeId: number,
    ): void;

    /**
     * Handle compilation result.
     *
     * This event is triggered when a code compilation was triggered.
     *
     * Note that this event is triggered for *any* compilation, even when the
     * compilation was done for a source/compiler this view is not attached to.
     * Therefore, it is smart to check that the updated compiler is the one the
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
    abstract onCompileResult(compilerId: number, compiler: CompilerInfo, result: CompilationResult): void;

    /**
     * Perform any clean-up events when the pane is closed.
     *
     * This is typically used to emit an analytics event for closing the pane,
     * unsubscribing from the event hub and disposing the monaco editor.
     */
    abstract close(): void;

    /** Initialize standard lifecycle hooks */
    protected registerStandardCallbacks(): void {
        this.paneRenaming.on('renamePane', this.updateState.bind(this));
        this.container.on('destroy', this.close.bind(this));
        this.container.on('resize', this.resize.bind(this));
        this.eventHub.on('compileResult', this.onCompileResult.bind(this));
        this.eventHub.on('compiler', this.onCompiler.bind(this));
        this.eventHub.on('compilerClose', this.onCompilerClose.bind(this));
        this.eventHub.on('settingsChange', this.onSettingsChange.bind(this));
        this.eventHub.on('shown', this.resize.bind(this));
        this.eventHub.on('resize', this.resize.bind(this));
    }

    /**
     * Produce a default name for the pane. Typical implementation
     * looks like this:
     *
     * ```ts
     * return 'Rust MIR Viewer';
     * ```
     */
    abstract getDefaultPaneName(): string;

    /** Generate "(Editor #1, Compiler #1)" tag */
    protected getPaneTag() {
        const {compilerName, editorId, treeId, compilerId} = this.compilerInfo;
        if (editorId) {
            return `${compilerName} (Editor #${editorId}, Compiler #${compilerId})`;
        } else {
            return `${compilerName} (Tree #${treeId}, Compiler #${compilerId})`;
        }
    }

    /** Get name for the pane */
    protected getPaneName() {
        return this.paneName ?? this.getDefaultPaneName() + ' ' + this.getPaneTag();
    }

    /** Update the pane's title, called when the pane name or compiler info changes */
    protected updateTitle() {
        this.container.setTitle(escapeHTML(this.getPaneName()));
    }

    /** Close the pane if the compiler this pane was attached to closes */
    protected onCompilerClose(compilerId: number) {
        if (this.compilerInfo.compilerId === compilerId) {
            _.defer(() => this.container.close());
        }
    }

    protected onSettingsChange(settings: SiteSettings) {
        this.settings = settings;
    }

    getCurrentState(): PaneState {
        const state = {
            id: this.compilerInfo.compilerId,
            compilerName: this.compilerInfo.compilerName,
            editorid: this.compilerInfo.editorId,
            treeid: this.compilerInfo.treeId,
        };
        this.paneRenaming.addState(state);
        return state;
    }

    updateState() {
        this.container.setState(this.getCurrentState());
    }

    abstract resize(): void;
}

/**
 * Basic container for a tool pane with a monaco editor in Compiler Explorer.
 *
 * Type parameter E indicates which monaco editor kind this pane hosts. Common
 * values are monaco.editor.IDiffEditor and monaco.ICodeEditor
 *
 * Type parameter S refers to a state interface for the pane
 */
export abstract class MonacoPane<E extends monaco.editor.IEditor, S> extends Pane<S> {
    editor: E;
    selection: monaco.Selection | undefined = undefined;
    fontScale: FontScale;

    protected constructor(hub: Hub, container: Container, state: S & MonacoPaneState) {
        super(hub, container, state);
        this.selection = state.selection;

        this.registerEditorActions();
    }

    override registerButtons(state: S): void {
        const editorRoot = this.domRoot.find('.monaco-placeholder')[0];
        this.editor = this.createEditor(editorRoot);
        this.fontScale = new FontScale(this.domRoot, state, this.editor);
    }

    override getCurrentState(): MonacoPaneState {
        const parent = super.getCurrentState();
        const state: MonacoPaneState = {
            selection: this.selection,
            ...parent,
        };
        this.fontScale.addState(state);
        return state;
    }

    resize() {
        _.defer(() => {
            const topBarHeight = utils.updateAndCalcTopBarHeight(this.domRoot, this.topBar, this.hideable);
            this.editor.layout({
                width: unwrap(this.domRoot.width()),
                height: unwrap(this.domRoot.height()) - topBarHeight,
            });
        });
    }

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

    protected override onSettingsChange(settings: SiteSettings) {
        super.onSettingsChange(settings);
        this.editor.updateOptions({
            contextmenu: settings.useCustomContextMenu,
            minimap: {
                enabled: settings.showMinimap,
            },
            fontFamily: settings.editorsFFont,
            fontLigatures: settings.editorsFLigatures,
        });
    }

    protected onDidChangeCursorSelection(event: monaco.editor.ICursorSelectionChangedEvent) {
        if (this.isAwaitingInitialResults) {
            this.selection = event.selection;
            this.updateState();
        }
    }

    /** Initialize standard lifecycle hooks */
    protected override registerStandardCallbacks(): void {
        super.registerStandardCallbacks();
        // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
        if (this.fontScale) this.fontScale.on('change', this.updateState.bind(this));
        this.eventHub.on('broadcastFontScale', (scale: number) => {
            this.fontScale.setScale(scale);
            this.updateState();
        });
        this.eventHub.on('printrequest', this.sendPrintData, this);
    }

    /**
     * Optionally overridable code for initializing monaco actions on the
     * editor instance
     */
    registerEditorActions(): void {}

    /**
     * Utility function to check if this is a code editor or something else (like a diff editor)
     */
    protected isStandaloneEditor(editor: monaco.editor.IEditor): editor is monaco.editor.IStandaloneCodeEditor {
        return editor.getEditorType() === 'vs.editor.ICodeEditor';
    }
    /**
     * Get the name of the pane to be displayed in the print view
     */
    abstract getPrintName(): string;

    /**
     * Send any printable content to the print view when requested
     */
    protected sendPrintData() {
        const editor = this.editor;
        if (this.isStandaloneEditor(editor)) {
            const model = (editor as monaco.editor.IStandaloneCodeEditor).getModel();
            if (model) {
                const lines = [...new Array(model.getLineCount()).keys()].map(i =>
                    monaco.editor.colorizeModelLine(model, i + 1),
                );
                const extra = this.getExtraPrintData();
                this.eventHub.emit(
                    'printdata',
                    `<h1>${this.getPrintName()}: ${escapeHTML(this.getPaneName())}</h1>` +
                        (extra ?? '') +
                        `<code>${lines.join('<br/>\n')}</code>`,
                );
            }
        }
    }

    /**
     * Provide additional info to be included below the header in the default sendPrintData
     */
    protected getExtraPrintData(): string | undefined {
        return undefined;
    }
}
