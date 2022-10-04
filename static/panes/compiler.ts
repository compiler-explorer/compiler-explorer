// Copyright (c) 2012, Compiler Explorer Authors
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
import {ga} from '../analytics';
import * as colour from '../colour';
import {Toggles} from '../widgets/toggles';
import {FontScale} from '../widgets/fontscale';
import * as Components from '../components';
import LruCache from 'lru-cache';
import {options} from '../options';
import * as monaco from 'monaco-editor';
import {Alert} from '../alert';
import bigInt from 'big-integer';
import {LibsWidget} from '../widgets/libs-widget';
import * as codeLensHandler from '../codelens-handler';
import * as monacoConfig from '../monaco-config';
import * as TimingWidget from '../widgets/timing-info-widget';
import {CompilerPicker} from '../compiler-picker';
import {CompilerService} from '../compiler-service';
import {Settings} from '../settings';
import * as utils from '../utils';
import * as LibUtils from '../lib-utils';
import {getAssemblyDocumentation} from '../api/api';
import {PaneRenaming} from '../widgets/pane-renaming';
import {MonacoPane} from './pane';
import {CompilerInfo} from '../../types/compiler.interfaces';
import {MonacoPaneState} from './pane.interfaces';
import {Hub} from '../hub';
import {Container} from 'golden-layout';
import {CompilerState} from './compiler.interfaces';
const toolIcons = require.context('../../views/resources/logos', false, /\.(png|svg)$/);

const OpcodeCache = new LruCache({
    max: 64 * 1024,
    length: function (n) {
        return JSON.stringify(n).length;
    },
});

function patchOldFilters(filters) {
    if (filters === undefined) return undefined;
    // Filters are of the form {filter: true|false¸ ...}. In older versions, we used
    // to suppress the {filter:false} form. This means we can't distinguish between
    // "filter not on" and "filter not present". In the latter case we want to default
    // the filter. In the former case we want the filter off. Filters now don't suppress
    // but there are plenty of permalinks out there with no filters set at all. Here
    // we manually set any missing filters to 'false' to recover the old behaviour of
    // "if it's not here, it's off".
    _.each(['binary', 'labels', 'directives', 'commentOnly', 'trim', 'intel'], function (oldFilter) {
        if (filters[oldFilter] === undefined) filters[oldFilter] = false;
    });
    return filters;
}

const languages = options.languages;

type LastResultType = {
    ppOutput: unknown;
    astOutput: unknown;
    irOutput: unknown;
    devices: unknown;
    rustMirOutput: unknown;
    rustMacroExpOutput: unknown;
    rustHirOutput: unknown;
    haskellCoreOutput: unknown;
    haskellStgOutput: unknown;
    haskellCmmOutput: unknown;
    gccDumpOutput: unknown;
    gnatDebugTreeOutput: unknown;
    gnatDebugOutput: unknown;
    optOutput: unknown;
};

type CurrentStateType = {
    id?: number;
    compiler: string;
    source: number | null;
    tree: any;
    options: unknown;
    filters: Record<string, unknown>;
    wantOptInfo: any;
    libs?: {name: string, ver: string}[];
    lang: any;
    selection: monaco.Selection | undefined;
    flagsViewOpen: any;
};

// Disable max line count only for the constructor. Turns out, it needs to do quite a lot of things
// eslint-disable-next-line max-statements
export class Compiler extends MonacoPane<monaco.editor.IStandaloneCodeEditor, CompilerState> {
    private compilerService: any;
    private id: number;
    private sourceTreeId: any;
    private sourceEditorId: number | null;
    private originalCompilerId: any;
    private infoByLang: {};
    private deferCompiles: boolean;
    private needsCompile: boolean;
    private deviceViewOpen: boolean;
    private options: unknown;
    private source: string;
    private assembly: any[];
    private colours: any[];
    private lastResult: LastResultType;
    private lastTimeTaken: number;
    private pendingRequestSentAt: number;
    private pendingCMakeRequestSentAt: number;
    private nextRequest: null;
    private nextCMakeRequest: null;
    private flagsViewOpen: any;
    private optViewOpen: boolean;
    private cfgViewOpen: boolean;
    private wantOptInfo: any;
    private decorations: {};
    private prevDecorations: any[];
    private labelDefinitions: {};
    private alertSystem: Alert;
    private awaitingInitialResults: boolean;
    private linkedFadeTimeoutId: number;
    private toolsMenu: JQuery<HTMLElement> | null;
    private revealJumpStack: any[];
    private compilerPicker: CompilerPicker;
    private compiler: CompilerInfo | null;
    private currentLangId: any;
    private filters: Toggles;
    private optButton: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private flagsButton: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private ppButton: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private astButton: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private irButton: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private llvmOptPipelineButton: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private deviceButton: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private gnatDebugTreeButton: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private gnatDebugButton: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private rustMirButton: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private rustMacroExpButton: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private haskellCoreButton: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private haskellStgButton: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private haskellCmmButton: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private gccDumpButton: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private cfgButton: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private executorButton: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private libsButton: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private compileInfoLabel: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private compileClearCache: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private outputBtn: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private outputTextCount: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private outputErrorCount: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private optionsField: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private initialOptionsFieldPlacehoder: JQuery<HTMLElement>;
    private prependOptions: JQuery<HTMLElement>;
    private fullCompilerName: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private fullTimingInfo: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private compilerLicenseButton: JQuery<HTMLElement>;
    private filterBinaryButton: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private filterBinaryTitle: JQuery<HTMLElement>;
    private filterExecuteButton: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private filterExecuteTitle: JQuery<HTMLElement>;
    private filterLabelsButton: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private filterLabelsTitle: JQuery<HTMLElement>;
    private filterDirectivesButton: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private filterDirectivesTitle: JQuery<HTMLElement>;
    private filterLibraryCodeButton: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private filterLibraryCodeTitle: JQuery<HTMLElement>;
    private filterCommentsButton: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private filterCommentsTitle: JQuery<HTMLElement>;
    private filterTrimButton: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private filterTrimTitle: JQuery<HTMLElement>;
    private filterIntelButton: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private filterIntelTitle: JQuery<HTMLElement>;
    private filterDemangleButton: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private filterDemangleTitle: JQuery<HTMLElement>;
    private noBinaryFiltersButtons: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private shortCompilerName: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private bottomBar: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private statusLabel: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private statusIcon: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private monacoPlaceholder: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private rustHirButton: JQuery<HTMLElement>;
    private libsWidget: LibsWidget | null;
    private isLabelCtxKey: monaco.editor.IContextKey<boolean>;
    private revealJumpStackHasElementsCtxKey: monaco.editor.IContextKey<boolean>;
    private isAsmKeywordCtxKey: monaco.editor.IContextKey<boolean>;

    private ppViewOpen: boolean;
    private astViewOpen: boolean;
    private irViewOpen: boolean;
    private llvmOptPipelineViewOpen: boolean;
    private gccDumpViewOpen: boolean;
    private gccDumpPassSelected: boolean;
    private treeDumpEnabled: boolean;
    private rtlDumpEnabled: boolean;
    private ipaDumpEnabled: boolean;
    private dumpFlags: any;
    private gnatDebugTreeViewOpen: boolean;
    private gnatDebugViewOpen: boolean;
    private rustMirViewOpen: boolean;
    private rustMacroExpViewOpen: boolean;
    private rustHirViewOpen: boolean;
    private haskellCoreViewOpen: boolean;
    private haskellStgViewOpen: boolean;
    private haskellCmmViewOpen: boolean;

    constructor(private readonly hub: Hub, container: Container, state: MonacoPaneState & CompilerState) {
        super(hub, container, state);
        this.compilerService = hub.compilerService;
        this.id = state.id || hub.nextCompilerId();
        this.sourceTreeId = state.tree ? state.tree : false;
        if (this.sourceTreeId) {
            this.sourceEditorId = null;
        } else {
            this.sourceEditorId = state.source || 1;
        }

        this.settings = Settings.getStoredSettings();
        this.originalCompilerId = state.compiler;
        this.initLangAndCompiler(state);
        this.infoByLang = {};
        this.deferCompiles = hub.deferred;
        this.needsCompile = false;
        this.deviceViewOpen = false;
        this.options = state.options || (options.compileOptions as any)[this.currentLangId];
        this.source = '';
        this.assembly = [];
        this.colours = [];
        this.lastResult = {
            astOutput: '',
            irOutput: '',
            gccDumpOutput: '',
            gnatDebugOutput: '',
            gnatDebugTreeOutput: '',
            rustMirOutput: '',
            rustMacroExpOutput: '',
            rustHirOutput: '',
            haskellCoreOutput: '',
            haskellStgOutput: '',
            haskellCmmOutput: '',
            devices: '',
            optOutput: '',
            ppOutput: '',
        };

        this.lastTimeTaken = 0;
        this.pendingRequestSentAt = 0;
        this.pendingCMakeRequestSentAt = 0;
        this.nextRequest = null;
        this.nextCMakeRequest = null;
        this.optViewOpen = false;
        this.flagsViewOpen = state.flagsViewOpen || false;
        this.cfgViewOpen = false;
        this.wantOptInfo = state.wantOptInfo;
        this.decorations = {};
        this.prevDecorations = [];
        this.labelDefinitions = {};
        this.alertSystem = new Alert();
        this.alertSystem.prefixMessage = 'Compiler #' + this.id;

        this.awaitingInitialResults = false;
        this.selection = state.selection;

        this.linkedFadeTimeoutId = -1;
        this.toolsMenu = null;

        this.revealJumpStack = [];

        this.paneRenaming = new PaneRenaming(this, state);

        this.initButtons(state);

        this.fontScale = new FontScale(this.domRoot, state, this.editor);
        this.compilerPicker = new CompilerPicker(
            this.domRoot,
            this.hub,
            this.currentLangId,
            this.compiler?.id ?? '',
            this.onCompilerChange.bind(this)
        );

        this.initLibraries(state);

        this.initEditorActions();
        this.initEditorCommands();

        this.initCallbacks();
        // Handle initial settings
        this.onSettingsChange(this.settings);
        this.sendCompiler();
        this.updateCompilerInfo();
        this.updateButtons();
        this.saveState();

        if (this.sourceTreeId) {
            this.compile();
        }
    }

    override getInitialHTML() {
        return $('#compiler').html();
    }

    override createEditor(editorRoot: HTMLElement) {
        var monacoDisassembly =
            (languages[this.currentLangId] ? languages[this.currentLangId].monacoDisassembly : null) || 'asm';

        return monaco.editor.create(
            editorRoot,
            monacoConfig.extendConfig(
                {
                    readOnly: true,
                    language: monacoDisassembly,
                    glyphMargin: !options.embedded,
                    //guides: false,
                    vimInUse: false,
                },
                this.settings
            )
        );
    }

    override registerOpeningAnalyticsEvent() {
        ga.proxy('send', {
            hitType: 'event',
            eventCategory: 'OpenViewPane',
            eventAction: 'Compiler',
        });
    }
    
    getEditorIdBySourcefile(sourcefile) {
        if (this.sourceTreeId) {
            var tree = this.hub.getTreeById(this.sourceTreeId);
            if (tree) {
                return tree.multifileService.getEditorIdByFilename(sourcefile.file);
            }
        } else {
            if (sourcefile !== null && (sourcefile.file === null || sourcefile.mainsource)) {
                return this.sourceEditorId;
            }
        }

        return false;
    };

    
    initLangAndCompiler(state) {
        var langId = state.lang;
        var compilerId = state.compiler;
        var result = this.compilerService.processFromLangAndCompiler(langId, compilerId);
        this.compiler = result.compiler;
        this.currentLangId = result.langId;
        this.updateLibraries();
    };

    
    override close() {
        codeLensHandler.unregister(this.id);
        this.eventHub.unsubscribe();
        this.eventHub.emit('compilerClose', this.id, this.sourceTreeId);
        this.editor.dispose();
    };

// eslint-disable-next-line max-statements
    
    initPanerButtons() {
        const outputConfig = Components.getOutput(this.id, this.sourceEditorId, this.sourceTreeId);

        this.container.layoutManager.createDragSource(this.outputBtn, outputConfig);
        this.outputBtn.click(
            () => {
                var insertPoint =
                    this.hub.findParentRowOrColumn(this.container) || this.container.layoutManager.root.contentItems[0];
                insertPoint.addChild(outputConfig);
            }
        );

        var cloneComponent = () => {
            var currentState = this.currentState();
            // Delete the saved id to force a new one
            delete currentState.id;
            return {
                type: 'component',
                componentName: 'compiler',
                componentState: currentState,
            };
        }
        var createOptView = () => {
            return Components.getOptViewWith(
                this.id,
                this.source as unknown as number,
                this.lastResult.optOutput,
                this.getCompilerName(),
                this.sourceEditorId,
                this.sourceTreeId
            );
        }

        var createFlagsView = () => {
            return Components.getFlagsViewWith(this.id, this.getCompilerName(), this.optionsField.val());
        }

        if (this.flagsViewOpen) {
            createFlagsView();
        }

        var createPpView = () => {
            return Components.getPpViewWith(
                this.id,
                this.source as unknown as number,
                this.lastResult.ppOutput,
                this.getCompilerName(),
                this.sourceEditorId ?? 0,
                this.sourceTreeId
            );
        };

        var createAstView = () => {
            return Components.getAstViewWith(
                this.id,
                this.source as unknown as number,
                this.lastResult.astOutput,
                this.getCompilerName(),
                this.sourceEditorId ?? 0,
                this.sourceTreeId
            );
        };

        var createIrView = () => {
            return Components.getIrViewWith(
                this.id,
                this.source,
                this.lastResult.irOutput,
                this.getCompilerName(),
                this.sourceEditorId ?? 0,
                this.sourceTreeId
            );
        };

        var createLLVMOptPipelineView = () => {
            return Components.getLLVMOptPipelineViewWith(
                this.id,
                this.getCompilerName(),
                this.sourceEditorId ?? 0,
                this.sourceTreeId
            );
        };

        var createDeviceView = () => {
            return Components.getDeviceViewWith(
                this.id,
                this.source,
                this.lastResult.devices,
                this.getCompilerName(),
                this.sourceEditorId ?? 0,
                this.sourceTreeId
            );
        };

        var createRustMirView = () => {
            return Components.getRustMirViewWith(
                this.id,
                this.source,
                this.lastResult.rustMirOutput,
                this.getCompilerName(),
                this.sourceEditorId ?? 0,
                this.sourceTreeId
            );
        };

        var createRustMacroExpView = () => {
            return Components.getRustMacroExpViewWith(
                this.id,
                this.source,
                this.lastResult.rustMacroExpOutput,
                this.getCompilerName(),
                this.sourceEditorId ?? 0,
                this.sourceTreeId
            );
        };

        var createRustHirView = () => {
            return Components.getRustHirViewWith(
                this.id,
                this.source,
                this.lastResult.rustHirOutput,
                this.getCompilerName(),
                this.sourceEditorId ?? 0,
                this.sourceTreeId
            );
        };

        var createHaskellCoreView = () => {
            return Components.getHaskellCoreViewWith(
                this.id,
                this.source,
                this.lastResult.haskellCoreOutput,
                this.getCompilerName(),
                this.sourceEditorId ?? 0,
                this.sourceTreeId
            );
        }

        var createHaskellStgView = () => {
            return Components.getHaskellStgViewWith(
                this.id,
                this.source,
                this.lastResult.haskellStgOutput,
                this.getCompilerName(),
                this.sourceEditorId ?? 0,
                this.sourceTreeId
            );
        }

        var createHaskellCmmView = () => {
            return Components.getHaskellCmmViewWith(
                this.id,
                this.source,
                this.lastResult.haskellCmmOutput,
                this.getCompilerName(),
                this.sourceEditorId ?? 0,
                this.sourceTreeId
            );
        }

        var createGccDumpView = () => {
            return Components.getGccDumpViewWith(
                this.id as unknown as string,
                this.getCompilerName(),
                this.sourceEditorId ?? 0,
                this.sourceTreeId,
                this.lastResult.gccDumpOutput
            );
        }

        var createGnatDebugTreeView = () => {
            return Components.getGnatDebugTreeViewWith(
                this.id,
                this.source,
                this.lastResult.gnatDebugTreeOutput,
                this.getCompilerName(),
                this.sourceEditorId ?? 0,
                this.sourceTreeId
            );
        };

        var createGnatDebugView = () => {
            return Components.getGnatDebugViewWith(
                this.id,
                this.source,
                this.lastResult.gnatDebugOutput,
                this.getCompilerName(),
                this.sourceEditorId ?? 0,
                this.sourceTreeId
            );
        };

        var createCfgView = () => {
            return Components.getCfgViewWith(this.id, this.sourceEditorId ?? 0, this.sourceTreeId);
        };

        const createExecutor = () => {
            const currentState = this.currentState();
            const editorId = currentState.source;
            const treeId = currentState.tree;
            const langId = currentState.lang;
            const compilerId = currentState.compiler;
            const libs = this.libsWidget?.getLibsInUse()?.map((item) => ({
                name: item.libId,
                ver: item.versionId,
            })) ?? [];

            return Components.getExecutorWith(editorId ?? 0, langId, compilerId, libs, currentState.options, treeId);
        };

        var newPaneDropdown = this.domRoot.find('.new-pane-dropdown');
        var togglePannerAdder = () => {
            newPaneDropdown.dropdown('toggle');
        };

        // Note that the .d.ts file lies. createDragSource returns the newly created DragSource
        this.container.layoutManager
            .createDragSource(this.domRoot.find('.btn.add-compiler'), cloneComponent())
            ._dragListener.on('dragStart', togglePannerAdder);

        this.domRoot.find('.btn.add-compiler').on('click',
            () => {
                var insertPoint =
                    this.hub.findParentRowOrColumn(this.container.parent) || this.container.layoutManager.root.contentItems[0];
                insertPoint.addChild(cloneComponent());
            }
        );

        this.container.layoutManager
            .createDragSource(this.optButton, createOptView())
            ._dragListener.on('dragStart', togglePannerAdder);

        this.optButton.click(
            () => {
                var insertPoint =
                    this.hub.findParentRowOrColumn(this.container.parent) || this.container.layoutManager.root.contentItems[0];
                insertPoint.addChild(createOptView());
            }
        );

        var popularArgumentsMenu = this.domRoot.find('div.populararguments div.dropdown-menu');
        this.container.layoutManager
            .createDragSource(this.flagsButton, createFlagsView())
            ._dragListener.on('dragStart', togglePannerAdder);

        this.flagsButton.click(
            () => {
                var insertPoint =
                    this.hub.findParentRowOrColumn(this.container.parent) || this.container.layoutManager.root.contentItems[0];
                insertPoint.addChild(createFlagsView());
            }
        );

        popularArgumentsMenu.append(this.flagsButton);

        this.container.layoutManager
            .createDragSource(this.ppButton, createPpView())
            ._dragListener.on('dragStart', togglePannerAdder);

        this.ppButton.click(
            () => {
                var insertPoint =
                    this.hub.findParentRowOrColumn(this.container.parent) || this.container.layoutManager.root.contentItems[0];
                insertPoint.addChild(createPpView());
            }
        );

        this.container.layoutManager
            .createDragSource(this.astButton, createAstView())
            ._dragListener.on('dragStart', togglePannerAdder);

        this.astButton.click(
            () => {
                var insertPoint =
                    this.hub.findParentRowOrColumn(this.container.parent) || this.container.layoutManager.root.contentItems[0];
                insertPoint.addChild(createAstView());
            }
        );

        this.container.layoutManager
            .createDragSource(this.irButton, createIrView())
            ._dragListener.on('dragStart', togglePannerAdder);

        this.irButton.click(
            () => {
                var insertPoint =
                    this.hub.findParentRowOrColumn(this.container.parent) || this.container.layoutManager.root.contentItems[0];
                insertPoint.addChild(createIrView());
            }
        );

        this.container.layoutManager
            .createDragSource(this.llvmOptPipelineButton, createLLVMOptPipelineView())
            ._dragListener.on('dragStart', togglePannerAdder);

        this.llvmOptPipelineButton.click(
            () =>{
                var insertPoint =
                    this.hub.findParentRowOrColumn(this.container.parent) || this.container.layoutManager.root.contentItems[0];
                insertPoint.addChild(createLLVMOptPipelineView());
            }
        );

        this.container.layoutManager
            .createDragSource(this.deviceButton, createDeviceView())
            ._dragListener.on('dragStart', togglePannerAdder);

        this.deviceButton.click(
            () =>{
                var insertPoint =
                    this.hub.findParentRowOrColumn(this.container.parent) || this.container.layoutManager.root.contentItems[0];
                insertPoint.addChild(createDeviceView());
            }
        );

        this.container.layoutManager
            .createDragSource(this.rustMirButton, createRustMirView())
            ._dragListener.on('dragStart', togglePannerAdder);

        this.rustMirButton.click(
            () =>{
                var insertPoint =
                    this.hub.findParentRowOrColumn(this.container.parent) || this.container.layoutManager.root.contentItems[0];
                insertPoint.addChild(createRustMirView());
            }
        );

        this.container.layoutManager
            .createDragSource(this.haskellCoreButton, createHaskellCoreView())
            ._dragListener.on('dragStart', togglePannerAdder);

        this.haskellCoreButton.click(
            () =>{
                var insertPoint =
                    this.hub.findParentRowOrColumn(this.container.parent) || this.container.layoutManager.root.contentItems[0];
                insertPoint.addChild(createHaskellCoreView());
            }
        );

        this.container.layoutManager
            .createDragSource(this.haskellStgButton, createHaskellStgView())
            ._dragListener.on('dragStart', togglePannerAdder);

        this.haskellStgButton.click(
            () =>{
                var insertPoint =
                    this.hub.findParentRowOrColumn(this.container.parent) || this.container.layoutManager.root.contentItems[0];
                insertPoint.addChild(createHaskellStgView());
            }
        );

        this.container.layoutManager
            .createDragSource(this.haskellCmmButton, createHaskellCmmView())
            ._dragListener.on('dragStart', togglePannerAdder);

        this.haskellCmmButton.click(
            () =>{
                var insertPoint =
                    this.hub.findParentRowOrColumn(this.container.parent) || this.container.layoutManager.root.contentItems[0];
                insertPoint.addChild(createHaskellCmmView());
            }
        );

        this.container.layoutManager
            .createDragSource(this.rustMacroExpButton, createRustMacroExpView())
            ._dragListener.on('dragStart', togglePannerAdder);

        this.rustMacroExpButton.click(
            () =>{
                var insertPoint =
                    this.hub.findParentRowOrColumn(this.container.parent) || this.container.layoutManager.root.contentItems[0];
                insertPoint.addChild(createRustMacroExpView());
            }
        );

        this.container.layoutManager
            .createDragSource(this.rustHirButton, createRustHirView())
            ._dragListener.on('dragStart', togglePannerAdder);

        this.rustHirButton.click(
            () =>{
                var insertPoint =
                    this.hub.findParentRowOrColumn(this.container.parent) || this.container.layoutManager.root.contentItems[0];
                insertPoint.addChild(createRustHirView());
            }
        );

        this.container.layoutManager
            .createDragSource(this.gccDumpButton, createGccDumpView())
            ._dragListener.on('dragStart', togglePannerAdder);

        this.gccDumpButton.click(
            () =>{
                var insertPoint =
                    this.hub.findParentRowOrColumn(this.container.parent) || this.container.layoutManager.root.contentItems[0];
                insertPoint.addChild(createGccDumpView());
            }
        );

        this.container.layoutManager
            .createDragSource(this.gnatDebugTreeButton, createGnatDebugTreeView())
            ._dragListener.on('dragStart', togglePannerAdder);

        this.gnatDebugTreeButton.click(
            () =>{
                var insertPoint =
                    this.hub.findParentRowOrColumn(this.container.parent) || this.container.layoutManager.root.contentItems[0];
                insertPoint.addChild(createGnatDebugTreeView());
            }
        );

        this.container.layoutManager
            .createDragSource(this.gnatDebugButton, createGnatDebugView())
            ._dragListener.on('dragStart', togglePannerAdder);

        this.gnatDebugButton.click(
            () =>{
                var insertPoint =
                    this.hub.findParentRowOrColumn(this.container.parent) || this.container.layoutManager.root.contentItems[0];
                insertPoint.addChild(createGnatDebugView());
            }
        );

        this.container.layoutManager
            .createDragSource(this.cfgButton, createCfgView())
            ._dragListener.on('dragStart', togglePannerAdder);

        this.cfgButton.click(
            () =>{
                var insertPoint =
                    this.hub.findParentRowOrColumn(this.container.parent) || this.container.layoutManager.root.contentItems[0];
                insertPoint.addChild(createCfgView());
            }
        );

        this.container.layoutManager
            .createDragSource(this.executorButton, createExecutor())
            ._dragListener.on('dragStart', togglePannerAdder);

        this.executorButton.click(
            () =>{
                var insertPoint =
                    this.hub.findParentRowOrColumn(this.container.parent) || this.container.layoutManager.root.contentItems[0];
                insertPoint.addChild(createExecutor());
            }
        );

        this.initToolButtons();
    };

    
    undefer() {
        this.deferCompiles = false;
        if (this.needsCompile) {
            this.compile();
        }
    };

    
    override resize() {
        _.defer(() => {
            var topBarHeight = utils.updateAndCalcTopBarHeight(this.domRoot, this.topBar, this.hideable);
            var bottomBarHeight = this.bottomBar.outerHeight(true);
            this.editor.layout({
                width: this.domRoot.width() ?? 0,
                height: this.domRoot.height() ?? 0 - topBarHeight - (bottomBarHeight ?? 0),
            });
        });
    };

// Returns a label name if it can be found in the given position, otherwise
// returns null.
    
    getLabelAtPosition(position) {
        var asmLine = this.assembly[position.lineNumber - 1];
        // Outdated position.lineNumber can happen (Between compilations?) - Check for those and skip
        if (asmLine) {
            var column = position.column;
            var labels = asmLine.labels || [];

            for (var i = 0; i < labels.length; ++i) {
                if (column >= labels[i].range.startCol && column < labels[i].range.endCol) {
                    return labels[i];
                }
            }
        }
        return null;
    };

// Jumps to a label definition related to a label which was found in the
// given position and highlights the given range. If no label can be found in
// the given positon it do nothing.
    
    jumpToLabel(position) {
        var label = this.getLabelAtPosition(position);

        if (!label) {
            return;
        }

        var labelDefLineNum = this.labelDefinitions[label.name];
        if (!labelDefLineNum) {
            return;
        }

        // Highlight the new range.
        var endLineContent = this.editor.getModel()?.getLineContent(labelDefLineNum);

        this.pushRevealJump();

        this.editor.setSelection(
            new monaco.Selection(labelDefLineNum, 0, labelDefLineNum, (endLineContent?.length ?? 0) + 1)
        );

        // Jump to the given line.
        this.editor.revealLineInCenter(labelDefLineNum);
    };

    
    pushRevealJump() {
        this.revealJumpStack.push(this.editor.saveViewState());
        this.revealJumpStackHasElementsCtxKey.set(true);
    };

    
    popAndRevealJump() {
        if (this.revealJumpStack.length > 0) {
            this.editor.restoreViewState(this.revealJumpStack.pop());
            this.revealJumpStackHasElementsCtxKey.set(this.revealJumpStack.length > 0);
        }
    };

    
    initEditorActions() {
        this.isLabelCtxKey = this.editor.createContextKey('isLabel', true);
        this.revealJumpStackHasElementsCtxKey = this.editor.createContextKey('hasRevealJumpStackElements', false);
        this.isAsmKeywordCtxKey = this.editor.createContextKey('isAsmKeyword', true);

        this.editor.addAction({
            id: 'jumptolabel',
            label: 'Jump to label',
            keybindings: [monaco.KeyMod.CtrlCmd | monaco.KeyCode.Enter],
            precondition: 'isLabel',
            contextMenuGroupId: 'navigation',
            contextMenuOrder: 1.5,
            run: (ed) => {
                var position = ed.getPosition();
                if (position != null) {
                    this.jumpToLabel(position);
                }
            },
        });

        // Hiding the 'Jump to label' context menu option if no label can be found
        // in the clicked position.
        const contextmenu = this.editor.getContribution('editor.contrib.contextmenu');
        // @ts-ignore
        const realMethod = contextmenu?._onContextMenu;
        // @ts-ignore
        contextmenu?._onContextMenu && contextmenu?._onContextMenu = (e) => {
            if (e && e.target && e.target.position) {
                if (this.isLabelCtxKey) {
                    var label = this.getLabelAtPosition(e.target.position);
                    this.isLabelCtxKey.set(label !== null);
                }

                if (this.isAsmKeywordCtxKey) {
                    if (!this.compiler?.supportsAsmDocs) {
                        // No need to show the "Show asm documentation" if it's just going to fail.
                        // This is useful for things like xtensa which define an instructionSet but have no docs associated
                        this.isAsmKeywordCtxKey.set(false);
                    } else {
                        var currentWord = this.editor.getModel()?.getWordAtPosition(e.target.position);
                        if (currentWord) {
                            currentWord.range = new monaco.Range(
                                e.target.position.lineNumber,
                                Math.max(currentWord.startColumn, 1),
                                e.target.position.lineNumber,
                                currentWord.endColumn
                            );
                            if (currentWord.word) {
                                this.isAsmKeywordCtxKey.set(this.isWordAsmKeyword(currentWord));
                            }
                        }
                    }
                }
            }
            realMethod.apply(contextmenu, arguments);
        };

        this.editor.addAction({
            id: 'returnfromreveal',
            label: 'Return from reveal jump',
            keybindings: [monaco.KeyMod.CtrlCmd | monaco.KeyMod.Shift | monaco.KeyCode.Enter],
            contextMenuGroupId: 'navigation',
            contextMenuOrder: 1.4,
            precondition: 'hasRevealJumpStackElements',
            run: () =>{
                this.popAndRevealJump();
            }
        });

        this.editor.addAction({
            id: 'viewsource',
            label: 'Scroll to source',
            keybindings: [monaco.KeyMod.CtrlCmd | monaco.KeyCode.F10],
            keybindingContext: undefined,
            contextMenuGroupId: 'navigation',
            contextMenuOrder: 1.5,
            run: (ed) => {
                var position = ed.getPosition();
                if (position != null) {
                    var desiredLine = position.lineNumber - 1;
                    var source = this.assembly[desiredLine].source;
                    if (source && source.line > 0) {
                        var editorId = this.getEditorIdBySourcefile(source);
                        if (editorId) {
                            // a null file means it was the user's source
                            this.eventHub.emit('editorLinkLine', editorId, source.line, -1, -1, true);
                        }
                    }
                }
            }
        });

        this.editor.addAction({
            id: 'viewasmdoc',
            label: 'View assembly documentation',
            keybindings: [monaco.KeyMod.CtrlCmd | monaco.KeyCode.F8],
            keybindingContext: undefined,
            precondition: 'isAsmKeyword',
            contextMenuGroupId: 'help',
            contextMenuOrder: 1.5,
            run: this.onAsmToolTip.bind(this),
        });

        this.editor.addAction({
            id: 'toggleColourisation',
            label: 'Toggle colourisation',
            keybindings: [monaco.KeyMod.CtrlCmd | monaco.KeyMod.Shift | monaco.KeyCode.F1],
            keybindingContext: undefined,
            run: () =>{
                this.eventHub.emit('modifySettings', {
                    colouriseAsm: !this.settings.colouriseAsm,
                });
            }
        });
    };

    
    initEditorCommands() {
        this.editor.addAction({
            id: 'dumpAsm',
            label: 'Developer: Dump asm',
            run: () =>{
                // eslint-disable-next-line no-console
                console.log(this.assembly);
            }
        });
    };

// Gets the filters that will actually be used (accounting for issues with binary
// mode etc).
    
    getEffectiveFilters() {
        if (!this.compiler) return {};
        var filters = this.filters.get();
        if (filters.binary && !this.compiler.supportsBinary) {
            delete filters.binary;
        }
        if (filters.execute && !this.compiler.supportsExecute) {
            delete filters.execute;
        }
        if (filters.libraryCode && !this.compiler.supportsLibraryCodeFilter) {
            delete filters.libraryCode;
        }
        _.each(this.compiler.disabledFilters, function(filter) {
            if (filters[filter]) {
                delete filters[filter];
            }
        });
        return filters;
    };

    
    findTools(content, tools) {
        if (content.componentName === 'tool') {
            if (content.componentState.compiler === this.id) {
                tools.push({
                    id: content.componentState.toolId,
                    args: content.componentState.args,
                    stdin: content.componentState.stdin,
                });
            }
        } else if (content.content) {
            _.each(
                content.content,
                (subcontent) => {
                    tools = this.findTools(subcontent, tools);
                }
            );
        }

        return tools;
    };

    
    getActiveTools(newToolSettings) {
        if (!this.compiler) return {};

        const tools = [];
        if (newToolSettings) {
            tools.push({
                id: newToolSettings.toolId,
                args: newToolSettings.args,
                stdin: newToolSettings.stdin,
            });
        }

        if (this.container.layoutManager.isInitialised) {
            var config = this.container.layoutManager.toConfig();
            return this.findTools(config, tools);
        } else {
            return tools;
        }
    };

    
    isToolActive(activetools, toolId) {
        return _.find(activetools, function(tool) {
            return tool.id === toolId;
        });
    };

    
    compile(bypassCache?, newTools?) {
        if (this.deferCompiles) {
            this.needsCompile = true;
            return;
        }

        type LibrariesType = {
            id: string;
            version: string;
        };

        this.needsCompile = false;
        this.compileInfoLabel.text(' - Compiling...');
        var options = {
            userArguments: this.options,
            compilerOptions: {
                producePp: this.ppViewOpen ? this.ppOptions : false,
                produceAst: this.astViewOpen,
                produceGccDump: {
                    opened: this.gccDumpViewOpen,
                    pass: this.gccDumpPassSelected,
                    treeDump: this.treeDumpEnabled,
                    rtlDump: this.rtlDumpEnabled,
                    ipaDump: this.ipaDumpEnabled,
                    dumpFlags: this.dumpFlags,
                },
                produceOptInfo: this.wantOptInfo,
                produceCfg: this.cfgViewOpen,
                produceGnatDebugTree: this.gnatDebugTreeViewOpen,
                produceGnatDebug: this.gnatDebugViewOpen,
                produceIr: this.irViewOpen,
                produceLLVMOptPipeline: this.llvmOptPipelineViewOpen ? this.llvmOptPipelineOptions : false,
                produceDevice: this.deviceViewOpen,
                produceRustMir: this.rustMirViewOpen,
                produceRustMacroExp: this.rustMacroExpViewOpen,
                produceRustHir: this.rustHirViewOpen,
                produceHaskellCore: this.haskellCoreViewOpen,
                produceHaskellStg: this.haskellStgViewOpen,
                produceHaskellCmm: this.haskellCmmViewOpen,
            },
            filters: this.getEffectiveFilters(),
            tools: this.getActiveTools(newTools),
            libraries: this.libsWidget?.getLibsInUse()?.map((item) => ({
                id: item.libId,
                version: item.versionId,
            })) ?? [],
        };

        if (this.sourceTreeId) {
            this.compileFromTree(options, bypassCache);
        } else {
            this.compileFromEditorSource(options, bypassCache);
        }
    };

    
    compileFromTree(options, bypassCache) {
        const tree = this.hub.getTreeById(this.sourceTreeId);
        if (!tree) {
            this.sourceTreeId = false;
            this.compileFromEditorSource(options, bypassCache);
            return;
        }

        const request = {
            source: tree.multifileService.getMainSource(),
            compiler: this.compiler ? this.compiler.id : '',
            options: options,
            lang: this.currentLangId,
            files: tree.multifileService.getFiles(),
            bypassCache: false,
        };

        var fetches: Promise<void>[] = [];
        fetches.push(
            this.compilerService.expand(request.source).then((contents) => {
                request.source = contents;
            })
        );

        for (var i = 0; i < request.files.length; i++) {
            var file = request.files[i];
            fetches.push(
                this.compilerService.expand(file.contents).then((contents) => {
                    file.contents = contents;
                })
            );
        }

        var self = this;
        Promise.all(fetches).then(() => {
            var treeState = tree.currentState();
            var cmakeProject = tree.multifileService.isACMakeProject();

            if (bypassCache) request.bypassCache = true;
            if (!self.compiler) {
                self.onCompileResponse(request, this.errorResult('<Please select a compiler>'), false);
            } else if (cmakeProject && request.source === '') {
                self.onCompileResponse(request, this.errorResult('<Please supply a CMakeLists.txt>'), false);
            } else {
                if (cmakeProject) {
                    request.options.compilerOptions.cmakeArgs = treeState.cmakeArgs;
                    request.options.compilerOptions.customOutputFilename = treeState.customOutputFilename;
                    self.sendCMakeCompile(request);
                } else {
                    self.sendCompile(request);
                }
            }
        });
    };

    
    compileFromEditorSource(options, bypassCache: boolean) {
        this.compilerService.expand(this.source).then(
            (expanded) => {
                const request = {
                    source: expanded || '',
                    compiler: this.compiler ? this.compiler.id : '',
                    options: options,
                    lang: this.currentLangId,
                    files: [],
                    bypassCache: false,
                };
                if (bypassCache) request.bypassCache = true;
                if (!this.compiler) {
                    this.onCompileResponse(request, this.errorResult('<Please select a compiler>'), false);
                } else {
                    this.sendCompile(request);
                }
            }
        );
    };

    
    sendCMakeCompile(request) {
        var onCompilerResponse = _.bind(this.onCMakeResponse, this);

        if (this.pendingCMakeRequestSentAt) {
            // If we have a request pending, then just store this request to do once the
            // previous request completes.
            this.nextCMakeRequest = request;
            return;
        }
        this.eventHub.emit('compiling', this.id, this.compiler);
        // Display the spinner
        this.handleCompilationStatus({code: 4});
        this.pendingCMakeRequestSentAt = Date.now();
        // After a short delay, give the user some indication that we're working on their
        // compilation.
        var progress = setTimeout(
            () =>{
                this.setAssembly({asm: this.fakeAsm('<Compiling...>')}, 0);
            },
            500
        );
        this.compilerService
            .submitCMake(request)
            .then(function(x) {
                clearTimeout(progress);
                onCompilerResponse(request, x.result, x.localCacheHit);
            })
            .catch(function(x) {
                clearTimeout(progress);
                var message = 'Unknown error';
                if (_.isString(x)) {
                    message = x;
                } else if (x) {
                    message = x.error || x.code || message;
                }
                onCompilerResponse(request, this.errorResult('<Compilation failed: ' + message + '>'), false);
            });
    };

    
    sendCompile(request) {
        var onCompilerResponse = this.onCompileResponse.bind(this);

        if (this.pendingRequestSentAt) {
            // If we have a request pending, then just store this request to do once the
            // previous request completes.
            this.nextRequest = request;
            return;
        }
        this.eventHub.emit('compiling', this.id, this.compiler);
        // Display the spinner
        this.handleCompilationStatus({code: 4});
        this.pendingRequestSentAt = Date.now();
        // After a short delay, give the user some indication that we're working on their
        // compilation.
        var progress = setTimeout(
            () =>{
                this.setAssembly({asm: this.fakeAsm('<Compiling...>')}, 0);
            },
            500
        );
        this.compilerService
            .submit(request)
            .then((x) => {
                clearTimeout(progress);
                onCompilerResponse(request, x.result, x.localCacheHit);
            })
            .catch((e) => {
                clearTimeout(progress);
                var message = 'Unknown error';
                if (_.isString(e)) {
                    message = e;
                } else if (e) {
                    message = e.error || e.code || e.message;
                    if (e.stack) {
                        // eslint-disable-next-line no-console
                        console.log(e);
                    }
                }
                onCompilerResponse(request, this.errorResult('<Compilation failed: ' + message + '>'), false);
            });
    };

    
    setNormalMargin() {
        this.editor.updateOptions({
            lineNumbers: true,
            lineNumbersMinChars: 1,
        });
    };

    
    setBinaryMargin() {
        this.editor.updateOptions({
            lineNumbersMinChars: 6,
            lineNumbers: _.bind(this.getBinaryForLine, this),
        });
    };


    getBinaryForLine(line) {
        var obj = this.assembly[line - 1];
        if (obj) {
            return obj.address ? obj.address.toString(16) : '';
        } else {
            return '???';
        }
    };


    setAssembly(result, filteredCount) {
        var asm = result.asm || this.fakeAsm('<No output>');
        this.assembly = asm;
        if (!this.editor || !this.editor.getModel()) return;
        var editorModel = this.editor.getModel();
        var msg = '<No assembly generated>';
        if (asm.length) {
            msg = _.pluck(asm, 'text').join('\n');
        } else if (filteredCount > 0) {
            msg = '<No assembly to display (~' + filteredCount + (filteredCount === 1 ? ' line' : ' lines') + ' filtered)>';
        }

        if (asm.length === 1 && result.code !== 0 && (result.stderr || result.stdout)) {
            msg += '\n\n# For more information see the output window';
            if (!this.isOutputOpened) {
                msg += '\n# To open the output window, click or drag the "Output" icon at the bottom of this window';
            }
        }

        editorModel.setValue(msg);

        if (!this.awaitingInitialResults) {
            if (this.selection) {
                this.editor.setSelection(this.selection);
                this.editor.revealLinesInCenter(this.selection.startLineNumber, this.selection.endLineNumber);
            }
            this.awaitingInitialResults = true;
        } else {
            var visibleRanges = this.editor.getVisibleRanges();
            var currentTopLine = visibleRanges.length > 0 ? visibleRanges[0].startLineNumber : 1;
            this.editor.revealLine(currentTopLine);
        }

        this.decorations.labelUsages = [];
        _.each(
            this.assembly,
            (obj, line) => {
                if (!obj.labels || !obj.labels.length) return;

                obj.labels.forEach((label) => {
                    this.decorations.labelUsages.push({
                        range: new monaco.Range(line + 1, label.range.startCol, line + 1, label.range.endCol),
                        options: {
                            inlineClassName: 'asm-label-link',
                            hoverMessage: [
                                {
                                    value: 'Ctrl + Left click to follow the label',
                                },
                            ],
                        },
                    });
                });
            }
        );
        this.updateDecorations();

        var codeLenses = [];
        if (this.getEffectiveFilters().binary || result.forceBinaryView) {
            this.setBinaryMargin();
            _.each(
                this.assembly,
                _.bind(function(obj, line) {
                    if (obj.opcodes) {
                        var address = obj.address ? obj.address.toString(16) : '';
                        codeLenses.push({
                            range: {
                                startLineNumber: line + 1,
                                startColumn: 1,
                                endLineNumber: line + 2,
                                endColumn: 1,
                            },
                            id: address,
                            command: {
                                title: obj.opcodes.join(' '),
                            },
                        });
                    }
                }, this)
            );
        } else {
            this.setNormalMargin();
        }

        if (this.settings.enableCodeLens) {
            codeLensHandler.registerLensesForCompiler(this.id, editorModel, codeLenses);

            var currentAsmLang = editorModel.getLanguageId();
            codeLensHandler.registerProviderForLanguage(currentAsmLang);
        } else {
            // Make sure the codelens is disabled
            codeLensHandler.unregister(this.id);
        }
    };

    private errorResult(text) {
        return {asm: this.fakeAsm(text), code: -1, stdout: '', stderr: ''};
    }

    private fakeAsm(text) {
        return [{text: text, source: null, fake: true}];
    }


    doNextCompileRequest() {
        if (this.nextRequest) {
            var next = this.nextRequest;
            this.nextRequest = null;
            this.sendCompile(next);
        }
    };


    doNextCMakeRequest() {
        if (this.nextCMakeRequest) {
            var next = this.nextCMakeRequest;
            this.nextCMakeRequest = null;
            this.sendCMakeCompile(next);
        }
    };


    onCMakeResponse(request, result, cached) {
        result.source = this.source;
        this.lastResult = result;
        var timeTaken = Math.max(0, Date.now() - this.pendingCMakeRequestSentAt);
        this.lastTimeTaken = timeTaken;
        var wasRealReply = this.pendingCMakeRequestSentAt > 0;
        this.pendingCMakeRequestSentAt = 0;

        this.handleCompileRequestAndResult(request, result, cached, wasRealReply, timeTaken);

        this.doNextCMakeRequest();
    };


    handleCompileRequestAndResult(request, result, cached, wasRealReply, timeTaken) {
        ga.proxy('send', {
            hitType: 'event',
            eventCategory: 'Compile',
            eventAction: request.compiler,
            eventLabel: request.options.userArguments,
            eventValue: cached ? 1 : 0,
        });
        ga.proxy('send', {
            hitType: 'timing',
            timingCategory: 'Compile',
            timingVar: request.compiler,
            timingValue: timeTaken,
        });

        // Delete trailing empty lines
        if (Array.isArray(result.asm)) {
            var indexToDiscard = _.findLastIndex(result.asm, function(line) {
                return !_.isEmpty(line.text);
            });
            result.asm.splice(indexToDiscard + 1, result.asm.length - indexToDiscard);
        }

        this.labelDefinitions = result.labelDefinitions || {};
        if (result.asm) {
            this.setAssembly(result, result.filteredCount || 0);
        } else if (result.result && result.result.asm) {
            this.setAssembly(result.result, result.result.filteredCount || 0);
        } else {
            result.asm = fakeAsm('<Compilation failed>');
            this.setAssembly(result, 0);
        }

        var stdout = result.stdout || [];
        var stderr = result.stderr || [];
        var failed = result.code ? result.code !== 0 : false;

        if (result.buildsteps) {
            _.each(result.buildsteps, function(step) {
                stdout = stdout.concat(step.stdout || []);
                stderr = stderr.concat(step.stderr || []);
                failed = failed | (step.code !== 0);
            });
        }

        this.handleCompilationStatus(CompilerService.calculateStatusIcon(result));
        this.outputTextCount.text(stdout.length);
        this.outputErrorCount.text(stderr.length);
        if (this.isOutputOpened || (stdout.length === 0 && stderr.length === 0)) {
            this.outputBtn.prop('title', '');
        } else {
            CompilerService.handleOutputButtonTitle(this.outputBtn, result);
        }
        var infoLabelText = '';
        if (cached) {
            infoLabelText = ' - cached';
        } else if (wasRealReply) {
            infoLabelText = ' - ' + timeTaken + 'ms';
        }

        if (result.asmSize) {
            infoLabelText += ' (' + result.asmSize + 'B)';
        }

        if (result.filteredCount && result.filteredCount > 0) {
            infoLabelText += ' ~' + result.filteredCount + (result.filteredCount === 1 ? ' line' : ' lines') + ' filtered';
        }

        this.compileInfoLabel.text(infoLabelText);

        if (result.result) {
            var wasCmake =
                result.buildsteps &&
                _.any(result.buildsteps, function(step) {
                    return step.step === 'cmake';
                });
            this.postCompilationResult(request, result.result, wasCmake);
        } else {
            this.postCompilationResult(request, result);
        }

        this.eventHub.emit('compileResult', this.id, this.compiler, result, languages[this.currentLangId]);
    };


    onCompileResponse(request, result, cached) {
        // Save which source produced this change. It should probably be saved earlier though
        result.source = this.source;
        this.lastResult = result;
        var timeTaken = Math.max(0, Date.now() - this.pendingRequestSentAt);
        this.lastTimeTaken = timeTaken;
        var wasRealReply = this.pendingRequestSentAt > 0;
        this.pendingRequestSentAt = 0;

        this.handleCompileRequestAndResult(request, result, cached, wasRealReply, timeTaken);

        this.doNextCompileRequest();
    };


    postCompilationResult(request, result, wasCmake) {
        if (result.popularArguments) {
            this.handlePopularArgumentsResult(result.popularArguments);
        } else if (this.compiler) {
            this.compilerService.requestPopularArguments(this.compiler.id, request.options.userArguments).then(
                (result) => {
                    if (result && result.result) {
                        this.handlePopularArgumentsResult(result.result);
                    }
                }
            );
        }

        this.updateButtons();

        this.checkForUnwiseArguments(result.compilationOptions, wasCmake);
        this.setCompilationOptionsPopover(result.compilationOptions ? result.compilationOptions.join(' ') : '');

        this.checkForHints(result);

        if (result.bbcdiskimage) {
            this.emulateBbcDisk(result.bbcdiskimage);
        } else if (result.speccytape) {
            this.emulateSpeccyTape(result.speccytape);
        } else if (result.miraclesms) {
            this.emulateMiracleSMS(result.miraclesms);
        }
    };


    emulateMiracleSMS(image) {
        var dialog = $('#miracleemu');

        this.alertSystem.notify(
            'Click ' +
            '<a target="_blank" id="miracle_emulink" style="cursor:pointer;" click="javascript:;">here</a>' +
            ' to emulate',
            {
                group: 'emulation',
                collapseSimilar: true,
                dismissTime: 10000,
                onBeforeShow: function(elem) {
                    elem.find('#miracle_emulink').on('click', function() {
                        dialog.modal();

                        var emuwindow = dialog.find('#miracleemuframe')[0].contentWindow;
                        var tmstr = Date.now();
                        emuwindow.location = 'https://xania.org/miracle/miracle.html?' + tmstr + '#b64sms=' + image;
                    });
                },
            }
        );
    };


    emulateSpeccyTape(image) {
        var dialog = $('#jsspeccyemu');

        this.alertSystem.notify(
            'Click ' +
            '<a target="_blank" id="jsspeccy_emulink" style="cursor:pointer;" click="javascript:;">here</a>' +
            ' to emulate',
            {
                group: 'emulation',
                collapseSimilar: true,
                dismissTime: 10000,
                onBeforeShow: function(elem) {
                    elem.find('#jsspeccy_emulink').on('click', function() {
                        dialog.modal();

                        var emuwindow = dialog.find('#speccyemuframe')[0].contentWindow;
                        var tmstr = Date.now();
                        emuwindow.location = 'https://static.ce-cdn.net/jsspeccy/index.html?' + tmstr + '#b64tape=' + image;
                    });
                },
            }
        );
    };


    emulateBbcDisk(bbcdiskimage) {
        var dialog = $('#jsbeebemu');

        this.alertSystem.notify(
            'Click <a target="_blank" id="emulink" style="cursor:pointer;" click="javascript:;">here</a> to emulate',
            {
                group: 'emulation',
                collapseSimilar: true,
                dismissTime: 10000,
                onBeforeShow: function(elem) {
                    elem.find('#emulink').on('click', function() {
                        dialog.modal();

                        var emuwindow = dialog.find('#jsbeebemuframe')[0].contentWindow;
                        var tmstr = Date.now();
                        emuwindow.location =
                            'https://bbc.godbolt.org/?' + tmstr + '#embed&autoboot&disc1=b64data:' + bbcdiskimage;
                    });
                },
            }
        );
    };


    onEditorChange(editor, source, langId, compilerId) {
        if (this.sourceTreeId) {
            var tree = this.hub.getTreeById(this.sourceTreeId);
            if (tree) {
                if (tree.multifileService.isEditorPartOfProject(editor)) {
                    if (this.settings.compileOnChange) {
                        this.compile();

                        return;
                    }
                }
            }
        }

        if (
            editor === this.sourceEditorId &&
            langId === this.currentLangId &&
            (compilerId === undefined || compilerId === this.id)
        ) {
            this.source = source;
            if (this.settings.compileOnChange) {
                this.compile();
            }
        }
    };


    onCompilerFlagsChange(compilerId, compilerFlags) {
        if (compilerId === this.id) {
            this.onOptionsChange(compilerFlags);
        }
    };


    onToolOpened(compilerId, toolSettings) {
        if (this.id === compilerId) {
            var toolId = toolSettings.toolId;

            var buttons = this.toolsMenu.find('button');
            $(buttons).each(
                _.bind(function(idx, button) {
                    var toolButton = $(button);
                    var toolName = toolButton.data('toolname');
                    if (toolId === toolName) {
                        toolButton.prop('disabled', true);
                    }
                }, this)
            );

            this.compile(false, toolSettings);
        }
    };


    onToolClosed(compilerId, toolSettings) {
        if (this.id === compilerId) {
            var toolId = toolSettings.toolId;

            var buttons = this.toolsMenu.find('button');
            $(buttons).each(
                _.bind(function(idx, button) {
                    var toolButton = $(button);
                    var toolName = toolButton.data('toolname');
                    if (toolId === toolName) {
                        toolButton.prop('disabled', !this.supportsTool(toolId));
                    }
                }, this)
            );
        }
    };


    onOutputOpened(compilerId) {
        if (this.id === compilerId) {
            this.isOutputOpened = true;
            this.outputBtn.prop('disabled', true);
            this.resendResult();
        }
    };


    onOutputClosed(compilerId) {
        if (this.id === compilerId) {
            this.isOutputOpened = false;
            this.outputBtn.prop('disabled', false);
        }
    };


    onOptViewClosed(id) {
        if (this.id === id) {
            this.wantOptInfo = false;
            this.optViewOpen = false;
            this.optButton.prop('disabled', this.optViewOpen);
        }
    };


    onFlagsViewClosed(id, compilerFlags) {
        if (this.id === id) {
            this.flagsViewOpen = false;
            this.optionsField.val(compilerFlags);
            this.optionsField.prop('disabled', this.flagsViewOpen);
            this.optionsField.prop('placeholder', this.initialOptionsFieldPlacehoder);
            this.flagsButton.prop('disabled', this.flagsViewOpen);

            this.compilerService.requestPopularArguments(this.compiler.id, compilerFlags).then(
                _.bind(function(result) {
                    if (result && result.result) {
                        this.handlePopularArgumentsResult(result.result);
                    }
                }, this)
            );

            this.saveState();
        }
    };


    onToolSettingsChange(id) {
        if (this.id === id) {
            this.compile();
        }
    };


    onPpViewOpened(id) {
        if (this.id === id) {
            this.ppButton.prop('disabled', true);
            this.ppViewOpen = true;
            // the pp view will request compilation once it populates its options so this.compile() is not called here
        }
    };


    onPpViewClosed(id) {
        if (this.id === id) {
            this.ppButton.prop('disabled', false);
            this.ppViewOpen = false;
        }
    };


    onPpViewOptionsUpdated(id, options, reqCompile) {
        if (this.id === id) {
            this.ppOptions = options;
            if (reqCompile) {
                this.compile();
            }
        }
    };


    onAstViewOpened(id) {
        if (this.id === id) {
            this.astButton.prop('disabled', true);
            this.astViewOpen = true;
            this.compile();
        }
    };


    onAstViewClosed(id) {
        if (this.id === id) {
            this.astButton.prop('disabled', false);
            this.astViewOpen = false;
        }
    };


    onIrViewOpened(id) {
        if (this.id === id) {
            this.irButton.prop('disabled', true);
            this.irViewOpen = true;
            this.compile();
        }
    };


    onIrViewClosed(id) {
        if (this.id === id) {
            this.irButton.prop('disabled', false);
            this.irViewOpen = false;
        }
    };


    onLLVMOptPipelineViewOpened(id) {
        if (this.id === id) {
            this.llvmOptPipelineButton.prop('disabled', true);
            this.llvmOptPipelineViewOpen = true;
            this.compile();
        }
    };


    onLLVMOptPipelineViewClosed(id) {
        if (this.id === id) {
            this.llvmOptPipelineButton.prop('disabled', false);
            this.llvmOptPipelineViewOpen = false;
        }
    };


    onLLVMOptPipelineViewOptionsUpdated(id, options, recompile) {
        if (this.id === id) {
            this.llvmOptPipelineOptions = options;
            if (recompile) {
                this.compile();
            }
        }
    };


    onDeviceViewOpened(id) {
        if (this.id === id) {
            this.deviceButton.prop('disabled', true);
            this.deviceViewOpen = true;
            this.compile();
        }
    };


    onDeviceViewClosed(id) {
        if (this.id === id) {
            this.deviceButton.prop('disabled', false);
            this.deviceViewOpen = false;
        }
    };


    onRustMirViewOpened(id) {
        if (this.id === id) {
            this.rustMirButton.prop('disabled', true);
            this.rustMirViewOpen = true;
            this.compile();
        }
    };


    onRustMirViewClosed(id) {
        if (this.id === id) {
            this.rustMirButton.prop('disabled', false);
            this.rustMirViewOpen = false;
        }
    };


    onHaskellCoreViewOpened(id) {
        if (this.id === id) {
            this.haskellCoreButton.prop('disabled', true);
            this.haskellCoreViewOpen = true;
            this.compile();
        }
    };


    onHaskellCoreViewClosed(id) {
        if (this.id === id) {
            this.haskellCoreButton.prop('disabled', false);
            this.haskellCoreViewOpen = false;
        }
    };


    onHaskellStgViewOpened(id) {
        if (this.id === id) {
            this.haskellStgButton.prop('disabled', true);
            this.haskellStgViewOpen = true;
            this.compile();
        }
    };


    onHaskellStgViewClosed(id) {
        if (this.id === id) {
            this.haskellStgButton.prop('disabled', false);
            this.haskellStgViewOpen = false;
        }
    };


    onHaskellCmmViewOpened(id) {
        if (this.id === id) {
            this.haskellCmmButton.prop('disabled', true);
            this.haskellCmmViewOpen = true;
            this.compile();
        }
    };


    onHaskellCmmViewClosed(id) {
        if (this.id === id) {
            this.haskellCmmButton.prop('disabled', false);
            this.haskellCmmViewOpen = false;
        }
    };


    onGnatDebugTreeViewOpened(id) {
        if (this.id === id) {
            this.gnatDebugTreeButton.prop('disabled', true);
            this.gnatDebugTreeViewOpen = true;
            this.compile();
        }
    };


    onGnatDebugTreeViewClosed(id) {
        if (this.id === id) {
            this.gnatDebugTreeButton.prop('disabled', false);
            this.gnatDebugTreeViewOpen = false;
        }
    };


    onGnatDebugViewOpened(id) {
        if (this.id === id) {
            this.gnatDebugButton.prop('disabled', true);
            this.gnatDebugViewOpen = true;
            this.compile();
        }
    };


    onGnatDebugViewClosed(id) {
        if (this.id === id) {
            this.gnatDebugButton.prop('disabled', false);
            this.gnatDebugViewOpen = false;
        }
    };


    onRustMacroExpViewOpened(id) {
        if (this.id === id) {
            this.rustMacroExpButton.prop('disabled', true);
            this.rustMacroExpViewOpen = true;
            this.compile();
        }
    };


    onRustMacroExpViewClosed(id) {
        if (this.id === id) {
            this.rustMacroExpButton.prop('disabled', false);
            this.rustMacroExpViewOpen = false;
        }
    };


    onRustHirViewOpened(id) {
        if (this.id === id) {
            this.rustHirButton.prop('disabled', true);
            this.rustHirViewOpen = true;
            this.compile();
        }
    };


    onRustHirViewClosed(id) {
        if (this.id === id) {
            this.rustHirButton.prop('disabled', false);
            this.rustHirViewOpen = false;
        }
    };


    onGccDumpUIInit(id) {
        if (this.id === id) {
            this.compile();
        }
    };


    onGccDumpFiltersChanged(id, filters, reqCompile) {
        if (this.id === id) {
            this.treeDumpEnabled = filters.treeDump !== false;
            this.rtlDumpEnabled = filters.rtlDump !== false;
            this.ipaDumpEnabled = filters.ipaDump !== false;
            this.dumpFlags = {
                address: filters.addressOption !== false,
                slim: filters.slimOption !== false,
                raw: filters.rawOption !== false,
                details: filters.detailsOption !== false,
                stats: filters.statsOption !== false,
                blocks: filters.blocksOption !== false,
                vops: filters.vopsOption !== false,
                lineno: filters.linenoOption !== false,
                uid: filters.uidOption !== false,
                all: filters.allOption !== false,
            };

            if (reqCompile) {
                this.compile();
            }
        }
    };


    onGccDumpPassSelected(id, passObject, reqCompile) {
        if (this.id === id) {
            this.gccDumpPassSelected = passObject;

            if (reqCompile && passObject !== null) {
                this.compile();
            }
        }
    };


    onGccDumpViewOpened(id) {
        if (this.id === id) {
            this.gccDumpButton.prop('disabled', true);
            this.gccDumpViewOpen = true;
        }
    };


    onGccDumpViewClosed(id) {
        if (this.id === id) {
            this.gccDumpButton.prop('disabled', !this.compiler.supportsGccDump);
            this.gccDumpViewOpen = false;

            delete this.gccDumpPassSelected;
            delete this.treeDumpEnabled;
            delete this.rtlDumpEnabled;
            delete this.ipaDumpEnabled;
            delete this.dumpFlags;
        }
    };


    onOptViewOpened(id) {
        if (this.id === id) {
            this.optViewOpen = true;
            this.wantOptInfo = true;
            this.optButton.prop('disabled', this.optViewOpen);
            this.compile();
        }
    };


    onFlagsViewOpened(id) {
        if (this.id === id) {
            this.flagsViewOpen = true;
            this.handlePopularArgumentsResult(false);
            this.optionsField.prop('disabled', this.flagsViewOpen);
            this.optionsField.val('');
            this.optionsField.prop('placeholder', 'see detailed flags window');
            this.flagsButton.prop('disabled', this.flagsViewOpen);
            this.compile();
            this.saveState();
        }
    };


    onCfgViewOpened(id) {
        if (this.id === id) {
            this.cfgButton.prop('disabled', true);
            this.cfgViewOpen = true;
            this.compile();
        }
    };


    onCfgViewClosed(id) {
        if (this.id === id) {
            this.cfgViewOpen = false;
            this.cfgButton.prop('disabled', this.getEffectiveFilters().binary);
        }
    };


    initFilterButtons() {
        this.filterBinaryButton = this.domRoot.find("[data-bind='binary']");
        this.filterBinaryTitle = this.filterBinaryButton.prop('title');

        this.filterExecuteButton = this.domRoot.find("[data-bind='execute']");
        this.filterExecuteTitle = this.filterExecuteButton.prop('title');

        this.filterLabelsButton = this.domRoot.find("[data-bind='labels']");
        this.filterLabelsTitle = this.filterLabelsButton.prop('title');

        this.filterDirectivesButton = this.domRoot.find("[data-bind='directives']");
        this.filterDirectivesTitle = this.filterDirectivesButton.prop('title');

        this.filterLibraryCodeButton = this.domRoot.find("[data-bind='libraryCode']");
        this.filterLibraryCodeTitle = this.filterLibraryCodeButton.prop('title');

        this.filterCommentsButton = this.domRoot.find("[data-bind='commentOnly']");
        this.filterCommentsTitle = this.filterCommentsButton.prop('title');

        this.filterTrimButton = this.domRoot.find("[data-bind='trim']");
        this.filterTrimTitle = this.filterTrimButton.prop('title');

        this.filterIntelButton = this.domRoot.find("[data-bind='intel']");
        this.filterIntelTitle = this.filterIntelButton.prop('title');

        this.filterDemangleButton = this.domRoot.find("[data-bind='demangle']");
        this.filterDemangleTitle = this.filterDemangleButton.prop('title');

        this.noBinaryFiltersButtons = this.domRoot.find('.nonbinary');
    };


    initButtons(state) {
        this.filters = new Toggles(this.domRoot.find('.filters'), patchOldFilters(state.filters));

        this.optButton = this.domRoot.find('.btn.view-optimization');
        this.flagsButton = this.domRoot.find('div.populararguments div.dropdown-menu button');
        this.ppButton = this.domRoot.find('.btn.view-pp');
        this.astButton = this.domRoot.find('.btn.view-ast');
        this.irButton = this.domRoot.find('.btn.view-ir');
        this.llvmOptPipelineButton = this.domRoot.find('.btn.view-llvm-opt-pipeline');
        this.deviceButton = this.domRoot.find('.btn.view-device');
        this.gnatDebugTreeButton = this.domRoot.find('.btn.view-gnatdebugtree');
        this.gnatDebugButton = this.domRoot.find('.btn.view-gnatdebug');
        this.rustMirButton = this.domRoot.find('.btn.view-rustmir');
        this.rustMacroExpButton = this.domRoot.find('.btn.view-rustmacroexp');
        this.rustMirButton = this.domRoot.find('.btn.view-rusthir');
        this.haskellCoreButton = this.domRoot.find('.btn.view-haskellCore');
        this.haskellStgButton = this.domRoot.find('.btn.view-haskellStg');
        this.haskellCmmButton = this.domRoot.find('.btn.view-haskellCmm');
        this.gccDumpButton = this.domRoot.find('.btn.view-gccdump');
        this.cfgButton = this.domRoot.find('.btn.view-cfg');
        this.executorButton = this.domRoot.find('.create-executor');
        this.libsButton = this.domRoot.find('.btn.show-libs');

        this.compileInfoLabel = this.domRoot.find('.compile-info');
        this.compileClearCache = this.domRoot.find('.clear-cache');

        this.outputBtn = this.domRoot.find('.output-btn');
        this.outputTextCount = this.domRoot.find('span.text-count');
        this.outputErrorCount = this.domRoot.find('span.err-count');

        this.optionsField = this.domRoot.find('.options');
        this.initialOptionsFieldPlacehoder = this.optionsField.prop('placeholder');
        this.prependOptions = this.domRoot.find('.prepend-options');
        this.fullCompilerName = this.domRoot.find('.full-compiler-name');
        this.fullTimingInfo = this.domRoot.find('.full-timing-info');
        this.compilerLicenseButton = this.domRoot.find('.compiler-license');
        this.setCompilationOptionsPopover(this.compiler ? this.compiler.options : null);
        // Dismiss on any click that isn't either in the opening element, inside
        // the popover or on any alert
        $(document).on(
            'mouseup',
            (e) => {
                const target = $(e.target);
                if (
                    !target.is(this.prependOptions) &&
                    this.prependOptions.has(target).length === 0 &&
                    target.closest('.popover').length === 0
                )
                    this.prependOptions.popover('hide');

                if (
                    !target.is(this.fullCompilerName) &&
                    this.fullCompilerName.has(target).length === 0 &&
                    target.closest('.popover').length === 0
                )
                    this.fullCompilerName.popover('hide');
            }
        );

        this.initFilterButtons(state);

        this.filterExecuteButton.toggle(options.supportsExecute);
        this.filterLibraryCodeButton.toggle(options.supportsLibraryCodeFilter);

        this.optionsField.val(this.options);

        this.shortCompilerName = this.domRoot.find('.short-compiler-name');
        this.compilerPicker = this.domRoot.find('.compiler-picker');
        this.setCompilerVersionPopover({version: '', fullVersion: ''}, '');

        this.topBar = this.domRoot.find('.top-bar');
        this.bottomBar = this.domRoot.find('.bottom-bar');
        this.statusLabel = this.domRoot.find('.status-text');

        this.hideable = this.domRoot.find('.hideable');
        this.statusIcon = this.domRoot.find('.status-icon');

        this.monacoPlaceholder = this.domRoot.find('.monaco-placeholder');

        this.initPanerButtons();
    };


    onLibsChanged() {
        this.saveState();
        this.compile();
    };


    initLibraries(state) {
        this.libsWidget = new LibsWidget(
            this.currentLangId,
            this.compiler,
            this.libsButton,
            state,
            this.onLibsChanged.bind(this),
            LibUtils.getSupportedLibraries(
                this.compiler ? this.compiler.libsArr : [],
                this.currentLangId,
                this.compiler ? this.compiler.remote : null
            )
        );
    };


    updateLibraries() {
        if (this.libsWidget) {
            var filteredLibraries = {};
            if (this.compiler) {
                filteredLibraries = LibUtils.getSupportedLibraries(
                    this.compiler.libsArr,
                    this.currentLangId,
                    this.compiler ? this.compiler.remote : null
                );
            }

            this.libsWidget.setNewLangId(this.currentLangId, this.compiler?.id ?? '', filteredLibraries);
        }
    };


    isSupportedTool(tool) {
        if (this.sourceTreeId) {
            return tool.tool.type === 'postcompilation';
        } else {
            return true;
        }
    };


    supportsTool(toolId) {
        if (!this.compiler) return;

        return _.find(
            this.compiler.tools,
            (tool) => {
                return tool.id === toolId && this.isSupportedTool(tool);
            }
        );
    };


    initToolButton(togglePannerAdder, button, toolId) {
        var createToolView = () =>{
            var args = '';
            var monacoStdin = false;
            var langTools = options.tools[this.currentLangId];
            if (langTools && langTools[toolId] && langTools[toolId].tool) {
                if (langTools[toolId].tool.args !== undefined) {
                    args = langTools[toolId].tool.args;
                }
                if (langTools[toolId].tool.monacoStdin !== undefined) {
                    monacoStdin = langTools[toolId].tool.monacoStdin;
                }
            }
            return Components.getToolViewWith(this.id as unknown as string, this.sourceEditorId ?? 0, toolId, args, monacoStdin, this.sourceTreeId);
        };

        this.container.layoutManager
            .createDragSource(button, createToolView())
            ._dragListener.on('dragStart', togglePannerAdder);

        button.click(
            () =>{
                button.prop('disabled', true);
                var insertPoint =
                    this.hub.findParentRowOrColumn(this.container) || this.container.layoutManager.root.contentItems[0];
                insertPoint.addChild(createToolView);
            }
        );
    };


    initToolButtons() {
        this.toolsMenu = this.domRoot.find('.new-tool-dropdown');
        var toggleToolDropdown = () =>{
            this.toolsMenu?.dropdown('toggle');
        };
        this.toolsMenu.empty();

        if (!this.compiler) return;

        var addTool = _.bind(function(toolName, title, toolIcon, toolIconDark) {
            var btn = $("<button class='dropdown-item btn btn-light btn-sm'>");
            btn.addClass('view-' + toolName);
            btn.data('toolname', toolName);
            if (toolIcon) {
                var light = toolIcons(toolIcon);
                var dark = toolIconDark ? toolIcons(toolIconDark) : light;
                btn.append(
                    '<span class="dropdown-icon fas">' +
                    '<img src="' +
                    light +
                    '" class="theme-light-only" width="16px" style="max-height: 16px"/>' +
                    '<img src="' +
                    dark +
                    '" class="theme-dark-only" width="16px" style="max-height: 16px"/>' +
                    '</span>'
                );
            } else {
                btn.append("<span class='dropdown-icon fas fa-cog'></span>");
            }
            btn.append(title);
            this.toolsMenu.append(btn);

            if (toolName !== 'none') {
                this.initToolButton(toggleToolDropdown, btn, toolName);
            }
        }, this);

        if (_.isEmpty(this.compiler.tools)) {
            addTool('none', 'No tools available');
        } else {
            _.each(
                this.compiler.tools,
                _.bind(function(tool) {
                    if (this.isSupportedTool(tool)) {
                        addTool(tool.tool.id, tool.tool.name, tool.tool.icon, tool.tool.darkIcon);
                    }
                }, this)
            );
        }
    };


    enableToolButtons() {
        var activeTools = this.getActiveTools();

        var buttons = this.toolsMenu.find('button');
        $(buttons).each(
            _.bind(function(idx, button) {
                var toolButton = $(button);
                var toolName = toolButton.data('toolname');
                toolButton.prop('disabled', !(this.supportsTool(toolName) && !this.isToolActive(activeTools, toolName)));
            }, this)
        );
    };

// eslint-disable-next-line max-statements

    updateButtons() {
        if (!this.compiler) return;
        var filters = this.getEffectiveFilters();
        // We can support intel output if the compiler supports it, or if we're compiling
        // to binary (as we can disassemble it however we like).
        var formatFilterTitle(button, title) {
            button.prop(
                'title',
                '[' +
                (button.hasClass('active') ? 'ON' : 'OFF') +
                '] ' +
                title +
                (button.prop('disabled') ? ' [LOCKED]' : '')
            );
        };
        var isIntelFilterDisabled = !this.compiler.supportsIntel && !filters.binary;
        this.filterIntelButton.prop('disabled', isIntelFilterDisabled);
        formatFilterTitle(this.filterIntelButton, this.filterIntelTitle);
        // Disable binary support on compilers that don't work with it.
        this.filterBinaryButton.prop('disabled', !this.compiler.supportsBinary);
        formatFilterTitle(this.filterBinaryButton, this.filterBinaryTitle);
        this.filterExecuteButton.prop('disabled', !this.compiler.supportsExecute);
        formatFilterTitle(this.filterExecuteButton, this.filterExecuteTitle);
        // Disable demangle for compilers where we can't access it
        this.filterDemangleButton.prop('disabled', !this.compiler.supportsDemangle);
        formatFilterTitle(this.filterDemangleButton, this.filterDemangleTitle);
        // Disable any of the options which don't make sense in binary mode.
        var noBinaryFiltersDisabled = !!filters.binary && !this.compiler.supportsFiltersInBinary;
        this.noBinaryFiltersButtons.prop('disabled', noBinaryFiltersDisabled);

        this.filterLibraryCodeButton.prop('disabled', !this.compiler.supportsLibraryCodeFilter);
        formatFilterTitle(this.filterLibraryCodeButton, this.filterLibraryCodeTitle);

        this.filterLabelsButton.prop('disabled', this.compiler.disabledFilters.indexOf('labels') !== -1);
        formatFilterTitle(this.filterLabelsButton, this.filterLabelsTitle);
        this.filterDirectivesButton.prop('disabled', this.compiler.disabledFilters.indexOf('directives') !== -1);
        formatFilterTitle(this.filterDirectivesButton, this.filterDirectivesTitle);
        this.filterCommentsButton.prop('disabled', this.compiler.disabledFilters.indexOf('commentOnly') !== -1);
        formatFilterTitle(this.filterCommentsButton, this.filterCommentsTitle);
        this.filterTrimButton.prop('disabled', this.compiler.disabledFilters.indexOf('trim') !== -1);
        formatFilterTitle(this.filterTrimButton, this.filterTrimTitle);

        if (this.flagsButton) {
            this.flagsButton.prop('disabled', this.flagsViewOpen);
        }
        this.optButton.prop('disabled', this.optViewOpen);
        this.ppButton.prop('disabled', this.ppViewOpen);
        this.astButton.prop('disabled', this.astViewOpen);
        this.irButton.prop('disabled', this.irViewOpen);
        this.llvmOptPipelineButton.prop('disabled', this.llvmOptPipelineViewOpen);
        this.deviceButton.prop('disabled', this.deviceViewOpen);
        this.rustMirButton.prop('disabled', this.rustMirViewOpen);
        this.haskellCoreButton.prop('disabled', this.haskellCoreViewOpen);
        this.haskellStgButton.prop('disabled', this.haskellStgViewOpen);
        this.haskellCmmButton.prop('disabled', this.haskellCmmViewOpen);
        this.rustMacroExpButton.prop('disabled', this.rustMacroExpViewOpen);
        this.rustHirButton.prop('disabled', this.rustHirViewOpen);
        this.cfgButton.prop('disabled', this.cfgViewOpen);
        this.gccDumpButton.prop('disabled', this.gccDumpViewOpen);
        this.gnatDebugTreeButton.prop('disabled', this.gnatDebugTreeViewOpen);
        this.gnatDebugButton.prop('disabled', this.gnatDebugViewOpen);
        // The executorButton does not need to be changed here, because you can create however
        // many executors as you want.

        this.optButton.toggle(!!this.compiler.supportsOptOutput);
        this.ppButton.toggle(!!this.compiler.supportsPpView);
        this.astButton.toggle(!!this.compiler.supportsAstView);
        this.irButton.toggle(!!this.compiler.supportsIrView);
        this.llvmOptPipelineButton.toggle(!!this.compiler.supportsLLVMOptPipelineView);
        this.deviceButton.toggle(!!this.compiler.supportsDeviceAsmView);
        this.rustMirButton.toggle(!!this.compiler.supportsRustMirView);
        this.rustMacroExpButton.toggle(!!this.compiler.supportsRustMacroExpView);
        this.rustHirButton.toggle(!!this.compiler.supportsRustHirView);
        this.haskellCoreButton.toggle(!!this.compiler.supportsHaskellCoreView);
        this.haskellStgButton.toggle(!!this.compiler.supportsHaskellStgView);
        this.haskellCmmButton.toggle(!!this.compiler.supportsHaskellCmmView);
        this.cfgButton.toggle(!!this.compiler.supportsCfg);
        this.gccDumpButton.toggle(!!this.compiler.supportsGccDump);
        this.gnatDebugTreeButton.toggle(!!this.compiler.supportsGnatDebugViews);
        this.gnatDebugButton.toggle(!!this.compiler.supportsGnatDebugViews);
        this.executorButton.toggle(!!this.compiler.supportsExecute);

        this.compilerLicenseButton.toggle(this.hasCompilerLicenseInfo());

        this.enableToolButtons();
    };


    hasCompilerLicenseInfo() {
        return (
            this.compiler.license &&
            (this.compiler.license.preamble || this.compiler.license.link || this.compiler.license.name)
        );
    };


    handlePopularArgumentsResult(result) {
        var popularArgumentsMenu = $(this.domRoot.find('div.populararguments div.dropdown-menu'));

        while (popularArgumentsMenu.children().length > 1) {
            popularArgumentsMenu.children()[1].remove();
        }

        if (result && !this.flagsViewOpen) {
            _.forEach(
                result,
                (arg, key) => {
                    var argumentButton = $(document.createElement('button'));
                    argumentButton.addClass('dropdown-item btn btn-light btn-sm');
                    argumentButton.attr('title', arg.description);
                    argumentButton.data('arg', key);
                    argumentButton.html(
                        "<div class='argmenuitem'>" +
                        "<span class='argtitle'>" +
                        _.escape(key) +
                        '</span>' +
                        "<span class='argdescription'>" +
                        arg.description +
                        '</span>' +
                        '</div>'
                    );

                    argumentButton.on(
                        'click',
                        () =>{
                            var button = argumentButton;
                            var curOptions = this.optionsField.val();
                            if (curOptions.length > 0) {
                                this.optionsField.val(curOptions + ' ' + button.data('arg'));
                            } else {
                                this.optionsField.val(button.data('arg'));
                            }

                            this.optionsField.change();
                        }
                    );

                    popularArgumentsMenu.append(argumentButton);
                }
            );
        }
    };


    generateLicenseInfo() {
        if (this.compiler) {
            // MSVC will take a while to add this
            if (!this.compiler.license) {
                return 'No license information to display for ' + this.compiler.name;
            }
            var result = '';
            var preamble = this.compiler.license.preamble;
            if (preamble) {
                result += preamble + '<br/>';
            }
            var name = this.compiler.license.name;
            var link = this.compiler.license.link;

            if (name || link) {
                result += this.compiler.name + ' is licensed under its ';

                if (link) {
                    var aText = name ? name : link;
                    result += '<a href="' + link + '" target="_blank">' + aText + '</a>';
                } else {
                    result += name;
                }

                result += ' license';
            }

            if (!result) {
                result = 'No license information to display for ' + this.compiler.name;
            }

            return result;
        }
        return 'No compiler selected';
    };


    onFontScale() {
        this.saveState();
    };

// Disable only for initListeners as there are more and more callbacks.
// eslint-disable-next-line max-statements

    initListeners() {
        this.filters.on('change', _.bind(this.onFilterChange, this));
        this.fontScale.on('change', _.bind(this.onFontScale, this));
        this.eventHub.on(
            'broadcastFontScale',
            (scale) => {
                this.fontScale.setScale(scale);
                this.saveState();
            }
        );
        this.paneRenaming.on('renamePane', this.saveState.bind(this));

        this.container.on('destroy', this.close, this);
        this.container.on('resize', this.resize, this);
        this.container.on('shown', this.resize, this);
        this.container.on(
            'open',
            () => {
                this.eventHub.emit('compilerOpen', this.id, this.sourceEditorId ?? 0, this.sourceTreeId);
            },
        );
        this.eventHub.on('editorChange', this.onEditorChange, this);
        this.eventHub.on('compilerFlagsChange', this.onCompilerFlagsChange, this);
        this.eventHub.on('editorClose', this.onEditorClose, this);
        this.eventHub.on('treeClose', this.onTreeClose, this);
        this.eventHub.on('colours', this.onColours, this);
        this.eventHub.on('coloursForCompiler', this.onColoursForCompiler, this);
        this.eventHub.on('resendCompilation', this.onResendCompilation, this);
        this.eventHub.on('findCompilers', this.sendCompiler, this);
        this.eventHub.on('compilerSetDecorations', this.onCompilerSetDecorations, this);
        this.eventHub.on('panesLinkLine', this.onPanesLinkLine, this);
        this.eventHub.on('settingsChange', this.onSettingsChange, this);
        this.eventHub.on('requestCompilation', this.onRequestCompilation, this);

        this.eventHub.on('toolSettingsChange', this.onToolSettingsChange, this);
        this.eventHub.on('toolOpened', this.onToolOpened, this);
        this.eventHub.on('toolClosed', this.onToolClosed, this);

        this.eventHub.on('optViewOpened', this.onOptViewOpened, this);
        this.eventHub.on('optViewClosed', this.onOptViewClosed, this);
        this.eventHub.on('flagsViewOpened', this.onFlagsViewOpened, this);
        this.eventHub.on('flagsViewClosed', this.onFlagsViewClosed, this);
        this.eventHub.on('ppViewOpened', this.onPpViewOpened, this);
        this.eventHub.on('ppViewClosed', this.onPpViewClosed, this);
        this.eventHub.on('ppViewOptionsUpdated', this.onPpViewOptionsUpdated, this);
        this.eventHub.on('astViewOpened', this.onAstViewOpened, this);
        this.eventHub.on('astViewClosed', this.onAstViewClosed, this);
        this.eventHub.on('irViewOpened', this.onIrViewOpened, this);
        this.eventHub.on('irViewClosed', this.onIrViewClosed, this);
        this.eventHub.on('llvmOptPipelineViewOpened', this.onLLVMOptPipelineViewOpened, this);
        this.eventHub.on('llvmOptPipelineViewClosed', this.onLLVMOptPipelineViewClosed, this);
        this.eventHub.on('llvmOptPipelineViewOptionsUpdated', this.onLLVMOptPipelineViewOptionsUpdated, this);
        this.eventHub.on('deviceViewOpened', this.onDeviceViewOpened, this);
        this.eventHub.on('deviceViewClosed', this.onDeviceViewClosed, this);
        this.eventHub.on('rustMirViewOpened', this.onRustMirViewOpened, this);
        this.eventHub.on('rustMirViewClosed', this.onRustMirViewClosed, this);
        this.eventHub.on('rustMacroExpViewOpened', this.onRustMacroExpViewOpened, this);
        this.eventHub.on('rustMacroExpViewClosed', this.onRustMacroExpViewClosed, this);
        this.eventHub.on('rustHirViewOpened', this.onRustHirViewOpened, this);
        this.eventHub.on('rustHirViewClosed', this.onRustHirViewClosed, this);
        this.eventHub.on('haskellCoreViewOpened', this.onHaskellCoreViewOpened, this);
        this.eventHub.on('haskellCoreViewClosed', this.onHaskellCoreViewClosed, this);
        this.eventHub.on('haskellStgViewOpened', this.onHaskellStgViewOpened, this);
        this.eventHub.on('haskellStgViewClosed', this.onHaskellStgViewClosed, this);
        this.eventHub.on('haskellCmmViewOpened', this.onHaskellCmmViewOpened, this);
        this.eventHub.on('haskellCmmViewClosed', this.onHaskellCmmViewClosed, this);
        this.eventHub.on('outputOpened', this.onOutputOpened, this);
        this.eventHub.on('outputClosed', this.onOutputClosed, this);

        this.eventHub.on('gccDumpPassSelected', this.onGccDumpPassSelected, this);
        this.eventHub.on('gccDumpFiltersChanged', this.onGccDumpFiltersChanged, this);
        this.eventHub.on('gccDumpViewOpened', this.onGccDumpViewOpened, this);
        this.eventHub.on('gccDumpViewClosed', this.onGccDumpViewClosed, this);
        this.eventHub.on('gccDumpUIInit', this.onGccDumpUIInit, this);

        this.eventHub.on('gnatDebugTreeViewOpened', this.onGnatDebugTreeViewOpened, this);
        this.eventHub.on('gnatDebugTreeViewClosed', this.onGnatDebugTreeViewClosed, this);
        this.eventHub.on('gnatDebugViewOpened', this.onGnatDebugViewOpened, this);
        this.eventHub.on('gnatDebugViewClosed', this.onGnatDebugViewClosed, this);

        this.eventHub.on('cfgViewOpened', this.onCfgViewOpened, this);
        this.eventHub.on('cfgViewClosed', this.onCfgViewClosed, this);
        this.eventHub.on('resize', this.resize, this);
        this.eventHub.on(
            'requestFilters',
            (id) => {
                if (id === this.id) {
                    this.eventHub.emit('filtersChange', this.id, this.getEffectiveFilters());
                }
            }
        );
        this.eventHub.on(
            'requestCompiler',
            (id) => {
                if (id === this.id) {
                    this.sendCompiler();
                }
            }
        );
        this.eventHub.on('languageChange', this.onLanguageChange, this);

        this.fullTimingInfo.off('click').click(
            () => {
                TimingWidget.displayCompilationTiming(this.lastResult, this.lastTimeTaken);
            }
        );
    };


    initCallbacks() {
        this.initListeners();

        var optionsChange = _.debounce(
            _.bind(function(e) {
                this.onOptionsChange($(e.target).val());
            }, this),
            800
        );

        this.optionsField.on('change', optionsChange).on('keyup', optionsChange);

        this.mouseMoveThrottledFunction = _.throttle(_.bind(this.onMouseMove, this), 50);
        this.outputEditor.onMouseMove(
            (e) => {
                this.mouseMoveThrottledFunction(e);
            }
        );

        this.cursorSelectionThrottledFunction = _.throttle(_.bind(this.onDidChangeCursorSelection, this), 500);
        this.outputEditor.onDidChangeCursorSelection(
            (e) => {
                this.cursorSelectionThrottledFunction(e);
            }
        );

        this.mouseUpThrottledFunction = _.throttle(_.bind(this.onMouseUp, this), 50);
        this.outputEditor.onMouseUp(
            (e) => {
                this.mouseUpThrottledFunction(e);
            }
        );

        this.compileClearCache.on(
            'click',
            () =>{
                this.compilerService.cache.reset();
                this.compile(true);
            }
        );

        this.compilerLicenseButton.on(
            'click',
            () =>{
                var title = this.compiler ? 'License for ' + this.compiler.name : 'No compiler selected';
                this.alertSystem.alert(title, this.generateLicenseInfo());
            }
        );

        // Dismiss the popover on escape.
        $(document).on(
            'keyup.editable',
            (e) => {
                if (e.which === 27) {
                    this.libsButton.popover('hide');
                }
            }
        );

        // Dismiss on any click that isn't either in the opening element, inside
        // the popover or on any alert
        $(document).on(
            'click',
            (e) {
                var elem = this.libsButton;
                var target = $(e.target);
                if (!target.is(elem) && elem.has(target).length === 0 && target.closest('.popover').length === 0) {
                    elem.popover('hide');
                }
            }
        );

        this.eventHub.on('initialised', this.undefer, this);
    };


    onOptionsChange(options) {
        if (this.options !== options) {
            this.options = options;
            this.saveState();
            this.compile();
            this.updateButtons();
            this.sendCompiler();
        }
    };

    private htmlEncode(rawStr) {
        return rawStr.replace(/[\u00A0-\u9999<>&]/g, function(i) {
            return '&#' + i.charCodeAt(0) + ';';
        });
    }


    checkForHints(result) {
        if (result.hints) {
            var self = this;
            result.hints.forEach((hint) => {
                self.alertSystem.notify(this.htmlEncode(hint), {
                    group: 'hints',
                    collapseSimilar: false,
                });
            });
        }
    };


    checkForUnwiseArguments(optionsArray, wasCmake) {
        if (!this.compiler) return;
        // Check if any options are in the unwiseOptions array and remember them
        var unwiseOptions = _.intersection(
            optionsArray,
            this.compiler.unwiseOptions.filter((opt) => {
                return opt !== '';
            })
        );

        var options = unwiseOptions.length === 1 ? 'Option ' : 'Options ';
        var names = unwiseOptions.join(', ');
        var are = unwiseOptions.length === 1 ? ' is ' : ' are ';
        var msg = options + names + are + 'not recommended, as behaviour might change based on server hardware.';

        if (_.contains(optionsArray, '-flto') && !this.filters.state.binary && !wasCmake) {
            this.alertSystem.notify('Option -flto is being used without Compile to Binary.', {
                group: 'unwiseOption',
                collapseSimilar: true,
            });
        }

        if (unwiseOptions.length > 0) {
            this.alertSystem.notify(msg, {
                group: 'unwiseOption',
                collapseSimilar: true,
            });
        }
    };


    updateCompilerInfo() {
        this.updateCompilerName();
        if (this.compiler) {
            if (this.compiler.notification) {
                this.alertSystem.notify(this.compiler.notification, {
                    group: 'compilerwarning',
                    alertClass: 'notification-info',
                    dismissTime: 7000,
                });
            }
            this.prependOptions.data('content', this.compiler.options);
        }
    };


    updateCompilerUI() {
        var panerDropdown = this.domRoot.find('.pane-dropdown');
        var togglePannerAdder = () => {
            panerDropdown.dropdown('toggle');
        };
        this.initToolButtons(togglePannerAdder);
        this.updateButtons();
        this.updateCompilerInfo();
        // Resize in case the new compiler name is too big
        this.resize();
    };


    onCompilerChange(value) {
        this.compiler = this.compilerService.findCompiler(this.currentLangId, value);

        this.deferCompiles = true;
        this.needsCompile = true;

        this.updateLibraries();
        this.saveState();
        this.updateCompilerUI();

        this.undefer();

        this.sendCompiler();
    };


    sendCompiler() {
        this.eventHub.emit('compiler', this.id, this.compiler, this.options, this.sourceEditorId, this.sourceTreeId);
    };


    onEditorClose(editor) {
        if (editor === this.sourceEditorId) {
            // We can't immediately close as an outer loop somewhere in GoldenLayout is iterating over
            // the hierarchy. We can't modify while it's being iterated over.
            this.close();
            _.defer(function(self) {
                self.container.close();
            }, this);
        }
    };


    onTreeClose(tree) {
        if (tree === this.sourceTreeId) {
            this.close();
            _.defer(function(self) {
                self.container.close();
            }, this);
        }
    };


    onFilterChange() {
        var filters = this.getEffectiveFilters();
        this.eventHub.emit('filtersChange', this.id, filters);
        this.saveState();
        this.compile();
        this.updateButtons();
    };


    currentState(): CurrentStateType {
        const state = {
            id: this.id,
            compiler: this.compiler ? this.compiler.id : '',
            source: this.sourceEditorId,
            tree: this.sourceTreeId,
            options: this.options,
            // NB must *not* be effective filters
            filters: this.filters.get(),
            wantOptInfo: this.wantOptInfo,
            libs: this.libsWidget?.get(),
            lang: this.currentLangId,
            selection: this.selection,
            flagsViewOpen: this.flagsViewOpen,
        };
        this.paneRenaming.addState(state);
        this.fontScale.addState(state);
        return state;
    };


    saveState() {
        this.container.setState(this.currentState());
    };


    onColours(editor, colours, scheme) {
        var asmColours = {} as Record<number, Record<number, number>>;
        _.each(
            this.assembly,
            (x, index) => {
                if (x.source && x.source.line > 0) {
                    var editorId = this.getEditorIdBySourcefile(x.source);
                    if (editorId === editor) {
                        if (!asmColours[editorId]) {
                            asmColours[editorId] = {};
                        }
                        asmColours[editorId][index] = colours[x.source.line - 1];
                    }
                }
            }
        );

        _.each(
            asmColours,
            (col) => {
                this.colours = colour.applyColours(this.editor, col, scheme, this.colours);
            }
        );
    };


    onColoursForCompiler(compilerId, colours, scheme) {
        if (this.id === compilerId) {
            this.colours = colour.applyColours(this.editor, colours, scheme, this.colours);
        }
    };


    getCompilerName() {
        return this.compiler ? this.compiler.name : 'No compiler set';
    };


    getLanguageName() {
        var lang = options.languages[this.currentLangId];
        return lang ? lang.name : '?';
    };


    override getPaneName() {
        var langName = this.getLanguageName();
        var compName = this.getCompilerName();
        if (this.sourceEditorId) {
            return compName + ' (' + langName + ', Editor #' + this.sourceEditorId + ', Compiler #' + this.id + ')';
        } else if (this.sourceTreeId) {
            return compName + ' (' + langName + ', Tree #' + this.sourceTreeId + ', Compiler #' + this.id + ')';
        } else {
            return '';
        }
    };


    override updateTitle() {
        var name = this.paneName ? this.paneName : this.getPaneName();
        this.container.setTitle(_.escape(name));
    };


    updateCompilerName() {
        var compilerName = this.getCompilerName();
        var compilerVersion = this.compiler ? this.compiler.version : '';
        var compilerFullVersion = this.compiler && this.compiler.fullVersion ? this.compiler.fullVersion : compilerVersion;
        var compilerNotification = this.compiler ? this.compiler.notification : '';
        this.shortCompilerName.text(compilerName);
        this.setCompilerVersionPopover({
            version: compilerVersion,
            fullVersion: compilerFullVersion
        }, compilerNotification);
        this.updateTitle();
    };


    resendResult() {
        if (!$.isEmptyObject(this.lastResult)) {
            this.eventHub.emit('compileResult', this.id, this.compiler, this.lastResult, this.currentLangId);
            return true;
        }
        return false;
    };


    onResendCompilation(id) {
        if (id === this.id) {
            this.resendResult();
        }
    };


    updateDecorations() {
        this.prevDecorations = this.editor.deltaDecorations(
            this.prevDecorations,
            _.flatten(_.values(this.decorations))
        );
    };


    clearLinkedLines() {
        this.decorations.linkedCode = [];
        this.updateDecorations();
    };


    onPanesLinkLine(compilerId, lineNumber, colBegin, colEnd, revealLine, sender, editorId) {
        if (Number(compilerId) === this.id) {
            const lineNums: number[] = [];
            const directlyLinkedLineNums: number[] = [];
            const signalFromAnotherPane = sender !== this.getPaneName();
            _.each(
                this.assembly,
                (asmLine, i) => {
                    if (asmLine.source && asmLine.source.line === lineNumber) {
                        var fileEditorId = this.getEditorIdBySourcefile(asmLine.source);
                        if (fileEditorId && editorId === fileEditorId) {
                            var line = i + 1;
                            lineNums.push(line);
                            var currentCol = asmLine.source.column;
                            if (signalFromAnotherPane && currentCol && colBegin <= currentCol && currentCol <= colEnd) {
                                directlyLinkedLineNums.push(line);
                            }
                        }
                    }
                }
            );

            if (revealLine && lineNums[0]) {
                this.pushRevealJump();
                this.hub.activateTabForContainer(this.container);
                this.editor.revealLineInCenter(lineNums[0]);
            }

            var lineClass = sender !== this.getPaneName() ? 'linked-code-decoration-line' : '';
            var linkedLinesDecoration = _.map(lineNums, function(line) {
                return {
                    range: new monaco.Range(line, 1, line, 1),
                    options: {
                        isWholeLine: true,
                        linesDecorationsClassName: 'linked-code-decoration-margin',
                        className: lineClass,
                    },
                };
            });
            var directlyLinkedLinesDecoration = _.map(directlyLinkedLineNums, function(line) {
                return {
                    range: new monaco.Range(line, 1, line, 1),
                    options: {
                        isWholeLine: true,
                        inlineClassName: 'linked-code-decoration-column',
                    },
                };
            });
            this.decorations.linkedCode = linkedLinesDecoration.concat(directlyLinkedLinesDecoration);
            if (this.linkedFadeTimeoutId !== -1) {
                clearTimeout(this.linkedFadeTimeoutId);
            }
            this.linkedFadeTimeoutId = setTimeout(
                () =>{
                    this.clearLinkedLines();
                    this.linkedFadeTimeoutId = -1;
                },
                5000
            );
            this.updateDecorations();
        }
    };


    onCompilerSetDecorations(id, lineNums, revealLine) {
        if (Number(id) === this.id) {
            if (revealLine && lineNums[0]) {
                this.pushRevealJump();
                this.editor.revealLineInCenter(lineNums[0]);
            }
            this.decorations.linkedCode = _.map(lineNums, function(line) {
                return {
                    range: new monaco.Range(line, 1, line, 1),
                    options: {
                        isWholeLine: true,
                        linesDecorationsClassName: 'linked-code-decoration-margin',
                        inlineClassName: 'linked-code-decoration-inline',
                    },
                };
            });
            this.updateDecorations();
        }
    };


    setCompilationOptionsPopover(content) {
        this.prependOptions.popover('dispose');
        this.prependOptions.popover({
            content: content || 'No options in use',
            template:
                '<div class="popover' +
                (content ? ' compiler-options-popover' : '') +
                '" role="tooltip"><div class="arrow"></div>' +
                '<h3 class="popover-header"></h3><div class="popover-body"></div></div>',
        });
    };


    setCompilerVersionPopover(version, notification) {
        this.fullCompilerName.popover('dispose');
        // `notification` contains HTML from a config file, so is 'safe'.
        // `version` comes from compiler output, so isn't, and is escaped.
        var bodyContent = $('<div>');
        var versionContent = $('<div>').html(_.escape(version.version));
        bodyContent.append(versionContent);
        if (version.fullVersion && version.fullVersion.trim() !== version.version.trim()) {
            var hiddenSection = $('<div>');
            var lines = _.map(version.fullVersion.split('\n'), function(line) {
                return _.escape(line);
            }).join('<br/>');
            var hiddenVersionText = $('<div>').html(lines).hide();
            var clickToExpandContent = $('<a>')
                .attr('href', 'javascript:;')
                .text('Toggle full version output')
                .on(
                    'click',
                    () =>{
                        versionContent.toggle();
                        hiddenVersionText.toggle();
                        this.fullCompilerName.popover('update');
                    }
                );
            hiddenSection.append(hiddenVersionText).append(clickToExpandContent);
            bodyContent.append(hiddenSection);
        }
        this.fullCompilerName.popover({
            html: true,
            title: notification
                ? $.parseHTML('<span>Compiler Version: ' + notification + '</span>')[0]
                : 'Full compiler version',
            content: bodyContent,
            template:
                '<div class="popover' +
                (version ? ' compiler-options-popover' : '') +
                '" role="tooltip">' +
                '<div class="arrow"></div>' +
                '<h3 class="popover-header"></h3><div class="popover-body"></div>' +
                '</div>',
        });
    };


    onRequestCompilation(editorId, treeId) {
        if (editorId === this.sourceEditorId || (treeId && treeId === this.sourceTreeId)) {
            this.compile();
        }
    };


    override onSettingsChange(newSettings) {
        var before = this.settings;
        this.settings = _.clone(newSettings);
        if (!before.lastHoverShowSource && this.settings.hoverShowSource) {
            this.onCompilerSetDecorations(this.id, []);
        }
        this.editor.updateOptions({
            contextmenu: this.settings.useCustomContextMenu,
            minimap: {
                enabled: this.settings.showMinimap && !options.embedded,
            },
            fontFamily: this.settings.editorsFFont,
            codeLensFontFamily: this.settings.editorsFFont,
            fontLigatures: this.settings.editorsFLigatures,
        });
    };

    private readonly hexLike = /^(#?[$]|0x)([0-9a-fA-F]+)$/;
    private readonly hexLike2 = /^(#?)([0-9a-fA-F]+)H$/;
    private readonly decimalLike = /^(#?)(-?[0-9]+)$/;

    private parseNumericValue(value) {
        var hexMatch = this.hexLike.exec(value) || this.hexLike2.exec(value);
        if (hexMatch) return bigInt(hexMatch[2], 16);

        var decMatch = this.decimalLike.exec(value);
        if (decMatch) return bigInt(decMatch[2]);

        return null;
    }

    private getNumericToolTip(value) {
        var numericValue = this.parseNumericValue(value);
        if (numericValue === null) return null;

        // Decimal representation.
        var result = numericValue.toString(10);

        // Hexadecimal representation.
        if (numericValue.isNegative()) {
            var masked = bigInt('ffffffffffffffff', 16).and(numericValue);
            result += ' = 0x' + masked.toString(16).toUpperCase();
        } else {
            result += ' = 0x' + numericValue.toString(16).toUpperCase();
        }

        // Printable ASCII character.
        if (numericValue.greaterOrEquals(0x20) && numericValue.lesserOrEquals(0x7e)) {
            var char = String.fromCharCode(numericValue.valueOf());
            result += " = '" + char + "'";
        }

        return result;
    }

    private getAsmInfo(opcode, instructionSet) {
        var cacheName = 'asm/' + (instructionSet ? instructionSet + '/' : '') + opcode;
        var cached = OpcodeCache.get(cacheName);
        if (cached) {
            if (cached.found) {
                return Promise.resolve(cached.data);
            }
            return Promise.reject(cached.data);
        }
        return new Promise(function(resolve, reject) {
            getAssemblyDocumentation({opcode: opcode, instructionSet: instructionSet})
                .then(function(response) {
                    response.json().then(function(body) {
                        if (response.status === 200) {
                            OpcodeCache.set(cacheName, {found: true, data: body});
                            resolve(body);
                        } else {
                            OpcodeCache.set(cacheName, {found: false, data: body.error});
                            reject(body.error);
                        }
                    });
                })
                .catch(function(error) {
                    reject('Fetch error: ' + error);
                });
        });
    }


    override onDidChangeCursorSelection(e) {
        if (this.awaitingInitialResults) {
            this.selection = e.selection;
            this.saveState();
        }
    };


    onMouseUp(e) {
        if (e === null || e.target === null || e.target.position === null) return;

        if (e.event.ctrlKey && e.event.leftButton) {
            this.jumpToLabel(e.target.position);
        }
    };


    onMouseMove(e) {
        if (e === null || e.target === null || e.target.position === null) return;
        var hoverShowSource = this.settings.hoverShowSource === true;
        if (this.assembly) {
            var hoverAsm = this.assembly[e.target.position.lineNumber - 1];
            if (hoverShowSource && hoverAsm) {
                this.clearLinkedLines();
                // We check that we actually have something to show at this point!
                var sourceLine = -1;
                var sourceColBegin = -1;
                var sourceColEnd = -1;
                if (hoverAsm.source) {
                    sourceLine = hoverAsm.source.line;
                    if (hoverAsm.source.column) {
                        sourceColBegin = hoverAsm.source.column;
                        sourceColEnd = sourceColBegin;
                    }

                    var editorId = this.getEditorIdBySourcefile(hoverAsm.source);
                    if (editorId) {
                        this.eventHub.emit('editorLinkLine', editorId, sourceLine, sourceColBegin, sourceColEnd, false);

                        this.eventHub.emit(
                            'panesLinkLine',
                            this.id,
                            sourceLine,
                            sourceColBegin,
                            sourceColEnd,
                            false,
                            this.getPaneName(),
                            //editorId
                        );
                    }
                }
            }
        }
        var currentWord = this.editor.getModel()?.getWordAtPosition(e.target.position);
        if (currentWord && currentWord.word) {
            var word = currentWord.word;
            var startColumn = currentWord.startColumn;
            // Avoid throwing an exception if somehow (How?) we have a non existent lineNumber.
            // c.f. https://sentry.io/matt-godbolt/compiler-explorer/issues/285270358/
            if (e.target.position.lineNumber <= this.editor.getModel()?.getLineCount() ?? 0) {
                // Hacky workaround to check for negative numbers.
                // c.f. https://github.com/compiler-explorer/compiler-explorer/issues/434
                const lineContent = this.editor.getModel()?.getLineContent(e.target.position.lineNumber);
                if (lineContent && lineContent[currentWord.startColumn - 2] === '-') {
                    word = '-' + word;
                    startColumn -= 1;
                }
            }
            currentWord.range = new monaco.Range(
                e.target.position.lineNumber,
                Math.max(startColumn, 1),
                e.target.position.lineNumber,
                currentWord.endColumn
            );
            var numericToolTip = this.getNumericToolTip(word);
            if (numericToolTip) {
                this.decorations.numericToolTip = {
                    range: currentWord.range,
                    options: {
                        isWholeLine: false,
                        hoverMessage: [
                            {
                                // We use double `` as numericToolTip may include a single ` character.
                                value: '``' + numericToolTip + '``',
                            },
                        ],
                    },
                };
                this.updateDecorations();
            }
            var hoverShowAsmDoc = this.settings.hoverShowAsmDoc === true;
            if (hoverShowAsmDoc && this.compiler && this.compiler.supportsAsmDocs && this.isWordAsmKeyword(currentWord)) {
                this.getAsmInfo(currentWord.word, this.compiler.instructionSet).then(
                    (response) => {
                        if (!response) return;
                        this.decorations.asmToolTip = {
                            range: currentWord.range,
                            options: {
                                isWholeLine: false,
                                hoverMessage: [
                                    {
                                        value: response.tooltip + '\n\nMore information available in the context menu.',
                                        isTrusted: true,
                                    },
                                ],
                            },
                        };
                        this.updateDecorations();
                    }
                );
            }
        }
    };


    getLineTokens(line) {
        var model = this.editor.getModel();
        if (!model || line > model.getLineCount()) return [];
        var flavour = model.getLanguageId();
        var tokens = monaco.editor.tokenize(model.getLineContent(line), flavour);
        return tokens.length > 0 ? tokens[0] : [];
    };


    isWordAsmKeyword(word) {
        return _.some(this.getLineTokens(word.range.startLineNumber), function(t) {
            return t.offset + 1 === word.startColumn && t.type === 'keyword.asm';
        });
    };


    onAsmToolTip(ed) {
        ga.proxy('send', {
            hitType: 'event',
            eventCategory: 'OpenModalPane',
            eventAction: 'AsmDocs',
        });
        var pos = ed.getPosition();
        if (!pos || !ed.getModel()) return;
        var word = ed.getModel().getWordAtPosition(pos);
        if (!word || !word.word) return;
        var opcode = word.word.toUpperCase();

        function newGitHubIssueUrl() {
            return (
                'https://github.com/compiler-explorer/compiler-explorer/issues/new?title=' +
                encodeURIComponent('[BUG] Problem with ' + opcode + ' opcode')
            );
        }

        function appendInfo(url) {
            return (
                '<br><br>For more information, visit <a href="' +
                url +
                '" target="_blank" rel="noopener noreferrer">the ' +
                opcode +
                ' documentation <sup><small class="fas fa-external-link-alt opens-new-window"' +
                ' title="Opens in a new window"></small></sup></a>.' +
                '<br>If the documentation for this opcode is wrong or broken in some way, ' +
                'please feel free to <a href="' +
                newGitHubIssueUrl() +
                '" target="_blank" rel="noopener noreferrer">' +
                'open an issue on GitHub <sup><small class="fas fa-external-link-alt opens-new-window" ' +
                'title="Opens in a new window"></small></sup></a>.'
            );
        }

        this.getAsmInfo(word.word, this.compiler?.instructionSet).then(
            (asmHelp: any) => {
                if (asmHelp) {
                    this.alertSystem.alert(opcode + ' help', asmHelp.html + appendInfo(asmHelp.url), function() {
                        ed.focus();
                        ed.setPosition(pos);
                    });
                } else {
                    this.alertSystem.notify('This token was not found in the documentation. Sorry!', {
                        group: 'notokenindocs',
                        alertClass: 'notification-error',
                        dismissTime: 5000,
                    });
                }
            },
            (rejection) => {
                this.alertSystem.notify(
                    'There was an error fetching the documentation for this opcode (' + rejection + ').',
                    {
                        group: 'notokenindocs',
                        alertClass: 'notification-error',
                        dismissTime: 5000,
                    }
                );
            }
        );
    };
    
    handleCompilationStatus(status) {
        CompilerService.handleCompilationStatus(this.statusLabel, this.statusIcon, status);
    };
    
    onLanguageChange(editorId, newLangId, treeId) {
        if (
            (this.sourceEditorId && this.sourceEditorId === editorId) ||
            (this.sourceTreeId && this.sourceTreeId === treeId)
        ) {
            var oldLangId = this.currentLangId;
            this.currentLangId = newLangId;
            // Store the current selected stuff to come back to it later in the same session (Not state stored!)
            this.infoByLang[oldLangId] = {
                compiler: this.compiler && this.compiler.id ? this.compiler.id : options.defaultCompiler[oldLangId],
                options: this.options,
            };
            var info = this.infoByLang[this.currentLangId] || {};
            this.deferCompiles = true;
            this.initLangAndCompiler({lang: newLangId, compiler: info.compiler});
            this.updateCompilersSelector(info);
            this.saveState();
            this.updateCompilerUI();
            this.setAssembly(this.fakeAsm(''));
            // this is a workaround to delay compilation further until the Editor sends a compile request
            this.needsCompile = false;

            this.undefer();
            this.sendCompiler();
        }
    }


    private getCurrentLangCompilers() {
        return this.compilerService.getCompilersForLang(this.currentLangId);
    }
    
    updateCompilersSelector(info) {
        this.compilerPicker.update(this.currentLangId, this.compiler?.id ?? '');
        this.options = info.options || '';
        this.optionsField.val(this.options);
    }
}
