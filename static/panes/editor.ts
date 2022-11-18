// Copyright (c) 2016, Compiler Explorer Authors
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
import $ from 'jquery';
import * as colour from '../colour';
import * as loadSaveLib from '../widgets/load-save';
import * as Components from '../components';
import * as monaco from 'monaco-editor';
import {Buffer} from 'buffer';
import {options} from '../options';
import {Alert} from '../alert';
import {ga} from '../analytics';
import * as monacoVim from 'monaco-vim';
import * as monacoConfig from '../monaco-config';
import * as quickFixesHandler from '../quick-fixes-handler';
import TomSelect from 'tom-select';
import {Settings, SiteSettings} from '../settings';
import '../formatter-registry';
import '../modes/_all';
import {MonacoPane} from './pane';
import {Hub} from '../hub';
import {MonacoPaneState} from './pane.interfaces';
import {Container} from 'golden-layout';
import {EditorState, LanguageSelectData} from './editor.interfaces';
import {Language, LanguageKey} from '../../types/languages.interfaces';
import {editor} from 'monaco-editor';
import IModelDeltaDecoration = editor.IModelDeltaDecoration;
import {MessageWithLocation, ResultLine} from '../../types/resultline/resultline.interfaces';
import {CompilerInfo} from '../../types/compiler.interfaces';
import {CompilationResult} from '../../types/compilation/compilation.interfaces';
import {Decoration, Motd} from '../motd.interfaces';
import type {escape_html} from 'tom-select/dist/types/utils';
import ICursorSelectionChangedEvent = editor.ICursorSelectionChangedEvent;
import {Compiler} from './compiler';

const loadSave = new loadSaveLib.LoadSave();
const languages = options.languages as Record<string, Language | undefined>;

type ResultLineWithSourcePane = ResultLine & {
    sourcePane: string;
};

// eslint-disable-next-line max-statements
export class Editor extends MonacoPane<monaco.editor.IStandaloneCodeEditor, EditorState> {
    private id: number;
    private ourCompilers: Record<string, boolean>;
    private ourExecutors: Record<number, boolean>;
    private httpRoot: string;
    private asmByCompiler: Record<string, ResultLine[] | undefined>;
    private defaultFileByCompiler: Record<number, string>;
    private busyCompilers: Record<number, boolean>;
    private colours: string[];
    private treeCompilers: Record<number, Record<number, boolean> | undefined>;
    private decorations: Record<string, IModelDeltaDecoration[] | undefined>;
    private prevDecorations: string[];
    private extraDecorations?: Decoration[];
    private fadeTimeoutId: NodeJS.Timeout | null;
    private editorSourceByLang: Record<LanguageKey, string | undefined>;
    private alertSystem: Alert;
    private filename: string | false;
    private awaitingInitialResults: boolean;
    private revealJumpStack: editor.ICodeEditorViewState[];
    private langKeys: string[];
    private legacyReadOnly?: boolean;
    private selectize: TomSelect;
    private lastChangeEmitted: string | null;
    private languageBtn: JQuery<HTMLElement>;
    public currentLanguage?: Language;
    private waitingForLanguage: boolean;
    private currentCursorPosition: JQuery<HTMLElement>;
    private mouseMoveThrottledFunction?: ((e: monaco.editor.IEditorMouseEvent) => void) & _.Cancelable;
    private cursorSelectionThrottledFunction?: (e: monaco.editor.ICursorSelectionChangedEvent) => void & _.Cancelable;
    private vimMode: any;
    private vimFlag: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private loadSaveButton: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private addExecutorButton: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private conformanceViewerButton: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private cppInsightsButton: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private quickBenchButton: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private languageInfoButton: JQuery;
    private nothingCtrlSSince?: number;
    private nothingCtrlSTimes?: number;
    private isCpp: editor.IContextKey<boolean>;
    private isClean: editor.IContextKey<boolean>;
    private debouncedEmitChange: (() => void) & _.Cancelable;
    private revealJumpStackHasElementsCtxKey: editor.IContextKey<boolean>;

    constructor(hub: Hub, state: MonacoPaneState & EditorState, container: Container) {
        super(hub, container, state);

        this.alertSystem = new Alert();
        this.alertSystem.prefixMessage = 'Editor #' + this.id;

        if (this.currentLanguage) this.onLanguageChange(this.currentLanguage.id, true);

        if (state.source !== undefined) {
            this.setSource(state.source);
        } else {
            this.updateEditorCode();
        }

        const startFolded = /^[/*#;]+\s*setup.*/;
        if (state.source && state.source.match(startFolded)) {
            // With reference to https://github.com/Microsoft/monaco-editor/issues/115
            // I tried that and it didn't work, but a delay of 500 seems to "be enough".
            // FIXME: Currently not working - No folding is performed
            setTimeout(() => {
                this.editor.setSelection(new monaco.Selection(1, 1, 1, 1));
                this.editor.focus();
                this.editor.getAction('editor.fold').run();
                //this.editor.clearSelection();
            }, 500);
        }

        if (this.settings.useVim) {
            this.enableVim();
        }

        // We suppress posting changes until the user has stopped typing by:
        // * Using _.debounce() to run emitChange on any key event or change
        //   only after a delay.
        // * Only actually triggering a change if the document text has changed from
        //   the previous emitted.
        this.lastChangeEmitted = null;
        this.onSettingsChange(this.settings);
        // this.editor.on("keydown", _.bind(function () {
        //     // Not strictly a change; but this suppresses changes until some time
        //     // after the last key down (be it an actual change or a just a cursor
        //     // movement etc).
        //     this.debouncedEmitChange();
        // }, this));
    }

    override initializeDefaults(): void {
        this.ourCompilers = {};
        this.ourExecutors = {};
        this.asmByCompiler = {};
        this.defaultFileByCompiler = {};
        this.busyCompilers = {};
        this.colours = [];
        this.treeCompilers = {};

        this.decorations = {};
        this.prevDecorations = [];
        this.extraDecorations = [];

        this.fadeTimeoutId = null;

        this.editorSourceByLang = {} as Record<LanguageKey, string | undefined>;

        this.awaitingInitialResults = false;

        this.revealJumpStack = [];
    }

    override registerOpeningAnalyticsEvent(): void {
        ga.proxy('send', {
            hitType: 'event',
            eventCategory: 'OpenViewPane',
            eventAction: 'Editor',
        });
        ga.proxy('send', {
            hitType: 'event',
            eventCategory: 'LanguageChange',
            eventAction: this.currentLanguage?.id,
        });
    }

    override getInitialHTML(): string {
        return $('#codeEditor').html();
    }

    override createEditor(editorRoot: HTMLElement): editor.IStandaloneCodeEditor {
        const editor = monaco.editor.create(
            editorRoot,
            // @ts-expect-error: options.readOnly and anything inside window.compilerExplorerOptions is unknown
            monacoConfig.extendConfig(
                {
                    readOnly:
                        !!options.readOnly ||
                        this.legacyReadOnly ||
                        // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
                        (window.compilerExplorerOptions && window.compilerExplorerOptions.mobileViewer),
                    glyphMargin: !options.embedded,
                },
                this.settings as SiteSettings
            )
        );

        editor.getModel()?.setEOL(monaco.editor.EndOfLineSequence.LF);

        return editor;
    }

    onMotd(motd: Motd): void {
        this.extraDecorations = motd.decorations;
        this.updateExtraDecorations();
    }

    updateExtraDecorations(): void {
        let decorationsDirty = false;
        this.extraDecorations?.forEach(decoration => {
            // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
            if (
                // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
                decoration.filter &&
                this.currentLanguage?.name &&
                decoration.filter.indexOf(this.currentLanguage.name.toLowerCase()) < 0
            )
                return;
            const match = this.editor.getModel()?.findNextMatch(
                decoration.regex,
                {
                    column: 1,
                    lineNumber: 1,
                },
                true,
                true,
                null,
                false
            );

            if (match !== this.decorations[decoration.name]) {
                decorationsDirty = true;
                this.decorations[decoration.name] = match
                    ? [{range: match.range, options: decoration.decoration}]
                    : undefined;
            }
        });

        // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
        if (decorationsDirty) this.updateDecorations();
    }

    // If compilerId is undefined, every compiler will be pinged
    maybeEmitChange(force?: boolean, compilerId?: number): void {
        const source = this.getSource();
        if (!force && source === this.lastChangeEmitted) return;

        this.updateExtraDecorations();

        this.lastChangeEmitted = source ?? null;
        this.eventHub.emit(
            'editorChange',
            this.id,
            this.lastChangeEmitted ?? '',
            this.currentLanguage?.id ?? '',
            compilerId
        );
    }

    override updateState(): void {
        const state = {
            id: this.id,
            source: this.getSource(),
            lang: this.currentLanguage?.id,
            selection: this.selection,
            filename: this.filename,
        };
        this.fontScale.addState(state);
        this.container.setState(state);

        this.updateButtons();
    }

    setSource(newSource: string): void {
        this.updateSource(newSource);

        if (window.compilerExplorerOptions.mobileViewer) {
            $(this.domRoot.find('.monaco-placeholder textarea')).hide();
        }
    }

    onNewSource(editorId: number, newSource: string): void {
        if (this.id === editorId) {
            this.setSource(newSource);
        }
    }

    getSource(): string | undefined {
        return this.editor.getModel()?.getValue();
    }

    getLanguageFromState(state: MonacoPaneState & EditorState): Language | undefined {
        let newLanguage = languages[this.langKeys[0]];
        this.waitingForLanguage = Boolean(state.source && !state.lang);
        if (this.settings.defaultLanguage && this.settings.defaultLanguage in languages) {
            newLanguage = languages[this.settings.defaultLanguage];
        }

        if (state.lang && state.lang in languages) {
            newLanguage = languages[state.lang];
        } else if (
            this.settings.newEditorLastLang &&
            this.hub.lastOpenedLangId &&
            this.hub.lastOpenedLangId in languages
        ) {
            newLanguage = languages[this.hub.lastOpenedLangId];
        }

        return newLanguage;
    }

    override registerCallbacks(): void {
        this.container.on('shown', this.resize, this);
        this.container.on('open', () => {
            this.eventHub.emit('editorOpen', this.id);
        });
        this.container.layoutManager.on('initialised', () => {
            // Once initialized, let everyone know what text we have.
            this.maybeEmitChange();
            // And maybe ask for a compilation (Will hit the cache most of the time)
            this.requestCompilation();
        });

        this.eventHub.on('treeCompilerEditorIncludeChange', this.onTreeCompilerEditorIncludeChange, this);
        this.eventHub.on('treeCompilerEditorExcludeChange', this.onTreeCompilerEditorExcludeChange, this);
        this.eventHub.on('coloursForEditor', this.onColoursForEditor, this);
        this.eventHub.on('compilerOpen', this.onCompilerOpen, this);
        this.eventHub.on('executorOpen', this.onExecutorOpen, this);
        this.eventHub.on('executorClose', this.onExecutorClose, this);
        this.eventHub.on('compiling', this.onCompiling, this);
        this.eventHub.on('executeResult', this.onExecuteResponse, this);
        this.eventHub.on('selectLine', this.onSelectLine, this);
        this.eventHub.on('editorSetDecoration', this.onEditorSetDecoration, this);
        this.eventHub.on('editorDisplayFlow', this.onEditorDisplayFlow, this);
        this.eventHub.on('editorLinkLine', this.onEditorLinkLine, this);
        this.eventHub.on('conformanceViewOpen', this.onConformanceViewOpen, this);
        this.eventHub.on('conformanceViewClose', this.onConformanceViewClose, this);
        this.eventHub.on('newSource', this.onNewSource, this);
        this.eventHub.on('motd', this.onMotd, this);
        this.eventHub.on('findEditors', this.sendEditor, this);
        this.eventHub.emit('requestMotd');

        this.debouncedEmitChange = _.debounce(() => {
            this.maybeEmitChange();
        }, this.settings.delayAfterChange);

        this.editor.getModel()?.onDidChangeContent(() => {
            this.debouncedEmitChange();
            this.updateState();
        });

        this.mouseMoveThrottledFunction = _.throttle(this.onMouseMove.bind(this), 50);

        this.editor.onMouseMove(e => {
            if (this.mouseMoveThrottledFunction) this.mouseMoveThrottledFunction(e);
        });

        if (window.compilerExplorerOptions.mobileViewer) {
            // workaround for issue with contextmenu not going away when tapping somewhere else on the screen
            this.editor.onDidChangeCursorSelection(() => {
                const contextmenu = $('div.context-view.monaco-menu-container');
                if (contextmenu.css('display') !== 'none') {
                    contextmenu.hide();
                }
            });
        }

        this.cursorSelectionThrottledFunction = _.throttle(
            this.onDidChangeCursorSelection.bind(this) as (
                e: editor.ICursorSelectionChangedEvent
            ) => void & _.Cancelable,
            500
        );
        this.editor.onDidChangeCursorSelection(e => {
            if (this.cursorSelectionThrottledFunction) this.cursorSelectionThrottledFunction(e);
        });

        this.editor.onDidFocusEditorText(_.bind(this.onDidFocusEditorText, this));
        this.editor.onDidBlurEditorText(_.bind(this.onDidBlurEditorText, this));
        this.editor.onDidChangeCursorPosition(_.bind(this.onDidChangeCursorPosition, this));

        this.eventHub.on('initialised', this.maybeEmitChange, this);

        $(document).on('keyup.editable', e => {
            // @ts-expect-error: Document and JQuery<HTMLElement> have no overlap
            if (e.target === this.domRoot.find('.monaco-placeholder .inputarea')[0]) {
                if (e.which === 27) {
                    this.onEscapeKey();
                } else if (e.which === 45) {
                    this.onInsertKey(e);
                }
            }
        });
    }

    sendEditor(): void {
        this.eventHub.emit('editorOpen', this.id);
    }

    onMouseMove(e: editor.IEditorMouseEvent): void {
        // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
        if (e !== null && e.target !== null && this.settings.hoverShowSource && e.target.position !== null) {
            this.clearLinkedLine();
            const pos = e.target.position;
            this.tryPanesLinkLine(pos.lineNumber, pos.column, false);
        }
    }

    override onDidChangeCursorSelection(e: ICursorSelectionChangedEvent): void {
        if (this.awaitingInitialResults) {
            this.selection = e.selection;
            this.updateState();
        }
    }

    onDidChangeCursorPosition(e: ICursorSelectionChangedEvent): void {
        // @ts-expect-error: 'position' is not a property of 'e'
        if (e.position) {
            // @ts-expect-error: 'position' is not a property of 'e'
            this.currentCursorPosition.text('(' + e.position.lineNumber + ', ' + e.position.column + ')');
        }
    }

    onDidFocusEditorText(): void {
        const position = this.editor.getPosition();
        if (position) {
            this.currentCursorPosition.text('(' + position.lineNumber + ', ' + position.column + ')');
        }
        this.currentCursorPosition.show();
    }

    onDidBlurEditorText(): void {
        this.currentCursorPosition.text('');
        this.currentCursorPosition.hide();
    }

    onEscapeKey(): void {
        // @ts-expect-error: IStandaloneCodeEditor is missing this property
        if (this.editor.vimInUse) {
            const currentState = monacoVim.VimMode.Vim.maybeInitVimState_(this.vimMode);
            if (currentState.insertMode) {
                monacoVim.VimMode.Vim.exitInsertMode(this.vimMode);
            } else if (currentState.visualMode) {
                monacoVim.VimMode.Vim.exitVisualMode(this.vimMode, false);
            }
        }
    }

    onInsertKey(event: JQuery.TriggeredEvent<Document, undefined, Document, Document>): void {
        // @ts-expect-error: IStandaloneCodeEditor is missing this property
        if (this.editor.vimInUse) {
            const currentState = monacoVim.VimMode.Vim.maybeInitVimState_(this.vimMode);
            if (!currentState.insertMode) {
                const insertEvent = {
                    preventDefault: event.preventDefault,
                    stopPropagation: event.stopPropagation,
                    browserEvent: {
                        key: 'i',
                        defaultPrevented: false,
                    },
                    keyCode: 39,
                };
                this.vimMode.handleKeyDown(insertEvent);
            }
        }
    }

    enableVim(): void {
        const statusElem = this.domRoot.find('#v-status')[0];
        const vimMode = monacoVim.initVimMode(this.editor, statusElem);
        this.vimMode = vimMode;
        this.vimFlag.prop('class', 'btn btn-info');
        // @ts-expect-error: IStandaloneCodeEditor is missing this property
        this.editor.vimInUse = true;
    }

    disableVim(): void {
        this.vimMode.dispose();
        this.domRoot.find('#v-status').html('');
        this.vimFlag.prop('class', 'btn btn-light');
        // @ts-expect-error: IStandaloneCodeEditor is missing this property
        this.editor.vimInUse = false;
    }

    override initializeGlobalDependentProperties(): void {
        super.initializeGlobalDependentProperties();

        this.httpRoot = window.httpRoot;
        this.langKeys = Object.keys(languages);
    }

    override initializeStateDependentProperties(state: MonacoPaneState & EditorState): void {
        super.initializeStateDependentProperties(state);

        this.id = state.id || this.hub.nextEditorId();

        this.filename = state.filename ?? false;
        this.selection = state.selection;
        this.legacyReadOnly = state.options && !!state.options.readOnly;

        this.currentLanguage = this.getLanguageFromState(state);
        if (!this.currentLanguage) {
            //this.currentLanguage = options.defaultCompiler;
        }
    }

    override registerButtons(state: MonacoPaneState & EditorState): void {
        super.registerButtons(state);

        this.topBar = this.domRoot.find('.top-bar');
        this.hideable = this.domRoot.find('.hideable');

        this.loadSaveButton = this.domRoot.find('.load-save');
        const paneAdderDropdown = this.domRoot.find('.add-pane');
        const addCompilerButton = this.domRoot.find('.btn.add-compiler');
        this.addExecutorButton = this.domRoot.find('.btn.add-executor');
        this.conformanceViewerButton = this.domRoot.find('.btn.conformance');
        const addEditorButton = this.domRoot.find('.btn.add-editor');
        const toggleVimButton = this.domRoot.find('#vim-flag');
        this.vimFlag = this.domRoot.find('#vim-flag');
        toggleVimButton.on('click', () => {
            // @ts-expect-error: IStandaloneCodeEditor is missing this property
            if (this.editor.vimInUse) {
                this.disableVim();
            } else {
                this.enableVim();
            }
        });

        // Ensure that the button is disabled if we don't have anything to select
        // Note that is might be disabled for other reasons beforehand
        if (this.langKeys.length <= 1) {
            this.languageBtn.prop('disabled', true);
        }

        const usableLanguages = Object.values(languages).filter(language => {
            return this.hub.compilerService.compilersByLang[language?.id ?? ''];
        });

        this.languageInfoButton = this.domRoot.find('.language-info');
        this.languageInfoButton.popover({});
        this.languageBtn = this.domRoot.find('.change-language');
        this.selectize = new TomSelect(this.languageBtn as any, {
            sortField: 'name',
            valueField: 'id',
            labelField: 'name',
            searchField: ['name'],
            placeholder: '🔍 Select a language...',
            options: _.map(usableLanguages, _.identity) as any[],
            items: this.currentLanguage?.id ? [this.currentLanguage.id] : [],
            dropdownParent: 'body',
            plugins: ['dropdown_input'],
            onChange: _.bind(this.onLanguageChange, this),
            closeAfterSelect: true,
            render: {
                option: this.renderSelectizeOption.bind(this),
                item: this.renderSelectizeItem.bind(this),
            },
        });

        // NB a new compilerConfig needs to be created every time; else the state is shared
        // between all compilers created this way. That leads to some nasty-to-find state
        // bugs e.g. https://github.com/compiler-explorer/compiler-explorer/issues/225
        const getCompilerConfig = () => {
            return Components.getCompiler(this.id, this.currentLanguage?.id ?? '');
        };

        const getExecutorConfig = () => {
            return Components.getExecutor(this.id, this.currentLanguage?.id ?? '');
        };

        const getConformanceConfig = () => {
            // TODO: this doesn't pass any treeid introduced by #3360
            return Components.getConformanceView(this.id, 0, this.getSource() ?? '', this.currentLanguage?.id ?? '');
        };

        const getEditorConfig = () => {
            return Components.getEditor();
        };

        const addPaneOpener = (dragSource, dragConfig) => {
            this.container.layoutManager
                .createDragSource(dragSource, dragConfig)
                // @ts-expect-error: createDragSource returns not void
                ._dragListener.on('dragStart', () => {
                    paneAdderDropdown.dropdown('toggle');
                });

            dragSource.on('click', () => {
                const insertPoint =
                    this.hub.findParentRowOrColumn(this.container.parent) ||
                    this.container.layoutManager.root.contentItems[0];
                insertPoint.addChild(dragConfig);
            });
        };

        addPaneOpener(addCompilerButton, getCompilerConfig);
        addPaneOpener(this.addExecutorButton, getExecutorConfig);
        addPaneOpener(this.conformanceViewerButton, getConformanceConfig);
        addPaneOpener(addEditorButton, getEditorConfig);

        this.initLoadSaver();
        $(this.domRoot).on('keydown', event => {
            if ((event.ctrlKey || event.metaKey) && String.fromCharCode(event.which).toLowerCase() === 's') {
                this.handleCtrlS(event);
            }
        });

        if (options.thirdPartyIntegrationEnabled) {
            this.cppInsightsButton = this.domRoot.find('.open-in-cppinsights');
            this.cppInsightsButton.on('mousedown', () => {
                this.updateOpenInCppInsights();
            });

            this.quickBenchButton = this.domRoot.find('.open-in-quickbench');
            this.quickBenchButton.on('mousedown', () => {
                this.updateOpenInQuickBench();
            });
        }

        this.currentCursorPosition = this.domRoot.find('.currentCursorPosition');
        this.currentCursorPosition.hide();
    }

    handleCtrlS(event: JQuery.KeyDownEvent<HTMLElement, undefined, HTMLElement, HTMLElement>): void {
        event.preventDefault();
        if (this.settings.enableCtrlStree && this.hub.hasTree()) {
            const trees = this.hub.trees;
            // todo: change when multiple trees are used
            // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
            if (trees && trees.length > 0) {
                trees[0].multifileService.includeByEditorId(this.id).then(() => {
                    trees[0].refresh();
                });
            }
        } else {
            if (this.settings.enableCtrlS === 'true') {
                if (this.currentLanguage) loadSave.setMinimalOptions(this.getSource() ?? '', this.currentLanguage);
                // @ts-expect-error: this.id is not a string
                if (!loadSave.onSaveToFile(this.id)) {
                    this.showLoadSaver();
                }
            } else if (this.settings.enableCtrlS === 'false') {
                this.emitShortLinkEvent();
            } else if (this.settings.enableCtrlS === '2') {
                this.runFormatDocumentAction();
            } else if (this.settings.enableCtrlS === '3') {
                this.handleCtrlSDoNothing();
            }
        }
    }

    handleCtrlSDoNothing(): void {
        if (this.nothingCtrlSTimes === undefined) {
            this.nothingCtrlSTimes = 0;
            this.nothingCtrlSSince = Date.now();
        } else {
            if (Date.now() - (this.nothingCtrlSSince ?? 0) > 5000) {
                this.nothingCtrlSTimes = undefined;
            } else if (this.nothingCtrlSTimes === 4) {
                const element = this.domRoot.find('.ctrlSNothing');
                element.show(100);
                setTimeout(function () {
                    element.hide();
                }, 2000);
                this.nothingCtrlSTimes = undefined;
            } else {
                this.nothingCtrlSTimes++;
            }
        }
    }

    updateButtons(): void {
        if (options.thirdPartyIntegrationEnabled) {
            if (this.currentLanguage?.id === 'c++') {
                this.cppInsightsButton.show();
                this.quickBenchButton.show();
            } else {
                this.cppInsightsButton.hide();
                this.quickBenchButton.hide();
            }
        }

        this.addExecutorButton.prop('disabled', !this.currentLanguage?.supportsExecute);
    }

    b64UTFEncode(str: string): string {
        return Buffer.from(
            encodeURIComponent(str).replace(/%([0-9A-F]{2})/g, (match, v) => {
                return String.fromCharCode(parseInt(v, 16));
            })
        ).toString('base64');
    }

    asciiEncodeJsonText(json: string): string {
        return json.replace(/[\u007F-\uFFFF]/g, chr => {
            // json unicode escapes must always be 4 characters long, so pad with leading zeros
            return '\\u' + ('0000' + chr.charCodeAt(0).toString(16)).substring(-4);
        });
    }

    getCompilerStates(): any[] {
        const states: any[] = [];

        for (const compilerIdStr of Object.keys(this.ourCompilers)) {
            const compilerId = parseInt(compilerIdStr);

            const glCompiler: Compiler | undefined = _.find(
                this.container.layoutManager.root.getComponentsByName('compiler'),
                function (c) {
                    return c.id === compilerId;
                }
            );

            if (glCompiler) {
                const state = glCompiler.getCurrentState();
                states.push(state);
            }
        }

        return states;
    }

    updateOpenInCppInsights(): void {
        if (options.thirdPartyIntegrationEnabled) {
            let cppStd = 'cpp2a';

            const compilers = this.getCompilerStates();
            compilers.forEach(compiler => {
                if (compiler.options.indexOf('-std=c++11') !== -1 || compiler.options.indexOf('-std=gnu++11') !== -1) {
                    cppStd = 'cpp11';
                } else if (
                    compiler.options.indexOf('-std=c++14') !== -1 ||
                    compiler.options.indexOf('-std=gnu++14') !== -1
                ) {
                    cppStd = 'cpp14';
                } else if (
                    compiler.options.indexOf('-std=c++17') !== -1 ||
                    compiler.options.indexOf('-std=gnu++17') !== -1
                ) {
                    cppStd = 'cpp17';
                } else if (
                    compiler.options.indexOf('-std=c++2a') !== -1 ||
                    compiler.options.indexOf('-std=gnu++2a') !== -1
                ) {
                    cppStd = 'cpp2a';
                } else if (compiler.options.indexOf('-std=c++98') !== -1) {
                    cppStd = 'cpp98';
                }
            });

            const maxURL = 8177; // apache's default maximum url length
            const maxCode = maxURL - ('/lnk?code=&std=' + cppStd + '&rev=1.0').length;
            let codeData = this.b64UTFEncode(this.getSource() ?? '');
            if (codeData.length > maxCode) {
                codeData = this.b64UTFEncode('/** Source too long to fit in a URL */\n');
            }

            const link = 'https://cppinsights.io/lnk?code=' + codeData + '&std=' + cppStd + '&rev=1.0';

            this.cppInsightsButton.attr('href', link);
        }
    }

    cleanupSemVer(semver: string): string | null {
        if (semver) {
            const semverStr = semver.toString();
            if (semverStr !== '' && semverStr.indexOf('(') === -1) {
                const vercomps = semverStr.split('.');
                return vercomps[0] + '.' + (vercomps[1] ? vercomps[1] : '0');
            }
        }

        return null;
    }

    updateOpenInQuickBench(): void {
        if (options.thirdPartyIntegrationEnabled) {
            type QuickBenchState = {
                text?: string;
                compiler?: string;
                optim?: string;
                cppVersion?: string;
                lib?: string;
            };

            const quickBenchState: QuickBenchState = {
                text: this.getSource(),
            };

            const compilers = this.getCompilerStates();

            compilers.forEach(compiler => {
                let knownCompiler = false;

                const compilerExtInfo = this.hub.compilerService.findCompiler(
                    this.currentLanguage?.id ?? '',
                    compiler.compiler
                );
                const semver = this.cleanupSemVer(compilerExtInfo.semver);
                let groupOrName = compilerExtInfo.baseName || compilerExtInfo.groupName || compilerExtInfo.name;
                if (semver && groupOrName) {
                    groupOrName = groupOrName.toLowerCase();
                    if (groupOrName.indexOf('gcc') !== -1) {
                        quickBenchState.compiler = 'gcc-' + semver;
                        knownCompiler = true;
                    } else if (groupOrName.indexOf('clang') !== -1) {
                        quickBenchState.compiler = 'clang-' + semver;
                        knownCompiler = true;
                    }
                }

                if (knownCompiler) {
                    const match = compiler.options.match(/-(O([0-3sg]|fast))/);
                    if (match !== null) {
                        if (match[2] === 'fast') {
                            quickBenchState.optim = 'F';
                        } else {
                            quickBenchState.optim = match[2].toUpperCase();
                        }
                    }

                    if (
                        compiler.options.indexOf('-std=c++11') !== -1 ||
                        compiler.options.indexOf('-std=gnu++11') !== -1
                    ) {
                        quickBenchState.cppVersion = '11';
                    } else if (
                        compiler.options.indexOf('-std=c++14') !== -1 ||
                        compiler.options.indexOf('-std=gnu++14') !== -1
                    ) {
                        quickBenchState.cppVersion = '14';
                    } else if (
                        compiler.options.indexOf('-std=c++17') !== -1 ||
                        compiler.options.indexOf('-std=gnu++17') !== -1
                    ) {
                        quickBenchState.cppVersion = '17';
                    } else if (
                        compiler.options.indexOf('-std=c++2a') !== -1 ||
                        compiler.options.indexOf('-std=gnu++2a') !== -1
                    ) {
                        quickBenchState.cppVersion = '20';
                    }

                    if (compiler.options.indexOf('-stdlib=libc++') !== -1) {
                        quickBenchState.lib = 'llvm';
                    }
                }
            });

            const link =
                'https://quick-bench.com/#' +
                Buffer.from(this.asciiEncodeJsonText(JSON.stringify(quickBenchState))).toString('base64');
            this.quickBenchButton.attr('href', link);
        }
    }

    changeLanguage(newLang: string): void {
        if (newLang === 'cmake' && languages.cmake) {
            this.selectize.addOption(languages.cmake);
        }
        this.selectize.setValue(newLang);
    }

    clearLinkedLine() {
        this.decorations.linkedCode = [];
        this.updateDecorations();
    }

    tryPanesLinkLine(thisLineNumber: number, column: number, reveal: boolean): void {
        const selectedToken = this.getTokenSpan(thisLineNumber, column);
        for (const compilerId of Object.keys(this.asmByCompiler)) {
            this.eventHub.emit(
                'panesLinkLine',
                Number(compilerId),
                thisLineNumber,
                selectedToken.colBegin,
                selectedToken.colEnd,
                reveal,
                this.getPaneName(),
                this.id
            );
        }
    }

    requestCompilation(): void {
        this.eventHub.emit('requestCompilation', this.id, false);
        if (this.settings.formatOnCompile) {
            this.runFormatDocumentAction();
        }

        this.hub.trees.forEach(tree => {
            if (tree.multifileService.isEditorPartOfProject(this.id)) {
                this.eventHub.emit('requestCompilation', this.id, tree.id);
            }
        });
    }

    override registerEditorActions(): void {
        this.editor.addAction({
            id: 'compile',
            label: 'Compile',
            keybindings: [monaco.KeyMod.CtrlCmd | monaco.KeyCode.Enter],
            keybindingContext: undefined,
            contextMenuGroupId: 'navigation',
            contextMenuOrder: 1.5,
            run: () => {
                // This change request is mostly superfluous
                this.maybeEmitChange();
                this.requestCompilation();
            },
        });

        this.revealJumpStackHasElementsCtxKey = this.editor.createContextKey('hasRevealJumpStackElements', false);

        this.editor.addAction({
            id: 'returnfromreveal',
            label: 'Return from reveal jump',
            keybindings: [monaco.KeyMod.CtrlCmd | monaco.KeyMod.Shift | monaco.KeyCode.Enter],
            contextMenuGroupId: 'navigation',
            contextMenuOrder: 1.4,
            precondition: 'hasRevealJumpStackElements',
            run: () => {
                this.popAndRevealJump();
            },
        });

        this.editor.addAction({
            id: 'toggleCompileOnChange',
            label: 'Toggle compile on change',
            keybindings: [monaco.KeyMod.CtrlCmd | monaco.KeyMod.Shift | monaco.KeyCode.Enter],
            keybindingContext: undefined,
            run: () => {
                this.eventHub.emit('modifySettings', {
                    compileOnChange: !this.settings.compileOnChange,
                });
                this.alertSystem.notify(
                    'Compile on change has been toggled ' + (this.settings.compileOnChange ? 'ON' : 'OFF'),
                    {
                        group: 'togglecompile',
                        alertClass: this.settings.compileOnChange ? 'notification-on' : 'notification-off',
                        dismissTime: 3000,
                    }
                );
            },
        });

        this.editor.addAction({
            id: 'toggleColourisation',
            label: 'Toggle colourisation',
            keybindings: [monaco.KeyMod.CtrlCmd | monaco.KeyMod.Shift | monaco.KeyCode.F1],
            keybindingContext: undefined,
            run: () => {
                this.eventHub.emit('modifySettings', {
                    colouriseAsm: !this.settings.colouriseAsm,
                });
            },
        });

        this.editor.addAction({
            id: 'viewasm',
            label: 'Reveal linked code',
            keybindings: [monaco.KeyMod.CtrlCmd | monaco.KeyCode.F10],
            keybindingContext: undefined,
            contextMenuGroupId: 'navigation',
            contextMenuOrder: 1.5,
            run: ed => {
                const pos = ed.getPosition();
                if (pos != null) {
                    this.tryPanesLinkLine(pos.lineNumber, pos.column, true);
                }
            },
        });

        this.isCpp = this.editor.createContextKey('isCpp', true);
        this.isCpp.set(this.currentLanguage?.id === 'c++');

        this.isClean = this.editor.createContextKey('isClean', true);
        this.isClean.set(this.currentLanguage?.id === 'clean');

        this.editor.addAction({
            id: 'cpprefsearch',
            label: 'Search on Cppreference',
            keybindings: [monaco.KeyMod.CtrlCmd | monaco.KeyCode.F8],
            keybindingContext: undefined,
            contextMenuGroupId: 'help',
            contextMenuOrder: 1.5,
            precondition: 'isCpp',
            run: this.searchOnCppreference.bind(this),
        });

        this.editor.addAction({
            id: 'clooglesearch',
            label: 'Search on Cloogle',
            keybindings: [monaco.KeyMod.CtrlCmd | monaco.KeyCode.F8],
            keybindingContext: undefined,
            contextMenuGroupId: 'help',
            contextMenuOrder: 1.5,
            precondition: 'isClean',
            run: this.searchOnCloogle.bind(this),
        });

        this.editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.F9, () => {
            this.runFormatDocumentAction();
        });

        this.editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyD, () => {
            this.editor.getAction('editor.action.duplicateSelection').run();
        });
    }

    emitShortLinkEvent(): void {
        if (this.settings.enableSharingPopover) {
            this.eventHub.emit('displaySharingPopover');
        } else {
            this.eventHub.emit('copyShortLinkToClip');
        }
    }

    runFormatDocumentAction(): void {
        this.editor.getAction('editor.action.formatDocument').run();
    }

    searchOnCppreference(ed: monaco.editor.ICodeEditor): void {
        const pos = ed.getPosition();
        if (!pos || !ed.getModel()) return;
        const word = ed.getModel()?.getWordAtPosition(pos);
        if (!word || !word.word) return;
        const preferredLanguage = this.getPreferredLanguageTag();
        // This list comes from the footer of the page
        const cpprefLangs = ['ar', 'cs', 'de', 'en', 'es', 'fr', 'it', 'ja', 'ko', 'pl', 'pt', 'ru', 'tr', 'zh'];
        // If navigator.languages is supported, we could be a bit more clever and look for a match there too
        let langTag = 'en';
        if (cpprefLangs.indexOf(preferredLanguage) !== -1) {
            langTag = preferredLanguage;
        }
        const url = 'https://' + langTag + '.cppreference.com/mwiki/index.php?search=' + encodeURIComponent(word.word);
        window.open(url, '_blank', 'noopener');
    }

    searchOnCloogle(ed: monaco.editor.ICodeEditor): void {
        const pos = ed.getPosition();
        if (!pos || !ed.getModel()) return;
        const word = ed.getModel()?.getWordAtPosition(pos);
        if (!word || !word.word) return;
        const url = 'https://cloogle.org/#' + encodeURIComponent(word.word);
        window.open(url, '_blank', 'noopener');
    }

    getPreferredLanguageTag(): string {
        let result = 'en';
        let lang = 'en';
        // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
        if (navigator) {
            // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
            if (navigator.languages && navigator.languages.length) {
                lang = navigator.languages[0];
            } else if (navigator.language) {
                lang = navigator.language;
            }
        }
        // navigator.language[s] is supposed to return strings, but hey, you never know
        if (lang !== result && _.isString(lang)) {
            const primaryLanguageSubtagIdx = lang.indexOf('-');
            result = lang.substring(0, primaryLanguageSubtagIdx).toLowerCase();
        }
        return result;
    }

    doesMatchEditor(otherSource?: string): boolean {
        return otherSource === this.getSource();
    }

    confirmOverwrite(yes: () => void): void {
        this.alertSystem.ask(
            'Changes were made to the code',
            'Changes were made to the code while it was being processed. Overwrite changes?',
            {yes: yes, no: undefined}
        );
    }

    updateSource(newSource: string): void {
        // Create something that looks like an edit operation for the whole text
        const operation = {
            range: this.editor.getModel()?.getFullModelRange(),
            forceMoveMarkers: true,
            text: newSource,
        };
        const nullFn = () => {
            return null;
        };

        const viewState = this.editor.saveViewState();
        // Add an undo stop so we don't go back further than expected
        this.editor.pushUndoStop();
        // Apply de edit. Note that we lose cursor position, but I've not found a better alternative yet
        // @ts-expect-error: See above comment maybe
        this.editor.getModel()?.pushEditOperations(viewState?.cursorState ?? null, [operation], nullFn);
        this.numberUsedLines();

        if (!this.awaitingInitialResults) {
            if (this.selection) {
                /*
                 * this setTimeout is a really crap workaround to fix #2150
                 * the TL;DR; is that we reach this point *before* GL has laid
                 * out the window, so we have no height
                 *
                 * If we revealLinesInCenter at this point the editor "does the right thing"
                 * and scrolls itself all the way to the line we requested.
                 *
                 * Unfortunately the editor thinks it is very small, so the "center"
                 * is the first line, and when the editor does resize eventually things are off.
                 *
                 * The workaround is to just delay things "long enough"
                 *
                 * This is bad and I feel bad.
                 */
                setTimeout(() => {
                    if (this.selection) {
                        this.editor.setSelection(this.selection);
                        this.editor.revealLinesInCenter(this.selection.startLineNumber, this.selection.endLineNumber);
                    }
                }, 500);
            }
            this.awaitingInitialResults = true;
        }
    }

    formatCurrentText(): void {
        const previousSource = this.getSource();
        const lang = this.currentLanguage;

        if (!Object.prototype.hasOwnProperty.call(lang, 'formatter')) {
            return this.alertSystem.notify('This language does not support in-editor formatting', {
                group: 'formatting',
                alertClass: 'notification-error',
            });
        }

        $.ajax({
            type: 'POST',
            url: window.location.origin + this.httpRoot + 'api/format/' + lang?.formatter,
            dataType: 'json', // Expected
            contentType: 'application/json', // Sent
            data: JSON.stringify({
                source: previousSource,
                base: this.settings.formatBase,
            }),
            success: result => {
                if (result.exit === 0) {
                    if (this.doesMatchEditor(previousSource)) {
                        this.updateSource(result.answer);
                    } else {
                        this.confirmOverwrite(this.updateSource.bind(this, result.answer));
                    }
                } else {
                    // Ops, the formatter itself failed!
                    this.alertSystem.notify('We encountered an error formatting your code: ' + result.answer, {
                        group: 'formatting',
                        alertClass: 'notification-error',
                    });
                }
            },
            error: (xhr, e_status, error) => {
                // Hopefully we have not exploded!
                if (xhr.responseText) {
                    try {
                        const res = JSON.parse(xhr.responseText);
                        error = res.answer || error;
                    } catch (e) {
                        // continue regardless of error
                    }
                }
                error = error || 'Unknown error';
                this.alertSystem.notify('We ran into some issues while formatting your code: ' + error, {
                    group: 'formatting',
                    alertClass: 'notification-error',
                });
            },
            cache: true,
        });
    }

    override resize(): void {
        super.resize();

        // Only update the options if needed
        if (this.settings.wordWrap) {
            this.editor.updateOptions({
                wordWrapColumn: this.editor.getLayoutInfo().viewportColumn,
            });
        }
    }

    override onSettingsChange(newSettings: SiteSettings): void {
        const before = this.settings;
        const after = newSettings;
        this.settings = {...newSettings};

        this.editor.updateOptions({
            autoIndent: this.settings.autoIndent ? 'advanced' : 'none',
            autoClosingBrackets: this.settings.autoCloseBrackets ? 'always' : 'never',
            // @ts-ignore useVim is added by the vim plugin, not present in base editor options
            useVim: this.settings.useVim,
            quickSuggestions: this.settings.showQuickSuggestions,
            contextmenu: this.settings.useCustomContextMenu,
            minimap: {
                enabled: this.settings.showMinimap && !options.embedded,
            },
            fontFamily: this.settings.editorsFFont,
            fontLigatures: this.settings.editorsFLigatures,
            wordWrap: this.settings.wordWrap ? 'bounded' : 'off',
            wordWrapColumn: this.editor.getLayoutInfo().viewportColumn, // Ensure the column count is up to date
        });

        if (before.hoverShowSource && !after.hoverShowSource) {
            this.onEditorSetDecoration(this.id, -1, false);
        }

        if (after.useVim && !before.useVim) {
            this.enableVim();
        } else if (!after.useVim && before.useVim) {
            this.disableVim();
        }

        this.editor.getModel()?.updateOptions({
            tabSize: this.settings.tabWidth,
            insertSpaces: this.settings.useSpaces,
        });

        this.numberUsedLines();
    }

    numberUsedLines(): void {
        if (_.any(this.busyCompilers)) return;

        if (!this.settings.colouriseAsm) {
            this.updateColours([]);
            return;
        }

        if (this.hub.hasTree()) {
            return;
        }

        const result: Record<number, boolean> = {};
        // First, note all lines used.
        for (const [compilerId, asm] of Object.entries(this.asmByCompiler)) {
            asm?.forEach(asmLine => {
                let foundInTrees = false;

                for (const [treeId, compilerIds] of Object.entries(this.treeCompilers)) {
                    if (compilerIds && compilerIds[compilerId]) {
                        const tree = this.hub.getTreeById(Number(treeId));
                        if (tree) {
                            const defaultFile = this.defaultFileByCompiler[compilerId];
                            foundInTrees = true;

                            if (asmLine.source && asmLine.source.line > 0) {
                                const sourcefilename = asmLine.source.file ? asmLine.source.file : defaultFile;
                                if (this.id === tree.multifileService.getEditorIdByFilename(sourcefilename)) {
                                    result[asmLine.source.line - 1] = true;
                                }
                            }
                        }
                    }
                }

                if (!foundInTrees) {
                    if (
                        asmLine.source &&
                        (asmLine.source.file === null || asmLine.source.mainsource) &&
                        asmLine.source.line > 0
                    ) {
                        result[asmLine.source.line - 1] = true;
                    }
                }
            });
        }
        // Now assign an ordinal to each used line.
        let ordinal = 0;
        Object.keys(result).forEach(k => {
            result[k] = ordinal++;
        });

        this.updateColours(result);
    }

    updateColours(colours) {
        this.colours = colour.applyColours(this.editor, colours, this.settings.colourScheme, this.colours);
        this.eventHub.emit('colours', this.id, colours, this.settings.colourScheme);
    }

    onCompilerOpen(compilerId: number, editorId: number, treeId: number | boolean): void {
        if (editorId === this.id) {
            // On any compiler open, rebroadcast our state in case they need to know it.
            if (this.waitingForLanguage) {
                const glCompiler = _.find(
                    this.container.layoutManager.root.getComponentsByName('compiler'),
                    function (c) {
                        return c.id === compilerId;
                    }
                );
                if (glCompiler) {
                    const selected = options.compilers.find(compiler => {
                        return compiler.id === glCompiler.originalCompilerId;
                    });
                    if (selected) {
                        this.changeLanguage(selected.lang);
                    }
                }
            }

            if (typeof treeId === 'number' && treeId > 0) {
                if (!this.treeCompilers[treeId]) {
                    this.treeCompilers[treeId] = {};
                }

                // @ts-expect-error: this.treeCompilers[treeId] is never undefined at this point
                this.treeCompilers[treeId][compilerId] = true;
            }
            this.ourCompilers[compilerId] = true;

            if (!treeId) {
                this.maybeEmitChange(true, compilerId);
            }
        }
    }

    onTreeCompilerEditorIncludeChange(treeId: number, editorId: number, compilerId: number): void {
        if (this.id === editorId) {
            this.onCompilerOpen(compilerId, editorId, treeId);
        }
    }

    onTreeCompilerEditorExcludeChange(treeId: number, editorId: number, compilerId: number): void {
        if (this.id === editorId) {
            this.onCompilerClose(compilerId);
        }
    }

    onColoursForEditor(editorId: number, colours: Record<number, number>, scheme: string): void {
        if (this.id === editorId) {
            this.colours = colour.applyColours(this.editor, colours, scheme, this.colours);
        }
    }

    onExecutorOpen(executorId: number, editorId: boolean | number): void {
        if (editorId === this.id) {
            this.maybeEmitChange(true);
            this.ourExecutors[executorId] = true;
        }
    }

    override onCompilerClose(compilerId: number): void {
        /*if (this.treeCompilers[treeId]) {
            delete this.treeCompilers[treeId][compilerId];
        }*/

        if (this.ourCompilers[compilerId]) {
            const model = this.editor.getModel();
            if (model) monaco.editor.setModelMarkers(model, String(compilerId), []);
            delete this.asmByCompiler[compilerId];
            delete this.busyCompilers[compilerId];
            delete this.ourCompilers[compilerId];
            delete this.defaultFileByCompiler[compilerId];
            this.numberUsedLines();
        }
    }

    onExecutorClose(id: number): void {
        if (this.ourExecutors[id]) {
            delete this.ourExecutors[id];
            const model = this.editor.getModel();
            if (model) monaco.editor.setModelMarkers(model, 'Executor ' + id, []);
        }
    }

    onCompiling(compilerId: number): void {
        if (!this.ourCompilers[compilerId]) return;
        this.busyCompilers[compilerId] = true;
    }

    addSource(arr: ResultLine[] | undefined, sourcePane: string): ResultLineWithSourcePane[] {
        if (arr) {
            const newArr: ResultLineWithSourcePane[] = arr.map(element => {
                return {
                    sourcePane: sourcePane,
                    ...element,
                };
            });

            return newArr;
        } else {
            return [];
        }
    }

    getAllOutputAndErrors(
        result: CompilationResult,
        compilerName: string,
        compilerId: number | string
    ): (ResultLine & {sourcePane: string})[] {
        const compilerTitle = compilerName + ' #' + compilerId;
        let all = this.addSource(result.stdout, compilerTitle);

        if (result.buildsteps) {
            _.each(result.buildsteps, step => {
                all = all.concat(this.addSource(step.stdout, compilerTitle));
                all = all.concat(this.addSource(step.stderr, compilerTitle));
            });
        }
        if (result.tools) {
            _.each(result.tools, tool => {
                all = all.concat(this.addSource(tool.stdout, tool.name + ' #' + compilerId));
                all = all.concat(this.addSource(tool.stderr, tool.name + ' #' + compilerId));
            });
        }
        all = all.concat(this.addSource(result.stderr, compilerTitle));

        return all;
    }

    collectOutputWidgets(output: (ResultLine & {sourcePane: string})[]): {
        fixes: monaco.languages.CodeAction[];
        widgets: editor.IMarkerData[];
    } {
        let fixes: monaco.languages.CodeAction[] = [];
        const editorModel = this.editor.getModel();
        const widgets = _.compact(
            output.map(obj => {
                if (!obj.tag) return;

                const trees = this.hub.trees;
                // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
                if (trees && trees.length > 0) {
                    if (obj.tag.file) {
                        if (this.id !== trees[0].multifileService.getEditorIdByFilename(obj.tag.file)) {
                            return;
                        }
                    } else {
                        if (this.id !== trees[0].multifileService.getMainSourceEditorId()) {
                            return;
                        }
                    }
                }

                let colBegin = 0;
                let colEnd = Infinity;
                let lineBegin = obj.tag.line;
                let lineEnd = obj.tag.line;
                if (obj.tag.column) {
                    if (obj.tag.endcolumn) {
                        colBegin = obj.tag.column;
                        colEnd = obj.tag.endcolumn;
                        lineBegin = obj.tag.line;
                        lineEnd = obj.tag.endline;
                    } else {
                        const span = this.getTokenSpan(obj.tag.line ?? 0, obj.tag.column);
                        colBegin = obj.tag.column;
                        colEnd = span.colEnd;
                        if (colEnd === obj.tag.column) colEnd = -1;
                    }
                }
                let link;
                if (obj.tag.link) {
                    link = {
                        value: obj.tag.link.text,
                        target: obj.tag.link.url,
                    };
                }

                const diag: monaco.editor.IMarkerData = {
                    severity: obj.tag.severity,
                    message: obj.tag.text,
                    source: obj.sourcePane,
                    startLineNumber: lineBegin ?? 0,
                    startColumn: colBegin,
                    endLineNumber: lineEnd ?? 0,
                    endColumn: colEnd,
                    code: link,
                };

                if (obj.tag.fixes && editorModel) {
                    fixes = fixes.concat(
                        obj.tag.fixes.map((fs, ind) => {
                            return {
                                title: fs.title,
                                diagnostics: [diag],
                                kind: 'quickfix',
                                edit: {
                                    edits: fs.edits.map(f => {
                                        return {
                                            resource: editorModel.uri,
                                            edit: {
                                                range: new monaco.Range(
                                                    f.line ?? 0,
                                                    f.column ?? 0,
                                                    f.endline ?? 0,
                                                    f.endcolumn ?? 0
                                                ),
                                                text: f.text,
                                            },
                                        };
                                    }),
                                },
                                isPreferred: ind === 0,
                            };
                        })
                    );
                }
                return diag;
            })
        );

        return {
            fixes: fixes,
            widgets: widgets,
        };
    }

    setDecorationTags(widgets: editor.IMarkerData[], ownerId: string): void {
        const editorModel = this.editor.getModel();
        if (editorModel) monaco.editor.setModelMarkers(editorModel, ownerId, widgets);

        this.decorations.tags = _.map(
            widgets,
            function (tag) {
                return {
                    range: new monaco.Range(tag.startLineNumber, tag.startColumn, tag.startLineNumber + 1, 1),
                    options: {
                        isWholeLine: false,
                        inlineClassName: 'error-code',
                    },
                };
            },
            this
        );

        this.updateDecorations();
    }

    setQuickFixes(fixes: monaco.languages.CodeAction[]): void {
        if (fixes.length) {
            const editorModel = this.editor.getModel();
            if (editorModel) {
                quickFixesHandler.registerQuickFixesForCompiler(this.id, editorModel, fixes);
                quickFixesHandler.registerProviderForLanguage(editorModel.getLanguageId());
            }
        } else {
            quickFixesHandler.unregister(this.id);
        }
    }

    override onCompileResult(compilerId: number, compiler: CompilerInfo, result: CompilationResult): void {
        // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
        if (!compiler || !this.ourCompilers[compilerId]) return;

        this.busyCompilers[compilerId] = false;

        const collectedOutput = this.collectOutputWidgets(
            this.getAllOutputAndErrors(result, compiler.name, compilerId)
        );

        this.setDecorationTags(collectedOutput.widgets, String(compilerId));
        this.setQuickFixes(collectedOutput.fixes);

        let asm: ResultLine[] = [];

        // @ts-expect-error: result has no property 'result'
        if (result.result && result.result.asm) {
            // @ts-expect-error: result has no property 'result'
            asm = result.result.asm;
        } else if (result.asm) {
            asm = result.asm;
        }

        if (result.devices && Array.isArray(asm)) {
            asm = asm.concat(
                Object.values(result.devices).flatMap(device => {
                    return device.asm ?? [];
                })
            );
        }

        this.asmByCompiler[compilerId] = asm;

        if (result.inputFilename) {
            this.defaultFileByCompiler[compilerId] = result.inputFilename;
        } else {
            this.defaultFileByCompiler[compilerId] = 'example' + this.currentLanguage?.extensions[0];
        }

        this.numberUsedLines();
    }

    onExecuteResponse(executorId: number, compiler: CompilerInfo, result: CompilationResult): void {
        if (this.ourExecutors[executorId]) {
            let output = this.getAllOutputAndErrors(result, compiler.name, 'Execution ' + executorId);
            if (result.buildResult) {
                output = output.concat(
                    this.getAllOutputAndErrors(result.buildResult, compiler.name, 'Executor ' + executorId)
                );
            }
            this.setDecorationTags(this.collectOutputWidgets(output).widgets, 'Executor ' + executorId);

            this.numberUsedLines();
        }
    }

    onSelectLine(id: number, lineNum: number): void {
        if (Number(id) === this.id) {
            this.editor.setSelection(new monaco.Selection(lineNum - 1, 0, lineNum, 0));
        }
    }

    // Returns a half-segment [a, b) for the token on the line lineNum
    // that spans across the column.
    // a - colStart points to the first character of the token
    // b - colEnd points to the character immediately following the token
    // e.g.: "this->callableMethod ( x, y );"
    //              ^a   ^column  ^b
    getTokenSpan(lineNum: number, column: number): {colBegin: number; colEnd: number} {
        const model = this.editor.getModel();
        if (model && (lineNum < 1 || lineNum > model.getLineCount())) {
            // #3592 Be forgiving towards parsing errors
            return {colBegin: 0, colEnd: 0};
        }

        if (model && lineNum <= model.getLineCount()) {
            const line = model.getLineContent(lineNum);
            if (0 < column && column <= line.length) {
                const tokens = monaco.editor.tokenize(line, model.getLanguageId());
                if (tokens.length > 0) {
                    let lastOffset = 0;
                    let lastWasString = false;
                    for (let i = 0; i < tokens[0].length; ++i) {
                        // Treat all the contiguous string tokens as one,
                        // For example "hello \" world" is treated as one token
                        // instead of 3 "string.cpp", "string.escape.cpp", "string.cpp"
                        if (tokens[0][i].type.startsWith('string')) {
                            if (lastWasString) {
                                continue;
                            }
                            lastWasString = true;
                        } else {
                            lastWasString = false;
                        }
                        const currentOffset = tokens[0][i].offset;
                        if (column <= currentOffset) {
                            return {colBegin: lastOffset + 1, colEnd: currentOffset + 1};
                        } else {
                            lastOffset = currentOffset;
                        }
                    }
                    return {colBegin: lastOffset + 1, colEnd: line.length + 1};
                }
            }
        }
        return {colBegin: column, colEnd: column + 1};
    }

    pushRevealJump(): void {
        const state = this.editor.saveViewState();
        if (state) this.revealJumpStack.push(state);
        this.revealJumpStackHasElementsCtxKey.set(true);
    }

    popAndRevealJump(): void {
        if (this.revealJumpStack.length > 0) {
            const state = this.revealJumpStack.pop();
            if (state) this.editor.restoreViewState(state);
            this.revealJumpStackHasElementsCtxKey.set(this.revealJumpStack.length > 0);
        }
    }

    onEditorLinkLine(editorId: number, lineNum: number, columnBegin: number, columnEnd: number, reveal: boolean): void {
        if (Number(editorId) === this.id) {
            if (reveal && lineNum) {
                this.pushRevealJump();
                this.hub.activateTabForContainer(this.container);
                this.editor.revealLineInCenter(lineNum);
            }
            this.decorations.linkedCode = [];
            if (lineNum && lineNum !== -1) {
                this.decorations.linkedCode.push({
                    range: new monaco.Range(lineNum, 1, lineNum, 1),
                    options: {
                        isWholeLine: true,
                        linesDecorationsClassName: 'linked-code-decoration-margin',
                        className: 'linked-code-decoration-line',
                    },
                });
            }

            if (lineNum > 0 && columnBegin !== -1) {
                const lastTokenSpan = this.getTokenSpan(lineNum, columnEnd);
                this.decorations.linkedCode.push({
                    range: new monaco.Range(lineNum, columnBegin, lineNum, lastTokenSpan.colEnd),
                    options: {
                        isWholeLine: false,
                        inlineClassName: 'linked-code-decoration-column',
                    },
                });
            }
            if (!this.settings.indefiniteLineHighlight) {
                if (this.fadeTimeoutId !== null) {
                    clearTimeout(this.fadeTimeoutId);
                }
                this.fadeTimeoutId = setTimeout(() => {
                    this.clearLinkedLine();
                    this.fadeTimeoutId = null;
                }, 5000);
            }
            this.updateDecorations();
        }
    }

    onEditorSetDecoration(id: number, lineNum: number, reveal: boolean, column?: number): void {
        if (Number(id) === this.id) {
            if (reveal && lineNum) {
                this.pushRevealJump();
                this.editor.revealLineInCenter(lineNum);
                this.editor.focus();
                this.editor.setPosition({column: column || 0, lineNumber: lineNum});
            }
            this.decorations.linkedCode = [];
            if (lineNum && lineNum !== -1) {
                this.decorations.linkedCode.push({
                    range: new monaco.Range(lineNum, 1, lineNum, 1),
                    options: {
                        isWholeLine: true,
                        linesDecorationsClassName: 'linked-code-decoration-margin',
                        inlineClassName: 'linked-code-decoration-inline',
                    },
                });
            }
            this.updateDecorations();
        }
    }

    onEditorDisplayFlow(id: number, flow: MessageWithLocation[]): void {
        if (Number(id) === this.id) {
            if (this.decorations.flows && this.decorations.flows.length) {
                this.decorations.flows = [];
            } else {
                this.decorations.flows = flow.map((ri, ind) => {
                    return {
                        range: new monaco.Range(
                            ri.line ?? 0,
                            ri.column ?? 0,
                            (ri.endline || ri.line) ?? 0,
                            (ri.endcolumn || ri.column) ?? 0
                        ),
                        options: {
                            before: {
                                content: ' ' + (ind + 1).toString() + ' ',
                                inlineClassName: 'flow-decoration',
                                cursorStops: monaco.editor.InjectedTextCursorStops.None,
                            },
                            inlineClassName: 'flow-highlight',
                            isWholeLine: false,
                            hoverMessage: {value: ri.text},
                        },
                    };
                });
            }
            this.updateDecorations();
        }
    }

    updateDecorations(): void {
        this.prevDecorations = this.editor.deltaDecorations(
            this.prevDecorations,
            _.compact(_.flatten(_.values(this.decorations)))
        );
    }

    onConformanceViewOpen(editorId: number): void {
        if (editorId === this.id) {
            this.conformanceViewerButton.attr('disabled', 1);
        }
    }

    onConformanceViewClose(editorId: number): void {
        if (editorId === this.id) {
            this.conformanceViewerButton.attr('disabled', null);
        }
    }

    showLoadSaver(): void {
        this.loadSaveButton.trigger('click');
    }

    initLoadSaver(): void {
        this.loadSaveButton.off('click').on('click', () => {
            if (this.currentLanguage) {
                loadSave.run(
                    (text, filename) => {
                        this.setSource(text);
                        this.setFilename(filename);
                        this.updateState();
                        this.maybeEmitChange(true);
                        this.requestCompilation();
                    },
                    this.getSource(),
                    this.currentLanguage
                );
            }
        });
    }

    onLanguageChange(newLangId: string, firstTime: boolean): void {
        if (newLangId in languages) {
            if (firstTime || newLangId !== this.currentLanguage?.id) {
                const oldLangId = this.currentLanguage?.id;
                this.currentLanguage = languages[newLangId];
                if (!this.waitingForLanguage && !this.settings.keepSourcesOnLangChange && newLangId !== 'cmake') {
                    this.editorSourceByLang[oldLangId ?? ''] = this.getSource();
                    this.updateEditorCode();
                }
                this.initLoadSaver();
                const editorModel = this.editor.getModel();
                if (editorModel && this.currentLanguage)
                    monaco.editor.setModelLanguage(editorModel, this.currentLanguage.monaco);
                this.isCpp.set(this.currentLanguage?.id === 'c++');
                this.isClean.set(this.currentLanguage?.id === 'clean');
                this.updateLanguageTooltip();
                this.updateTitle();
                this.updateState();
                // Broadcast the change to other panels
                this.eventHub.emit('languageChange', this.id, newLangId);
                this.decorations = {};
                if (!firstTime) {
                    this.maybeEmitChange(true);
                    this.requestCompilation();

                    ga.proxy('send', {
                        hitType: 'event',
                        eventCategory: 'LanguageChange',
                        eventAction: newLangId,
                    });
                }
            }
            this.waitingForLanguage = false;
        }
    }

    override getDefaultPaneName(): string {
        return 'Editor';
    }

    override getPaneName(): string {
        if (this.filename) {
            return this.filename;
        } else {
            return this.currentLanguage?.name + ' source #' + this.id;
        }
    }

    setFilename(name: string): void {
        this.filename = name;
        this.updateTitle();
        this.updateState();
    }

    override updateTitle(): void {
        const name = this.getPaneName();
        const customName = this.paneName ? this.paneName : name;
        if (name.endsWith('CMakeLists.txt')) {
            this.changeLanguage('cmake');
        }
        this.container.setTitle(_.escape(customName));
    }

    // Called every time we change language, so we get the relevant code
    updateEditorCode(): void {
        this.setSource(
            this.editorSourceByLang[this.currentLanguage?.id ?? ''] ||
                languages[this.currentLanguage?.id ?? '']?.example
        );
    }

    override close(): void {
        this.eventHub.unsubscribe();
        this.eventHub.emit('editorClose', this.id);
        this.editor.dispose();
        this.hub.removeEditor(this.id);
    }

    getSelectizeRenderHtml(
        data: LanguageSelectData,
        escape: typeof escape_html,
        width: number,
        height: number
    ): string {
        let result =
            '<div class="d-flex" style="align-items: center">' +
            '<div class="mr-1 d-flex" style="align-items: center">' +
            '<img src="' +
            (data.logoData ? data.logoData : '') +
            '" class="' +
            (data.logoDataDark ? 'theme-light-only' : '') +
            '" width="' +
            width +
            '" style="max-height: ' +
            height +
            'px"/>';
        if (data.logoDataDark) {
            result +=
                '<img src="' +
                data.logoDataDark +
                '" class="theme-dark-only" width="' +
                width +
                '" style="max-height: ' +
                height +
                'px"/>';
        }

        result += '</div><div';
        if (data.tooltip) {
            result += ' title="' + data.tooltip + '"';
        }
        result += '>' + escape(data.name) + '</div></div>';
        return result;
    }

    renderSelectizeOption(data: LanguageSelectData, escape: typeof escape_html) {
        return this.getSelectizeRenderHtml(data, escape, 23, 23);
    }

    renderSelectizeItem(data: LanguageSelectData, escape: typeof escape_html) {
        return this.getSelectizeRenderHtml(data, escape, 20, 20);
    }

    onCompiler(compilerId: number, compiler: unknown, options: string, editorId: number, treeId: number): void {}

    updateLanguageTooltip() {
        this.languageInfoButton.popover('dispose');
        if (this.currentLanguage?.tooltip) {
            this.languageInfoButton.popover({
                title: 'More info about this language',
                content: this.currentLanguage.tooltip,
                container: 'body',
                trigger: 'focus',
                placement: 'left',
            });
            this.languageInfoButton.show();
            this.languageInfoButton.prop('title', this.currentLanguage.tooltip);
        } else {
            this.languageInfoButton.hide();
        }
    }
}
