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

import $ from 'jquery';
import {MultifileFile, MultifileService, MultifileServiceState} from '../multifile-service.js';
import {LineColouring} from '../line-colouring.js';
import * as utils from '../utils.js';
import {Settings, SiteSettings} from '../settings.js';
import {PaneRenaming} from '../widgets/pane-renaming.js';
import {Hub} from '../hub.js';
import {EventHub} from '../event-hub.js';
import {Alert} from '../widgets/alert.js';
import * as Components from '../components.js';
import {ga} from '../analytics.js';
import TomSelect from 'tom-select';
import {Toggles} from '../widgets/toggles.js';
import {options} from '../options.js';
import {saveAs} from 'file-saver';
import {Container} from 'golden-layout';
import _ from 'underscore';
import {assert, unwrap, unwrapString} from '../assert.js';

const languages = options.languages;

export interface TreeState extends MultifileServiceState {
    id: number;
    cmakeArgs: string;
    customOutputFilename: string;
}

export class Tree {
    public readonly id: number;
    private container: Container;
    private domRoot: JQuery;
    private readonly hub: Hub;
    private eventHub: EventHub;
    private readonly settings: SiteSettings;
    private httpRoot: string;
    private readonly alertSystem: Alert;
    private root: JQuery;
    private rowTemplate: JQuery;
    private namedItems: JQuery;
    private unnamedItems: JQuery;
    private langKeys: string[];
    private cmakeArgsInput: JQuery;
    private customOutputFilenameInput: JQuery;
    public multifileService: MultifileService;
    private lineColouring: LineColouring;
    private readonly ourCompilers: Record<number, boolean>;
    private readonly busyCompilers: Record<number, boolean>;
    private readonly asmByCompiler: Record<number, any>;
    private selectize: TomSelect;
    private languageBtn: JQuery;
    private toggleCMakeButton: Toggles;
    private debouncedEmitChange: () => void = () => {};
    private hideable: JQuery;
    private readonly topBar: JQuery;
    private paneName: string;
    private paneRenaming: PaneRenaming;

    constructor(hub: Hub, container: Container, state: TreeState) {
        this.id = state.id || hub.nextTreeId();
        this.container = container;
        this.domRoot = container.getElement();
        this.domRoot.html($('#tree').html());
        this.hub = hub;
        this.eventHub = hub.createEventHub();
        this.settings = Settings.getStoredSettings();

        this.httpRoot = window.httpRoot;

        this.alertSystem = new Alert();
        this.alertSystem.prefixMessage = 'Tree #' + this.id;

        this.root = this.domRoot.find('.tree');
        this.rowTemplate = $('#tree-editor-tpl');
        this.namedItems = this.domRoot.find('.named-editors');
        this.unnamedItems = this.domRoot.find('.unnamed-editors');
        this.hideable = this.domRoot.find('.hideable');
        this.topBar = this.domRoot.find('.top-bar.mainbar');

        this.langKeys = Object.keys(languages);

        this.cmakeArgsInput = this.domRoot.find('.cmake-arguments');
        this.customOutputFilenameInput = this.domRoot.find('.cmake-customOutputFilename');

        const usableLanguages = Object.values(languages).filter(language => {
            return hub.compilerService.getCompilersForLang(language.id);
        });

        if (!(state.compilerLanguageId as any)) {
            state.compilerLanguageId = this.settings.defaultLanguage ?? 'c++';
        }

        this.multifileService = new MultifileService(this.hub, this.alertSystem, state);
        this.lineColouring = new LineColouring(this.multifileService);
        this.ourCompilers = {};
        this.busyCompilers = {};
        this.asmByCompiler = {};

        this.paneRenaming = new PaneRenaming(this, state);

        this.initInputs(state);
        this.initButtons(state);
        this.initCallbacks();
        this.onSettingsChange(this.settings);

        assert(this.languageBtn[0] instanceof HTMLSelectElement);
        this.selectize = new TomSelect(this.languageBtn[0], {
            sortField: 'name',
            valueField: 'id',
            labelField: 'name',
            searchField: ['name'],
            options: usableLanguages,
            items: [this.multifileService.getLanguageId()],
            dropdownParent: 'body',
            plugins: ['input_autogrow'],
            onChange: (val: any) => this.onLanguageChange(val),
        });

        this.onLanguageChange(this.multifileService.getLanguageId());

        ga.proxy('send', {
            hitType: 'event',
            eventCategory: 'OpenViewPane',
            eventAction: 'Tree',
        });

        this.refresh();
        this.eventHub.emit('findEditors');
    }

    private initInputs(state: TreeState) {
        if (state.cmakeArgs) {
            this.cmakeArgsInput.val(state.cmakeArgs);
        }

        if (state.customOutputFilename) {
            this.customOutputFilenameInput.val(state.customOutputFilename);
        }
    }

    private getCmakeArgs(): string {
        return unwrapString(this.cmakeArgsInput.val());
    }

    private getCustomOutputFilename(): string {
        return _.escape(unwrapString(this.customOutputFilenameInput.val()));
    }

    public currentState(): TreeState {
        const state = {
            id: this.id,
            cmakeArgs: this.getCmakeArgs(),
            customOutputFilename: this.getCustomOutputFilename(),
            ...this.multifileService.getState(),
        };
        this.paneRenaming.addState(state);
        return state;
    }

    private updateState() {
        const state = this.currentState();
        this.container.setState(state);

        this.updateButtons(state);
    }

    private initCallbacks() {
        this.container.on('resize', this.resize, this);
        this.container.on('shown', this.resize, this);
        this.container.on('open', () => {
            this.eventHub.emit('treeOpen', this.id);
        });
        this.container.on('destroy', this.close, this);

        this.paneRenaming.on('renamePane', this.updateState.bind(this));

        this.eventHub.on('editorOpen', this.onEditorOpen, this);
        this.eventHub.on('editorClose', this.onEditorClose, this);
        this.eventHub.on('compilerOpen', this.onCompilerOpen, this);
        this.eventHub.on('compilerClose', this.onCompilerClose, this);

        this.eventHub.on('compileResult', this.onCompileResponse, this);

        this.toggleCMakeButton.on('change', this.onToggleCMakeChange.bind(this));

        this.cmakeArgsInput.on('change', this.updateCMakeArgs.bind(this));
        this.customOutputFilenameInput.on('change', this.updateCustomOutputFilename.bind(this));
    }

    private updateCMakeArgs() {
        this.updateState();

        this.debouncedEmitChange();
    }

    private updateCustomOutputFilename() {
        this.updateState();

        this.debouncedEmitChange();
    }

    private onToggleCMakeChange() {
        const isOn = this.toggleCMakeButton.get().isCMakeProject;
        this.multifileService.setAsCMakeProject(isOn);

        this.domRoot.find('.cmake-project').prop('title', '[' + (isOn ? 'ON' : 'OFF') + '] CMake project');
        this.updateState();
    }

    private onLanguageChange(newLangId: string) {
        if (newLangId in languages) {
            this.multifileService.setLanguageId(newLangId);
            this.eventHub.emit('languageChange', false, newLangId, this.id);
        }

        this.toggleCMakeButton.enableToggle('isCMakeProject', this.multifileService.isCompatibleWithCMake());

        this.refresh();
    }

    private sendCompilerChangesToEditor(compilerId: number) {
        this.multifileService.forEachOpenFile((file: MultifileFile) => {
            if (file.isIncluded) {
                this.eventHub.emit('treeCompilerEditorIncludeChange', this.id, file.editorId, compilerId);
            } else {
                this.eventHub.emit('treeCompilerEditorExcludeChange', this.id, file.editorId, compilerId);
            }
        });

        this.eventHub.emit('resendCompilation', compilerId);
    }

    private sendCompileRequests() {
        this.eventHub.emit('requestCompilation', false, this.id);
    }

    private sendChangesToAllEditors() {
        for (const compilerId in this.ourCompilers) {
            this.sendCompilerChangesToEditor(parseInt(compilerId));
        }
    }

    private onCompilerOpen(compilerId: number, unused, treeId: number | boolean) {
        if (treeId === this.id) {
            this.ourCompilers[compilerId] = true;
            this.sendCompilerChangesToEditor(compilerId);
        }
    }

    private onCompilerClose(compilerId: number, treeId: number | boolean) {
        if (treeId === this.id) {
            delete this.ourCompilers[compilerId];
        }
    }

    private onEditorOpen(editorId: number) {
        const file = this.multifileService.getFileByEditorId(editorId);
        if (file) return;

        this.multifileService.addFileForEditorId(editorId);
        this.refresh();
        this.sendChangesToAllEditors();
    }

    private onEditorClose(editorId: number) {
        const file = this.multifileService.getFileByEditorId(editorId);

        if (file) {
            file.isOpen = false;
            const editor = this.hub.getEditorById(editorId);
            file.langId = editor?.currentLanguage?.id ?? '';
            file.content = editor?.getSource() ?? '';
            file.editorId = -1;
        }

        this.refresh();
    }

    private removeFile(fileId: number) {
        const file = this.multifileService.removeFileByFileId(fileId);
        if (file) {
            if (file.isOpen) {
                const editor = this.hub.getEditorById(file.editorId);
                if (editor) {
                    editor.container.close();
                }
            }
        }

        this.refresh();
    }

    private removeFileByFilename(filename: string) {
        const file = this.multifileService.removeFileByFilename(filename);
        if (file) {
            if (file.isOpen) {
                const editor = this.hub.getEditorById(file.editorId);
                if (editor) {
                    editor.container.close();
                }
            }
        }

        this.refresh();
    }

    private addRowToTreelist(file: MultifileFile) {
        const item = $(this.rowTemplate.children()[0].cloneNode(true));
        const stageButton = item.find('.stage-file');
        const unstageButton = item.find('.unstage-file');
        const renameButton = item.find('.rename-file');
        const deleteButton = item.find('.delete-file');

        item.data('fileId', file.fileId);
        if (file.filename) {
            item.find('.filename').text(file.filename);
        } else if (file.editorId > 0) {
            const editor = this.hub.getEditorById(file.editorId);
            if (editor) {
                item.find('.filename').text(editor.getPaneName());
            } else {
                // wait for editor to appear first
                return;
            }
        } else {
            item.find('.filename').text('Unknown file');
        }

        item.on('click', e => {
            const fileId = $(e.currentTarget).data('fileId');
            this.editFile(fileId);
        });

        renameButton.on('click', async e => {
            const fileId = $(e.currentTarget).parent('li').data('fileId');
            await this.multifileService.renameFile(fileId);
            this.refresh();
        });

        deleteButton.on('click', e => {
            const fileId = $(e.currentTarget).parent('li').data('fileId');
            const file = this.multifileService.getFileByFileId(fileId);
            if (file) {
                this.alertSystem.ask(
                    'Delete file',
                    `Are you sure you want to delete ${file.filename ? _.escape(file.filename) : 'this file'}?`,
                    {
                        yes: () => {
                            this.removeFile(fileId);
                        },
                        yesClass: 'btn-danger',
                        yesHtml: 'Delete',
                        noClass: 'btn-primary',
                        noHtml: 'Cancel',
                    },
                );
            }
        });

        stageButton.on('click', async e => {
            const fileId = $(e.currentTarget).parent('li').data('fileId');
            await this.moveToInclude(fileId);
        });
        unstageButton.on('click', async e => {
            const fileId = $(e.currentTarget).parent('li').data('fileId');
            await this.moveToExclude(fileId);
        });
        stageButton.toggle(!file.isIncluded);
        unstageButton.toggle(file.isIncluded);
        // @ts-ignore TODO type mismatch
        (file.isIncluded ? this.namedItems : this.unnamedItems).append(item);
    }

    refresh() {
        this.updateState();

        this.namedItems.html('');
        this.unnamedItems.html('');

        this.multifileService.forEachFile((file: MultifileFile) => this.addRowToTreelist(file));
    }

    private editFile(fileId: number) {
        const file = this.multifileService.getFileByFileId(fileId);
        if (file) {
            if (!file.isOpen) {
                const dragConfig = this.getConfigForNewEditor(file);
                file.isOpen = true;

                this.hub.addInEditorStackIfPossible(dragConfig);
            } else {
                const editor = this.hub.getEditorById(file.editorId);
                this.hub.activateTabForContainer(editor?.container);
            }

            this.sendChangesToAllEditors();
        }
    }

    private async moveToInclude(fileId: number) {
        await this.multifileService.includeByFileId(fileId);

        this.refresh();
        this.sendChangesToAllEditors();
    }

    private async moveToExclude(fileId: number) {
        await this.multifileService.excludeByFileId(fileId);
        this.refresh();
        this.sendChangesToAllEditors();
    }

    private bindClickToOpenPane(dragSource, dragConfig) {
        (this.container.layoutManager.createDragSource(dragSource, dragConfig.bind(this)) as any)._dragListener.on(
            'dragStart',
            () => {
                this.domRoot.find('.add-pane').dropdown('toggle');
            },
        );

        dragSource.on('click', () => {
            this.hub.addInEditorStackIfPossible(dragConfig.bind(this));
        });
    }

    private getConfigForNewCompiler() {
        return Components.getCompilerForTree(this.id, this.currentState().compilerLanguageId);
    }

    private getConfigForNewExecutor() {
        return Components.getExecutorForTree(this.id, this.currentState().compilerLanguageId);
    }

    private getConfigForNewEditor(file: MultifileFile | undefined) {
        let editor;
        const editorId = this.hub.nextEditorId();

        if (file) {
            file.editorId = editorId;
            editor = Components.getEditor(file.langId, editorId);

            editor.componentState.source = file.content;
            if (file.filename) {
                editor.componentState.filename = file.filename;
            }
        } else {
            editor = Components.getEditor(this.multifileService.getLanguageId(), editorId);
        }

        return editor;
    }

    private static getFormattedDateTime() {
        const d = new Date();
        const t = x => x.slice(-2);
        // Hopefully some day we can use the temporal api to make this less of a pain
        return (
            `${d.getFullYear()} ${t('0' + (d.getMonth() + 1))} ${t('0' + d.getDate())}` +
            `${t('0' + d.getHours())} ${t('0' + d.getMinutes())} ${t('0' + d.getSeconds())}`
        );
    }

    private static triggerSaveAs(blob) {
        const dt = Tree.getFormattedDateTime();
        saveAs(blob, `project-${dt}.zip`);
    }

    private initButtons(state: TreeState) {
        const addCompilerButton = this.domRoot.find('.add-compiler');
        const addExecutorButton = this.domRoot.find('.add-executor');
        const addEditorButton = this.domRoot.find('.add-editor');
        const saveProjectButton = this.domRoot.find('.save-project-to-file');

        saveProjectButton.on('click', async () => {
            await this.multifileService.saveProjectToZipfile(Tree.triggerSaveAs.bind(this));
        });

        const loadProjectFromFile = this.domRoot.find('.load-project-from-file') as JQuery<HTMLInputElement>;
        loadProjectFromFile.on('change', async e => {
            const files = e.target.files;
            if (files && files.length > 0) {
                await this.openZipFile(files[0]);
            }

            loadProjectFromFile.val('');
        });

        this.bindClickToOpenPane(addCompilerButton, this.getConfigForNewCompiler);
        this.bindClickToOpenPane(addExecutorButton, this.getConfigForNewExecutor);
        this.bindClickToOpenPane(addEditorButton, this.getConfigForNewEditor);

        this.languageBtn = this.domRoot.find('.change-language');
        if (!(this.languageBtn[0] instanceof HTMLSelectElement)) {
            throw new Error('.language-button is not an HTMLSelectElement');
        }
        if (this.langKeys.length <= 1) {
            this.languageBtn.prop('disabled', true);
        }

        this.toggleCMakeButton = new Toggles(
            this.domRoot.find('.options'),
            state as unknown as Record<string, boolean>,
        );

        let drophereHideTimeout;
        this.root.on('dragover', ev => {
            ev.preventDefault();

            if (drophereHideTimeout) clearTimeout(drophereHideTimeout);

            const drophere = this.root.find('.drophere');
            drophere.show();
        });

        this.root.on('dragleave', () => {
            const drophere = this.root.find('.drophere');
            drophereHideTimeout = setTimeout(() => {
                drophere.hide();
            }, 1000);
        });

        this.root.on('drop', async (ev: any) => {
            ev.preventDefault();

            const drophere = this.root.find('.drophere');
            drophere.hide();

            const dataTransfer = ev.originalEvent.dataTransfer;
            if (dataTransfer.items) {
                [...dataTransfer.items].forEach(async (item, i) => {
                    if (item.kind === 'file') {
                        const file = item.getAsFile();
                        if (file.name.endsWith('.zip')) {
                            this.openZipFile(file);
                        } else {
                            await this.addSingleFile(file);
                        }
                    }
                });
            } else {
                [...dataTransfer.files].forEach(async (file, i) => {
                    if (file.name.endsWith('.zip')) {
                        this.openZipFile(file);
                    } else {
                        await this.addSingleFile(file);
                    }
                });
            }
        });
    }

    private async openZipFile(htmlfile) {
        if (!htmlfile.name.toLowerCase().endsWith('.zip')) {
            this.alertSystem.alert('Load project file', 'Projects can only be loaded from .zip files', {isError: true});
            return;
        }

        this.multifileService.forEachFile((file: MultifileFile) => {
            this.removeFile(file.fileId);
        });

        await this.multifileService.loadProjectFromFile(htmlfile, (file: MultifileFile) => {
            this.refresh();
            if (file.filename === 'CMakeLists.txt') {
                // todo: find a way to toggle on CMake checkbox...
                this.editFile(file.fileId);
            }
        });
    }

    private async askForOverwriteAndDo(filename): Promise<void> {
        return new Promise((resolve, reject) => {
            if (this.multifileService.fileExists(filename)) {
                this.alertSystem.ask('Overwrite file', `${_.escape(filename)} already exists, overwrite this file?`, {
                    yes: () => {
                        this.removeFileByFilename(filename);
                        resolve();
                    },
                    no: () => {
                        reject();
                    },
                    onClose: () => {
                        reject();
                    },
                    yesClass: 'btn-danger',
                    yesHtml: 'Yes',
                    noClass: 'btn-primary',
                    noHtml: 'No',
                });
            } else {
                resolve();
            }
        });
    }

    private async addSingleFile(htmlfile): Promise<void> {
        try {
            await this.askForOverwriteAndDo(htmlfile.name);

            return new Promise(resolve => {
                const fr = new FileReader();
                fr.onload = () => {
                    this.multifileService.addNewTextFile(htmlfile.name, fr.result?.toString() || '');
                    this.refresh();
                    resolve();
                };
                fr.readAsText(htmlfile);
            });
        } catch {
            // expected when user says no
        }
    }

    private numberUsedLines() {
        if (_.any(this.busyCompilers)) return;

        if (!this.settings.colouriseAsm) {
            this.updateColoursNone();
            return;
        }

        this.lineColouring.clear();

        for (const [compilerId, asm] of Object.entries(this.asmByCompiler)) {
            if (asm) {
                this.lineColouring.addFromAssembly(parseInt(compilerId), asm);
            }
        }

        this.lineColouring.calculate();

        this.updateColours();
    }

    private updateColours() {
        for (const compilerId in this.ourCompilers) {
            const id: number = parseInt(compilerId);
            this.eventHub.emit(
                'coloursForCompiler',
                id,
                this.lineColouring.getColoursForCompiler(id),
                this.settings.colourScheme,
            );
        }

        this.multifileService.forEachOpenFile((file: MultifileFile) => {
            this.eventHub.emit(
                'coloursForEditor',
                file.editorId,
                this.lineColouring.getColoursForEditor(file.editorId),
                this.settings.colourScheme,
            );
        });
    }

    private updateColoursNone() {
        for (const compilerId in this.ourCompilers) {
            this.eventHub.emit('coloursForCompiler', parseInt(compilerId), {}, this.settings.colourScheme);
        }

        this.multifileService.forEachOpenFile((file: MultifileFile) => {
            this.eventHub.emit('coloursForEditor', file.editorId, {}, this.settings.colourScheme);
        });
    }

    private onCompileResponse(compilerId: number, compiler, result) {
        if (!this.ourCompilers[compilerId]) return;

        this.busyCompilers[compilerId] = false;

        // todo: parse errors and warnings and relate them to lines in the code
        // note: requires info about the filename, do we currently have that?

        // eslint-disable-next-line max-len
        // {"text":"/tmp/compiler-explorer-compiler2021428-7126-95g4xc.zfo8p/example.cpp:4:21: error: expected ‘;’ before ‘}’ token"}

        if (result.result && result.result.asm) {
            this.asmByCompiler[compilerId] = result.result.asm;
        } else {
            this.asmByCompiler[compilerId] = result.asm;
        }

        this.numberUsedLines();
    }

    private updateButtons(state: TreeState) {
        if (state.isCMakeProject) {
            this.cmakeArgsInput.parent().removeClass('d-none');
            this.customOutputFilenameInput.parent().removeClass('d-none');
        } else {
            this.cmakeArgsInput.parent().addClass('d-none');
            this.customOutputFilenameInput.parent().addClass('d-none');
        }
    }

    private resize() {
        utils.updateAndCalcTopBarHeight(this.domRoot, this.topBar, this.hideable);

        const mainbarHeight = unwrap(this.topBar.outerHeight(true));
        const argsHeight = unwrap(this.domRoot.find('.panel-args').outerHeight(true));
        const outputfileHeight = unwrap(this.domRoot.find('.panel-outputfile').outerHeight(true));
        const innerHeight = unwrap(this.domRoot.innerHeight());

        this.root.height(innerHeight - mainbarHeight - argsHeight - outputfileHeight);
    }

    private onSettingsChange(newSettings) {
        this.debouncedEmitChange = _.debounce(() => {
            this.sendCompileRequests();
        }, newSettings.delayAfterChange);
    }

    private getPaneName() {
        return `Tree #${this.id}`;
    }

    private updateTitle() {
        const name = this.paneName ? this.paneName : this.getPaneName();
        this.container.setTitle(_.escape(name));
    }

    private close() {
        this.eventHub.unsubscribe();
        this.eventHub.emit('treeClose', this.id);
        this.hub.removeTree(this.id);
        $('#add-tree').prop('disabled', false);
    }
}
