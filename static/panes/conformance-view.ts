// Copyright (c) 2017, Compiler Explorer Authors
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

import {options} from '../options';
import _ from 'underscore';
import $ from 'jquery';
import {ga} from '../analytics';
import * as Components from '../components';
import {CompilerLibs, LibsWidget} from '../widgets/libs-widget';
import {CompilerPicker} from '../widgets/compiler-picker';
import * as utils from '../utils';
import * as LibUtils from '../lib-utils';
import {PaneRenaming} from '../widgets/pane-renaming';
import {CompilerService} from '../compiler-service';
import {Pane} from './pane';
import {Hub} from '../hub';
import {Container} from 'golden-layout';
import {PaneState} from './pane.interfaces';
import {ConformanceViewState} from './conformance-view.interfaces';
import {Library, LibraryVersion} from '../options.interfaces';
import {CompilerInfo} from '../../types/compiler.interfaces';
import {CompilationResult} from '../../types/compilation/compilation.interfaces';
import {Lib} from '../widgets/libs-widget.interfaces';
import {SourceAndFiles} from '../download-service';

type ConformanceStatus = {
    allowCompile: boolean;
    allowAdd: boolean;
};

type CompilerEntry = {
    parent: JQuery<HTMLElement>;
    picker: CompilerPicker | null;
    optionsField: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]> | null;
    statusIcon: JQuery<HTMLElement> | null;
    prependOptions: JQuery<HTMLElement> | null;
};

type CompileChildLibraries = {
    id: string;
    version: string;
};

type AddCompilerPickerConfig = {
    compilerId: string;
    options: string | number | string[];
};

export class Conformance extends Pane<ConformanceViewState> {
    private libsWidget: LibsWidget;
    private compilerService: CompilerService;
    private readonly maxCompilations: number;
    private langId: string;
    private source: string;
    private sourceNeedsExpanding: boolean;
    private compilerPickers: CompilerEntry[];
    private expandedSourceAndFiles: SourceAndFiles | null;
    private currentLibs: Lib[];
    private status: ConformanceStatus;
    private readonly stateByLang: Record<string, ConformanceViewState>;
    private libsButton: JQuery<HTMLElement>;
    private conformanceContentRoot: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private selectorList: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private addCompilerButton: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private selectorTemplate: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private lastState?: ConformanceViewState;

    constructor(hub: Hub, container: Container, state: PaneState & ConformanceViewState) {
        super(hub, container, state);
        this.compilerService = hub.compilerService;
        this.maxCompilations = options.cvCompilerCountMax;
        this.langId = state.langId || _.keys(options.languages)[0];
        this.source = state.source ?? '';
        this.sourceNeedsExpanding = true;
        this.expandedSourceAndFiles = null;

        this.status = {
            allowCompile: false,
            allowAdd: true,
        };
        this.stateByLang = {};

        this.paneRenaming = new PaneRenaming(this, state);

        this.initButtons();
        this.initCallbacks();
        this.initFromState(state);
        this.initLibraries(state);
        this.handleToolbarUI();

        // Dismiss the popover on escape.
        $(document).on('keyup.editable', e => {
            if (e.which === 27) {
                this.libsButton.popover('hide');
            }
        });

        // Dismiss on any click that isn't either in the opening element, inside
        // the popover or on any alert
        $(document).on('click', e => {
            const elem = this.libsButton;
            const target = $(e.target);
            if (
                !target.is(elem) &&
                elem.has(target as unknown as Element).length === 0 &&
                target.closest('.popover').length === 0
            ) {
                elem.popover('hide');
            }
        });
    }

    getInitialHTML(): string {
        return $('#conformance').html();
    }

    registerOpeningAnalyticsEvent(): void {
        ga.proxy('send', {
            hitType: 'event',
            eventCategory: 'OpenViewPane',
            eventAction: 'Conformance',
        });
    }

    onLibsChanged(): void {
        const newLibs = this.libsWidget.get();
        if (newLibs !== this.currentLibs) {
            this.currentLibs = newLibs;
            this.saveState();
            this.compileAll();
        }
    }

    initLibraries(state: PaneState & ConformanceViewState): void {
        const compilerIds = this.getCurrentCompilersIds();
        this.libsWidget = new LibsWidget(
            this.langId,
            compilerIds.join('|'),
            this.libsButton,
            state,
            this.onLibsChanged.bind(this),
            // @ts-expect-error: Typescript does not detect that this is correct
            this.getOverlappingLibraries(Array.isArray(compilerIds) ? compilerIds : [compilerIds])
        );
        // No callback is done on initialization, so make sure we store the current libs
        this.currentLibs = this.libsWidget.get();
    }

    initButtons(): void {
        this.conformanceContentRoot = this.domRoot.find('.conformance-wrapper');
        this.selectorList = this.domRoot.find('.compiler-list');
        this.addCompilerButton = this.domRoot.find('.add-compiler');
        this.selectorTemplate = $('#compiler-selector').find('.form-row');
        this.topBar = this.domRoot.find('.top-bar');
        this.libsButton = this.topBar.find('.show-libs');
        this.hideable = this.domRoot.find('.hideable');
    }

    initCallbacks(): void {
        this.container.on('destroy', () => {
            this.eventHub.unsubscribe();
            if (this.compilerInfo.editorId) this.eventHub.emit('conformanceViewClose', this.compilerInfo.editorId);
        });

        this.paneRenaming.on('renamePane', this.saveState.bind(this));

        this.container.on('destroy', this.close, this);
        this.container.on('open', () => {
            if (this.compilerInfo.editorId) this.eventHub.emit('conformanceViewOpen', this.compilerInfo.editorId);
        });

        this.container.on('resize', this.resize, this);
        this.container.on('shown', this.resize, this);
        this.eventHub.on('resize', this.resize, this);
        this.eventHub.on('editorChange', this.onEditorChange, this);
        this.eventHub.on('editorClose', this.onEditorClose, this);
        this.eventHub.on('languageChange', this.onLanguageChange, this);

        this.addCompilerButton.on('click', () => {
            this.addCompilerPicker();
            this.saveState();
        });
    }

    override getPaneName(): string {
        return 'Conformance Viewer (Editor #' + this.compilerInfo.editorId + ')';
    }

    override updateTitle(): void {
        let compilerText = '';
        // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
        if (this.compilerPickers && this.compilerPickers.length !== 0) {
            compilerText = ' ' + this.compilerPickers.length + '/' + this.maxCompilations;
        }
        const name = this.paneName ? this.paneName + compilerText : this.getPaneName() + compilerText;
        this.container.setTitle(_.escape(name));
    }

    addCompilerPicker(config?: AddCompilerPickerConfig): void {
        if (!config) {
            config = {
                // Compiler id which is being used
                compilerId: '',
                // Options which are in use
                options: options.compileOptions[this.langId],
            };
        }
        const newSelector = this.selectorTemplate.clone();
        const newCompilerEntry: CompilerEntry = {
            parent: newSelector,
            picker: null,
            optionsField: null,
            statusIcon: null,
            prependOptions: null,
        };

        const onOptionsChange = _.debounce(() => {
            this.saveState();
            this.compileChild(newCompilerEntry);
        }, 800);

        newCompilerEntry.optionsField = newSelector
            .find('.conformance-options')
            .val(config.options)
            .on('change', onOptionsChange)
            .on('keyup', onOptionsChange);

        newSelector
            .find('.close')
            .not('.extract-compiler')
            .not('.copy-compiler')
            .on('click', () => {
                this.removeCompilerPicker(newCompilerEntry);
            });

        newSelector.find('.close.copy-compiler').on('click', () => {
            const config: AddCompilerPickerConfig = {
                compilerId: newCompilerEntry.picker?.lastCompilerId ?? '',
                options: newCompilerEntry.optionsField?.val() || '',
            };
            this.copyCompilerPicker(config);
        });

        newCompilerEntry.statusIcon = newSelector.find('.status-icon');
        newCompilerEntry.prependOptions = newSelector.find('.prepend-options');
        const popCompilerButton = newSelector.find('.extract-compiler');

        const onCompilerChange = (compilerId: string) => {
            popCompilerButton.toggleClass('d-none', !compilerId);
            this.saveState();
            // Hide the results icon when a new compiler is selected
            this.handleStatusIcon(newCompilerEntry.statusIcon, {code: 0});
            const compiler = this.compilerService.findCompiler(this.langId, compilerId);
            if (compiler) this.setCompilationOptionsPopover(newCompilerEntry.prependOptions, compiler.options);
            this.updateLibraries();
            this.compileChild(newCompilerEntry);
        };

        newCompilerEntry.picker = new CompilerPicker(
            $(newSelector[0]),
            this.hub,
            this.langId,
            config.compilerId,
            onCompilerChange
        );

        const getCompilerConfig = () => {
            return Components.getCompilerWith(
                this.compilerInfo.editorId ?? 0,
                undefined,
                newCompilerEntry.optionsField?.val(),
                newCompilerEntry.picker?.lastCompilerId ?? '',
                this.langId,
                this.lastState?.libs
            );
        };

        // The .d.ts for GL lies. You can pass a function that returns the config as a second parameter
        this.container.layoutManager.createDragSource(popCompilerButton, getCompilerConfig as any);

        popCompilerButton.on('click', () => {
            const insertPoint =
                this.hub.findParentRowOrColumn(this.container.parent) ||
                this.container.layoutManager.root.contentItems[0];
            insertPoint.addChild(getCompilerConfig());
        });

        this.selectorList.append(newSelector);

        // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
        if (!this.compilerPickers) this.compilerPickers = [];
        this.compilerPickers.push(newCompilerEntry);

        this.handleToolbarUI();
    }

    override onCompiler(
        compilerId: number,
        compiler: CompilerInfo,
        options: string,
        editorId: number,
        treeId: number
    ): void {}

    setCompilationOptionsPopover(element: JQuery<HTMLElement> | null, content: string): void {
        element?.popover('dispose');
        element?.popover({
            content: content || 'No options in use',
            template:
                '<div class="popover' +
                (content ? ' compiler-options-popover' : '') +
                '" role="tooltip"><div class="arrow"></div>' +
                '<h3 class="popover-header"></h3><div class="popover-body"></div></div>',
        });
    }

    removeCompilerPicker(compilerEntry: CompilerEntry): void {
        this.compilerPickers = _.reject(this.compilerPickers, function (entry) {
            return compilerEntry.picker?.id === entry.picker?.id;
        });
        compilerEntry.picker?.tomSelect?.close();
        compilerEntry.parent.remove();

        this.updateLibraries();
        this.handleToolbarUI();
        this.saveState();
    }

    copyCompilerPicker(config: AddCompilerPickerConfig): void {
        this.addCompilerPicker(config);
        this.compileChild(this.compilerPickers.at(-1));
        this.saveState();
    }

    async expandToFiles(): Promise<SourceAndFiles> {
        if (this.sourceNeedsExpanding || !this.expandedSourceAndFiles) {
            const expanded = await this.compilerService.expandToFiles(this.source);
            this.expandedSourceAndFiles = expanded;
            this.sourceNeedsExpanding = false;
            return expanded;
        }
        return Promise.resolve(this.expandedSourceAndFiles);
    }

    onEditorChange(editorId: number, newSource: string, langId: string): void {
        if (editorId === this.compilerInfo.editorId && this.source !== newSource) {
            this.langId = langId;
            this.source = newSource;
            this.sourceNeedsExpanding = true;
            this.compileAll();
        }
    }

    onEditorClose(editorId: number): void {
        if (editorId === this.compilerInfo.editorId) {
            this.close();
            _.defer(function (self) {
                self.container.close();
            }, this);
        }
    }

    private hasResultAnyOutput(result: CompilationResult): boolean {
        // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
        return (result.stdout || []).length > 0 || (result.stderr || []).length > 0;
    }

    handleCompileOutIcon(element: JQuery<HTMLElement>, result: CompilationResult) {
        const hasOutput = this.hasResultAnyOutput(result);
        element.toggleClass('d-none', !hasOutput);
        if (hasOutput) {
            CompilerService.handleOutputButtonTitle(element, result);
        }
    }

    onCompileResponse(compilerEntry: CompilerEntry, result: CompilationResult) {
        let compilationOptions = '';
        if (result.compilationOptions) {
            compilationOptions = result.compilationOptions.join(' ');
        }

        this.setCompilationOptionsPopover(compilerEntry.prependOptions, compilationOptions);

        this.handleCompileOutIcon(compilerEntry.parent.find('.compiler-out'), result);

        this.handleStatusIcon(compilerEntry.statusIcon, CompilerService.calculateStatusIcon(result));
        this.saveState();
    }

    private getCompilerId(compilerEntry?: CompilerEntry): string | string[] {
        if (compilerEntry && compilerEntry.picker && compilerEntry.picker.tomSelect) {
            return compilerEntry.picker.tomSelect.getValue();
        }
        return '';
    }

    compileChild(compilerEntry) {
        const compilerId = this.getCompilerId(compilerEntry);
        if (compilerId === '') return;
        // Hide previous status icons
        this.handleStatusIcon(compilerEntry.statusIcon, {code: 4});

        this.expandToFiles().then(expanded => {
            const request = {
                source: expanded.source,
                compiler: compilerId,
                options: {
                    userArguments: compilerEntry.optionsField.val() || '',
                    filters: {},
                    compilerOptions: {produceAst: false, produceOptInfo: false, skipAsm: true},
                    libraries: [] as CompileChildLibraries[],
                },
                lang: this.langId,
                files: expanded.files,
            };

            this.currentLibs.forEach(item => {
                request.options.libraries.push({
                    id: item.name,
                    version: item.ver,
                });
            });

            // This error function ensures that the user will know we had a problem (As we don't save asm)
            this.compilerService
                .submit(request)
                .then((x: any) => {
                    this.onCompileResponse(compilerEntry, x.result);
                })
                .catch(x => {
                    this.onCompileResponse(compilerEntry, {
                        asm: [],
                        code: -1,
                        stdout: [],
                        stderr: x.error || x.message || x,
                        timedOut: false,
                    });
                });
        });
    }

    compileAll(): void {
        this.compilerPickers.forEach(this.compileChild.bind(this));
    }

    handleToolbarUI(): void {
        const compilerCount = this.compilerPickers.length;

        // Only allow new compilers if we allow for more
        this.addCompilerButton.prop('disabled', compilerCount >= this.maxCompilations);

        this.updateTitle();
    }

    handleStatusIcon(statusIcon, status): void {
        CompilerService.handleCompilationStatus(null, statusIcon, status);
    }

    currentState(): ConformanceViewState {
        // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
        if (!this.compilerPickers) this.compilerPickers = [];

        const compilers = this.compilerPickers.map(compilerEntry => ({
            compilerId: this.getCompilerId(compilerEntry),
            options: compilerEntry.optionsField?.val() || '',
        }));
        const state = {
            editorid: this.compilerInfo.editorId,
            langId: this.langId,
            compilers: compilers,
            libs: this.currentLibs,
        };
        this.paneRenaming.addState(state);
        return state;
    }

    saveState(): void {
        this.lastState = this.currentState();
        this.container.setState(this.lastState);
    }

    override resize(): void {
        // The pane becomes unusable long before this hides the icons
        // Added either way just in case we ever add more icons to this pane
        const topBarHeight = utils.updateAndCalcTopBarHeight(this.domRoot, this.topBar, this.hideable);
        this.conformanceContentRoot.outerHeight(this.domRoot.height() ?? 0 - topBarHeight);
    }

    getOverlappingLibraries(compilerIds: string[]): CompilerLibs {
        const compilers = compilerIds.map(compilerId => {
            return this.compilerService.findCompiler(this.langId, compilerId);
        });

        const langId = this.langId;

        let libraries: Record<string, Library | false> = {};
        let first = true;
        compilers.map(compiler => {
            if (compiler) {
                const filteredLibraries = LibUtils.getSupportedLibraries(compiler.libsArr, langId, compiler.remote);

                if (first) {
                    libraries = _.extend({}, filteredLibraries);
                    first = false;
                } else {
                    const libsInCommon = _.intersection(_.keys(libraries), _.keys(filteredLibraries));

                    for (const libKey in libraries) {
                        const lib = libraries[libKey];
                        if (lib && libsInCommon.includes(libKey)) {
                            const versionsInCommon = _.intersection(
                                Object.keys(lib.versions),
                                Object.keys(filteredLibraries[libKey].versions)
                            );

                            lib.versions = _.pick(lib.versions, (version, versionkey) => {
                                return versionsInCommon.includes(versionkey);
                            }) as Record<string, LibraryVersion>; // TODO(jeremy-rifkin)
                        } else {
                            libraries[libKey] = false;
                        }
                    }

                    libraries = _.omit(libraries, lib => {
                        return !lib || _.isEmpty(lib.versions);
                    }) as Record<string, Library>; // TODO(jeremy-rifkin)
                }
            }
        });

        return libraries as CompilerLibs; // TODO(jeremy-rifkin)
    }

    getCurrentCompilersIds() {
        return _.uniq(
            this.compilerPickers
                .map(compilerEntry => {
                    return this.getCompilerId(compilerEntry);
                })
                .filter(compilerId => {
                    return compilerId !== '';
                })
        );
    }

    updateLibraries(): void {
        const compilerIds = this.getCurrentCompilersIds();
        this.libsWidget.setNewLangId(
            this.langId,
            compilerIds.join('|'),
            // @ts-expect-error: This is actually ok
            this.getOverlappingLibraries(Array.isArray(compilerIds) ? compilerIds : [compilerIds])
        );
    }

    onLanguageChange(editorId: number | boolean, newLangId: string): void {
        if (editorId === this.compilerInfo.editorId && this.langId !== newLangId) {
            const oldLangId = this.langId;
            this.stateByLang[oldLangId] = this.currentState();

            this.langId = newLangId;
            this.compilerPickers.forEach(compilerEntry => {
                compilerEntry.picker?.tomSelect?.close();
                compilerEntry.parent.remove();
            });
            this.compilerPickers = [];
            const langState = this.stateByLang[newLangId];
            this.initFromState(langState);
            this.updateLibraries();
            this.handleToolbarUI();
            this.saveState();
        }
    }

    override close(): void {
        this.eventHub.unsubscribe();
        this.compilerPickers.forEach(compilerEntry => {
            compilerEntry.picker?.tomSelect?.close();
            compilerEntry.parent.remove();
        });
        if (this.compilerInfo.editorId) this.eventHub.emit('conformanceViewClose', this.compilerInfo.editorId);
    }

    initFromState(state?: ConformanceViewState): void {
        if (state && state.compilers) {
            this.lastState = state;
            _.each(state.compilers, _.bind(this.addCompilerPicker, this));
        } else {
            this.lastState = this.currentState();
        }
    }

    getDefaultPaneName(): string {
        return '';
    }

    onCompileResult(compilerId: number, compiler: unknown, result: unknown): void {}
}
