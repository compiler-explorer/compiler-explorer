// Copyright (c) 2022, Compiler Explorer Authors
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
import * as monaco from 'monaco-editor';
import * as sifter from '@orchidjs/sifter';
import {Container} from 'golden-layout';
import TomSelect from 'tom-select';
import scrollIntoView from 'scroll-into-view-if-needed';

import {MonacoPane} from './pane.js';
import {OptPipelineViewState} from './opt-pipeline.interfaces.js';
import {MonacoPaneState} from './pane.interfaces.js';

import {extendConfig} from '../monaco-config.js';
import {Hub} from '../hub.js';
import * as utils from '../utils.js';
import {Toggles} from '../widgets/toggles.js';

import {
    OptPipelineBackendOptions,
    OptPipelineOutput,
    OptPipelineResults,
} from '../compilation/opt-pipeline-output.interfaces.js';
import {unwrap, unwrapString} from '../assert.js';
import {CompilationResult} from '../compilation/compilation.interfaces.js';
import {CompilerInfo} from '../compiler.interfaces.js';
import {escapeHTML} from '../../shared/common-utils.js';

const MIN_SIDEBAR_WIDTH = 100;

export class OptPipeline extends MonacoPane<monaco.editor.IStandaloneDiffEditor, OptPipelineViewState> {
    results: OptPipelineResults = {};
    compiler: CompilerInfo | null;
    groupName: JQuery;
    passesColumn: JQuery;
    passesFilter: JQuery;
    passesList: JQuery;
    passesColumnResizer: JQuery;
    body: JQuery;
    clickCallback: (e: JQuery.ClickEvent) => void;
    keydownCallback: (e: JQuery.KeyDownEvent) => void;
    isPassListSelected = false;
    groupSelector: TomSelect;
    originalModel: any;
    modifiedModel: any;
    options: Toggles;
    filters: Toggles;
    state: OptPipelineViewState;
    lastOptions: OptPipelineBackendOptions = {
        filterDebugInfo: true,
        filterIRMetadata: false,
        fullModule: false,
        noDiscardValueNames: true,
        demangle: true,
        libraryFunctions: true,
    };
    resizeStartX: number;
    resizeStartWidth: number;
    resizeDragMoveBind: (e: MouseEvent) => void;
    resizeDragEndBind: (e: MouseEvent) => void;
    firstResults = true;

    constructor(hub: Hub, container: Container, state: OptPipelineViewState & MonacoPaneState) {
        super(hub, container, state);
        this.groupName = this.domRoot.find('.opt-group-name');
        this.updateGroupName();
        this.passesColumn = this.domRoot.find('.passes-column');
        this.passesFilter = this.domRoot.find('.passes-filter');
        this.passesList = this.domRoot.find('.passes-list');
        this.body = this.domRoot.find('.opt-pipeline-body');

        if (state.sidebarWidth === 0) {
            _.defer(() => {
                state.sidebarWidth = parseInt(
                    unwrap(document.defaultView).getComputedStyle(this.passesColumn.get()[0]).width,
                    10,
                );
                state.sidebarWidth = Math.max(state.sidebarWidth, MIN_SIDEBAR_WIDTH);
                this.resize();
                this.updateState();
            });
        } else {
            state.sidebarWidth = Math.max(state.sidebarWidth, MIN_SIDEBAR_WIDTH);
            this.passesColumn.get()[0].style.width = state.sidebarWidth + 'px';
        }
        this.state = state;
        this.upgradeStateFields();
        const selector = this.domRoot.get()[0].getElementsByClassName('group-selector')[0];
        if (!(selector instanceof HTMLSelectElement)) {
            throw new Error('.group-selector is not an HTMLSelectElement');
        }
        this.groupSelector = new TomSelect(selector, {
            valueField: 'value',
            labelField: 'title',
            searchField: ['title'],
            dropdownParent: 'body',
            plugins: ['input_autogrow'],
            sortField: 'title',
            maxOptions: 1000,
            onChange: (e: string) => this.selectGroup(e),
        });
        this.groupSelector.on('dropdown_close', () => {
            // scroll back to the selection on the next open
            const selection = unwrap(this.groupSelector).getOption(this.state.selectedGroup);
            this.groupSelector.setActiveOption(selection);
        });
        this.clickCallback = this.onClickCallback.bind(this);
        this.keydownCallback = this.onKeydownCallback.bind(this);
        $(document).on('click', this.clickCallback);
        $(document).on('keydown', this.keydownCallback);
        this.eventHub.emit('optPipelineViewOpened', this.compilerInfo.compilerId);
        this.eventHub.emit('requestSettings');
        this.emitOptions(true);
        this.passesFilter.on('input', _.debounce(this.onFiltersChange.bind(this), 250));
        this.updateButtons();
    }

    upgradeStateFields() {
        // `selectedGroup` replaces `selectedFunction`
        if (this.state.selectedFunction) {
            this.state.selectedGroup = this.state.selectedFunction;
            delete this.state.selectedFunction;
        }
    }

    override initializeStateDependentProperties(state: OptPipelineViewState & MonacoPaneState) {
        const langId = state.lang;
        const compilerId = state.compiler;
        if (langId && compilerId) {
            const result = this.hub.compilerService.processFromLangAndCompiler(langId, compilerId);
            this.compiler = result?.compiler ?? null;
        } else {
            // With older state that's missing `lang` and `compiler`,
            // we fallback to previous functionality (the compiler info is
            // currently only used to tweak the UI for newer languages that did
            // not offer this view previously).
            this.compiler = null;
        }
    }

    override getInitialHTML(): string {
        return $('#opt-pipeline').html();
    }

    getMonacoLanguage(): string {
        let monacoLanguage = 'llvm-ir';
        if (this.compiler) {
            monacoLanguage = this.compiler.optPipeline?.monacoLanguage ?? 'llvm-ir';
        }
        return monacoLanguage;
    }

    override createEditor(editorRoot: HTMLElement): void {
        const monacoLanguage = this.getMonacoLanguage();
        this.editor = monaco.editor.createDiffEditor(
            editorRoot,
            extendConfig({
                language: monacoLanguage,
                readOnly: true,
                glyphMargin: true,
                lineNumbersMinChars: 3,
            }),
        );
        this.originalModel = monaco.editor.createModel('', monacoLanguage);
        this.modifiedModel = monaco.editor.createModel('', monacoLanguage);
        this.editor.setModel({original: this.originalModel, modified: this.modifiedModel});
    }

    updateEditor() {
        const monacoLanguage = this.getMonacoLanguage();
        monaco.editor.setModelLanguage(this.originalModel, monacoLanguage);
        monaco.editor.setModelLanguage(this.modifiedModel, monacoLanguage);
    }

    override getPrintName() {
        return '<Unimplemented>';
    }

    override sendPrintData() {
        // nop
    }

    override getDefaultPaneName(): string {
        return 'Opt Pipeline Viewer';
    }

    override registerButtons(state: OptPipelineViewState) {
        super.registerButtons(state);
        this.options = new Toggles(this.domRoot.find('.options'), state as unknown as Record<string, boolean>);
        this.options.on('change', this.onOptionsChange.bind(this));
        this.filters = new Toggles(this.domRoot.find('.filters'), state as unknown as Record<string, boolean>);
        this.filters.on('change', this.onOptionsChange.bind(this));

        this.passesColumnResizer = this.domRoot.find('.passes-column-resizer');
        this.passesColumnResizer.get()[0].addEventListener('mousedown', this.initResizeDrag.bind(this), false);
    }

    updateButtons() {
        if (!this.compiler || !this.compiler.optPipeline) return;

        const {supportedOptions, supportedFilters, initialOptionsState, initialFiltersState} =
            this.compiler.optPipeline;
        if (supportedOptions) {
            for (const key of ['dump-full-module', '-fno-discard-value-names', 'demangle-symbols']) {
                this.options.enableToggle(key, supportedOptions.includes(key));
            }
        }
        if (supportedFilters) {
            for (const key of ['filter-debug-info', 'filter-instruction-metadata']) {
                this.filters.enableToggle(key, supportedFilters.includes(key));
            }
        }
        if (initialOptionsState) {
            for (const key in initialOptionsState) {
                this.options.set(key, initialOptionsState[key]);
            }
        }
        if (initialFiltersState) {
            for (const key in initialFiltersState) {
                this.filters.set(key, initialFiltersState[key]);
            }
        }
    }

    initResizeDrag(e: MouseEvent) {
        // taken from SO
        this.resizeStartX = e.clientX;
        this.resizeStartWidth = parseInt(
            unwrap(document.defaultView).getComputedStyle(this.passesColumn.get()[0]).width,
            10,
        );
        this.resizeDragMoveBind = this.resizeDragMove.bind(this);
        this.resizeDragEndBind = this.resizeDragEnd.bind(this);
        document.documentElement.addEventListener('mousemove', this.resizeDragMoveBind, false);
        document.documentElement.addEventListener('mouseup', this.resizeDragEndBind, false);
    }

    resizeDragMove(e: MouseEvent) {
        let width = this.resizeStartWidth + e.clientX - this.resizeStartX;
        if (width < MIN_SIDEBAR_WIDTH) {
            width = MIN_SIDEBAR_WIDTH;
        }
        this.passesColumn.get()[0].style.width = width + 'px';
        this.state.sidebarWidth = width;
        this.resize();
    }

    resizeDragEnd(e: MouseEvent) {
        document.documentElement.removeEventListener('mousemove', this.resizeDragMoveBind, false);
        document.documentElement.removeEventListener('mouseup', this.resizeDragEndBind, false);
    }

    emitOptions(force = false) {
        const options = this.options.get();
        const filters = this.filters.get();
        // TODO: Make use of filter-inconsequential-passes on the back end? Maybe provide a specific function arg to
        // the backend? Would be a data transfer optimization.
        const newOptions: OptPipelineBackendOptions = {
            //'filter-inconsequential-passes': options['filter-inconsequential-passes'],
            filterDebugInfo: filters['filter-debug-info'],
            filterIRMetadata: filters['filter-instruction-metadata'],
            fullModule: options['dump-full-module'],
            noDiscardValueNames: options['-fno-discard-value-names'],
            demangle: options['demangle-symbols'],
            libraryFunctions: options['library-functions'],
        };
        let changed = false;
        for (const k in newOptions) {
            const key = k as keyof OptPipelineBackendOptions;
            if (newOptions[key] !== this.lastOptions[key]) {
                changed = true;
            }
        }
        this.lastOptions = newOptions;
        if (changed || force) {
            this.eventHub.emit('optPipelineViewOptionsUpdated', this.compilerInfo.compilerId, newOptions, true);
        }
    }

    onOptionsChange() {
        // Redo pass sidebar
        this.selectGroup(this.state.selectedGroup);
        // Inform compiler of the options
        this.emitOptions();
    }

    override onCompileResult(compilerId: number, compiler: CompilerInfo, result: CompilationResult): void {
        if (this.compilerInfo.compilerId !== compilerId) return;
        if (result.optPipelineOutput) {
            const output: OptPipelineOutput = unwrap(result.optPipelineOutput);
            if (output.error) {
                this.editor
                    .getModel()
                    ?.original.setValue(
                        `<An error occurred while generating the optimization pipeline output: ${output.error}>`,
                    );
                this.editor.getModel()?.modified.setValue('');
            }
            this.updateResults(output.results);
        } else if (compiler.optPipeline) {
            this.updateResults({});
            this.editor.getModel()?.original.setValue('<Error>');
            this.editor.getModel()?.modified.setValue('');
        }
    }

    override onCompiler(
        compilerId: number,
        compiler: CompilerInfo | null,
        options: string,
        editorId: number,
        treeId: number,
    ): void {
        if (this.compilerInfo.compilerId !== compilerId) return;
        this.compilerInfo.compilerName = compiler ? compiler.name : '';
        this.compilerInfo.editorId = editorId;
        this.compilerInfo.treeId = treeId;
        this.updateTitle();
        this.compiler = compiler;
        this.updateGroupName();
        this.updateButtons();
        this.updateEditor();
        if (compiler && !compiler.optPipeline) {
            //this.editor.setValue('<Opt pipeline output is not supported for this compiler>');
        }
    }

    updateGroupName() {
        if (!this.compiler) return;
        const groupNameText = this.compiler.optPipeline?.groupName || 'Function';
        this.groupName.text(`${groupNameText}: `);
    }

    updateResults(results: OptPipelineResults): void {
        this.results = results;
        //const groups = Object.keys(result);
        let selectedGroup = this.state.selectedGroup; // one of the .clear calls below will end up resetting this
        this.groupSelector.clear();
        this.groupSelector.clearOptions();
        const keys = Object.keys(results);
        if (keys.length === 0) {
            this.groupSelector.addOption({
                title: '<No groups available>',
                value: '<No groups available>',
            });
        }
        for (const fn of keys) {
            this.groupSelector.addOption({
                title: fn,
                value: fn,
            });
        }
        this.passesList.empty();
        if (keys.length > 0) {
            if (selectedGroup === '' || !(selectedGroup in results)) {
                selectedGroup = keys[0];
            }
            this.groupSelector.setValue(selectedGroup);
        } else {
            // restore this.selectedGroup, next time the compilation results aren't errors the selected group will
            // still be the same
            this.state.selectedGroup = selectedGroup;
        }
    }

    selectGroup(name: string) {
        this.state.selectedGroup = name;
        if (!(name in this.results)) {
            return;
        }
        const filterInconsequentialPasses = this.filters.get()['filter-inconsequential-passes'];
        const passes = this.results[name];
        this.passesList.empty();
        let isFirstMachinePass = true;
        const newPasses: any = [];
        for (const [i, pass] of passes.entries()) {
            if (filterInconsequentialPasses && !pass.irChanged) {
                continue;
            }
            let className = pass.irChanged ? 'changed' : '';
            if (pass.machine && isFirstMachinePass) {
                className += ' firstMachinePass';
                isFirstMachinePass = false;
            }
            newPasses.push({
                name: pass.name,
                div: `<div data-i="${i}" class="pass ${className}">${escapeHTML(pass.name)}</div>`,
            });
        }
        this.filterPasses(newPasses);
        const passDivs = this.passesList.find('.pass');
        passDivs.on('click', e => {
            const target = e.target;
            this.passesList.find('.active').removeClass('active');
            $(target).addClass('active');
            this.displayPass(parseInt(unwrap(target.getAttribute('data-i'))));
        });
        // try to select a pass
        if (this.state.selectedIndex >= passes.length) {
            this.state.selectedIndex = 0;
        }
        const selectedPassDiv = this.passesList.find(`[data-i=${this.state.selectedIndex}]`);
        selectedPassDiv.addClass('active');
        // displayPass updates state
        this.displayPass(this.state.selectedIndex);
        // if loading from a url center the active pass
        if (this.firstResults) {
            this.firstResults = false;
            const activePass = this.passesList.find('.active').get(0);
            if (activePass) {
                scrollIntoView(activePass, {
                    scrollMode: 'if-needed',
                    block: 'center',
                });
            }
        }
    }

    private filterPasses(newPasses: any) {
        const filterValue = unwrapString(this.passesFilter.val()).trim();
        if (filterValue === '') {
            for (const pass of newPasses) {
                this.passesList.append(pass.div);
            }
        } else {
            const searcher = new sifter.Sifter(newPasses, {diacritics: false});
            const filteredPasses = searcher.search(filterValue, {
                fields: ['name'],
                conjunction: 'and',
                sort: 'name',
            });
            for (const result of filteredPasses.items) {
                this.passesList.append(newPasses[result.id].div);
            }
        }
    }

    displayPass(i: number) {
        if (this.state.selectedGroup in this.results && i < this.results[this.state.selectedGroup].length) {
            this.state.selectedIndex = i;
            const pass = this.results[this.state.selectedGroup][i];
            const before = pass.before.map(x => x.text).join('\n');
            const after = pass.after.map(x => x.text).join('\n');
            this.editor.getModel()?.original.setValue(before);
            this.editor.getModel()?.modified.setValue(after);
            this.updateState();
        }
    }

    onClickCallback(e: JQuery.ClickEvent) {
        this.isPassListSelected = unwrap(this.passesList.get(0)).contains(e.target);
    }

    onKeydownCallback(e: JQuery.KeyDownEvent) {
        if (this.isPassListSelected) {
            if (e.key === 'ArrowUp') {
                e.preventDefault();
                const active = this.passesList.find('.active');
                const prev = active.prev().get(0);
                if (prev) {
                    active.removeClass('active');
                    $(prev).addClass('active');
                    scrollIntoView(prev, {
                        scrollMode: 'if-needed',
                        block: 'nearest',
                    });
                    this.displayPass(parseInt(unwrap(prev.getAttribute('data-i'))));
                }
            }
            if (e.key === 'ArrowDown') {
                e.preventDefault();
                const active = this.passesList.find('.active');
                const next = active.next().get(0);
                if (next) {
                    active.removeClass('active');
                    $(next).addClass('active');
                    scrollIntoView(next, {
                        scrollMode: 'if-needed',
                        block: 'nearest',
                    });
                    this.displayPass(parseInt(unwrap(next.getAttribute('data-i'))));
                }
            }
        }
    }

    onFiltersChange(_: any): void {
        this.selectGroup(this.state.selectedGroup);
    }

    override getCurrentState() {
        return {
            ...this.options.get(),
            ...this.filters.get(),
            ...super.getCurrentState(),
            selectedGroup: this.state.selectedGroup,
            selectedIndex: this.state.selectedIndex,
            sidebarWidth: this.state.sidebarWidth,
        };
    }

    override resize() {
        _.defer(() => {
            const topBarHeight = utils.updateAndCalcTopBarHeight(this.domRoot, this.topBar, this.hideable);
            const otherWidth = unwrap(this.passesColumn.width()) + unwrap(this.passesColumnResizer.width());
            const domWidth = unwrap(this.domRoot.width());
            if (otherWidth > domWidth) {
                this.passesColumn.get()[0].style.width = domWidth + 'px';
            }
            this.editor.layout({
                width: domWidth - otherWidth,
                height: unwrap(this.domRoot.height()) - topBarHeight,
            });
            unwrap(this.body.get(0)).style.height = unwrap(this.domRoot.height()) - topBarHeight + 'px';
        });
    }

    override close(): void {
        $(document).off('click', this.clickCallback);
        $(document).off('keydown', this.keydownCallback);
        this.eventHub.unsubscribe();
        this.eventHub.emit('optPipelineViewClosed', this.compilerInfo.compilerId);
        this.editor.dispose();
    }
}
