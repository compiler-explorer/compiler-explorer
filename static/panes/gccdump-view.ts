// Copyright (c) 2017, Marc Poulhiès - Kalray Inc.
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

import {Container} from 'golden-layout';
import {Hub} from '../hub';

import TomSelect from 'tom-select';
import {Toggles} from '../widgets/toggles';

import * as monaco from 'monaco-editor';
import {MonacoPane} from './pane';
import {MonacoPaneState} from './pane.interfaces';
import * as monacoConfig from '../monaco-config';

import {GccDumpFiltersState, GccDumpViewState, GccDumpViewSelectedPass} from './gccdump-view.interfaces';

import {ga} from '../analytics';
import {assert} from '../assert';

export class GccDump extends MonacoPane<monaco.editor.IStandaloneCodeEditor, GccDumpViewState> {
    selectize: TomSelect;
    uiIsReady: boolean;
    filters: Toggles;
    dumpFiltersButtons: JQuery<HTMLElement>;
    dumpTreesButton: JQuery<HTMLElement>;
    dumpTreesTitle: string;
    dumpRtlButton: JQuery<HTMLElement>;
    dumpRtlTitle: string;
    dumpIpaButton: JQuery<HTMLElement>;
    dumpIpaTitle: string;
    optionAddressButton: JQuery<HTMLElement>;
    optionAddressTitle: string;
    optionSlimButton: JQuery<HTMLElement>;
    optionSlimTitle: string;
    optionRawButton: JQuery<HTMLElement>;
    optionRawTitle: string;
    optionDetailsButton: JQuery<HTMLElement>;
    optionDetailsTitle: string;
    optionStatsButton: JQuery<HTMLElement>;
    optionStatsTitle: string;
    optionBlocksButton: JQuery<HTMLElement>;
    optionBlocksTitle: string;
    optionVopsButton: JQuery<HTMLElement>;
    optionVopsTitle: string;
    optionLinenoButton: JQuery<HTMLElement>;
    optionLinenoTitle: string;
    optionUidButton: JQuery<HTMLElement>;
    optionUidTitle: string;
    optionAllButton: JQuery<HTMLElement>;
    optionAllTitle: string;
    inhibitPassSelect = false;
    cursorSelectionThrottledFunction: ((e: any) => void) & _.Cancelable;
    selectedPass: string | null = null;

    constructor(hub: Hub, container: Container, state: GccDumpViewState & MonacoPaneState) {
        super(hub, container, state);

        if (state.selectedPass && typeof state.selectedPass === 'string') {
            // To keep URL format stable wrt GccDump, only a string of the form 'r.expand' is stored.
            // Old links also have the pass number prefixed but this can be ignored.
            // Create the object that will be used instead of this bare string.
            const selectedPassRe = /[0-9]*(i|t|r)\.([\w-_]*)/;
            const passType = {
                i: 'ipa',
                r: 'rtl',
                t: 'tree',
            };
            const match = state.selectedPass.match(selectedPassRe);
            if (match) {
                const selectedPassO: GccDumpViewSelectedPass = {
                    filename_suffix: match[1] + '.' + match[2],
                    name: match[2] + ' (' + passType[match[1]] + ')',
                    command_prefix: '-fdump-' + passType[match[1]] + '-' + match[2],

                    // FIXME(dkm): maybe this could be avoided by better typing.
                    selectedPass: null,
                };

                this.eventHub.emit('gccDumpPassSelected', this.compilerInfo.compilerId, selectedPassO, false);
            }
        }

        // until we get our first result from compilation backend with all fields,
        // disable UI callbacks.
        this.uiIsReady = false;
        this.onUiNotReady();

        this.eventHub.emit('gccDumpFiltersChanged', this.compilerInfo.compilerId, this.getEffectiveFilters(), false);

        this.updateButtons();
        this.updateState();

        // UI is ready, request compilation to get passes list and
        // current output (if any)
        this.eventHub.emit('gccDumpUIInit', this.compilerInfo.compilerId);
    }

    override getInitialHTML(): string {
        return $('#gccdump').html();
    }

    override createEditor(editorRoot: HTMLElement): monaco.editor.IStandaloneCodeEditor {
        return monaco.editor.create(
            editorRoot,
            monacoConfig.extendConfig({
                readOnly: true,
                glyphMargin: true,
                lineNumbersMinChars: 3,
                dropdownParent: 'body',
            })
        );
    }

    override registerOpeningAnalyticsEvent() {
        ga.proxy('send', {
            hitType: 'event',
            eventCategory: 'OpenViewPane',
            eventAction: 'GccDump',
        });
    }

    override registerButtons(state: GccDumpViewState & MonacoPaneState) {
        super.registerButtons(state);

        const gccdump_picker = this.domRoot.find('.gccdump-pass-picker').get(0);
        if (!(gccdump_picker instanceof HTMLSelectElement)) {
            throw new Error('.gccdump-pass-picker is not an HTMLSelectElement');
        }
        assert(gccdump_picker instanceof HTMLSelectElement);
        this.selectize = new TomSelect(gccdump_picker, {
            sortField: undefined, // do not sort
            valueField: 'name',
            labelField: 'name',
            searchField: ['name'],
            options: [],
            items: [],
            plugins: ['input_autogrow'],
            maxOptions: 500,
        });

        this.filters = new Toggles(this.domRoot.find('.dump-filters'), state as any as Record<string, boolean>);

        this.dumpFiltersButtons = this.domRoot.find('.dump-filters .btn');

        this.dumpTreesButton = this.domRoot.find("[data-bind='treeDump']");
        this.dumpTreesTitle = this.dumpTreesButton.prop('title');

        this.dumpRtlButton = this.domRoot.find("[data-bind='rtlDump']");
        this.dumpRtlTitle = this.dumpRtlButton.prop('title');

        this.dumpIpaButton = this.domRoot.find("[data-bind='ipaDump']");
        this.dumpIpaTitle = this.dumpIpaButton.prop('title');

        this.optionAddressButton = this.domRoot.find("[data-bind='addressOption']");
        this.optionAddressTitle = this.optionAddressButton.prop('title');

        this.optionSlimButton = this.domRoot.find("[data-bind='slimOption']");
        this.optionSlimTitle = this.optionSlimButton.prop('title');

        this.optionRawButton = this.domRoot.find("[data-bind='rawOption']");
        this.optionRawTitle = this.optionRawButton.prop('title');

        this.optionDetailsButton = this.domRoot.find("[data-bind='detailsOption']");
        this.optionDetailsTitle = this.optionDetailsButton.prop('title');

        this.optionStatsButton = this.domRoot.find("[data-bind='statsOption']");
        this.optionStatsTitle = this.optionStatsButton.prop('title');

        this.optionBlocksButton = this.domRoot.find("[data-bind='blocksOption']");
        this.optionBlocksTitle = this.optionBlocksButton.prop('title');

        this.optionVopsButton = this.domRoot.find("[data-bind='vopsOption']");
        this.optionVopsTitle = this.optionVopsButton.prop('title');

        this.optionLinenoButton = this.domRoot.find("[data-bind='linenoOption']");
        this.optionLinenoTitle = this.optionLinenoButton.prop('title');

        this.optionUidButton = this.domRoot.find("[data-bind='uidOption']");
        this.optionUidTitle = this.optionUidButton.prop('title');

        this.optionAllButton = this.domRoot.find("[data-bind='allOption']");
        this.optionAllTitle = this.optionAllButton.prop('title');
    }

    override registerCallbacks() {
        this.filters.on('change', this.onFilterChange.bind(this));
        this.selectize.on('change', this.onPassSelect.bind(this));

        this.eventHub.emit('gccDumpViewOpened', this.compilerInfo.compilerId);
        this.eventHub.emit('requestSettings');

        this.container.on('resize', this.resize, this);
        this.container.on('shown', this.resize, this);

        this.cursorSelectionThrottledFunction = _.throttle(this.onDidChangeCursorSelection.bind(this), 500);
        this.editor.onDidChangeCursorSelection(e => {
            this.cursorSelectionThrottledFunction(e);
        });
    }

    updateButtons() {
        const formatButtonTitle = (button, title) =>
            button.prop('title', '[' + (button.hasClass('active') ? 'ON' : 'OFF') + '] ' + title);
        formatButtonTitle(this.dumpTreesButton, this.dumpTreesTitle);
        formatButtonTitle(this.dumpRtlButton, this.dumpRtlTitle);
        formatButtonTitle(this.dumpIpaButton, this.dumpIpaTitle);
        formatButtonTitle(this.optionAddressButton, this.optionAddressTitle);
        formatButtonTitle(this.optionSlimButton, this.optionSlimTitle);
        formatButtonTitle(this.optionRawButton, this.optionRawTitle);
        formatButtonTitle(this.optionDetailsButton, this.optionDetailsTitle);
        formatButtonTitle(this.optionStatsButton, this.optionStatsTitle);
        formatButtonTitle(this.optionBlocksButton, this.optionBlocksTitle);
        formatButtonTitle(this.optionVopsButton, this.optionVopsTitle);
        formatButtonTitle(this.optionLinenoButton, this.optionLinenoTitle);
        formatButtonTitle(this.optionUidButton, this.optionUidTitle);
        formatButtonTitle(this.optionAllButton, this.optionAllTitle);
    }

    // Disable view's menu when invalid compiler has been
    // selected after view is opened.
    onUiNotReady() {
        // disable drop down menu and buttons
        this.selectize.disable();
        this.dumpFiltersButtons.prop('disabled', true);
    }

    onUiReady() {
        // enable drop down menu and buttons
        this.selectize.enable();

        this.dumpFiltersButtons.prop('disabled', false);
    }

    onPassSelect(passId: string) {
        const selectedPass = this.selectize.options[passId] as unknown as GccDumpViewSelectedPass;

        if (this.inhibitPassSelect !== true) {
            this.eventHub.emit('gccDumpPassSelected', this.compilerInfo.compilerId, selectedPass, true);
        }

        // To keep shared URL compatible, we keep on storing only a string in the
        // state and stick to the original format.
        // Previously, we were simply storing the full file suffix (the part after [...]):
        //    [file.c.]123t.expand
        // We don't have the number now, but we can store the file suffix without this number
        // (the number is useless and should probably have never been there in the
        // first place).

        this.selectedPass = selectedPass.filename_suffix;
        this.updateState();
    }

    // Called after result from new compilation received
    // if gccDumpOutput is false, cleans the select menu
    updatePass(filters, selectize, gccDumpOutput) {
        const passes = gccDumpOutput ? gccDumpOutput.all : [];

        // we are changing selectize but don't want any callback to
        // trigger new compilation
        this.inhibitPassSelect = true;

        selectize.clear(true);
        selectize.clearOptions();

        for (const p of passes) {
            selectize.addOption(p);
        }

        if (gccDumpOutput.selectedPass) selectize.addItem(gccDumpOutput.selectedPass.name, true);
        else selectize.clear(true);

        this.eventHub.emit('gccDumpPassSelected', this.compilerInfo.compilerId, gccDumpOutput.selectedPass, false);

        this.inhibitPassSelect = false;
    }

    override onCompileResult(id, compiler, result) {
        if (this.compilerInfo.compilerId !== id || !compiler) return;

        const model = this.editor.getModel();
        if (model) {
            if (result.gccDumpOutput && result.gccDumpOutput.syntaxHighlight) {
                monaco.editor.setModelLanguage(model, 'gccdump-rtl-gimple');
            } else {
                monaco.editor.setModelLanguage(model, 'plaintext');
            }
        }
        if (compiler.supportsGccDump && result.gccDumpOutput) {
            const currOutput = result.gccDumpOutput.currentPassOutput;

            // if result contains empty selected pass, probably means
            // we requested an invalid/outdated pass.
            if (!result.gccDumpOutput.selectedPass) {
                this.selectize.clear(true);
                this.selectedPass = null;
            }
            this.updatePass(this.filters, this.selectize, result.gccDumpOutput);
            this.showGccDumpResults(currOutput);

            // enable UI on first successful compilation or after an invalid compiler selection (eg. clang)
            if (!this.uiIsReady) {
                this.uiIsReady = true;
                this.onUiReady();
            }
        } else {
            this.selectize.clear(true);
            this.selectedPass = null;
            this.updatePass(this.filters, this.selectize, false);
            this.uiIsReady = false;
            this.onUiNotReady();
            if (!compiler.supportsGccDump) {
                this.showGccDumpResults('<Tree/RTL output is not supported for this compiler (GCC only)>');
            } else {
                this.showGccDumpResults('<Tree/RTL output is empty>');
            }
        }
        this.updateState();
    }

    override getDefaultPaneName() {
        return 'GCC Tree/RTL Viewer';
    }

    showGccDumpResults(results) {
        this.editor.setValue(results);

        if (!this.isAwaitingInitialResults) {
            if (this.selection) {
                this.editor.setSelection(this.selection);
                this.editor.revealLinesInCenter(this.selection.startLineNumber, this.selection.endLineNumber);
            }
            this.isAwaitingInitialResults = true;
        }
    }

    override onCompiler(compilerId: number, compiler, options: unknown, editorId: number, treeId: number) {
        if (compilerId === this.compilerInfo.compilerId) {
            this.compilerInfo.compilerName = compiler ? compiler.name : '';
            this.compilerInfo.editorId = editorId;
            this.compilerInfo.treeId = treeId;
            this.updateTitle();
            // TODO(jeremy-rifkin): Panes like ast-view handle the case here where the compiler doesn't support
            // the view
        }
    }

    override onCompilerClose(id) {
        if (id === this.compilerInfo.compilerId) {
            // We can't immediately close as an outer loop somewhere in GoldenLayout is iterating over
            // the hierarchy. We can't modify while it's being iterated over.
            this.close();
            _.defer(function (self) {
                self.container.close();
            }, this);
        }
    }

    getEffectiveFilters(): GccDumpFiltersState {
        // This cast only works if gccdump.pug and gccdump-view.interfaces are
        // kept synchronized. See comment in gccdump-view.interfaces.ts.

        return this.filters.get() as unknown as GccDumpFiltersState;
    }

    onFilterChange() {
        this.updateState();
        this.updateButtons();

        if (this.inhibitPassSelect !== true) {
            this.eventHub.emit('gccDumpFiltersChanged', this.compilerInfo.compilerId, this.getEffectiveFilters(), true);
        }
    }

    override getCurrentState() {
        const parent = super.getCurrentState();
        const filters = this.getEffectiveFilters(); // TODO: Validate somehow?
        const state: MonacoPaneState & GccDumpViewState = {
            // filters needs to come first, the entire state is given to the toggles and we don't want to override
            // properties such as selectedPass with obsolete values
            ...filters,

            selectedPass: this.selectedPass,

            // See FIXME(dkm) comment in gccdump-view.interfaces.ts.
            filename_suffix: this.selectedPass,
            name: null,
            command_prefix: null,

            ...parent,
        };
        // TODO(jeremy-rifkin)
        return state as any;
    }

    override close() {
        this.eventHub.unsubscribe();
        this.eventHub.emit('gccDumpViewClosed', this.compilerInfo.compilerId);
        this.editor.dispose();
    }
}
