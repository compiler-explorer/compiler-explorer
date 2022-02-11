// Copyright (c) 2017, Marc Poulhiès - Kalray Inc.
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

'use strict';

var FontScale = require('../fontscale').FontScale;
var monaco = require('monaco-editor');
var Toggles = require('../toggles').Toggles;
require('../modes/gccdump-rtl-gimple-mode');
var _ = require('underscore');
var $ = require('jquery');
var ga = require('../analytics').ga;
var monacoConfig = require('../monaco-config');
var TomSelect = require('tom-select');
var utils = require('../utils');
var PaneRenaming = require('../pane-renaming').PaneRenaming;


function GccDump(hub, container, state) {
    this.container = container;
    this.eventHub = hub.createEventHub();
    this.domRoot = container.getElement();
    this.domRoot.html($('#gccdump').html());
    var root = this.domRoot.find('.monaco-placeholder');

    this.gccDumpEditor = monaco.editor.create(root[0], monacoConfig.extendConfig({
        readOnly: true,
        glyphMargin: true,
        lineNumbersMinChars: 3,
        dropdownParent: 'body',
    }));

    this.initButtons(state);

    var gccdump_picker = this.domRoot[0].querySelector('.gccdump-pass-picker');
    this.selectize = new TomSelect(gccdump_picker, {
        sortField: -1, // do not sort
        valueField: 'name',
        labelField: 'name',
        searchField: ['name'],
        options: [],
        items: [],
        plugins: ['input_autogrow'],
        maxOptions: 500,
    });

    // this is used to save internal state.
    this.state = {};

    this.state._compilerid = state._compilerid;
    this.state._editorid = state._editorid;
    this._compilerName = state._compilerName;

    this.awaitingInitialResults = false;
    this.selection = state.selection;

    this.initCallbacks();

    if (state && state.selectedPass) {
        this.state.selectedPass = state.selectedPass;

        // To keep URL format stable wrt GccDump, only a string of the form 'r.expand' is stored.
        // Old links also have the pass number prefixed but this can be ignored.
        // Create the object that will be used instead of this bare string.
        var selectedPassRe = /[0-9]*(i|t|r)\.([\w-_]*)/;
        var passType = {
            i: 'ipa',
            r: 'rtl',
            t: 'tree',
        };
        var match = state.selectedPass.match(selectedPassRe);
        var selectedPassO = {
            filename_suffix: match[1] + '.' + match[2],
            name: match[2] + ' (' + passType[match[1]] + ')',
            command_prefix: '-fdump-' + passType[match[1]] + '-' + match[2],
        };

        this.eventHub.emit('gccDumpPassSelected', this.state._compilerid, selectedPassO, false);
    }

    // until we get our first result from compilation backend with all fields,
    // disable UI callbacks.
    this.uiIsReady = false;
    this.onUiNotReady();

    this.eventHub.emit('gccDumpFiltersChanged', this.state._compilerid, this.getEffectiveFilters(), false);

    this.updateButtons();
    this.saveState();
    this.updateTitle();

    // UI is ready, request compilation to get passes list and
    // current output (if any)
    this.eventHub.emit('gccDumpUIInit', this.state._compilerid);
    ga.proxy('send', {
        hitType: 'event',
        eventCategory: 'OpenViewPane',
        eventAction: 'GccDump',
    });
}

GccDump.prototype.initButtons = function (state) {
    this.filters = new Toggles(this.domRoot.find('.dump-filters'), state);
    this.fontScale = new FontScale(this.domRoot, state, this.gccDumpEditor);

    this.topBar = this.domRoot.find('.top-bar');
    this.dumpFiltersButtons = this.domRoot.find('.dump-filters .btn');

    this.dumpTreesButton = this.domRoot.find('[data-bind=\'treeDump\']');
    this.dumpTreesTitle = this.dumpTreesButton.prop('title');

    this.dumpRtlButton = this.domRoot.find('[data-bind=\'rtlDump\']');
    this.dumpRtlTitle = this.dumpRtlButton.prop('title');

    this.dumpIpaButton = this.domRoot.find('[data-bind=\'ipaDump\']');
    this.dumpIpaTitle = this.dumpIpaButton.prop('title');

    this.optionAddressButton = this.domRoot.find('[data-bind=\'addressOption\']');
    this.optionAddressTitle = this.optionAddressButton.prop('title');

    this.optionSlimButton = this.domRoot.find('[data-bind=\'slimOption\']');
    this.optionSlimTitle = this.optionSlimButton.prop('title');

    this.optionRawButton = this.domRoot.find('[data-bind=\'rawOption\']');
    this.optionRawTitle = this.optionRawButton.prop('title');

    this.optionDetailsButton = this.domRoot.find('[data-bind=\'detailsOption\']');
    this.optionDetailsTitle = this.optionDetailsButton.prop('title');

    this.optionStatsButton = this.domRoot.find('[data-bind=\'statsOption\']');
    this.optionStatsTitle = this.optionStatsButton.prop('title');

    this.optionBlocksButton = this.domRoot.find('[data-bind=\'blocksOption\']');
    this.optionBlocksTitle = this.optionBlocksButton.prop('title');

    this.optionVopsButton = this.domRoot.find('[data-bind=\'vopsOption\']');
    this.optionVopsTitle = this.optionVopsButton.prop('title');

    this.optionLinenoButton = this.domRoot.find('[data-bind=\'linenoOption\']');
    this.optionLinenoTitle = this.optionLinenoButton.prop('title');

    this.optionUidButton = this.domRoot.find('[data-bind=\'uidOption\']');
    this.optionUidTitle = this.optionUidButton.prop('title');

    this.optionAllButton = this.domRoot.find('[data-bind=\'allOption\']');
    this.optionAllTitle = this.optionAllButton.prop('title');

    this.hideable = this.domRoot.find('.hideable');
};

GccDump.prototype.initCallbacks = function () {
    this.filters.on('change', _.bind(this.onFilterChange, this));
    this.selectize.on('change', _.bind(this.onPassSelect, this));

    this.fontScale.on('change', _.bind(this.saveState, this));

    this.eventHub.on('compileResult', this.onCompileResult, this);
    this.eventHub.on('compiler', this.onCompiler, this);
    this.eventHub.on('compilerClose', this.onCompilerClose, this);
    this.eventHub.on('settingsChange', this.onSettingsChange, this);

    this.eventHub.emit('gccDumpViewOpened', this.state._compilerid);
    this.eventHub.emit('requestSettings');
    this.container.on('destroy', this.close, this);

    this.container.on('resize', this.resize, this);
    this.container.on('shown', this.resize, this);
    PaneRenaming.registerCallback(this);

    this.cursorSelectionThrottledFunction =
        _.throttle(_.bind(this.onDidChangeCursorSelection, this), 500);
    this.gccDumpEditor.onDidChangeCursorSelection(_.bind(function (e) {
        this.cursorSelectionThrottledFunction(e);
    }, this));
};

GccDump.prototype.updateButtons = function () {
    var formatButtonTitle = function (button, title) {
        button.prop('title', '[' + (button.hasClass('active') ? 'ON' : 'OFF') + '] ' + title);
    };
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
};

// Disable view's menu when invalid compiler has been
// selected after view is opened.
GccDump.prototype.onUiNotReady = function () {
    // disable drop down menu and buttons
    this.selectize.disable();
    this.dumpFiltersButtons.prop('disabled', true);
};

GccDump.prototype.onUiReady = function () {
    // enable drop down menu and buttons
    this.selectize.enable();

    this.dumpFiltersButtons.prop('disabled', false);
};

GccDump.prototype.onPassSelect = function (passId) {
    var selectedPass = this.selectize.options[passId];

    if (this.inhibitPassSelect !== true) {
        this.eventHub.emit('gccDumpPassSelected', this.state._compilerid, selectedPass, true);
    }

    // To keep shared URL compatible, we keep on storing only a string in the
    // state and stick to the original format.
    // Previously, we were simply storing the full file suffix (the part after [...]):
    //    [file.c.]123t.expand
    // We don't have the number now, but we can store the file suffix without this number
    // (the number is useless and should probably have never been there in the
    // first place).

    this.state.selectedPass = selectedPass.filename_suffix;
    this.saveState();
};

GccDump.prototype.resize = function () {
    var topBarHeight = utils.updateAndCalcTopBarHeight(this.domRoot, this.topBar, this.hideable);
    this.gccDumpEditor.layout({
        width: this.domRoot.width(),
        height: this.domRoot.height() - topBarHeight,
    });
};

// Called after result from new compilation received
// if gccDumpOutput is false, cleans the select menu
GccDump.prototype.updatePass = function (filters, selectize, gccDumpOutput) {
    var passes = gccDumpOutput ? gccDumpOutput.all : [];

    // we are changing selectize but don't want any callback to
    // trigger new compilation
    this.inhibitPassSelect = true;

    selectize.clear(true);
    selectize.clearOptions(true);

    _.each(passes, function (p) {
        selectize.addOption(p);
    }, this);

    if (gccDumpOutput.selectedPass)
        selectize.addItem(gccDumpOutput.selectedPass.name, true);
    else
        selectize.clear(true);

    this.eventHub.emit('gccDumpPassSelected', this.state._compilerid, gccDumpOutput.selectedPass, false);

    this.inhibitPassSelect = false;
};

GccDump.prototype.onCompileResult = function (id, compiler, result) {
    if (this.state._compilerid !== id || !compiler) return;

    if (result.gccDumpOutput && result.gccDumpOutput.syntaxHighlight) {
        monaco.editor.setModelLanguage(this.gccDumpEditor.getModel(), 'gccdump-rtl-gimple');
    } else {
        monaco.editor.setModelLanguage(this.gccDumpEditor.getModel(), 'plaintext');
    }
    if (compiler.supportsGccDump && result.gccDumpOutput) {
        var currOutput = result.gccDumpOutput.currentPassOutput;

        // if result contains empty selected pass, probably means
        // we requested an invalid/outdated pass.
        if (!result.gccDumpOutput.selectedPass) {
            this.selectize.clear(true);
            this.state.selectedPass = null;
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
        this.state.selectedPass = null;
        this.updatePass(this.filters, this.selectize, false);
        this.uiIsReady = false;
        this.onUiNotReady();
        if (!compiler.supportsGccDump) {
            this.showGccDumpResults('<Tree/RTL output is not supported for this compiler (GCC only)>');
        } else {
            this.showGccDumpResults('<Tree/RTL output is empty>');
        }
    }
    this.saveState();
};

GccDump.prototype.getPaneName = function () {
    return 'GCC Tree/RTL Viewer ' + (this._compilerName || '') +
        ' (Editor #' + this.state._editorid + ', Compiler #' + this.state._compilerid + ')';
};

GccDump.prototype.updateTitle = function () {
    var name = this.paneName ? this.paneName : this.getPaneName();
    this.container.setTitle(_.escape(name));
};

GccDump.prototype.showGccDumpResults = function (results) {
    this.gccDumpEditor.setValue(results);

    if (!this.awaitingInitialResults) {
        if (this.selection) {
            this.gccDumpEditor.setSelection(this.selection);
            this.gccDumpEditor.revealLinesInCenter(this.selection.startLineNumber,
                this.selection.endLineNumber);
        }
        this.awaitingInitialResults = true;
    }
};

GccDump.prototype.onCompiler = function (id, compiler, options, editorid) {
    if (id === this.state._compilerid) {
        this._compilerName = compiler ? compiler.name : '';
        this.state._editorid = editorid;
        this.updateTitle();
    }
};

GccDump.prototype.onCompilerClose = function (id) {
    if (id === this.state._compilerid) {
        // We can't immediately close as an outer loop somewhere in GoldenLayout is iterating over
        // the hierarchy. We can't modify while it's being iterated over.
        this.close();
        _.defer(function (self) {
            self.container.close();
        }, this);
    }
};

GccDump.prototype.getEffectiveFilters = function () {
    return this.filters.get();
};

GccDump.prototype.onFilterChange = function () {
    this.saveState();
    this.updateButtons();

    if (this.inhibitPassSelect !== true) {
        this.eventHub.emit('gccDumpFiltersChanged', this.state._compilerid, this.getEffectiveFilters(), true);
    }
};

GccDump.prototype.saveState = function () {
    var state = this.currentState();
    this.container.setState(state);
    this.fontScale.addState(state);
};

GccDump.prototype.currentState = function () {
    var filters = this.getEffectiveFilters();
    return {
        _compilerid: this.state._compilerid,
        _editorid: this.state._editorid,
        selectedPass: this.state.selectedPass,
        treeDump: filters.treeDump,
        rtlDump: filters.rtlDump,
        ipaDump: filters.ipaDump,
        addressOption: filters.addressOption,
        slimOption: filters.slimOption,
        rawOption: filters.rawOption,
        detailsOption: filters.detailsOption,
        statsOption: filters.statsOption,
        blocksOption: filters.blocksOption,
        vopsOption: filters.vopsOption,
        linenoOption: filters.linenoOption,
        uidOption: filters.uidOption,
        allOption: filters.allOption,
        selection: this.selection,
    };
};

GccDump.prototype.onSettingsChange = function (newSettings) {
    this.gccDumpEditor.updateOptions({
        contextmenu: newSettings.useCustomContextMenu,
        minimap: {
            enabled: newSettings.showMinimap,
        },
        fontFamily: newSettings.editorsFFont,
        fontLigatures: newSettings.editorsFLigatures,
    });
};

GccDump.prototype.onDidChangeCursorSelection = function (e) {
    if (this.awaitingInitialResults) {
        this.selection = e.selection;
        this.saveState();
    }
};

GccDump.prototype.close = function () {
    this.eventHub.unsubscribe();
    this.eventHub.emit('gccDumpViewClosed', this.state._compilerid);
    this.gccDumpEditor.dispose();
};

module.exports = {
    GccDump: GccDump,
};
