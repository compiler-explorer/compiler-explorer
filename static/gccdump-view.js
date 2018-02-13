// Copyright (c) 2012-2017, Simon Brand
// Copyright (c) 2017, Marc Poulhiès - Kalray Inc.
//
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

var FontScale = require('fontscale');
var monaco = require('monaco');
var Toggles = require('toggles');
require('gccdump-rtl-gimple-mode');
var _ = require('underscore');
var $ = require('jquery');

require('selectize');

function GccDump(hub, container, state) {
    this.container = container;
    this.eventHub = hub.createEventHub();
    this.domRoot = container.getElement();
    this.domRoot.html($('#gccdump').html());
    this.filters = new Toggles(this.domRoot.find('.dump-filters'), state);

    this._currentDecorations = [];

    // until we get our first result from compilation backend with all fields,
    // disable UI callbacks.
    this.uiIsReady = false;
    // disable filter buttons
    this.domRoot.find('.dump-filters .btn')
        .each(function () {
            $(this).addClass('disabled');
        });

    this.gccDumpEditor = monaco.editor.create(this.domRoot.find('.monaco-placeholder')[0], {
        value: '',
        scrollBeyondLastLine: false,
        readOnly: true,
        glyphMargin: true,
        quickSuggestions: false,
        fixedOverflowWidgets: true,
        fontFamily: 'monospace',
        minimap: {
            maxColumn: 80
        },
        lineNumbersMinChars: 3
    });

    var selectize = this.domRoot.find('.gccdump-pass-picker').selectize({
        sortField: 'name',
        valueField: 'name',
        labelField: 'name',
        searchField: ['name'],
        options: [],
        items: []
    });

    this.selectize = selectize[0].selectize;
    this.selectize.disable();

    // this is used to save internal state.
    this.state = {};

    this.state._compilerid = state._compilerid;
    this.state._editorid = state._editorid;
    this._compilerName = state._compilerName;

    this.fontScale = new FontScale(this.domRoot, state, this.gccDumpEditor);
    this.fontScale.on('change', _.bind(this.saveState, this));

    this.eventHub.on('compileResult', this.onCompileResult, this);
    this.eventHub.on('compiler', this.onCompiler, this);
    this.eventHub.on('compilerClose', this.onCompilerClose, this);
    this.eventHub.on('settingsChange', this.onSettingsChange, this);

    this.eventHub.emit('gccDumpViewOpened', this.state._compilerid);
    this.eventHub.emit('requestSettings');
    this.container.on('destroy', this.close, this);

    container.on('resize', this.resize, this);
    container.on('shown', this.resize, this);

    if (state && state.selectedPass) {
        this.state.selectedPass = state.selectedPass;
        this.eventHub.emit('gccDumpPassSelected',
            this.state._compilerid,
            state.selectedPass,
            false);
    }

    this.eventHub.emit('gccDumpFiltersChanged',
        this.state._compilerid,
        this.getEffectiveFilters(),
        false);

    this.saveState();
    this.setTitle();

    // UI is ready, request compilation to get passes list and
    // current output (if any)
    this.eventHub.emit('gccDumpUIInit', this.state._compilerid);
}

// Disable view's menu when invalid compiler has been
// selected after view is opened.
GccDump.prototype.onUiNotReady = function () {
    this.filters.off('change');
    this.selectize.off('change');

    // disable drop down menu and buttons
    this.selectize.disable();
    this.domRoot.find('.dump-filters .btn')
        .each(function () {
            $(this).addClass('disabled');
        });
};

GccDump.prototype.onUiReady = function () {
    this.filters.on('change', _.bind(this.onFilterChange, this));
    this.selectize.on('change', _.bind(this.onPassSelect, this));

    // enable drop down menu and buttons
    this.selectize.enable();
    this.domRoot.find('.dump-filters .btn')
        .each(function () {
            $(this).removeClass('disabled');
        });
};

GccDump.prototype.onPassSelect = function (passId) {
    if (this.inhibitPassSelect !== true) {
        this.eventHub.emit('gccDumpPassSelected',
            this.state._compilerid,
            passId,
            true);
    }
    this.state.selectedPass = passId;
    this.saveState();
};

// TODO: de-dupe with compiler etc
GccDump.prototype.resize = function () {
    var topBarHeight = this.domRoot.find('.top-bar').outerHeight(true);
    this.gccDumpEditor.layout({
        width: this.domRoot.width(),
        height: this.domRoot.height() - topBarHeight
    });
};

// Called after result from new compilation received
// if gccDumpOutput is false, cleans the select menu
GccDump.prototype.updatePass = function (filters, selectize, gccDumpOutput) {
    var passes = gccDumpOutput ? gccDumpOutput.all : [];

    // we are changing selectize but don't want any callback to
    // trigger new compilation
    this.inhibitPassSelect = true;

    _.each(selectize.options, function (p) {
        if (passes.indexOf(p.name) === -1) {
            selectize.removeOption(p.name);
        }
    }, this);

    _.each(passes, function (p) {
        selectize.addOption({
            name: p
        });
    }, this);

    if (gccDumpOutput.selectedPass && gccDumpOutput.selectedPass !== '') {
        selectize.addItem(gccDumpOutput.selectedPass, true);
    } else {
        selectize.clear(true);
    }

    this.eventHub.emit('gccDumpPassSelected',
        this.state._compilerid,
        gccDumpOutput.selectedPass,
        false);

    this.inhibitPassSelect = false;
};

GccDump.prototype.onCompileResult = function (id, compiler, result) {
    if (this.state._compilerid !== id) return;

    if (result.hasGccDumpOutput && result.gccDumpOutput.syntaxHighlight) {
        monaco.editor.setModelLanguage(this.gccDumpEditor.getModel(), 'gccdump-rtl-gimple');
    } else {
        monaco.editor.setModelLanguage(this.gccDumpEditor.getModel(), null);
    }

    if (result.hasGccDumpOutput) {
        var currOutput = result.gccDumpOutput.currentPassOutput;

        // if result contains empty selected pass, probably means
        // we requested an invalid/outdated pass.
        if (result.gccDumpOutput.selectedPass === '') {
            this.selectize.clear(true);
            this.state.selectedPass = '';
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
        this.state.selectedPass = '';
        this.updatePass(this.filters, this.selectize, false);
        this.showGccDumpResults('<No output, this view is only supported on GCC compilers>');
        this.uiIsReady = false;
        this.onUiNotReady();
    }

    this.saveState();
};

GccDump.prototype.setTitle = function () {
    this.container.setTitle((this._compilerName || '') +
        ' GCC Tree/RTL Viewer (Editor #' + this.state._editorid + ', Compiler #' + this.state._compilerid + ')');
};

GccDump.prototype.showGccDumpResults = function (results) {
    this.gccDumpEditor.setValue(results);
};

GccDump.prototype.onCompiler = function (id, compiler, options, editorid) {
    if (id === this.state._compilerid) {
        this._compilerName = compiler ? compiler.name : '';
        this.state._editorid = editorid;
        this.setTitle();
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

    if (this.inhibitPassSelect !== true) {
        this.eventHub.emit('gccDumpFiltersChanged',
            this.state._compilerid,
            this.getEffectiveFilters(),
            true);
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
        rtlDump: filters.rtlDump
    };
};

GccDump.prototype.onSettingsChange = function (newSettings) {
    this.gccDumpEditor.updateOptions({
        minimap: {
            enabled: newSettings.showMinimap
        }
    });
};

GccDump.prototype.close = function () {
    this.eventHub.unsubscribe();
    this.eventHub.emit('gccDumpViewClosed', this.state._compilerid);
    this.gccDumpEditor.dispose();
};

module.exports = {
    GccDump: GccDump
};
