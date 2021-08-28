// Copyright (c) 2021, Tom Ritter
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

var FontScale = require('../fontscale');
var monaco = require('monaco-editor');
var _ = require('underscore');
var $ = require('jquery');
var ga = require('../analytics');
var monacoConfig = require('../monaco-config');
var local = require('../local');

require('../modes/asm-mode');

function Flags(hub, container, state) {
    state = state || {};
    this.container = container;
    this.eventHub = hub.createEventHub();
    this.domRoot = container.getElement();
    this.domRoot.html($('#flags').html());
    this.source = state.source || '';
    var root = this.domRoot.find('.monaco-placeholder');

    this.settings = JSON.parse(local.get('settings', '{}'));

    var value = '';
    if (state.compilerFlags) {
        value = state.compilerFlags.replace(/ /g, '\n');
    }
    this.editor = monaco.editor.create(root[0], monacoConfig.extendConfig({
        value: value,
        language: 'plaintext',
        readOnly: false,
        glyphMargin: true,
    }));

    this._compilerid = state.id;
    this._compilerName = state.compilerName;

    this.awaitingInitialResults = false;
    this.selection = state.selection;

    this.initButtons(state);
    this.initCallbacks();

    this.setTitle();
    this.onSettingsChange(this.settings);
    this.eventHub.emit('flagsViewOpened', this._compilerid);
    ga.proxy('send', {
        hitType: 'event',
        eventCategory: 'OpenViewPane',
        eventAction: 'detailedCompilerFlags',
    });
}

Flags.prototype.initButtons = function (state) {
    this.fontScale = new FontScale(this.domRoot, state, this.editor);

    this.topBar = this.domRoot.find('.top-bar');
};

Flags.prototype.initCallbacks = function () {
    this.fontScale.on('change', _.bind(this.updateState, this));

    this.eventHub.on('compiler', this.onCompiler, this);
    this.eventHub.on('compilerClose', this.onCompilerClose, this);
    this.eventHub.on('settingsChange', this.onSettingsChange, this);
    this.eventHub.on('resize', this.resize, this);
    this.container.on('destroy', this.close, this);
    this.eventHub.emit('requestSettings');
    this.eventHub.emit('findCompilers');

    this.container.on('resize', this.resize, this);
    this.container.on('shown', this.resize, this);

    this.container.layoutManager.on('initialised', function () {
        // Once initialized, let everyone know what text we have.
        this.maybeEmitChange();
    }, this);
    this.eventHub.on('initialised', this.maybeEmitChange, this);

    this.editor.getModel().onDidChangeContent(_.bind(function () {
        this.debouncedEmitChange();
        this.updateState();
    }, this));

    this.cursorSelectionThrottledFunction =
        _.throttle(_.bind(this.onDidChangeCursorSelection, this), 500);
    this.editor.onDidChangeCursorSelection(_.bind(function (e) {
        this.cursorSelectionThrottledFunction(e);
    }, this));
};

Flags.prototype.getPaneName = function () {
    return 'Detailed Compiler Flags ' + this._compilerName + ' (Compiler #' + this._compilerid + ')';
};

Flags.prototype.setTitle = function () {
    this.container.setTitle(this.getPaneName());
};

Flags.prototype.onCompiler = function (id, compiler) {
    if (id === this._compilerid) {
        this._compilerName = compiler ? compiler.name : '';
        this.setTitle();
    }
};

Flags.prototype.resize = function () {
    var topBarHeight = this.topBar.outerHeight(true);
    this.editor.layout({
        width: this.domRoot.width(),
        height: this.domRoot.height() - topBarHeight,
    });
};

Flags.prototype.updateState = function () {
    this.container.setState(this.currentState());
};

Flags.prototype.currentState = function () {
    var state = {
        id: this._compilerid,
        selection: this.selection,
        compilerFlags: this.getOptions(),
    };
    this.fontScale.addState(state);
    return state;
};

Flags.prototype.close = function () {
    this.eventHub.unsubscribe();
    this.eventHub.emit('flagsViewClosed', this._compilerid, this.getOptions());
    this.editor.dispose();
};

Flags.prototype.onCompilerClose = function (id) {
    if (id === this._compilerid) {
        // We can't immediately close as an outer loop somewhere in GoldenLayout is iterating over
        // the hierarchy. We can't modify while it's being iterated over.
        this.close();
        _.defer(function (self) {
            self.container.close();
        }, this);
    }
};

Flags.prototype.onSettingsChange = function (newSettings) {
    this.editor.updateOptions({
        contextmenu: newSettings.useCustomContextMenu,
        minimap: {
            enabled: newSettings.showMinimap,
        },
        fontFamily: newSettings.editorsFFont,
        fontLigatures: newSettings.editorsFLigatures,
    });

    this.debouncedEmitChange = _.debounce(_.bind(function () {
        this.maybeEmitChange();
    }, this), newSettings.delayAfterChange);
};

Flags.prototype.onDidChangeCursorSelection = function (e) {
    if (this.awaitingInitialResults) {
        this.selection = e.selection;
        this.updateState();
    }
};

Flags.prototype.getOptions = function () {
    var lines = this.editor.getModel().getValue();
    return lines.replace(/\n/g, ' ');
};

Flags.prototype.maybeEmitChange = function (force) {
    var options = this.getOptions();
    if (!force && options === this.lastChangeEmitted) return;

    this.lastChangeEmitted = options;
    this.eventHub.emit('compilerFlagsChange', this._compilerid, this.lastChangeEmitted);
};

module.exports = {
    Flags: Flags,
};
