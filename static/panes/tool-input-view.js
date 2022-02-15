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

var FontScale = require('../fontscale').FontScale;
var monaco = require('monaco-editor');
var _ = require('underscore');
var $ = require('jquery');
var ga = require('../analytics').ga;
var monacoConfig = require('../monaco-config');
var Settings = require('../settings').Settings;
var PaneRenaming = require('../pane-renaming').PaneRenaming;

require('../modes/asm-mode');

function ToolInputView(hub, container, state) {
    state = state || {};
    this.container = container;
    this.eventHub = hub.createEventHub();
    this.domRoot = container.getElement();
    this.domRoot.html($('#tool-input').html());
    this.source = state.source || '';
    var root = this.domRoot.find('.monaco-placeholder');

    this.settings = Settings.getStoredSettings();

    this.editor = monaco.editor.create(root[0], monacoConfig.extendConfig({
        value: '',
        language: 'plaintext',
        readOnly: false,
        glyphMargin: true,
    }));

    this._toolId = state.toolId;
    this._toolName = state.toolName;
    this._compilerId = state.compilerId;
    this.selection = state.selection || '';
    this.shouldSetSelectionInitially = !!this.selection;

    this.initButtons(state);
    this.initCallbacks();

    this.updateTitle();
    this.onSettingsChange(this.settings);
    this.eventHub.emit('toolInputViewOpened', this._toolid);
    ga.proxy('send', {
        hitType: 'event',
        eventCategory: 'OpenViewPane',
        eventAction: 'toolInputView',
    });
}

ToolInputView.prototype.initButtons = function (state) {
    this.fontScale = new FontScale(this.domRoot, state, this.editor);

    this.topBar = this.domRoot.find('.top-bar');
};

ToolInputView.prototype.initCallbacks = function () {
    this.fontScale.on('change', _.bind(this.updateState, this));

    this.eventHub.on('compilerClose', this.onCompilerClose, this);
    this.eventHub.on('toolClosed', this.onToolClose, this);
    this.eventHub.on('toolInputViewCloseRequest', this.onToolInputViewCloseRequest, this);
    this.eventHub.on('settingsChange', this.onSettingsChange, this);
    this.eventHub.on('setToolInput', this.onSetToolInput, this);

    this.container.on('resize', this.resize, this);
    this.container.on('shown', this.resize, this);
    this.container.on('destroy', this.close, this);
    PaneRenaming.registerCallback(this);

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

ToolInputView.prototype.getPaneName =  function () {
    return 'Tool Input ' + this._toolName + ' (Compiler #' + this._compilerId + ')';
};

ToolInputView.prototype.updateTitle = function () {
    var name = this.paneName ? this.paneName : this.getPaneName();
    this.container.setTitle(_.escape(name));
};

ToolInputView.prototype.resize = function () {
    var topBarHeight = this.topBar.outerHeight(true);
    this.editor.layout({
        width: this.domRoot.width(),
        height: this.domRoot.height() - topBarHeight,
    });
};

ToolInputView.prototype.updateState = function () {
    this.container.setState(this.currentState());
};

ToolInputView.prototype.currentState = function () {
    var state = {
        toolId: this._toolId,
        toolName: this._toolName,
        compilerId: this._compilerId,
        selection: this.selection,
    };
    this.fontScale.addState(state);
    return state;
};

ToolInputView.prototype.close = function () {
    this.eventHub.unsubscribe();
    this.eventHub.emit('toolInputViewClosed', this._compilerId, this._toolId, this.getInput());
    this.editor.dispose();
};

ToolInputView.prototype.onToolClose = function (compilerId, toolSettings) {
    if (this._compilerId === compilerId && this._toolId === toolSettings.toolId) {
        // We can't immediately close as an outer loop somewhere in GoldenLayout is iterating over
        // the hierarchy. We can't modify while it's being iterated over.
        this.close();
        _.defer(function (self) {
            self.container.close();
        }, this);
    }
};

ToolInputView.prototype.onToolInputViewCloseRequest = function (compilerId, toolId) {
    if (this._compilerId === compilerId && this._toolId === toolId) {
        this.close();
        _.defer(function (self) {
            self.container.close();
        }, this);
    }
};

ToolInputView.prototype.onCompilerClose = function (id) {
    if (id === this._compilerId) {
        // We can't immediately close as an outer loop somewhere in GoldenLayout is iterating over
        // the hierarchy. We can't modify while it's being iterated over.
        this.close();
        _.defer(function (self) {
            self.container.close();
        }, this);
    }
};

ToolInputView.prototype.onSettingsChange = function (newSettings) {
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

ToolInputView.prototype.onDidChangeCursorSelection = function (e) {
    // On initialization this callback fires with the default selection
    // overwriting any selection from state. If we are awaiting initial
    // selection setting then don't update our selection.
    if (!this.shouldSetSelectionInitially) {
        this.selection = e.selection;
        this.updateState();
    }
};

ToolInputView.prototype.onSetToolInput = function (compilerId, toolId, value) {
    if (this._compilerId === compilerId && this._toolId === toolId) {
        var ret = this.editor.getModel().setValue(value);
        if (this.shouldSetSelectionInitially && this.selection) {
            this.editor.setSelection(this.selection);
            this.editor.revealLinesInCenter(
                this.selection.startLineNumber, this.selection.endLineNumber);
            this.shouldSetSelectionInitially = false;
        }
        return ret;
    }
};

ToolInputView.prototype.getInput = function () {
    if (!this.editor.getModel()) {
        return '';
    }
    return this.editor.getModel().getValue();
};

ToolInputView.prototype.maybeEmitChange = function (force) {
    var input = this.getInput();
    if (!force && input === this.lastChangeEmitted) return;

    this.lastChangeEmitted = input;
    this.eventHub.emit('toolInputChange', this._compilerId, this._toolId, this.lastChangeEmitted);
};

module.exports = {
    ToolInputView: ToolInputView,
};
