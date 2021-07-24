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

var FontScale = require('../fontscale');
var ga = require('../analytics');
var _ = require('underscore');
var $ = require('jquery');
var monaco = require('monaco-editor');
var monacoConfig = require('../monaco-config');

function RustMir(hub, container, state) {
    this.container = container;
    this.eventHub = hub.createEventHub();
    this.domRoot = container.getElement();
    this.domRoot.html($('#rustmir').html());

    var root = this.domRoot.find('.monaco-placeholder');
    this.rustMirEditor = monaco.editor.create(root[0], monacoConfig.extendConfig({
        language: 'rust',
        readOnly: true,
        glyphMargin: true,
        lineNumbersMinChars: 3,
    }));

    this._compilerid = state.id;
    this._compilerName = state.compilerName;
    this._editorid = state.editorid;

    this.awaitingInitialResults = false;
    this.selection = state.selection;

    this.settings = {};
    this.colours = [];
    this.irCode = [];

    this.initButtons(state);
    this.initCallbacks();

    if (state && state.rustMirOutput) {
        this.showRustMirResults(state.rustMirOutput);
    }
    this.setTitle();

    ga.proxy('send', {
        hitType: 'event',
        eventCategory: 'OpenViewPane',
        eventAction: 'RustMir',
    });
}

RustMir.prototype.initButtons = function (state) {
    this.fontScale = new FontScale(this.domRoot, state, this.rustMirEditor);

    this.topBar = this.domRoot.find('.top-bar');
};

RustMir.prototype.initCallbacks = function () {
    this.fontScale.on('change', this.updateState, this);
    this.container.on('destroy', this.close, this);
    this.container.on('resize', this.resize, this);

    this.eventHub.on('compileResult', this.onCompileResult, this);
    this.eventHub.on('compiler', this.onCompiler, this);
    this.eventHub.on('compilerClose', this.onCompilerClose, this);
    this.eventHub.on('settingsChange', this.onSettingsChange, this);

    this.eventHub.emit('rustMirViewOpened', this._compilerid);
    this.eventHub.emit('requestSettings');

    this.eventHub.on('shown', this.resize, this);
    this.eventHub.on('resize', this.resize, this);

    this.cursorSelectionThrottledFunction =
        _.throttle(_.bind(this.onDidChangeCursorSelection, this), 500);
    this.rustMirEditor.onDidChangeCursorSelection(_.bind(function (e) {
        this.cursorSelectionThrottledFunction(e);
    }, this));
};

// TODO: de-dupe with compiler etc
RustMir.prototype.resize = function () {
    var topBarHeight = this.topBar.outerHeight(true);
    this.rustMirEditor.layout({
        width: this.domRoot.width(),
        height: this.domRoot.height() - topBarHeight,
    });
};

RustMir.prototype.onCompileResult = function (id, compiler, result) {
    if (this._compilerid !== id) return;
    if (result.hasRustMirOutput) {
        this.showRustMirResults(result.rustMirOutput);
    } else if (compiler.supportsRustMirView) {
        this.showRustMirResults([{text: '<No output>'}]);
    }
};

RustMir.prototype.getCurrentEditorLanguage = function() {
    return this.rustMirEditor.getModel().getModeId();
};

RustMir.prototype.getPaneName = function () {
    return this._compilerName + ' Rust MIR Viewer (Editor #' + this._editorid + ', Compiler # '
        + this._compilerid + ')';
};

RustMir.prototype.setTitle = function () {
    this.container.setTitle(this.getPaneName());
};

RustMir.prototype.showRustMirResults = function (rustMirCode) {
    if (!this.rustMirEditor) return;
    this.rustMirEditor.getModel().setValue(rustMirCode.length
        ? _.pluck(rustMirCode, 'text').join('\n')
        : '<No Rust MIR generated>');

    if (!this.awaitingInitialResults) {
        if (this.selection) {
            this.rustMirEditor.setSelection(this.selection);
            this.rustMirEditor.revealLinesInCenter(this.selection.startLineNumber,
                this.selection.endLineNumber);
        }
        this.awaitingInitialResults = true;
    }
};

RustMir.prototype.onCompiler = function (id, compiler, options, editorid) {
    if (this._compilerid === id) {
        this._compilerName = compiler ? compiler.name : '';
        this._editorid = editorid;
        this.setTitle();
        if (compiler && !compiler.supportsRustMirView) {
            this.rustMirEditor.setValue('<Rust MIR output is not supported for this compiler>');
        }
    }
};

RustMir.prototype.onCompilerClose = function (id) {
    if (this._compilerid === id) {
        _.defer(function (self) {
            self.container.close();
        }, this);
    }
};

RustMir.prototype.updateState = function () {
    this.container.setState(this.currentState());
};

RustMir.prototype.currentState = function () {
    var state = {
        id: this._compilerid,
        editorid: this._editorid,
        selection: this.selection,
    };
    this.fontScale.addState(state);
    return state;
};

RustMir.prototype.onDidChangeCursorSelection = function (e) {
    if (this.awaitingInitialResults) {
        this.selection = e.selection;
        this.updateState();
    }
};

RustMir.prototype.onSettingsChange = function (newSettings) {
    this.settings = newSettings;
    this.rustMirEditor.updateOptions({
        contextmenu: newSettings.useCustomContextMenu,
        minimap: {
            enabled: newSettings.showMinimap,
        },
        fontFamily: newSettings.editorFFont,
        fontLigatures: newSettings.editorsFLigatures,
    });
};

RustMir.prototype.close = function () {
    this.eventHub.unsubscribe();
    this.eventHub.emit('rustMirViewClosed', this._compilerid);
    this.rustMirEditor.dispose();
};

module.exports = {
    RustMir: RustMir,
};
