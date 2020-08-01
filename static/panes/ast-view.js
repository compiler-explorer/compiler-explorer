// Copyright (c) 2017, Simon Brand
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
"use strict";

var FontScale = require('../fontscale');
var monaco = require('monaco-editor');
var _ = require('underscore');
var $ = require('jquery');
var ga = require('../analytics');

function Ast(hub, container, state) {
    this.container = container;
    this.eventHub = hub.createEventHub();
    this.domRoot = container.getElement();
    this.domRoot.html($('#ast').html());
    this.astEditor = monaco.editor.create(this.domRoot.find(".monaco-placeholder")[0], {
        value: "",
        scrollBeyondLastLine: false,
        language: 'plaintext',
        readOnly: true,
        glyphMargin: true,
        fontFamily: 'Consolas, "Liberation Mono", Courier, monospace',
        quickSuggestions: false,
        fixedOverflowWidgets: true,
        minimap: {
            maxColumn: 80
        },
        lineNumbersMinChars: 3
    });

    this._compilerid = state.id;
    this._compilerName = state.compilerName;
    this._editorid = state.editorid;

    this.awaitingInitialResults = false;
    this.selection = state.selection;

    this.initButtons(state);
    this.initCallbacks();

    if (state && state.astOutput) {
        this.showAstResults(state.astOutput);
    }
    this.setTitle();

    ga.proxy('send', {
        hitType: 'event',
        eventCategory: 'OpenViewPane',
        eventAction: 'Ast'
    });
}

Ast.prototype.initButtons = function (state) {
    this.fontScale = new FontScale(this.domRoot, state, this.astEditor);

    this.topBar = this.domRoot.find(".top-bar");
};

Ast.prototype.initCallbacks = function () {
    this.fontScale.on('change', _.bind(this.updateState, this));

    this.container.on('destroy', this.close, this);

    this.eventHub.on('compileResult', this.onCompileResult, this);
    this.eventHub.on('compiler', this.onCompiler, this);
    this.eventHub.on('compilerClose', this.onCompilerClose, this);
    this.eventHub.on('settingsChange', this.onSettingsChange, this);
    this.eventHub.emit('astViewOpened', this._compilerid);
    this.eventHub.emit('requestSettings');

    this.container.on('resize', this.resize, this);
    this.container.on('shown', this.resize, this);

    this.cursorSelectionThrottledFunction =
        _.throttle(_.bind(this.onDidChangeCursorSelection, this), 500);
    this.astEditor.onDidChangeCursorSelection(_.bind(function (e) {
        this.cursorSelectionThrottledFunction(e);
    }, this));
};

// TODO: de-dupe with compiler etc
Ast.prototype.resize = function () {
    var topBarHeight = this.topBar.outerHeight(true);
    this.astEditor.layout({
        width: this.domRoot.width(),
        height: this.domRoot.height() - topBarHeight
    });
};

Ast.prototype.onCompileResult = function (id, compiler, result, lang) {
    if (this._compilerid !== id) return;

    if (result.hasAstOutput) {
        this.showAstResults(result.astOutput);
    }
    else if (compiler.supportsAstView) {
        this.showAstResults("<No output>");
    }

    if (lang && lang.monaco && this.getCurrentEditorLanguage() !== lang.monaco) {
        monaco.editor.setModelLanguage(this.astEditor.getModel(), lang.monaco);
    }
};

// Monaco language id of the current editor
Ast.prototype.getCurrentEditorLanguage = function () {
    return this.astEditor.getModel().getModeId();
};

Ast.prototype.setTitle = function () {
    this.container.setTitle(
        this._compilerName + " Ast Viewer (Editor #" + this._editorid + ", Compiler #" + this._compilerid + ")"
    );
};

Ast.prototype.getDisplayableAst = function (astResult) {
    return "**" + astResult.astType + "** - " + astResult.displayString;
};

Ast.prototype.showAstResults = function (results) {
    this.astEditor.setValue(results);

    if (!this.awaitingInitialResults) {
        if (this.selection) {
            this.astEditor.setSelection(this.selection);
            this.astEditor.revealLinesInCenter(this.selection.startLineNumber,
                this.selection.endLineNumber);
        }
        this.awaitingInitialResults = true;
    }
};

Ast.prototype.onCompiler = function (id, compiler, options, editorid) {
    if (id === this._compilerid) {
        this._compilerName = compiler ? compiler.name : '';
        this._editorid = editorid;
        this.setTitle();
        if (compiler && !compiler.supportsAstView) {
            this.astEditor.setValue("<AST output is not supported for this compiler>");
        }
    }
};

Ast.prototype.onCompilerClose = function (id) {
    if (id === this._compilerid) {
        // We can't immediately close as an outer loop somewhere in GoldenLayout is iterating over
        // the hierarchy. We can't modify while it's being iterated over.
        _.defer(function (self) {
            self.container.close();
        }, this);
    }
};

Ast.prototype.updateState = function () {
    this.container.setState(this.currentState());
};

Ast.prototype.currentState = function () {
    var state = {
        id: this._compilerid,
        editorid: this._editorid,
        selection: this.selection
    };
    this.fontScale.addState(state);
    return state;
};

Ast.prototype.onCompilerClose = function (id) {
    if (id === this._compilerid) {
        // We can't immediately close as an outer loop somewhere in GoldenLayout is iterating over
        // the hierarchy. We can't modify while it's being iterated over.
        this.close();
        _.defer(function (self) {
            self.container.close();
        }, this);
    }
};

Ast.prototype.onSettingsChange = function (newSettings) {
    this.astEditor.updateOptions({
        contextmenu: newSettings.useCustomContextMenu,
        minimap: {
            enabled: newSettings.showMinimap
        },
        fontFamily: newSettings.editorsFFont,
        fontLigatures: newSettings.editorsFLigatures
    });
};

Ast.prototype.onDidChangeCursorSelection = function (e) {
    if (this.awaitingInitialResults) {
        this.selection = e.selection;
        this.updateState();
    }
};

Ast.prototype.close = function () {
    this.eventHub.unsubscribe();
    this.eventHub.emit("astViewClosed", this._compilerid);
    this.astEditor.dispose();
};

module.exports = {
    Ast: Ast
};
