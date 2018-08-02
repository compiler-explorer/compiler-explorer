// Copyright (c) 2018, Compiler Explorer Authors
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
var monaco = require('../monaco');
var _ = require('underscore');
var $ = require('jquery');
var ga = require('../analytics');
var colour = require('../colour');

function Ir(hub, container, state) {
    this.container = container;
    this.eventHub = hub.createEventHub();
    this.domRoot = container.getElement();
    this.domRoot.html($('#ir').html());
    this._currentDecorations = [];
    this.irEditor = monaco.editor.create(this.domRoot.find(".monaco-placeholder")[0], {
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

    this.colours = [];
    this.ir = [];

    this.initEditorActions();
    this.initButtons(state);
    this.initCallbacks();

    if (state && state.irOutput) {
        this.showIrResults(state.irOutput);
    }
    this.setTitle();

    ga.proxy('send', {
        hitType: 'event',
        eventCategory: 'OpenViewPane',
        eventAction: 'Ir'
    });
}

Ir.prototype.initEditorActions = function () {
    this.irEditor.addAction({
        id: 'viewsource',
        label: 'Scroll to source',
        keybindings: [monaco.KeyMod.CtrlCmd | monaco.KeyCode.F10],
        keybindingContext: null,
        contextMenuGroupId: 'navigation',
        contextMenuOrder: 1.5,
        run: _.bind(function (ed) {
            var desiredLine = ed.getPosition().lineNumber - 1;
            var source = this.ir[desiredLine].source;
            if (source.file === null) {
                // a null file means it was the user's source
                this.eventHub.emit('editorSetDecoration', this.sourceEditorId, source.line, true);
            } else {
                // TODO: some indication this asm statement came from elsewhere
                this.eventHub.emit('editorSetDecoration', this.sourceEditorId, -1, false);
            }
        }, this)
    });
};

Ir.prototype.initButtons = function (state) {
    this.fontScale = new FontScale(this.domRoot, state, this.irEditor);

    this.topBar = this.domRoot.find(".top-bar");
};

Ir.prototype.initCallbacks = function () {
    this.fontScale.on('change', _.bind(this.updateState, this));

    this.container.on('destroy', this.close, this);

    this.eventHub.on('compileResult', this.onCompileResult, this);
    this.eventHub.on('compiler', this.onCompiler, this);
    this.eventHub.on('compilerClose', this.onCompilerClose, this);
    this.eventHub.on('settingsChange', this.onSettingsChange, this);
    this.eventHub.on('colours', this.onColours, this);
    this.eventHub.emit('irViewOpened', this._compilerid);
    this.eventHub.emit('requestSettings');

    this.container.on('resize', this.resize, this);
    this.container.on('shown', this.resize, this);
};

Ir.prototype.onColours = function (editorid, colours, scheme) {
    if (editorid === this._editorid) {
        var irColours = {};
        _.each(this.assembly, function (x, index) {
            if (x.source && x.source.file === null && x.source.line > 0) {
                irColours[index] = colours[x.source.line - 1];
            }
        });
        this.colours = colour.applyColours(this.irEditor, irColours, scheme, this.colours);
    }
};

Ir.prototype.resize = function () {
    var topBarHeight = this.topBar.outerHeight(true);
    this.irEditor.layout({
        width: this.domRoot.width(),
        height: this.domRoot.height() - topBarHeight
    });
};

Ir.prototype.onCompileResult = function (id, compiler, result, lang) {
    if (this._compilerid !== id) return;
    if (result.hasIrOutput) {
        this.ir = result.irOutput;
        this.showIrResults(result.irOutput);
    }
    else if (compiler.supportsIrView) {
        this.ir = {};
        this.showIrResults("<No output>");
    }

    if (lang && lang.monaco && this.getCurrentEditorLanguage() !== lang.monaco) {
        monaco.editor.setModelLanguage(this.irEditor.getModel(), lang.monaco);
    }
};

// Monaco language id of the current editor
Ir.prototype.getCurrentEditorLanguage = function () {
    return this.irEditor.getModel().getModeId();
};

Ir.prototype.setTitle = function () {
    this.container.setTitle(
        this._compilerName + " IR Viewer (Editor #" + this._editorid + ", Compiler #" + this._compilerid + ")"
    );
};

Ir.prototype.showIrResults = function (results) {
    var currentTopLine = this.irEditor.getCompletelyVisibleLinesRangeInViewport().startLineNumber;
    this.irEditor.setValue(_.map(results, function (line) {return line.text;}).join('\n'));
    this.irEditor.revealLine(currentTopLine);
};

Ir.prototype.onCompiler = function (id, compiler, options, editorid) {
    if (id === this._compilerid) {
        this._compilerName = compiler ? compiler.name : '';
        this._editorid = editorid;
        this.setTitle();
        if (compiler && !compiler.supportsIrView) {
            this.irEditor.setValue("<IR output is not supported for this compiler>");
        }
    }
};

Ir.prototype.onCompilerClose = function (id) {
    if (id === this._compilerid) {
        // We can't immediately close as an outer loop somewhere in GoldenLayout is iterating over
        // the hierarchy. We can't modify while it's being iterated over.
        _.defer(function (self) {
            self.container.close();
        }, this);
    }
};

Ir.prototype.updateState = function () {
    this.container.setState(this.currentState());
};

Ir.prototype.currentState = function () {
    var state = {
        id: this._compilerid,
        editorid: this._editorid
    };
    this.fontScale.addState(state);
    return state;
};

Ir.prototype.onCompilerClose = function (id) {
    if (id === this._compilerid) {
        // We can't immediately close as an outer loop somewhere in GoldenLayout is iterating over
        // the hierarchy. We can't modify while it's being iterated over.
        this.close();
        _.defer(function (self) {
            self.container.close();
        }, this);
    }
};

Ir.prototype.onSettingsChange = function (newSettings) {
    this.irEditor.updateOptions({
        contextmenu: newSettings.useCustomContextMenu,
        minimap: {
            enabled: newSettings.showMinimap
        }
    });
};

Ir.prototype.close = function () {
    this.eventHub.unsubscribe();
    this.eventHub.emit("irViewClosed", this._compilerid);
    this.irEditor.dispose();
};

module.exports = {
    Ir: Ir
};
