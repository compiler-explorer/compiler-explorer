// Copyright (c) 2017, Jared Wyles
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

require('../modes/asm-mode');


function Opt(hub, container, state) {
    state = state || {};
    this.container = container;
    this.eventHub = hub.createEventHub();
    this.domRoot = container.getElement();
    this.domRoot.html($('#opt').html());
    this.source = state.source || '';
    this._currentDecorations = [];
    var root = this.domRoot.find('.monaco-placeholder');

    this.optEditor = monaco.editor.create(root[0], monacoConfig.extendConfig({
        value: this.source,
        language: 'plaintext',
        readOnly: true,
        glyphMargin: true,
    }));

    this._compilerid = state.id;
    this._compilerName = state.compilerName;
    this._editorid = state.editorid;

    this.awaitingInitialResults = false;
    this.selection = state.selection;

    this.isCompilerSupported = false;

    this.initButtons(state);
    this.initCallbacks();

    if (state && state.optOutput) {
        this.showOptResults(state.optOutput);
    }
    this.setTitle();
    this.eventHub.emit('optViewOpened', this._compilerid);
    ga.proxy('send', {
        hitType: 'event',
        eventCategory: 'OpenViewPane',
        eventAction: 'Opt',
    });
}

Opt.prototype.initButtons = function (state) {
    this.fontScale = new FontScale(this.domRoot, state, this.optEditor);

    this.topBar = this.domRoot.find('.top-bar');
};

Opt.prototype.initCallbacks = function () {
    this.fontScale.on('change', _.bind(this.updateState, this));

    this.eventHub.on('compileResult', this.onCompileResult, this);
    this.eventHub.on('compiler', this.onCompiler, this);
    this.eventHub.on('compilerClose', this.onCompilerClose, this);
    this.eventHub.on('settingsChange', this.onSettingsChange, this);
    this.eventHub.on('resize', this.resize, this);
    this.container.on('destroy', this.close, this);
    this.eventHub.emit('requestSettings');
    this.eventHub.emit('findCompilers');

    this.container.on('resize', this.resize, this);
    this.container.on('shown', this.resize, this);

    this.cursorSelectionThrottledFunction =
        _.throttle(_.bind(this.onDidChangeCursorSelection, this), 500);
    this.optEditor.onDidChangeCursorSelection(_.bind(function (e) {
        this.cursorSelectionThrottledFunction(e);
    }, this));
};

Opt.prototype.onCompileResult = function (id, compiler, result, lang) {
    if (this._compilerid !== id || !this.isCompilerSupported) return;
    this.source = result.source;
    this.optEditor.setValue(this.source);
    if (result.hasOptOutput) {
        this.showOptResults(result.optOutput);
    }
    if (lang && lang.monaco && this.getCurrentEditorLanguage() !== lang.monaco) {
        monaco.editor.setModelLanguage(this.optEditor.getModel(), lang.monaco);
    }

    if (!this.awaitingInitialResults) {
        if (this.selection) {
            this.optEditor.setSelection(this.selection);
            this.optEditor.revealLinesInCenter(this.selection.startLineNumber,
                this.selection.endLineNumber);
        }
        this.awaitingInitialResults = true;
    }
};

// Monaco language id of the current editor
Opt.prototype.getCurrentEditorLanguage = function () {
    return this.optEditor.getModel().getModeId();
};

Opt.prototype.setTitle = function () {
    this.container.setTitle(
        this._compilerName + ' Opt Viewer (Editor #' + this._editorid + ', Compiler #' + this._compilerid + ')');
};

Opt.prototype.getDisplayableOpt = function (optResult) {
    return {
        value: '**' + optResult.optType + '** - ' + optResult.displayString,
        isTrusted: false,
    };
};

Opt.prototype.showOptResults = function (results) {
    var opt = [];

    results = _.filter(results, function (x) {
        return x.DebugLoc !== undefined;
    });

    results = _.groupBy(results, function (x) {
        return x.DebugLoc.Line;
    });

    _.mapObject(results, function (value, key) {
        var linenumber = Number(key);
        var className = value.reduce(function (acc, x) {
            if (x.optType === 'Missed' || acc === 'Missed') {
                return 'Missed';
            } else if (x.optType === 'Passed' || acc === 'Passed') {
                return 'Passed';
            }
            return x.optType;
        }, '');
        var contents = _.map(value, this.getDisplayableOpt, this);
        opt.push({
            range: new monaco.Range(linenumber, 1, linenumber, Infinity),
            options: {
                isWholeLine: true,
                glyphMarginClassName: 'opt-decoration.' + className.toLowerCase(),
                hoverMessage: contents,
                glyphMarginHoverMessage: contents,
            },
        });
    }, this);

    this._currentDecorations = this.optEditor.deltaDecorations(this._currentDecorations, opt);
};

Opt.prototype.onCompiler = function (id, compiler) {
    if (id === this._compilerid) {
        this._compilerName = compiler ? compiler.name : '';
        this.setTitle();
        this.isCompilerSupported = compiler ? compiler.supportsOptOutput : false;
        if (!this.isCompilerSupported) {
            this.optEditor.setValue('<OPT output is not supported for this compiler>');
        }
    }
};

Opt.prototype.resize = function () {
    var topBarHeight = this.topBar.outerHeight(true);
    this.optEditor.layout({
        width: this.domRoot.width(),
        height: this.domRoot.height() - topBarHeight,
    });
};

Opt.prototype.updateState = function () {
    this.container.setState(this.currentState());
};

Opt.prototype.currentState = function () {
    var state = {
        id: this._compilerid,
        editorid: this._editorid,
        selection: this.selection,
    };
    this.fontScale.addState(state);
    return state;
};

Opt.prototype.close = function () {
    this.eventHub.unsubscribe();
    this.eventHub.emit('optViewClosed', this._compilerid);
    this.optEditor.dispose();
};

Opt.prototype.onCompilerClose = function (id) {
    if (id === this._compilerid) {
        // We can't immediately close as an outer loop somewhere in GoldenLayout is iterating over
        // the hierarchy. We can't modify while it's being iterated over.
        this.close();
        _.defer(function (self) {
            self.container.close();
        }, this);
    }
};

Opt.prototype.onSettingsChange = function (newSettings) {
    this.optEditor.updateOptions({
        contextmenu: newSettings.useCustomContextMenu,
        minimap: {
            enabled: newSettings.showMinimap,
        },
        fontFamily: newSettings.editorsFFont,
        fontLigatures: newSettings.editorsFLigatures,
    });
};

Opt.prototype.onDidChangeCursorSelection = function (e) {
    if (this.awaitingInitialResults) {
        this.selection = e.selection;
        this.updateState();
    }
};

module.exports = {
    Opt: Opt,
};
