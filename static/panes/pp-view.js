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

'use strict';

var FontScale = require('../fontscale').FontScale;
var Toggles = require('../toggles').Toggles;
var monaco = require('monaco-editor');
var _ = require('underscore');
var $ = require('jquery');
var ga = require('../analytics').ga;
var monacoConfig = require('../monaco-config');

function PP(hub, container, state) {
    this.container = container;
    this.eventHub = hub.createEventHub();
    this.domRoot = container.getElement();
    this.domRoot.html($('#pp').html());

    var root = this.domRoot.find('.monaco-placeholder');
    this.ppEditor = monaco.editor.create(root[0], monacoConfig.extendConfig({
        language: 'plaintext',
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

    this.ppCode = [];

    this.initButtons(state);
    this.options = new Toggles(this.domRoot.find('.options'), state);
    this.options.on('change', _.bind(this.onOptionsChange, this));

    this.initCallbacks();

    this.onOptionsChange();

    this.initialized = true;

    if (state && state.ppOutput) {
        this.showPpResults(state.ppOutput);
    } else {
        this.showCompilationLoadingMessage();
    }

    this.setTitle();

    ga.proxy('send', {
        hitType: 'event',
        eventCategory: 'OpenViewPane',
        eventAction: 'PP',
    });
}

PP.prototype.initButtons = function (state) {
    this.fontScale = new FontScale(this.domRoot, state, this.ppEditor);
    this.topBar = this.domRoot.find('.top-bar');
};

PP.prototype.initCallbacks = function () {
    this.fontScale.on('change', _.bind(this.updateState, this));

    this.container.on('destroy', this.close, this);

    var onColoursOnCompile = this.eventHub.mediateDependentCalls(this.onColours, this.onCompileResult);

    this.eventHub.on('compileResult', onColoursOnCompile.dependencyProxy, this);
    this.eventHub.on('compiler', this.onCompiler, this);
    this.eventHub.on('compilerClose', this.onCompilerClose, this);
    this.eventHub.on('settingsChange', this.onSettingsChange, this);
    this.eventHub.emit('ppViewOpened', this._compilerid);
    this.eventHub.emit('requestSettings');

    this.container.on('resize', this.resize, this);
    this.container.on('shown', this.resize, this);
};

PP.prototype.onOptionsChange = function () {
    var options = this.options.get();
    this.updateState();
    // update parameters for the compiler and recompile
    this.showCompilationLoadingMessage();
    this.eventHub.emit('ppViewOptionsUpdated', this._compilerid, {
        'filter-headers': options['filter-headers'],
        'clang-format': options['clang-format'],
    }, true);
};

PP.prototype.showCompilationLoadingMessage = function () {
    this.showPpResults([{text: '<Compiling...>'}]);
};

PP.prototype.resize = function () {
    var topBarHeight = this.topBar.outerHeight(true);
    this.ppEditor.layout({
        width: this.domRoot.width(),
        height: this.domRoot.height() - topBarHeight,
    });
};

PP.prototype.onCompileResult = function (id, compiler, result, lang) {
    if (this._compilerid !== id) return;

    if (result.hasPpOutput) {
        this.showPpResults(result.ppOutput);
    } else if (compiler.supportsPpView) {
        this.showPpResults([{text: '<No output>'}]);
    }

    if (lang && lang.monaco && this.getCurrentEditorLanguage() !== lang.monaco) {
        monaco.editor.setModelLanguage(this.ppEditor.getModel(), lang.monaco);
    }
};

// Monaco language id of the current editor
PP.prototype.getCurrentEditorLanguage = function () {
    return this.ppEditor.getModel().getLanguageId();
};

PP.prototype.getPaneName = function () {
    return 'Preprocessor Output ' + this._compilerName +
        ' (Editor #' + this._editorid + ', Compiler #' + this._compilerid + ')';
};

PP.prototype.setTitle = function () {
    this.container.setTitle(this.getPaneName());
};

PP.prototype.getDisplayablePp = function (ppResult) {
    return '**' + ppResult.ppType + '** - ' + ppResult.displayString;
};

PP.prototype.showPpResults = function (results) {
    var fullText = results.map(function (x) {
        return x.text;
    }).join('\n');
    this.ppEditor.setValue(fullText);
    this.ppCode = results;

    if (!this.awaitingInitialResults) {
        if (this.selection) {
            this.ppEditor.setSelection(this.selection);
            this.ppEditor.revealLinesInCenter(this.selection.startLineNumber,
                this.selection.endLineNumber);
        }
        this.awaitingInitialResults = true;
    }
};

PP.prototype.onCompiler = function (id, compiler, options, editorid) {
    if (id === this._compilerid) {
        this._compilerName = compiler ? compiler.name : '';
        this._editorid = editorid;
        this.setTitle();
        if (compiler && !compiler.supportsPpView) {
            this.ppEditor.setValue('<Preprocessor output is not supported for this compiler>');
        }
    }
};

PP.prototype.onCompilerClose = function (id) {
    if (id === this._compilerid) {
        // We can't immediately close as an outer loop somewhere in GoldenLayout is iterating over
        // the hierarchy. We can't modify while it's being iterated over.
        _.defer(function (self) {
            self.container.close();
        }, this);
    }
};

PP.prototype.updateState = function () {
    this.container.setState(this.currentState());
};

PP.prototype.currentState = function () {
    var options = this.options.get();
    var state = {
        id: this._compilerid,
        editorid: this._editorid,
        selection: this.selection,
        'filter-headers': options['filter-headers'],
        'clang-format': options['clang-format'],
    };
    this.fontScale.addState(state);
    return state;
};

PP.prototype.onCompilerClose = function (id) {
    if (id === this._compilerid) {
        // We can't immediately close as an outer loop somewhere in GoldenLayout is iterating over
        // the hierarchy. We can't modify while it's being iterated over.
        this.close();
        _.defer(function (self) {
            self.container.close();
        }, this);
    }
};

PP.prototype.onSettingsChange = function (newSettings) {
    this.settings = newSettings;
    this.ppEditor.updateOptions({
        contextmenu: newSettings.useCustomContextMenu,
        minimap: {
            enabled: newSettings.showMinimap,
        },
        fontFamily: newSettings.editorsFFont,
        fontLigatures: newSettings.editorsFLigatures,
    });
};

PP.prototype.close = function () {
    this.eventHub.unsubscribe();
    this.eventHub.emit('ppViewClosed', this._compilerid);
    this.ppEditor.dispose();
};

module.exports = {
    PP: PP,
};
