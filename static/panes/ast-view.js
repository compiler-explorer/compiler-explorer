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

var FontScale = require('../fontscale');
var monaco = require('monaco-editor');
var _ = require('underscore');
var $ = require('jquery');
var colour = require('../colour');
var ga = require('../analytics');
var monacoConfig = require('../monaco-config');

function Ast(hub, container, state) {
    this.container = container;
    this.eventHub = hub.createEventHub();
    this.domRoot = container.getElement();
    this.domRoot.html($('#ast').html());

    this.decorations = {};
    this.prevDecorations = [];
    var root = this.domRoot.find('.monaco-placeholder');
    this.astEditor = monaco.editor.create(root[0], monacoConfig.extendConfig({
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

    this.colours = [];
    this.astCode = [];
    this.lastColours = [];
    this.lastColourScheme = {};


    this.initButtons(state);
    this.initCallbacks();

    if (state && state.astOutput) {
        this.showAstResults(state.astOutput);
    }
    this.setTitle();

    ga.proxy('send', {
        hitType: 'event',
        eventCategory: 'OpenViewPane',
        eventAction: 'Ast',
    });
}

Ast.prototype.initButtons = function (state) {
    this.fontScale = new FontScale(this.domRoot, state, this.astEditor);

    this.topBar = this.domRoot.find('.top-bar');
};

Ast.prototype.initCallbacks = function () {
    this.linkedFadeTimeoutId = -1;
    this.mouseMoveThrottledFunction = _.throttle(_.bind(this.onMouseMove, this), 50);
    this.astEditor.onMouseMove(_.bind(function (e) {
        this.mouseMoveThrottledFunction(e);
    }, this));

    this.fontScale.on('change', _.bind(this.updateState, this));

    this.container.on('destroy', this.close, this);

    this.eventHub.on('compileResult', this.onCompileResult, this);
    this.eventHub.on('compiler', this.onCompiler, this);
    this.eventHub.on('colours', this.onColours, this);
    this.eventHub.on('panesLinkLine', this.onPanesLinkLine, this);
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
        height: this.domRoot.height() - topBarHeight,
    });
};

Ast.prototype.onCompileResult = function (id, compiler, result, lang) {
    if (this._compilerid !== id) return;

    if (result.hasAstOutput) {
        this.showAstResults(result.astOutput);
    }
    else if (compiler.supportsAstView) {
        this.showAstResults([{text: '<No output>'}]);
    }

    if (lang && lang.monaco && this.getCurrentEditorLanguage() !== lang.monaco) {
        monaco.editor.setModelLanguage(this.astEditor.getModel(), lang.monaco);
    }

    // Copied over from ir-view.js:onCompileResponse
    // Why call this explicitly instead of just listening to the "colours" event?
    // Because the recolouring happens before this editors value is set using "showIrResults".
    this.onColours(this._compilerid, this.lastColours, this.lastColourScheme);
};

// Monaco language id of the current editor
Ast.prototype.getCurrentEditorLanguage = function () {
    return this.astEditor.getModel().getModeId();
};

Ast.prototype.getPaneName = function () {
    return this._compilerName + ' Ast Viewer (Editor #' + this._editorid + ', Compiler #' + this._compilerid + ')';
};

Ast.prototype.setTitle = function () {
    this.container.setTitle(this.getPaneName());
};

Ast.prototype.getDisplayableAst = function (astResult) {
    return '**' + astResult.astType + '** - ' + astResult.displayString;
};

Ast.prototype.showAstResults = function (results) {
    var fullText = results.map(function (x) { return x.text; }).join('\n');
    this.astEditor.setValue(fullText);
    this.astCode = results;

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
            this.astEditor.setValue('<AST output is not supported for this compiler>');
        }
    }
};

Ast.prototype.onColours = function (id, colours, scheme) {
    if (id === this._compilerid) {
        this.lastColours = colours;
        this.lastColourScheme = scheme;

        var astColours = {};
        _.each(this.astCode, function (x, index) {
            if (x.source && x.source.from && x.source.to &&
                x.source.from <= x.source.to && x.source.to < x.source.from + 100) {
                var i;
                for (i = x.source.from; i <= x.source.to; ++i) {
                    if (colours[i - 1] !== undefined) {
                        astColours[index] = colours[i - 1];
                        break;
                    }
                }
            }
        });
        this.colours = colour.applyColours(this.astEditor, astColours, scheme, this.colours);
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
        selection: this.selection,
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
    this.settings = newSettings;
    this.astEditor.updateOptions({
        contextmenu: newSettings.useCustomContextMenu,
        minimap: {
            enabled: newSettings.showMinimap,
        },
        fontFamily: newSettings.editorsFFont,
        fontLigatures: newSettings.editorsFLigatures,
    });
};

Ast.prototype.onMouseMove = function (e) {
    if (e === null || e.target === null || e.target.position === null) return;
    if (this.settings.hoverShowSource === true && this.astCode) {
        this.clearLinkedLines();
        var hoverCode = this.astCode[e.target.position.lineNumber - 1];
        if (hoverCode) {
            // We check that we actually have something to show at this point!
            var sourceLine = (hoverCode.source && hoverCode.source.from && hoverCode.source.to)
                ? hoverCode.source.from
                : -1;
            this.eventHub.emit('editorLinkLine', this._editorid, sourceLine, -1, false);
            this.eventHub.emit('panesLinkLine', this._compilerid, sourceLine, false, this.getPaneName());
        }
    }
};

Ast.prototype.onDidChangeCursorSelection = function (e) {
    if (this.awaitingInitialResults) {
        this.selection = e.selection;
        this.updateState();
    }
};

Ast.prototype.updateDecorations = function () {
    this.prevDecorations = this.astEditor.deltaDecorations(
        this.prevDecorations, _.flatten(_.values(this.decorations)));
};

Ast.prototype.clearLinkedLines = function () {
    this.decorations.linkedCode = [];
    this.updateDecorations();
};

Ast.prototype.onPanesLinkLine = function (compilerId, lineNumber, revealLine, sender) {
    if (Number(compilerId) === this._compilerid) {
        var lineNums = [];
        _.each(this.astCode, function (astLine, i) {
            if (astLine.source && astLine.source.from <= lineNumber && lineNumber <= astLine.source.to) {
                var line = i + 1;
                lineNums.push(line);
            }
        });
        if (revealLine && lineNums[0]) this.astEditor.revealLineInCenter(lineNums[0]);
        var lineClass = sender !== this.getPaneName() ? 'linked-code-decoration-line' : '';
        this.decorations.linkedCode = _.map(lineNums, function (line) {
            return {
                range: new monaco.Range(line, 1, line, 1),
                options: {
                    isWholeLine: true,
                    linesDecorationsClassName: 'linked-code-decoration-margin',
                    className: lineClass,
                },
            };
        });
        if (this.linkedFadeTimeoutId !== -1) {
            clearTimeout(this.linkedFadeTimeoutId);
        }
        this.linkedFadeTimeoutId = setTimeout(_.bind(function () {
            this.clearLinkedLines();
            this.linkedFadeTimeoutId = -1;
        }, this), 5000);
        this.updateDecorations();
    }
};

Ast.prototype.close = function () {
    this.eventHub.unsubscribe();
    this.eventHub.emit('astViewClosed', this._compilerid);
    this.astEditor.dispose();
};

module.exports = {
    Ast: Ast,
};
