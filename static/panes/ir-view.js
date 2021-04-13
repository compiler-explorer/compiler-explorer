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

'use strict';

var FontScale = require('../fontscale');
var monaco = require('monaco-editor');
var _ = require('underscore');
var $ = require('jquery');
var colour = require('../colour');
var ga = require('../analytics');
var monacoConfig = require('../monaco-config');

function Ir(hub, container, state) {
    this.container = container;
    this.eventHub = hub.createEventHub();
    this.domRoot = container.getElement();
    this.domRoot.html($('#ir').html());

    this.decorations = {};
    this.prevDecorations = [];
    var root = this.domRoot.find('.monaco-placeholder');

    this.irEditor = monaco.editor.create(root[0], monacoConfig.extendConfig({
        language: 'llvm-ir',
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
    this.initEditorActions();

    if (state && state.irOutput) {
        this.showIrResults(state.irOutput);
    }
    this.setTitle();

    ga.proxy('send', {
        hitType: 'event',
        eventCategory: 'OpenViewPane',
        eventAction: 'Ir',
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
            var source = this.irCode[desiredLine].source;
            if (source !== null && source.file === null) {
                // a null file means it was the user's source
                this.eventHub.emit('editorLinkLine', this._editorid, source.line, -1, -1, true);
            }
        }, this),
    });
};

Ir.prototype.initButtons = function (state) {
    this.fontScale = new FontScale(this.domRoot, state, this.irEditor);

    this.topBar = this.domRoot.find('.top-bar');
};

Ir.prototype.initCallbacks = function () {
    this.linkedFadeTimeoutId = -1;
    this.mouseMoveThrottledFunction = _.throttle(_.bind(this.onMouseMove, this), 50);
    this.irEditor.onMouseMove(_.bind(function (e) {
        this.mouseMoveThrottledFunction(e);
    }, this));

    this.cursorSelectionThrottledFunction =
        _.throttle(_.bind(this.onDidChangeCursorSelection, this), 500);
    this.irEditor.onDidChangeCursorSelection(_.bind(function (e) {
        this.cursorSelectionThrottledFunction(e);
    }, this));

    this.fontScale.on('change', _.bind(this.updateState, this));

    this.container.on('destroy', this.close, this);

    var onColoursOnCompile = this.eventHub.mediateDependentCalls(this.onColours, this.onCompileResponse);

    this.eventHub.on('compileResult', onColoursOnCompile.dependencyProxy, this);
    this.eventHub.on('compiler', this.onCompiler, this);
    this.eventHub.on('colours', onColoursOnCompile.dependentProxy, this);
    this.eventHub.on('panesLinkLine', this.onPanesLinkLine, this);
    this.eventHub.on('compilerClose', this.onCompilerClose, this);
    this.eventHub.on('settingsChange', this.onSettingsChange, this);
    this.eventHub.emit('irViewOpened', this._compilerid);
    this.eventHub.emit('requestSettings');

    this.container.on('resize', this.resize, this);
    this.container.on('shown', this.resize, this);
};

// TODO: de-dupe with compiler etc
Ir.prototype.resize = function () {
    var topBarHeight = this.topBar.outerHeight(true);
    this.irEditor.layout({
        width: this.domRoot.width(),
        height: this.domRoot.height() - topBarHeight,
    });
};

Ir.prototype.onCompileResponse = function (id, compiler, result) {
    if (this._compilerid !== id) return;
    if (result.hasIrOutput) {
        this.showIrResults(result.irOutput);
    } else if (compiler.supportsIrView) {
        this.showIrResults([{text: '<No output>'}]);
    }
};

Ir.prototype.getPaneName = function () {
    return this._compilerName + ' IR Viewer (Editor #' + this._editorid + ', Compiler #' + this._compilerid + ')';
};

Ir.prototype.setTitle = function () {
    this.container.setTitle(this.getPaneName());
};

Ir.prototype.showIrResults = function (irCode) {
    if (!this.irEditor) return;
    this.irCode = irCode;
    this.irEditor.getModel().setValue(irCode.length ? _.pluck(irCode, 'text').join('\n') : '<No IR generated>');

    if (!this.awaitingInitialResults) {
        if (this.selection) {
            this.irEditor.setSelection(this.selection);
            this.irEditor.revealLinesInCenter(this.selection.startLineNumber,
                this.selection.endLineNumber);
        }
        this.awaitingInitialResults = true;
    }
};

Ir.prototype.onCompiler = function (id, compiler, options, editorid) {
    if (id === this._compilerid) {
        this._compilerName = compiler ? compiler.name : '';
        this._editorid = editorid;
        this.setTitle();
        if (compiler && !compiler.supportsIrView) {
            this.irEditor.setValue('<IR output is not supported for this compiler>');
        }
    }
};

Ir.prototype.onColours = function (id, colours, scheme) {
    if (id === this._compilerid) {
        var irColours = {};
        _.each(this.irCode, function (x, index) {
            if (x.source && x.source.file === null && x.source.line > 0 && colours[x.source.line - 1] !== undefined) {
                irColours[index] = colours[x.source.line - 1];
            }
        });
        this.colours = colour.applyColours(this.irEditor, irColours, scheme, this.colours);
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
        editorid: this._editorid,
        selection: this.selection,
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
    this.settings = newSettings;
    this.irEditor.updateOptions({
        contextmenu: newSettings.useCustomContextMenu,
        minimap: {
            enabled: newSettings.showMinimap,
        },
        fontFamily: newSettings.editorsFFont,
        fontLigatures: newSettings.editorsFLigatures,
    });
};

Ir.prototype.onMouseMove = function (e) {
    if (e === null || e.target === null || e.target.position === null) return;
    if (this.settings.hoverShowSource === true && this.irCode) {
        this.clearLinkedLines();
        var hoverCode = this.irCode[e.target.position.lineNumber - 1];
        if (hoverCode) {
            // We check that we actually have something to show at this point!
            var sourceLine = -1;
            var sourceColBegin = -1;
            var sourceColEnd = -1;
            if (hoverCode.source && !hoverCode.source.file) {
                sourceLine = hoverCode.source.line;
                if (hoverCode.source.column) {
                    sourceColBegin = hoverCode.source.column;
                    sourceColEnd = sourceColBegin;
                }
            }
            this.eventHub.emit('editorLinkLine', this._editorid, sourceLine, sourceColBegin, sourceColEnd, false);
            this.eventHub.emit('panesLinkLine', this._compilerid,
                sourceLine, sourceColBegin, sourceColEnd,
                false, this.getPaneName());
        }
    }
};

Ir.prototype.onDidChangeCursorSelection = function (e) {
    if (this.awaitingInitialResults) {
        this.selection = e.selection;
        this.updateState();
    }
};


Ir.prototype.updateDecorations = function () {
    this.prevDecorations = this.irEditor.deltaDecorations(
        this.prevDecorations, _.flatten(_.values(this.decorations)));
};

Ir.prototype.clearLinkedLines = function () {
    this.decorations.linkedCode = [];
    this.updateDecorations();
};

Ir.prototype.onPanesLinkLine = function (compilerId, lineNumber, colBegin, colEnd, revealLine, sender) {
    if (Number(compilerId) === this._compilerid) {
        var lineNums = [];
        var directlyLinkedLineNums = [];
        var signalFromAnotherPane = sender !== this.getPaneName();
        _.each(this.irCode, function (irLine, i) {
            if (irLine.source && irLine.source.file === null && irLine.source.line === lineNumber) {
                var line = i + 1;
                lineNums.push(line);
                var currentCol = irLine.source.column;
                if (signalFromAnotherPane && currentCol && colBegin <= currentCol && currentCol <= colEnd) {
                    directlyLinkedLineNums.push(line);
                }
            }
        });
        if (revealLine && lineNums[0]) this.irEditor.revealLineInCenter(lineNums[0]);
        var lineClass = sender !== this.getPaneName() ? 'linked-code-decoration-line' : '';
        var linkedLineDecorations = _.map(lineNums, function (line) {
            return {
                range: new monaco.Range(line, 1, line, 1),
                options: {
                    isWholeLine: true,
                    linesDecorationsClassName: 'linked-code-decoration-margin',
                    className: lineClass,
                },
            };
        });
        var directlyLinkedLineDecorations = _.map(directlyLinkedLineNums, function (line) {
            return {
                range: new monaco.Range(line, 1, line, 1),
                options: {
                    isWholeLine: true,
                    inlineClassName: 'linked-code-decoration-column',
                },
            };
        });
        this.decorations.linkedCode = linkedLineDecorations.concat(directlyLinkedLineDecorations);
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

Ir.prototype.close = function () {
    this.eventHub.unsubscribe();
    this.eventHub.emit('irViewClosed', this._compilerid);
    this.irEditor.dispose();
};

module.exports = {
    Ir: Ir,
};
