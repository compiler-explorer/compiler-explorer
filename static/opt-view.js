// Copyright (c) 2012-2017, Jared Wyles
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

define(function (require) {
    "use strict";

    var FontScale = require('fontscale');
    var monaco = require('monaco');
    var options = require('options');
    var _ = require('underscore');
    var $ = require('jquery');

    require('asm-mode');
    require('selectize');

    function Opt(hub, container, state) {
        state = state || {};
        this.container = container;
        this.eventHub = hub.createEventHub();
        this.domRoot = container.getElement();
        this.domRoot.html($('#opt').html());
        this.compilers = {};
        this.code = state.source || "";
        this._currentDecorations = [];
        this.optEditor = monaco.editor.create(this.domRoot.find(".monaco-placeholder")[0], {
            value: this.code,
            scrollBeyondLastLine: false,
            language: 'cppp', //we only support cpp(p) for now
            readOnly: true,
            glyphMargin: true,
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
        this.fontScale = new FontScale(this.domRoot, state, this.optEditor);
        this.fontScale.on('change', _.bind(this.updateState, this));

        this.eventHub.on('compileResult', this.onCompileResult, this);
        this.eventHub.on('compiler', this.onCompiler, this);
        this.eventHub.on('compilerClose', this.onCompilerClose, this);
        this.eventHub.on('editorChange', this.onEditorChange, this);
        this.eventHub.on('settingsChange', this.onSettingsChange, this);
        this.eventHub.on('resize', this.resize, this);
        this.container.on('destroy', function () {
            this.eventHub.emit("optViewClosed", this._compilerid);
            this.eventHub.unsubscribe();
            this.optEditor.dispose();
        }, this);
        this.eventHub.emit('requestSettings');

        container.on('resize', this.resize, this);
        container.on('shown', this.resize, this);
        if (state && state.optOutput) {
            this.showOptResults(state.optOutput);
        }
        this.setTitle();
        this.eventHub.emit("optViewOpened", this._compilerid);
    }

    // TODO: de-dupe with compiler etc
    Opt.prototype.resize = function () {
        var topBarHeight = this.domRoot.find(".top-bar").outerHeight(true);
        this.optEditor.layout({
            width: this.domRoot.width(),
            height: this.domRoot.height() - topBarHeight
        });
    };

    Opt.prototype.onEditorChange = function (id, source) {
        if (this._editorid == id) {
            this.code = source;
            this.optEditor.setValue(source);
        }
    };
    Opt.prototype.onCompileResult = function (id, compiler, result) {
        if (result.hasOptOutput && this._compilerid == id) {
            this.showOptResults(result.optOutput);
        }
    };
    Opt.prototype.setTitle = function () {
        this.container.setTitle(this._compilerName + " Opt Viewer (Editor #" + this._editorid + ", Compiler #" + this._compilerid + ")");
    };

    Opt.prototype.getDisplayableOpt = function (optResult) {
        return "**" + optResult.optType + "** - " + optResult.displayString;
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
                if (x.optType == "Missed" || acc == "Missed") {
                    return "Missed";
                } else if (x.optType == "Passed" || acc == "Passed") {
                    return "Passed";
                }
                return x.optType;
            }, "");
            var contents = _.map(value, this.getDisplayableOpt, this);
            opt.push({
                range: new monaco.Range(linenumber, 1, linenumber, Infinity),
                options: {
                    isWholeLine: true,
                    glyphMarginClassName: "opt-decoration." + className.toLowerCase(),
                    hoverMessage: contents,
                    glyphMarginHoverMessage: contents
                }
            });
        }, this);

        this._currentDecorations = this.optEditor.deltaDecorations(this._currentDecorations, opt);
    };

    Opt.prototype.onCompiler = function (id, compiler, options, editorid) {
        if (compiler && !compiler.supportsOptOutput) {
            this.code = this.optEditor.getValue();
            this.optEditor.setValue("<" + compiler.version + " does not support the optimisation view>");
            return;
        }

        if (id == this._compilerid) {
            this._compilerName = compiler ? compiler.name : '';
            this._editorid = editorid;
            this.setTitle();
            this.optEditor.setValue(this.code);
        }
    };

    Opt.prototype.onCompilerClose = function (id) {
        delete this.compilers[id];
    };

    Opt.prototype.updateState = function () {
    };

    Opt.prototype.onSettingsChange = function (newSettings) {
        this.optEditor.updateOptions({
            minimap: {
                enabled: newSettings.showMinimap
            }
        });
    };

    return {
        Opt: Opt
    };
});
