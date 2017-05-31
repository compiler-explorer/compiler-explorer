// Copyright (c) 2012-2017, Simon Brand
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

    require('asm-mode');
    require('selectize');

    function Ast(hub, container, state) {
        this.container = container;
        this.eventHub = hub.createEventHub();
        this.domRoot = container.getElement();
        this.domRoot.html($('#ast').html());
        this.compilers = {};
        this._currentDecorations = [];
        this.astEditor = monaco.editor.create(this.domRoot.find(".monaco-placeholder")[0], {
            value: state.source || "",
            scrollBeyondLastLine: false,
            language: 'cpp', //we only support cpp for now
            readOnly: true,
            glyphMargin: true,
            quickSuggestions: false,
            fixedOverflowWidgets: true
        });

        this._compilerid = state.id;
        this._compilerName = state.compilerName;
        this._editorid = state.editorid;
        this.fontScale = new FontScale(this.domRoot, state, this.astEditor);
        this.fontScale.on('change', _.bind(this.updateState, this));

        this.eventHub.on('compileResult', this.onCompileResult, this);
        this.eventHub.on('compiler', this.onCompiler, this);
        this.eventHub.on('compilerClose', this.onCompilerClose, this);
        this.eventHub.on('editorChange', this.onEditorChange, this);
        this.eventHub.on('themeChange', this.onThemeChange, this);
        this.eventHub.emit('requestTheme');

        this.container.on('destroy', function () {
            this.eventHub.emit("astViewClosed", this._compilerid);
            this.eventHub.unsubscribe();
            this.astEditor.dispose();
        }, this);

        container.on('resize', this.resize, this);
        container.on('shown', this.resize, this);
        if(state && state.astOutput) {
              this.showAstResults(state.astOutput);
        }
        this.setTitle();
    }

    // TODO: de-dupe with compiler etc
    Ast.prototype.resize = function () {
        var topBarHeight = this.domRoot.find(".top-bar").outerHeight(true);
        this.astEditor.layout({
            width: this.domRoot.width(),
            height: this.domRoot.height() - topBarHeight
        });
    };

    Ast.prototype.onEditorChange = function(id, source) {
    };
    
    Ast.prototype.onCompileResult = function (id, compiler, result) {
        if(result.hasAstOutput && this._compilerid == id) {
            this.showAstResults(result.astOutput);
        }
    };
    Ast.prototype.setTitle = function () {
          this.container.setTitle(this._compilerName + " Ast Viewer (Editor #" + this._editorid + ", Compiler #" + this._compilerid + ")");
    };

    Ast.prototype.getDisplayableAst = function (astResult) {
       return "**" + astResult.astType + "** - " + astResult.displayString;
    };

    Ast.prototype.showAstResults = function(results) {
        this.astEditor.setValue(results);
    };

    Ast.prototype.onCompiler = function (id, compiler, options, editorid) {
        if(id == this._compilerid) {
            this._compilerName = compiler.name;
            this._editorid = editorid;
            this.setTitle();
        }
    };

    Ast.prototype.onCompilerClose = function (id) {
        delete this.compilers[id];
    };

    Ast.prototype.onThemeChange = function (newTheme) {
        if (this.astEditor) {
            this.astEditor.updateOptions({theme: newTheme.monaco});
        }
    };


    Ast.prototype.updateState = function () {
    };

    return {
        Ast: Ast
    };
});
