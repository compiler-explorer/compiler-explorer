// Copyright (c) 2012-2017, Matt Godbolt
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

    const FontScale = require('fontscale');
    const monaco = require('monaco');
    const _ = require('underscore');

    require('asm-mode');
    require('selectize');

    function State(id, model) {
        this.id = id;
        this.model = model;
        this.compiler = null;
        this.result = null;
    }

    State.prototype.update = function (id, compiler, result) {
        if (this.id !== id) return false;
        this.compiler = compiler;
        this.result = result;
        const asm = result.asm || [];
        this.model.setValue(_.pluck(asm, 'text').join("\n"));
        return true;
    };

    function Diff(hub, container, state) {
        this.container = container;
        this.eventHub = hub.createEventHub();
        this.domRoot = container.getElement();
        this.domRoot.html($('#diff').html());
        this.compilers = {};

        this.outputEditor = monaco.editor.createDiffEditor(this.domRoot.find(".monaco-placeholder")[0], {
            fontFamily: 'Fira Mono',
            scrollBeyondLastLine: false,
            readOnly: true,
            language: 'asm'
        });

        this.lhs = new State(state.lhs, monaco.editor.createModel('', 'asm'));
        this.rhs = new State(state.rhs, monaco.editor.createModel('', 'asm'));
        this.outputEditor.setModel({original: this.lhs.model, modified: this.rhs.model});

        const self = this;
        const selectize = this.domRoot.find(".diff-picker").selectize({
            sortField: 'name',
            valueField: 'id',
            labelField: 'name',
            searchField: ['name'],
            options: [],
            items: [],
            render: {
                option: function (item, escape) {
                    return '<div>' +
                        '<span class="compiler">' + escape(item.compiler.name) + '</span>' +
                        '<span class="options">' + escape(item.options) + '</span>' +
                        '<ul class="meta">' +
                        '<li class="editor">Editor #' + escape(item.editorId) + '</li>' +
                        '<li class="compilerId">Compiler #' + escape(item.id) + '</li>' +
                        '</ul></div>';
                }
            }
        }).on('change', function () {
            let compiler = self.compilers[$(this).val()];
            if (!compiler) return;
            if ($(this).hasClass('lhs')) {
                self.lhs.compiler = compiler;
                self.lhs.id = compiler.id;
            } else {
                self.rhs.compiler = compiler;
                self.rhs.id = compiler.id;
            }
            self.onDiffSelect(compiler.id);
        });
        this.selectize = {lhs: selectize[0].selectize, rhs: selectize[1].selectize};

        this.fontScale = new FontScale(this.domRoot, state, this.outputEditor);
        this.fontScale.on('change', _.bind(this.updateState, this));

        this.eventHub.on('compileResult', this.onCompileResult, this);
        this.eventHub.on('compiler', this.onCompiler, this);
        this.eventHub.on('compilerClose', this.onCompilerClose, this);
        this.eventHub.on('themeChange', this.onThemeChange, this);
        this.container.on('destroy', function () {
            this.eventHub.unsubscribe();
            this.outputEditor.dispose();
        }, this);
        container.on('resize', this.resize, this);
        container.on('shown', this.resize, this);

        this.eventHub.emit('resendCompilation', this.lhs.id);
        this.eventHub.emit('resendCompilation', this.rhs.id);
        this.eventHub.emit('findCompilers');
        this.eventHub.emit('requestTheme');

        this.updateCompilerNames();
        this.updateCompilers();
    }

    // TODO: de-dupe with compiler etc
    Diff.prototype.resize = function () {
        const topBarHeight = this.domRoot.find(".top-bar").outerHeight(true);
        this.outputEditor.layout({
            width: this.domRoot.width(),
            height: this.domRoot.height() - topBarHeight
        });
    };

    Diff.prototype.onDiffSelect = function (id) {
        this.eventHub.emit('resendCompilation', id);
        this.updateCompilerNames();
        this.updateState();
    };

    Diff.prototype.onCompileResult = function (id, compiler, result) {
        // both sides must be updated, don't be tempted to rewrite this as
        // var changes = lhs.update() || rhs.update();
        const lhsChanged = this.lhs.update(id, compiler, result);
        const rhsChanged = this.rhs.update(id, compiler, result);
        if (lhsChanged || rhsChanged) {
            this.updateCompilerNames();
        }
    };

    Diff.prototype.onCompiler = function (id, compiler, options, editorId) {
        if (!compiler) return;
        options = options || "";
        let name = compiler.name + " " + options;
        // TODO: selectize doesn't play nicely with CSS tricks for truncation; this is the best I can do
        // There's a plugin at: http://www.benbybenjacobs.com/blog/2014/04/09/no-wrap-plugin-for-selectize-dot-js
        // but it doesn't look easy to integrate.
        const maxLength = 30;
        if (name.length > maxLength - 3) name = name.substr(0, maxLength - 3) + "...";
        this.compilers[id] = {
            id: id,
            name: name,
            options: options,
            editorId: editorId,
            compiler: compiler
        };
        if (!this.lhs.id) {
            this.lhs.compiler = this.compilers[id];
            this.lhs.id = id;
            this.onDiffSelect(id);
        } else if (!this.rhs.id) {
            this.rhs.compiler = this.compilers[id];
            this.rhs.id = id;
            this.onDiffSelect(id);
        }
        this.updateCompilers();
    };

    Diff.prototype.onCompilerClose = function (id) {
        delete this.compilers[id];
        this.updateCompilers();
    };

    Diff.prototype.updateCompilerNames = function () {
        let name = "Diff";
        if (this.lhs.compiler && this.rhs.compiler)
            name += " " + this.lhs.compiler.name + " vs " + this.rhs.compiler.name;
        this.container.setTitle(name);
    };

    Diff.prototype.updateCompilersFor = function (selectize, id) {
        selectize.clearOptions();
        _.each(this.compilers, function (compiler) {
            selectize.addOption(compiler);
        }, this);
        if (this.compilers[id]) {
            selectize.setValue(id);
        }
    };

    Diff.prototype.updateCompilers = function () {
        this.updateCompilersFor(this.selectize.lhs, this.lhs.id);
        this.updateCompilersFor(this.selectize.rhs, this.rhs.id);
    };

    Diff.prototype.updateState = function () {
        const state = {
            lhs: this.lhs.id,
            rhs: this.rhs.id
        };
        this.fontScale.addState(state);
        this.container.setState(state);
    };

    Diff.prototype.onThemeChange = function (newTheme) {
        if (this.outputEditor)
            this.outputEditor.updateOptions({theme: newTheme.monaco});
    };

    return {
        Diff: Diff,
        getComponent: function (lhs, rhs) {
            return {
                type: 'component',
                componentName: 'diff',
                componentState: {lhs: lhs, rhs: rhs},
            };
        }
    };
});
