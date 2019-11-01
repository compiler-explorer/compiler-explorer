// Copyright (c) 2017, Matt Godbolt
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

require('../modes/asm-mode');
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
    var asm = result.asm || [];
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
        fontFamily: 'Consolas, "Liberation Mono", Courier, monospace',
        scrollBeyondLastLine: false,
        readOnly: true,
        language: 'asm'
    });

    this.lhs = new State(state.lhs, monaco.editor.createModel('', 'asm'));
    this.rhs = new State(state.rhs, monaco.editor.createModel('', 'asm'));
    this.outputEditor.setModel({original: this.lhs.model, modified: this.rhs.model});

    var selectize = this.domRoot.find(".diff-picker").selectize({
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
        },
        dropdownParent: 'body'
    }).on('change', _.bind(function (e) {
        var target = $(e.target);
        var compiler = this.compilers[target.val()];
        if (!compiler) return;
        if (target.hasClass('lhs')) {
            this.lhs.compiler = compiler;
            this.lhs.id = compiler.id;
        } else {
            this.rhs.compiler = compiler;
            this.rhs.id = compiler.id;
        }
        this.onDiffSelect(compiler.id);
    }, this));
    this.selectize = {lhs: selectize[0].selectize, rhs: selectize[1].selectize};

    this.initButtons(state);
    this.initCallbacks();

    this.updateCompilerNames();
    this.updateCompilers();
    ga.proxy('send', {
        hitType: 'event',
        eventCategory: 'OpenViewPane',
        eventAction: 'Diff'
    });
}

// TODO: de-dupe with compiler etc
Diff.prototype.resize = function () {
    var topBarHeight = this.topBar.outerHeight(true);
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
    var lhsChanged = this.lhs.update(id, compiler, result);
    var rhsChanged = this.rhs.update(id, compiler, result);
    if (lhsChanged || rhsChanged) {
        this.updateCompilerNames();
    }
};

Diff.prototype.initButtons = function (state) {
    this.fontScale = new FontScale(this.domRoot, state, this.outputEditor);

    this.topBar = this.domRoot.find(".top-bar");
};

Diff.prototype.initCallbacks = function () {
    this.fontScale.on('change', _.bind(this.updateState, this));

    this.eventHub.on('compileResult', this.onCompileResult, this);
    this.eventHub.on('compiler', this.onCompiler, this);
    this.eventHub.on('compilerClose', this.onCompilerClose, this);
    this.eventHub.on('settingsChange', this.onSettingsChange, this);
    this.eventHub.on('themeChange', this.onThemeChange, this);
    this.container.on('destroy', function () {
        this.eventHub.unsubscribe();
        this.outputEditor.dispose();
    }, this);
    this.container.on('resize', this.resize, this);
    this.container.on('shown', this.resize, this);

    this.eventHub.emit('resendCompilation', this.lhs.id);
    this.eventHub.emit('resendCompilation', this.rhs.id);
    this.eventHub.emit('findCompilers');
    this.eventHub.emit('requestTheme');
    this.eventHub.emit('requestSettings');
};

Diff.prototype.onCompiler = function (id, compiler, options, editorId) {
    if (!compiler) return;
    options = options || "";
    var name = compiler.name + " " + options;
    // TODO: selectize doesn't play nicely with CSS tricks for truncation; this is the best I can do
    // There's a plugin at: http://www.benbybenjacobs.com/blog/2014/04/09/no-wrap-plugin-for-selectize-dot-js
    // but it doesn't look easy to integrate.
    var maxLength = 30;
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
    var name = "Diff";
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
    var state = {
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

Diff.prototype.onSettingsChange =  function (newSettings) {
    this.outputEditor.updateOptions({
        minimap: {
            enabled: newSettings.showMinimap
        },
        fontFamily: newSettings.editorsFFont,
        fontLigatures: newSettings.enableLigatures
    });
};

module.exports = {
    Diff: Diff,
    getComponent: function (lhs, rhs) {
        return {
            type: 'component',
            componentName: 'diff',
            componentState: {lhs: lhs, rhs: rhs}
        };
    }
};
