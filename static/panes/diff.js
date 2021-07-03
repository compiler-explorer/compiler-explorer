// Copyright (c) 2017, Compiler Explorer Authors
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
var TomSelect = require('tom-select');

require('../modes/asm-mode');


// note that these variables are saved to state, so don't change, only add to it
var
    DiffType_ASM = 0,
    DiffType_CompilerStdOut = 1,
    DiffType_CompilerStdErr = 2,
    DiffType_ExecStdOut = 3,
    DiffType_ExecStdErr = 4;

function State(id, model, difftype) {
    this.id = id;
    this.model = model;
    this.compiler = null;
    this.result = null;
    this.difftype = difftype;
}

State.prototype.update = function (id, compiler, result) {
    if (this.id !== id) return false;
    this.compiler = compiler;
    this.result = result;
    this.refresh();

    return true;
};

State.prototype.refresh = function () {
    var output = [];
    if (this.result) {
        switch (this.difftype) {
            case DiffType_ASM:
                output = this.result.asm || [];
                break;
            case DiffType_CompilerStdOut:
                output = this.result.stdout || [];
                break;
            case DiffType_CompilerStdErr:
                output = this.result.stderr || [];
                break;
            case DiffType_ExecStdOut:
                if (this.result.execResult)
                    output = this.result.execResult.stdout || [];
                break;
            case DiffType_ExecStdErr:
                if (this.result.execResult)
                    output = this.result.execResult.stderr || [];
                break;
        }
    }
    this.model.setValue(_.pluck(output, 'text').join('\n'));
};

function getItemDisplayTitle(item) {
    if (typeof item.id === 'string') {
        var p = item.id.indexOf('_exec');
        if (p !== -1) {
            return 'Executor #' + item.id.substr(0, p);
        }
    }

    return 'Compiler #' + item.id;
}

function Diff(hub, container, state) {
    this.container = container;
    this.eventHub = hub.createEventHub();
    this.domRoot = container.getElement();
    this.domRoot.html($('#diff').html());
    this.compilers = {};
    var root = this.domRoot.find('.monaco-placeholder');

    this.outputEditor = monaco.editor.createDiffEditor(root[0], {
        fontFamily: 'Consolas, "Liberation Mono", Courier, monospace',
        scrollBeyondLastLine: true,
        readOnly: true,
        language: 'asm',
    });

    this.lhs = new State(state.lhs, monaco.editor.createModel('', 'asm'), state.lhsdifftype || DiffType_ASM);
    this.rhs = new State(state.rhs, monaco.editor.createModel('', 'asm'), state.rhsdifftype || DiffType_ASM);
    this.outputEditor.setModel({original: this.lhs.model, modified: this.rhs.model});

    this.selectize = {};

    this.domRoot[0].querySelectorAll('.difftype-picker').forEach(_.bind(function (picker) {

        var instance = new TomSelect(picker, {
            sortField: 'name',
            valueField: 'id',
            labelField: 'name',
            searchField: ['name'],
            options: [
                {id: DiffType_ASM, name: 'Assembly'},
                {id: DiffType_CompilerStdOut, name: 'Compiler stdout'},
                {id: DiffType_CompilerStdErr, name: 'Compiler stderr'},
                {id: DiffType_ExecStdOut, name: 'Execution stdout'},
                {id: DiffType_ExecStdErr, name: 'Execution stderr'},
            ],
            items: [],
            render: {
                option: function (item, escape) {
                    return '<div>' + escape(item.name) + '</div>';
                },
            },
            dropdownParent: 'body',
            plugins: ['input_autogrow'],
            onChange: _.bind(function (value) {
                if (picker.classList.contains('lhsdifftype')) {
                    this.lhs.difftype = parseInt(value);
                    this.lhs.refresh();
                } else {
                    this.rhs.difftype = parseInt(value);
                    this.rhs.refresh();
                }
                this.updateState();
            }, this),
        });

        if (picker.classList.contains('lhsdifftype')) {
            this.selectize.lhsdifftype = instance;
        } else {
            this.selectize.rhsdifftype = instance;
        }

    }, this));


    this.domRoot[0].querySelectorAll('.diff-picker').forEach(_.bind(function (picker) {
        var instance = new TomSelect(picker, {
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
                        '<li class="compilerId">' + escape(getItemDisplayTitle(item)) + '</li>' +
                        '</ul></div>';
                },
            },
            dropdownParent: 'body',
            plugins: ['input_autogrow'],
            onChange: _.bind(function (value) {

                var compiler = this.compilers[value];
                if (!compiler) return;
                if (picker.classList.contains('lhs')) {
                    this.lhs.compiler = compiler;
                    this.lhs.id = compiler.id;
                } else {
                    this.rhs.compiler = compiler;
                    this.rhs.id = compiler.id;
                }
                this.onDiffSelect(compiler.id);
            }, this),
        });

        if (picker.classList.contains('lhs')) {
            this.selectize.lhs = instance;
        } else {
            this.selectize.rhs = instance;
        }
    }, this));


    this.initButtons(state);
    this.initCallbacks();

    this.updateCompilerNames();
    this.updateCompilers();
    ga.proxy('send', {
        hitType: 'event',
        eventCategory: 'OpenViewPane',
        eventAction: 'Diff',
    });
}

// TODO: de-dupe with compiler etc
Diff.prototype.resize = function () {
    var topBarHeight = this.topBar.outerHeight(true);
    this.outputEditor.layout({
        width: this.domRoot.width(),
        height: this.domRoot.height() - topBarHeight,
    });
};

Diff.prototype.onDiffSelect = function (id) {
    this.requestResendResult(id);
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

Diff.prototype.onExecuteResult = function (id, compiler, result) {
    var compileResult = _.assign({}, result.buildResult);
    compileResult.execResult = {
        code: result.code,
        stdout: result.stdout,
        stderr: result.stderr,
    };

    this.onCompileResult(id + '_exec', compiler, compileResult);
};

Diff.prototype.initButtons = function (state) {
    this.fontScale = new FontScale(this.domRoot, state, this.outputEditor);

    this.topBar = this.domRoot.find('.top-bar');
};

Diff.prototype.initCallbacks = function () {
    this.fontScale.on('change', _.bind(this.updateState, this));

    this.eventHub.on('compileResult', this.onCompileResult, this);
    this.eventHub.on('executeResult', this.onExecuteResult, this);
    this.eventHub.on('compiler', this.onCompiler, this);
    this.eventHub.on('compilerClose', this.onCompilerClose, this);
    this.eventHub.on('executor', this.onExecutor, this);
    this.eventHub.on('executorClose', this.onExecutorClose, this);
    this.eventHub.on('settingsChange', this.onSettingsChange, this);
    this.eventHub.on('themeChange', this.onThemeChange, this);
    this.container.on('destroy', function () {
        this.eventHub.unsubscribe();
        this.outputEditor.dispose();
    }, this);
    this.container.on('resize', this.resize, this);
    this.container.on('shown', this.resize, this);

    this.requestResendResult(this.lhs.id);
    this.requestResendResult(this.rhs.id);

    this.eventHub.emit('findCompilers');
    this.eventHub.emit('findExecutors');

    this.eventHub.emit('requestTheme');
    this.eventHub.emit('requestSettings');
};

Diff.prototype.requestResendResult = function (id) {
    if (typeof id === 'string') {
        var p = id.indexOf('_exec');
        if (p !== -1) {
            var execId = parseInt(id.substr(0, p));
            this.eventHub.emit('resendExecution', execId);
        }
    } else {
        this.eventHub.emit('resendCompilation', id);
    }
};

Diff.prototype.onCompiler = function (id, compiler, options, editorId) {
    if (!compiler) return;
    options = options || '';
    var name = compiler.name + ' ' + options;
    // TODO: selectize doesn't play nicely with CSS tricks for truncation; this is the best I can do
    // There's a plugin at: http://www.benbybenjacobs.com/blog/2014/04/09/no-wrap-plugin-for-selectize-dot-js
    // but it doesn't look easy to integrate.
    var maxLength = 30;
    if (name.length > maxLength - 3) name = name.substr(0, maxLength - 3) + '...';
    this.compilers[id] = {
        id: id,
        name: name,
        options: options,
        editorId: editorId,
        compiler: compiler,
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

Diff.prototype.onExecutor = function (id, compiler, options, editorId) {
    this.onCompiler(id + '_exec', compiler, options, editorId);
};

Diff.prototype.onCompilerClose = function (id) {
    delete this.compilers[id];
    this.updateCompilers();
};

Diff.prototype.onExecutorClose = function (id) {
    this.onCompilerClose(id + '_exec');
};

Diff.prototype.updateCompilerNames = function () {
    var name = 'Diff';
    if (this.lhs.compiler && this.rhs.compiler)
        name += ' ' + this.lhs.compiler.name + ' vs ' + this.rhs.compiler.name;
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

    this.selectize.lhsdifftype.setValue(this.lhs.difftype || DiffType_ASM);
    this.selectize.rhsdifftype.setValue(this.rhs.difftype || DiffType_ASM);
};

Diff.prototype.updateState = function () {
    var state = {
        lhs: this.lhs.id,
        rhs: this.rhs.id,
        lhsdifftype: this.lhs.difftype,
        rhsdifftype: this.rhs.difftype,
    };
    this.fontScale.addState(state);
    this.container.setState(state);
};

Diff.prototype.onThemeChange = function (newTheme) {
    if (this.outputEditor)
        this.outputEditor.updateOptions({theme: newTheme.monaco});
};

Diff.prototype.onSettingsChange = function (newSettings) {
    this.outputEditor.updateOptions({
        minimap: {
            enabled: newSettings.showMinimap,
        },
        fontFamily: newSettings.editorsFFont,
        fontLigatures: newSettings.editorsFLigatures,
    });
};

module.exports = {
    Diff: Diff,
    getComponent: function (lhs, rhs) {
        return {
            type: 'component',
            componentName: 'diff',
            componentState: {lhs: lhs, rhs: rhs},
        };
    },
};
