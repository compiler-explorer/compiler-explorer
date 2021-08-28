// Copyright (c) 2016, Compiler Explorer Authors
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

var _ = require('underscore');
var $ = require('jquery');
var FontScale = require('../fontscale');
var AnsiToHtml = require('../ansi-to-html');
var Toggles = require('../toggles');
var ga = require('../analytics');

function makeAnsiToHtml(color) {
    return new AnsiToHtml({
        fg: color ? color : '#333',
        bg: '#f5f5f5',
        stream: true,
        escapeXML: true,
    });
}

function Output(hub, container, state) {
    this.container = container;
    this.compilerId = state.compiler;
    this.editorId = state.editor;
    this.treeId = state.tree;
    this.hub = hub;
    this.eventHub = hub.createEventHub();
    this.domRoot = container.getElement();
    this.domRoot.html($('#compiler-output').html());
    this.contentRoot = this.domRoot.find('.content');
    this.optionsToolbar = this.domRoot.find('.options-toolbar');
    this.compilerName = '';
    this.fontScale = new FontScale(this.domRoot, state, '.content');
    this.fontScale.on('change', _.bind(function () {
        this.saveState();
    }, this));
    this.normalAnsiToHtml = makeAnsiToHtml();
    this.errorAnsiToHtml = makeAnsiToHtml('red');

    this.initButtons();
    this.initCallbacks(state);

    this.onOptionsChange();
    this.updateCompilerName();
    ga.proxy('send', {
        hitType: 'event',
        eventCategory: 'OpenViewPane',
        eventAction: 'Output',
    });
}

Output.prototype.initCallbacks = function (state) {
    this.options = new Toggles(this.domRoot.find('.options'), state);
    this.options.on('change', _.bind(this.onOptionsChange, this));

    this.container.on('resize', this.resize, this);
    this.container.on('shown', this.resize, this);
    this.container.on('destroy', this.close, this);

    this.eventHub.on('compiling', this.onCompiling, this);
    this.eventHub.on('compileResult', this.onCompileResult, this);
    this.eventHub.on('compilerClose', this.onCompilerClose, this);
    this.eventHub.emit('outputOpened', this.compilerId);
};

Output.prototype.getEffectiveOptions = function () {
    return this.options.get();
};

Output.prototype.resize = function () {
    this.contentRoot.height(this.domRoot.height() - this.optionsToolbar.height() - 5);
};

Output.prototype.onOptionsChange = function () {
    var options = this.getEffectiveOptions();
    this.contentRoot.toggleClass('wrap', options.wrap);
    this.wrapButton.prop('title', '[' + (options.wrap ? 'ON' : 'OFF') + '] ' + this.wrapTitle);

    this.saveState();
};

Output.prototype.initButtons = function () {
    this.wrapButton = this.domRoot.find('.wrap-lines');
    this.wrapTitle = this.wrapButton.prop('title');
};

Output.prototype.currentState = function () {
    var options = this.getEffectiveOptions();
    var state = {
        compiler: this.compilerId,
        editor: this.editorId,
        tree: this.treeId,
        wrap: options.wrap,
    };
    this.fontScale.addState(state);
    return state;
};

Output.prototype.saveState = function () {
    this.container.setState(this.currentState());
};

Output.prototype.addOutputLines = function (result) {
    _.each((result.stdout || []).concat(result.stderr || []), function (obj) {
        var lineNumber = obj.tag ? obj.tag.line : obj.line;
        var columnNumber = obj.tag ? obj.tag.column : -1;
        if (obj.text === '') {
            this.add('<br/>');
        } else {
            this.add(this.normalAnsiToHtml.toHtml(obj.text), lineNumber, columnNumber, obj.tag ? obj.tag.file : false);
        }
    }, this);
};

Output.prototype.onCompiling = function (compilerId) {
    if (this.compilerId === compilerId) {
        this.setCompileStatus(true);
    }
};

Output.prototype.onCompileResult = function (id, compiler, result) {
    if (id !== this.compilerId) return;
    if (compiler) this.compilerName = compiler.name;

    this.contentRoot.empty();

    if (result.buildsteps) {
        _.each(result.buildsteps, _.bind(function (step) {
            this.add('Step ' + step.step + ' returned: ' + step.code);
            this.addOutputLines(step);
        }, this));
    } else {
        this.addOutputLines(result);
        if (!result.execResult) {
            this.add('Compiler returned: ' + result.code);
        } else {
            this.add('ASM generation compiler returned: ' + result.code);
            this.addOutputLines(result.execResult.buildResult);
            this.add('Execution build compiler returned: ' + result.execResult.buildResult.code);
        }
    }

    if (result.execResult && (result.execResult.didExecute || result.didExecute)) {
        this.add('Program returned: ' + result.execResult.code);
        if (result.execResult.stderr.length || result.execResult.stdout.length) {
            _.each(result.execResult.stderr, function (obj) {
                // Conserve empty lines as they are discarded by ansiToHtml
                if (obj.text === '') {
                    this.programOutput('<br/>');
                } else {
                    this.programOutput(this.errorAnsiToHtml.toHtml(obj.text), 'red');
                }
            }, this);

            _.each(result.execResult.stdout, function (obj) {
                // Conserve empty lines as they are discarded by ansiToHtml
                if (obj.text === '') {
                    this.programOutput('<br/>');
                } else {
                    this.programOutput(this.normalAnsiToHtml.toHtml(obj.text));
                }
            }, this);
        }
    }
    this.setCompileStatus(false);
    this.updateCompilerName();
};

Output.prototype.programOutput = function (msg, color) {
    var elem = $('<div/>').appendTo(this.contentRoot)
        .html(msg)
        .addClass('program-exec-output');

    if (color)
        elem.css('color', color);
};

Output.prototype.getEditorIdByFilename = function (filename) {
    var tree = this.hub.getTreeById(this.treeId);
    if (tree) {
        return tree.multifileService.getEditorIdByFilename(filename);
    }
    return false;
};

Output.prototype.emitEditorLinkLine = function (lineNum, column, filename, goto) {
    if (this.editorId) {
        this.eventHub.emit('editorLinkLine', this.editorId, lineNum, column, column + 1, goto);
    } else if (filename) {
        var editorId = this.getEditorIdByFilename(filename);
        if (editorId) {
            this.eventHub.emit('editorLinkLine', editorId, lineNum, column, column + 1, goto);
        }
    }
};

Output.prototype.add = function (msg, lineNum, column, filename) {
    var elem = $('<div/>').appendTo(this.contentRoot);
    if (lineNum) {
        elem.html(
            $('<span class="linked-compiler-output-line"></span>')
                .html(msg)
                .on('click', _.bind(function (e) {
                    this.emitEditorLinkLine(lineNum, column, filename, true);
                    // do not bring user to the top of index.html
                    // http://stackoverflow.com/questions/3252730
                    e.preventDefault();
                    return false;
                }, this))
                .on('mouseover', _.bind(function () {
                    this.emitEditorLinkLine(lineNum, column, filename, false);
                }, this))
        );
    } else {
        elem.html(msg);
    }
};

Output.prototype.getPaneName = function () {
    var name = 'Output';
    if (this.compilerName) name += ' of ' + this.compilerName;
    name += ' (Compiler #' + this.compilerId + ')';
    return name;
};

Output.prototype.updateCompilerName = function () {
    this.container.setTitle(this.getPaneName());
};

Output.prototype.onCompilerClose = function (id) {
    if (id === this.compilerId) {
        // We can't immediately close as an outer loop somewhere in GoldenLayout is iterating over
        // the hierarchy. We can't modify while it's being iterated over.
        this.close();
        _.defer(function (self) {
            self.container.close();
        }, this);
    }
};

Output.prototype.close = function () {
    this.eventHub.emit('outputClosed', this.compilerId);
    this.eventHub.unsubscribe();
};

Output.prototype.setCompileStatus = function (isCompiling) {
    this.contentRoot.toggleClass('compiling', isCompiling);
};

module.exports = {
    Output: Output,
};
