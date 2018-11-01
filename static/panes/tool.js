// Copyright (c) 2018, Compiler Explorer Team
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
        escapeXML: true
    });
}

function Tool(hub, container, state) {
    this.container = container;
    this.compilerId = state.compiler;
    this.editorId = state.editor;
    this.toolId = state.toolId;
    this.toolName = "Tool";
    this.compilerService = hub.compilerService;
    this.eventHub = hub.createEventHub();
    this.domRoot = container.getElement();
    this.domRoot.html($('#tool-output').html());
    this.contentRoot = this.domRoot.find('.content');
    this.optionsToolbar = this.domRoot.find('.options-toolbar');
    this.compilerName = "";
    this.fontScale = new FontScale(this.domRoot, state, ".content");
    this.fontScale.on('change', _.bind(function () {
        this.saveState();
    }, this));
    this.normalAnsiToHtml = makeAnsiToHtml();
    this.errorAnsiToHtml = makeAnsiToHtml('red');

    this.optionsField = this.domRoot.find('input.options');

    this.initButtons();
    this.options = new Toggles(this.domRoot.find('.options'), state);
    this.options.on('change', _.bind(this.onOptionsChange, this));

    this.initArgs(state);

    this.container.on('resize', this.resize, this);
    this.container.on('shown', this.resize, this);
    this.container.on('destroy', this.close, this);

    this.eventHub.on('compileResult', this.onCompileResult, this);
    this.eventHub.on('compilerClose', this.onCompilerClose, this);

    this.onOptionsChange();
    this.updateCompilerName();

    ga.proxy('send', {
        hitType: 'event',
        eventCategory: 'OpenViewPane',
        eventAction: 'Tool'
    });

    this.eventHub.emit('toolOpened', this.compilerId, this.currentState());
}

Tool.prototype.initArgs = function (state) {
    if (this.optionsField) {
        var optionsChange = _.debounce(_.bind(function (e) {
            this.onOptionsChange($(e.target).val());

            this.eventHub.emit('toolSettingsChange', this.compilerId);
        }, this), 800);

        this.optionsField
            .on('change', optionsChange)
            .on('keyup', optionsChange);

        if (state.args) {
            this.optionsField.val(state.args);
        }
    }
};

Tool.prototype.getInputArgs = function () {
    if (this.optionsField) {
        return this.optionsField.val();
    } else {
        return "";
    }
};

Tool.prototype.getEffectiveOptions = function () {
    return this.options.get();
};

Tool.prototype.resize = function () {
    this.contentRoot.height(this.domRoot.height() - this.optionsToolbar.height() - 5);
};

Tool.prototype.onOptionsChange = function () {
    var options = this.getEffectiveOptions();
    this.contentRoot.toggleClass('wrap', options.wrap);
    this.wrapButton.prop('title', '[' + (options.wrap ? 'ON' : 'OFF') + '] ' + this.wrapTitle);

    this.saveState();
};

Tool.prototype.initButtons = function () {
    this.wrapButton = this.domRoot.find('.wrap-lines');
    this.wrapTitle = this.wrapButton.prop('title');
};

Tool.prototype.currentState = function () {
    var options = this.getEffectiveOptions();
    var state = {
        compiler: this.compilerId,
        editor: this.editorId,
        wrap: options.wrap,
        toolId: this.toolId,
        args: this.getInputArgs()
    };
    this.fontScale.addState(state);
    return state;
};

Tool.prototype.saveState = function () {
    this.container.setState(this.currentState());
};

Tool.prototype.onCompileResult = function (id, compiler, result) {
    if (id !== this.compilerId) return;
    if (compiler) this.compilerName = compiler.name;

    this.contentRoot.empty();

    var toolResult = null;
    if (result && result.tools) {
        toolResult = _.find(result.tools, function (tool) {
            return (tool.id === this.toolId);
        }, this);
    }

    if (toolResult) {
        _.each((toolResult.stdout || []).concat(toolResult.stderr || []), function (obj) {
            this.add(this.normalAnsiToHtml.toHtml(obj.text), obj.tag ? obj.tag.line : obj.line);
        }, this);
    
        this.toolName = toolResult.name;
        this.add(this.toolName + " returned: " + toolResult.code);
    
        this.updateCompilerName();

        if (toolResult.sourcechanged) {
            this.eventHub.emit('newSource', this.editorId, toolResult.newsource);
        }
    } else {
        this.add("No tool result");
    }
};

Tool.prototype.add = function (msg, lineNum) {
    var elem = $('<p></p>').appendTo(this.contentRoot);
    if (lineNum) {
        elem.html(
            $('<a></a>')
                .prop('href', 'javascript:;')
                .html(msg)
                .click(_.bind(function (e) {
                    this.eventHub.emit('editorSetDecoration', this.editorId, lineNum, true);
                    e.preventDefault();
                    return false;
                }, this))
                .on('mouseover', _.bind(function () {
                    this.eventHub.emit('editorSetDecoration', this.editorId, lineNum, false);
                }, this))
        );
    } else {
        elem.html(msg);
    }
};

Tool.prototype.updateCompilerName = function () {
    var name = this.toolName + " #" + this.compilerId;
    if (this.compilerName) name += " with " + this.compilerName;
    this.container.setTitle(name);
};

Tool.prototype.onCompilerClose = function (id) {
    if (id === this.compilerId) {
        this.close();
        _.defer(function (self) {
            self.container.close();
        }, this);
    }
};

Tool.prototype.close = function () {
    this.eventHub.emit('toolClosed', this.compilerId, this.currentState());
    this.eventHub.unsubscribe();
};

module.exports = {
    Tool: Tool
};
