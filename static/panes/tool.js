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

var _ = require('underscore');
var $ = require('jquery');
var FontScale = require('../fontscale');
var AnsiToHtml = require('../ansi-to-html');
var Toggles = require('../toggles');
var ga = require('../analytics');
var monaco = require('monaco-editor');
var monacoConfig = require('../monaco-config');
var ceoptions = require('../options');
require('../modes/asm6502-mode');

function makeAnsiToHtml(color) {
    return new AnsiToHtml({
        fg: color ? color : '#333',
        bg: '#f5f5f5',
        stream: true,
        escapeXML: true,
    });
}

function Tool(hub, container, state) {
    this.container = container;
    this.compilerId = state.compiler;
    this.editorId = state.editor;
    this.toolId = state.toolId;
    this.toolName = 'Tool';
    this.compilerService = hub.compilerService;
    this.eventHub = hub.createEventHub();
    this.domRoot = container.getElement();
    this.domRoot.html($('#tool-output').html());
    this.editorContentRoot = this.domRoot.find('.monaco-placeholder');
    this.plainContentRoot = this.domRoot.find('pre.content');
    this.optionsToolbar = this.domRoot.find('.options-toolbar');
    this.badLangToolbar = this.domRoot.find('.bad-lang');
    this.compilerName = '';
    this.normalAnsiToHtml = makeAnsiToHtml();
    this.errorAnsiToHtml = makeAnsiToHtml('red');

    this.optionsField = this.domRoot.find('input.options');
    this.stdinField = this.domRoot.find('textarea.tool-stdin');

    this.outputEditor = monaco.editor.create(this.editorContentRoot[0], monacoConfig.extendConfig({
        readOnly: true,
        language: 'text',
        fontFamily: 'courier new',
        lineNumbersMinChars: 5,
        renderIndentGuides: false,
    }));

    this.fontScale = new FontScale(this.domRoot, state, '.content');
    this.fontScale.on('change', _.bind(function () {
        this.saveState();
    }, this));

    this.initButtons(state);
    this.options = new Toggles(this.domRoot.find('.options'), state);
    this.options.on('change', _.bind(this.onOptionsChange, this));

    this.initArgs(state);
    this.initCallbacks();

    this.onOptionsChange();
    this.updateCompilerName();

    ga.proxy('send', {
        hitType: 'event',
        eventCategory: 'OpenViewPane',
        eventAction: 'Tool',
    });

    this.eventHub.emit('toolOpened', this.compilerId, this.currentState());
    this.eventHub.emit('requestSettings');
}

Tool.prototype.initCallbacks = function () {
    this.container.on('resize', this.resize, this);
    this.container.on('shown', this.resize, this);
    this.container.on('destroy', this.close, this);

    this.eventHub.on('compileResult', this.onCompileResult, this);
    this.eventHub.on('compilerClose', this.onCompilerClose, this);
    this.eventHub.on('settingsChange', this.onSettingsChange, this);
    this.eventHub.on('languageChange', this.onLanguageChange, this);

    this.toggleArgs.on('click', _.bind(function () {
        this.togglePanel(this.toggleArgs, this.panelArgs);
    }, this));

    this.toggleStdin.on('click', _.bind(function () {
        this.togglePanel(this.toggleStdin, this.panelStdin);
    }, this));

    if (MutationObserver !== undefined) {
        new MutationObserver(_.bind(this.resize, this)).observe(this.stdinField[0], {
            attributes: true, attributeFilter: ['style'],
        });
    }
};

Tool.prototype.onLanguageChange = function (editorId, newLangId) {
    if (this.editorId === editorId) {
        var tools = ceoptions.tools[newLangId];
        this.toggleUsable(tools && tools[this.toolId]);
    }
};

Tool.prototype.toggleUsable = function (isUsable) {
    if (isUsable) {
        this.plainContentRoot.css('opacity', '1');
        this.badLangToolbar.hide();
        this.optionsToolbar.show();
    } else {
        this.plainContentRoot.css('opacity', '0.5');
        this.optionsToolbar.hide();
        this.badLangToolbar.show();
    }
};

Tool.prototype.onSettingsChange = function (newSettings) {
    this.outputEditor.updateOptions({
        contextmenu: newSettings.useCustomContextMenu,
        minimap: {
            enabled: newSettings.showMinimap,
        },
        fontFamily: newSettings.editorsFFont,
        fontLigatures: newSettings.editorsFLigatures,
    });
};

Tool.prototype.initArgs = function (state) {
    var optionsChange = _.debounce(_.bind(function (e) {
        this.onOptionsChange($(e.target).val());

        this.eventHub.emit('toolSettingsChange', this.compilerId);
    }, this), 800);

    if (this.optionsField) {
        this.optionsField
            .on('change', optionsChange)
            .on('keyup', optionsChange);

        if (state.args) {
            this.optionsField.val(state.args);
        }
    }

    if (this.stdinField) {
        this.stdinField
            .on('change', optionsChange)
            .on('keyup', optionsChange);

        if (state.stdin) {
            this.stdinField.val(state.stdin);
        }
    }
};

Tool.prototype.getInputArgs = function () {
    if (this.optionsField) {
        return this.optionsField.val();
    } else {
        return '';
    }
};

Tool.prototype.getInputStdin = function () {
    if (this.stdinField) {
        return this.stdinField.val();
    } else {
        return '';
    }
};

Tool.prototype.getEffectiveOptions = function () {
    return this.options.get();
};

Tool.prototype.resize = function () {
    var barsHeight = this.optionsToolbar.outerHeight() + 2;
    if (!this.panelArgs.hasClass('d-none')) {
        barsHeight += this.panelArgs.outerHeight();
    }
    if (!this.panelStdin.hasClass('d-none')) {
        barsHeight += this.panelStdin.outerHeight();
    }

    this.outputEditor.layout({
        width: this.domRoot.width(),
        height: this.domRoot.height() - barsHeight,
    });

    this.plainContentRoot.height(this.domRoot.height() - barsHeight);
};

Tool.prototype.onOptionsChange = function () {
    var options = this.getEffectiveOptions();
    this.plainContentRoot.toggleClass('wrap', options.wrap);
    this.wrapButton.prop('title', '[' + (options.wrap ? 'ON' : 'OFF') + '] ' + this.wrapTitle);

    this.saveState();
};

Tool.prototype.initButtons = function (state) {
    this.wrapButton = this.domRoot.find('.wrap-lines');
    this.wrapTitle = this.wrapButton.prop('title');

    this.panelArgs = this.domRoot.find('.panel-args');
    this.panelStdin = this.domRoot.find('.panel-stdin');

    this.initToggleButtons(state);
};

Tool.prototype.initToggleButtons = function (state) {
    this.toggleArgs = this.domRoot.find('.toggle-args');
    this.toggleStdin = this.domRoot.find('.toggle-stdin');

    if (state.argsPanelShown === true) {
        this.showPanel(this.toggleArgs, this.panelArgs);
    }

    if (state.stdinPanelShown === true) {
        this.showPanel(this.toggleStdin, this.panelStdin);
    }
};

Tool.prototype.showPanel = function (button, panel) {
    panel.removeClass('d-none');
    button.addClass('active');
    this.resize();
};

Tool.prototype.hidePanel = function (button, panel) {
    panel.addClass('d-none');
    button.removeClass('active');
    this.resize();
};

Tool.prototype.togglePanel = function (button, panel) {
    if (panel.hasClass('d-none')) {
        this.showPanel(button, panel);
    } else {
        this.hidePanel(button, panel);
    }
    this.saveState();
};

Tool.prototype.currentState = function () {
    var options = this.getEffectiveOptions();
    var state = {
        compiler: this.compilerId,
        editor: this.editorId,
        wrap: options.wrap,
        toolId: this.toolId,
        args: this.getInputArgs(),
        stdin: this.getInputStdin(),
        stdinPanelShown: !this.panelStdin.hasClass('d-none'),
        argsPanelShow: !this.panelArgs.hasClass('d-none'),
    };
    this.fontScale.addState(state);
    return state;
};

Tool.prototype.saveState = function () {
    this.container.setState(this.currentState());
};

Tool.prototype.setLanguage = function (languageId) {
    if (languageId) {
        this.options.enableToggle('wrap', false);
        monaco.editor.setModelLanguage(this.outputEditor.getModel(), languageId);
        this.outputEditor.setValue('');
        this.fontScale.setTarget(this.outputEditor);
        $(this.plainContentRoot).hide();
        $(this.editorContentRoot).show();
    } else {
        this.options.enableToggle('wrap', true);
        this.plainContentRoot.empty();
        this.fontScale.setTarget('.content');
        $(this.editorContentRoot).hide();
        $(this.plainContentRoot).show();
    }
};

Tool.prototype.onCompileResult = function (id, compiler, result) {
    try{
        if (id !== this.compilerId) return;
        if (compiler) this.compilerName = compiler.name;

        var foundTool = _.find(compiler.tools, function (tool) {
            return (tool.tool.id === this.toolId);
        }, this);

        this.toggleUsable(foundTool);

        var toolResult = null;
        if (result && result.tools) {
            toolResult = _.find(result.tools, function (tool) {
                return (tool.id === this.toolId);
            }, this);
        }

        var toolInfo = null;
        if (compiler && compiler.tools) {
            toolInfo = _.find(compiler.tools, function (tool) {
                return (tool.tool.id === this.toolId);
            }, this);
        }

        if (toolInfo) {
            this.toggleStdin.prop('disabled', false);

            if (toolInfo.tool.stdinHint) {
                this.stdinField.prop('placeholder', toolInfo.tool.stdinHint);
                if (toolInfo.tool.stdinHint === 'disabled') {
                    this.toggleStdin.prop('disabled', true);
                } else {
                    this.showPanel(this.toggleStdin, this.panelStdin);
                }
            } else {
                this.stdinField.prop('placeholder', 'Tool stdin...');
            }
        }

        if (toolResult) {
            if (toolResult.languageId && (toolResult.languageId === 'stderr')) {
                toolResult.languageId = false;
            }

            this.setLanguage(toolResult.languageId);

            if (toolResult.languageId) {
                this.setEditorContent(_.pluck(toolResult.stdout, 'text').join('\n'));
            } else {
                _.each((toolResult.stdout || []).concat(toolResult.stderr || []), function (obj) {
                    if (obj.text === '') {
                        this.add('<br/>');
                    } else {
                        this.add(this.normalAnsiToHtml.toHtml(obj.text), obj.tag ? obj.tag.line : obj.line);
                    }
                }, this);
            }

            this.toolName = toolResult.name;
            this.updateCompilerName();

            if (toolResult.sourcechanged) {
                this.eventHub.emit('newSource', this.editorId, toolResult.newsource);
            }
        } else {
            this.setEditorContent('No tool result');
        }
    } catch(e) {
        this.setLanguage(false);
        this.add('javascript error: ' + e.message);
    }
};

Tool.prototype.add = function (msg, lineNum) {
    var elem = $('<div/>').appendTo(this.plainContentRoot);
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

Tool.prototype.setEditorContent = function (content) {
    if (!this.outputEditor || !this.outputEditor.getModel()) return;
    var editorModel = this.outputEditor.getModel();
    var visibleRanges = this.outputEditor.getVisibleRanges();
    var currentTopLine = visibleRanges.length > 0 ? visibleRanges[0].startLineNumber : 1;
    editorModel.setValue(content);
    this.outputEditor.revealLine(currentTopLine);
    this.setNormalContent();
};

Tool.prototype.setNormalContent = function () {
    this.outputEditor.updateOptions({
        lineNumbers: true,
        codeLens: false,
    });
    if (this.codeLensProvider) {
        this.codeLensProvider.dispose();
    }
};

Tool.prototype.updateCompilerName = function () {
    var name = this.toolName + ' #' + this.compilerId;
    if (this.compilerName) name += ' with ' + this.compilerName;
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
    this.outputEditor.dispose();
};

module.exports = {
    Tool: Tool,
};
