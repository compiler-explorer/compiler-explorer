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
var FontScale = require('../widgets/fontscale').FontScale;
var AnsiToHtml = require('../ansi-to-html').Filter;
var Toggles = require('../widgets/toggles').Toggles;
var ga = require('../analytics').ga;
var Components = require('../components');
var monaco = require('monaco-editor');
var monacoConfig = require('../monaco-config');
var ceoptions = require('../options').options;
var utils = require('../utils');
var PaneRenaming = require('../widgets/pane-renaming').PaneRenaming;

function makeAnsiToHtml(color) {
    return new AnsiToHtml({
        fg: color ? color : '#333',
        bg: '#f5f5f5',
        stream: true,
        escapeXML: true,
    });
}

function Tool(hub, container, state) {
    this.hub = hub;
    this.container = container;
    this.compilerId = state.compiler;
    this.editorId = state.editor;
    this.treeId = state.tree;
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
    this.monacoStdin = state.monacoStdin || false;
    this.monacoEditorOpen = state.monacoEditorOpen || false;
    this.monacoEditorHasBeenAutoOpened = state.monacoEditorHasBeenAutoOpened || false;
    this.monacoStdinField = '';
    this.normalAnsiToHtml = makeAnsiToHtml();

    this.optionsField = this.domRoot.find('input.options');
    this.localStdinField = this.domRoot.find('textarea.tool-stdin');

    this.outputEditor = monaco.editor.create(
        this.editorContentRoot[0],
        monacoConfig.extendConfig({
            readOnly: true,
            language: 'text',
            fontFamily: 'courier new',
            lineNumbersMinChars: 5,
            guides: false,
        })
    );

    this.fontScale = new FontScale(this.domRoot, state, '.content');
    this.fontScale.on(
        'change',
        _.bind(function () {
            this.saveState();
        }, this)
    );

    this.createToolInputView = _.bind(function () {
        return Components.getToolInputViewWith(this.compilerId, this.toolId, this.toolName);
    }, this);

    this.initButtons(state);
    this.options = new Toggles(this.domRoot.find('.options'), state);
    this.options.on('change', _.bind(this.onOptionsChange, this));

    this.paneRenaming = new PaneRenaming(this, state);

    this.initArgs(state);
    this.initCallbacks();

    this.onOptionsChange();

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

    this.paneRenaming.on('renamePane', this.saveState.bind(this));

    this.eventHub.on('compileResult', this.onCompileResult, this);
    this.eventHub.on('compilerClose', this.onCompilerClose, this);
    this.eventHub.on('settingsChange', this.onSettingsChange, this);
    this.eventHub.on('languageChange', this.onLanguageChange, this);
    this.eventHub.on('toolInputChange', this.onToolInputChange, this);
    this.eventHub.on('toolInputViewClosed', this.onToolInputViewClosed, this);

    this.toggleArgs.on(
        'click',
        _.bind(function () {
            this.togglePanel(this.toggleArgs, this.panelArgs);
        }, this)
    );

    this.toggleStdin.on(
        'click',
        _.bind(function () {
            if (!this.monacoStdin) {
                this.togglePanel(this.toggleStdin, this.panelStdin);
            } else {
                if (!this.monacoEditorOpen) {
                    this.openMonacoEditor();
                } else {
                    this.monacoEditorOpen = false;
                    this.toggleStdin.removeClass('active');
                    this.eventHub.emit('toolInputViewCloseRequest', this.compilerId, this.toolId);
                }
            }
        }, this)
    );

    if (MutationObserver !== undefined) {
        new MutationObserver(_.bind(this.resize, this)).observe(this.localStdinField[0], {
            attributes: true,
            attributeFilter: ['style'],
        });
    }
};

Tool.prototype.onLanguageChange = function (editorId, newLangId) {
    if (this.editorId && this.editorId === editorId) {
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
        codeLensFontFamily: newSettings.editorsFFont,
        fontLigatures: newSettings.editorsFLigatures,
    });
};

Tool.prototype.initArgs = function (state) {
    var optionsChange = _.debounce(
        _.bind(function (e) {
            this.onOptionsChange($(e.target).val());

            this.eventHub.emit('toolSettingsChange', this.compilerId);
        }, this),
        800
    );

    if (this.optionsField) {
        this.optionsField.on('change', optionsChange).on('keyup', optionsChange);

        if (state.args) {
            this.optionsField.val(state.args);
        }
    }

    if (this.localStdinField) {
        this.localStdinField.on('change', optionsChange).on('keyup', optionsChange);

        if (state.stdin) {
            if (!this.monacoStdin) {
                this.localStdinField.val(state.stdin);
            } else {
                this.eventHub.emit('setToolInput', this.compilerId, this.toolId, state.stdin);
            }
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

Tool.prototype.onToolInputChange = function (compilerId, toolId, input) {
    if (this.compilerId === compilerId && this.toolId === toolId) {
        this.monacoStdinField = input;
        this.onOptionsChange();
        this.eventHub.emit('toolSettingsChange', this.compilerId);
    }
};

Tool.prototype.onToolInputViewClosed = function (compilerId, toolId, input) {
    if (this.compilerId === compilerId && this.toolId === toolId) {
        // Duplicate close messages have been seen, with the second having no value.
        // If we have a current value and the new value is empty, ignore the message.
        if (this.monacoStdinField && input) {
            this.monacoStdinField = input;
            this.monacoEditorOpen = false;
            this.toggleStdin.removeClass('active');

            this.onOptionsChange();
            this.eventHub.emit('toolSettingsChange', this.compilerId);
        }
    }
};

Tool.prototype.getInputStdin = function () {
    if (!this.monacoStdin) {
        if (this.localStdinField) {
            return this.localStdinField.val();
        } else {
            return '';
        }
    } else {
        return this.monacoStdinField;
    }
};

Tool.prototype.openMonacoEditor = function () {
    this.monacoEditorHasBeenAutoOpened = true; // just in case we get here in an unexpected way
    this.monacoEditorOpen = true;
    this.toggleStdin.addClass('active');
    var insertPoint =
        this.hub.findParentRowOrColumn(this.container) || this.container.layoutManager.root.contentItems[0];
    insertPoint.addChild(this.createToolInputView);
    this.onOptionsChange();
    this.eventHub.emit('setToolInput', this.compilerId, this.toolId, this.monacoStdinField);
};

Tool.prototype.getEffectiveOptions = function () {
    return this.options.get();
};

Tool.prototype.resize = function () {
    utils.updateAndCalcTopBarHeight(this.domRoot, this.optionsToolbar, this.hideable);
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

    this.hideable = this.domRoot.find('.hideable');

    this.initToggleButtons(state);
};

Tool.prototype.initToggleButtons = function (state) {
    this.toggleArgs = this.domRoot.find('.toggle-args');
    this.toggleStdin = this.domRoot.find('.toggle-stdin');

    if (state.argsPanelShown === true) {
        this.showPanel(this.toggleArgs, this.panelArgs);
    }

    if (state.stdinPanelShown === true) {
        if (!this.monacoStdin) {
            this.showPanel(this.toggleStdin, this.panelStdin);
        } else {
            if (!this.monacoEditorOpen) {
                this.openMonacoEditor();
            }
        }
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
        tree: this.treeId,
        wrap: options.wrap,
        toolId: this.toolId,
        args: this.getInputArgs(),
        stdin: this.getInputStdin(),
        stdinPanelShown:
            (this.monacoStdin && this.monacoEditorOpen) || (this.panelStdin && !this.panelStdin.hasClass('d-none')),
        monacoStdin: this.monacoStdin,
        monacoEditorOpen: this.monacoEditorOpen,
        monacoEditorHasBeenAutoOpened: this.monacoEditorHasBeenAutoOpened,
        argsPanelShow: !this.panelArgs.hasClass('d-none'),
    };
    this.paneRenaming.addState(state);
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
    try {
        if (id !== this.compilerId) return;
        if (compiler) this.compilerName = compiler.name;

        var foundTool = _.find(
            compiler.tools,
            function (tool) {
                return tool.tool.id === this.toolId;
            },
            this
        );

        this.toggleUsable(foundTool);

        var toolResult = null;
        if (result && result.tools) {
            toolResult = _.find(
                result.tools,
                function (tool) {
                    return tool.id === this.toolId;
                },
                this
            );
        } else if (result && result.result && result.result.tools) {
            toolResult = _.find(
                result.result.tools,
                function (tool) {
                    return tool.id === this.toolId;
                },
                this
            );
        }

        var toolInfo = null;
        if (compiler && compiler.tools) {
            toolInfo = _.find(
                compiler.tools,
                function (tool) {
                    return tool.tool.id === this.toolId;
                },
                this
            );
        }

        if (toolInfo) {
            this.toggleStdin.prop('disabled', false);

            if (this.monacoStdin && !this.monacoEditorOpen && !this.monacoEditorHasBeenAutoOpened) {
                this.monacoEditorHasBeenAutoOpened = true;
                this.openMonacoEditor();
            } else if (!this.monacoStdin && toolInfo.tool.stdinHint) {
                this.localStdinField.prop('placeholder', toolInfo.tool.stdinHint);
                if (toolInfo.tool.stdinHint === 'disabled') {
                    this.toggleStdin.prop('disabled', true);
                } else {
                    this.showPanel(this.toggleStdin, this.panelStdin);
                }
            } else {
                this.localStdinField.prop('placeholder', 'Tool stdin...');
            }
        }

        if (toolResult) {
            if (toolResult.languageId && toolResult.languageId === 'stderr') {
                toolResult.languageId = false;
            }

            this.setLanguage(toolResult.languageId);

            if (toolResult.languageId) {
                this.setEditorContent(_.pluck(toolResult.stdout, 'text').join('\n'));
            } else {
                _.each(
                    (toolResult.stdout || []).concat(toolResult.stderr || []),
                    function (obj) {
                        if (obj.text === '') {
                            this.add('<br/>');
                        } else {
                            this.add(this.normalAnsiToHtml.toHtml(obj.text), obj.tag ? obj.tag.line : obj.line);
                        }
                    },
                    this
                );
            }

            this.toolName = toolResult.name;
            this.updateTitle();

            if (toolResult.sourcechanged && this.editorId) {
                this.eventHub.emit('newSource', this.editorId, toolResult.newsource);
            }
        } else {
            this.setEditorContent('No tool result');
        }
    } catch (e) {
        this.setLanguage(false);
        this.add('javascript error: ' + e.message);
    }
};

Tool.prototype.add = function (msg, lineNum) {
    var elem = $('<div/>').appendTo(this.plainContentRoot);
    if (lineNum && this.editorId) {
        elem.html(
            $('<a></a>')
                .prop('href', 'javascript:;')
                .html(msg)
                .on(
                    'click',
                    _.bind(function (e) {
                        this.eventHub.emit('editorSetDecoration', this.editorId, lineNum, true);
                        e.preventDefault();
                        return false;
                    }, this)
                )
                .on(
                    'mouseover',
                    _.bind(function () {
                        this.eventHub.emit('editorSetDecoration', this.editorId, lineNum, false);
                    }, this)
                )
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

Tool.prototype.getPaneName = function () {
    var name = this.toolName + ' #' + this.compilerId;
    if (this.compilerName) name += ' with ' + this.compilerName;
    return name;
};

Tool.prototype.updateTitle = function () {
    var name = this.paneName ? this.paneName : this.getPaneName();
    this.container.setTitle(_.escape(name));
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
