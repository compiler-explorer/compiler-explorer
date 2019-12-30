// Copyright (c) 2012, Matt Godbolt
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
var $ = require('jquery');
var _ = require('underscore');
var ga = require('../analytics');
var colour = require('../colour');
var Toggles = require('../toggles');
var FontScale = require('../fontscale');
var Promise = require('es6-promise').Promise;
var Components = require('../components');
var LruCache = require('lru-cache');
var options = require('../options');
var monaco = require('monaco-editor');
var Alert = require('../alert');
var bigInt = require('big-integer');
var local = require('../local');
var Libraries = require('../libs-widget');
require('../modes/asm-mode');
require('../modes/ptx-mode');

require('selectize');

var OpcodeCache = new LruCache({
    max: 64 * 1024,
    length: function (n) {
        return JSON.stringify(n).length;
    }
});

function patchOldFilters(filters) {
    if (filters === undefined) return undefined;
    // Filters are of the form {filter: true|false¸ ...}. In older versions, we used
    // to suppress the {filter:false} form. This means we can't distinguish between
    // "filter not on" and "filter not present". In the latter case we want to default
    // the filter. In the former case we want the filter off. Filters now don't suppress
    // but there are plenty of permalinks out there with no filters set at all. Here
    // we manually set any missing filters to 'false' to recover the old behaviour of
    // "if it's not here, it's off".
    _.each(['binary', 'labels', 'directives', 'commentOnly', 'trim', 'intel'], function (oldFilter) {
        if (filters[oldFilter] === undefined) filters[oldFilter] = false;
    });
    return filters;
}

var languages = options.languages;

function Compiler(hub, container, state) {
    this.container = container;
    this.hub = hub;
    this.eventHub = hub.createEventHub();
    this.compilerService = hub.compilerService;
    this.domRoot = container.getElement();
    this.domRoot.html($('#compiler').html());
    this.id = state.id || hub.nextCompilerId();
    this.sourceEditorId = state.source || 1;
    this.settings = JSON.parse(local.get('settings', '{}'));
    this.originalCompilerId = state.compiler;
    this.initLangAndCompiler(state);
    this.infoByLang = {};
    this.deferCompiles = hub.deferred;
    this.needsCompile = false;
    this.options = state.options || options.compileOptions[this.currentLangId];
    this.source = '';
    this.assembly = [];
    this.colours = [];
    this.lastResult = {};
    this.pendingRequestSentAt = 0;
    this.nextRequest = null;
    this.optViewOpen = false;
    this.cfgViewOpen = false;
    this.wantOptInfo = state.wantOptInfo;
    this.decorations = {};
    this.prevDecorations = [];
    this.labelDefinitions = {};
    this.alertSystem = new Alert();
    this.alertSystem.prefixMessage = "Compiler #" + this.id + ": ";

    this.awaitingInitialResults = false;
    this.selection = state.selection;

    this.linkedFadeTimeoutId = -1;
    this.toolsMenu = null;

    this.initButtons(state);

    this.outputEditor = monaco.editor.create(this.monacoPlaceholder[0], {
        scrollBeyondLastLine: false,
        readOnly: true,
        language: languages[this.currentLangId].monacoDisassembly || 'asm',
        fontFamily: this.settings.editorsFFont,
        glyphMargin: !options.embedded,
        fixedOverflowWidgets: true,
        minimap: {
            maxColumn: 80
        },
        lineNumbersMinChars: 1,
        renderIndentGuides: false,
        fontLigatures: this.settings.editorsFLigatures
    });

    this.codeLensProvider = null;
    this.fontScale = new FontScale(this.domRoot, state, this.outputEditor);

    this.compilerPicker.selectize({
        sortField: this.compilerService.getSelectizerOrder(),
        valueField: 'id',
        labelField: 'name',
        searchField: ['name'],
        optgroupField: 'group',
        optgroups: this.compilerService.getGroupsInUse(this.currentLangId),
        lockOptgroupOrder: true,
        options: _.map(this.getCurrentLangCompilers(), _.identity),
        items: this.compiler ? [this.compiler.id] : [],
        dropdownParent: 'body',
        closeAfterSelect: true
    }).on('change', _.bind(function (e) {
        var val = $(e.target).val();
        if (val) {
            ga.proxy('send', {
                hitType: 'event',
                eventCategory: 'SelectCompiler',
                eventAction: val
            });
            this.onCompilerChange(val);
        }
    }, this));

    this.compilerSelectizer = this.compilerPicker[0].selectize;

    this.initLibraries(state);

    this.initEditorActions();

    this.initCallbacks();
    // Handle initial settings
    this.onSettingsChange(this.settings);
    this.sendCompiler();
    this.updateCompilerInfo();
    this.updateButtons();
    this.saveState();
    ga.proxy('send', {
        hitType: 'event',
        eventCategory: 'OpenViewPane',
        eventAction: 'Compiler'
    });
}

Compiler.prototype.initLangAndCompiler = function (state) {
    var langId = state.lang;
    var compilerId = state.compiler;
    var result = this.compilerService.processFromLangAndCompiler(langId, compilerId);
    this.compiler = result.compiler;
    this.currentLangId = result.langId;
    this.updateLibraries();
};

Compiler.prototype.close = function () {
    this.eventHub.unsubscribe();
    this.eventHub.emit('compilerClose', this.id);
    this.outputEditor.dispose();
};

Compiler.prototype.initPanerButtons = function () {
    var outputConfig = _.bind(function () {
        return Components.getOutput(this.id, this.sourceEditorId);
    }, this);

    this.container.layoutManager.createDragSource(this.outputBtn, outputConfig);
    this.outputBtn.click(_.bind(function () {
        var insertPoint = this.hub.findParentRowOrColumn(this.container) ||
            this.container.layoutManager.root.contentItems[0];
        insertPoint.addChild(outputConfig);
    }, this));

    var cloneComponent = _.bind(function () {
        var currentState = this.currentState();
        // Delete the saved id to force a new one
        delete currentState.id;
        return {
            type: 'component',
            componentName: 'compiler',
            componentState: currentState
        };
    }, this);
    var createOptView = _.bind(function () {
        return Components.getOptViewWith(this.id, this.source, this.lastResult.optOutput, this.getCompilerName(),
            this.sourceEditorId);
    }, this);

    var createAstView = _.bind(function () {
        return Components.getAstViewWith(this.id, this.source, this.lastResult.astOutput, this.getCompilerName(),
            this.sourceEditorId);
    }, this);

    var createIrView = _.bind(function () {
        return Components.getIrViewWith(this.id, this.source, this.lastResult.irOutput, this.getCompilerName(),
            this.sourceEditorId);
    }, this);

    var createGccDumpView = _.bind(function () {
        return Components.getGccDumpViewWith(this.id, this.getCompilerName(), this.sourceEditorId,
            this.lastResult.gccDumpOutput);
    }, this);

    var createCfgView = _.bind(function () {
        return Components.getCfgViewWith(this.id, this.sourceEditorId);
    }, this);

    var panerDropdown = this.domRoot.find('.pane-dropdown');
    var togglePannerAdder = function () {
        panerDropdown.dropdown('toggle');
    };

    this.container.layoutManager
        .createDragSource(this.domRoot.find('.btn.add-compiler'), cloneComponent)
        ._dragListener.on('dragStart', togglePannerAdder);

    this.domRoot.find('.btn.add-compiler').click(_.bind(function () {
        var insertPoint = this.hub.findParentRowOrColumn(this.container) ||
            this.container.layoutManager.root.contentItems[0];
        insertPoint.addChild(cloneComponent);
    }, this));

    this.container.layoutManager
        .createDragSource(this.optButton, createOptView)
        ._dragListener.on('dragStart', togglePannerAdder);

    this.optButton.click(_.bind(function () {
        var insertPoint = this.hub.findParentRowOrColumn(this.container) ||
            this.container.layoutManager.root.contentItems[0];
        insertPoint.addChild(createOptView);
    }, this));

    this.container.layoutManager
        .createDragSource(this.astButton, createAstView)
        ._dragListener.on('dragStart', togglePannerAdder);

    this.astButton.click(_.bind(function () {
        var insertPoint = this.hub.findParentRowOrColumn(this.container) ||
            this.container.layoutManager.root.contentItems[0];
        insertPoint.addChild(createAstView);
    }, this));

    this.container.layoutManager
        .createDragSource(this.irButton, createIrView)
        ._dragListener.on('dragStart', togglePannerAdder);

    this.irButton.click(_.bind(function () {
        var insertPoint = this.hub.findParentRowOrColumn(this.container) ||
            this.container.layoutManager.root.contentItems[0];
        insertPoint.addChild(createIrView);
    }, this));

    this.container.layoutManager
        .createDragSource(this.gccDumpButton, createGccDumpView)
        ._dragListener.on('dragStart', togglePannerAdder);

    this.gccDumpButton.click(_.bind(function () {
        var insertPoint = this.hub.findParentRowOrColumn(this.container) ||
            this.container.layoutManager.root.contentItems[0];
        insertPoint.addChild(createGccDumpView);
    }, this));

    this.container.layoutManager
        .createDragSource(this.cfgButton, createCfgView)
        ._dragListener.on('dragStart', togglePannerAdder);

    this.cfgButton.click(_.bind(function () {
        var insertPoint = this.hub.findParentRowOrColumn(this.container) ||
            this.container.layoutManager.root.contentItems[0];
        insertPoint.addChild(createCfgView);
    }, this));

    this.initToolButtons(togglePannerAdder);
};

Compiler.prototype.undefer = function () {
    this.deferCompiles = false;
    if (this.needsCompile) {
        this.compile();
    }
};

Compiler.prototype.updateAndCalcTopBarHeight = function () {
    // If we save vertical space by hiding stuff that's OK to hide
    // when thin, then hide that stuff.
    this.hideable.show();
    var topBarHeightMax = this.topBar.outerHeight(true);
    this.hideable.hide();
    var topBarHeightMin = this.topBar.outerHeight(true);
    var topBarHeight = topBarHeightMin;
    if (topBarHeightMin === topBarHeightMax) {
        this.hideable.show();
    }

    return topBarHeight;
};

Compiler.prototype.resize = function () {
    var topBarHeight = this.updateAndCalcTopBarHeight();
    var bottomBarHeight = this.bottomBar.outerHeight(true);
    this.outputEditor.layout({
        width: this.domRoot.width(),
        height: this.domRoot.height() - topBarHeight - bottomBarHeight
    });
};

// Returns a label name if it can be found in the given position, otherwise
// returns null.
Compiler.prototype.getLabelAtPosition = function (position) {
    var asmLine = this.assembly[position.lineNumber - 1];
    var column = position.column;
    var labels = asmLine.labels;

    for (var i = 0; i < labels.length; ++i) {
        if (column >= labels[i].range.startCol &&
            column < labels[i].range.endCol
        ) {
            return labels[i];
        }
    }

    return null;
};

// Jumps to a label definition related to a label which was found in the
// given position and highlights the given range. If no label can be found in
// the given positon it do nothing.
Compiler.prototype.jumpToLabel = function (position) {
    var label = this.getLabelAtPosition(position);

    if (!label) {
        return;
    }

    var labelDefLineNum = this.labelDefinitions[label.name];
    if (!labelDefLineNum) {
        return;
    }

    // Highlight the new range.
    var endLineContent =
        this.outputEditor.getModel().getLineContent(labelDefLineNum);

    this.outputEditor.setSelection(new monaco.Selection(
        labelDefLineNum, 0,
        labelDefLineNum, endLineContent.length + 1));

    // Jump to the given line.
    this.outputEditor.revealLineInCenter(labelDefLineNum);
};

Compiler.prototype.initEditorActions = function () {
    this.isLabelCtxKey = this.outputEditor.createContextKey('isLabel', true);

    this.outputEditor.addAction({
        id: 'jumptolabel',
        label: 'Jump to label',
        keybindings: [monaco.KeyMod.CtrlCmd | monaco.KeyCode.Enter],
        precondition: 'isLabel',
        contextMenuGroupId: 'navigation',
        contextMenuOrder: 1.5,
        run: _.bind(function (ed) {
            this.jumpToLabel(ed.getPosition());
        }, this)
    });

    // Hiding the 'Jump to label' context menu option if no label can be found
    // in the clicked position.
    var contextmenu = this.outputEditor.getContribution('editor.contrib.contextmenu');
    var realMethod = contextmenu._onContextMenu;
    contextmenu._onContextMenu = _.bind(function (e) {
        if (this.isLabelCtxKey && e.target.position) {
            var label = this.getLabelAtPosition(e.target.position);
            this.isLabelCtxKey.set(label);
        }
        realMethod.apply(contextmenu, arguments);
    }, this);

    this.outputEditor.addAction({
        id: 'viewsource',
        label: 'Scroll to source',
        keybindings: [monaco.KeyMod.CtrlCmd | monaco.KeyCode.F10],
        keybindingContext: null,
        contextMenuGroupId: 'navigation',
        contextMenuOrder: 1.5,
        run: _.bind(function (ed) {
            var desiredLine = ed.getPosition().lineNumber - 1;
            var source = this.assembly[desiredLine].source;
            if (source !== null && source.file === null) {
                // a null file means it was the user's source
                this.eventHub.emit('editorLinkLine', this.sourceEditorId, source.line, -1, true);
            }
        }, this)
    });

    this.outputEditor.addAction({
        id: 'viewasmdoc',
        label: 'View x86-64 opcode doc',
        keybindings: [monaco.KeyMod.CtrlCmd | monaco.KeyCode.F8],
        keybindingContext: null,
        contextMenuGroupId: 'help',
        contextMenuOrder: 1.5,
        run: _.bind(this.onAsmToolTip, this)
    });

    this.outputEditor.addAction({
        id: 'toggleColourisation',
        label: 'Toggle colourisation',
        keybindings: [monaco.KeyMod.CtrlCmd | monaco.KeyMod.Shift | monaco.KeyCode.F1],
        keybindingContext: null,
        run: _.bind(function () {
            this.eventHub.emit('modifySettings', {
                colouriseAsm: !this.settings.colouriseAsm
            });
        }, this)
    });

};

// Gets the filters that will actually be used (accounting for issues with binary
// mode etc).
Compiler.prototype.getEffectiveFilters = function () {
    if (!this.compiler) return {};
    var filters = this.filters.get();
    if (filters.binary && !this.compiler.supportsBinary) {
        delete filters.binary;
    }
    if (filters.execute && !this.compiler.supportsExecute) {
        delete filters.execute;
    }
    if (filters.libraryCode && !this.compiler.supportsLibraryCodeFilter) {
        delete filters.libraryCode;
    }
    _.each(this.compiler.disabledFilters, function (filter) {
        if (filters[filter]) {
            delete filters[filter];
        }
    });
    return filters;
};

Compiler.prototype.findTools = function (content, tools) {
    if (content.componentName === "tool") {
        if (
            (content.componentState.editor === this.sourceEditorId) &&
            (content.componentState.compiler === this.id)) {
            tools.push({
                id: content.componentState.toolId,
                args: content.componentState.args,
                stdin: content.componentState.stdin
            });
        }
    } else if (content.content) {
        _.each(content.content, function (subcontent) {
            tools = this.findTools(subcontent, tools);
        }, this);
    }

    return tools;
};

Compiler.prototype.getActiveTools = function (newToolSettings) {
    if (!this.compiler) return {};

    var tools = [];
    if (newToolSettings) {
        tools.push({
            id: newToolSettings.toolId,
            args: newToolSettings.args,
            stdin: newToolSettings.stdin
        });
    }

    if (this.container.layoutManager.isInitialised) {
        var config = this.container.layoutManager.toConfig();
        return this.findTools(config, tools);
    } else {
        return tools;
    }
};

Compiler.prototype.isToolActive = function (activetools, toolId) {
    return _.find(activetools, function (tool) {
        return tool.id === toolId;
    });
};

Compiler.prototype.compile = function (bypassCache, newTools) {
    if (this.deferCompiles) {
        this.needsCompile = true;
        return;
    }
    this.needsCompile = false;
    this.compileTimeLabel.text(' - Compiling...');
    var options = {
        userArguments: this.options,
        compilerOptions: {
            produceAst: this.astViewOpen,
            produceGccDump: {
                opened: this.gccDumpViewOpen,
                pass: this.gccDumpPassSelected,
                treeDump: this.treeDumpEnabled,
                rtlDump: this.rtlDumpEnabled
            },
            produceOptInfo: this.wantOptInfo,
            produceCfg: this.cfgViewOpen,
            produceIr: this.irViewOpen
        },
        filters: this.getEffectiveFilters(),
        tools: this.getActiveTools(newTools),
        libraries: []
    };

    _.each(this.libsWidget.getLibsInUse(), function (item) {
        options.libraries.push({
            id: item.libId,
            version: item.versionId
        });
    });

    this.compilerService.expand(this.source).then(_.bind(function (expanded) {
        var request = {
            source: expanded || '',
            compiler: this.compiler ? this.compiler.id : '',
            options: options,
            lang: this.currentLangId
        };
        if (bypassCache) request.bypassCache = true;
        if (!this.compiler) {
            this.onCompileResponse(request, errorResult('<Please select a compiler>'), false);
        } else {
            this.sendCompile(request);
        }
    }, this));
};

Compiler.prototype.sendCompile = function (request) {
    var onCompilerResponse = _.bind(this.onCompileResponse, this);

    if (this.pendingRequestSentAt) {
        // If we have a request pending, then just store this request to do once the
        // previous request completes.
        this.nextRequest = request;
        return;
    }
    this.eventHub.emit('compiling', this.id, this.compiler);
    // Display the spinner
    this.handleCompilationStatus({code: 4});
    this.pendingRequestSentAt = Date.now();
    // After a short delay, give the user some indication that we're working on their
    // compilation.
    var progress = setTimeout(_.bind(function () {
        this.setAssembly(fakeAsm('<Compiling...>'));
    }, this), 500);
    this.compilerService.submit(request)
        .then(function (x) {
            clearTimeout(progress);
            onCompilerResponse(request, x.result, x.localCacheHit);
        })
        .catch(function (x) {
            clearTimeout(progress);
            var message = "Unknown error";
            if (_.isString(x)) {
                message = x;
            } else if (x) {
                message = x.error || x.code;
            }
            onCompilerResponse(request,
                errorResult('<Compilation failed: ' + message + '>'), false);
        });
};

Compiler.prototype.setNormalMargin = function () {
    this.outputEditor.updateOptions({
        lineNumbers: true,
        lineNumbersMinChars: 1
    });
};

Compiler.prototype.setBinaryMargin = function () {
    this.outputEditor.updateOptions({
        lineNumbersMinChars: 6,
        lineNumbers: _.bind(this.getBinaryForLine, this)
    });
};

Compiler.prototype.getBinaryForLine = function (line) {
    var obj = this.assembly[line - 1];
    if (obj) {
        return obj.address ? obj.address.toString(16) : '';
    } else {
        return '???';
    }
};

Compiler.prototype.setAssembly = function (asm) {
    this.assembly = asm;
    if (!this.outputEditor || !this.outputEditor.getModel()) return;
    var editorModel = this.outputEditor.getModel();
    editorModel.setValue(asm.length ? _.pluck(asm, 'text').join('\n') : "<No assembly generated>");

    if (!this.awaitingInitialResults) {
        if (this.selection) {
            this.outputEditor.setSelection(this.selection);
            this.outputEditor.revealLinesInCenter(
                this.selection.startLineNumber, this.selection.endLineNumber);
        }
        this.awaitingInitialResults = true;
    } else {
        var visibleRanges = this.outputEditor.getVisibleRanges();
        var currentTopLine =
            visibleRanges.length > 0 ? visibleRanges[0].startLineNumber : 1;
        this.outputEditor.revealLine(currentTopLine);
    }

    this.decorations.labelUsages = [];
    _.each(this.assembly, _.bind(function (obj, line) {
        if (!obj.labels || !obj.labels.length) return;

        obj.labels.forEach(function (label) {
            this.decorations.labelUsages.push({
                range: new monaco.Range(line + 1, label.range.startCol,
                    line + 1, label.range.endCol),
                options: {
                    inlineClassName: 'asm-label-link'
                }
            });
        }, this);
    }, this));
    this.updateDecorations();

    if (this.getEffectiveFilters().binary) {
        this.setBinaryMargin();
        if (this.codeLensProvider !== null) {
            this.codeLensProvider.dispose();
        }
        var codeLenses = [];
        _.each(this.assembly, _.bind(function (obj, line) {
            if (obj.opcodes) {
                var address = obj.address ? obj.address.toString(16) : '';
                codeLenses.push({
                    range: {
                        startLineNumber: line + 1,
                        startColumn: 1,
                        endLineNumber: line + 2,
                        endColumn: 1
                    },
                    id: address,
                    command: {
                        title: obj.opcodes.join(' ')
                    }
                });
            }
        }, this));
        var currentAsmLang = editorModel.getModeId();
        this.codeLensProvider = monaco.languages.registerCodeLensProvider(currentAsmLang, {
            provideCodeLenses: function (model) {
                if (model === editorModel)
                    return codeLenses;
                else
                    return [];
            }
        });
    } else {
        this.setNormalMargin();
        if (this.codeLensProvider !== null) {
            this.codeLensProvider.dispose();
        }
    }
};

function errorResult(text) {
    return {asm: fakeAsm(text), code: -1, stdout: '', stderr: ''};
}

function fakeAsm(text) {
    return [{text: text, source: null, fake: true}];
}

Compiler.prototype.onCompileResponse = function (request, result, cached) {
    // Delete trailing empty lines
    if ($.isArray(result.asm)) {
        var indexToDiscard = _.findLastIndex(result.asm, function (line) {
            return !_.isEmpty(line.text);
        });
        result.asm.splice(indexToDiscard + 1, result.asm.length - indexToDiscard);
    }
    // Save which source produced this change. It should probably be saved earlier though
    result.source = this.source;
    this.lastResult = result;
    var timeTaken = Math.max(0, Date.now() - this.pendingRequestSentAt);
    var wasRealReply = this.pendingRequestSentAt > 0;
    this.pendingRequestSentAt = 0;
    ga.proxy('send', {
        hitType: 'event',
        eventCategory: 'Compile',
        eventAction: request.compiler,
        eventLabel: request.options.userArguments,
        eventValue: cached ? 1 : 0
    });
    ga.proxy('send', {
        hitType: 'timing',
        timingCategory: 'Compile',
        timingVar: request.compiler,
        timingValue: timeTaken
    });

    this.labelDefinitions = result.labelDefinitions || {};
    this.setAssembly(result.asm || fakeAsm('<No output>'));

    var stdout = result.stdout || [];
    var stderr = result.stderr || [];

    var allText = _.pluck(stdout.concat(stderr), 'text').join('\n');
    var failed = result.code !== 0;
    var warns = !failed && !!allText;
    this.handleCompilationStatus({code: failed ? 3 : (warns ? 2 : 1), compilerOut: result.code});
    this.outputTextCount.text(stdout.length);
    this.outputErrorCount.text(stderr.length);
    if (this.isOutputOpened) {
        this.outputBtn.prop('title', '');
    } else {
        this.outputBtn.prop('title', allText.replace(/\x1b\[[0-9;]*m(.\[K)?/g, ''));
    }
    var timeLabelText = '';
    if (cached) {
        timeLabelText = ' - cached';
    } else if (wasRealReply) {
        timeLabelText = ' - ' + timeTaken + 'ms';
    }

    if (result.asmSize !== undefined) {
        timeLabelText += ' (' + result.asmSize + 'B)';
    }

    this.compileTimeLabel.text(timeLabelText);

    this.postCompilationResult(request, result);
    this.eventHub.emit('compileResult', this.id, this.compiler, result, languages[this.currentLangId]);

    if (this.nextRequest) {
        var next = this.nextRequest;
        this.nextRequest = null;
        this.sendCompile(next);
    }
};

Compiler.prototype.postCompilationResult = function (request, result) {
    if (result.popularArguments) {
        this.handlePopularArgumentsResult(result.popularArguments);
    } else {
        this.compilerService.requestPopularArguments(this.compiler.id, request.options.userArguments).then(
            _.bind(function (result) {
                if (result && result.result) {
                    this.handlePopularArgumentsResult(result.result);
                }
            }, this)
        );
    }

    this.updateButtons();

    this.checkForUnwiseArguments(result.compilationOptions);
    this.setCompilationOptionsPopover(result.compilationOptions ? result.compilationOptions.join(' ') : '');
};

Compiler.prototype.onEditorChange = function (editor, source, langId, compilerId) {
    if (editor === this.sourceEditorId && langId === this.currentLangId &&
        (compilerId === undefined || compilerId === this.id)) {
        this.source = source;
        if (this.settings.compileOnChange) {
            this.compile();
        }
    }
};

Compiler.prototype.onToolOpened = function (compilerId, toolSettings) {
    if (this.id === compilerId) {
        var toolId = toolSettings.toolId;

        var buttons = this.toolsMenu.find('button');
        $(buttons).each(_.bind(function (idx, button) {
            var toolButton = $(button);
            var toolName = toolButton.data('toolname');
            if (toolId === toolName) {
                toolButton.prop('disabled', true);
            }
        }, this));

        this.compile(false, toolSettings);
    }
};

Compiler.prototype.onToolClosed = function (compilerId, toolSettings) {
    if (this.id === compilerId) {
        var toolId = toolSettings.toolId;

        var buttons = this.toolsMenu.find('button');
        $(buttons).each(_.bind(function (idx, button) {
            var toolButton = $(button);
            var toolName = toolButton.data('toolname');
            if (toolId === toolName) {
                toolButton.prop('disabled', !this.supportsTool(toolId));
            }
        }, this));
    }
};

Compiler.prototype.onOutputOpened = function (compilerId) {
    if (this.id === compilerId) {
        this.isOutputOpened = true;
        this.outputBtn.prop('disabled', true);
        this.resendResult();
    }
};

Compiler.prototype.onOutputClosed = function (compilerId) {
    if (this.id === compilerId) {
        this.isOutputOpened = false;
        this.outputBtn.prop('disabled', false);
    }
};

Compiler.prototype.onOptViewClosed = function (id) {
    if (this.id === id) {
        this.wantOptInfo = false;
        this.optViewOpen = false;
        this.optButton.prop('disabled', this.optViewOpen);
    }
};

Compiler.prototype.onAstViewOpened = function (id) {
    if (this.id === id) {
        this.astButton.prop('disabled', true);
        this.astViewOpen = true;
        this.compile();
    }
};

Compiler.prototype.onToolSettingsChange = function (id) {
    if (this.id === id) {
        this.compile();
    }
};

Compiler.prototype.onAstViewClosed = function (id) {
    if (this.id === id) {
        this.astButton.prop('disabled', false);
        this.astViewOpen = false;
    }
};

Compiler.prototype.onIrViewOpened = function (id) {
    if (this.id === id) {
        this.irButton.prop('disabled', true);
        this.irViewOpen = true;
        this.compile();
    }
};

Compiler.prototype.onIrViewClosed = function (id) {
    if (this.id === id) {
        this.irButton.prop('disabled', false);
        this.irViewOpen = false;
    }
};

Compiler.prototype.onGccDumpUIInit = function (id) {
    if (this.id === id) {
        this.compile();
    }
};

Compiler.prototype.onGccDumpFiltersChanged = function (id, filters, reqCompile) {
    if (this.id === id) {
        this.treeDumpEnabled = (filters.treeDump !== false);
        this.rtlDumpEnabled = (filters.rtlDump !== false);

        if (reqCompile) {
            this.compile();
        }
    }
};

Compiler.prototype.onGccDumpPassSelected = function (id, passId, reqCompile) {
    if (this.id === id) {
        this.gccDumpPassSelected = passId;

        if (reqCompile && passId !== '') {
            this.compile();
        }
    }
};

Compiler.prototype.onGccDumpViewOpened = function (id) {
    if (this.id === id) {
        this.gccDumpButton.prop('disabled', true);
        this.gccDumpViewOpen = true;
    }
};

Compiler.prototype.onGccDumpViewClosed = function (id) {
    if (this.id === id) {
        this.gccDumpButton.prop('disabled', !this.compiler.supportsGccDump);
        this.gccDumpViewOpen = false;

        delete this.gccDumpPassSelected;
        delete this.treeDumpEnabled;
        delete this.rtlDumpEnabled;
    }
};

Compiler.prototype.onOptViewOpened = function (id) {
    if (this.id === id) {
        this.optViewOpen = true;
        this.wantOptInfo = true;
        this.optButton.prop('disabled', this.optViewOpen);
        this.compile();
    }
};

Compiler.prototype.onCfgViewOpened = function (id) {
    if (this.id === id) {
        this.cfgButton.prop('disabled', true);
        this.cfgViewOpen = true;
        this.compile();
    }
};

Compiler.prototype.onCfgViewClosed = function (id) {
    if (this.id === id) {
        this.cfgViewOpen = false;
        this.cfgButton.prop('disabled', this.getEffectiveFilters().binary);
    }
};

Compiler.prototype.initButtons = function (state) {
    this.filters = new Toggles(this.domRoot.find('.filters'), patchOldFilters(state.filters));

    this.optButton = this.domRoot.find('.btn.view-optimization');
    this.astButton = this.domRoot.find('.btn.view-ast');
    this.irButton = this.domRoot.find('.btn.view-ir');
    this.gccDumpButton = this.domRoot.find('.btn.view-gccdump');
    this.cfgButton = this.domRoot.find('.btn.view-cfg');
    this.libsButton = this.domRoot.find('.btn.show-libs');

    this.compileTimeLabel = this.domRoot.find('.compile-time');
    this.compileClearCache = this.domRoot.find('.clear-cache');

    this.outputBtn = this.domRoot.find('.output-btn');
    this.outputTextCount = this.domRoot.find('span.text-count');
    this.outputErrorCount = this.domRoot.find('span.err-count');

    this.optionsField = this.domRoot.find('.options');
    this.prependOptions = this.domRoot.find('.prepend-options');
    this.fullCompilerName = this.domRoot.find('.full-compiler-name');
    this.setCompilationOptionsPopover(this.compiler ? this.compiler.options : null);
    // Dismiss on any click that isn't either in the opening element, inside
    // the popover or on any alert
    $(document).on('mouseup', _.bind(function (e) {
        var target = $(e.target);
        if (!target.is(this.prependOptions) && this.prependOptions.has(target).length === 0 &&
            target.closest('.popover').length === 0)
            this.prependOptions.popover("hide");

        if (!target.is(this.fullCompilerName) && this.fullCompilerName.has(target).length === 0 &&
            target.closest('.popover').length === 0)
            this.fullCompilerName.popover("hide");
    }, this));

    this.filterBinaryButton = this.domRoot.find("[data-bind='binary']");
    this.filterBinaryTitle = this.filterBinaryButton.prop('title');

    this.filterExecuteButton = this.domRoot.find("[data-bind='execute']");
    this.filterExecuteTitle = this.filterExecuteButton.prop('title');

    this.filterLabelsButton = this.domRoot.find("[data-bind='labels']");
    this.filterLabelsTitle = this.filterLabelsButton.prop('title');

    this.filterDirectivesButton = this.domRoot.find("[data-bind='directives']");
    this.filterDirectivesTitle = this.filterDirectivesButton.prop('title');

    this.filterLibraryCodeButton = this.domRoot.find("[data-bind='libraryCode']");
    this.filterLibraryCodeTitle = this.filterLibraryCodeButton.prop('title');

    this.filterCommentsButton = this.domRoot.find("[data-bind='commentOnly']");
    this.filterCommentsTitle = this.filterCommentsButton.prop('title');

    this.filterTrimButton = this.domRoot.find("[data-bind='trim']");
    this.filterTrimTitle = this.filterTrimButton.prop('title');

    this.filterIntelButton = this.domRoot.find("[data-bind='intel']");
    this.filterIntelTitle = this.filterIntelButton.prop('title');

    this.filterDemangleButton = this.domRoot.find("[data-bind='demangle']");
    this.filterDemangleTitle = this.filterDemangleButton.prop('title');

    this.noBinaryFiltersButtons = this.domRoot.find('.nonbinary');
    this.filterExecuteButton.toggle(options.supportsExecute);
    this.filterLibraryCodeButton.toggle(options.supportsLibraryCodeFilter);
    this.optionsField.val(this.options);

    this.shortCompilerName = this.domRoot.find('.short-compiler-name');
    this.compilerPicker = this.domRoot.find('.compiler-picker');
    this.setCompilerVersionPopover('');

    this.topBar = this.domRoot.find('.top-bar');
    this.bottomBar = this.domRoot.find('.bottom-bar');
    this.statusLabel = this.domRoot.find('.status-text');

    this.hideable = this.domRoot.find('.hideable');
    this.statusIcon = this.domRoot.find('.status-icon');

    this.monacoPlaceholder = this.domRoot.find('.monaco-placeholder');

    this.initPanerButtons();
};

Compiler.prototype.onLibsChanged = function () {
    this.saveState();
    this.compile();
};

Compiler.prototype.initLibraries = function (state) {
    this.libsWidget = new Libraries.Widget(this.currentLangId, this.compiler, this.libsButton,
        state, _.bind(this.onLibsChanged, this));
};

Compiler.prototype.updateLibraries = function () {
    if (this.libsWidget) this.libsWidget.setNewLangId(this.currentLangId, this.compiler.id, this.compiler.libs);
};

Compiler.prototype.supportsTool = function (toolId) {
    if (!this.compiler) return;

    return _.find(this.compiler.tools, function (tool) {
        return (tool.tool.id === toolId);
    });
};

Compiler.prototype.initToolButton = function (togglePannerAdder, button, toolId) {
    var createToolView = _.bind(function () {
        return Components.getToolViewWith(this.id, this.sourceEditorId, toolId, "");
    }, this);

    this.container.layoutManager
        .createDragSource(button, createToolView)
        ._dragListener.on('dragStart', togglePannerAdder);

    button.click(_.bind(function () {
        button.prop("disabled", true);
        var insertPoint = this.hub.findParentRowOrColumn(this.container) ||
            this.container.layoutManager.root.contentItems[0];
        insertPoint.addChild(createToolView);
    }, this));
};

Compiler.prototype.initToolButtons = function (togglePannerAdder) {
    this.toolsMenu = this.domRoot.find('.toolsmenu');
    this.toolsMenu.empty();

    if (!this.compiler) return;

    var addTool = _.bind(function (toolName, title) {
        var btn = $("<button class='dropdown-item btn btn-light btn-sm'>");
        btn.addClass('.view-' + toolName);
        btn.data('toolname', toolName);
        btn.append("<span class='dropdown-icon fas fa-cog' />" + title);
        this.toolsMenu.append(btn);

        if (toolName !== "none") {
            this.initToolButton(togglePannerAdder, btn, toolName);
        }
    }, this);

    if (_.isEmpty(this.compiler.tools)) {
        addTool("none", "No tools available");
    } else {
        _.each(this.compiler.tools, function (tool) {
            addTool(tool.tool.id, tool.tool.name);
        });
    }
};

Compiler.prototype.enableToolButtons = function () {
    var activeTools = this.getActiveTools();

    var buttons = this.toolsMenu.find('button');
    $(buttons).each(_.bind(function (idx, button) {
        var toolButton = $(button);
        var toolName = toolButton.data('toolname');
        toolButton.prop('disabled',
            !(this.supportsTool(toolName)
            && !this.isToolActive(activeTools, toolName)));
    }, this));
};

Compiler.prototype.updateButtons = function () {
    if (!this.compiler) return;
    var filters = this.getEffectiveFilters();
    // We can support intel output if the compiler supports it, or if we're compiling
    // to binary (as we can disassemble it however we like).
    var formatFilterTitle = function (button, title) {
        button.prop('title', '[' + (button.hasClass('active') ? 'ON' : 'OFF') + '] ' + title +
            (button.prop('disabled') ? ' [LOCKED]' : ''));
    };
    var isIntelFilterDisabled = !this.compiler.supportsIntel && !filters.binary;
    this.filterIntelButton.prop('disabled', isIntelFilterDisabled);
    formatFilterTitle(this.filterIntelButton, this.filterIntelTitle);
    // Disable binary support on compilers that don't work with it.
    this.filterBinaryButton.prop('disabled', !this.compiler.supportsBinary);
    formatFilterTitle(this.filterBinaryButton, this.filterBinaryTitle);
    this.filterExecuteButton.prop('disabled', !this.compiler.supportsExecute);
    formatFilterTitle(this.filterExecuteButton, this.filterExecuteTitle);
    // Disable demangle for compilers where we can't access it
    this.filterDemangleButton.prop('disabled', !this.compiler.supportsDemangle);
    formatFilterTitle(this.filterDemangleButton, this.filterDemangleTitle);
    // Disable any of the options which don't make sense in binary mode.
    var noBinaryFiltersDisabled = !!filters.binary && !this.compiler.supportsFiltersInBinary;
    this.noBinaryFiltersButtons.prop('disabled', noBinaryFiltersDisabled);

    this.filterLibraryCodeButton.prop('disabled', !this.compiler.supportsLibraryCodeFilter);
    formatFilterTitle(this.filterLibraryCodeButton, this.filterLibraryCodeTitle);

    this.filterLabelsButton.prop('disabled', this.compiler.disabledFilters.indexOf('labels') !== -1);
    formatFilterTitle(this.filterLabelsButton, this.filterLabelsTitle);
    this.filterDirectivesButton.prop('disabled', this.compiler.disabledFilters.indexOf('directives') !== -1);
    formatFilterTitle(this.filterDirectivesButton, this.filterDirectivesTitle);
    this.filterCommentsButton.prop('disabled', this.compiler.disabledFilters.indexOf('commentOnly') !== -1);
    formatFilterTitle(this.filterCommentsButton, this.filterCommentsTitle);
    this.filterTrimButton.prop('disabled', this.compiler.disabledFilters.indexOf('trim') !== -1);
    formatFilterTitle(this.filterTrimButton, this.filterTrimTitle);

    this.optButton.prop('disabled', this.optViewOpen || !this.compiler.supportsOptOutput);
    this.astButton.prop('disabled', this.astViewOpen || !this.compiler.supportsAstView);
    this.irButton.prop('disabled', this.irViewOpen || !this.compiler.supportsIrView);
    this.cfgButton.prop('disabled', this.cfgViewOpen || !this.compiler.supportsCfg);
    this.gccDumpButton.prop('disabled', this.gccDumpViewOpen || !this.compiler.supportsGccDump);

    this.enableToolButtons();
};

Compiler.prototype.handlePopularArgumentsResult = function (result) {
    var popularArgumentsMenu = this.domRoot.find("div.populararguments div.dropdown-menu");
    popularArgumentsMenu.html("");

    if (result) {
        var addedOption = false;

        _.forEach(result, _.bind(function (arg, key) {
            var argumentButton = $(document.createElement("button"));
            argumentButton.addClass('dropdown-item btn btn-light btn-sm');
            argumentButton.attr("title", arg.description);
            argumentButton.data("arg", key);
            argumentButton.html(
                "<div class='argmenuitem'>" +
                "<span class='argtitle'>" + _.escape(key) + "</span>" +
                "<span class='argdescription'>" + arg.description + "</span>" +
                "</div>");

            argumentButton.click(_.bind(function () {
                var button = argumentButton;
                var curOptions = this.optionsField.val();
                if (curOptions.length > 0) {
                    this.optionsField.val(curOptions + " " + button.data("arg"));
                } else {
                    this.optionsField.val(button.data("arg"));
                }

                this.optionsField.change();
            }, this));

            popularArgumentsMenu.append(argumentButton);
            addedOption = true;
        }, this));

        if (!addedOption) {
            $("div.populararguments").hide();
        } else {
            $("div.populararguments").show();
        }
    } else {
        $("div.populararguments").hide();
    }
};

Compiler.prototype.onFontScale = function () {
    this.saveState();
};

Compiler.prototype.initListeners = function () {
    this.filters.on('change', _.bind(this.onFilterChange, this));
    this.fontScale.on('change', _.bind(this.onFontScale, this));

    this.container.on('destroy', this.close, this);
    this.container.on('resize', this.resize, this);
    this.container.on('shown', this.resize, this);
    this.container.on('open', function () {
        this.eventHub.emit('compilerOpen', this.id, this.sourceEditorId);
    }, this);
    this.eventHub.on('editorChange', this.onEditorChange, this);
    this.eventHub.on('editorClose', this.onEditorClose, this);
    this.eventHub.on('colours', this.onColours, this);
    this.eventHub.on('resendCompilation', this.onResendCompilation, this);
    this.eventHub.on('findCompilers', this.sendCompiler, this);
    this.eventHub.on('compilerSetDecorations', this.onCompilerSetDecorations, this);
    this.eventHub.on('panesLinkLine', this.onPanesLinkLine, this);
    this.eventHub.on('settingsChange', this.onSettingsChange, this);
    this.eventHub.on('requestCompilation', this.onRequestCompilation, this);

    this.eventHub.on('toolSettingsChange', this.onToolSettingsChange, this);
    this.eventHub.on('toolOpened', this.onToolOpened, this);
    this.eventHub.on('toolClosed', this.onToolClosed, this);

    this.eventHub.on('optViewOpened', this.onOptViewOpened, this);
    this.eventHub.on('optViewClosed', this.onOptViewClosed, this);
    this.eventHub.on('astViewOpened', this.onAstViewOpened, this);
    this.eventHub.on('astViewClosed', this.onAstViewClosed, this);
    this.eventHub.on('irViewOpened', this.onIrViewOpened, this);
    this.eventHub.on('irViewClosed', this.onIrViewClosed, this);
    this.eventHub.on('outputOpened', this.onOutputOpened, this);
    this.eventHub.on('outputClosed', this.onOutputClosed, this);

    this.eventHub.on('gccDumpPassSelected', this.onGccDumpPassSelected, this);
    this.eventHub.on('gccDumpFiltersChanged', this.onGccDumpFiltersChanged, this);
    this.eventHub.on('gccDumpViewOpened', this.onGccDumpViewOpened, this);
    this.eventHub.on('gccDumpViewClosed', this.onGccDumpViewClosed, this);
    this.eventHub.on('gccDumpUIInit', this.onGccDumpUIInit, this);

    this.eventHub.on('cfgViewOpened', this.onCfgViewOpened, this);
    this.eventHub.on('cfgViewClosed', this.onCfgViewClosed, this);
    this.eventHub.on('resize', this.resize, this);
    this.eventHub.on('requestFilters', function (id) {
        if (id === this.id) {
            this.eventHub.emit('filtersChange', this.id, this.getEffectiveFilters());
        }
    }, this);
    this.eventHub.on('requestCompiler', function (id) {
        if (id === this.id) {
            this.sendCompiler();
        }
    }, this);
    this.eventHub.on('languageChange', this.onLanguageChange, this);
};

Compiler.prototype.initCallbacks = function () {
    this.initListeners();

    var optionsChange = _.debounce(_.bind(function (e) {
        this.onOptionsChange($(e.target).val());
    }, this), 800);

    this.optionsField
        .on('change', optionsChange)
        .on('keyup', optionsChange);

    this.mouseMoveThrottledFunction = _.throttle(_.bind(this.onMouseMove, this), 50);
    this.outputEditor.onMouseMove(_.bind(function (e) {
        this.mouseMoveThrottledFunction(e);
    }, this));

    this.cursorSelectionThrottledFunction =
        _.throttle(_.bind(this.onDidChangeCursorSelection, this), 500);
    this.outputEditor.onDidChangeCursorSelection(_.bind(function (e) {
        this.cursorSelectionThrottledFunction(e);
    }, this));

    this.mouseUpThrottledFunction = _.throttle(_.bind(this.onMouseUp, this), 50);
    this.outputEditor.onMouseUp(_.bind(function (e) {
        this.mouseUpThrottledFunction(e);
    }, this));

    this.compileClearCache.on('click', _.bind(function () {
        this.compilerService.cache.reset();
        this.compile(true);
    }, this));

    // Dismiss the popover on escape.
    $(document).on('keyup.editable', _.bind(function (e) {
        if (e.which === 27) {
            this.libsButton.popover('hide');
        }
    }, this));

    // Dismiss on any click that isn't either in the opening element, inside
    // the popover or on any alert
    $(document).on('click', _.bind(function (e) {
        var elem = this.libsButton;
        var target = $(e.target);
        if (!target.is(elem) && elem.has(target).length === 0 && target.closest('.popover').length === 0) {
            elem.popover('hide');
        }
    }, this));

    this.eventHub.on('initialised', this.undefer, this);
};

Compiler.prototype.onOptionsChange = function (options) {
    if (this.options !== options) {
        this.options = options;
        this.saveState();
        this.compile();
        this.updateButtons();
        this.sendCompiler();
    }
};

Compiler.prototype.checkForUnwiseArguments = function (optionsArray) {
    // Check if any options are in the unwiseOptions array and remember them
    var unwiseOptions = _.intersection(optionsArray, this.compiler.unwiseOptions);

    var options = unwiseOptions.length === 1 ? "Option " : "Options ";
    var names = unwiseOptions.join(", ");
    var are = unwiseOptions.length === 1 ? " is " : " are ";
    var msg = options + names + are + "not recommended, as behaviour might change based on server hardware.";

    if (unwiseOptions.length > 0) {
        this.alertSystem.notify(msg, {group: 'unwiseOption', collapseSimilar: true});
    }
};

Compiler.prototype.updateCompilerInfo = function () {
    this.updateCompilerName();
    if (this.compiler) {
        if (this.compiler.notification) {
            this.alertSystem.notify(this.compiler.notification, {
                group: 'compilerwarning',
                alertClass: 'notification-info',
                dismissTime: 5000
            });
        }
        this.prependOptions.data('content', this.compiler.options);
    }
};

Compiler.prototype.updateCompilerUI = function () {
    var panerDropdown = this.domRoot.find('.pane-dropdown');
    var togglePannerAdder = function () {
        panerDropdown.dropdown('toggle');
    };
    this.initToolButtons(togglePannerAdder);
    this.updateButtons();
    this.updateCompilerInfo();
    // Resize in case the new compiler name is too big
    this.resize();
};

Compiler.prototype.onCompilerChange = function (value) {
    this.compiler = this.compilerService.findCompiler(this.currentLangId, value);
    this.updateLibraries();
    this.saveState();
    this.compile();
    this.updateCompilerUI();
    this.sendCompiler();
};

Compiler.prototype.sendCompiler = function () {
    this.eventHub.emit('compiler', this.id, this.compiler, this.options, this.sourceEditorId);
};

Compiler.prototype.onEditorClose = function (editor) {
    if (editor === this.sourceEditorId) {
        // We can't immediately close as an outer loop somewhere in GoldenLayout is iterating over
        // the hierarchy. We can't modify while it's being iterated over.
        this.close();
        _.defer(function (self) {
            self.container.close();
        }, this);
    }
};

Compiler.prototype.onFilterChange = function () {
    var filters = this.getEffectiveFilters();
    this.eventHub.emit('filtersChange', this.id, filters);
    this.saveState();
    this.compile();
    this.updateButtons();
};

Compiler.prototype.currentState = function () {
    var state = {
        id: this.id,
        compiler: this.compiler ? this.compiler.id : '',
        source: this.sourceEditorId,
        options: this.options,
        // NB must *not* be effective filters
        filters: this.filters.get(),
        wantOptInfo: this.wantOptInfo,
        libs: this.libsWidget.get(),
        lang: this.currentLangId,
        selection: this.selection
    };
    this.fontScale.addState(state);
    return state;
};

Compiler.prototype.saveState = function () {
    this.container.setState(this.currentState());
};

Compiler.prototype.onColours = function (editor, colours, scheme) {
    if (editor === this.sourceEditorId) {
        var asmColours = {};
        _.each(this.assembly, function (x, index) {
            if (x.source && x.source.file === null && x.source.line > 0) {
                asmColours[index] = colours[x.source.line - 1];
            }
        });
        this.colours = colour.applyColours(this.outputEditor, asmColours, scheme, this.colours);
    }
};

Compiler.prototype.getCompilerName = function () {
    return this.compiler ? this.compiler.name : 'No compiler set';
};


Compiler.prototype.getLanguageName = function () {
    var lang = options.languages[this.currentLangId];
    return lang ? lang.name : '?';
};

Compiler.prototype.getPaneName = function () {
    var langName = this.getLanguageName();
    var compName = this.getCompilerName();
    return compName + ' (Editor #' + this.sourceEditorId + ', Compiler #' + this.id + ') ' + langName;
};

Compiler.prototype.updateCompilerName = function () {
    var compilerName = this.getCompilerName();
    var compilerVersion = this.compiler ? this.compiler.version : '';
    this.container.setTitle(this.getPaneName());
    this.shortCompilerName.text(compilerName);
    this.setCompilerVersionPopover(compilerVersion);
};

Compiler.prototype.resendResult = function () {
    if (!$.isEmptyObject(this.lastResult)) {
        this.eventHub.emit('compileResult', this.id, this.compiler, this.lastResult);
        return true;
    }
    return false;
};

Compiler.prototype.onResendCompilation = function (id) {
    if (id === this.id) {
        this.resendResult();
    }
};

Compiler.prototype.updateDecorations = function () {
    this.prevDecorations = this.outputEditor.deltaDecorations(
        this.prevDecorations, _.flatten(_.values(this.decorations)));
};

Compiler.prototype.clearLinkedLines = function () {
    this.decorations.linkedCode = [];
    this.updateDecorations();
};

Compiler.prototype.onPanesLinkLine = function (compilerId, lineNumber, revealLine, sender) {
    if (Number(compilerId) === this.id) {
        var lineNums = [];
        _.each(this.assembly, function (asmLine, i) {
            if (asmLine.source && asmLine.source.file === null && asmLine.source.line === lineNumber) {
                lineNums.push(i + 1);
            }
        });
        if (revealLine && lineNums[0]) this.outputEditor.revealLineInCenter(lineNums[0]);
        var lineClass = sender !== this.getPaneName() ? 'linked-code-decoration-line' : '';
        this.decorations.linkedCode = _.map(lineNums, function (line) {
            return {
                range: new monaco.Range(line, 1, line, 1),
                options: {
                    isWholeLine: true,
                    linesDecorationsClassName: 'linked-code-decoration-margin',
                    className: lineClass
                }
            };
        });
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

Compiler.prototype.onCompilerSetDecorations = function (id, lineNums, revealLine) {
    if (Number(id) === this.id) {
        if (revealLine && lineNums[0]) this.outputEditor.revealLineInCenter(lineNums[0]);
        this.decorations.linkedCode = _.map(lineNums, function (line) {
            return {
                range: new monaco.Range(line, 1, line, 1),
                options: {
                    isWholeLine: true,
                    linesDecorationsClassName: 'linked-code-decoration-margin',
                    inlineClassName: 'linked-code-decoration-inline'
                }
            };
        });
        this.updateDecorations();
    }
};

Compiler.prototype.setCompilationOptionsPopover = function (content) {
    this.prependOptions.popover('dispose');
    this.prependOptions.popover({
        content: content || 'No options in use',
        template: '<div class="popover' +
            (content ? ' compiler-options-popover' : '') +
            '" role="tooltip"><div class="arrow"></div>' +
            '<h3 class="popover-header"></h3><div class="popover-body"></div></div>'
    });
};

Compiler.prototype.setCompilerVersionPopover = function (version) {
    this.fullCompilerName.popover('dispose');
    this.fullCompilerName.popover({
        content: version || '',
        template: '<div class="popover' +
            (version ? ' compiler-options-popover' : '') +
            '" role="tooltip"><div class="arrow"></div>' +
            '<h3 class="popover-header"></h3><div class="popover-body"></div></div>'
    });
};

Compiler.prototype.onRequestCompilation = function (editorId) {
    if (editorId === this.sourceEditorId) {
        this.compile();
    }
};

Compiler.prototype.onSettingsChange = function (newSettings) {
    var before = this.settings;
    this.settings = _.clone(newSettings);
    if (!before.lastHoverShowSource && this.settings.hoverShowSource) {
        this.onCompilerSetDecorations(this.id, []);
    }
    this.outputEditor.updateOptions({
        contextmenu: this.settings.useCustomContextMenu,
        minimap: {
            enabled: this.settings.showMinimap && !options.embedded
        },
        fontFamily: this.settings.editorsFFont,
        fontLigatures: this.settings.editorsFLigatures
    });
};

var hexLike = /^(#?[$]|0x)([0-9a-fA-F]+)$/;
var hexLike2 = /^(#?)([0-9a-fA-F]+)H$/;
var decimalLike = /^(#?)(-?[0-9]+)$/;

function getNumericToolTip(value) {
    var match = hexLike.exec(value) || hexLike2.exec(value);
    if (match) {
        return value + ' = ' + bigInt(match[2], 16).toString(10);
    }
    match = decimalLike.exec(value);
    if (match) {
        var asBig = bigInt(match[2]);
        if (asBig.isNegative()) {
            asBig = bigInt('ffffffffffffffff', 16).and(asBig);
        }
        return value + ' = 0x' + asBig.toString(16).toUpperCase();
    }

    return null;
}

function getAsmInfo(opcode) {
    var cacheName = 'asm/' + opcode;
    var cached = OpcodeCache.get(cacheName);
    if (cached) {
        return Promise.resolve(cached.found ? cached.result : null);
    }
    var base = window.httpRoot;
    return new Promise(function (resolve, reject) {
        $.ajax({
            type: 'GET',
            url: window.location.origin + base + 'api/asm/' + opcode,
            dataType: 'json',  // Expected,
            contentType: 'text/plain',  // Sent
            success: function (result) {
                OpcodeCache.set(cacheName, result);
                resolve(result.found ? result.result : null);
            },
            error: function (result) {
                reject(result);
            },
            cache: true
        });
    });
}

Compiler.prototype.onDidChangeCursorSelection = function (e) {
    if (this.awaitingInitialResults) {
        this.selection = e.selection;
        this.saveState();
    }
};

Compiler.prototype.onMouseUp = function (e) {
    if (e === null || e.target === null || e.target.position === null) return;

    if (e.event.ctrlKey && e.event.leftButton) {
        this.jumpToLabel(e.target.position);
    }
};

Compiler.prototype.onMouseMove = function (e) {
    if (e === null || e.target === null || e.target.position === null) return;
    if (this.settings.hoverShowSource === true && this.assembly) {
        this.clearLinkedLines();
        var hoverAsm = this.assembly[e.target.position.lineNumber - 1];
        if (hoverAsm) {
            // We check that we actually have something to show at this point!
            var sourceLine = hoverAsm.source && !hoverAsm.source.file ? hoverAsm.source.line : -1;
            this.eventHub.emit('editorLinkLine', this.sourceEditorId, sourceLine, -1, false);
            this.eventHub.emit('panesLinkLine', this.id, sourceLine, false, this.getPaneName());
        }
    }
    var currentWord = this.outputEditor.getModel().getWordAtPosition(e.target.position);
    if (currentWord && currentWord.word) {
        var word = currentWord.word;
        var startColumn = currentWord.startColumn;
        // Avoid throwing an exception if somehow (How?) we have a non existent lineNumber.
        // c.f. https://sentry.io/matt-godbolt/compiler-explorer/issues/285270358/
        if (e.target.position.lineNumber <= this.outputEditor.getModel().getLineCount()) {
            // Hacky workaround to check for negative numbers.
            // c.f. https://github.com/mattgodbolt/compiler-explorer/issues/434
            var lineContent = this.outputEditor.getModel().getLineContent(e.target.position.lineNumber);
            if (lineContent[currentWord.startColumn - 2] === '-') {
                word = '-' + word;
                startColumn -= 1;
            }
        }
        currentWord.range = new monaco.Range(e.target.position.lineNumber, Math.max(startColumn, 1),
            e.target.position.lineNumber, currentWord.endColumn);
        var numericToolTip = getNumericToolTip(word);
        if (numericToolTip) {
            this.decorations.numericToolTip = {
                range: currentWord.range,
                options: {
                    isWholeLine: false, hoverMessage: [{
                        value: '`' + numericToolTip + '`'
                    }]
                }
            };
            this.updateDecorations();
        }

        if (this.getEffectiveFilters().intel) {
            var lineTokens = _.bind(function (model, line) {
                //Force line's state to be accurate
                if (line > model.getLineCount()) return [];
                var flavour = model.getModeId();
                var tokens = monaco.editor.tokenize(model.getLineContent(line), flavour);
                return tokens.length > 0 ? tokens[0] : [];
            }, this);

            if (this.settings.hoverShowAsmDoc === true &&
                _.some(lineTokens(this.outputEditor.getModel(), currentWord.range.startLineNumber), function (t) {
                    return t.offset + 1 === currentWord.startColumn && t.type === 'keyword.asm';
                })) {
                getAsmInfo(currentWord.word).then(_.bind(function (response) {
                    if (!response) return;
                    this.decorations.asmToolTip = {
                        range: currentWord.range,
                        options: {
                            isWholeLine: false,
                            hoverMessage: [{
                                value: response.tooltip + '\n\nMore information available in the context menu.',
                                isTrusted: true
                            }]
                        }
                    };
                    this.updateDecorations();
                }, this));
            }
        }
    }
};

Compiler.prototype.onAsmToolTip = function (ed) {
    ga.proxy('send', {
        hitType: 'event',
        eventCategory: 'OpenModalPane',
        eventAction: 'AsmDocs'
    });
    if (!this.getEffectiveFilters().intel) return;
    var pos = ed.getPosition();
    var word = ed.getModel().getWordAtPosition(pos);
    if (!word || !word.word) return;
    var opcode = word.word.toUpperCase();

    function newGitHubIssueUrl() {
        return 'https://github.com/mattgodbolt/compiler-explorer/issues/new?title=' +
            encodeURIComponent("[BUG] Problem with " + opcode + " opcode");
    }

    function appendInfo(url) {
        return '<br><br>For more information, visit <a href="' + url +
            '" target="_blank" rel="noopener noreferrer">the ' + opcode +
            ' documentation <sup><small class="fas fa-external-link-alt opens-new-window"' +
            ' title="Opens in a new window"></small></sup></a>.' +
            '<br>If the documentation for this opcode is wrong or broken in some way, ' +
            'please feel free to <a href="' + newGitHubIssueUrl() + '" target="_blank" rel="noopener noreferrer">' +
            'open an issue on GitHub <sup><small class="fas fa-external-link-alt opens-new-window" ' +
            'title="Opens in a new window"></small></sup></a>.';
    }

    getAsmInfo(word.word).then(_.bind(function (asmHelp) {
        if (asmHelp) {
            this.alertSystem.alert(opcode + ' help', asmHelp.html + appendInfo(asmHelp.url), function () {
                ed.focus();
                ed.setPosition(pos);
            });
        } else {
            this.alertSystem.notify('This token was not found in the documentation. Sorry!', {
                group: 'notokenindocs',
                alertClass: 'notification-error',
                dismissTime: 3000
            });
        }
    }, this), _.bind(function (rejection) {
        this.alertSystem
            .notify('There was an error fetching the documentation for this opcode (' + rejection + ').', {
                group: 'notokenindocs',
                alertClass: 'notification-error',
                dismissTime: 3000
            });
    }, this));
};

Compiler.prototype.handleCompilationStatus = function (status) {
    if (!this.statusLabel || !this.statusIcon) return;

    function ariaLabel() {
        // Compiling...
        if (status.code === 4) return "Compiling";
        if (status.compilerOut === 0) {
            // StdErr.length > 0
            if (status.code === 3) return "Compilation succeeded with errors";
            // StdOut.length > 0
            if (status.code === 2) return "Compilation succeeded with warnings";
            return "Compilation succeeded";
        } else {
            // StdErr.length > 0
            if (status.code === 3) return "Compilation failed with errors";
            // StdOut.length > 0
            if (status.code === 2) return "Compilation failed with warnings";
            return "Compilation failed";
        }
    }

    function color() {
        // Compiling...
        if (status.code === 4) return "black";
        if (status.compilerOut === 0) {
            // StdErr.length > 0
            if (status.code === 3) return "#FF6645";
            // StdOut.length > 0
            if (status.code === 2) return "#FF6500";
            return "#12BB12";
        } else {
            // StdErr.length > 0
            if (status.code === 3) return "#FF1212";
            // StdOut.length > 0
            if (status.code === 2) return "#BB8700";
            return "#FF6645";
        }
    }

    this.statusIcon
        .removeClass()
        .addClass('status-icon fas')
        .css('color', color())
        .toggle(status.code !== 0)
        .prop('aria-label', ariaLabel())
        .prop('data-status', status.code)
        .toggleClass('fa-spinner', status.code === 4)
        .toggleClass('fa-times-circle', status.code === 3)
        .toggleClass('fa-check-circle', status.code === 1 || status.code === 2);

    this.statusLabel
        .toggleClass('error', status === 3)
        .toggleClass('warning', status === 2);
};

Compiler.prototype.onLanguageChange = function (editorId, newLangId) {
    if (this.sourceEditorId === editorId) {
        var oldLangId = this.currentLangId;
        this.currentLangId = newLangId;
        // Store the current selected stuff to come back to it later in the same session (Not state stored!)
        this.infoByLang[oldLangId] = {
            compiler: this.compiler && this.compiler.id ? this.compiler.id : options.defaultCompiler[oldLangId],
            options: this.options
        };
        var info = this.infoByLang[this.currentLangId] || {};
        this.initLangAndCompiler({lang: newLangId, compiler: info.compiler});
        this.updateCompilersSelector(info);
        this.updateCompilerUI();
        this.sendCompiler();
        this.saveState();
    }
};

Compiler.prototype.getCurrentLangCompilers = function () {
    return this.compilerService.getCompilersForLang(this.currentLangId);
};

Compiler.prototype.updateCompilersSelector = function (info) {
    this.compilerSelectizer.clearOptions(true);
    this.compilerSelectizer.clearOptionGroups();
    _.each(this.compilerService.getGroupsInUse(this.currentLangId), function (group) {
        this.compilerSelectizer.addOptionGroup(group.value, {label: group.label});
    }, this);
    this.compilerSelectizer.load(_.bind(function (callback) {
        callback(_.map(this.getCurrentLangCompilers(), _.identity));
    }, this));
    this.compilerSelectizer.setValue([this.compiler ? this.compiler.id : null], true);
    this.options = info.options || "";
    this.optionsField.val(this.options);
};

module.exports = {
    Compiler: Compiler
};
