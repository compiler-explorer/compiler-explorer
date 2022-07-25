// Copyright (c) 2012, Compiler Explorer Authors
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
var ga = require('../analytics').ga;
var colour = require('../colour');
var Toggles = require('../widgets/toggles').Toggles;
var FontScale = require('../widgets/fontscale').FontScale;
var Promise = require('es6-promise').Promise;
var Components = require('../components');
var LruCache = require('lru-cache');
var options = require('../options').options;
var monaco = require('monaco-editor');
var Alert = require('../alert').Alert;
var bigInt = require('big-integer');
var LibsWidget = require('../widgets/libs-widget').LibsWidget;
var codeLensHandler = require('../codelens-handler');
var monacoConfig = require('../monaco-config');
var TimingWidget = require('../widgets/timing-info-widget');
var CompilerPicker = require('../compiler-picker').CompilerPicker;
var CompilerService = require('../compiler-service').CompilerService;
var Settings = require('../settings').Settings;
var utils = require('../utils');
var LibUtils = require('../lib-utils');
var getAssemblyDocumentation = require('../api/api').getAssemblyDocumentation;
var PaneRenaming = require('../widgets/pane-renaming').PaneRenaming;
var toolIcons = require.context('../../views/resources/logos', false, /\.(png|svg)$/);

var OpcodeCache = new LruCache({
    max: 64 * 1024,
    length: function (n) {
        return JSON.stringify(n).length;
    },
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

// Disable max line count only for the constructor. Turns out, it needs to do quite a lot of things
// eslint-disable-next-line max-statements
function Compiler(hub, container, state) {
    this.container = container;
    this.hub = hub;
    this.eventHub = hub.createEventHub();
    this.compilerService = hub.compilerService;
    this.domRoot = container.getElement();
    this.domRoot.html($('#compiler').html());
    this.id = state.id || hub.nextCompilerId();
    this.sourceTreeId = state.tree ? state.tree : false;
    if (this.sourceTreeId) {
        this.sourceEditorId = false;
    } else {
        this.sourceEditorId = state.source || 1;
    }
    this.settings = Settings.getStoredSettings();
    this.originalCompilerId = state.compiler;
    this.initLangAndCompiler(state);
    this.infoByLang = {};
    this.deferCompiles = hub.deferred;
    this.needsCompile = false;
    this.deviceViewOpen = false;
    this.options = state.options || options.compileOptions[this.currentLangId];
    this.source = '';
    this.assembly = [];
    this.colours = [];
    this.lastResult = {};
    this.lastTimeTaken = 0;
    this.pendingRequestSentAt = 0;
    this.pendingCMakeRequestSentAt = 0;
    this.nextRequest = null;
    this.nextCMakeRequest = null;
    this.optViewOpen = false;
    this.flagsViewOpen = state.flagsViewOpen || false;
    this.cfgViewOpen = false;
    this.wantOptInfo = state.wantOptInfo;
    this.decorations = {};
    this.prevDecorations = [];
    this.labelDefinitions = {};
    this.alertSystem = new Alert();
    this.alertSystem.prefixMessage = 'Compiler #' + this.id;

    this.awaitingInitialResults = false;
    this.selection = state.selection;

    this.linkedFadeTimeoutId = -1;
    this.toolsMenu = null;

    this.revealJumpStack = [];

    this.paneRenaming = new PaneRenaming(this, state);

    this.initButtons(state);

    var monacoDisassembly = 'asm';
    // Bandaid fix to not have to include monacoDisassembly everywhere in languages.js
    if (languages[this.currentLangId]) {
        switch (languages[this.currentLangId].id) {
            case 'cuda':
                monacoDisassembly = 'ptx';
                break;
            case 'ruby':
                monacoDisassembly = 'asmruby';
                break;
            case 'mlir':
                monacoDisassembly = 'mlir';
                break;
        }
    }

    this.outputEditor = monaco.editor.create(
        this.monacoPlaceholder[0],
        monacoConfig.extendConfig(
            {
                readOnly: true,
                language: monacoDisassembly,
                glyphMargin: !options.embedded,
                guides: false,
                vimInUse: false,
            },
            this.settings
        )
    );

    this.fontScale = new FontScale(this.domRoot, state, this.outputEditor);
    this.compilerPicker = new CompilerPicker(
        this.domRoot,
        this.hub,
        this.currentLangId,
        this.compiler ? this.compiler.id : null,
        _.bind(this.onCompilerChange, this)
    );

    this.initLibraries(state);

    this.initEditorActions();
    this.initEditorCommands();

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
        eventAction: 'Compiler',
    });

    if (this.sourceTreeId) {
        this.compile();
    }
}

Compiler.prototype.getEditorIdBySourcefile = function (sourcefile) {
    if (this.sourceTreeId) {
        var tree = this.hub.getTreeById(this.sourceTreeId);
        if (tree) {
            return tree.multifileService.getEditorIdByFilename(sourcefile.file);
        }
    } else {
        if (sourcefile !== null && (sourcefile.file === null || sourcefile.mainsource)) {
            return this.sourceEditorId;
        }
    }

    return false;
};

Compiler.prototype.initLangAndCompiler = function (state) {
    var langId = state.lang;
    var compilerId = state.compiler;
    var result = this.compilerService.processFromLangAndCompiler(langId, compilerId);
    this.compiler = result.compiler;
    this.currentLangId = result.langId;
    this.updateLibraries();
};

Compiler.prototype.close = function () {
    codeLensHandler.unregister(this.id);
    this.eventHub.unsubscribe();
    this.eventHub.emit('compilerClose', this.id, this.sourceTreeId);
    this.outputEditor.dispose();
};

// eslint-disable-next-line max-statements
Compiler.prototype.initPanerButtons = function () {
    var outputConfig = _.bind(function () {
        return Components.getOutput(this.id, this.sourceEditorId, this.sourceTreeId);
    }, this);

    this.container.layoutManager.createDragSource(this.outputBtn, outputConfig);
    this.outputBtn.click(
        _.bind(function () {
            var insertPoint =
                this.hub.findParentRowOrColumn(this.container) || this.container.layoutManager.root.contentItems[0];
            insertPoint.addChild(outputConfig);
        }, this)
    );

    var cloneComponent = _.bind(function () {
        var currentState = this.currentState();
        // Delete the saved id to force a new one
        delete currentState.id;
        return {
            type: 'component',
            componentName: 'compiler',
            componentState: currentState,
        };
    }, this);
    var createOptView = _.bind(function () {
        return Components.getOptViewWith(
            this.id,
            this.source,
            this.lastResult.optOutput,
            this.getCompilerName(),
            this.sourceEditorId,
            this.sourceTreeId
        );
    }, this);

    var createFlagsView = _.bind(function () {
        return Components.getFlagsViewWith(this.id, this.getCompilerName(), this.optionsField.val());
    }, this);

    if (this.flagsViewOpen) {
        createFlagsView();
    }

    var createPpView = _.bind(function () {
        return Components.getPpViewWith(
            this.id,
            this.source,
            this.lastResult.ppOutput,
            this.getCompilerName(),
            this.sourceEditorId,
            this.sourceTreeId
        );
    }, this);

    var createAstView = _.bind(function () {
        return Components.getAstViewWith(
            this.id,
            this.source,
            this.lastResult.astOutput,
            this.getCompilerName(),
            this.sourceEditorId,
            this.sourceTreeId
        );
    }, this);

    var createIrView = _.bind(function () {
        return Components.getIrViewWith(
            this.id,
            this.source,
            this.lastResult.irOutput,
            this.getCompilerName(),
            this.sourceEditorId,
            this.sourceTreeId
        );
    }, this);

    var createLLVMOptPipelineView = _.bind(function () {
        return Components.getLLVMOptPipelineViewWith(
            this.id,
            this.getCompilerName(),
            this.sourceEditorId,
            this.sourceTreeId
        );
    }, this);

    var createDeviceView = _.bind(function () {
        return Components.getDeviceViewWith(
            this.id,
            this.source,
            this.lastResult.devices,
            this.getCompilerName(),
            this.sourceEditorId,
            this.sourceTreeId
        );
    }, this);

    var createRustMirView = _.bind(function () {
        return Components.getRustMirViewWith(
            this.id,
            this.source,
            this.lastResult.rustMirOutput,
            this.getCompilerName(),
            this.sourceEditorId,
            this.sourceTreeId
        );
    }, this);

    var createRustMacroExpView = _.bind(function () {
        return Components.getRustMacroExpViewWith(
            this.id,
            this.source,
            this.lastResult.rustMacroExpOutput,
            this.getCompilerName(),
            this.sourceEditorId,
            this.sourceTreeId
        );
    }, this);

    var createRustHirView = _.bind(function () {
        return Components.getRustHirViewWith(
            this.id,
            this.source,
            this.lastResult.rustHirOutput,
            this.getCompilerName(),
            this.sourceEditorId,
            this.sourceTreeId
        );
    }, this);

    var createHaskellCoreView = _.bind(function () {
        return Components.getHaskellCoreViewWith(
            this.id,
            this.source,
            this.lastResult.haskellCoreOutput,
            this.getCompilerName(),
            this.sourceEditorId,
            this.sourceTreeId
        );
    }, this);

    var createHaskellStgView = _.bind(function () {
        return Components.getHaskellStgViewWith(
            this.id,
            this.source,
            this.lastResult.haskellStgOutput,
            this.getCompilerName(),
            this.sourceEditorId,
            this.sourceTreeId
        );
    }, this);

    var createHaskellCmmView = _.bind(function () {
        return Components.getHaskellCmmViewWith(
            this.id,
            this.source,
            this.lastResult.haskellCmmOutput,
            this.getCompilerName(),
            this.sourceEditorId,
            this.sourceTreeId
        );
    }, this);

    var createGccDumpView = _.bind(function () {
        return Components.getGccDumpViewWith(
            this.id,
            this.getCompilerName(),
            this.sourceEditorId,
            this.sourceTreeId,
            this.lastResult.gccDumpOutput
        );
    }, this);

    var createGnatDebugTreeView = _.bind(function () {
        return Components.getGnatDebugTreeViewWith(
            this.id,
            this.source,
            this.lastResult.gnatDebugTreeOutput,
            this.getCompilerName(),
            this.sourceEditorId,
            this.sourceTreeId
        );
    }, this);

    var createGnatDebugView = _.bind(function () {
        return Components.getGnatDebugViewWith(
            this.id,
            this.source,
            this.lastResult.gnatDebugOutput,
            this.getCompilerName(),
            this.sourceEditorId,
            this.sourceTreeId
        );
    }, this);

    var createCfgView = _.bind(function () {
        return Components.getCfgViewWith(this.id, this.sourceEditorId, this.sourceTreeId);
    }, this);

    var createExecutor = _.bind(function () {
        var currentState = this.currentState();
        var editorId = currentState.source;
        var treeId = currentState.tree;
        var langId = currentState.lang;
        var compilerId = currentState.compiler;
        var libs = [];
        _.each(this.libsWidget.getLibsInUse(), function (item) {
            libs.push({
                name: item.libId,
                ver: item.versionId,
            });
        });
        return Components.getExecutorWith(editorId, langId, compilerId, libs, currentState.options, treeId);
    }, this);

    var panerDropdown = this.domRoot.find('.pane-dropdown');
    var togglePannerAdder = function () {
        panerDropdown.dropdown('toggle');
    };

    this.container.layoutManager
        .createDragSource(this.domRoot.find('.btn.add-compiler'), cloneComponent)
        ._dragListener.on('dragStart', togglePannerAdder);

    this.domRoot.find('.btn.add-compiler').click(
        _.bind(function () {
            var insertPoint =
                this.hub.findParentRowOrColumn(this.container) || this.container.layoutManager.root.contentItems[0];
            insertPoint.addChild(cloneComponent);
        }, this)
    );

    this.container.layoutManager
        .createDragSource(this.optButton, createOptView)
        ._dragListener.on('dragStart', togglePannerAdder);

    this.optButton.click(
        _.bind(function () {
            var insertPoint =
                this.hub.findParentRowOrColumn(this.container) || this.container.layoutManager.root.contentItems[0];
            insertPoint.addChild(createOptView);
        }, this)
    );

    var popularArgumentsMenu = this.domRoot.find('div.populararguments div.dropdown-menu');
    this.container.layoutManager
        .createDragSource(this.flagsButton, createFlagsView)
        ._dragListener.on('dragStart', togglePannerAdder);

    this.flagsButton.click(
        _.bind(function () {
            var insertPoint =
                this.hub.findParentRowOrColumn(this.container) || this.container.layoutManager.root.contentItems[0];
            insertPoint.addChild(createFlagsView);
        }, this)
    );

    popularArgumentsMenu.append(this.flagsButton);

    this.container.layoutManager
        .createDragSource(this.ppButton, createPpView)
        ._dragListener.on('dragStart', togglePannerAdder);

    this.ppButton.click(
        _.bind(function () {
            var insertPoint =
                this.hub.findParentRowOrColumn(this.container) || this.container.layoutManager.root.contentItems[0];
            insertPoint.addChild(createPpView);
        }, this)
    );

    this.container.layoutManager
        .createDragSource(this.astButton, createAstView)
        ._dragListener.on('dragStart', togglePannerAdder);

    this.astButton.click(
        _.bind(function () {
            var insertPoint =
                this.hub.findParentRowOrColumn(this.container) || this.container.layoutManager.root.contentItems[0];
            insertPoint.addChild(createAstView);
        }, this)
    );

    this.container.layoutManager
        .createDragSource(this.irButton, createIrView)
        ._dragListener.on('dragStart', togglePannerAdder);

    this.irButton.click(
        _.bind(function () {
            var insertPoint =
                this.hub.findParentRowOrColumn(this.container) || this.container.layoutManager.root.contentItems[0];
            insertPoint.addChild(createIrView);
        }, this)
    );

    this.container.layoutManager
        .createDragSource(this.llvmOptPipelineButton, createLLVMOptPipelineView)
        ._dragListener.on('dragStart', togglePannerAdder);

    this.llvmOptPipelineButton.click(
        _.bind(function () {
            var insertPoint =
                this.hub.findParentRowOrColumn(this.container) || this.container.layoutManager.root.contentItems[0];
            insertPoint.addChild(createLLVMOptPipelineView);
        }, this)
    );

    this.container.layoutManager
        .createDragSource(this.deviceButton, createDeviceView)
        ._dragListener.on('dragStart', togglePannerAdder);

    this.deviceButton.click(
        _.bind(function () {
            var insertPoint =
                this.hub.findParentRowOrColumn(this.container) || this.container.layoutManager.root.contentItems[0];
            insertPoint.addChild(createDeviceView);
        }, this)
    );

    this.container.layoutManager
        .createDragSource(this.rustMirButton, createRustMirView)
        ._dragListener.on('dragStart', togglePannerAdder);

    this.rustMirButton.click(
        _.bind(function () {
            var insertPoint =
                this.hub.findParentRowOrColumn(this.container) || this.container.layoutManager.root.contentItems[0];
            insertPoint.addChild(createRustMirView);
        }, this)
    );

    this.container.layoutManager
        .createDragSource(this.haskellCoreButton, createHaskellCoreView)
        ._dragListener.on('dragStart', togglePannerAdder);

    this.haskellCoreButton.click(
        _.bind(function () {
            var insertPoint =
                this.hub.findParentRowOrColumn(this.container) || this.container.layoutManager.root.contentItems[0];
            insertPoint.addChild(createHaskellCoreView);
        }, this)
    );

    this.container.layoutManager
        .createDragSource(this.haskellStgButton, createHaskellStgView)
        ._dragListener.on('dragStart', togglePannerAdder);

    this.haskellStgButton.click(
        _.bind(function () {
            var insertPoint =
                this.hub.findParentRowOrColumn(this.container) || this.container.layoutManager.root.contentItems[0];
            insertPoint.addChild(createHaskellStgView);
        }, this)
    );

    this.container.layoutManager
        .createDragSource(this.haskellCmmButton, createHaskellCmmView)
        ._dragListener.on('dragStart', togglePannerAdder);

    this.haskellCmmButton.click(
        _.bind(function () {
            var insertPoint =
                this.hub.findParentRowOrColumn(this.container) || this.container.layoutManager.root.contentItems[0];
            insertPoint.addChild(createHaskellCmmView);
        }, this)
    );

    this.container.layoutManager
        .createDragSource(this.rustMacroExpButton, createRustMacroExpView)
        ._dragListener.on('dragStart', togglePannerAdder);

    this.rustMacroExpButton.click(
        _.bind(function () {
            var insertPoint =
                this.hub.findParentRowOrColumn(this.container) || this.container.layoutManager.root.contentItems[0];
            insertPoint.addChild(createRustMacroExpView);
        }, this)
    );

    this.container.layoutManager
        .createDragSource(this.rustHirButton, createRustHirView)
        ._dragListener.on('dragStart', togglePannerAdder);

    this.rustHirButton.click(
        _.bind(function () {
            var insertPoint =
                this.hub.findParentRowOrColumn(this.container) || this.container.layoutManager.root.contentItems[0];
            insertPoint.addChild(createRustHirView);
        }, this)
    );

    this.container.layoutManager
        .createDragSource(this.gccDumpButton, createGccDumpView)
        ._dragListener.on('dragStart', togglePannerAdder);

    this.gccDumpButton.click(
        _.bind(function () {
            var insertPoint =
                this.hub.findParentRowOrColumn(this.container) || this.container.layoutManager.root.contentItems[0];
            insertPoint.addChild(createGccDumpView);
        }, this)
    );

    this.container.layoutManager
        .createDragSource(this.gnatDebugTreeButton, createGnatDebugTreeView)
        ._dragListener.on('dragStart', togglePannerAdder);

    this.gnatDebugTreeButton.click(
        _.bind(function () {
            var insertPoint =
                this.hub.findParentRowOrColumn(this.container) || this.container.layoutManager.root.contentItems[0];
            insertPoint.addChild(createGnatDebugTreeView);
        }, this)
    );

    this.container.layoutManager
        .createDragSource(this.gnatDebugButton, createGnatDebugView)
        ._dragListener.on('dragStart', togglePannerAdder);

    this.gnatDebugButton.click(
        _.bind(function () {
            var insertPoint =
                this.hub.findParentRowOrColumn(this.container) || this.container.layoutManager.root.contentItems[0];
            insertPoint.addChild(createGnatDebugView);
        }, this)
    );

    this.container.layoutManager
        .createDragSource(this.cfgButton, createCfgView)
        ._dragListener.on('dragStart', togglePannerAdder);

    this.cfgButton.click(
        _.bind(function () {
            var insertPoint =
                this.hub.findParentRowOrColumn(this.container) || this.container.layoutManager.root.contentItems[0];
            insertPoint.addChild(createCfgView);
        }, this)
    );

    this.container.layoutManager
        .createDragSource(this.executorButton, createExecutor)
        ._dragListener.on('dragStart', togglePannerAdder);

    this.executorButton.click(
        _.bind(function () {
            var insertPoint =
                this.hub.findParentRowOrColumn(this.container) || this.container.layoutManager.root.contentItems[0];
            insertPoint.addChild(createExecutor);
        }, this)
    );

    this.initToolButtons(togglePannerAdder);
};

Compiler.prototype.undefer = function () {
    this.deferCompiles = false;
    if (this.needsCompile) {
        this.compile();
    }
};

Compiler.prototype.resize = function () {
    _.defer(function (self) {
        var topBarHeight = utils.updateAndCalcTopBarHeight(self.domRoot, self.topBar, self.hideable);
        var bottomBarHeight = self.bottomBar.outerHeight(true);
        self.outputEditor.layout({
            width: self.domRoot.width(),
            height: self.domRoot.height() - topBarHeight - bottomBarHeight,
        });
    }, this);
};

// Returns a label name if it can be found in the given position, otherwise
// returns null.
Compiler.prototype.getLabelAtPosition = function (position) {
    var asmLine = this.assembly[position.lineNumber - 1];
    // Outdated position.lineNumber can happen (Between compilations?) - Check for those and skip
    if (asmLine) {
        var column = position.column;
        var labels = asmLine.labels || [];

        for (var i = 0; i < labels.length; ++i) {
            if (column >= labels[i].range.startCol && column < labels[i].range.endCol) {
                return labels[i];
            }
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
    var endLineContent = this.outputEditor.getModel().getLineContent(labelDefLineNum);

    this.pushRevealJump();

    this.outputEditor.setSelection(
        new monaco.Selection(labelDefLineNum, 0, labelDefLineNum, endLineContent.length + 1)
    );

    // Jump to the given line.
    this.outputEditor.revealLineInCenter(labelDefLineNum);
};

Compiler.prototype.pushRevealJump = function () {
    this.revealJumpStack.push(this.outputEditor.saveViewState());
    this.revealJumpStackHasElementsCtxKey.set(true);
};

Compiler.prototype.popAndRevealJump = function () {
    if (this.revealJumpStack.length > 0) {
        this.outputEditor.restoreViewState(this.revealJumpStack.pop());
        this.revealJumpStackHasElementsCtxKey.set(this.revealJumpStack.length > 0);
    }
};

Compiler.prototype.initEditorActions = function () {
    this.isLabelCtxKey = this.outputEditor.createContextKey('isLabel', true);
    this.revealJumpStackHasElementsCtxKey = this.outputEditor.createContextKey('hasRevealJumpStackElements', false);
    this.isAsmKeywordCtxKey = this.outputEditor.createContextKey('isAsmKeyword', true);

    this.outputEditor.addAction({
        id: 'jumptolabel',
        label: 'Jump to label',
        keybindings: [monaco.KeyMod.CtrlCmd | monaco.KeyCode.Enter],
        precondition: 'isLabel',
        contextMenuGroupId: 'navigation',
        contextMenuOrder: 1.5,
        run: _.bind(function (ed) {
            var position = ed.getPosition();
            if (position != null) {
                this.jumpToLabel(position);
            }
        }, this),
    });

    // Hiding the 'Jump to label' context menu option if no label can be found
    // in the clicked position.
    var contextmenu = this.outputEditor.getContribution('editor.contrib.contextmenu');
    var realMethod = contextmenu._onContextMenu;
    contextmenu._onContextMenu = _.bind(function (e) {
        if (e && e.target && e.target.position) {
            if (this.isLabelCtxKey) {
                var label = this.getLabelAtPosition(e.target.position);
                this.isLabelCtxKey.set(label !== null);
            }

            if (this.isAsmKeywordCtxKey) {
                if (!this.compiler.supportsAsmDocs) {
                    // No need to show the "Show asm documentation" if it's just going to fail.
                    // This is useful for things like xtensa which define an instructionSet but have no docs associated
                    this.isAsmKeywordCtxKey.set(false);
                } else {
                    var currentWord = this.outputEditor.getModel().getWordAtPosition(e.target.position);
                    if (currentWord) {
                        currentWord.range = new monaco.Range(
                            e.target.position.lineNumber,
                            Math.max(currentWord.startColumn, 1),
                            e.target.position.lineNumber,
                            currentWord.endColumn
                        );
                        if (currentWord.word) {
                            this.isAsmKeywordCtxKey.set(this.isWordAsmKeyword(currentWord));
                        }
                    }
                }
            }
        }
        realMethod.apply(contextmenu, arguments);
    }, this);

    this.outputEditor.addAction({
        id: 'returnfromreveal',
        label: 'Return from reveal jump',
        keybindings: [monaco.KeyMod.CtrlCmd | monaco.KeyMod.Shift | monaco.KeyCode.Enter],
        contextMenuGroupId: 'navigation',
        contextMenuOrder: 1.4,
        precondition: 'hasRevealJumpStackElements',
        run: _.bind(function () {
            this.popAndRevealJump();
        }, this),
    });

    this.outputEditor.addAction({
        id: 'viewsource',
        label: 'Scroll to source',
        keybindings: [monaco.KeyMod.CtrlCmd | monaco.KeyCode.F10],
        keybindingContext: null,
        contextMenuGroupId: 'navigation',
        contextMenuOrder: 1.5,
        run: _.bind(function (ed) {
            var position = ed.getPosition();
            if (position != null) {
                var desiredLine = position.lineNumber - 1;
                var source = this.assembly[desiredLine].source;
                if (source && source.line > 0) {
                    var editorId = this.getEditorIdBySourcefile(source);
                    if (editorId) {
                        // a null file means it was the user's source
                        this.eventHub.emit('editorLinkLine', editorId, source.line, -1, -1, true);
                    }
                }
            }
        }, this),
    });

    this.outputEditor.addAction({
        id: 'viewasmdoc',
        label: 'View assembly documentation',
        keybindings: [monaco.KeyMod.CtrlCmd | monaco.KeyCode.F8],
        keybindingContext: null,
        precondition: 'isAsmKeyword',
        contextMenuGroupId: 'help',
        contextMenuOrder: 1.5,
        run: _.bind(this.onAsmToolTip, this),
    });

    this.outputEditor.addAction({
        id: 'toggleColourisation',
        label: 'Toggle colourisation',
        keybindings: [monaco.KeyMod.CtrlCmd | monaco.KeyMod.Shift | monaco.KeyCode.F1],
        keybindingContext: null,
        run: _.bind(function () {
            this.eventHub.emit('modifySettings', {
                colouriseAsm: !this.settings.colouriseAsm,
            });
        }, this),
    });
};

Compiler.prototype.initEditorCommands = function () {
    this.outputEditor.addAction({
        id: 'dumpAsm',
        label: 'Developer: Dump asm',
        run: _.bind(function () {
            // eslint-disable-next-line no-console
            console.log(this.assembly);
        }, this),
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
    if (content.componentName === 'tool') {
        if (content.componentState.compiler === this.id) {
            tools.push({
                id: content.componentState.toolId,
                args: content.componentState.args,
                stdin: content.componentState.stdin,
            });
        }
    } else if (content.content) {
        _.each(
            content.content,
            function (subcontent) {
                tools = this.findTools(subcontent, tools);
            },
            this
        );
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
            stdin: newToolSettings.stdin,
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
    this.compileInfoLabel.text(' - Compiling...');
    var options = {
        userArguments: this.options,
        compilerOptions: {
            producePp: this.ppViewOpen ? this.ppOptions : false,
            produceAst: this.astViewOpen,
            produceGccDump: {
                opened: this.gccDumpViewOpen,
                pass: this.gccDumpPassSelected,
                treeDump: this.treeDumpEnabled,
                rtlDump: this.rtlDumpEnabled,
                ipaDump: this.ipaDumpEnabled,
                dumpFlags: this.dumpFlags,
            },
            produceOptInfo: this.wantOptInfo,
            produceCfg: this.cfgViewOpen,
            produceGnatDebugTree: this.gnatDebugTreeViewOpen,
            produceGnatDebug: this.gnatDebugViewOpen,
            produceIr: this.irViewOpen,
            produceLLVMOptPipeline: this.llvmOptPipelineViewOpen ? this.llvmOptPipelineOptions : false,
            produceDevice: this.deviceViewOpen,
            produceRustMir: this.rustMirViewOpen,
            produceRustMacroExp: this.rustMacroExpViewOpen,
            produceRustHir: this.rustHirViewOpen,
            produceHaskellCore: this.haskellCoreViewOpen,
            produceHaskellStg: this.haskellStgViewOpen,
            produceHaskellCmm: this.haskellCmmViewOpen,
        },
        filters: this.getEffectiveFilters(),
        tools: this.getActiveTools(newTools),
        libraries: [],
    };

    _.each(this.libsWidget.getLibsInUse(), function (item) {
        options.libraries.push({
            id: item.libId,
            version: item.versionId,
        });
    });

    if (this.sourceTreeId) {
        this.compileFromTree(options, bypassCache);
    } else {
        this.compileFromEditorSource(options, bypassCache);
    }
};

Compiler.prototype.compileFromTree = function (options, bypassCache) {
    var tree = this.hub.getTreeById(this.sourceTreeId);
    if (!tree) {
        this.sourceTreeId = false;
        this.compileFromEditorSource(options, bypassCache);
        return;
    }

    var request = {
        source: tree.multifileService.getMainSource(),
        compiler: this.compiler ? this.compiler.id : '',
        options: options,
        lang: this.currentLangId,
        files: tree.multifileService.getFiles(),
    };

    var fetches = [];
    fetches.push(
        this.compilerService.expand(request.source).then(function (contents) {
            request.source = contents;
        })
    );

    for (var i = 0; i < request.files.length; i++) {
        var file = request.files[i];
        fetches.push(
            this.compilerService.expand(file.contents).then(function (contents) {
                file.contents = contents;
            })
        );
    }

    var self = this;
    Promise.all(fetches).then(function () {
        var treeState = tree.currentState();
        var cmakeProject = tree.multifileService.isACMakeProject();

        if (bypassCache) request.bypassCache = true;
        if (!self.compiler) {
            self.onCompileResponse(request, errorResult('<Please select a compiler>'), false);
        } else if (cmakeProject && request.source === '') {
            self.onCompileResponse(request, errorResult('<Please supply a CMakeLists.txt>'), false);
        } else {
            if (cmakeProject) {
                request.options.compilerOptions.cmakeArgs = treeState.cmakeArgs;
                request.options.compilerOptions.customOutputFilename = treeState.customOutputFilename;
                self.sendCMakeCompile(request);
            } else {
                self.sendCompile(request);
            }
        }
    });
};

Compiler.prototype.compileFromEditorSource = function (options, bypassCache) {
    this.compilerService.expand(this.source).then(
        _.bind(function (expanded) {
            var request = {
                source: expanded || '',
                compiler: this.compiler ? this.compiler.id : '',
                options: options,
                lang: this.currentLangId,
                files: [],
            };
            if (bypassCache) request.bypassCache = true;
            if (!this.compiler) {
                this.onCompileResponse(request, errorResult('<Please select a compiler>'), false);
            } else {
                this.sendCompile(request);
            }
        }, this)
    );
};

Compiler.prototype.sendCMakeCompile = function (request) {
    var onCompilerResponse = _.bind(this.onCMakeResponse, this);

    if (this.pendingCMakeRequestSentAt) {
        // If we have a request pending, then just store this request to do once the
        // previous request completes.
        this.nextCMakeRequest = request;
        return;
    }
    this.eventHub.emit('compiling', this.id, this.compiler);
    // Display the spinner
    this.handleCompilationStatus({code: 4});
    this.pendingCMakeRequestSentAt = Date.now();
    // After a short delay, give the user some indication that we're working on their
    // compilation.
    var progress = setTimeout(
        _.bind(function () {
            this.setAssembly({asm: fakeAsm('<Compiling...>')}, 0);
        }, this),
        500
    );
    this.compilerService
        .submitCMake(request)
        .then(function (x) {
            clearTimeout(progress);
            onCompilerResponse(request, x.result, x.localCacheHit);
        })
        .catch(function (x) {
            clearTimeout(progress);
            var message = 'Unknown error';
            if (_.isString(x)) {
                message = x;
            } else if (x) {
                message = x.error || x.code || message;
            }
            onCompilerResponse(request, errorResult('<Compilation failed: ' + message + '>'), false);
        });
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
    var progress = setTimeout(
        _.bind(function () {
            this.setAssembly({asm: fakeAsm('<Compiling...>')}, 0);
        }, this),
        500
    );
    this.compilerService
        .submit(request)
        .then(function (x) {
            clearTimeout(progress);
            onCompilerResponse(request, x.result, x.localCacheHit);
        })
        .catch(function (e) {
            clearTimeout(progress);
            var message = 'Unknown error';
            if (_.isString(e)) {
                message = e;
            } else if (e) {
                message = e.error || e.code || e.message;
                if (e.stack) {
                    // eslint-disable-next-line no-console
                    console.log(e);
                }
            }
            onCompilerResponse(request, errorResult('<Compilation failed: ' + message + '>'), false);
        });
};

Compiler.prototype.setNormalMargin = function () {
    this.outputEditor.updateOptions({
        lineNumbers: true,
        lineNumbersMinChars: 1,
    });
};

Compiler.prototype.setBinaryMargin = function () {
    this.outputEditor.updateOptions({
        lineNumbersMinChars: 6,
        lineNumbers: _.bind(this.getBinaryForLine, this),
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

Compiler.prototype.setAssembly = function (result, filteredCount) {
    var asm = result.asm || fakeAsm('<No output>');
    this.assembly = asm;
    if (!this.outputEditor || !this.outputEditor.getModel()) return;
    var editorModel = this.outputEditor.getModel();
    var msg = '<No assembly generated>';
    if (asm.length) {
        msg = _.pluck(asm, 'text').join('\n');
    } else if (filteredCount > 0) {
        msg = '<No assembly to display (~' + filteredCount + (filteredCount === 1 ? ' line' : ' lines') + ' filtered)>';
    }

    if (asm.length === 1 && result.code !== 0 && (result.stderr || result.stdout)) {
        msg += '\n\n# For more information see the output window';
        if (!this.isOutputOpened) {
            msg += '\n# To open the output window, click or drag the "Output" icon at the bottom of this window';
        }
    }

    editorModel.setValue(msg);

    if (!this.awaitingInitialResults) {
        if (this.selection) {
            this.outputEditor.setSelection(this.selection);
            this.outputEditor.revealLinesInCenter(this.selection.startLineNumber, this.selection.endLineNumber);
        }
        this.awaitingInitialResults = true;
    } else {
        var visibleRanges = this.outputEditor.getVisibleRanges();
        var currentTopLine = visibleRanges.length > 0 ? visibleRanges[0].startLineNumber : 1;
        this.outputEditor.revealLine(currentTopLine);
    }

    this.decorations.labelUsages = [];
    _.each(
        this.assembly,
        _.bind(function (obj, line) {
            if (!obj.labels || !obj.labels.length) return;

            obj.labels.forEach(function (label) {
                this.decorations.labelUsages.push({
                    range: new monaco.Range(line + 1, label.range.startCol, line + 1, label.range.endCol),
                    options: {
                        inlineClassName: 'asm-label-link',
                        hoverMessage: [
                            {
                                value: 'Ctrl + Left click to follow the label',
                            },
                        ],
                    },
                });
            }, this);
        }, this)
    );
    this.updateDecorations();

    var codeLenses = [];
    if (this.getEffectiveFilters().binary || result.forceBinaryView) {
        this.setBinaryMargin();
        _.each(
            this.assembly,
            _.bind(function (obj, line) {
                if (obj.opcodes) {
                    var address = obj.address ? obj.address.toString(16) : '';
                    codeLenses.push({
                        range: {
                            startLineNumber: line + 1,
                            startColumn: 1,
                            endLineNumber: line + 2,
                            endColumn: 1,
                        },
                        id: address,
                        command: {
                            title: obj.opcodes.join(' '),
                        },
                    });
                }
            }, this)
        );
    } else {
        this.setNormalMargin();
    }

    if (this.settings.enableCodeLens) {
        codeLensHandler.registerLensesForCompiler(this.id, editorModel, codeLenses);

        var currentAsmLang = editorModel.getLanguageId();
        codeLensHandler.registerProviderForLanguage(currentAsmLang);
    } else {
        // Make sure the codelens is disabled
        codeLensHandler.unregister(this.id);
    }
};

function errorResult(text) {
    return {asm: fakeAsm(text), code: -1, stdout: '', stderr: ''};
}

function fakeAsm(text) {
    return [{text: text, source: null, fake: true}];
}

Compiler.prototype.doNextCompileRequest = function () {
    if (this.nextRequest) {
        var next = this.nextRequest;
        this.nextRequest = null;
        this.sendCompile(next);
    }
};

Compiler.prototype.doNextCMakeRequest = function () {
    if (this.nextCMakeRequest) {
        var next = this.nextCMakeRequest;
        this.nextCMakeRequest = null;
        this.sendCMakeCompile(next);
    }
};

Compiler.prototype.onCMakeResponse = function (request, result, cached) {
    result.source = this.source;
    this.lastResult = result;
    var timeTaken = Math.max(0, Date.now() - this.pendingCMakeRequestSentAt);
    this.lastTimeTaken = timeTaken;
    var wasRealReply = this.pendingCMakeRequestSentAt > 0;
    this.pendingCMakeRequestSentAt = 0;

    this.handleCompileRequestAndResult(request, result, cached, wasRealReply, timeTaken);

    this.doNextCMakeRequest();
};

Compiler.prototype.handleCompileRequestAndResult = function (request, result, cached, wasRealReply, timeTaken) {
    ga.proxy('send', {
        hitType: 'event',
        eventCategory: 'Compile',
        eventAction: request.compiler,
        eventLabel: request.options.userArguments,
        eventValue: cached ? 1 : 0,
    });
    ga.proxy('send', {
        hitType: 'timing',
        timingCategory: 'Compile',
        timingVar: request.compiler,
        timingValue: timeTaken,
    });

    // Delete trailing empty lines
    if (Array.isArray(result.asm)) {
        var indexToDiscard = _.findLastIndex(result.asm, function (line) {
            return !_.isEmpty(line.text);
        });
        result.asm.splice(indexToDiscard + 1, result.asm.length - indexToDiscard);
    }

    this.labelDefinitions = result.labelDefinitions || {};
    if (result.asm) {
        this.setAssembly(result, result.filteredCount || 0);
    } else if (result.result && result.result.asm) {
        this.setAssembly(result.result, result.result.filteredCount || 0);
    } else {
        result.asm = fakeAsm('<Compilation failed>');
        this.setAssembly(result, 0);
    }

    var stdout = result.stdout || [];
    var stderr = result.stderr || [];
    var failed = result.code ? result.code !== 0 : false;

    if (result.buildsteps) {
        _.each(result.buildsteps, function (step) {
            stdout = stdout.concat(step.stdout || []);
            stderr = stderr.concat(step.stderr || []);
            failed = failed | (step.code !== 0);
        });
    }

    this.handleCompilationStatus(CompilerService.calculateStatusIcon(result));
    this.outputTextCount.text(stdout.length);
    this.outputErrorCount.text(stderr.length);
    if (this.isOutputOpened || (stdout.length === 0 && stderr.length === 0)) {
        this.outputBtn.prop('title', '');
    } else {
        CompilerService.handleOutputButtonTitle(this.outputBtn, result);
    }
    var infoLabelText = '';
    if (cached) {
        infoLabelText = ' - cached';
    } else if (wasRealReply) {
        infoLabelText = ' - ' + timeTaken + 'ms';
    }

    if (result.asmSize) {
        infoLabelText += ' (' + result.asmSize + 'B)';
    }

    if (result.filteredCount && result.filteredCount > 0) {
        infoLabelText += ' ~' + result.filteredCount + (result.filteredCount === 1 ? ' line' : ' lines') + ' filtered';
    }

    this.compileInfoLabel.text(infoLabelText);

    if (result.result) {
        var wasCmake =
            result.buildsteps &&
            _.any(result.buildsteps, function (step) {
                return step.step === 'cmake';
            });
        this.postCompilationResult(request, result.result, wasCmake);
    } else {
        this.postCompilationResult(request, result);
    }

    this.eventHub.emit('compileResult', this.id, this.compiler, result, languages[this.currentLangId]);
};

Compiler.prototype.onCompileResponse = function (request, result, cached) {
    // Save which source produced this change. It should probably be saved earlier though
    result.source = this.source;
    this.lastResult = result;
    var timeTaken = Math.max(0, Date.now() - this.pendingRequestSentAt);
    this.lastTimeTaken = timeTaken;
    var wasRealReply = this.pendingRequestSentAt > 0;
    this.pendingRequestSentAt = 0;

    this.handleCompileRequestAndResult(request, result, cached, wasRealReply, timeTaken);

    this.doNextCompileRequest();
};

Compiler.prototype.postCompilationResult = function (request, result, wasCmake) {
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

    this.checkForUnwiseArguments(result.compilationOptions, wasCmake);
    this.setCompilationOptionsPopover(result.compilationOptions ? result.compilationOptions.join(' ') : '');

    this.checkForHints(result);

    if (result.bbcdiskimage) {
        this.emulateBbcDisk(result.bbcdiskimage);
    }
};

Compiler.prototype.emulateBbcDisk = function (bbcdiskimage) {
    var dialog = $('#jsbeebemu');

    this.alertSystem.notify(
        'Click <a target="_blank" id="emulink" style="cursor:pointer;" click="javascript:;">here</a> to emulate',
        {
            group: 'emulation',
            collapseSimilar: true,
            dismissTime: 10000,
            onBeforeShow: function (elem) {
                elem.find('#emulink').on('click', function () {
                    dialog.modal();

                    var emuwindow = dialog.find('#jsbeebemuframe')[0].contentWindow;
                    var tmstr = Date.now();
                    emuwindow.location =
                        'https://bbc.godbolt.org/?' + tmstr + '#embed&autoboot&disc1=b64data:' + bbcdiskimage;
                });
            },
        }
    );
};

Compiler.prototype.onEditorChange = function (editor, source, langId, compilerId) {
    if (this.sourceTreeId) {
        var tree = this.hub.getTreeById(this.sourceTreeId);
        if (tree) {
            if (tree.multifileService.isEditorPartOfProject(editor)) {
                if (this.settings.compileOnChange) {
                    this.compile();

                    return;
                }
            }
        }
    }

    if (
        editor === this.sourceEditorId &&
        langId === this.currentLangId &&
        (compilerId === undefined || compilerId === this.id)
    ) {
        this.source = source;
        if (this.settings.compileOnChange) {
            this.compile();
        }
    }
};

Compiler.prototype.onCompilerFlagsChange = function (compilerId, compilerFlags) {
    if (compilerId === this.id) {
        this.onOptionsChange(compilerFlags);
    }
};

Compiler.prototype.onToolOpened = function (compilerId, toolSettings) {
    if (this.id === compilerId) {
        var toolId = toolSettings.toolId;

        var buttons = this.toolsMenu.find('button');
        $(buttons).each(
            _.bind(function (idx, button) {
                var toolButton = $(button);
                var toolName = toolButton.data('toolname');
                if (toolId === toolName) {
                    toolButton.prop('disabled', true);
                }
            }, this)
        );

        this.compile(false, toolSettings);
    }
};

Compiler.prototype.onToolClosed = function (compilerId, toolSettings) {
    if (this.id === compilerId) {
        var toolId = toolSettings.toolId;

        var buttons = this.toolsMenu.find('button');
        $(buttons).each(
            _.bind(function (idx, button) {
                var toolButton = $(button);
                var toolName = toolButton.data('toolname');
                if (toolId === toolName) {
                    toolButton.prop('disabled', !this.supportsTool(toolId));
                }
            }, this)
        );
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

Compiler.prototype.onFlagsViewClosed = function (id, compilerFlags) {
    if (this.id === id) {
        this.flagsViewOpen = false;
        this.optionsField.val(compilerFlags);
        this.optionsField.prop('disabled', this.flagsViewOpen);
        this.optionsField.prop('placeholder', this.initialOptionsFieldPlacehoder);
        this.flagsButton.prop('disabled', this.flagsViewOpen);

        this.compilerService.requestPopularArguments(this.compiler.id, compilerFlags).then(
            _.bind(function (result) {
                if (result && result.result) {
                    this.handlePopularArgumentsResult(result.result);
                }
            }, this)
        );

        this.saveState();
    }
};

Compiler.prototype.onToolSettingsChange = function (id) {
    if (this.id === id) {
        this.compile();
    }
};

Compiler.prototype.onPpViewOpened = function (id) {
    if (this.id === id) {
        this.ppButton.prop('disabled', true);
        this.ppViewOpen = true;
        // the pp view will request compilation once it populates its options so this.compile() is not called here
    }
};

Compiler.prototype.onPpViewClosed = function (id) {
    if (this.id === id) {
        this.ppButton.prop('disabled', false);
        this.ppViewOpen = false;
    }
};

Compiler.prototype.onPpViewOptionsUpdated = function (id, options, reqCompile) {
    if (this.id === id) {
        this.ppOptions = options;
        if (reqCompile) {
            this.compile();
        }
    }
};

Compiler.prototype.onAstViewOpened = function (id) {
    if (this.id === id) {
        this.astButton.prop('disabled', true);
        this.astViewOpen = true;
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

Compiler.prototype.onLLVMOptPipelineViewOpened = function (id) {
    if (this.id === id) {
        this.llvmOptPipelineButton.prop('disabled', true);
        this.llvmOptPipelineViewOpen = true;
        this.compile();
    }
};

Compiler.prototype.onLLVMOptPipelineViewClosed = function (id) {
    if (this.id === id) {
        this.llvmOptPipelineButton.prop('disabled', false);
        this.llvmOptPipelineViewOpen = false;
    }
};

Compiler.prototype.onLLVMOptPipelineViewOptionsUpdated = function (id, options, recompile) {
    if (this.id === id) {
        console.log(options);
        this.llvmOptPipelineOptions = options;
        if (recompile) {
            this.compile();
        }
    }
};

Compiler.prototype.onDeviceViewOpened = function (id) {
    if (this.id === id) {
        this.deviceButton.prop('disabled', true);
        this.deviceViewOpen = true;
        this.compile();
    }
};

Compiler.prototype.onDeviceViewClosed = function (id) {
    if (this.id === id) {
        this.deviceButton.prop('disabled', false);
        this.deviceViewOpen = false;
    }
};

Compiler.prototype.onRustMirViewOpened = function (id) {
    if (this.id === id) {
        this.rustMirButton.prop('disabled', true);
        this.rustMirViewOpen = true;
        this.compile();
    }
};

Compiler.prototype.onRustMirViewClosed = function (id) {
    if (this.id === id) {
        this.rustMirButton.prop('disabled', false);
        this.rustMirViewOpen = false;
    }
};

Compiler.prototype.onHaskellCoreViewOpened = function (id) {
    if (this.id === id) {
        this.haskellCoreButton.prop('disabled', true);
        this.haskellCoreViewOpen = true;
        this.compile();
    }
};

Compiler.prototype.onHaskellCoreViewClosed = function (id) {
    if (this.id === id) {
        this.haskellCoreButton.prop('disabled', false);
        this.haskellCoreViewOpen = false;
    }
};

Compiler.prototype.onHaskellStgViewOpened = function (id) {
    if (this.id === id) {
        this.haskellStgButton.prop('disabled', true);
        this.haskellStgViewOpen = true;
        this.compile();
    }
};

Compiler.prototype.onHaskellStgViewClosed = function (id) {
    if (this.id === id) {
        this.haskellStgButton.prop('disabled', false);
        this.haskellStgViewOpen = false;
    }
};

Compiler.prototype.onHaskellCmmViewOpened = function (id) {
    if (this.id === id) {
        this.haskellCmmButton.prop('disabled', true);
        this.haskellCmmViewOpen = true;
        this.compile();
    }
};

Compiler.prototype.onHaskellCmmViewClosed = function (id) {
    if (this.id === id) {
        this.haskellCmmButton.prop('disabled', false);
        this.haskellCmmViewOpen = false;
    }
};

Compiler.prototype.onGnatDebugTreeViewOpened = function (id) {
    if (this.id === id) {
        this.gnatDebugTreeButton.prop('disabled', true);
        this.gnatDebugTreeViewOpen = true;
        this.compile();
    }
};

Compiler.prototype.onGnatDebugTreeViewClosed = function (id) {
    if (this.id === id) {
        this.gnatDebugTreeButton.prop('disabled', false);
        this.gnatDebugTreeViewOpen = false;
    }
};

Compiler.prototype.onGnatDebugViewOpened = function (id) {
    if (this.id === id) {
        this.gnatDebugButton.prop('disabled', true);
        this.gnatDebugViewOpen = true;
        this.compile();
    }
};

Compiler.prototype.onGnatDebugViewClosed = function (id) {
    if (this.id === id) {
        this.gnatDebugButton.prop('disabled', false);
        this.gnatDebugViewOpen = false;
    }
};

Compiler.prototype.onRustMacroExpViewOpened = function (id) {
    if (this.id === id) {
        this.rustMacroExpButton.prop('disabled', true);
        this.rustMacroExpViewOpen = true;
        this.compile();
    }
};

Compiler.prototype.onRustMacroExpViewClosed = function (id) {
    if (this.id === id) {
        this.rustMacroExpButton.prop('disabled', false);
        this.rustMacroExpViewOpen = false;
    }
};

Compiler.prototype.onRustHirViewOpened = function (id) {
    if (this.id === id) {
        this.rustHirButton.prop('disabled', true);
        this.rustHirViewOpen = true;
        this.compile();
    }
};

Compiler.prototype.onRustHirViewClosed = function (id) {
    if (this.id === id) {
        this.rustHirButton.prop('disabled', false);
        this.rustHirViewOpen = false;
    }
};

Compiler.prototype.onGccDumpUIInit = function (id) {
    if (this.id === id) {
        this.compile();
    }
};

Compiler.prototype.onGccDumpFiltersChanged = function (id, filters, reqCompile) {
    if (this.id === id) {
        this.treeDumpEnabled = filters.treeDump !== false;
        this.rtlDumpEnabled = filters.rtlDump !== false;
        this.ipaDumpEnabled = filters.ipaDump !== false;
        this.dumpFlags = {
            address: filters.addressOption !== false,
            slim: filters.slimOption !== false,
            raw: filters.rawOption !== false,
            details: filters.detailsOption !== false,
            stats: filters.statsOption !== false,
            blocks: filters.blocksOption !== false,
            vops: filters.vopsOption !== false,
            lineno: filters.linenoOption !== false,
            uid: filters.uidOption !== false,
            all: filters.allOption !== false,
        };

        if (reqCompile) {
            this.compile();
        }
    }
};

Compiler.prototype.onGccDumpPassSelected = function (id, passObject, reqCompile) {
    if (this.id === id) {
        this.gccDumpPassSelected = passObject;

        if (reqCompile && passObject !== null) {
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
        delete this.ipaDumpEnabled;
        delete this.dumpFlags;
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

Compiler.prototype.onFlagsViewOpened = function (id) {
    if (this.id === id) {
        this.flagsViewOpen = true;
        this.handlePopularArgumentsResult(false);
        this.optionsField.prop('disabled', this.flagsViewOpen);
        this.optionsField.val('');
        this.optionsField.prop('placeholder', 'see detailed flags window');
        this.flagsButton.prop('disabled', this.flagsViewOpen);
        this.compile();
        this.saveState();
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

Compiler.prototype.initFilterButtons = function () {
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
};

Compiler.prototype.initButtons = function (state) {
    this.filters = new Toggles(this.domRoot.find('.filters'), patchOldFilters(state.filters));

    this.optButton = this.domRoot.find('.btn.view-optimization');
    this.flagsButton = this.domRoot.find('div.populararguments div.dropdown-menu button');
    this.ppButton = this.domRoot.find('.btn.view-pp');
    this.astButton = this.domRoot.find('.btn.view-ast');
    this.irButton = this.domRoot.find('.btn.view-ir');
    this.llvmOptPipelineButton = this.domRoot.find('.btn.view-llvm-opt-pipeline');
    this.deviceButton = this.domRoot.find('.btn.view-device');
    this.gnatDebugTreeButton = this.domRoot.find('.btn.view-gnatdebugtree');
    this.gnatDebugButton = this.domRoot.find('.btn.view-gnatdebug');
    this.rustMirButton = this.domRoot.find('.btn.view-rustmir');
    this.rustMacroExpButton = this.domRoot.find('.btn.view-rustmacroexp');
    this.rustHirButton = this.domRoot.find('.btn.view-rusthir');
    this.haskellCoreButton = this.domRoot.find('.btn.view-haskellCore');
    this.haskellStgButton = this.domRoot.find('.btn.view-haskellStg');
    this.haskellCmmButton = this.domRoot.find('.btn.view-haskellCmm');
    this.gccDumpButton = this.domRoot.find('.btn.view-gccdump');
    this.cfgButton = this.domRoot.find('.btn.view-cfg');
    this.executorButton = this.domRoot.find('.create-executor');
    this.libsButton = this.domRoot.find('.btn.show-libs');

    this.compileInfoLabel = this.domRoot.find('.compile-info');
    this.compileClearCache = this.domRoot.find('.clear-cache');

    this.outputBtn = this.domRoot.find('.output-btn');
    this.outputTextCount = this.domRoot.find('span.text-count');
    this.outputErrorCount = this.domRoot.find('span.err-count');

    this.optionsField = this.domRoot.find('.options');
    this.initialOptionsFieldPlacehoder = this.optionsField.prop('placeholder');
    this.prependOptions = this.domRoot.find('.prepend-options');
    this.fullCompilerName = this.domRoot.find('.full-compiler-name');
    this.fullTimingInfo = this.domRoot.find('.full-timing-info');
    this.setCompilationOptionsPopover(this.compiler ? this.compiler.options : null);
    // Dismiss on any click that isn't either in the opening element, inside
    // the popover or on any alert
    $(document).on(
        'mouseup',
        _.bind(function (e) {
            var target = $(e.target);
            if (
                !target.is(this.prependOptions) &&
                this.prependOptions.has(target).length === 0 &&
                target.closest('.popover').length === 0
            )
                this.prependOptions.popover('hide');

            if (
                !target.is(this.fullCompilerName) &&
                this.fullCompilerName.has(target).length === 0 &&
                target.closest('.popover').length === 0
            )
                this.fullCompilerName.popover('hide');
        }, this)
    );

    this.initFilterButtons(state);

    this.filterExecuteButton.toggle(options.supportsExecute);
    this.filterLibraryCodeButton.toggle(options.supportsLibraryCodeFilter);

    this.optionsField.val(this.options);

    this.shortCompilerName = this.domRoot.find('.short-compiler-name');
    this.compilerPicker = this.domRoot.find('.compiler-picker');
    this.setCompilerVersionPopover({version: '', fullVersion: ''}, '');

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
    this.libsWidget = new LibsWidget(
        this.currentLangId,
        this.compiler,
        this.libsButton,
        state,
        _.bind(this.onLibsChanged, this),
        LibUtils.getSupportedLibraries(
            this.compiler ? this.compiler.libsArr : [],
            this.currentLangId,
            this.compiler ? this.compiler.remote : null
        )
    );
};

Compiler.prototype.updateLibraries = function () {
    if (this.libsWidget) {
        var filteredLibraries = {};
        if (this.compiler) {
            filteredLibraries = LibUtils.getSupportedLibraries(
                this.compiler.libsArr,
                this.currentLangId,
                this.compiler ? this.compiler.remote : null
            );
        }

        this.libsWidget.setNewLangId(this.currentLangId, this.compiler ? this.compiler.id : false, filteredLibraries);
    }
};

Compiler.prototype.isSupportedTool = function (tool) {
    if (this.sourceTreeId) {
        return tool.tool.type === 'postcompilation';
    } else {
        return true;
    }
};

Compiler.prototype.supportsTool = function (toolId) {
    if (!this.compiler) return;

    return _.find(
        this.compiler.tools,
        _.bind(function (tool) {
            return tool.tool.id === toolId && this.isSupportedTool(tool);
        }, this)
    );
};

Compiler.prototype.initToolButton = function (togglePannerAdder, button, toolId) {
    var createToolView = _.bind(function () {
        var args = '';
        var monacoStdin = false;
        var langTools = options.tools[this.currentLangId];
        if (langTools && langTools[toolId] && langTools[toolId].tool) {
            if (langTools[toolId].tool.args !== undefined) {
                args = langTools[toolId].tool.args;
            }
            if (langTools[toolId].tool.monacoStdin !== undefined) {
                monacoStdin = langTools[toolId].tool.monacoStdin;
            }
        }
        return Components.getToolViewWith(this.id, this.sourceEditorId, toolId, args, monacoStdin, this.sourceTreeId);
    }, this);

    this.container.layoutManager
        .createDragSource(button, createToolView)
        ._dragListener.on('dragStart', togglePannerAdder);

    button.click(
        _.bind(function () {
            button.prop('disabled', true);
            var insertPoint =
                this.hub.findParentRowOrColumn(this.container) || this.container.layoutManager.root.contentItems[0];
            insertPoint.addChild(createToolView);
        }, this)
    );
};

Compiler.prototype.initToolButtons = function (togglePannerAdder) {
    this.toolsMenu = this.domRoot.find('.toolsmenu');
    this.toolsMenu.empty();

    if (!this.compiler) return;

    var addTool = _.bind(function (toolName, title, toolIcon, toolIconDark) {
        var btn = $("<button class='dropdown-item btn btn-light btn-sm'>");
        btn.addClass('view-' + toolName);
        btn.data('toolname', toolName);
        if (toolIcon) {
            const light = toolIcons(toolIcon);
            const dark = toolIconDark ? toolIcons(toolIconDark) : light;
            btn.append(
                `<span class="dropdown-icon fas">
                <img src="${light}" class="theme-light-only" width="16px" style="max-height: 16px"/>
                <img src="${dark}" class="theme-dark-only" width="16px" style="max-height: 16px"/>
                </span>`
            );
        } else {
            btn.append("<span class='dropdown-icon fas fa-cog'></span>");
        }
        btn.append(title);
        this.toolsMenu.append(btn);

        if (toolName !== 'none') {
            this.initToolButton(togglePannerAdder, btn, toolName);
        }
    }, this);

    if (_.isEmpty(this.compiler.tools)) {
        addTool('none', 'No tools available');
    } else {
        _.each(
            this.compiler.tools,
            _.bind(function (tool) {
                if (this.isSupportedTool(tool)) {
                    addTool(tool.tool.id, tool.tool.name, tool.tool.icon, tool.tool.darkIcon);
                }
            }, this)
        );
    }
};

Compiler.prototype.enableToolButtons = function () {
    var activeTools = this.getActiveTools();

    var buttons = this.toolsMenu.find('button');
    $(buttons).each(
        _.bind(function (idx, button) {
            var toolButton = $(button);
            var toolName = toolButton.data('toolname');
            toolButton.prop('disabled', !(this.supportsTool(toolName) && !this.isToolActive(activeTools, toolName)));
        }, this)
    );
};

// eslint-disable-next-line max-statements
Compiler.prototype.updateButtons = function () {
    if (!this.compiler) return;
    var filters = this.getEffectiveFilters();
    // We can support intel output if the compiler supports it, or if we're compiling
    // to binary (as we can disassemble it however we like).
    var formatFilterTitle = function (button, title) {
        button.prop(
            'title',
            '[' +
                (button.hasClass('active') ? 'ON' : 'OFF') +
                '] ' +
                title +
                (button.prop('disabled') ? ' [LOCKED]' : '')
        );
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

    if (this.flagsButton) {
        this.flagsButton.prop('disabled', this.flagsViewOpen);
    }
    this.optButton.prop('disabled', this.optViewOpen);
    this.ppButton.prop('disabled', this.ppViewOpen);
    this.astButton.prop('disabled', this.astViewOpen);
    this.irButton.prop('disabled', this.irViewOpen);
    this.llvmOptPipelineButton.prop('disabled', this.llvmOptPipelineViewOpen);
    this.deviceButton.prop('disabled', this.deviceViewOpen);
    this.rustMirButton.prop('disabled', this.rustMirViewOpen);
    this.haskellCoreButton.prop('disabled', this.haskellCoreViewOpen);
    this.haskellStgButton.prop('disabled', this.haskellStgViewOpen);
    this.haskellCmmButton.prop('disabled', this.haskellCmmViewOpen);
    this.rustMacroExpButton.prop('disabled', this.rustMacroExpViewOpen);
    this.rustHirButton.prop('disabled', this.rustHirViewOpen);
    this.cfgButton.prop('disabled', this.cfgViewOpen);
    this.gccDumpButton.prop('disabled', this.gccDumpViewOpen);
    this.gnatDebugTreeButton.prop('disabled', this.gnatDebugTreeViewOpen);
    this.gnatDebugButton.prop('disabled', this.gnatDebugViewOpen);
    // The executorButton does not need to be changed here, because you can create however
    // many executors as you want.

    this.optButton.toggle(!!this.compiler.supportsOptOutput);
    this.ppButton.toggle(!!this.compiler.supportsPpView);
    this.astButton.toggle(!!this.compiler.supportsAstView);
    this.irButton.toggle(!!this.compiler.supportsIrView);
    this.llvmOptPipelineButton.toggle(!!this.compiler.supportsLLVMOptPipelineView);
    this.deviceButton.toggle(!!this.compiler.supportsDeviceAsmView);
    this.rustMirButton.toggle(!!this.compiler.supportsRustMirView);
    this.rustMacroExpButton.toggle(!!this.compiler.supportsRustMacroExpView);
    this.rustHirButton.toggle(!!this.compiler.supportsRustHirView);
    this.haskellCoreButton.toggle(!!this.compiler.supportsHaskellCoreView);
    this.haskellStgButton.toggle(!!this.compiler.supportsHaskellStgView);
    this.haskellCmmButton.toggle(!!this.compiler.supportsHaskellCmmView);
    this.cfgButton.toggle(!!this.compiler.supportsCfg);
    this.gccDumpButton.toggle(!!this.compiler.supportsGccDump);
    this.gnatDebugTreeButton.toggle(!!this.compiler.supportsGnatDebugViews);
    this.gnatDebugButton.toggle(!!this.compiler.supportsGnatDebugViews);
    this.executorButton.toggle(!!this.compiler.supportsExecute);

    this.enableToolButtons();
};

Compiler.prototype.handlePopularArgumentsResult = function (result) {
    var popularArgumentsMenu = $(this.domRoot.find('div.populararguments div.dropdown-menu'));

    while (popularArgumentsMenu.children().length > 1) {
        popularArgumentsMenu.children()[1].remove();
    }

    if (result && !this.flagsViewOpen) {
        _.forEach(
            result,
            _.bind(function (arg, key) {
                var argumentButton = $(document.createElement('button'));
                argumentButton.addClass('dropdown-item btn btn-light btn-sm');
                argumentButton.attr('title', arg.description);
                argumentButton.data('arg', key);
                argumentButton.html(
                    "<div class='argmenuitem'>" +
                        "<span class='argtitle'>" +
                        _.escape(key) +
                        '</span>' +
                        "<span class='argdescription'>" +
                        arg.description +
                        '</span>' +
                        '</div>'
                );

                argumentButton.on(
                    'click',
                    _.bind(function () {
                        var button = argumentButton;
                        var curOptions = this.optionsField.val();
                        if (curOptions.length > 0) {
                            this.optionsField.val(curOptions + ' ' + button.data('arg'));
                        } else {
                            this.optionsField.val(button.data('arg'));
                        }

                        this.optionsField.change();
                    }, this)
                );

                popularArgumentsMenu.append(argumentButton);
            }, this)
        );
    }
};

Compiler.prototype.onFontScale = function () {
    this.saveState();
};

// Disable only for initListeners as there are more and more callbacks.
// eslint-disable-next-line max-statements
Compiler.prototype.initListeners = function () {
    this.filters.on('change', _.bind(this.onFilterChange, this));
    this.fontScale.on('change', _.bind(this.onFontScale, this));
    this.eventHub.on(
        'broadcastFontScale',
        _.bind(function (scale) {
            this.fontScale.setScale(scale);
            this.saveState();
        }, this)
    );
    this.paneRenaming.on('renamePane', this.saveState.bind(this));

    this.container.on('destroy', this.close, this);
    this.container.on('resize', this.resize, this);
    this.container.on('shown', this.resize, this);
    this.container.on(
        'open',
        function () {
            this.eventHub.emit('compilerOpen', this.id, this.sourceEditorId, this.sourceTreeId);
        },
        this
    );
    this.eventHub.on('editorChange', this.onEditorChange, this);
    this.eventHub.on('compilerFlagsChange', this.onCompilerFlagsChange, this);
    this.eventHub.on('editorClose', this.onEditorClose, this);
    this.eventHub.on('treeClose', this.onTreeClose, this);
    this.eventHub.on('colours', this.onColours, this);
    this.eventHub.on('coloursForCompiler', this.onColoursForCompiler, this);
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
    this.eventHub.on('flagsViewOpened', this.onFlagsViewOpened, this);
    this.eventHub.on('flagsViewClosed', this.onFlagsViewClosed, this);
    this.eventHub.on('ppViewOpened', this.onPpViewOpened, this);
    this.eventHub.on('ppViewClosed', this.onPpViewClosed, this);
    this.eventHub.on('ppViewOptionsUpdated', this.onPpViewOptionsUpdated, this);
    this.eventHub.on('astViewOpened', this.onAstViewOpened, this);
    this.eventHub.on('astViewClosed', this.onAstViewClosed, this);
    this.eventHub.on('irViewOpened', this.onIrViewOpened, this);
    this.eventHub.on('irViewClosed', this.onIrViewClosed, this);
    this.eventHub.on('llvmOptPipelineViewOpened', this.onLLVMOptPipelineViewOpened, this);
    this.eventHub.on('llvmOptPipelineViewClosed', this.onLLVMOptPipelineViewClosed, this);
    this.eventHub.on('llvmOptPipelineViewOptionsUpdated', this.onLLVMOptPipelineViewOptionsUpdated, this);
    this.eventHub.on('deviceViewOpened', this.onDeviceViewOpened, this);
    this.eventHub.on('deviceViewClosed', this.onDeviceViewClosed, this);
    this.eventHub.on('rustMirViewOpened', this.onRustMirViewOpened, this);
    this.eventHub.on('rustMirViewClosed', this.onRustMirViewClosed, this);
    this.eventHub.on('rustMacroExpViewOpened', this.onRustMacroExpViewOpened, this);
    this.eventHub.on('rustMacroExpViewClosed', this.onRustMacroExpViewClosed, this);
    this.eventHub.on('rustHirViewOpened', this.onRustHirViewOpened, this);
    this.eventHub.on('rustHirViewClosed', this.onRustHirViewClosed, this);
    this.eventHub.on('haskellCoreViewOpened', this.onHaskellCoreViewOpened, this);
    this.eventHub.on('haskellCoreViewClosed', this.onHaskellCoreViewClosed, this);
    this.eventHub.on('haskellStgViewOpened', this.onHaskellStgViewOpened, this);
    this.eventHub.on('haskellStgViewClosed', this.onHaskellStgViewClosed, this);
    this.eventHub.on('haskellCmmViewOpened', this.onHaskellCmmViewOpened, this);
    this.eventHub.on('haskellCmmViewClosed', this.onHaskellCmmViewClosed, this);
    this.eventHub.on('outputOpened', this.onOutputOpened, this);
    this.eventHub.on('outputClosed', this.onOutputClosed, this);

    this.eventHub.on('gccDumpPassSelected', this.onGccDumpPassSelected, this);
    this.eventHub.on('gccDumpFiltersChanged', this.onGccDumpFiltersChanged, this);
    this.eventHub.on('gccDumpViewOpened', this.onGccDumpViewOpened, this);
    this.eventHub.on('gccDumpViewClosed', this.onGccDumpViewClosed, this);
    this.eventHub.on('gccDumpUIInit', this.onGccDumpUIInit, this);

    this.eventHub.on('gnatDebugTreeViewOpened', this.onGnatDebugTreeViewOpened, this);
    this.eventHub.on('gnatDebugTreeViewClosed', this.onGnatDebugTreeViewClosed, this);
    this.eventHub.on('gnatDebugViewOpened', this.onGnatDebugViewOpened, this);
    this.eventHub.on('gnatDebugViewClosed', this.onGnatDebugViewClosed, this);

    this.eventHub.on('cfgViewOpened', this.onCfgViewOpened, this);
    this.eventHub.on('cfgViewClosed', this.onCfgViewClosed, this);
    this.eventHub.on('resize', this.resize, this);
    this.eventHub.on(
        'requestFilters',
        function (id) {
            if (id === this.id) {
                this.eventHub.emit('filtersChange', this.id, this.getEffectiveFilters());
            }
        },
        this
    );
    this.eventHub.on(
        'requestCompiler',
        function (id) {
            if (id === this.id) {
                this.sendCompiler();
            }
        },
        this
    );
    this.eventHub.on('languageChange', this.onLanguageChange, this);

    this.fullTimingInfo.off('click').click(
        _.bind(function () {
            TimingWidget.displayCompilationTiming(this.lastResult, this.lastTimeTaken);
        }, this)
    );
};

Compiler.prototype.initCallbacks = function () {
    this.initListeners();

    var optionsChange = _.debounce(
        _.bind(function (e) {
            this.onOptionsChange($(e.target).val());
        }, this),
        800
    );

    this.optionsField.on('change', optionsChange).on('keyup', optionsChange);

    this.mouseMoveThrottledFunction = _.throttle(_.bind(this.onMouseMove, this), 50);
    this.outputEditor.onMouseMove(
        _.bind(function (e) {
            this.mouseMoveThrottledFunction(e);
        }, this)
    );

    this.cursorSelectionThrottledFunction = _.throttle(_.bind(this.onDidChangeCursorSelection, this), 500);
    this.outputEditor.onDidChangeCursorSelection(
        _.bind(function (e) {
            this.cursorSelectionThrottledFunction(e);
        }, this)
    );

    this.mouseUpThrottledFunction = _.throttle(_.bind(this.onMouseUp, this), 50);
    this.outputEditor.onMouseUp(
        _.bind(function (e) {
            this.mouseUpThrottledFunction(e);
        }, this)
    );

    this.compileClearCache.on(
        'click',
        _.bind(function () {
            this.compilerService.cache.reset();
            this.compile(true);
        }, this)
    );

    // Dismiss the popover on escape.
    $(document).on(
        'keyup.editable',
        _.bind(function (e) {
            if (e.which === 27) {
                this.libsButton.popover('hide');
            }
        }, this)
    );

    // Dismiss on any click that isn't either in the opening element, inside
    // the popover or on any alert
    $(document).on(
        'click',
        _.bind(function (e) {
            var elem = this.libsButton;
            var target = $(e.target);
            if (!target.is(elem) && elem.has(target).length === 0 && target.closest('.popover').length === 0) {
                elem.popover('hide');
            }
        }, this)
    );

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

function htmlEncode(rawStr) {
    return rawStr.replace(/[\u00A0-\u9999<>&]/g, function (i) {
        return '&#' + i.charCodeAt(0) + ';';
    });
}

Compiler.prototype.checkForHints = function (result) {
    if (result.hints) {
        var self = this;
        result.hints.forEach(function (hint) {
            self.alertSystem.notify(htmlEncode(hint), {
                group: 'hints',
                collapseSimilar: false,
            });
        });
    }
};

Compiler.prototype.checkForUnwiseArguments = function (optionsArray, wasCmake) {
    // Check if any options are in the unwiseOptions array and remember them
    var unwiseOptions = _.intersection(
        optionsArray,
        _.filter(this.compiler.unwiseOptions, function (opt) {
            return opt !== '';
        })
    );

    var options = unwiseOptions.length === 1 ? 'Option ' : 'Options ';
    var names = unwiseOptions.join(', ');
    var are = unwiseOptions.length === 1 ? ' is ' : ' are ';
    var msg = options + names + are + 'not recommended, as behaviour might change based on server hardware.';

    if (_.contains(optionsArray, '-flto') && !this.filters.state.binary && !wasCmake) {
        this.alertSystem.notify('Option -flto is being used without Compile to Binary.', {
            group: 'unwiseOption',
            collapseSimilar: true,
        });
    }

    if (unwiseOptions.length > 0) {
        this.alertSystem.notify(msg, {
            group: 'unwiseOption',
            collapseSimilar: true,
        });
    }
};

Compiler.prototype.updateCompilerInfo = function () {
    this.updateCompilerName();
    if (this.compiler) {
        if (this.compiler.notification) {
            this.alertSystem.notify(this.compiler.notification, {
                group: 'compilerwarning',
                alertClass: 'notification-info',
                dismissTime: 7000,
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

    this.deferCompiles = true;
    this.needsCompile = true;

    this.updateLibraries();
    this.saveState();
    this.updateCompilerUI();

    this.undefer();

    this.sendCompiler();
};

Compiler.prototype.sendCompiler = function () {
    this.eventHub.emit('compiler', this.id, this.compiler, this.options, this.sourceEditorId, this.sourceTreeId);
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

Compiler.prototype.onTreeClose = function (tree) {
    if (tree === this.sourceTreeId) {
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
        tree: this.sourceTreeId,
        options: this.options,
        // NB must *not* be effective filters
        filters: this.filters.get(),
        wantOptInfo: this.wantOptInfo,
        libs: this.libsWidget.get(),
        lang: this.currentLangId,
        selection: this.selection,
        flagsViewOpen: this.flagsViewOpen,
    };
    this.paneRenaming.addState(state);
    this.fontScale.addState(state);
    return state;
};

Compiler.prototype.saveState = function () {
    this.container.setState(this.currentState());
};

Compiler.prototype.onColours = function (editor, colours, scheme) {
    var asmColours = {};
    _.each(
        this.assembly,
        _.bind(function (x, index) {
            if (x.source && x.source.line > 0) {
                var editorId = this.getEditorIdBySourcefile(x.source);
                if (editorId === editor) {
                    if (!asmColours[editorId]) {
                        asmColours[editorId] = {};
                    }
                    asmColours[editorId][index] = colours[x.source.line - 1];
                }
            }
        }, this)
    );

    _.each(
        asmColours,
        _.bind(function (col) {
            this.colours = colour.applyColours(this.outputEditor, col, scheme, this.colours);
        }, this)
    );
};

Compiler.prototype.onColoursForCompiler = function (compilerId, colours, scheme) {
    if (this.id === compilerId) {
        this.colours = colour.applyColours(this.outputEditor, colours, scheme, this.colours);
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
    if (this.sourceEditorId) {
        return compName + ' (' + langName + ', Editor #' + this.sourceEditorId + ', Compiler #' + this.id + ')';
    } else if (this.sourceTreeId) {
        return compName + ' (' + langName + ', Tree #' + this.sourceTreeId + ', Compiler #' + this.id + ')';
    } else {
        return '';
    }
};

Compiler.prototype.updateTitle = function () {
    var name = this.paneName ? this.paneName : this.getPaneName();
    this.container.setTitle(_.escape(name));
};

Compiler.prototype.updateCompilerName = function () {
    var compilerName = this.getCompilerName();
    var compilerVersion = this.compiler ? this.compiler.version : '';
    var compilerFullVersion = this.compiler && this.compiler.fullVersion ? this.compiler.fullVersion : compilerVersion;
    var compilerNotification = this.compiler ? this.compiler.notification : '';
    this.shortCompilerName.text(compilerName);
    this.setCompilerVersionPopover({version: compilerVersion, fullVersion: compilerFullVersion}, compilerNotification);
    this.updateTitle();
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
        this.prevDecorations,
        _.flatten(_.values(this.decorations))
    );
};

Compiler.prototype.clearLinkedLines = function () {
    this.decorations.linkedCode = [];
    this.updateDecorations();
};

Compiler.prototype.onPanesLinkLine = function (compilerId, lineNumber, colBegin, colEnd, revealLine, sender, editorId) {
    if (Number(compilerId) === this.id) {
        var lineNums = [];
        var directlyLinkedLineNums = [];
        var signalFromAnotherPane = sender !== this.getPaneName();
        _.each(
            this.assembly,
            _.bind(function (asmLine, i) {
                if (asmLine.source && asmLine.source.line === lineNumber) {
                    var fileEditorId = this.getEditorIdBySourcefile(asmLine.source);
                    if (fileEditorId && editorId === fileEditorId) {
                        var line = i + 1;
                        lineNums.push(line);
                        var currentCol = asmLine.source.column;
                        if (signalFromAnotherPane && currentCol && colBegin <= currentCol && currentCol <= colEnd) {
                            directlyLinkedLineNums.push(line);
                        }
                    }
                }
            }, this)
        );

        if (revealLine && lineNums[0]) {
            this.pushRevealJump();
            this.hub.activateTabForContainer(this.container);
            this.outputEditor.revealLineInCenter(lineNums[0]);
        }

        var lineClass = sender !== this.getPaneName() ? 'linked-code-decoration-line' : '';
        var linkedLinesDecoration = _.map(lineNums, function (line) {
            return {
                range: new monaco.Range(line, 1, line, 1),
                options: {
                    isWholeLine: true,
                    linesDecorationsClassName: 'linked-code-decoration-margin',
                    className: lineClass,
                },
            };
        });
        var directlyLinkedLinesDecoration = _.map(directlyLinkedLineNums, function (line) {
            return {
                range: new monaco.Range(line, 1, line, 1),
                options: {
                    isWholeLine: true,
                    inlineClassName: 'linked-code-decoration-column',
                },
            };
        });
        this.decorations.linkedCode = linkedLinesDecoration.concat(directlyLinkedLinesDecoration);
        if (this.linkedFadeTimeoutId !== -1) {
            clearTimeout(this.linkedFadeTimeoutId);
        }
        this.linkedFadeTimeoutId = setTimeout(
            _.bind(function () {
                this.clearLinkedLines();
                this.linkedFadeTimeoutId = -1;
            }, this),
            5000
        );
        this.updateDecorations();
    }
};

Compiler.prototype.onCompilerSetDecorations = function (id, lineNums, revealLine) {
    if (Number(id) === this.id) {
        if (revealLine && lineNums[0]) {
            this.pushRevealJump();
            this.outputEditor.revealLineInCenter(lineNums[0]);
        }
        this.decorations.linkedCode = _.map(lineNums, function (line) {
            return {
                range: new monaco.Range(line, 1, line, 1),
                options: {
                    isWholeLine: true,
                    linesDecorationsClassName: 'linked-code-decoration-margin',
                    inlineClassName: 'linked-code-decoration-inline',
                },
            };
        });
        this.updateDecorations();
    }
};

Compiler.prototype.setCompilationOptionsPopover = function (content) {
    this.prependOptions.popover('dispose');
    this.prependOptions.popover({
        content: content || 'No options in use',
        template:
            '<div class="popover' +
            (content ? ' compiler-options-popover' : '') +
            '" role="tooltip"><div class="arrow"></div>' +
            '<h3 class="popover-header"></h3><div class="popover-body"></div></div>',
    });
};

Compiler.prototype.setCompilerVersionPopover = function (version, notification) {
    this.fullCompilerName.popover('dispose');
    // `notification` contains HTML from a config file, so is 'safe'.
    // `version` comes from compiler output, so isn't, and is escaped.
    var bodyContent = $('<div>');
    var versionContent = $('<div>').html(_.escape(version.version));
    bodyContent.append(versionContent);
    if (version.fullVersion && version.fullVersion.trim() !== version.version.trim()) {
        var hiddenSection = $('<div>');
        var lines = _.map(version.fullVersion.split('\n'), function (line) {
            return _.escape(line);
        }).join('<br/>');
        var hiddenVersionText = $('<div>').html(lines).hide();
        var clickToExpandContent = $('<a>')
            .attr('href', 'javascript:;')
            .text('Toggle full version output')
            .on(
                'click',
                _.bind(function () {
                    versionContent.toggle();
                    hiddenVersionText.toggle();
                    this.fullCompilerName.popover('update');
                }, this)
            );
        hiddenSection.append(hiddenVersionText).append(clickToExpandContent);
        bodyContent.append(hiddenSection);
    }
    this.fullCompilerName.popover({
        html: true,
        title: notification
            ? $.parseHTML('<span>Compiler Version: ' + notification + '</span>')[0]
            : 'Full compiler version',
        content: bodyContent,
        template:
            '<div class="popover' +
            (version ? ' compiler-options-popover' : '') +
            '" role="tooltip">' +
            '<div class="arrow"></div>' +
            '<h3 class="popover-header"></h3><div class="popover-body"></div>' +
            '</div>',
    });
};

Compiler.prototype.onRequestCompilation = function (editorId, treeId) {
    if (editorId === this.sourceEditorId || (treeId && treeId === this.sourceTreeId)) {
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
            enabled: this.settings.showMinimap && !options.embedded,
        },
        fontFamily: this.settings.editorsFFont,
        codeLensFontFamily: this.settings.editorsFFont,
        fontLigatures: this.settings.editorsFLigatures,
    });
};

var hexLike = /^(#?[$]|0x)([0-9a-fA-F]+)$/;
var hexLike2 = /^(#?)([0-9a-fA-F]+)H$/;
var decimalLike = /^(#?)(-?[0-9]+)$/;

function parseNumericValue(value) {
    var hexMatch = hexLike.exec(value) || hexLike2.exec(value);
    if (hexMatch) return bigInt(hexMatch[2], 16);

    var decMatch = decimalLike.exec(value);
    if (decMatch) return bigInt(decMatch[2]);

    return null;
}

function getNumericToolTip(value) {
    var numericValue = parseNumericValue(value);
    if (numericValue === null) return null;

    // Decimal representation.
    var result = numericValue.toString(10);

    // Hexadecimal representation.
    if (numericValue.isNegative()) {
        var masked = bigInt('ffffffffffffffff', 16).and(numericValue);
        result += ' = 0x' + masked.toString(16).toUpperCase();
    } else {
        result += ' = 0x' + numericValue.toString(16).toUpperCase();
    }

    // Printable ASCII character.
    if (numericValue.greaterOrEquals(0x20) && numericValue.lesserOrEquals(0x7e)) {
        var char = String.fromCharCode(numericValue.valueOf());
        result += " = '" + char + "'";
    }

    return result;
}

function getAsmInfo(opcode, instructionSet) {
    var cacheName = 'asm/' + (instructionSet ? instructionSet + '/' : '') + opcode;
    var cached = OpcodeCache.get(cacheName);
    if (cached) {
        if (cached.found) {
            return Promise.resolve(cached.data);
        }
        return Promise.reject(cached.data);
    }
    return new Promise(function (resolve, reject) {
        getAssemblyDocumentation({opcode: opcode, instructionSet: instructionSet})
            .then(function (response) {
                response.json().then(function (body) {
                    if (response.status === 200) {
                        OpcodeCache.set(cacheName, {found: true, data: body});
                        resolve(body);
                    } else {
                        OpcodeCache.set(cacheName, {found: false, data: body.error});
                        reject(body.error);
                    }
                });
            })
            .catch(function (error) {
                reject('Fetch error: ' + error);
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
    var hoverShowSource = this.settings.hoverShowSource === true;
    if (this.assembly) {
        var hoverAsm = this.assembly[e.target.position.lineNumber - 1];
        if (hoverShowSource && hoverAsm) {
            this.clearLinkedLines();
            // We check that we actually have something to show at this point!
            var sourceLine = -1;
            var sourceColBegin = -1;
            var sourceColEnd = -1;
            if (hoverAsm.source) {
                sourceLine = hoverAsm.source.line;
                if (hoverAsm.source.column) {
                    sourceColBegin = hoverAsm.source.column;
                    sourceColEnd = sourceColBegin;
                }

                var editorId = this.getEditorIdBySourcefile(hoverAsm.source);
                if (editorId) {
                    this.eventHub.emit('editorLinkLine', editorId, sourceLine, sourceColBegin, sourceColEnd, false);

                    this.eventHub.emit(
                        'panesLinkLine',
                        this.id,
                        sourceLine,
                        sourceColBegin,
                        sourceColEnd,
                        false,
                        this.getPaneName(),
                        editorId
                    );
                }
            }
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
            // c.f. https://github.com/compiler-explorer/compiler-explorer/issues/434
            var lineContent = this.outputEditor.getModel().getLineContent(e.target.position.lineNumber);
            if (lineContent[currentWord.startColumn - 2] === '-') {
                word = '-' + word;
                startColumn -= 1;
            }
        }
        currentWord.range = new monaco.Range(
            e.target.position.lineNumber,
            Math.max(startColumn, 1),
            e.target.position.lineNumber,
            currentWord.endColumn
        );
        var numericToolTip = getNumericToolTip(word);
        if (numericToolTip) {
            this.decorations.numericToolTip = {
                range: currentWord.range,
                options: {
                    isWholeLine: false,
                    hoverMessage: [
                        {
                            // We use double `` as numericToolTip may include a single ` character.
                            value: '``' + numericToolTip + '``',
                        },
                    ],
                },
            };
            this.updateDecorations();
        }
        var hoverShowAsmDoc = this.settings.hoverShowAsmDoc === true;
        if (hoverShowAsmDoc && this.compiler && this.compiler.supportsAsmDocs && this.isWordAsmKeyword(currentWord)) {
            getAsmInfo(currentWord.word, this.compiler.instructionSet).then(
                _.bind(function (response) {
                    if (!response) return;
                    this.decorations.asmToolTip = {
                        range: currentWord.range,
                        options: {
                            isWholeLine: false,
                            hoverMessage: [
                                {
                                    value: response.tooltip + '\n\nMore information available in the context menu.',
                                    isTrusted: true,
                                },
                            ],
                        },
                    };
                    this.updateDecorations();
                }, this)
            );
        }
    }
};

Compiler.prototype.getLineTokens = function (line) {
    var model = this.outputEditor.getModel();
    if (!model || line > model.getLineCount()) return [];
    var flavour = model.getLanguageId();
    var tokens = monaco.editor.tokenize(model.getLineContent(line), flavour);
    return tokens.length > 0 ? tokens[0] : [];
};

Compiler.prototype.isWordAsmKeyword = function (word) {
    return _.some(this.getLineTokens(word.range.startLineNumber), function (t) {
        return t.offset + 1 === word.startColumn && t.type === 'keyword.asm';
    });
};

Compiler.prototype.onAsmToolTip = function (ed) {
    ga.proxy('send', {
        hitType: 'event',
        eventCategory: 'OpenModalPane',
        eventAction: 'AsmDocs',
    });
    var pos = ed.getPosition();
    if (!pos || !ed.getModel()) return;
    var word = ed.getModel().getWordAtPosition(pos);
    if (!word || !word.word) return;
    var opcode = word.word.toUpperCase();

    function newGitHubIssueUrl() {
        return (
            'https://github.com/compiler-explorer/compiler-explorer/issues/new?title=' +
            encodeURIComponent('[BUG] Problem with ' + opcode + ' opcode')
        );
    }

    function appendInfo(url) {
        return (
            '<br><br>For more information, visit <a href="' +
            url +
            '" target="_blank" rel="noopener noreferrer">the ' +
            opcode +
            ' documentation <sup><small class="fas fa-external-link-alt opens-new-window"' +
            ' title="Opens in a new window"></small></sup></a>.' +
            '<br>If the documentation for this opcode is wrong or broken in some way, ' +
            'please feel free to <a href="' +
            newGitHubIssueUrl() +
            '" target="_blank" rel="noopener noreferrer">' +
            'open an issue on GitHub <sup><small class="fas fa-external-link-alt opens-new-window" ' +
            'title="Opens in a new window"></small></sup></a>.'
        );
    }

    getAsmInfo(word.word, this.compiler.instructionSet).then(
        _.bind(function (asmHelp) {
            if (asmHelp) {
                this.alertSystem.alert(opcode + ' help', asmHelp.html + appendInfo(asmHelp.url), function () {
                    ed.focus();
                    ed.setPosition(pos);
                });
            } else {
                this.alertSystem.notify('This token was not found in the documentation. Sorry!', {
                    group: 'notokenindocs',
                    alertClass: 'notification-error',
                    dismissTime: 5000,
                });
            }
        }, this),
        _.bind(function (rejection) {
            this.alertSystem.notify(
                'There was an error fetching the documentation for this opcode (' + rejection + ').',
                {
                    group: 'notokenindocs',
                    alertClass: 'notification-error',
                    dismissTime: 5000,
                }
            );
        }, this)
    );
};

Compiler.prototype.handleCompilationStatus = function (status) {
    CompilerService.handleCompilationStatus(this.statusLabel, this.statusIcon, status);
};

Compiler.prototype.onLanguageChange = function (editorId, newLangId, treeId) {
    if (
        (this.sourceEditorId && this.sourceEditorId === editorId) ||
        (this.sourceTreeId && this.sourceTreeId === treeId)
    ) {
        var oldLangId = this.currentLangId;
        this.currentLangId = newLangId;
        // Store the current selected stuff to come back to it later in the same session (Not state stored!)
        this.infoByLang[oldLangId] = {
            compiler: this.compiler && this.compiler.id ? this.compiler.id : options.defaultCompiler[oldLangId],
            options: this.options,
        };
        var info = this.infoByLang[this.currentLangId] || {};
        this.deferCompiles = true;
        this.initLangAndCompiler({lang: newLangId, compiler: info.compiler});
        this.updateCompilersSelector(info);
        this.saveState();
        this.updateCompilerUI();
        this.setAssembly(fakeAsm(''));
        // this is a workaround to delay compilation further until the Editor sends a compile request
        this.needsCompile = false;

        this.undefer();
        this.sendCompiler();
    }
};

Compiler.prototype.getCurrentLangCompilers = function () {
    return this.compilerService.getCompilersForLang(this.currentLangId);
};

Compiler.prototype.updateCompilersSelector = function (info) {
    this.compilerPicker.update(this.currentLangId, this.compiler ? this.compiler.id : null);
    this.options = info.options || '';
    this.optionsField.val(this.options);
};

module.exports = {
    Compiler: Compiler,
};
