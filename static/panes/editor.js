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
var colour = require('../colour');
var loadSaveLib = require('../load-save');
var FontScale = require('../fontscale').FontScale;
var Components = require('../components');
var monaco = require('monaco-editor');
var options = require('../options').options;
var Alert = require('../alert').Alert;
var ga = require('../analytics').ga;
var monacoVim = require('monaco-vim');
var monacoConfig = require('../monaco-config');
var TomSelect = require('tom-select');
var Settings = require('../settings');
var utils = require('../utils');
require('../formatter-registry');
require('../modes/_all');

var loadSave = new loadSaveLib.LoadSave();
var languages = options.languages;

// eslint-disable-next-line max-statements
function Editor(hub, state, container) {
    this.id = state.id || hub.nextEditorId();
    this.container = container;
    this.domRoot = container.getElement();
    this.domRoot.html($('#codeEditor').html());
    this.hub = hub;
    this.eventHub = hub.createEventHub();
    // Should probably be its own function somewhere
    this.settings = Settings.getStoredSettings();
    this.ourCompilers = {};
    this.ourExecutors = {};
    this.httpRoot = window.httpRoot;
    this.widgetsByCompiler = {};
    this.asmByCompiler = {};
    this.defaultFileByCompiler = {};
    this.busyCompilers = {};
    this.colours = [];
    this.treeCompilers = {};

    this.decorations = {};
    this.prevDecorations = [];
    this.extraDecorations = [];

    this.fadeTimeoutId = -1;

    this.editorSourceByLang = {};
    this.alertSystem = new Alert();
    this.alertSystem.prefixMessage = 'Editor #' + this.id + ': ';

    this.filename = state.filename || false;

    this.awaitingInitialResults = false;
    this.selection = state.selection;

    this.revealJumpStack = [];

    this.langKeys = _.keys(languages);
    this.initLanguage(state);

    var root = this.domRoot.find('.monaco-placeholder');
    var legacyReadOnly = state.options && !!state.options.readOnly;
    this.editor = monaco.editor.create(root[0], monacoConfig.extendConfig({
        language: this.currentLanguage.monaco,
        readOnly: !!options.readOnly || legacyReadOnly ||
            (window.compilerExplorerOptions && window.compilerExplorerOptions.mobileViewer),
        glyphMargin: !options.embedded,
    }, this.settings));
    this.editor.getModel().setEOL(monaco.editor.EndOfLineSequence.LF);

    if (state.source !== undefined) {
        this.setSource(state.source);
    } else {
        this.updateEditorCode();
    }

    var startFolded = /^[/*#;]+\s*setup.*/;
    if (state.source && state.source.match(startFolded)) {
        // With reference to https://github.com/Microsoft/monaco-editor/issues/115
        // I tried that and it didn't work, but a delay of 500 seems to "be enough".
        // FIXME: Currently not working - No folding is performed
        setTimeout(_.bind(function () {
            this.editor.setSelection(new monaco.Selection(1, 1, 1, 1));
            this.editor.focus();
            this.editor.getAction('editor.fold').run();
            //this.editor.clearSelection();
        }, this), 500);
    }

    this.initEditorActions();
    this.initButtons(state);
    this.initCallbacks();

    if (this.settings.useVim) {
        this.enableVim();
    }

    var usableLanguages = _.filter(languages, function (language) {
        return hub.compilerService.compilersByLang[language.id];
    });

    this.selectize = new TomSelect(this.languageBtn, {
        sortField: 'name',
        valueField: 'id',
        labelField: 'name',
        searchField: ['name'],
        placeholder: '🔍 Select a language...',
        options: _.map(usableLanguages, _.identity),
        items: [this.currentLanguage.id],
        dropdownParent: 'body',
        plugins: ['dropdown_input'],
        onChange: _.bind(this.onLanguageChange, this),
        closeAfterSelect: true,
    });


    // We suppress posting changes until the user has stopped typing by:
    // * Using _.debounce() to run emitChange on any key event or change
    //   only after a delay.
    // * Only actually triggering a change if the document text has changed from
    //   the previous emitted.
    this.lastChangeEmitted = null;
    this.onSettingsChange(this.settings);
    // this.editor.on("keydown", _.bind(function () {
    //     // Not strictly a change; but this suppresses changes until some time
    //     // after the last key down (be it an actual change or a just a cursor
    //     // movement etc).
    //     this.debouncedEmitChange();
    // }, this));

    this.updateTitle();
    this.updateState();
    ga.proxy('send', {
        hitType: 'event',
        eventCategory: 'OpenViewPane',
        eventAction: 'Editor',
    });
    ga.proxy('send', {
        hitType: 'event',
        eventCategory: 'LanguageChange',
        eventAction: this.currentLanguage.id,
    });
}

Editor.prototype.onMotd = function (motd) {
    this.extraDecorations = motd.decorations;
    this.updateExtraDecorations();
};

Editor.prototype.updateExtraDecorations = function () {
    var decorationsDirty = false;
    _.each(this.extraDecorations, _.bind(function (decoration) {
        if (decoration.filter && decoration.filter.indexOf(this.currentLanguage.name.toLowerCase()) < 0) return;
        var match = this.editor.getModel().findNextMatch(decoration.regex, {
            column: 1,
            lineNumber: 1,
        }, true, true, null, false);
        if (match !== this.decorations[decoration.name]) {
            decorationsDirty = true;
            this.decorations[decoration.name] = match ? [{range: match.range, options: decoration.decoration}] : null;
        }
    }, this));
    if (decorationsDirty)
        this.updateDecorations();
};

// If compilerId is undefined, every compiler will be pinged
Editor.prototype.maybeEmitChange = function (force, compilerId) {
    var source = this.getSource();
    if (!force && source === this.lastChangeEmitted) return;

    this.updateExtraDecorations();

    this.lastChangeEmitted = source;
    this.eventHub.emit('editorChange', this.id, this.lastChangeEmitted, this.currentLanguage.id, compilerId);
};

Editor.prototype.updateState = function () {
    var state = {
        id: this.id,
        source: this.getSource(),
        lang: this.currentLanguage.id,
        selection: this.selection,
        filename: this.filename,
    };
    this.fontScale.addState(state);
    this.container.setState(state);

    this.updateButtons();
};

Editor.prototype.setSource = function (newSource) {
    this.updateSource(newSource);

    if (window.compilerExplorerOptions.mobileViewer) {
        $(this.domRoot.find('.monaco-placeholder textarea')).hide();
    }
};

Editor.prototype.onNewSource = function (editorId, newSource) {
    if (this.id === editorId) {
        this.setSource(newSource);
    }
};

Editor.prototype.getSource = function () {
    return this.editor.getModel().getValue();
};

Editor.prototype.initLanguage = function (state) {
    this.currentLanguage = languages[this.langKeys[0]];
    this.waitingForLanguage = state.source && !state.lang;
    if (languages[this.settings.defaultLanguage]) {
        this.currentLanguage = languages[this.settings.defaultLanguage];
    }
    if (languages[state.lang]) {
        this.currentLanguage = languages[state.lang];
    } else if (this.settings.newEditorLastLang && languages[this.hub.lastOpenedLangId]) {
        this.currentLanguage = languages[this.hub.lastOpenedLangId];
    }
};

Editor.prototype.initCallbacks = function () {
    this.fontScale.on('change', _.bind(this.updateState, this));

    this.container.on('resize', this.resize, this);
    this.container.on('shown', this.resize, this);
    this.container.on('open', _.bind(function () {
        this.eventHub.emit('editorOpen', this.id, this);
    }, this));
    this.container.on('destroy', this.close, this);
    this.container.layoutManager.on('initialised', function () {
        // Once initialized, let everyone know what text we have.
        this.maybeEmitChange();
        // And maybe ask for a compilation (Will hit the cache most of the time)
        this.requestCompilation();
    }, this);

    this.eventHub.on('treeCompilerEditorIncludeChange', this.onTreeCompilerEditorIncludeChange, this);
    this.eventHub.on('treeCompilerEditorExcludeChange', this.onTreeCompilerEditorExcludeChange, this);
    this.eventHub.on('coloursForEditor', this.onColoursForEditor, this);
    this.eventHub.on('compilerOpen', this.onCompilerOpen, this);
    this.eventHub.on('executorOpen', this.onExecutorOpen, this);
    this.eventHub.on('compilerClose', this.onCompilerClose, this);
    this.eventHub.on('compiling', this.onCompiling, this);
    this.eventHub.on('compileResult', this.onCompileResponse, this);
    this.eventHub.on('selectLine', this.onSelectLine, this);
    this.eventHub.on('editorSetDecoration', this.onEditorSetDecoration, this);
    this.eventHub.on('editorLinkLine', this.onEditorLinkLine, this);
    this.eventHub.on('settingsChange', this.onSettingsChange, this);
    this.eventHub.on('conformanceViewOpen', this.onConformanceViewOpen, this);
    this.eventHub.on('conformanceViewClose', this.onConformanceViewClose, this);
    this.eventHub.on('resize', this.resize, this);
    this.eventHub.on('newSource', this.onNewSource, this);
    this.eventHub.on('motd', this.onMotd, this);
    this.eventHub.on('findEditors', this.sendEditor, this);
    this.eventHub.emit('requestMotd');

    this.editor.getModel().onDidChangeContent(_.bind(function () {
        this.debouncedEmitChange();
        this.updateState();
    }, this));

    this.mouseMoveThrottledFunction = _.throttle(_.bind(this.onMouseMove, this), 50);

    this.editor.onMouseMove(_.bind(function (e) {
        this.mouseMoveThrottledFunction(e);
    }, this));

    if (window.compilerExplorerOptions.mobileViewer) {
        // workaround for issue with contextmenu not going away when tapping somewhere else on the screen
        this.editor.onDidChangeCursorSelection(_.bind(function () {
            var contextmenu = $('div.context-view.monaco-menu-container');
            if (contextmenu.css('display') !== 'none') {
                contextmenu.hide();
            }
        }, this));
    }

    this.cursorSelectionThrottledFunction =
        _.throttle(_.bind(this.onDidChangeCursorSelection, this), 500);
    this.editor.onDidChangeCursorSelection(_.bind(function (e) {
        this.cursorSelectionThrottledFunction(e);
    }, this));

    this.editor.onDidFocusEditorText(_.bind(this.onDidFocusEditorText, this));
    this.editor.onDidBlurEditorText(_.bind(this.onDidBlurEditorText, this));
    this.editor.onDidChangeCursorPosition(_.bind(this.onDidChangeCursorPosition,this));

    this.eventHub.on('initialised', this.maybeEmitChange, this);

    $(document).on('keyup.editable', _.bind(function (e) {
        if (e.target === this.domRoot.find('.monaco-placeholder .inputarea')[0]) {
            if (e.which === 27) {
                this.onEscapeKey(e);
            } else if (e.which === 45) {
                this.onInsertKey(e);
            }
        }
    }, this));
};

Editor.prototype.sendEditor = function () {
    this.eventHub.emit('editorOpen', this.id, this);
};

Editor.prototype.onMouseMove = function (e) {
    if (e !== null && e.target !== null && this.settings.hoverShowSource && e.target.position !== null) {
        var pos = e.target.position;
        this.tryPanesLinkLine(pos.lineNumber, pos.column, false);
    }
};

Editor.prototype.onDidChangeCursorSelection = function (e) {
    if (this.awaitingInitialResults) {
        this.selection = e.selection;
        this.updateState();
    }
};

Editor.prototype.onDidChangeCursorPosition = function (e) {
    if (e.position) {
        this.currentCursorPosition.text('(' + e.position.lineNumber + ', ' + e.position.column + ')');
    }
};

Editor.prototype.onDidFocusEditorText = function () {
    this.currentCursorPosition.show();
};

Editor.prototype.onDidBlurEditorText = function () {
    this.currentCursorPosition.text('');
    this.currentCursorPosition.hide();
};

Editor.prototype.onEscapeKey = function () {
    if (this.editor.vimInUse) {
        var currentState = monacoVim.VimMode.Vim.maybeInitVimState_(this.vimMode);
        if (currentState.insertMode) {
            monacoVim.VimMode.Vim.exitInsertMode(this.vimMode);
        } else if (currentState.visualMode) {
            monacoVim.VimMode.Vim.exitVisualMode(this.vimMode, false);
        }
    }
};

Editor.prototype.onInsertKey = function (event) {
    if (this.editor.vimInUse) {
        var currentState = monacoVim.VimMode.Vim.maybeInitVimState_(this.vimMode);
        if (!currentState.insertMode) {
            var insertEvent = {
                preventDefault: event.preventDefault,
                stopPropagation: event.stopPropagation,
                browserEvent: {
                    key: 'i',
                    defaultPrevented: false,
                },
                keyCode: 39,
            };
            this.vimMode.handleKeyDown(insertEvent);
        }
    }
};

Editor.prototype.enableVim = function () {
    this.vimMode = monacoVim.initVimMode(this.editor, this.domRoot.find('#v-status')[0]);
    this.vimFlag.prop('class', 'btn btn-info');
    this.editor.vimInUse = true;
};

Editor.prototype.disableVim = function () {
    this.vimMode.dispose();
    this.domRoot.find('#v-status').html('');
    this.vimFlag.prop('class', 'btn btn-light');
    this.editor.vimInUse = false;
};

Editor.prototype.initButtons = function (state) {
    this.fontScale = new FontScale(this.domRoot, state, this.editor);
    this.languageBtn = this.domRoot.find('.change-language');
    // Ensure that the button is disabled if we don't have nothing to select
    // Note that is might be disabled for other reasons beforehand
    if (this.langKeys.length <= 1) {
        this.languageBtn.prop('disabled', true);
    }
    this.topBar = this.domRoot.find('.top-bar');
    this.hideable = this.domRoot.find('.hideable');

    this.loadSaveButton = this.domRoot.find('.load-save');
    var paneAdderDropdown = this.domRoot.find('.add-pane');
    var addCompilerButton = this.domRoot.find('.btn.add-compiler');
    this.addExecutorButton = this.domRoot.find('.btn.add-executor');
    this.conformanceViewerButton = this.domRoot.find('.btn.conformance');
    var addEditorButton = this.domRoot.find('.btn.add-editor');
    var toggleVimButton = this.domRoot.find('#vim-flag');
    this.vimFlag = this.domRoot.find('#vim-flag');
    toggleVimButton.on('click', _.bind(function () {
        if (this.editor.vimInUse) {
            this.disableVim();
        } else {
            this.enableVim();
        }
    }, this));

    // NB a new compilerConfig needs to be created every time; else the state is shared
    // between all compilers created this way. That leads to some nasty-to-find state
    // bugs e.g. https://github.com/compiler-explorer/compiler-explorer/issues/225
    var getCompilerConfig = _.bind(function () {
        return Components.getCompiler(this.id, this.currentLanguage.id);
    }, this);

    var getExecutorConfig = _.bind(function () {
        return Components.getExecutor(this.id, this.currentLanguage.id);
    }, this);

    var getConformanceConfig = _.bind(function () {
        return Components.getConformanceView(this.id, this.getSource(), this.currentLanguage.id);
    }, this);

    var getEditorConfig = _.bind(function () {
        return Components.getEditor();
    }, this);

    var addPaneOpener = _.bind(function (dragSource, dragConfig) {
        this.container.layoutManager
            .createDragSource(dragSource, dragConfig)
            ._dragListener.on('dragStart', function () {
                paneAdderDropdown.dropdown('toggle');
            });

        dragSource.on('click', _.bind(function () {
            var insertPoint = this.hub.findParentRowOrColumn(this.container) ||
                this.container.layoutManager.root.contentItems[0];
            insertPoint.addChild(dragConfig);
        }, this));
    }, this);

    addPaneOpener(addCompilerButton, getCompilerConfig);
    addPaneOpener(this.addExecutorButton, getExecutorConfig);
    addPaneOpener(this.conformanceViewerButton, getConformanceConfig);
    addPaneOpener(addEditorButton, getEditorConfig);

    this.initLoadSaver();
    $(this.domRoot).on('keydown', _.bind(function (event) {
        if ((event.ctrlKey || event.metaKey) && String.fromCharCode(event.which).toLowerCase() === 's') {
            event.preventDefault();
            if (this.settings.enableCtrlStree && this.hub.hasTree()) {
                var trees = this.hub.trees;
                // todo: change when multiple trees are used
                if (trees && trees.length > 0) {
                    trees[0].multifileService.includeByEditorId(this.id).then(_.bind(function () {
                        trees[0].refresh();
                    }, this));
                }
            } else if (this.settings.enableCtrlS) {
                loadSave.setMinimalOptions(this.getSource(), this.currentLanguage);
                if (!loadSave.onSaveToFile(this.id)) {
                    this.showLoadSaver();
                }
            } else {
                this.eventHub.emit('displaySharingPopover');
            }
        }
    }, this));

    if (options.thirdPartyIntegrationEnabled) {
        this.cppInsightsButton = this.domRoot.find('.open-in-cppinsights');
        this.cppInsightsButton.on('mousedown', _.bind(function () {
            this.updateOpenInCppInsights();
        }, this));

        this.quickBenchButton = this.domRoot.find('.open-in-quickbench');
        this.quickBenchButton.on('mousedown', _.bind(function () {
            this.updateOpenInQuickBench();
        }, this));
    }

    this.currentCursorPosition = this.domRoot.find('.currentCursorPosition');
    this.currentCursorPosition.hide();
};

Editor.prototype.updateButtons = function () {
    if (options.thirdPartyIntegrationEnabled) {
        if (this.currentLanguage.id === 'c++') {
            this.cppInsightsButton.show();
            this.quickBenchButton.show();
        } else {
            this.cppInsightsButton.hide();
            this.quickBenchButton.hide();
        }
    }

    this.addExecutorButton.prop('disabled', !this.currentLanguage.supportsExecute);
};

Editor.prototype.b64UTFEncode = function (str) {
    return btoa(encodeURIComponent(str).replace(/%([0-9A-F]{2})/g, function (match, v) {
        return String.fromCharCode(parseInt(v, 16));
    }));
};

Editor.prototype.asciiEncodeJsonText = function (json) {
    return json.replace(/[\u007F-\uFFFF]/g, function (chr) {
        // json unicode escapes must always be 4 characters long, so pad with leading zeros
        return '\\u' + ('0000' + chr.charCodeAt(0).toString(16)).substr(-4);
    });
};

Editor.prototype.getCompilerStates = function () {
    var states = [];

    _.each(this.ourCompilers, _.bind(function (val, compilerIdStr) {
        var compilerId = parseInt(compilerIdStr);

        var glCompiler = _.find(this.container.layoutManager.root.getComponentsByName('compiler'), function (c) {
            return c.id === compilerId;
        });

        if (glCompiler) {
            var state = glCompiler.currentState();
            states.push(state);
        }
    }, this));

    return states;
};

Editor.prototype.updateOpenInCppInsights = function () {
    if (options.thirdPartyIntegrationEnabled) {
        var cppStd = 'cpp2a';

        var compilers = this.getCompilerStates();
        _.each(compilers, _.bind(function (compiler) {
            if ((compiler.options.indexOf('-std=c++11') !== -1) ||
                (compiler.options.indexOf('-std=gnu++11') !== -1)) {
                cppStd = 'cpp11';
            } else if ((compiler.options.indexOf('-std=c++14') !== -1) ||
                (compiler.options.indexOf('-std=gnu++14') !== -1)) {
                cppStd = 'cpp14';
            } else if ((compiler.options.indexOf('-std=c++17') !== -1) ||
                (compiler.options.indexOf('-std=gnu++17') !== -1)) {
                cppStd = 'cpp17';
            } else if ((compiler.options.indexOf('-std=c++2a') !== -1) ||
                (compiler.options.indexOf('-std=gnu++2a') !== -1)) {
                cppStd = 'cpp2a';
            } else if (compiler.options.indexOf('-std=c++98') !== -1) {
                cppStd = 'cpp98';
            }
        }, this));

        var link = 'https://cppinsights.io/lnk?code='
            + this.b64UTFEncode(this.getSource()) + '&std=' + cppStd + '&rev=1.0';

        this.cppInsightsButton.attr('href', link);
    }
};

Editor.prototype.cleanupSemVer = function (semver) {
    if (semver) {
        var semverStr = semver.toString();
        if ((semverStr !== '') && (semverStr.indexOf('(') === -1)) {
            var vercomps = semverStr.split('.');
            return vercomps[0] + '.' + (vercomps[1] ? vercomps[1] : '0');
        }
    }

    return false;
};

Editor.prototype.updateOpenInQuickBench = function () {
    if (options.thirdPartyIntegrationEnabled) {
        var quickBenchState = {
            text: this.getSource(),
        };

        var compilers = this.getCompilerStates();

        _.each(compilers, _.bind(function (compiler) {
            var knownCompiler = false;

            var compilerExtInfo = this.hub.compilerService.findCompiler(this.currentLanguage.id, compiler.compiler);
            var semver = this.cleanupSemVer(compilerExtInfo.semver);
            var groupOrName =
                compilerExtInfo.baseName ? compilerExtInfo.baseName :
                    compilerExtInfo.groupName ? compilerExtInfo.groupName : compilerExtInfo.name;

            if (semver && groupOrName) {
                groupOrName = groupOrName.toLowerCase();
                if (groupOrName.indexOf('gcc') !== -1) {
                    quickBenchState.compiler = 'gcc-' + semver;
                    knownCompiler = true;
                } else if (groupOrName.indexOf('clang') !== -1) {
                    quickBenchState.compiler = 'clang-' + semver;
                    knownCompiler = true;
                }
            }

            if (knownCompiler) {
                var match = compiler.options.match(/-(O([0-3sg]|fast))/);
                if (match !== null) {
                    if (match[2] === 'fast') {
                        quickBenchState.optim = 'F';
                    } else {
                        quickBenchState.optim = match[2].toUpperCase();
                    }
                }

                if ((compiler.options.indexOf('-std=c++11') !== -1) ||
                    (compiler.options.indexOf('-std=gnu++11') !== -1)) {
                    quickBenchState.cppVersion = '11';
                } else if ((compiler.options.indexOf('-std=c++14') !== -1) ||
                    (compiler.options.indexOf('-std=gnu++14') !== -1)) {
                    quickBenchState.cppVersion = '14';
                } else if ((compiler.options.indexOf('-std=c++17') !== -1) ||
                    (compiler.options.indexOf('-std=gnu++17') !== -1)) {
                    quickBenchState.cppVersion = '17';
                } else if ((compiler.options.indexOf('-std=c++2a') !== -1) ||
                    (compiler.options.indexOf('-std=gnu++2a') !== -1)) {
                    quickBenchState.cppVersion = '20';
                }

                if ((compiler.options.indexOf('-stdlib=libc++') !== -1)) {
                    quickBenchState.lib = 'llvm';
                }
            }
        }, this));

        var link = 'http://quick-bench.com/#' + btoa(this.asciiEncodeJsonText(JSON.stringify(quickBenchState)));
        this.quickBenchButton.attr('href', link);
    }
};

Editor.prototype.changeLanguage = function (newLang) {
    if (newLang === 'cmake') {
        this.selectize.addOption(languages.cmake);
    }
    this.selectize.setValue(newLang);
};

Editor.prototype.clearLinkedLine = function () {
    this.decorations.linkedCode = [];
    this.updateDecorations();
};

Editor.prototype.tryPanesLinkLine = function (thisLineNumber, column, reveal) {
    var selectedToken = this.getTokenSpan(thisLineNumber, column);
    _.each(this.asmByCompiler, _.bind(function (asms, compilerId) {
        this.eventHub.emit('panesLinkLine', compilerId, thisLineNumber,
            selectedToken.colBegin, selectedToken.colEnd, reveal, undefined, this.id);
    }, this));
};

Editor.prototype.requestCompilation = function () {
    this.eventHub.emit('requestCompilation', this.id);

    _.each(this.hub.trees, _.bind(function (tree) {
        if (tree.multifileService.isEditorPartOfProject(this.id)) {
            this.eventHub.emit('requestCompilation', this.id, tree.id);
        }
    }, this));
};

Editor.prototype.initEditorActions = function () {
    this.editor.addAction({
        id: 'compile',
        label: 'Compile',
        keybindings: [monaco.KeyMod.CtrlCmd | monaco.KeyCode.Enter],
        keybindingContext: null,
        contextMenuGroupId: 'navigation',
        contextMenuOrder: 1.5,
        run: _.bind(function () {
            // This change request is mostly superfluous
            this.maybeEmitChange();
            this.requestCompilation();
        }, this),
    });

    this.revealJumpStackHasElementsCtxKey = this.editor.createContextKey('hasRevealJumpStackElements', false);

    this.editor.addAction({
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

    this.editor.addAction({
        id: 'toggleCompileOnChange',
        label: 'Toggle compile on change',
        keybindings: [monaco.KeyMod.CtrlCmd | monaco.KeyMod.Shift | monaco.KeyCode.Enter],
        keybindingContext: null,
        run: _.bind(function () {
            this.eventHub.emit('modifySettings', {
                compileOnChange: !this.settings.compileOnChange,
            });
            this.alertSystem
                .notify('Compile on change has been toggled ' + (this.settings.compileOnChange ? 'ON' : 'OFF'), {
                    group: 'togglecompile',
                    alertClass: this.settings.compileOnChange ? 'notification-on' : 'notification-off',
                    dismissTime: 3000,
                });
        }, this),
    });

    this.editor.addAction({
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

    this.editor.addAction({
        id: 'viewasm',
        label: 'Reveal linked code',
        keybindings: [monaco.KeyMod.CtrlCmd | monaco.KeyCode.F10],
        keybindingContext: null,
        contextMenuGroupId: 'navigation',
        contextMenuOrder: 1.5,
        run: _.bind(function (ed) {
            var pos = ed.getPosition();
            this.tryPanesLinkLine(pos.lineNumber, pos.column, true);
        }, this),
    });

    this.isCpp = this.editor.createContextKey('isCpp', true);
    this.isCpp.set(this.currentLanguage.id === 'c++');

    this.isClean = this.editor.createContextKey('isClean', true);
    this.isClean.set(this.currentLanguage.id === 'clean');

    this.editor.addAction({
        id: 'cpprefsearch',
        label: 'Search on Cppreference',
        keybindings: [monaco.KeyMod.CtrlCmd | monaco.KeyCode.F8],
        keybindingContext: null,
        contextMenuGroupId: 'help',
        contextMenuOrder: 1.5,
        precondition: 'isCpp',
        run: _.bind(this.searchOnCppreference, this),
    });

    this.editor.addAction({
        id: 'clooglesearch',
        label: 'Search on Cloogle',
        keybindings: [monaco.KeyMod.CtrlCmd | monaco.KeyCode.F8],
        keybindingContext: null,
        contextMenuGroupId: 'help',
        contextMenuOrder: 1.5,
        precondition: 'isClean',
        run: _.bind(this.searchOnCloogle, this),
    });

    this.editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.F9, _.bind(function () {
        this.editor.getAction('editor.action.formatDocument').run();
    }, this));

    this.editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyD, _.bind(function () {
        this.editor.getAction('editor.action.duplicateSelection').run();
    }, this));
};

Editor.prototype.searchOnCppreference = function (ed) {
    var pos = ed.getPosition();
    var word = ed.getModel().getWordAtPosition(pos);
    if (!word || !word.word) return;
    var preferredLanguage = this.getPreferredLanguageTag();
    // This list comes from the footer of the page
    var cpprefLangs = ['ar', 'cs', 'de', 'en', 'es', 'fr', 'it', 'ja', 'ko', 'pl', 'pt', 'ru', 'tr', 'zh'];
    // If navigator.languages is supported, we could be a bit more clever and look for a match there too
    var langTag = 'en';
    if (cpprefLangs.indexOf(preferredLanguage) !== -1) {
        langTag = preferredLanguage;
    }
    var url = 'https://' +
        langTag +
        '.cppreference.com/mwiki/index.php?search=' +
        encodeURIComponent(word.word);
    window.open(url, '_blank', 'noopener');
};

Editor.prototype.searchOnCloogle = function (ed) {
    var pos = ed.getPosition();
    var word = ed.getModel().getWordAtPosition(pos);
    if (!word || !word.word) return;
    var url = 'https://cloogle.org/#' + encodeURIComponent(word.word);
    window.open(url, '_blank', 'noopener');
};

Editor.prototype.getPreferredLanguageTag = function () {
    var result = 'en';
    var lang = 'en';
    if (navigator) {
        if (navigator.languages && navigator.languages.length) {
            lang = navigator.languages[0];
        } else if (navigator.language) {
            lang = navigator.language;
        }
    }
    // navigator.language[s] is supposed to return strings, but hey, you never know
    if (lang !== result && _.isString(lang)) {
        var primaryLanguageSubtagIdx = lang.indexOf('-');
        result = lang.substr(0, primaryLanguageSubtagIdx).toLowerCase();
    }
    return result;
};

Editor.prototype.doesMatchEditor = function (otherSource) {
    return otherSource === this.getSource();
};

Editor.prototype.confirmOverwrite = function (yes) {
    this.alertSystem.ask('Changes were made to the code',
        'Changes were made to the code while it was being processed. Overwrite changes?',
        {yes: yes, no: null});
};

Editor.prototype.updateSource = function (newSource) {
    // Create something that looks like an edit operation for the whole text
    var operation = {
        range: this.editor.getModel().getFullModelRange(),
        forceMoveMarkers: true,
        text: newSource,
    };
    var nullFn = function () {
        return null;
    };
    var viewState = this.editor.saveViewState();
    // Add a undo stop so we don't go back further than expected
    this.editor.pushUndoStop();
    // Apply de edit. Note that we lose cursor position, but I've not found a better alternative yet
    this.editor.getModel().pushEditOperations(viewState.cursorState, [operation], nullFn);
    this.numberUsedLines();

    if (!this.awaitingInitialResults) {
        if (this.selection) {
            /*
             * this setTimeout is a really crap workaround to fix #2150
             * the TL;DR; is that we reach this point *before* GL has laid
             * out the window, so we have no height
             *
             * If we revealLinesInCenter at this point the editor "does the right thing"
             * and scrolls itself all the way to the line we requested.
             *
             * Unfortunately the editor thinks it is very small, so the "center"
             * is the first line, and when the editor does resize eventually things are off.
             *
             * The workaround is to just delay things "long enough"
             *
             * This is bad and I feel bad.
             */
            setTimeout(_.bind(function () {
                this.editor.setSelection(this.selection);
                this.editor.revealLinesInCenter(this.selection.startLineNumber,
                    this.selection.endLineNumber);
            }, this), 500);
        }
        this.awaitingInitialResults = true;
    }
};

Editor.prototype.formatCurrentText = function () {
    var previousSource = this.getSource();
    var lang = this.currentLanguage;

    if (!Object.prototype.hasOwnProperty.call(lang, 'formatter')) {
        return this.alertSystem.notify('This language does not support in-editor formatting' , {
            group: 'formatting',
            alertClass: 'notification-error',
        });
    }

    $.ajax({
        type: 'POST',
        url: window.location.origin + this.httpRoot + 'api/format/' + lang.formatter,
        dataType: 'json',  // Expected
        contentType: 'application/json',  // Sent
        data: JSON.stringify({
            source: previousSource,
            base: this.settings.formatBase,
        }),
        success: _.bind(function (result) {
            if (result.exit === 0) {
                if (this.doesMatchEditor(previousSource)) {
                    this.updateSource(result.answer);
                } else {
                    this.confirmOverwrite(_.bind(function () {
                        this.updateSource(result.answer);
                    }, this), null);
                }
            } else {
                // Ops, the formatter itself failed!
                this.alertSystem.notify('We encountered an error formatting your code: ' + result.answer, {
                    group: 'formatting',
                    alertClass: 'notification-error',
                });
            }
        }, this),
        error: _.bind(function (xhr, e_status, error) {
            // Hopefully we have not exploded!
            if (xhr.responseText) {
                try {
                    var res = JSON.parse(xhr.responseText);
                    error = res.answer || error;
                } catch (e) {
                    // continue regardless of error
                }
            }
            error = error || 'Unknown error';
            this.alertSystem.notify('We ran into some issues while formatting your code: ' + error, {
                group: 'formatting',
                alertClass: 'notification-error',
            });
        }, this),
        cache: true,
    });
};

Editor.prototype.resize = function () {
    var topBarHeight = utils.updateAndCalcTopBarHeight(this.domRoot, this.topBar, this.hideable);

    this.editor.layout({
        width: this.domRoot.width(),
        height: this.domRoot.height() - topBarHeight,
    });

    // Only update the options if needed
    if (this.settings.wordWrap) {
        this.editor.updateOptions({
            wordWrapColumn: this.editor.getLayoutInfo().viewportColumn,
        });
    }
};

Editor.prototype.onSettingsChange = function (newSettings) {
    var before = this.settings;
    var after = newSettings;
    this.settings = _.clone(newSettings);

    this.editor.updateOptions({
        autoIndent: this.settings.autoIndent ? 'advanced' : 'none',
        autoClosingBrackets: this.settings.autoCloseBrackets,
        useVim: this.settings.useVim,
        quickSuggestions: this.settings.showQuickSuggestions,
        contextmenu: this.settings.useCustomContextMenu,
        minimap: {
            enabled: this.settings.showMinimap && !options.embedded,
        },
        fontFamily: this.settings.editorsFFont,
        fontLigatures: this.settings.editorsFLigatures,
        wordWrap: this.settings.wordWrap ? 'bounded' : 'off',
        wordWrapColumn: this.editor.getLayoutInfo().viewportColumn, // Ensure the column count is up to date
    });

    // Unconditionally send editor changes. The compiler only compiles when needed
    this.debouncedEmitChange = _.debounce(_.bind(function () {
        this.maybeEmitChange();
    }, this), after.delayAfterChange);

    if (before.hoverShowSource && !after.hoverShowSource) {
        this.onEditorSetDecoration(this.id, -1, false);
    }

    if (after.useVim && !before.useVim) {
        this.enableVim();
    } else if (!after.useVim && before.useVim) {
        this.disableVim();
    }

    if (this.editor.getModel()) {
        this.editor.getModel().updateOptions({
            tabSize: this.settings.tabWidth,
            insertSpaces: this.settings.useSpaces,
        });
    }

    this.numberUsedLines();
};

Editor.prototype.numberUsedLines = function () {
    if (_.any(this.busyCompilers)) return;

    if (!this.settings.colouriseAsm) {
        this.updateColours([]);
        return;
    }

    if (this.hub.hasTree()) {
        return;
    }

    var result = {};
    // First, note all lines used.
    _.each(this.asmByCompiler, _.bind(function (asm, compilerId) {
        _.each(asm, _.bind(function (asmLine) {
            var foundInTrees = false;

            _.each(this.treeCompilers, _.bind(function (compilerIds, treeId) {
                if (compilerIds[compilerId]) {
                    var tree = this.hub.getTreeById(Number(treeId));
                    if (tree) {
                        var defaultFile = this.defaultFileByCompiler[compilerId];
                        foundInTrees = true;

                        if (asmLine.source && asmLine.source.line > 0) {
                            var sourcefilename = asmLine.source.file ? asmLine.source.file : defaultFile;
                            if (this.id === tree.multifileService.getEditorIdByFilename(sourcefilename)) {
                                result[asmLine.source.line - 1] = true;
                            }
                        }
                    }
                }
            }, this));

            if (!foundInTrees) {
                if (asmLine.source &&
                    (asmLine.source.file === null || asmLine.source.mainsource) &&
                    asmLine.source.line > 0) {
                    result[asmLine.source.line - 1] = true;
                }
            }
        }, this));
    }, this));
    // Now assign an ordinal to each used line.
    var ordinal = 0;
    _.each(result, function (v, k) {
        result[k] = ordinal++;
    });

    this.updateColours(result);
};

Editor.prototype.updateColours = function (colours) {
    this.colours = colour.applyColours(this.editor, colours, this.settings.colourScheme, this.colours);
    this.eventHub.emit('colours', this.id, colours, this.settings.colourScheme);
};

Editor.prototype.onCompilerOpen = function (compilerId, editorId, treeId) {
    if (editorId === this.id) {
        // On any compiler open, rebroadcast our state in case they need to know it.
        if (this.waitingForLanguage) {
            var glCompiler = _.find(this.container.layoutManager.root.getComponentsByName('compiler'), function (c) {
                return c.id === compilerId;
            });
            if (glCompiler) {
                var selected = _.find(options.compilers, function (compiler) {
                    return compiler.id === glCompiler.originalCompilerId;
                });
                if (selected) {
                    this.changeLanguage(selected.lang);
                }
            }
        }

        if (treeId > 0) {
            if (!this.treeCompilers[treeId]) {
                this.treeCompilers[treeId] = {};
            }
            this.treeCompilers[treeId][compilerId] = true;
        }
        this.ourCompilers[compilerId] = true;

        if (!treeId) {
            this.maybeEmitChange(true, compilerId);
        }
    }
};

Editor.prototype.onTreeCompilerEditorIncludeChange = function (treeId, editorId, compilerId) {
    if (this.id === editorId) {
        this.onCompilerOpen(compilerId, editorId, treeId);
    }
};

Editor.prototype.onTreeCompilerEditorExcludeChange = function (treeId, editorId, compilerId) {
    if (this.id === editorId) {
        this.onCompilerClose(compilerId);
    }
};

Editor.prototype.onColoursForEditor = function (editorId, colours, scheme) {
    if (this.id === editorId) {
        this.colours = colour.applyColours(this.editor, colours, scheme, this.colours);
    }
};

Editor.prototype.onExecutorOpen = function (executorId, editorId) {
    if (editorId === this.id) {
        this.maybeEmitChange(true);
        this.ourExecutors[executorId] = true;
    }
};

Editor.prototype.onCompilerClose = function (compilerId, unused, treeId) {
    if (this.treeCompilers[treeId]) {
        delete this.treeCompilers[treeId][compilerId];
    }

    if (this.ourCompilers[compilerId]) {
        monaco.editor.setModelMarkers(this.editor.getModel(), compilerId, []);
        delete this.widgetsByCompiler[compilerId];
        delete this.asmByCompiler[compilerId];
        delete this.busyCompilers[compilerId];
        delete this.ourCompilers[compilerId];
        delete this.defaultFileByCompiler[compilerId];
        this.numberUsedLines();
    }
};

Editor.prototype.onExecutorClose = function (id) {
    if (this.ourExecutors[id]) {
        delete this.ourExecutors[id];
    }
};

Editor.prototype.onCompiling = function (compilerId) {
    if (!this.ourCompilers[compilerId]) return;
    this.busyCompilers[compilerId] = true;
};

Editor.prototype.getAllOutputAndErrors = function (result) {
    var all = (result.stdout || []);
    if (result.buildsteps) {
        _.each(result.buildsteps, function (step) {
            all = all.concat(step.stdout);
        });

        _.each(result.buildsteps, function (step) {
            all = all.concat(step.stderr);
        });
    }
    all = all.concat(result.stderr || []);

    return all;
};

Editor.prototype.onCompileResponse = function (compilerId, compiler, result) {
    if (!this.ourCompilers[compilerId]) return;

    this.busyCompilers[compilerId] = false;
    var output = this.getAllOutputAndErrors(result);
    var widgets = _.compact(_.map(output, function (obj) {
        if (!obj.tag) return;

        var trees = this.hub.trees;
        if (trees && trees.length > 0) {
            if (obj.tag.file) {
                if (this.id !== trees[0].multifileService.getEditorIdByFilename(obj.tag.file)) {
                    return;
                }
            } else {
                if (this.id !== trees[0].multifileService.getMainSourceEditorId()) {
                    return;
                }
            }
        }

        var severity = 3; // error
        if (obj.tag.text.match(/^warning/)) severity = 2;
        if (obj.tag.text.match(/^note/)) severity = 1;
        var colBegin = 0;
        var colEnd = Infinity;
        if (obj.tag.column) {
            var span = this.getTokenSpan(obj.tag.line, obj.tag.column);
            colBegin = obj.tag.column;
            if (span.colEnd === obj.tag.column) colEnd = -1;
            else colEnd = span.colEnd + 1;
        }
        return {
            severity: severity,
            message: obj.tag.text,
            source: compiler.name + ' #' + compilerId,
            startLineNumber: obj.tag.line,
            startColumn: colBegin,
            endLineNumber: obj.tag.line,
            endColumn: colEnd,
        };
    }, this));
    monaco.editor.setModelMarkers(this.editor.getModel(), compilerId, widgets);
    this.decorations.tags = _.map(widgets, function (tag) {
        return {
            range: new monaco.Range(tag.startLineNumber, tag.startColumn, tag.startLineNumber + 1, 1),
            options: {
                isWholeLine: false,
                inlineClassName: 'error-code',
            },
        };
    }, this);
    this.updateDecorations();

    if (result.result && result.result.asm) {
        this.asmByCompiler[compilerId] = result.result.asm;
    } else {
        this.asmByCompiler[compilerId] = result.asm;
    }

    if (result.inputFilename) {
        this.defaultFileByCompiler[compilerId] = result.inputFilename;
    } else {
        this.defaultFileByCompiler[compilerId] = 'example' + this.currentLanguage.extensions[0];
    }

    this.numberUsedLines();
};

Editor.prototype.onSelectLine = function (id, lineNum) {
    if (Number(id) === this.id) {
        this.editor.setSelection({line: lineNum - 1, ch: 0}, {line: lineNum, ch: 0});
    }
};

// Returns a half-segment [a, b) for the token on the line lineNum
// that spans across the column.
// a - colStart points to the first character of the token
// b - colEnd points to the character immediately following the token
// e.g.: "this->callableMethod ( x, y );"
//              ^a   ^column  ^b
Editor.prototype.getTokenSpan = function (lineNum, column) {
    var model = this.editor.getModel();
    if (lineNum <= model.getLineCount()) {
        var line = model.getLineContent(lineNum);
        if (0 < column && column < line.length) {
            var tokens = monaco.editor.tokenize(line, model.getLanguageId());
            if (tokens.length > 0) {
                var lastOffset = 0;
                var lastWasString = false;
                for (var i = 0; i < tokens[0].length; ++i) {
                    // Treat all the contiguous string tokens as one,
                    // For example "hello \" world" is treated as one token
                    // instead of 3 "string.cpp", "string.escape.cpp", "string.cpp"
                    if (tokens[0][i].type.startsWith('string')) {
                        if (lastWasString) {
                            continue;
                        }
                        lastWasString = true;
                    } else {
                        lastWasString = false;
                    }
                    var currentOffset = tokens[0][i].offset;
                    if (column <= currentOffset) {
                        return {colBegin: lastOffset, colEnd: currentOffset};
                    } else {
                        lastOffset = currentOffset;
                    }
                }
                return {colBegin: lastOffset, colEnd: line.length};
            }
        }
    }
    return {colBegin: column, colEnd: column + 1};
};

Editor.prototype.pushRevealJump = function () {
    this.revealJumpStack.push(this.editor.saveViewState());
    this.revealJumpStackHasElementsCtxKey.set(true);
};

Editor.prototype.popAndRevealJump = function () {
    if (this.revealJumpStack.length > 0) {
        this.editor.restoreViewState(this.revealJumpStack.pop());
        this.revealJumpStackHasElementsCtxKey.set(this.revealJumpStack.length > 0);
    }
};

Editor.prototype.onEditorLinkLine = function (editorId, lineNum, columnBegin, columnEnd, reveal) {
    if (Number(editorId) === this.id) {
        if (reveal && lineNum) {
            this.pushRevealJump();
            this.editor.revealLineInCenter(lineNum);
        }
        this.decorations.linkedCode = lineNum === -1 || !lineNum ? [] : [{
            range: new monaco.Range(lineNum, 1, lineNum, 1),
            options: {
                isWholeLine: true,
                linesDecorationsClassName: 'linked-code-decoration-margin',
                className: 'linked-code-decoration-line',
            },
        }];

        if (lineNum > 0 && columnBegin !== -1) {
            var lastTokenSpan = this.getTokenSpan(lineNum, columnEnd);
            this.decorations.linkedCode.push({
                range: new monaco.Range(lineNum, columnBegin, lineNum, lastTokenSpan.colEnd + 1),
                options: {
                    isWholeLine: false,
                    inlineClassName: 'linked-code-decoration-column',
                },
            });
        }

        if (this.fadeTimeoutId !== -1) {
            clearTimeout(this.fadeTimeoutId);
        }
        this.fadeTimeoutId = setTimeout(_.bind(function () {
            this.clearLinkedLine();
            this.fadeTimeoutId = -1;
        }, this), 5000);

        this.updateDecorations();
    }
};

Editor.prototype.onEditorSetDecoration = function (id, lineNum, reveal) {
    if (Number(id) === this.id) {
        if (reveal && lineNum) {
            this.pushRevealJump();
            this.editor.revealLineInCenter(lineNum);
        }
        this.decorations.linkedCode = lineNum === -1 || !lineNum ? [] : [{
            range: new monaco.Range(lineNum, 1, lineNum, 1),
            options: {
                isWholeLine: true,
                linesDecorationsClassName: 'linked-code-decoration-margin',
                inlineClassName: 'linked-code-decoration-inline',
            },
        }];
        this.updateDecorations();
    }
};

Editor.prototype.updateDecorations = function () {
    this.prevDecorations = this.editor.deltaDecorations(
        this.prevDecorations,
        _.compact(_.flatten(_.values(this.decorations))));
};

Editor.prototype.onConformanceViewOpen = function (editorId) {
    if (editorId === this.id) {
        this.conformanceViewerButton.attr('disabled', true);
    }
};

Editor.prototype.onConformanceViewClose = function (editorId) {
    if (editorId === this.id) {
        this.conformanceViewerButton.attr('disabled', false);
    }
};

Editor.prototype.showLoadSaver = function () {
    this.loadSaveButton.click();
};

Editor.prototype.initLoadSaver = function () {
    this.loadSaveButton
        .off('click')
        .click(_.bind(function () {
            loadSave.run(_.bind(function (text, filename) {
                this.setSource(text);
                this.setFilename(filename);
                this.updateState();
                this.maybeEmitChange(true);
                this.requestCompilation();
            }, this), this.getSource(), this.currentLanguage);
        }, this));
};

Editor.prototype.onLanguageChange = function (newLangId) {
    if (languages[newLangId]) {
        if (newLangId !== this.currentLanguage.id) {
            var oldLangId = this.currentLanguage.id;
            this.currentLanguage = languages[newLangId];
            if (!this.waitingForLanguage && !this.settings.keepSourcesOnLangChange && (newLangId !== 'cmake')) {
                this.editorSourceByLang[oldLangId] = this.getSource();
                this.updateEditorCode();
            }
            this.initLoadSaver();
            monaco.editor.setModelLanguage(this.editor.getModel(), this.currentLanguage.monaco);
            this.isCpp.set(this.currentLanguage.id === 'c++');
            this.isClean.set(this.currentLanguage.id === 'clean');
            this.updateTitle();
            this.updateState();
            // Broadcast the change to other panels
            this.eventHub.emit('languageChange', this.id, newLangId);
            this.maybeEmitChange(true);
            this.requestCompilation();
            ga.proxy('send', {
                hitType: 'event',
                eventCategory: 'LanguageChange',
                eventAction: newLangId,
            });
        }
        this.waitingForLanguage = false;
    }
};

Editor.prototype.getPaneName = function () {
    if (this.filename) {
        return this.filename;
    } else {
        return this.currentLanguage.name + ' source #' + this.id;
    }
};

Editor.prototype.setFilename = function (name) {
    this.filename = name;
    this.updateTitle();
    this.updateState();
};

Editor.prototype.updateTitle = function () {
    var name = this.getPaneName();
    if (name.endsWith('CMakeLists.txt')) {
        this.changeLanguage('cmake');
    }
    this.container.setTitle(_.escape(name));
};

// Called every time we change language, so we get the relevant code
Editor.prototype.updateEditorCode = function () {
    this.setSource(this.editorSourceByLang[this.currentLanguage.id] || languages[this.currentLanguage.id].example);
};

Editor.prototype.close = function () {
    this.eventHub.unsubscribe();
    this.eventHub.emit('editorClose', this.id);
    this.editor.dispose();
    this.hub.removeEditor(this.id);
};

module.exports = {
    Editor: Editor,
};
