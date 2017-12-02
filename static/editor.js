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
    var _ = require('underscore');
    var $ = require('jquery');
    var colour = require('colour');
    var loadSaveLib = require('loadSave');
    var FontScale = require('fontscale');
    var Components = require('components');
    var monaco = require('monaco');
    var options = require('options');
    var Alert = require('alert');
    require('./cppp-mode');
    require('./d-mode');
    require('./rust-mode');
    require('./ispc-mode');
    require('./haskell-mode');
    require('./swift-mode');
    require('./fpc-mode');
    require('selectize');

    var loadSave = new loadSaveLib.LoadSave();

    var languages = {
        'c++': {
            id: 'c++',
            name: 'C++',
            monaco: 'cppp',
            extensions: ['.cpp', '.cxx', '.h', '.hpp', '.hxx', '.c']
        },
        cppx: {
            id: 'cppx',
            name: 'Cppx',
            monaco: 'cppp',
            extensions: ['.cppx', '.cpp', '.cxx', '.h', '.hpp', '.hxx', '.c']
        },
        c: {
            id: 'c',
            name: 'C',
            monaco: 'c',
            extensions: ['.c', '.h']
        },
        rust: {
            id: 'rust',
            name: 'Rust',
            monaco: 'rust',
            extensions: ['.rs']
        },
        d: {
            id: 'd',
            name: 'D',
            monaco: 'd',
            extensions: ['.d']
        },
        go: {
            id: 'go',
            name: 'Go',
            monaco: 'go',
            extensions: ['.go']
        },
        ispc: {
            id: 'ispc',
            name: 'ispc',
            monaco: 'ispc',
            extensions: ['.ispc']
        },
        haskell: {
            id: 'haskell',
            name: 'Haskell',
            monaco: 'haskell',
            extensions: ['.haskell']
        },
        swift: {
            id: 'swift',
            name: 'Swift',
            monaco: 'swift',
            extensions: ['.swift']
        }
    };

    function Editor(hub, state, container, lang, defaultSrc) {
        this.id = state.id || hub.nextEditorId();
        this.container = container;
        this.domRoot = container.getElement();
        this.domRoot.html($('#codeEditor').html());
        this.eventHub = hub.createEventHub();
        this.settings = {};
        this.ourCompilers = {};

        this.widgetsByCompiler = {};
        this.asmByCompiler = {};
        this.busyCompilers = {};
        this.colours = [];
        this.lastCompilerIDResponse = -1;

        this.decorations = {};
        this.prevDecorations = [];

        this.fadeTimeoutId = -1;

        this.languageBtn = this.domRoot.find('.change-language');
        this.currentLanguage = state.lang ? languages[state.lang] : languages["c++"];

        var root = this.domRoot.find(".monaco-placeholder");
        var legacyReadOnly = state.options && !!state.options.readOnly;
        this.editor = monaco.editor.create(root[0], {
            value: state.source || defaultSrc || "",
            scrollBeyondLastLine: false,
            language: this.currentLanguage.monaco,
            fontFamily: 'Fira Mono',
            readOnly: !!options.readOnly || legacyReadOnly,
            glyphMargin: !options.embedded,
            quickSuggestions: false,
            fixedOverflowWidgets: true,
            minimap: {
                maxColumn: 80
            },
            folding: true,
            lineNumbersMinChars: options.embedded ? 1 : 3,
            emptySelectionClipboard: true,
            autoIndent: true
        });

        var startFolded = /^[/*#;]+\s*setup.*/;
        if (state.source && state.source.match(startFolded)) {
            var foldAction = this.editor.getAction('editor.fold');
            foldAction.run();
        }

        this.editor.addAction({
            id: 'compile',
            label: 'Compile',
            keybindings: [monaco.KeyMod.CtrlCmd | monaco.KeyCode.Enter],
            keybindingContext: null,
            contextMenuGroupId: 'navigation',
            contextMenuOrder: 1.5,
            run: _.bind(function () {
                this.maybeEmitChange();
            }, this)
        });

        this.editor.addAction({
            id: 'toggleCompileOnChange',
            label: 'Toggle compile on change',
            keybindings: [monaco.KeyMod.CtrlCmd | monaco.KeyMod.Shift | monaco.KeyCode.Enter],
            keybindingContext: null,
            run: _.bind(function () {
                this.eventHub.emit('modifySettings', {
                    compileOnChange: !this.settings.compileOnChange
                });
                new Alert().notify('Compile on code change has been toggled ' + (this.settings.compileOnChange ? 'ON' : 'OFF'), {
                    group: "togglecompile",
                    alertClass: this.settings.compileOnChange ? "notification-on" : "notification-off",
                    dismissTime: 3000
                });
            }, this)
        });

        this.editor.addAction({
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

        this.editor.addAction({
            id: 'viewasm',
            label: 'Scroll to assembly',
            keybindings: [monaco.KeyMod.CtrlCmd | monaco.KeyCode.F10],
            keybindingContext: null,
            contextMenuGroupId: 'navigation',
            contextMenuOrder: 1.5,
            run: function (ed) {
                tryCompilerLinkLine(ed.getPosition().lineNumber, true);
            }
        });

        var tryCompilerLinkLine = _.bind(function (thisLineNumber, reveal) {
            _.each(this.asmByCompiler, _.bind(function (asms, compilerId) {
                var targetLines = [];
                _.each(asms, function (asmLine, i) {
                    if (asmLine.source && asmLine.source.file === null &&
                        asmLine.source.line == thisLineNumber) {
                        targetLines.push(i + 1);
                    }
                });
                this.eventHub.emit('compilerSetDecorations', compilerId, targetLines, reveal);
            }, this));
        }, this);

        var clearCompilerLinkedLines = _.bind(function () {
            _.each(this.asmByCompiler, _.bind(function (asms, compilerId) {
                this.eventHub.emit('compilerSetDecorations', compilerId, -1, false);
            }, this));
        }, this);

        this.editor.onMouseLeave(_.bind(function () {
            this.fadeTimeoutId = setTimeout(_.bind(function () {
                clearCompilerLinkedLines();
                this.fadeTimeoutId = -1;
            }, this), 5000);
        }, this));

        this.mouseMoveThrottledFunction = _.throttle(_.bind(function (e) {
                if (e !== null && e.target !== null && this.settings.hoverShowSource && e.target.position !== null) {
                    tryCompilerLinkLine(e.target.position.lineNumber, false);
                }
            }, this), 250);

        this.editor.onMouseMove(_.bind(function (e) {
            this.mouseMoveThrottledFunction(e);
            // This can't be throttled or we can clear a timeout where we're already outside
            if (this.fadeTimeoutId !== -1) {
                clearTimeout(this.fadeTimeoutId);
                this.fadeTimeoutId = -1;
            }
        }, this));

        this.updateEditorLayout = _.bind(function () {
            var topBarHeight = this.domRoot.find(".top-bar").outerHeight(true) || 0;
            this.editor.layout({width: this.domRoot.width(), height: this.domRoot.height() - topBarHeight});
        }, this);

        this.fontScale = new FontScale(this.domRoot, state, this.editor);
        this.fontScale.on('change', _.bind(this.updateState, this));

        this.languageBtn.selectize({
            sortField: 'name',
            valueField: 'id',
            labelField: 'name',
            searchField: ['name'],
            options: _.map(languages, function (lang) { return lang; }),
            items: [this.currentLanguage.id]
        }).on('change', _.bind(function (e) {
            this.onLanguageChange($(e.target).val());
        }, this));

        // We suppress posting changes until the user has stopped typing by:
        // * Using _.debounce() to run emitChange on any key event or change
        //   only after a delay.
        // * Only actually triggering a change if the document text has changed from
        //   the previous emitted.
        this.lastChangeEmitted = null;
        this.onSettingsChange({});
        this.editor.getModel().onDidChangeContent(_.bind(function () {
            this.debouncedEmitChange();
            this.updateState();
        }, this));
        // this.editor.on("keydown", _.bind(function () {
        //     // Not strictly a change; but this suppresses changes until some time
        //     // after the last key down (be it an actual change or a just a cursor
        //     // movement etc).
        //     this.debouncedEmitChange();
        // }, this));

        container.on('resize', this.updateEditorLayout);
        container.on('shown', this.updateEditorLayout);
        container.on('open', _.bind(function () {
            this.eventHub.emit('editorOpen', this.id);
        }, this));
        container.on('destroy', _.bind(function () {
            this.eventHub.unsubscribe();
            this.eventHub.emit('editorClose', this.id);
            this.editor.dispose();
        }, this));
        this.container.layoutManager.on('initialised', function () {
            // Once initialized, let everyone know what text we have.
            this.maybeEmitChange();
        }, this);

        this.eventHub.on('compilerOpen', this.onCompilerOpen, this);
        this.eventHub.on('compilerClose', this.onCompilerClose, this);
        this.eventHub.on('compiling', this.onCompiling, this);
        this.eventHub.on('compileResult', this.onCompileResponse, this);
        this.eventHub.on('selectLine', this.onSelectLine, this);
        this.eventHub.on('editorSetDecoration', this.onEditorSetDecoration, this);
        this.eventHub.on('settingsChange', this.onSettingsChange, this);
        this.eventHub.on('conformanceViewOpen', this.onConformanceViewOpen, this);
        this.eventHub.on('conformanceViewClose', this.onConformanceViewClose, this);
        this.eventHub.on('resize', this.updateEditorLayout, this);
        this.eventHub.emit('requestSettings');

        // NB a new compilerConfig needs to be created every time; else the state is shared
        // between all compilers created this way. That leads to some nasty-to-find state
        // bugs e.g. https://github.com/mattgodbolt/compiler-explorer/issues/225
        var compilerConfig = _.bind(function () {
            return Components.getCompiler(this.id);
        }, this);

        var addCompilerButton = this.domRoot.find('.btn.add-compiler');

        this.container.layoutManager.createDragSource(
            addCompilerButton, compilerConfig);
        addCompilerButton.click(_.bind(function () {
            var insertPoint = hub.findParentRowOrColumn(this.container) ||
                this.container.layoutManager.root.contentItems[0];
            insertPoint.addChild(compilerConfig);
        }, this));


        var conformanceConfig = _.bind(function () {
            return Components.getConformanceView(this.id, this.getSource());
        }, this);

        this.conformanceViewerButton = this.domRoot.find('.btn.conformance');

        this.container.layoutManager.createDragSource(
            this.conformanceViewerButton, conformanceConfig);
        this.conformanceViewerButton.click(_.bind(function () {
            var insertPoint = hub.findParentRowOrColumn(this.container) ||
                this.container.layoutManager.root.contentItems[0];
            insertPoint.addChild(conformanceConfig);
        }, this));

        this.updateState();
    }

    Editor.prototype.maybeEmitChange = function (force) {
        var source = this.getSource();
        if (!force && source === this.lastChangeEmitted) return;
        this.lastChangeEmitted = source;
        this.eventHub.emit('editorChange', this.id, this.lastChangeEmitted);
    };

    Editor.prototype.updateState = function () {
        var state = {
            id: this.id,
            source: this.getSource(),
            lang: this.currentLanguage.id
        };
        this.fontScale.addState(state);
        this.container.setState(state);
    };

    Editor.prototype.setSource = function (newSource) {
        this.editor.getModel().setValue(newSource);
    };

    Editor.prototype.getSource = function () {
        return this.editor.getModel().getValue();
    };

    Editor.prototype.onSettingsChange = function (newSettings) {
        var before = this.settings;
        var after = newSettings;
        this.settings = _.clone(newSettings);

        this.editor.updateOptions({
            autoClosingBrackets: this.settings.autoCloseBrackets,
            tabSize: this.settings.tabWidth,
            quickSuggestions: this.settings.showQuickSuggestions,
            contextmenu: this.settings.useCustomContextMenu,
            minimap: {
                enabled: this.settings.showMinimap && !options.embedded
            }
        });

        // TODO: bug when:
        // * Turn off auto.
        // * edit code
        // * change compiler or compiler options (out of date code is used)
        var bDac = before.compileOnChange ? before.delayAfterChange : 0;
        var aDac = after.compileOnChange ? after.delayAfterChange : 0;
        if (bDac !== aDac || !this.debouncedEmitChange) {
            if (aDac) {
                this.debouncedEmitChange = _.debounce(_.bind(function () {
                    this.maybeEmitChange();
                }, this), after.delayAfterChange);
                this.maybeEmitChange(true);
            } else {
                this.debouncedEmitChange = _.noop;
            }
        }

        if (before.hoverShowSource && !after.hoverShowSource) {
            this.onEditorSetDecoration(this.id, -1, false);
        }

        this.numberUsedLines();
    };

    Editor.prototype.numberUsedLines = function () {
        var result = {};
        // First, note all lines used.
        _.each(this.asmByCompiler, function (asm) {
            _.each(asm, function (asmLine) {
                // If the line has a source indicator, and the source indicator is null (i.e. the
                // user's input file), then tag it as used.
                if (asmLine.source && asmLine.source.file === null)
                    result[asmLine.source.line - 1] = true;
            });
        });
        // Now assign an ordinal to each used line.
        var ordinal = 0;
        _.each(result, function (v, k) {
            result[k] = ordinal++;
        });

        if (_.any(this.busyCompilers)) return;
        this.updateColours(this.settings.colouriseAsm ? result : []);
    };

    Editor.prototype.updateColours = function (colours) {
        this.colours = colour.applyColours(this.editor, colours, this.settings.colourScheme, this.colours);
        this.eventHub.emit('colours', this.id, colours, this.settings.colourScheme);
    };

    Editor.prototype.onCompilerOpen = function (compilerId, editorId) {
        if (editorId === this.id) {
            // On any compiler open, rebroadcast our state in case they need to know it.
            this.maybeEmitChange(true);
            this.ourCompilers[compilerId] = true;
        }
    };

    Editor.prototype.onCompilerClose = function (compilerId) {
        if (!this.ourCompilers[compilerId]) return;
        monaco.editor.setModelMarkers(this.editor.getModel(), compilerId, []);
        delete this.widgetsByCompiler[compilerId];
        delete this.asmByCompiler[compilerId];
        delete this.busyCompilers[compilerId];
        this.numberUsedLines();
    };

    Editor.prototype.onCompiling = function (compilerId) {
        if (!this.ourCompilers[compilerId]) return;
        this.busyCompilers[compilerId] = true;
    };

    Editor.prototype.onCompileResponse = function (compilerId, compiler, result) {
        if (!this.ourCompilers[compilerId]) return;
        this.busyCompilers[compilerId] = false;
        var output = (result.stdout || []).concat(result.stderr || []);
        var widgets = _.compact(_.map(output, function (obj) {
            if (!obj.tag) return;
            var severity = 3; // error
            if (obj.tag.text.match(/^warning/)) severity = 2;
            if (obj.tag.text.match(/^note/)) severity = 1;
            return {
                severity: severity,
                message: obj.tag.text,
                source: compiler.name + ' #' + compilerId,
                startLineNumber: obj.tag.line,
                startColumn: obj.tag.column || 0,
                endLineNumber: obj.tag.line,
                endColumn: -1
            };
        }, this));
        monaco.editor.setModelMarkers(this.editor.getModel(), compilerId, widgets);
        this.decorations.tags = _.map(widgets, function (tag) {
            return {
                range: new monaco.Range(tag.startLineNumber, tag.startColumn, tag.startLineNumber + 1, 1),
                options: {
                    isWholeLine: false,
                    inlineClassName: "error-code"
                }
            };
        }, this);
        this.updateDecorations();
        this.asmByCompiler[compilerId] = result.asm;
        this.lastCompilerIDResponse = compilerId;
        this.numberUsedLines();
    };

    Editor.prototype.onSelectLine = function (id, lineNum) {
        if (id === this.id) {
            this.editor.setSelection({line: lineNum - 1, ch: 0}, {line: lineNum, ch: 0});
        }
    };

    Editor.prototype.onEditorSetDecoration = function (id, lineNum, reveal) {
        if (id === this.id) {
            if (reveal && lineNum)
                this.editor.revealLineInCenter(lineNum);
            this.decorations.linkedCode = lineNum === -1 || !lineNum ?
                []
                :
                [
                    {
                        range: new monaco.Range(lineNum, 1, lineNum, 1),
                        options: {
                            isWholeLine: true,
                            linesDecorationsClassName: 'linked-code-decoration-margin',
                            inlineClassName: 'linked-code-decoration-inline'
                        }
                    }
                ];
            this.updateDecorations();
        }
    };

    Editor.prototype.updateDecorations = function () {
        this.prevDecorations = this.editor.deltaDecorations(
            this.prevDecorations, _.flatten(_.values(this.decorations), true));
    };

    Editor.prototype.onConformanceViewOpen = function (editorId) {
        if (editorId == this.id) {
            this.conformanceViewerButton.attr("disabled", true);
        }
    };

    Editor.prototype.onConformanceViewClose = function (editorId) {
        if (editorId == this.id) {
            this.conformanceViewerButton.attr("disabled", false);
        }
    };

    Editor.prototype.onLanguageChange = function (newLangId) {
        if (newLangId !== this.currentLanguage.id) {
            this.currentLanguage = languages[newLangId] || languages.cpp;
            monaco.editor.setModelLanguage(this.editor.getModel(), this.currentLanguage.monaco);
            this.domRoot.find('.load-save')
                .off('click')
                .click(_.bind(function () {
                    loadSave.run(_.bind(function (text) {
                        this.editor.setValue(text);
                        this.updateState();
                        this.maybeEmitChange();
                    }, this), this.getSource(), this.currentLanguage.extensions);
                }, this));
            this.container.setTitle(this.currentLanguage.name + " source #" + this.id);
            this.updateState();
            // Broadcast the change to other panels
            this.eventHub.emit("languageChange", this.id, newLangId);
        }
    };

    return {
        Editor: Editor
    };
});
