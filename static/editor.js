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
    var Sharing = require('sharing');
    var Components = require('components');
    var monaco = require('monaco');
    var options = require('options');
    var Alert = require('alert');
    require('./d-mode');
    require('./rust-mode');

    var loadSave = new loadSaveLib.LoadSave();

    function Editor(hub, state, container, lang, defaultSrc) {
        var self = this;
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

        this.decorations = [];

        var cmMode;
        switch (lang.toLowerCase()) {
            default:
                cmMode = "cpp";
                break;
            case "c":
                cmMode = "cpp";
                break;
            case "rust":
                cmMode = "rust";
                break;
            case "d":
                cmMode = "d";
                break;
            case "go":
                cmMode = "go";
                break;
        }

        var root = this.domRoot.find(".monaco-placeholder");
        var legacyReadOnly = state.options && !!state.options.readOnly;
        this.editor = monaco.editor.create(root[0], {
            value: state.source || defaultSrc || "",
            scrollBeyondLastLine: false,
            language: cmMode,
            fontFamily: 'Fira Mono',
            readOnly: !!options.readOnly || legacyReadOnly,
            glyphMargin: true,
            quickSuggestions: false,
            fixedOverflowWidgets: true
        });

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

        function tryCompilerSelectLine(thisLineNumber, reveal) {
            _.each(self.asmByCompiler, function (asms, compilerId) {
                var targetLines = [];
                _.each(asms, function (asmLine, i) {
                    if (asmLine.source == thisLineNumber) {
                        targetLines.push(i + 1);
                    }
                });
                self.eventHub.emit('compilerSetDecorations', compilerId, targetLines, reveal);
            });
        }

        this.editor.addAction({
            id: 'viewasm',
            label: 'Scroll to assembly',
            keybindings: [monaco.KeyMod.CtrlCmd | monaco.KeyCode.F10],
            keybindingContext: null,
            contextMenuGroupId: 'navigation',
            contextMenuOrder: 1.5,
            run: function (ed) {
                tryCompilerSelectLine(ed.getPosition().lineNumber, true);
            }
        });

        this.mouseMoveThrottledFunction = _.throttle(function (e) {
                if (e !== null && e.target !== null && self.settings.hoverShowSource === true && e.target.position !== null) {
                    tryCompilerSelectLine(e.target.position.lineNumber, false);
                }
            },
            250
        );

        this.editor.onMouseMove(function (e) {
            self.mouseMoveThrottledFunction(e);
        });

        this.fontScale = new FontScale(this.domRoot, state, this.editor);
        this.fontScale.on('change', _.bind(this.updateState, this));

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

        function layout() {
            var topBarHeight = self.domRoot.find(".top-bar").outerHeight(true) || 0;
            self.editor.layout({width: self.domRoot.width(), height: self.domRoot.height() - topBarHeight});
        }

        this.domRoot.find('.load-save').click(_.bind(function () {
            loadSave.run(_.bind(function (text) {
                this.editor.setValue(text);
                this.updateState();
                this.maybeEmitChange();
            }, this), this.getSource());
        }, this));

        container.on('resize', layout);
        container.on('shown', layout);
        container.on('open', function () {
            self.eventHub.emit('editorOpen', self.id);
        });
        container.on('destroy', function () {
            self.eventHub.unsubscribe();
            self.eventHub.emit('editorClose', self.id);
            self.editor.dispose();
        });
        container.setTitle(lang + " source #" + self.id);
        this.container.layoutManager.on('initialised', function () {
            // Once initialized, let everyone know what text we have.
            self.maybeEmitChange();
        });

        this.eventHub.on('compilerOpen', this.onCompilerOpen, this);
        this.eventHub.on('compilerClose', this.onCompilerClose, this);
        this.eventHub.on('compiling', this.onCompiling, this);
        this.eventHub.on('compileResult', this.onCompileResponse, this);
        this.eventHub.on('selectLine', this.onSelectLine, this);
        this.eventHub.on('editorSetDecoration', this.onEditorSetDecoration, this);
        this.eventHub.on('settingsChange', this.onSettingsChange, this);
        this.eventHub.emit('requestSettings');

        // NB a new compilerConfig needs to be created every time; else the state is shared
        // between all compilers created this way. That leads to some nasty-to-find state
        // bugs e.g. https://github.com/mattgodbolt/compiler-explorer/issues/225
        var compilerConfig = _.bind(function () {
            return Components.getCompiler(this.id);
        }, this);

        this.container.layoutManager.createDragSource(
            this.domRoot.find('.btn.add-compiler'), compilerConfig());
        this.domRoot.find('.btn.add-compiler').click(_.bind(function () {
            var insertPoint = hub.findParentRowOrColumn(this.container) ||
                this.container.layoutManager.root.contentItems[0];
            insertPoint.addChild(compilerConfig());
        }, this));

        this.updateState();
    }

    Editor.prototype.maybeEmitChange = function (force) {
        var source = this.getSource();
        if (!force && source == this.lastChangeEmitted) return;
        this.lastChangeEmitted = source;
        this.eventHub.emit('editorChange', this.id, this.lastChangeEmitted);
    };

    Editor.prototype.updateState = function () {
        var state = {
            id: this.id,
            source: this.getSource()
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

        this.editor.updateOptions({autoClosingBrackets: this.settings.autoCloseBrackets, tabSize: this.settings.tabWidth});
        if (before.tabWidth !== after.tabWidth) {
            this.editor.getModel().updateOptions({tabSize: this.settings.tabWidth});
            // TODO: We could use an auto reindentation here, but currently there is no method on Monaco
        }

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
                if (asmLine.source) result[asmLine.source - 1] = true;
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
                source: compiler.name,
                startLineNumber: obj.tag.line,
                startColumn: obj.tag.column || 0,
                endLineNumber: obj.tag.line,
                endColumn: -1
            };
        }, this));
        monaco.editor.setModelMarkers(this.editor.getModel(), compilerId, widgets);
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
            if (reveal)
                this.editor.revealLineInCenter(lineNum);
            this.decorations = this.editor.deltaDecorations(this.decorations,
                lineNum === -1 || lineNum === null ? [] : [
                        {
                            range: new monaco.Range(lineNum, 1, lineNum, 1),
                            options: {
                                linesDecorationsClassName: 'linked-code-decoration'
                            }
                        }
                    ]);
        }
    };

    return {
        Editor: Editor
    };
});
