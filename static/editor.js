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

    const _ = require('underscore');
    const $ = require('jquery');
    const colour = require('colour');
    const loadSaveLib = require('loadSave');
    const FontScale = require('fontscale');
    const Components = require('components');
    const monaco = require('monaco');
    const options = require('options');
    const Alert = require('alert');
    require('./d-mode');
    require('./rust-mode');

    const loadSave = new loadSaveLib.LoadSave();

    function Editor(hub, state, container, lang, defaultSrc) {
        const self = this;
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

        let cmMode;
        // The first one is used as the default file extension when saving to local file.
        // All of them are used as the contents of the accept attribute of the file input
        let extensions = [];
        switch (lang.toLowerCase()) {
            default:
                cmMode = "cpp";
                extensions = ['.cpp', '.cxx', '.h', '.hpp', '.hxx'];
                break;
            case "c":
                cmMode = "cpp";
                extensions = ['.cpp', '.cxx', '.h', '.hpp', '.hxx'];
                break;
            case "rust":
                cmMode = "rust";
                extensions = ['.rs'];
                break;
            case "d":
                cmMode = "d";
                extensions = ['.d'];
                break;
            case "go":
                cmMode = "go";
                extensions = ['.go'];
                break;
        }

        const root = this.domRoot.find(".monaco-placeholder");
        const legacyReadOnly = state.options && !!state.options.readOnly;
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

        function tryCompilerLinkLine(thisLineNumber, reveal) {
            _.each(self.asmByCompiler, function (asms, compilerId) {
                const targetLines = [];
                _.each(asms, function (asmLine, i) {
                    if (asmLine.source == thisLineNumber) {
                        targetLines.push(i + 1);
                    }
                });
                self.eventHub.emit('compilerSetDecorations', compilerId, targetLines, reveal);
            });
        }

        function clearCompilerLinkedLines() {
            _.each(self.asmByCompiler, function (asms, compilerId) {
                self.eventHub.emit('compilerSetDecorations', compilerId, -1, false);
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
                tryCompilerLinkLine(ed.getPosition().lineNumber, true);
            }
        });

        this.editor.onMouseLeave(function () {
            self.fadeTimeoutId = setTimeout(function () {
                clearCompilerLinkedLines();
                self.fadeTimeoutId = -1;
            }, 5000);
        });

        this.mouseMoveThrottledFunction = _.throttle(function (e) {
                if (e !== null && e.target !== null && self.settings.hoverShowSource === true && e.target.position !== null) {
                    tryCompilerLinkLine(e.target.position.lineNumber, false);
                }
            },
            250
        );

        this.editor.onMouseMove(function (e) {
            self.mouseMoveThrottledFunction(e);
            // This can't be throttled or we can clear a timeout where we're already outside
            if (self.fadeTimeoutId !== -1) {
                clearTimeout(self.fadeTimeoutId);
                self.fadeTimeoutId = -1;
            }
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
            const topBarHeight = self.domRoot.find(".top-bar").outerHeight(true) || 0;
            self.editor.layout({width: self.domRoot.width(), height: self.domRoot.height() - topBarHeight});
        }

        this.domRoot.find('.load-save').click(_.bind(function () {
            loadSave.run(_.bind(function (text) {
                this.editor.setValue(text);
                this.updateState();
                this.maybeEmitChange();
            }, this), this.getSource(), extensions);
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
        this.eventHub.on('themeChange', this.onThemeChange, this);
        this.eventHub.emit('requestSettings');
        this.eventHub.emit('requestTheme');

        // NB a new compilerConfig needs to be created every time; else the state is shared
        // between all compilers created this way. That leads to some nasty-to-find state
        // bugs e.g. https://github.com/mattgodbolt/compiler-explorer/issues/225
        const compilerConfig = _.bind(function () {
            return Components.getCompiler(this.id);
        }, this);

        this.container.layoutManager.createDragSource(
            this.domRoot.find('.btn.add-compiler'), compilerConfig());
        this.domRoot.find('.btn.add-compiler').click(_.bind(function () {
            const insertPoint = hub.findParentRowOrColumn(this.container) ||
                this.container.layoutManager.root.contentItems[0];
            insertPoint.addChild(compilerConfig());
        }, this));

        this.updateState();
    }

    Editor.prototype.maybeEmitChange = function (force) {
        const source = this.getSource();
        if (!force && source === this.lastChangeEmitted) return;
        this.lastChangeEmitted = source;
        this.eventHub.emit('editorChange', this.id, this.lastChangeEmitted);
    };

    Editor.prototype.updateState = function () {
        const state = {
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
        const before = this.settings;
        const after = newSettings;
        this.settings = _.clone(newSettings);

        this.editor.updateOptions({
            autoClosingBrackets: this.settings.autoCloseBrackets,
            tabSize: this.settings.tabWidth,
            quickSuggestions: this.settings.showQuickSuggestions
        });

        // TODO: bug when:
        // * Turn off auto.
        // * edit code
        // * change compiler or compiler options (out of date code is used)
        const bDac = before.compileOnChange ? before.delayAfterChange : 0;
        const aDac = after.compileOnChange ? after.delayAfterChange : 0;
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
        const result = {};
        // First, note all lines used.
        _.each(this.asmByCompiler, function (asm) {
            _.each(asm, function (asmLine) {
                if (asmLine.source) result[asmLine.source - 1] = true;
            });
        });
        // Now assign an ordinal to each used line.
        let ordinal = 0;
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
        const output = (result.stdout || []).concat(result.stderr || []);
        const widgets = _.compact(_.map(output, function (obj) {
            if (!obj.tag) return;
            let severity = 3; // error
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

    Editor.prototype.onThemeChange = function (newTheme) {
        if (this.editor)
            this.editor.updateOptions({theme: newTheme.monaco});
    };

    Editor.prototype.updateDecorations = function () {
        this.prevDecorations = this.editor.deltaDecorations(
            this.prevDecorations, _.flatten(_.values(this.decorations), true));
    };

    return {
        Editor: Editor
    };
});
