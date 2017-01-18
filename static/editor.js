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
    var Toggles = require('toggles');
    var loadSaveLib = require('loadSave');
    var FontScale = require('fontscale');
    var Sharing = require('sharing');
    var Components = require('components');
    var monaco = require('monaco');

    var loadSave = new loadSaveLib.LoadSave();

    function Editor(hub, state, container, lang, defaultSrc) {
        var self = this;
        this.id = state.id || hub.nextId();
        this.container = container;
        this.domRoot = container.getElement();
        this.domRoot.html($('#codeEditor').html());
        this.eventHub = hub.createEventHub();

        this.widgetsByCompiler = {};
        this.asmByCompiler = {};
        this.busyCompilers = {};
        this.colours = [];
        this.options = new Toggles(this.domRoot.find('.options'), state.options);
        this.options.on('change', _.bind(this.onOptionsChange, this));

        var cmMode;
        switch (lang.toLowerCase()) {
            default:
                cmMode = "cpp";
                break;
            case "c":
                cmMode = "cpp";
                break;
            case "rust":  // TODO
                cmMode = "cpp";
                break;
            case "d":  // TODO
                cmMode = "cpp";
                break;
            case "go":
                cmMode = "go";
                break;
        }

        var root = this.domRoot.find(".monaco-placeholder");
        this.editor = monaco.editor.create(root[0], {
            value: state.source || defaultSrc || "",
            scrollBeyondLastLine: false,
            language: cmMode
        });

        this.editor.addAction({
            id: 'compile',
            label: 'Compile',
            keybindings: [monaco.KeyMod.CtrlCmd | monaco.KeyCode.Enter],
            run: _.bind(function () {
                this.maybeEmitChange();
            }, this)
        });

        this.fontScale = new FontScale(this.domRoot, state, this.editor);
        this.fontScale.on('change', _.bind(this.updateState, this));

        // We suppress posting changes until the user has stopped typing by:
        // * Using _.debounce() to run emitChange on any key event or change
        //   only after a delay.
        // * Only actually triggering a change if the document text has changed from
        //   the previous emitted.
        this.lastChangeEmitted = null;
        var ChangeDebounceMs = 800;
        this.debouncedEmitChange = _.debounce(function () {
            if (self.options.get().compileOnChange) self.maybeEmitChange();
        }, ChangeDebounceMs);
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
            var topBarHeight = self.domRoot.find(".top-bar").outerHeight(true);
            self.editor.layout({width: self.domRoot.width(), height: self.domRoot.height() - topBarHeight});
        }

        this.domRoot.find('.load-save').click(_.bind(function () {
            loadSave.run(_.bind(function (text) {
                this.editor.setValue(text);
                this.updateState();
                this.maybeEmitChange();
            }, this));
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

        Sharing.initShareButton(this.domRoot.find('.share'), container.layoutManager);

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
            source: this.getSource(),
            options: this.options.get()
        };
        this.fontScale.addState(state);
        this.container.setState(state);
    };

    Editor.prototype.getSource = function () {
        return this.editor.getModel().getValue();
    };

    Editor.prototype.onOptionsChange = function (before, after) {
        this.updateState();
        // TODO: bug when:
        // * Turn off auto.
        // * edit code
        // * change compiler or compiler options (out of date code is used)
        if (after.compileOnChange && !before.compileOnChange) {
            // If we've just enabled "compile on change"; forcibly send a change
            // which will recolourise as required.
            this.maybeEmitChange(true);
        } else if (before.colouriseAsm !== after.colouriseAsm) {
            // if the colourise option has been toggled...recompute colours
            this.numberUsedLines();
        }
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
        this.updateColours(this.options.get().colouriseAsm ? result : []);
    };

    Editor.prototype.updateColours = function (colours) {
        this.colours = colour.applyColours(this.editor, colours, this.colours);
        this.eventHub.emit('colours', this.id, colours);
    };

    Editor.prototype.onCompilerClose = function (compilerId) {
        monaco.editor.setModelMarkers(this.editor.getModel(), compilerId, []);
        delete this.widgetsByCompiler[compilerId];
        delete this.asmByCompiler[compilerId];
        delete this.busyCompilers[compilerId];
        this.numberUsedLines();
    };

    Editor.prototype.onCompilerOpen = function () {
        // On any compiler open, rebroadcast our state in case they need to know it.
        this.maybeEmitChange(true);
    };

    Editor.prototype.onCompiling = function (compilerId) {
        this.busyCompilers[compilerId] = true;
    };

    Editor.prototype.onCompileResponse = function (compilerId, compiler, result) {
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
        this.numberUsedLines();
    };

    Editor.prototype.onSelectLine = function (id, lineNum) {
        if (id === this.id) {
            this.editor.setSelection({line: lineNum - 1, ch: 0}, {line: lineNum, ch: 0});
        }
    };

    return {
        Editor: Editor
    };
});
