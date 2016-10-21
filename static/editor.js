// Copyright (c) 2012-2016, Matt Godbolt
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
    var CodeMirror = require('codemirror');
    var _ = require('underscore');
    var $ = require('jquery');
    var colour = require('colour');
    var Toggles = require('toggles');
    var compiler = require('compiler');
    var loadSaveLib = require('loadSave');
    var FontScale = require('fontscale');

    require('codemirror/mode/clike/clike');
    require('codemirror/mode/d/d');
    require('codemirror/mode/go/go');
    require('codemirror/mode/rust/rust');

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
        this.options = new Toggles(this.domRoot.find('.options'), state.options);
        this.options.on('change', _.bind(this.onOptionsChange, this));

        var cmMode;
        switch (lang.toLowerCase()) {
            default:
                cmMode = "text/x-c++src";
                break;
            case "c":
                cmMode = "text/x-c";
                break;
            case "rust":
                cmMode = "text/x-rustsrc";
                break;
            case "d":
                cmMode = "text/x-d";
                break;
            case "go":
                cmMode = "text/x-go";
                break;
        }

        this.editor = CodeMirror.fromTextArea(this.domRoot.find("textarea")[0], {
            lineNumbers: true,
            matchBrackets: true,
            useCPP: true,
            dragDrop: false,
            extraKeys: {"Alt-F": false}, // see https://github.com/mattgodbolt/gcc-explorer/pull/131
            mode: cmMode
        });

        this.fontScale = new FontScale(this.domRoot, state);
        this.fontScale.on('change', _.bind(this.updateState, this));

        if (state.source) {
            this.editor.setValue(state.source);
        } else if (defaultSrc) {
            this.editor.setValue(defaultSrc);
        }

        // We suppress posting changes until the user has stopped typing by:
        // * Using _.debounce() to run emitChange on any key event or change
        //   only after a delay.
        // * Only actually triggering a change if the document text has changed from
        //   the previous emitted.
        this.lastChangeEmitted = null;
        var ChangeDebounceMs = 500;
        this.debouncedEmitChange = _.debounce(function () {
            if (self.options.get().compileOnChange) self.maybeEmitChange();
        }, ChangeDebounceMs);
        this.editor.on("change", _.bind(function () {
            this.debouncedEmitChange();
            this.updateState();
        }, this));
        this.editor.on("keydown", _.bind(function () {
            // Not strictly a change; but this suppresses changes until some time
            // after the last key down (be it an actual change or a just a cursor
            // movement etc).
            this.debouncedEmitChange();
        }, this));

        // A small debounce used to give multiple compilers a chance to respond
        // before unioning the colours and updating them. Another approach to reducing
        // flicker as multiple compilers update is to track those compilers which
        // are busy, and only union/update colours when all are complete.
        var ColourDebounceMs = 200;
        this.debouncedUpdateColours = _.debounce(function (colours) {
            self.updateColours(colours);
        }, ColourDebounceMs);

        function refresh() {
            self.editor.refresh();
        }

        function resize() {
            var topBarHeight = self.domRoot.find(".top-bar").outerHeight(true);
            self.editor.setSize(self.domRoot.width(), self.domRoot.height() - topBarHeight);
            refresh();
        }

        this.domRoot.find('.load-save').click(_.bind(function () {
            loadSave.run(_.bind(function (text) {
                this.editor.setValue(text);
                this.updateState();
                this.maybeEmitChange();
            }, this));
        }, this));

        container.on('resize', resize);
        container.on('shown', refresh);
        container.on('open', function () {
            self.eventHub.emit('editorOpen', self.id);
        });
        container.on('destroy', function () {
            self.eventHub.unsubscribe();
            self.eventHub.emit('editorClose', self.id);
        });
        container.setTitle(lang + " source #" + self.id);
        this.container.layoutManager.on('initialised', function () {
            // Once initialized, let everyone know what text we have.
            self.maybeEmitChange();
        });

        this.eventHub.on('compilerOpen', this.onCompilerOpen, this);
        this.eventHub.on('compilerClose', this.onCompilerClose, this);
        this.eventHub.on('compileResult', this.onCompileResponse, this);
        this.eventHub.on('selectLine', this.onSelectLine, this);

        var compilerConfig = compiler.getComponent(this.id);

        this.container.layoutManager.createDragSource(
            this.domRoot.find('.btn.add-compiler'), compilerConfig);
        this.domRoot.find('.btn.add-compiler').click(_.bind(function () {
            var insertPoint = hub.findParentRowOrColumn(this.container) ||
                this.container.layoutManager.root.contentItems[0];
            insertPoint.addChild(compilerConfig);
        }, this));
    }

    Editor.prototype.maybeEmitChange = function (force) {
        var source = this.getSource();
        if (!force && source == this.lastChangeEmitted) return;
        this.lastChangeEmitted = this.getSource();
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
        return this.editor.getValue();
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

    function makeErrorNode(text, compiler) {
        var clazz = "error";
        if (text.match(/^warning/)) clazz = "warning";
        if (text.match(/^note/)) clazz = "note";
        var node = $('.template .inline-msg').clone();
        node.find('.icon').addClass(clazz);
        node.find(".msg").text(text);
        node.find(".compiler").text(compiler);
        return node[0];
    }

    Editor.prototype.removeWidgets = function (widgets) {
        var self = this;
        _.each(widgets, function (widget) {
            self.editor.removeLineWidget(widget);
        });
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

        this.debouncedUpdateColours(this.options.get().colouriseAsm ? result : []);
    };

    Editor.prototype.updateColours = function (colours) {
        colour.applyColours(this.editor, colours);
        this.eventHub.emit('colours', this.id, colours);
    };

    Editor.prototype.onCompilerClose = function (compilerId) {
        this.removeWidgets(this.widgetsByCompiler[compilerId]);
        delete this.widgetsByCompiler[compilerId];
        delete this.asmByCompiler[compilerId];
        this.numberUsedLines();
    };

    Editor.prototype.onCompilerOpen = function () {
        // On any compiler open, rebroadcast our state in case they need to know it.
        this.maybeEmitChange(true);
    };

    Editor.prototype.onCompileResponse = function (compilerId, compiler, result) {
        var output = (result.stdout || []).concat(result.stderr || []);
        var self = this;
        this.removeWidgets(this.widgetsByCompiler[compilerId]);
        var widgets = [];
        _.each(output, function (obj) {
            if (obj.tag) {
                var widget = self.editor.addLineWidget(
                    obj.tag.line - 1,
                    makeErrorNode(obj.tag.text, compiler.name),
                    {coverGutter: false, noHScroll: true});
                widgets.push(widget);
            }
        }, this);
        this.widgetsByCompiler[compilerId] = widgets;
        this.asmByCompiler[compilerId] = result.asm;
        this.numberUsedLines();
    };

    Editor.prototype.onSelectLine = function (id, lineNum) {
        if (id === this.id) {
            this.editor.setSelection({line: lineNum - 1, ch: 0}, {line: lineNum, ch: 0});
        }
    };

    return {
        Editor: Editor,
        getComponent: function (id) {
            return {
                type: 'component',
                componentName: 'codeEditor',
                componentState: {id: id},
                isClosable: false
            };
        },
        getComponentWith: function (id, source, options) {
            return {
                type: 'component',
                componentName: 'codeEditor',
                componentState: {id: id, source: source, options: options},
                isClosable: false
            };
        }
    };
});
