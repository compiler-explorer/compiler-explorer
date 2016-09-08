define(function (require) {
    "use strict";
    var CodeMirror = require('codemirror');
    var _ = require('underscore');
    var colour = require('colour');
    var Toggles = require('toggles')
    var compiler = require('compiler');

    require('codemirror/mode/clike/clike');
    require('codemirror/mode/d/d');
    require('codemirror/mode/go/go');
    require('codemirror/mode/rust/rust');

    function Editor(hub, state, container, lang, defaultSrc) {
        var self = this;
        this.id = state.id || hub.nextId();
        this.container = container;
        this.domRoot = container.getElement();
        this.domRoot.html($('#codeEditor').html());
        this.eventHub = container.layoutManager.eventHub;

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
            mode: cmMode
        });

        if (state.src) {
            this.editor.setValue(state.src);
        } else if (defaultSrc) {
            this.editor.setValue(defaultSrc);
        }

        // With reference to "fix typing '#' in emacs mode"
        // https://github.com/mattgodbolt/gcc-explorer/pull/131
        this.editor.setOption("extraKeys", {
            "Alt-F": false
        });

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

        function resize() {
            var topBarHeight = self.domRoot.find(".top-bar").outerHeight(true);
            self.editor.setSize(self.domRoot.width(), self.domRoot.height() - topBarHeight);
            self.editor.refresh();
        }

        container.on('resize', resize);
        container.on('open', function () {
            self.eventHub.emit('editorOpen', self.id);
        });
        container.on('destroy', function () {
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

        var compilerConfig = compiler.getComponent(this.id);
        this.container.layoutManager.createDragSource(
            this.domRoot.find('.btn.add-compiler'), compilerConfig);
    }

    Editor.prototype.maybeEmitChange = function (force) {
        var source = this.getSource();
        if (!force && source == this.lastChangeEmitted) return;
        this.lastChangeEmitted = this.getSource();
        this.eventHub.emit('editorChange', this.id, this.lastChangeEmitted);
    };

    Editor.prototype.updateState = function () {
        this.container.setState({
            id: this.id,
            src: this.getSource(),
            options: this.options.get()
        });
    };

    Editor.prototype.getSource = function () {
        return this.editor.getValue();
    };

    Editor.prototype.onOptionsChange = function (before, after) {
        this.updateState();
        // TODO: bug with options and filters: initial click seems to get lost!
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
        var node = $('<div class="' + clazz + ' inline-msg"><span class="icon">!!</span><span class="compiler">: </span><span class="msg"></span></div>');
        node.find(".msg").text(text);
        node.find(".compiler").text(compiler);
        return node[0];
    }

    function parseLines(lines, callback) {
        var re = /^\/tmp\/[^:]+:([0-9]+)(:([0-9]+))?:\s+(.*)/;
        $.each(lines.split('\n'), function (_, line) {
            line = line.trim();
            if (line !== "") {
                var match = line.match(re);
                if (match) {
                    callback(parseInt(match[1]), match[4].trim());
                } else {
                    callback(null, line);
                }
            }
        });
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
        var output = (result.stdout || "") + (result.stderr || "");
        var self = this;
        this.removeWidgets(this.widgetsByCompiler[compilerId]);
        var widgets = [];
        parseLines(output, function (lineNum, msg) {
            if (lineNum) {
                var widget = self.editor.addLineWidget(
                    lineNum - 1,
                    makeErrorNode(msg, compiler.name),
                    {coverGutter: false, noHScroll: true});
                widgets.push(widget);
            }
        });
        this.widgetsByCompiler[compilerId] = widgets;
        this.asmByCompiler[compilerId] = result.asm;
        this.numberUsedLines();
    };

    return {
        Editor: Editor,
        getComponent: function (id) {
            return {
                type: 'component',
                componentName: 'codeEditor',
                componentState: {id: id}
            };
        }
    };
});