define(function (require) {
    "use strict";
    var CodeMirror = require('codemirror');
    var _ = require('underscore');
    var colour = require('colour');
    require('codemirror/mode/clike/clike');
    require('codemirror/mode/d/d');
    require('codemirror/mode/go/go');
    require('codemirror/mode/rust/rust');

    function Editor(hub, state, container, lang, defaultSrc) {
        var self = this;
        this.id = state.id || hub.nextId();
        this.eventHub = container.layoutManager.eventHub;
        this.widgetsByCompiler = {};
        this.asmByCompiler = {};

        this.container = container;
        var domRoot = container.getElement();
        domRoot.html($('#codeEditor').html());

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

        this.editor = CodeMirror.fromTextArea(domRoot.find("textarea")[0], {
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
            self.maybeEmitChange();
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
            self.editor.setSize(domRoot.width(), domRoot.height());
            self.editor.refresh();
        }

        container.on('resize', resize);
        container.on('open', function () {
            self.eventHub.emit('editorOpen', self.id);
        });
        container.on('close', function () {
            self.eventHub.emit('editorClose', self.id);
        });
        container.setTitle(lang + " source");
        this.container.layoutManager.on('initialised', function () {
            // Once initialized, let everyone know what text we have.
            self.maybeEmitChange();
        });

        this.eventHub.on('compilerClose', this.onCompilerClose, this);
        this.eventHub.on('compileResult', this.onCompileResponse, this);
    }

    Editor.prototype.maybeEmitChange = function () {
        var source = this.getSource();
        if (source == this.lastChangeEmitted) return;
        this.lastChangeEmitted = this.getSource();
        this.eventHub.emit('editorChange', this.id, this.lastChangeEmitted);
    };

    Editor.prototype.updateState = function () {
        this.container.setState({
            id: this.id,
            src: this.getSource()
        });
    };

    Editor.prototype.getSource = function () {
        return this.editor.getValue();
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

        // TODO: make colourise an option on the editor not the asm views
        this.debouncedUpdateColours(result);
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

    return Editor;
});