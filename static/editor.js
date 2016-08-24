define(function (require) {
    "use strict";
    var CodeMirror = require('codemirror');
    require('codemirror/mode/clike/clike');
    require('codemirror/mode/d/d');
    require('codemirror/mode/go/go');
    require('codemirror/mode/rust/rust');

    function Editor(hub, state, container, lang, defaultSrc) {
        var self = this;
        this.id = state.id || hub.nextEditorId();

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
        this.editor.on("change", function () {
            hub.onEditorChange(self);
            self.updateState();
        });

        function resize() {
            self.editor.setSize(domRoot.width(), domRoot.height());
            self.editor.refresh();
        }

        container.on('resize', resize);
        container.on('open', resize);
        container.on('close', function () {
            hub.removeEditor(self);
        });
        container.setTitle(lang + " source");
    }

    Editor.prototype.updateState = function () {
        this.container.setState({
            id: this.id,
            src: this.getSource()
        });
    };

    Editor.prototype.getSource = function () {
        return this.editor.getValue();
    };

    Editor.prototype.getId = function () {
        return this.id;
    };

    return Editor;
});