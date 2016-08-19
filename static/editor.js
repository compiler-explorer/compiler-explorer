define(function (require) {
    "use strict";
    var CodeMirror = require('codemirror');
    require('codemirror/mode/clike/clike');
    require('codemirror/mode/d/d');
    require('codemirror/mode/go/go');
    require('codemirror/mode/rust/rust');
    require('asm-mode');

    return function Editor(container, lang) {
        var domRoot = container.getElement();
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

        var cppEditor = CodeMirror.fromTextArea(domRoot.find("textarea")[0], {
            lineNumbers: true,
            matchBrackets: true,
            useCPP: true,
            mode: cmMode
        });

        // With reference to "fix typing '#' in emacs mode"
        // https://github.com/mattgodbolt/gcc-explorer/pull/131
        cppEditor.setOption("extraKeys", {
            "Alt-F": false
        });
        // cppEditor.on("change", function () {
        //     if ($('.autocompile').hasClass('active')) {
        //         onEditorChange();
        //     }
        // });
        function resize() {
            cppEditor.setSize(domRoot.width(), domRoot.height());
            cppEditor.refresh();
        }

        container.on('resize', resize);
        container.on('open', resize);
        container.setTitle(lang + " source");
    };
});