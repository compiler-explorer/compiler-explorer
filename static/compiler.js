define(function (require) {
    "use strict";
    var CodeMirror = require('codemirror');
    var $ = require('jquery');
    require('asm-mode');

    var options = require('options');
    var compilers = options.compilers;

    function Compiler(hub, container, state) {
        var self = this;
        var domRoot = container.getElement();
        domRoot.html($('#compiler').html());

        var outputEditor = CodeMirror.fromTextArea(domRoot.find("textarea")[0], {
            lineNumbers: true,
            mode: "text/x-asm",
            readOnly: true,
            gutters: ['CodeMirror-linenumbers'],
            lineWrapping: true
        });

        function resize() {
            outputEditor.setSize(domRoot.width(), domRoot.height());
            outputEditor.refresh();
        }

        container.on('resize', resize);
        container.on('open', resize);
        container.setTitle("Compiled");
        container.on('close', function () {
            hub.removeCompiler(self);
        });

        this.source = state.source || 1;
        this.sourceEditor = null;
    }

    var debouncedAjax = _.debounce($.ajax, 500);

    Compiler.prototype.compile = function (fromEditor) {
        var self = this;
        if (!this.sourceEditor) return;
        var request = {
            fromEditor: fromEditor,
            source: this.sourceEditor.getSource(),
            compiler: compilers[0].id, // TODO
            options: "-O1", // TODO
            filters: {}
        };
        debouncedAjax({
            type: 'POST',
            url: '/compile',
            dataType: 'json',
            contentType: 'application/json',
            data: JSON.stringify(request),
            success: function (result) {
                self.onCompileResponse(request, result);
            },
            error: function (xhr, e_status, error) {
                console.log("AJAX request failed, reason : " + error);
            },
            cache: false
        });
    };

    Compiler.prototype.onCompileResponse = function (request, result) {
        console.log(request, result);
    };

    Compiler.prototype.onEditorListChange = function () {
        // TODO: if we can't find our source, select none?
        // TODO: Update dropdown of source
    };
    Compiler.prototype.onEditorChange = function (editor) {
        if (editor.getId() == this.source) {
            this.sourceEditor = editor;
            this.compile();
        }
    };

    return Compiler;
});