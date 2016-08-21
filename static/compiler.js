define(function (require) {
    "use strict";
    var CodeMirror = require('codemirror');
    var $ = require('jquery');
    var _ = require('underscore');
    require('asm-mode');
    require('selectize');

    var options = require('options');
    var compilers = options.compilers;

    function Compiler(hub, container, state) {
        var self = this;
        var domRoot = container.getElement();
        domRoot.html($('#compiler').html());
        var compilerId = compilers[0].id;
        domRoot.find(".compiler").selectize({
            sortField: 'name',
            valueField: 'id',
            labelField: 'name',
            options: compilers,
            items: [compilerId],  // TODO persist and depersist from state
            openOnFocus: true
        }).on('change', function () {
            self.onCompilerChange($(this).val());
        });
        var optionsChange = function (){
            self.onOptionsChange($(this).val());
        };
        domRoot.find(".options")
            .on("change", optionsChange)
            .on("keyup", optionsChange);

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
        this.compiler = compilerId;
        this.options = "";
    }

    var debouncedAjax = _.debounce($.ajax, 500);

    Compiler.prototype.compile = function (fromEditor) {
        var self = this;
        if (!this.sourceEditor || !this.compiler) return;  // TODO blank out the output?
        var request = {
            fromEditor: fromEditor,
            source: this.sourceEditor.getSource(),
            compiler: this.compiler,
            options: this.options,
            filters: {}  // TODO
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
                console.log("AJAX request failed, reason : " + error);  // TODO better error handling
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
        // TODO: persist and depersist
        if (editor.getId() == this.source) {
            this.sourceEditor = editor;
            this.compile();
        }
    };
    Compiler.prototype.onOptionsChange = function (options) {
        // TODO: persist and dep
        this.options = options;
        this.compile();
    };
    Compiler.prototype.onCompilerChange = function (value) {
        this.compiler = value;  // TODO check validity?
        // TODO: persist
        this.compile();
    };

    return Compiler;
});