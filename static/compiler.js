define(function (require) {
    "use strict";
    var CodeMirror = require('codemirror');
    var $ = require('jquery');
    var _ = require('underscore');
    var ga = require('analytics').ga;
    require('asm-mode');
    require('selectize');

    var options = require('options');
    var compilers = options.compilers;

    function Compiler(hub, container, state) {
        var self = this;
        this.domRoot = container.getElement();
        this.domRoot.html($('#compiler').html());
        var compilerId = compilers[0].id;
        this.domRoot.find(".compiler").selectize({
            sortField: 'name',
            valueField: 'id',
            labelField: 'name',
            searchField: ['name'],
            options: compilers,
            items: [compilerId],  // TODO persist and depersist from state
            openOnFocus: true
        }).on('change', function () {
            self.onCompilerChange($(this).val());
        });
        var optionsChange = function () {
            self.onOptionsChange($(this).val());
        };
        this.domRoot.find(".options")
            .on("change", optionsChange)
            .on("keyup", optionsChange);

        var outputEditor = CodeMirror.fromTextArea(this.domRoot.find("textarea")[0], {
            lineNumbers: true,
            mode: "text/x-asm",
            readOnly: true,
            gutters: ['CodeMirror-linenumbers'],
            lineWrapping: true
        });
        this.outputEditor = outputEditor;

        function resize() {
            var topBarHeight = self.domRoot.find(".top-bar").outerHeight(true);
            outputEditor.setSize(self.domRoot.width(), self.domRoot.height() - topBarHeight);
            outputEditor.refresh();
        }

        container.on('resize', resize);
        container.on('open', resize);
        container.setTitle("Compiled");
        container.on('close', function () {
            hub.removeCompiler(self);
        });

        this.domRoot.find(".filters .btn input").on('change', function () {
            self.onFilterChange();
        });

        this.source = state.source || 1;
        this.sourceEditor = null;
        this.compiler = compilerId;
        this.options = "";
        this.filters = {};
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
            filters: this.filters
        };

        request.timestamp = Date.now();
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

    Compiler.prototype.setAssembly = function (assembly) {
        var self = this;
        this.outputEditor.operation(function () {
            self.outputEditor.setValue(_.pluck(assembly, 'text').join("\n"));
        });
    };

    function fakeAsm(text) {
        return [{text: text, source: null, fake: true}];
    }

    Compiler.prototype.onCompileResponse = function (request, result) {
        ga('send', 'event', 'Compile', request.compiler, request.options, result.code);
        ga('send', 'timing', 'Compile', 'Timing', Date.now() - request.timestamp)
        this.setAssembly(result.asm || fakeAsm("[no output]"));
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

    Compiler.prototype.onFilterChange = function () {
        this.filters = {};
        var self = this;
        _.each(this.domRoot.find(".filters .btn.active input"), function (a) {
            self.filters[$(a).val()] = true;
        });
        this.compile();
    };

    return Compiler;
});