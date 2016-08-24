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
        this.container = container;
        this.domRoot = container.getElement();
        this.domRoot.html($('#compiler').html());

        this.source = state.source || 1;
        this.sourceEditor = null;
        this.compiler = state.compiler || compilers[0].id;
        this.options = state.options || "";
        this.filters = state.filters || {}; // TODO default values

        this.domRoot.find(".compiler").selectize({
            sortField: 'name',
            valueField: 'id',
            labelField: 'name',
            searchField: ['name'],
            options: compilers,
            items: [this.compiler],
            openOnFocus: true
        }).on('change', function () {
            self.onCompilerChange($(this).val());
        });
        var optionsChange = function () {
            self.onOptionsChange($(this).val());
        };
        this.domRoot.find(".options")
            .val(this.options)
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

        this.domRoot.find(".filters .btn input")
            .on('change', function () {
                self.onFilterChange();
            })
            .each(function () {
                $(this).parent().toggleClass('active', !!self.filters[$(this).val()]);
            });

        container.on('resize', resize);
        container.on('open', resize);
        container.setTitle("Compiled");
        container.on('close', function () {
            hub.removeCompiler(self);
        });
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
        if (editor.getId() == this.source) {
            this.sourceEditor = editor;
            this.compile();
        }
    };
    Compiler.prototype.onOptionsChange = function (options) {
        this.options = options;
        this.saveState();
        this.compile();
    };
    Compiler.prototype.onCompilerChange = function (value) {
        this.compiler = value;  // TODO check validity?
        this.saveState();
        this.compile();
    };

    Compiler.prototype.onFilterChange = function () {
        this.filters = {};
        var self = this;
        _.each(this.domRoot.find(".filters .btn.active input"), function (a) {
            self.filters[$(a).val()] = true;
        });
        this.saveState();
        this.compile();
    };

    Compiler.prototype.saveState = function () {
        this.container.setState({
            compiler: this.compiler,
            options: this.options,
            source: this.editor,
            filters: this.filters
        });
    };

    return Compiler;
});