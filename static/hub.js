define(function (require) {
    "use strict";

    var _ = require('underscore');
    var options = require('options');
    var Editor = require('editor');
    var Compiler = require('compiler');

    function Hub(layout, defaultSrc) {
        this.layout = layout;
        this.defaultSrc = defaultSrc;
        this.compilers = [];
        this.editors = {};
        this.initialised = false;

        var self = this;
        layout.registerComponent('codeEditor', function (container, state) {
            return self.codeEditorFactory(container, state);
        });
        layout.registerComponent('compilerOutput', function (container, state) {
            return self.compilerOutputFactory(container, state);
        });
        layout.init();

        this.initialised = true;

        _.each(this.editors, function (editor) {
            self.onEditorChange(editor);
        });
        this.onEditorListChange();
    }

    Hub.prototype.addCompiler = function (compiler) {
        this.compilers.push(compiler);
    };
    Hub.prototype.removeCompiler = function (compiler) {
        this.compilers = _.without(this.compilers, compiler);
    };

    Hub.prototype.addEditor = function (editor) {
        this.editors[editor.getId()] = editor;
        this.onEditorListChange();
    };
    Hub.prototype.removeEditor = function (editor) {
        delete this.editors[editor.getId()];
        this.onEditorListChange();
    };

    Hub.prototype.onEditorListChange = function () {
        if (!this.initialised) return;
        _.each(this.compilers, function (compiler) {
            compiler.onEditorListChange();
        });
    };

    Hub.prototype.nextEditorId = function () {
        for (var i = 1; i < 100000; ++i) {
            if (!this.editors[i]) return i;
        }
        throw "Ran out of ids!?";
    };

    Hub.prototype.codeEditorFactory = function (container, state) {
        var editor = new Editor(this, state, container, options.language, this.defaultSrc);
        this.addEditor(editor);
        return editor;
    };

    Hub.prototype.compilerOutputFactory = function (container, state) {
        var compiler = new Compiler(this, container, state);
        this.addCompiler(compiler);
        return compiler;
    };

    Hub.prototype.onEditorChange = function (editor) {
        if (!this.initialised) return;
        _.each(this.compilers, function (compiler) {
            compiler.onEditorChange(editor);
        });
    };

    return Hub;
});