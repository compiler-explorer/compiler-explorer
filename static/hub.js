define(function (require) {
    "use strict";

    var _ = require('underscore');
    var options = require('options');
    var Editor = require('editor');
    var Compiler = require('compiler');

    function Hub(layout, defaultSrc) {
        this.layout = layout;
        this.defaultSrc = defaultSrc;
        this.ids = {};

        var self = this;
        layout.registerComponent('codeEditor', function (container, state) {
            return self.codeEditorFactory(container, state);
        });
        layout.registerComponent('compilerOutput', function (container, state) {
            return self.compilerOutputFactory(container, state);
        });
        var removeId = function (id) {
            self.ids[id] = false;
        };
        layout.eventHub.on('editorClose', removeId)
        layout.eventHub.on('compilerClose', removeId);
        layout.init();
    }

    Hub.prototype.nextId = function () {
        for (var i = 1; i < 100000; ++i) {
            if (!this.ids[i]) {
                this.ids[i] = true;
                return i;
            }
        }
        throw "Ran out of ids!?";
    };

    Hub.prototype.codeEditorFactory = function (container, state) {
        return new Editor(this, state, container, options.language, this.defaultSrc);
    };

    Hub.prototype.compilerOutputFactory = function (container, state) {
        return new Compiler(this, container, state);
    };

    return Hub;
});