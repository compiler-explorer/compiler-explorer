define(function (require) {
    "use strict";

    var _ = require('underscore');
    var options = require('options');
    var editor = require('editor');
    var compiler = require('compiler');

    function Hub(layout, defaultSrc) {
        this.layout = layout;
        this.defaultSrc = defaultSrc;
        this.ids = {};

        var self = this;
        layout.registerComponent(editor.getComponent().componentName,
            function (container, state) {
                return self.codeEditorFactory(container, state);
            });
        layout.registerComponent(compiler.getComponent().componentName,
            function (container, state) {
                return self.compilerOutputFactory(container, state);
            });
        var removeId = function (id) {
            self.ids[id] = false;
        };
        layout.eventHub.on('editorClose', removeId);
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
        return new editor.Editor(this, state, container, options.language, this.defaultSrc);
    };

    Hub.prototype.compilerOutputFactory = function (container, state) {
        return new compiler.Compiler(this, container, state);
    };

    function WrappedEventHub(eventHub) {
        this.eventHub = eventHub;
        this.subscriptions = [];
    }

    WrappedEventHub.prototype.emit = function () {
        this.eventHub.emit.apply(this.eventHub, arguments);
    };
    WrappedEventHub.prototype.on = function (event, callback, context) {
        this.eventHub.on(event, callback, context);
        this.subscriptions.push({evt: event, fn: callback, ctx: context});
    };
    WrappedEventHub.prototype.unsubscribe = function () {
        _.each(this.subscriptions, _.bind(function (obj) {
            this.eventHub.off(obj.evt, obj.fn, obj.ctx);
        }, this));
    };

    Hub.prototype.createEventHub = function () {
        return new WrappedEventHub(this.layout.eventHub);
    };

    return Hub;
});