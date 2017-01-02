// Copyright (c) 2012-2017, Matt Godbolt
//
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without 
// modification, are permitted provided that the following conditions are met:
// 
//     * Redistributions of source code must retain the above copyright notice, 
//       this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright 
//       notice, this list of conditions and the following disclaimer in the 
//       documentation and/or other materials provided with the distribution.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
// POSSIBILITY OF SUCH DAMAGE.

define(function (require) {
    "use strict";

    var _ = require('underscore');
    var options = require('options');
    var editor = require('editor');
    var compiler = require('compiler');
    var output = require('output');
    var Components = require('components');

    function Hub(layout, defaultSrc) {
        this.layout = layout;
        this.defaultSrc = defaultSrc;
        this.ids = {};

        var self = this;
        layout.registerComponent(Components.getEditor().componentName,
            function (container, state) {
                return self.codeEditorFactory(container, state);
            });
        layout.registerComponent(Components.getCompiler().componentName,
            function (container, state) {
                return self.compilerFactory(container, state);
            });
        layout.registerComponent(Components.getOutput().componentName,
            function (container, state) {
                return self.outputFactory(container, state);
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

    Hub.prototype.compilerFactory = function (container, state) {
        return new compiler.Compiler(this, container, state);
    };

    Hub.prototype.outputFactory = function (container, state) {
        return new output.Output(this, container, state);
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

    Hub.prototype.findParentRowOrColumn = function (elem) {
        while (elem) {
            if (elem.isRow || elem.isColumn) return elem;
            elem = elem.parent;
        }
        return elem;
    };

    return Hub;
});