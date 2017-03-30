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
    var diff = require('diff');

    function Ids() {
        this.used = {};
    }

    Ids.prototype.add = function (id) {
        this.used[id] = true;
    };
    Ids.prototype.remove = function (id) {
        delete this.used[id];
    };
    Ids.prototype.next = function () {
        for (var i = 1; i < 100000; ++i) {
            if (!this.used[i]) {
                this.used[i] = true;
                return i;
            }
        }
        throw "Ran out of ids!?";
    };

    function Hub(layout, defaultSrc) {
        this.layout = layout;
        this.defaultSrc = defaultSrc;
        this.editorIds = new Ids();
        this.compilerIds = new Ids();

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
        layout.registerComponent(diff.getComponent().componentName,
            function (container, state) {
                return self.diffFactory(container, state);
            });

        layout.eventHub.on('editorOpen', function (id) {
            this.editorIds.add(id);
        }, this);
        layout.eventHub.on('editorClose', function (id) {
            this.editorIds.remove(id);
        }, this);
        layout.eventHub.on('compilerOpen', function (id) {
            this.compilerIds.add(id);
        }, this);
        layout.eventHub.on('compilerClose', function (id) {
            this.compilerIds.remove(id);
        }, this);
        layout.init();
    }

    Hub.prototype.nextEditorId = function () {
        return this.editorIds.next();
    };
    Hub.prototype.nextCompilerId = function () {
        return this.compilerIds.next();
    };

    Hub.prototype.codeEditorFactory = function (container, state) {
        // Ensure editors are closable: some older versions had 'isClosable' false.
        // NB there doesn't seem to be a better way to do this than reach into the config and rely on the fact nothing
        // has used it yet.
        container.parent.config.isClosable = true;
        return new editor.Editor(this, state, container, options.language, this.defaultSrc);
    };

    Hub.prototype.compilerFactory = function (container, state) {
        return new compiler.Compiler(this, container, state);
    };

    Hub.prototype.outputFactory = function (container, state) {
        return new output.Output(this, container, state);
    };
    Hub.prototype.diffFactory = function (container, state) {
        return new diff.Diff(this, container, state);
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

    Hub.prototype.addAtRoot = function (newElem) {
        var rootFirstItem = this.layout.root.contentItems[0];
        if (rootFirstItem) {
            if (rootFirstItem.isRow || rootFirstItem.isColumn) {
                rootFirstItem.addChild(newElem);
            } else {
                var newRow = this.layout.createContentItem({type: 'row'}, this.layout.root);
                this.layout.root.replaceChild(rootFirstItem, newRow);
                newRow.addChild(rootFirstItem);
                newRow.addChild(newElem);
            }
        } else {
            this.layout.root.addChild({
                type: 'row',
                content: [newElem]
            });
        }
    };

    return Hub;
});
