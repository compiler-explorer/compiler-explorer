// Copyright (c) 2016, Compiler Explorer Authors
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

'use strict';

var _ = require('underscore');
var Sentry = require('@sentry/browser');
var editor = require('./panes/editor');
var compiler = require('./panes/compiler');
var executor = require('./panes/executor');
var output = require('./panes/output');
var tool = require('./panes/tool');
var Components = require('components');
var diff = require('./panes/diff');
var optView = require('./panes/opt-view');
var astView = require('./panes/ast-view');
var irView = require('./panes/ir-view');
var gccDumpView = require('./panes/gccdump-view');
var cfgView = require('./panes/cfg-view');
var conformanceView = require('./panes/conformance-view');
var CompilerService = require('compiler-service');

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
    throw 'Ran out of ids!?';
};

function Hub(layout, subLangId, defaultLangId) {
    this.layout = layout;
    this.editorIds = new Ids();
    this.compilerIds = new Ids();
    this.executorIds = new Ids();
    this.compilerService = new CompilerService(layout.eventHub);
    this.deferred = true;
    this.deferredEmissions = [];
    this.lastOpenedLangId = null;
    this.subdomainLangId = subLangId || undefined;
    this.defaultLangId = defaultLangId;

    // FIXME
    // We can't avoid this self as _ is undefined at this point
    var self = this;

    layout.registerComponent(Components.getEditor().componentName,
        function (container, state) {
            return self.codeEditorFactory(container, state);
        });
    layout.registerComponent(Components.getCompiler().componentName,
        function (container, state) {
            return self.compilerFactory(container, state);
        });
    layout.registerComponent(Components.getExecutor().componentName,
        function (container, state) {
            return self.executorFactory(container, state);
        });
    layout.registerComponent(Components.getOutput().componentName,
        function (container, state) {
            return self.outputFactory(container, state);
        });
    layout.registerComponent(Components.getToolViewWith().componentName,
        function (container, state) {
            return self.toolFactory(container, state);
        });
    layout.registerComponent(diff.getComponent().componentName,
        function (container, state) {
            return self.diffFactory(container, state);
        });
    layout.registerComponent(Components.getOptView().componentName,
        function (container, state) {
            return self.optViewFactory(container, state);
        });
    layout.registerComponent(Components.getAstView().componentName,
        function (container, state) {
            return self.astViewFactory(container, state);
        });
    layout.registerComponent(Components.getIrView().componentName,
        function (container, state) {
            return self.irViewFactory(container, state);
        });
    layout.registerComponent(Components.getGccDumpView().componentName,
        function (container, state) {
            return self.gccDumpViewFactory(container, state);
        });
    layout.registerComponent(Components.getCfgView().componentName,
        function (container, state) {
            return self.cfgViewFactory(container, state);
        });
    layout.registerComponent(Components.getConformanceView().componentName,
        function (container, state) {
            return self.confomanceFactory(container, state);
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
    layout.eventHub.on('executorOpen', function (id) {
        this.executorIds.add(id);
    }, this);
    layout.eventHub.on('executorClose', function (id) {
        this.executorIds.remove(id);
    }, this);
    layout.eventHub.on('languageChange', function (editorId, langId) {
        this.lastOpenedLangId = langId;
    }, this);
    layout.init();
    this.undefer();
    layout.eventHub.emit('initialised');
}

Hub.prototype.undefer = function () {
    this.deferred = false;
    var eventHub = this.layout.eventHub;
    _.each(this.deferredEmissions, function (args) {
        eventHub.emit.apply(eventHub, args);
    });
    this.deferredEmissions = [];
};

Hub.prototype.nextEditorId = function () {
    return this.editorIds.next();
};

Hub.prototype.nextCompilerId = function () {
    return this.compilerIds.next();
};

Hub.prototype.nextExecutorId = function () {
    return this.executorIds.next();
};

Hub.prototype.codeEditorFactory = function (container, state) {
    // Ensure editors are closable: some older versions had 'isClosable' false.
    // NB there doesn't seem to be a better way to do this than reach into the config and rely on the fact nothing
    // has used it yet.
    container.parent.config.isClosable = true;
    return new editor.Editor(this, state, container);
};

Hub.prototype.compilerFactory = function (container, state) {
    return new compiler.Compiler(this, container, state);
};

Hub.prototype.executorFactory = function (container, state) {
    return new executor.Executor(this, container, state);
};

Hub.prototype.outputFactory = function (container, state) {
    return new output.Output(this, container, state);
};

Hub.prototype.toolFactory = function (container, state) {
    return new tool.Tool(this, container, state);
};

Hub.prototype.diffFactory = function (container, state) {
    return new diff.Diff(this, container, state);
};

Hub.prototype.optViewFactory = function (container, state) {
    return new optView.Opt(this, container, state);
};

Hub.prototype.astViewFactory = function (container, state) {
    return new astView.Ast(this, container, state);
};

Hub.prototype.irViewFactory = function (container, state) {
    return new irView.Ir(this, container, state);
};

Hub.prototype.gccDumpViewFactory = function (container, state) {
    return new gccDumpView.GccDump(this, container, state);
};

Hub.prototype.cfgViewFactory = function (container, state) {
    return new cfgView.Cfg(this, container, state);
};

Hub.prototype.confomanceFactory = function (container, state) {
    return new conformanceView.Conformance(this, container, state);
};

function WrappedEventHub(hub, eventHub) {
    this.hub = hub;
    this.eventHub = eventHub;
    this.subscriptions = [];
}

WrappedEventHub.prototype.emit = function () {
    // Events are deferred during initialisation to allow all the components to install their listeners before
    // all the emits are done. This fixes some ordering issues.
    if (this.hub.deferred) {
        this.hub.deferredEmissions.push(arguments);
    } else {
        this.eventHub.emit.apply(this.eventHub, arguments);
    }
};

WrappedEventHub.prototype.on = function (event, callback, context) {
    this.eventHub.on(event, callback, context);
    this.subscriptions.push({evt: event, fn: callback, ctx: context});
};

WrappedEventHub.prototype.unsubscribe = function () {
    _.each(this.subscriptions, _.bind(function (obj) {
        try {
            this.eventHub.off(obj.evt, obj.fn, obj.ctx);
        } catch (e) {
            Sentry.captureMessage('Can not unsubscribe from ' + obj.evt.toString());
            Sentry.captureException(e);
        }
    }, this));
    this.subscriptions = [];
};

WrappedEventHub.prototype.mediateDependentCalls = function (dependent, dependency) {
    var dependencyExecuted = false;
    var lastDependentArgs = null;
    var dependencyProxy = function () {
        dependency.apply(this, arguments);
        dependencyExecuted = true;
        if (lastDependentArgs) {
            dependent.apply(this, lastDependentArgs);
            lastDependentArgs = null;
        }
    };
    var dependentProxy = function () {
        if (dependencyExecuted) {
            dependent.apply(this, arguments);
        } else {
            lastDependentArgs = arguments;
        }
    };
    return {dependencyProxy: dependencyProxy,
        dependentProxy: dependentProxy};
};

Hub.prototype.createEventHub = function () {
    return new WrappedEventHub(this, this.layout.eventHub);
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
            content: [newElem],
        });
    }
};

module.exports = Hub;
