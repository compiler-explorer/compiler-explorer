// Copyright (c) 2017, Najjar Chedy
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

var $ = require('jquery');
var vis = require('vis');
var _ = require('underscore');

require('selectize');
require("vis/dist/vis.css");

function Cfg(hub, container, state) {
    this.container = container;
    this.eventHub = hub.createEventHub();
    this.domRoot = container.getElement();
    this.domRoot.html($('#cfg').html());
    this.defaultCfgOutput = {nodes: [{id: 0, shape: 'box', label: 'No Output'}], edges: []};
    this.binaryModeSupport = {
        nodes: [{
            id: 0,
            shape: 'box',
            label: 'Cfg mode cannot be used when the binary filter is set'
        }], edges: []
    };
    // Note that this might be outdated if no functions were present when creating the link, but that's handled
    // by selectize
    this.currentFunc = state.selectedFn || '';
    this.functions = [];
    this.networkOpts = {
        autoResize: true,
        locale: 'en',
        edges: {
            arrows: {to: {enabled: true}},
            smooth: {
                enabled: true,
                type: "dynamic",
                roundness: 1
            },
            physics: true
        },
        nodes: {
            font: {face: 'Consolas, "Liberation Mono", Courier, monospace', align: 'left'}
        },
        layout: {
            hierarchical: {
                enabled: true,
                direction: 'UD',
                nodeSpacing: 100,
                levelSeparation: 150
            }
        },
        physics: {
            enabled: true,
            hierarchicalRepulsion: {
                nodeDistance: 160
            }
        },
        interaction: {
            navigationButtons: false,
            keyboard: {
                enabled: true,
                speed: {x: 10, y: 10, zoom: 0.03},
                bindToWindow: false
            }
        }
    };

    this.cfgVisualiser = new vis.Network(this.domRoot.find('.graph-placeholder')[0],
        this.defaultCfgOutput, this.networkOpts);
    this.domRoot.find('.show-hide-btn').on('click', _.bind(function () {
        this.networkOpts.interaction.navigationButtons = !this.networkOpts.interaction.navigationButtons;
        this.cfgVisualiser.setOptions(this.networkOpts);
    }, this));
    this.domRoot.find('.enable-physics-btn').on('click', _.bind(function () {
        this.networkOpts.physics.enabled = !this.networkOpts.physics.enabled;
        // change only physics.enabled option to preserve current node locations
        this.cfgVisualiser.setOptions({
            physics: {enabled: this.networkOpts.physics.enabled}
        });
    }, this));

    this.compilerId = state.id;
    this._editorid = state.editorid;
    this._binaryFilter = false;

    this.functionPicker = $(this.domRoot).find('.function-picker').selectize({
        sortField: 'name',
        valueField: 'name',
        labelField: 'name',
        searchField: ['name']
    }).on('change', _.bind(function (e) {
        var selectedFn = this.functions[e.target.value];
        if (selectedFn) {
            this.currentFunc = e.target.value;
            this.showCfgResults({
                nodes: selectedFn.nodes,
                edges: selectedFn.edges
            });
            this.cfgVisualiser.selectNodes([selectedFn.nodes[0].id]);
            this.saveState();
        }
    }, this));

    this.eventHub.on('compilerClose', this.onCompilerClose, this);
    this.eventHub.on('compileResult', this.onCompileResult, this);
    this.eventHub.on('compiler', this.onCompiler, this);
    this.eventHub.on('filtersChange', this.onFiltersChange, this);

    this.container.on('destroy', this.close, this);
    this.container.on('resize', this.resize, this);
    this.container.on('shown', this.resize, this);
    this.eventHub.emit('cfgViewOpened', this.compilerId);
    this.eventHub.emit('requestFilters', this.compilerId);
    this.eventHub.emit('requestCompiler', this.compilerId);

    this.adaptStructure = function (names) {
        return _.map(names, function (name) {
            return {name: name};
        });
    };

    this.setTitle();
}

Cfg.prototype.onCompileResult = function (id, compiler, result) {
    if (this.compilerId === id) {
        var functionNames = [];
        if (this.supportsCfg && !$.isEmptyObject(result.cfg)) {
            this.functions = result.cfg;
            functionNames = Object.keys(this.functions);
            if (functionNames.indexOf(this.currentFunc) === -1) {
                this.currentFunc = functionNames[0];
            }
            this.showCfgResults({
                nodes: this.functions[this.currentFunc].nodes,
                edges: this.functions[this.currentFunc].edges
            });
            this.cfgVisualiser.selectNodes([this.functions[this.currentFunc].nodes[0].id]);
        } else {
            // We don't reset the current function here as we would lose the saved one if this happened at the begining
            // (Hint: It *does* happen)
            this.showCfgResults(this._binaryFilter ? this.binaryModeSupport : this.defaultCfgOutput);
        }

        this.functionPicker[0].selectize.clearOptions();
        this.functionPicker[0].selectize.addOption(functionNames.length ? this.adaptStructure(functionNames) : {name: 'The input does not contain functions'});
        this.functionPicker[0].selectize.refreshOptions(false);

        this.functionPicker[0].selectize.clear();
        this.functionPicker[0].selectize.addItem(functionNames.length ? this.currentFunc : 'The input does not contain any function', true);
        this.saveState();
    }
};
Cfg.prototype.onCompiler = function (id, compiler) {
    if (id === this.compilerId) {
        this._compilerName = compiler ? compiler.name : '';
        this.supportsCfg = compiler.supportsCfg;
        this.setTitle();
    }
};

Cfg.prototype.onFiltersChange = function (id, filters) {
    if (this.compilerId === id) {
        this._binaryFilter = filters.binary;
    }
};

Cfg.prototype.resize = function () {
    if (this.cfgVisualiser.canvas) {
        var height = this.domRoot.height() - this.domRoot.find('.top-bar').outerHeight(true);
        this.cfgVisualiser.setSize('100%', height.toString());
        this.cfgVisualiser.redraw();
    }
};

Cfg.prototype.setTitle = function () {
    this.container.setTitle(this._compilerName + ' Graph Viewer (Editor #' + this._editorid + ', Compiler #' + this.compilerId + ')');
};

Cfg.prototype.assignLevels = function (data) {
    var nodes = [];
    var idToIdx = [];
    for (var i in data.nodes) {
        var node = data.nodes[i];
        idToIdx[node.id] = i;
        nodes.push({
            edges: [],
            dagEdges: [],
            index: i,
            id: node.id,
            level: 0,
            state: 0,
            inCount: 0
        });
    }
    var isEdgeValid = function (edge) {
        return edge.from in idToIdx && edge.to in idToIdx;
    };
    data.edges.forEach(function (edge) {
        if (isEdgeValid(edge)) {
            nodes[idToIdx[edge.from]].edges.push(idToIdx[edge.to]);
        }
    });

    var dfs = function (node) { // choose which edges will be back-edges
        node.state = 1;
        node.edges.forEach(function (targetIndex) {
            var target = nodes[targetIndex];
            if (target.state !== 1) {
                if (target.state === 0) {
                    dfs(target);
                }
                node.dagEdges.push(targetIndex);
                target.inCount += 1;
            }
        });
        node.state = 2;
    };
    var markLevels = function (node) {
        node.dagEdges.forEach(function (targetIndex) {
            var target = nodes[targetIndex];
            target.level = Math.max(target.level, node.level + 1);
            if (--target.inCount === 0) {
                markLevels(target);
            }
        });
    };
    nodes.forEach(function (node) {
        if (node.state === 0) {
            dfs(node);
            node.level = 1;
            markLevels(node);
        }
    });
    nodes.forEach(function (node) {
        data.nodes[node.index]['level'] = node.level;
    });
    data.edges.forEach(function (edge) {
        if (isEdgeValid(edge)) {
            var nodeA = nodes[idToIdx[edge.from]];
            var nodeB = nodes[idToIdx[edge.to]];
            if (nodeA.level >= nodeB.level) {
                edge.physics = false;
            } else {
                edge.physics = true;
                var diff = (nodeB.level - nodeA.level);
                edge.length = diff * (200 - 5 * (Math.min(5, diff)));
            }
        } else {
            edge.physics = false;
        }
    });
};

Cfg.prototype.showCfgResults = function (data) {
    this.assignLevels(data);
    this.cfgVisualiser.setData(data);
};


Cfg.prototype.onCompilerClose = function (compilerId) {
    if (this.compilerId === compilerId) {
        // We can't immediately close as an outer loop somewhere in GoldenLayout is iterating over
        // the hierarchy. We can't modify while it's being iterated over.
        this.close();
        _.defer(function (self) {
            self.container.close();
        }, this);
    }
};

Cfg.prototype.close = function () {
    this.eventHub.unsubscribe();
    this.eventHub.emit('cfgViewClosed', this.compilerId);
    this.cfgVisualiser.destroy();
};


Cfg.prototype.saveState = function () {
    this.container.setState(this.currentState());
};

Cfg.prototype.currentState = function () {
    return {
        id: this.compilerId,
        editorid: this._editorid,
        selectedFn: this.currentFunc
    };
};


module.exports = {
    Cfg: Cfg
};

