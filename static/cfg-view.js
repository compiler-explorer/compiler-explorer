// Copyright (c) 2012-2017, Najjar Chedy
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

    var $ = require('jquery');
    var vis = require('vis');
    var _ = require('underscore');
    require('asm-mode');
    require('selectize');

    function Cfg(hub, container, state) {
        this.container = container;
        this.eventHub = hub.createEventHub();
        this.domRoot = container.getElement();
        this.domRoot.html($('#cfg').html());
        this.functions = state.cfgResult;
        this.defaultCfgOutput = {nodes: [{id: 0, label: 'No Output'}], edges: []};
        this.fnNames = this.functions ? Object.keys(this.functions) : [];
        this.currentFunc = this.fnNames.length ? this.fnNames[0] : "";

        this.networkOpts = {
            autoResize: true,
            locale: 'en',
            edges: {
                arrows: {to: {enabled: true}},
                smooth: {enabled: true},
                physics: false
            },
            nodes: {
                font: {face: 'monospace', align: 'left'}
            },
            layout: {
                improvedLayout: true,
                hierarchical: {
                    enabled: true,
                    sortMethod: "directed",
                    direction: "LR", // LR means Upside/down for some reason!
                    nodeSpacing: 100,
                    levelSeparation: 100
                }
            },
            physics: {
                hierarchicalRepulsion: {
                    nodeDistance: 125
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

        this.cfgVisualiser = new vis.Network(this.domRoot.find(".graph-placeholder")[0],
            this.defaultCfgOutput, this.networkOpts);
        this.restButton = this.domRoot.find(".show-hide-btn")
            .on('click', _.bind(function () {
                this.networkOpts.interaction.navigationButtons = !this.networkOpts.interaction.navigationButtons;
                this.cfgVisualiser.setOptions(this.networkOpts);
            }, this));

        this._compilerid = state.id;
        this._compilerName = state.compilerName;
        this._editorid = state.editorid;

        this.eventHub.on('compileResult', this.onCompileResult, this);
        this.eventHub.on('compiler', this.onCompiler, this);
        this.eventHub.emit('cfgViewOpened', this._compilerid);
        this.container.on('destroy', function () {
            this.eventHub.emit("cfgViewClosed", this._compilerid, this.cfgVisualiser);
            this.eventHub.unsubscribe();
        }, this);

        container.on('resize', this.resize, this);
        container.on('shown', this.resize, this);

        this.adaptStructure = function (names) {
            var options = [];

            for (var i = 0; i < names.length; ++i) {
                options.push({name: names[i]});
            }
            return options;
        };

        this.select = this.domRoot.find(".function-picker").selectize({
            sortField: 'name',
            valueField: 'name',
            labelField: 'name',
            searchField: ['name'],
            options: this.fnNames.length ? this.adaptStructure(this.fnNames) : [{name: "The input does not contain any function"}],
            items: this.fnNames.length ? [this.currentFunc] : ["Please select a function"]
        }).on('change', _.bind(function (event) {
            this.onFunctionChange(this.functions, event.target.value);
        }, this));

        this.setTitle();
    }

    Cfg.prototype.onCompileResult = function (id, compiler, result) {
        if (this._compilerid === id) {
            if (result.supportsCfg && !$.isEmptyObject(result.cfg)) {
                this.functions = result.cfg;
                this.fnNames = Object.keys(this.functions);
                if (this.fnNames.indexOf(this.currentFunc) === -1)
                    this.currentFunc = this.fnNames[0];
                this.showCfgResults({
                    'nodes': this.functions[this.currentFunc].nodes,
                    'edges': this.functions[this.currentFunc].edges
                });
                this.cfgVisualiser.selectNodes([this.functions[this.currentFunc].nodes[0].id]);
            } else {
                this.showCfgResults(this.defaultCfgOutput);
                this.currentFunc = "";
                this.fnNames = [];
            }

            this.select[0].selectize.destroy();
            this.select = this.domRoot.find(".function-picker").selectize({
                sortField: 'name',
                valueField: 'name',
                labelField: 'name',
                searchField: ['name'],
                options: this.fnNames.length ? this.adaptStructure(this.fnNames) : [{name: "The input does not contain any function"}],
                items: this.fnNames.length ? [this.currentFunc] : ["Please select a function"]
            }).on('change', _.bind(function (event) {
                this.onFunctionChange(this.functions, event.target.value);
            }, this));
        }
    };

    Cfg.prototype.setTitle = function () {
        this.container.setTitle(this._compilerName + " Graph Viewer (Editor #" + this._editorid + ", Compiler #" + this._compilerid + ")");
    };

    Cfg.prototype.showCfgResults = function (data) {
        this.cfgVisualiser.setData(data);
    };

    Cfg.prototype.onCompiler = function (id, compiler, options, editorid) {
        if (id === this._compilerid) {
            this._compilerName = compiler.name;
            this._editorid = editorid;
            this.setTitle();
        }
    };

    Cfg.prototype.onFunctionChange = function (functions, name) {
        if (functions[name]) {
            this.currentFunc = name;
            this.showCfgResults({
                'nodes': functions[name].nodes,
                'edges': functions[name].edges
            });
            this.cfgVisualiser.selectNodes([functions[name].nodes[0].id]);
        }
    };

    Cfg.prototype.resize = function () {
        var height = this.domRoot.height() - this.domRoot.find(".top-bar").outerHeight(true);
        this.cfgVisualiser.setSize('100%', height.toString());
        this.cfgVisualiser.redraw();
    };

    return {
        Cfg: Cfg
    };
});

