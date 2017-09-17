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

define(function(require){
   "use strict";
   
   var FontScale = require('fontscale');
   var options = require('options');
   var vis = require('vis');
   
    require('asm-mode');
    require('selectize');
    
   
   
   function Cfg(hub, container, state){
        this.container = container;
        this.eventHub = hub.createEventHub();
        this.domRoot = container.getElement();
        this.domRoot.html($('#cfg').html());
        this.functions = state.cfgResult;
        this.defaultCfgOuput = {'nodes': [{id: 0, label: 'No Output'}], 'edges': []};
        
        this.fnNames = Object.keys(this.functions);
        this. currentFunc = this.fnNames.length? this.fnNames[0]: "";
             
        this.compilers = {};
        

        var opts = {
            autoResize: true,
            height: '99%',
            width: '99%',
            locale: 'en',
            edges: {
                arrows: {to: {enabled: true}},
                smooth: {enabled: false}
            },
            nodes: {
                font: {'face': 'monospace', 'align': 'left'}
            },
            layout: {
                improvedLayout: true,
                "hierarchical": {
                    "enabled": true,
                    "sortMethod": "directed",
                    "direction": "UD",
                    nodeSpacing: 200,
                    levelSeparation: 200,
                }
            },
            physics: {
                hierarchicalRepulsion: {
                    nodeDistance: 300
                },
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
                                             this.defaultCfgOuput, opts);
        
        this._compilerid = state.id;
        this._compilerName = state.compilerName;
        this._editorid = state.editorid;
        
        this.eventHub.on('compileResult', this.onCompileResult, this);
        this.eventHub.on('compiler', this.onCompiler, this);
        this.eventHub.on('compilerClose', this.onCompilerClose, this);
        //this.eventHub.on('editorChange', this.onEditorChange, this);
        this.eventHub.emit('cfgViewOpened', this._compilerid);
        this.container.on('destroy', function () {
            this.eventHub.emit("cfgViewClosed", this._compilerid, this.cfgVisualiser);
            this.eventHub.unsubscribe();
        }, this);
        
        this.adaptStructure = function(names){
            var options = [];
            
            for(var i = 0; i < names.length; ++i){
                options.push({name:names[i]});
            }
            return options;
        }
        
        
        
        var self = this;
        
        this.select = this.domRoot.find(".function-picker").selectize({
            sortField: 'name',
            valueField: 'name',
            labelField: 'name',
            searchField: ['name'],
            options: this.fnNames.length ? this.adaptStructure(this.fnNames) : [{name:"no functions"}],
            items: this.fnNames.length ? [this.currentFunc] : [{name:"no functions"}]
        }).on('change', function(event){
            self.onFunctionChange(self.functions, this.value);
        });
        this.setTitle();
   }
   
   Cfg.prototype.onCompileResult = function (id, compiler, result) {
        if (this._compilerid == id) {
            if (result.supportCfg) {
                this.functions = result.cfg;
                this.fnNames = Object.keys(this.functions);
                if(!this.fnNames.includes(this.currentFunc))
                    this.currentFunc = this.fnNames[0];
                this.showCfgResults({
                    'nodes': this.functions[this.currentFunc].nodes,
                    'edges': this.functions[this.currentFunc].edges
                });
                this.domRoot.find(".function-picker")[0].selectize.destroy();;
                this.select = this.domRoot.find(".function-picker").selectize({
                    sortField: 'name',
                    valueField: 'name',
                    labelField: 'name',
                    searchField: ['name'],
                    options: this.fnNames.length ? this.adaptStructure(this.fnNames) : [{name: "no functions"}],
                    items: this.fnNames.length ? [this.currentFunc] : [{name: "no functions"}]
                });

                
            } else {
                this.showCfgResults(this.defaultCfgOuput);
            }

        }
    };
   
   Cfg.prototype.setTitle = function () {
          this.container.setTitle(this._compilerName + " Graph Viewer (Editor #" + this._editorid + ", Compiler #" + this._compilerid + ")");
    };


    Cfg.prototype.showCfgResults = function(data) {
        this.cfgVisualiser.setData(data);
    };

    Cfg.prototype.onCompiler = function (id, compiler, options, editorid) {
        if(id == this._compilerid) {
            this._compilerName = compiler.name;
            this._editorid = editorid;
            this.setTitle();
        }
    };

    Cfg.prototype.onCompilerClose = function (id) {
        delete this.compilers[id];
    };
    
    Cfg.prototype.onFunctionChange = function (functions, name) {

        if (functions[name]) {
            this.showCfgResults({
                'nodes': functions[name].nodes,
                'edges': functions[name].edges
            });
        }

        
        //if(functions)
      
    };

    Cfg.prototype.updateState = function () {
    };

    return {
        Cfg: Cfg
    };
});