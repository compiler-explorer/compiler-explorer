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
        this.compilers = {};
        
        var opts = {
              autoResize: true,
              locale: 'en',
              edges: {
                arrows: { to: {enabled: true}},
                smooth: { enabled: false}
              },
              nodes: {
                  font: {'face': 'monospace', 'align': 'left'}
              },
              layout: {
                "hierarchical": {
                  "enabled": true,
                  "sortMethod": "directed",
                  "direction": "UD",
                  nodeSpacing: 300,
                  levelSeparation: 200
                }
              },
              physics:  {
                 hierarchicalRepulsion: {
                   nodeDistance: 300
                 }
                }
            };
        
        this.cfgVisualiser = new vis.Network(this.domRoot.find(".graph-placeholder")[0], {'nodes':[{id:0, label:'0'}],'edges':[]}, opts); 
        
        this._compilerid = state.id;
        this._compilerName = state.compilerName;
        this._editorid = state.editorid;
        
        /*this.fontScale = new FontScale(this.domRoot, state, this.cfgVisualiser);
        this.fontScale.on('change', _.bind(this.updateState, this));*/
        
        this.eventHub.on('compileResult', this.onCompileResult, this);
        this.eventHub.on('compiler', this.onCompiler, this);
        this.eventHub.on('compilerClose', this.onCompilerClose, this);
        this.eventHub.emit('cfgViewOpened', this._compilerid);
        this.container.on('destroy', function () {
            this.eventHub.emit("cfgViewClosed", this._compilerid);
            this.eventHub.unsubscribe();
        }, this);
   }
   
   Cfg.prototype.onCompileResult = function (id, compiler, result) {
        if (this._compilerid == id) {
            //if (result.hasCfg) {
                this.showCfgResults({
                                     'nodes': new vis.DataSet(result.cfg.n[0]),
                                     'edges': new vis.DataSet(result.cfg.e[0])
                                    });
            //}

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

    Cfg.prototype.updateState = function () {
    };

    return {
        Cfg: Cfg
    };
});