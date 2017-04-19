// Copyright (c) 2012-2017, Jared Wyles
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

    var FontScale = require('fontscale');
    var monaco = require('monaco');
    var options = require('options');
    var hoverContent = {};

    monaco.languages.registerHoverProvider('cpp', {
	    provideHover: function(model, position) {
            if(hoverContent[position.lineNumber]) {
                return {
                    range: new monaco.Range(1, 1, position.lineNumber, model.getLineMaxColumn(position.lineNumber)),
				    contents: hoverContent[position.lineNumber]
                }
            }
        }
    });

    require('asm-mode');
    require('selectize');

    function Opt(hub, container, state) {
        this.container = container;
        this.eventHub = hub.createEventHub();
        this.domRoot = container.getElement();
        this.domRoot.html($('#opt').html());
        this.compilers = {};
        this.optEditor = monaco.editor.create(this.domRoot.find(".monaco-placeholder")[0], {
            value: state.source || "",
            scrollBeyondLastLine: false,
            language: 'cpp', //we only support cpp for now
            readOnly: true,
            glyphMargin: true,
            quickSuggestions: false,
            fixedOverflowWidgets: true
        });

        this.fontScale = new FontScale(this.domRoot, state, this.optEditor);
        this.fontScale.on('change', _.bind(this.updateState, this));

        this.eventHub.on('compileResult', this.onCompileResult, this);
        this.eventHub.on('compiler', this.onCompiler, this);
        this.eventHub.on('compilerClose', this.onCompilerClose, this);
        this.eventHub.on('editorChange', this.onEditorChange, this);


        this.container.on('destroy', function () {
            this.eventHub.unsubscribe();
            this.optEditor.dispose();
        }, this);
        container.on('resize', this.resize, this);
        container.on('shown', this.resize, this);
        if(state && state.optOutput) {
              this.showOptResults(state.optOutput);
        }
    }

    // TODO: de-dupe with compiler etc
    Opt.prototype.resize = function () {
        var topBarHeight = this.domRoot.find(".top-bar").outerHeight(true);
        this.optEditor.layout({
            width: this.domRoot.width(),
            height: this.domRoot.height() - topBarHeight
        });
    };
    Opt.prototype.onEditorChange = function(id, source) {
        this.optEditor.setValue(source);
    }
    Opt.prototype.onCompileResult = function (id, compiler, result) {
        if(result.hasOptOutput) {
            this.showOptResults(result.optOutput);
        }
    };
    Opt.prototype.getDisplayableOpt = function (optResult) {
       return "**" + optResult.optType + "** - " + optResult.displayString;
    }
    Opt.prototype.showOptResults = function(results) {
        var opt = [],
            hasPassed = false;

        hoverContent = {};
        results = _.groupBy(results, function(x) {
            return x.DebugLoc.Line;
        });
        _.mapObject(results, function(value, key) {
            var linenumber = Number(key);
            var className = value.reduce(function(acc, x) {
                if(acc && acc !== "Analysis" && x.optType !== acc) {
                    return "Mixed";
                } else {
                    return x.optType;
                }
            },"");
            opt.push({
                range: new monaco.Range(linenumber,1,linenumber,1), 
                options: { 
                    isWholeLine: true, 
                    glyphMarginClassName: "opt-decoration." + className.toLowerCase()
                }
            });
            hoverContent[linenumber] = value.map(function(x) {
                return this.getDisplayableOpt(x);
            }, this);
        }, this);
        
        this.optEditor.deltaDecorations([], opt);
    }


    Opt.prototype.onCompiler = function (id, compiler, options, editorId) {
    
    };

    Opt.prototype.onCompilerClose = function (id) {
        delete this.compilers[id];
    };


    Opt.prototype.updateState = function () {
    };

    return {
        Opt: Opt,
        getComponent: function (lhs, rhs) {
            return {
                type: 'component',
                componentName: 'opt',
                componentState: { },
            };
        }
    };
});