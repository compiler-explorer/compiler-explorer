// Copyright (c) 2012-2016, Matt Godbolt
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
    var CodeMirror = require('codemirror');
    var $ = require('jquery');
    var _ = require('underscore');
    var ga = require('analytics').ga;
    var colour = require('colour');
    var Toggles = require('toggles');
    require('asm-mode');
    require('selectize');

    var options = require('options');
    var compilers = options.compilers;
    var compilersById = _.object(_.pluck(compilers, "id"), compilers);

    function Compiler(hub, container, state) {
        var self = this;
        this.container = container;
        this.eventHub = hub.createEventHub();
        this.domRoot = container.getElement();
        this.domRoot.html($('#compiler').html());

        this.id = state.id || hub.nextId();
        this.sourceEditorId = state.source || 1;
        this.compiler = compilersById[state.compiler] || compilersById[options.defaultCompiler];
        this.options = state.options || options.compileOptions;
        this.filters = new Toggles(this.domRoot.find(".filters"), state.filters);
        this.source = "";
        this.assembly = [];
        this.lastRequestRespondedTo = "";

        this.debouncedAjax = _.debounce($.ajax, 250);

        this.domRoot.find(".compiler").selectize({
            sortField: 'name',
            valueField: 'id',
            labelField: 'name',
            searchField: ['name'],
            options: compilers,
            items: [this.compiler.id],
            openOnFocus: true
        }).on('change', function () {
            self.onCompilerChange($(this).val());
        });
        var optionsChange = function () {
            self.onOptionsChange($(this).val());
        };
        this.domRoot.find(".options")
            .val(this.options)
            .on("change", optionsChange)
            .on("keyup", optionsChange);

        // Hide the binary option if the global options has it disabled.
        this.domRoot.find("[data-bind='binary']").toggle(options.supportsBinary);

        var outputEditor = CodeMirror.fromTextArea(this.domRoot.find("textarea")[0], {
            lineNumbers: true,
            mode: "text/x-asm",
            readOnly: true,
            gutters: ['CodeMirror-linenumbers'],
            lineWrapping: true
        });
        this.outputEditor = outputEditor;

        function resize() {
            var topBarHeight = self.domRoot.find(".top-bar").outerHeight(true);
            var bottomBarHeight = self.domRoot.find(".bottom-bar").outerHeight(true);
            outputEditor.setSize(self.domRoot.width(), self.domRoot.height() - topBarHeight - bottomBarHeight);
            outputEditor.refresh();
        }

        this.filters.on('change', _.bind(this.onFilterChange, this));

        container.on('destroy', function () {
            self.eventHub.unsubscribe();
            self.eventHub.emit('compilerClose', self.id);
        }, this);
        container.on('resize', resize);
        container.on('open', function () {
            self.eventHub.emit('compilerOpen', self.id);
        });
        self.eventHub.on('editorChange', this.onEditorChange, this);
        self.eventHub.on('editorClose', this.onEditorClose, this);
        self.eventHub.on('colours', this.onColours, this);
        this.updateCompilerName();
        this.updateButtons();
    }

    // Gets the filters that will actually be used (accounting for issues with binary
    // mode etc).
    Compiler.prototype.getEffectiveFilters = function () {
        if (!this.compiler) return {};
        var filters = this.filters.get();
        if (filters.binary && !this.compiler.supportsBinary) {
            delete filters.binary;
        }
        return filters;
    };

    Compiler.prototype.compile = function () {
        var self = this;
        var request = {
            source: this.source || "",
            compiler: this.compiler ? this.compiler.id : "",
            options: this.options,
            filters: this.getEffectiveFilters()
        };

        if (!this.compiler) {
            this.onCompileResponse(request, errorResult("Please select a compiler"));
            return;
        }

        var cacheableRequest = JSON.stringify(request);
        if (cacheableRequest === this.lastRequestRespondedTo) return;
        // only set the request timestamp after checking cache; else we'll always fetch
        request.timestamp = Date.now();

        this.debouncedAjax({
            type: 'POST',
            url: '/compile',
            dataType: 'json',
            contentType: 'application/json',
            data: JSON.stringify(request),
            success: _.bind(function (result) {
                if (result.okToCache) {
                    this.lastRequestRespondedTo = cacheableRequest;
                } else {
                    this.lastRequestRespondedTo = "";
                }
                self.onCompileResponse(request, result);
            }, this),
            error: _.bind(function (xhr, e_status, error) {
                this.lastRequestRespondedTo = "";
                self.onCompileResponse(request, errorResult("Remote compilation failed: " + error));
            }, this),
            cache: false
        });
    };

    Compiler.prototype.setAssembly = function (assembly) {
        this.assembly = assembly;
        this.outputEditor.setValue(_.pluck(assembly, 'text').join("\n"));

        var addrToAddrDiv = {};
        _.each(this.assembly, _.bind(function (obj, line) {
            var address = obj.address ? obj.address.toString(16) : "";
            var div = $("<div class='address cm-number'>" + address + "</div>");
            addrToAddrDiv[address] = {div: div, line: line};
            this.outputEditor.setGutterMarker(line, 'address', div[0]);
        }, this));

        _.each(this.assembly, _.bind(function (obj, line) {
            var opcodes = $("<div class='opcodes'></div>");
            if (obj.opcodes) {
                var title = [];
                _.each(obj.opcodes, function (op) {
                    var opcodeNum = "00" + op.toString(16);
                    opcodeNum = opcodeNum.substr(opcodeNum.length - 2);
                    title.push(opcodeNum);
                    var opcode = $("<span class='opcode'>" + opcodeNum + "</span>");
                    opcodes.append(opcode);
                });
                opcodes.attr('title', title.join(" "));
            }
            this.outputEditor.setGutterMarker(line, 'opcodes', opcodes[0]);
            if (obj.links) {
                _.each(obj.links, _.bind(function (link) {
                    var from = {line: line, ch: link.offset};
                    var to = {line: line, ch: link.offset + link.length};
                    var address = link.to.toString(16);
                    var thing = $("<a href='#' class='cm-number'>" + address + "</a>");
                    this.outputEditor.markText(
                        from, to, {replacedWith: thing[0], handleMouseEvents: false});
                    var dest = addrToAddrDiv[address];
                    if (dest) {
                        var editor = this.outputEditor;
                        thing.hover(function (e) {
                            var entered = e.type == "mouseenter";
                            dest.div.toggleClass("highlighted", entered);
                            thing.toggleClass("highlighted", entered);
                        });
                        thing.on('click', function (e) {
                            editor.scrollIntoView({line: dest.line, ch: 0}, 30);
                            dest.div.toggleClass("highlighted", false);
                            thing.toggleClass("highlighted", false);
                            e.preventDefault();
                        });
                    }
                }, this));
            }
        }, this));
    };

    function errorResult(text) {
        return {asm: fakeAsm(text), code: -1, stdout: "", stderr: ""};
    }

    function fakeAsm(text) {
        return [{text: text, source: null, fake: true}];
    }

    Compiler.prototype.onCompileResponse = function (request, result) {
        ga('send', 'event', 'Compile', request.compiler, request.options, result.code);
        ga('send', 'timing', 'Compile', 'Timing', Date.now() - request.timestamp);
        this.outputEditor.operation(_.bind(function () {
            this.setAssembly(result.asm || fakeAsm("[no output]"));
            if (request.filters.binary) {
                this.outputEditor.setOption('lineNumbers', false);
                this.outputEditor.setOption('gutters', ['address', 'opcodes']);
            } else {
                this.outputEditor.setOption('lineNumbers', true);
                this.outputEditor.setOption('gutters', ['CodeMirror-linenumbers']);
            }
        }, this));
        var status = this.domRoot.find(".status");
        var allText = result.stdout + result.stderr;
        var failed = result.code !== 0;
        var warns = !failed && !!allText;
        status.toggleClass('error', failed);
        status.toggleClass('warning', warns);
        status.attr('title', allText);
        this.eventHub.emit('compileResult', this.id, this.compiler, result);
    };

    Compiler.prototype.onEditorChange = function (editor, source) {
        if (editor === this.sourceEditorId) {
            this.source = source;
            this.compile();
        }
    };

    Compiler.prototype.updateButtons = function () {
        if (!this.compiler) return;
        var filters = this.getEffectiveFilters();
        // We can support intel output if the compiler supports it, or if we're compiling
        // to binary (as we can disassemble it however we like).
        var intelAsm = this.compiler.intelAsm || filters.binary;
        this.domRoot.find("[data-bind='intel']").toggleClass("disabled", !intelAsm);
        // Disable binary support on compilers that don't work with it.
        this.domRoot.find("[data-bind='binary']")
            .toggleClass("disabled", !this.compiler.supportsBinary);
        // Disable any of the options which don't make sense in binary mode
        this.domRoot.find('.nonbinary').toggleClass("disabled", !!filters.binary);
    };

    Compiler.prototype.onOptionsChange = function (options) {
        this.options = options;
        this.saveState();
        this.compile();
        this.updateButtons();
    };

    Compiler.prototype.onCompilerChange = function (value) {
        this.compiler = compilersById[value];
        this.saveState();
        this.compile();
        this.updateButtons();
        this.updateCompilerName();
    };

    Compiler.prototype.onEditorClose = function (editor) {
        if (editor === this.sourceEditorId) {
            this.container.close();
        }
    };

    Compiler.prototype.onFilterChange = function () {
        this.saveState();
        this.compile();
        this.updateButtons();
    };

    Compiler.prototype.saveState = function () {
        this.container.setState({
            compiler: this.compiler ? this.compiler.id : "",
            options: this.options,
            source: this.editor,
            filters: this.filters.get()  // NB must *not* be effective filters
        });
    };

    Compiler.prototype.onColours = function (editor, colours) {
        if (editor == this.sourceEditorId) {
            var asmColours = {};
            this.assembly.forEach(function (x, index) {
                if (x.source) asmColours[index] = colours[x.source - 1];
            });
            colour.applyColours(this.outputEditor, asmColours);
        }
    };

    Compiler.prototype.updateCompilerName = function () {
        var compilerName = this.compiler ? this.compiler.name : "no compiler set";
        var compilerVersion = this.compiler ? this.compiler.version : "";
        this.container.setTitle("#" + this.sourceEditorId + " with " + compilerName);
        this.domRoot.find(".full-compiler-name").text(compilerVersion);
    };

    return {
        Compiler: Compiler,
        getComponent: function (editorId) {
            return {
                type: 'component',
                componentName: 'compilerOutput',
                componentState: {source: editorId}
            };
        },
        getComponentWith: function (editorId, filters, options, compilerId) {
            return {
                type: 'component',
                componentName: 'compilerOutput',
                componentState: {
                    source: editorId,
                    filters: filters,
                    options: options,
                    compiler: compilerId
                }
            };
        }
    };
});