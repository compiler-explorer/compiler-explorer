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
    var $ = require('jquery');
    var _ = require('underscore');
    var ga = require('analytics').ga;
    var colour = require('colour');
    var Toggles = require('toggles');
    var FontScale = require('fontscale');
    var Promise = require('es6-promise').Promise;
    var Components = require('components');
    var LruCache = require('lru-cache');
    var monaco = require('monaco');
    require('asm-mode');

    require('selectize');

    var options = require('options');
    var compilers = options.compilers;
    var compilersById = {};
    _.forEach(compilers, function (compiler) {
        compilersById[compiler.id] = compiler;
        if (compiler.alias) compilersById[compiler.alias] = compiler;
    });
    var Cache = new LruCache({
        max: 200 * 1024,
        length: function (n) {
            return JSON.stringify(n).length;
        }
    });

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
        this.colours = [];
        this.lastResult = null;
        this.pendingRequestSentAt = 0;
        this.nextRequest = null;
        this.settings = {};

        this.domRoot.find(".compiler-picker").selectize({
            sortField: 'name',
            valueField: 'id',
            labelField: 'name',
            searchField: ['name'],
            options: compilers,
            items: this.compiler ? [this.compiler.id] : []
        }).on('change', function () {
            self.onCompilerChange($(this).val());
        });
        var optionsChange = _.debounce(function () {
            self.onOptionsChange($(this).val());
        }, 800);
        this.domRoot.find(".options")
            .val(this.options)
            .on("change", optionsChange)
            .on("keyup", optionsChange);

        // Hide the binary option if the global options has it disabled.
        this.domRoot.find("[data-bind='binary']").toggle(options.supportsBinary);

        this.outputEditor = monaco.editor.create(this.domRoot.find(".monaco-placeholder")[0], {
            scrollBeyondLastLine: false,
            readOnly: true,
            language: 'asm'
        });

        this.outputEditor.addAction({
            id: 'viewcode',
            label: 'View code',
            keybindings: [monaco.KeyMod.CtrlCmd | monaco.KeyCode.F10],
            keybindingContext: null,
            contextMenuGroupId: 'navigation',
            contextMenuOrder: 1.5,
            run: function(ed) {
                var desiredLine = ed.getPosition().lineNumber - 1;
                self.eventHub.emit('editorSelectLine', self.sourceEditorId, self.assembly[desiredLine].source);
            }
        });

        this.outputEditor.onMouseMove(function (e) {
            if (self.settings.hoverShowSource == true) {
                var desiredLine = e.target.position.lineNumber - 1;
                self.eventHub.emit('editorSelectLine', self.sourceEditorId, self.assembly[desiredLine].source);
            }
        });

        this.fontScale = new FontScale(this.domRoot, state, this.outputEditor);
        this.fontScale.on('change', _.bind(function () {
            this.saveState();
            this.updateFontScale();
        }, this));

        this.filters.on('change', _.bind(this.onFilterChange, this));

        container.on('destroy', function () {
            self.eventHub.unsubscribe();
            self.eventHub.emit('compilerClose', self.id);
        }, this);
        container.on('resize', this.resize, this);
        container.on('shown', this.resize, this);
        container.on('open', function () {
            self.eventHub.emit('compilerOpen', self.id);
            self.updateFontScale();
        });
        this.eventHub.on('editorChange', this.onEditorChange, this);
        this.eventHub.on('editorClose', this.onEditorClose, this);
        this.eventHub.on('colours', this.onColours, this);
        this.eventHub.on('resendCompilation', this.onResendCompilation, this);
        this.eventHub.on('findCompilers', this.sendCompiler, this);
        this.eventHub.on('compilerSelectLine', this.onCompilerSelectLine, this);
        this.eventHub.on('settingsChange', this.onSettingsChange, this);
        this.sendCompiler();
        this.updateCompilerName();
        this.updateButtons();

        var outputConfig = _.bind(function () {
            return Components.getOutput(this.id, this.sourceEditorId);
        }, this);
        this.container.layoutManager.createDragSource(this.domRoot.find(".status").parent(), outputConfig);
        this.domRoot.find(".status").parent().click(_.bind(function () {
            var insertPoint = hub.findParentRowOrColumn(this.container) ||
                this.container.layoutManager.root.contentItems[0];
            insertPoint.addChild(outputConfig());
        }, this));

        function cloneComponent() {
            return {
                type: 'component',
                componentName: 'compiler',
                componentState: self.currentState()
            };
        }

        this.container.layoutManager.createDragSource(
            this.domRoot.find('.btn.add-compiler'), cloneComponent);
        this.domRoot.find('.btn.add-compiler').click(_.bind(function () {
            var insertPoint = hub.findParentRowOrColumn(this.container) ||
                this.container.layoutManager.root.contentItems[0];
            insertPoint.addChild(cloneComponent());
        }, this));

        this.saveState();
    }

    // TODO: need to call resize if either .top-bar or .bottom-bar resizes, which needs some work.
    // Issue manifests if you make a window where one compiler is small enough that the buttons spill onto two lines:
    // reload the page and the bottom-bar is off the bottom until you scroll a tiny bit.
    Compiler.prototype.resize = function () {
        var topBarHeight = this.domRoot.find(".top-bar").outerHeight(true);
        var bottomBarHeight = this.domRoot.find(".bottom-bar").outerHeight(true);
        this.outputEditor.layout({
            width: this.domRoot.width(),
            height: this.domRoot.height() - topBarHeight - bottomBarHeight
        });
    };

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
        var request = {
            source: this.source || "",
            compiler: this.compiler ? this.compiler.id : "",
            options: this.options,
            filters: this.getEffectiveFilters()
        };

        if (!this.compiler) {
            this.onCompileResponse(request, errorResult("<Please select a compiler>"), false);
            return;
        }

        this.sendCompile(request);
    };

    Compiler.prototype.sendCompile = function (request) {
        if (this.pendingRequestSentAt) {
            // If we have a request pending, then just store this request to do once the
            // previous request completes.
            this.nextRequest = request;
            return;
        }
        this.eventHub.emit('compiling', this.id, this.compiler);
        var jsonRequest = JSON.stringify(request);
        var cachedResult = Cache.get(jsonRequest);
        if (cachedResult) {
            this.onCompileResponse(request, cachedResult, true);
            return;
        }

        this.pendingRequestSentAt = Date.now();
        // After a short delay, give the user some indication that we're working on their
        // compilation.
        var progress = setTimeout(_.bind(function () {
            this.setAssembly(fakeAsm("<Compiling...>"));
        }, this), 500);
        $.ajax({
            type: 'POST',
            url: '/api/compiler/' + encodeURIComponent(request.compiler) + '/compile',
            dataType: 'json',
            contentType: 'application/json',
            data: jsonRequest,
            success: _.bind(function (result) {
                clearTimeout(progress);
                if (result.okToCache) {
                    Cache.set(jsonRequest, result);
                }
                this.onCompileResponse(request, result, false);
            }, this),
            error: _.bind(function (xhr, e_status, error) {
                clearTimeout(progress);
                this.onCompileResponse(request, errorResult("<Remote compilation failed: " + error + ">"), false);
            }, this),
            cache: false
        });
    };

    Compiler.prototype.getBinaryForLine = function (line) {
        var obj = this.assembly[line - 1];
        var address = obj.address ? obj.address.toString(16) : "";
        var opcodes = '<div class="opcodes" title="' + (obj.opcodes || []).join(" ") + '">';
        _.each(obj.opcodes, function (op) {
            opcodes += ('<span class="opcode">' + op + '</span>');
        });
        return '<div class="address">' + address + '</div>' + opcodes + '</div>';
    };

    // TODO: use ContentWidgets? OverlayWidgets?
    // Use highlight providers? hover providers? highlight providers?
    Compiler.prototype.setAssembly = function (assembly) {
        this.assembly = assembly;
        this.outputEditor.getModel().setValue(_.pluck(assembly, 'text').join("\n"));
        var addrToAddrDiv = {};
        var decorations = [];
        _.each(this.assembly, _.bind(function (obj, line) {
            var address = obj.address ? obj.address.toString(16) : "";
        //     var div = $("<div class='address cm-number'>" + address + "</div>");
            addrToAddrDiv[address] = {div: "moo", line: line};
        }, this));

        _.each(this.assembly, _.bind(function (obj, line) {
            if (obj.links) {
                _.each(obj.links, _.bind(function (link) {
                    var address = link.to.toString(16);
                    // var thing = $("<a href='#' class='cm-number'>" + address + "</a>");
                    // this.outputEditor.markText(
                    //     from, to, {replacedWith: thing[0], handleMouseEvents: false});
                    var dest = addrToAddrDiv[address];
                    if (dest) {
                        decorations.push({
                            range: new monaco.Range(line, link.offset, line, link.offset + link.length),
                            options: {}
                        });
                        // var editor = this.outputEditor;
                        // thing.hover(function (e) {
                        //     var entered = e.type == "mouseenter";
                        //     dest.div.toggleClass("highlighted", entered);
                        //     thing.toggleClass("highlighted", entered);
                        // });
                        // thing.on('click', function (e) {
                        //     editor.scrollIntoView({line: dest.line, ch: 0}, 30);
                        //     dest.div.toggleClass("highlighted", false);
                        //     thing.toggleClass("highlighted", false);
                        //     e.preventDefault();
                        // });
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

    Compiler.prototype.onCompileResponse = function (request, result, cached) {
        this.lastResult = result;
        var timeTaken = Math.max(0, Date.now() - this.pendingRequestSentAt);
        var wasRealReply = this.pendingRequestSentAt > 0;
        this.pendingRequestSentAt = 0;
        ga('send', {
            hitType: 'event',
            eventCategory: 'Compile',
            eventAction: request.compiler,
            eventLabel: request.options,
            eventValue: cached ? 1 : 0
        });
        ga('send', {
            hitType: 'timing',
            timingCategory: 'Compile',
            timingVar: request.compiler,
            timingValue: timeTaken
        });
        this.setAssembly(result.asm || fakeAsm("<No output>"));
        if (request.filters.binary) {
            this.outputEditor.updateOptions({
                lineNumbers: _.bind(this.getBinaryForLine, this),
                lineNumbersMinChars: 18
            });
        } else {
            this.outputEditor.updateOptions({
                lineNumbers: true,
                lineNumbersMinChars: 5
            });
        }
        var status = this.domRoot.find(".status");
        var allText = _.pluck((result.stdout || []).concat(result.stderr | []), 'text').join("\n");
        var failed = result.code !== 0;
        var warns = !failed && !!allText;
        status.toggleClass('error', failed);
        status.toggleClass('warning', warns);
        status.parent().attr('title', allText);
        var compileTime = this.domRoot.find('.compile-time');
        if (cached) {
            compileTime.text("- cached");
        } else if (wasRealReply) {
            compileTime.text("- " + timeTaken + "ms");
        } else {
            compileTime.text("");
        }
        this.eventHub.emit('compileResult', this.id, this.compiler, result);

        if (this.nextRequest) {
            var next = this.nextRequest;
            this.nextRequest = null;
            this.sendCompile(next);
        }
    };

    Compiler.prototype.expand = function (source) {
        var includeFind = /^\s*#include\s*["<](https?:\/\/[^>"]+)[>"]$/;
        var lines = source.split("\n");
        var promises = [];
        _.each(lines, function (line, lineNumZeroBased) {
            var match = line.match(includeFind);
            if (match) {
                promises.push(new Promise(function (resolve, reject) {
                    var req = $.get(match[1], function (data) {
                        data = '# 1 "' + match[1] + '"\n' + data + '\n\n# ' +
                            (lineNumZeroBased + 1) + ' "<stdin>"\n';

                        lines[lineNumZeroBased] = data;
                        resolve();
                    });
                    req.fail(function () {
                        resolve();
                    });
                }));
            }
        });
        return Promise.all(promises).then(function () {
            return lines.join("\n");
        });
    };

    Compiler.prototype.onEditorChange = function (editor, source) {
        if (editor === this.sourceEditorId) {
            this.expand(source).then(_.bind(function (expanded) {
                this.source = expanded;
                this.compile();
            }, this));
        }
    };

    Compiler.prototype.updateButtons = function () {
        if (!this.compiler) return;
        var filters = this.getEffectiveFilters();
        // We can support intel output if the compiler supports it, or if we're compiling
        // to binary (as we can disassemble it however we like).
        var intelAsm = this.compiler.supportsIntel || filters.binary;
        this.domRoot.find("[data-bind='intel']").toggleClass("disabled", !intelAsm);
        // Disable binary support on compilers that don't work with it.
        this.domRoot.find("[data-bind='binary']")
            .toggleClass("disabled", !this.compiler.supportsBinary);
        // Disable any of the options which don't make sense in binary mode.
        var filtersDisabled = !!filters.binary && !this.compiler.supportsFiltersInBinary;
        this.domRoot.find('.nonbinary').toggleClass("disabled", filtersDisabled);
    };

    Compiler.prototype.onOptionsChange = function (options) {
        this.options = options;
        this.saveState();
        this.compile();
        this.updateButtons();
        this.sendCompiler();
    };

    Compiler.prototype.onCompilerChange = function (value) {
        this.compiler = compilersById[value];
        this.saveState();
        this.compile();
        this.updateButtons();
        this.updateCompilerName();
        this.sendCompiler();
    };

    Compiler.prototype.sendCompiler = function () {
        this.eventHub.emit('compiler', this.id, this.compiler, this.options);
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

    Compiler.prototype.currentState = function () {
        var state = {
            compiler: this.compiler ? this.compiler.id : "",
            options: this.options,
            source: this.editor,
            filters: this.filters.get()  // NB must *not* be effective filters
        };
        this.fontScale.addState(state);
        return state;
    };

    Compiler.prototype.saveState = function () {
        this.container.setState(this.currentState());
    };

    Compiler.prototype.updateFontScale = function () {
        this.eventHub.emit('compilerFontScale', this.id, this.fontScale.scale);
    };

    Compiler.prototype.onColours = function (editor, colours) {
        if (editor == this.sourceEditorId) {
            var asmColours = {};
            _.each(this.assembly, function (x, index) {
                if (x.source) asmColours[index] = colours[x.source - 1];
            });
            this.colours = colour.applyColours(this.outputEditor, asmColours, this.colours);
        }
    };

    Compiler.prototype.updateCompilerName = function () {
        var compilerName = this.compiler ? this.compiler.name : "no compiler set";
        var compilerVersion = this.compiler ? this.compiler.version : "";
        this.container.setTitle("#" + this.sourceEditorId + " with " + compilerName);
        this.domRoot.find(".full-compiler-name").text(compilerVersion);
    };

    Compiler.prototype.onResendCompilation = function (id) {
        if (id == this.id && this.lastResult) {
            this.eventHub.emit('compileResult', this.id, this.compiler, this.lastResult);
        }
    };

    Compiler.prototype.onCompilerSelectLine = function (id, lineNum) {
        if (id === this.id && lineNum != null) {
            this.outputEditor.setSelection({positionColumn: 0, positionLineNumber: lineNum + 1, selectionStartColumn: 0,
                    selectionStartLineNumber: lineNum});
        }
    };

    Compiler.prototype.onSettingsChange = function (newSettings) {
        this.settings = _.clone(newSettings);
    };

    return {
        Compiler: Compiler
    };
});
