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
    var options = require('options');
    var monaco = require('monaco');
    var Alert = require('alert');
    var bigInt = require('big-integer');
    require('asm-mode');

    require('selectize');

    var OpcodeCache = new LruCache({
        max: 64 * 1024,
        length: function (n) {
            return JSON.stringify(n).length;
        }
    });

    function Compiler(hub, container, state) {
        var self = this;
        this.container = container;
        this.eventHub = hub.createEventHub();
        this.compilerService = hub.compilerService;
        this.domRoot = container.getElement();
        this.domRoot.html($('#compiler').html());

        this.id = state.id || hub.nextCompilerId();
        this.sourceEditorId = state.source || 1;
        this.compiler = this.compilerService.getCompilerById(state.compiler) ||
            this.compilerService.getCompilerById(options.defaultCompiler);
        this.options = state.options || options.compileOptions;
        this.filters = new Toggles(this.domRoot.find(".filters"), state.filters);
        this.source = "";
        this.assembly = [];
        this.colours = [];
        this.lastResult = {};
        this.pendingRequestSentAt = 0;
        this.nextRequest = null;
        this.settings = {};
        this.wantOptInfo = state.wantOptInfo;
        this.decorations = {};
        this.prevDecorations = [];
        this.optButton = this.domRoot.find('.btn.view-optimization');
        this.astButton = this.domRoot.find('.btn.view-ast');

        this.linkedFadeTimeoutId = -1;

        this.domRoot.find(".compiler-picker").selectize({
            sortField: 'name',
            valueField: 'id',
            labelField: 'name',
            searchField: ['name'],
            options: options.compilers,
            items: this.compiler ? [this.compiler.id] : []
        }).on('change', function () {
            ga('send', {
                hitType: 'event',
                eventCategory: 'SelectCompiler',
                eventAction: $(this).val()
            });
            self.onCompilerChange($(this).val());
        });
        var optionsChange = _.debounce(function () {
            self.onOptionsChange($(this).val());
        }, 800);
        this.optionsField = this.domRoot.find(".options");
        this.optionsField 
            .val(this.options)
            .on("change", optionsChange)
            .on("keyup", optionsChange);

        // Hide the binary option if the global options has it disabled.
        this.domRoot.find("[data-bind='binary']").toggle(options.supportsBinary);
        this.domRoot.find("[data-bind='execute']").toggle(options.supportsExecute);

        this.outputEditor = monaco.editor.create(this.domRoot.find(".monaco-placeholder")[0], {
            scrollBeyondLastLine: false,
            readOnly: true,
            language: 'asm',
            fontFamily: 'Fira Mono',
            glyphMargin: !options.embedded,
            fixedOverflowWidgets: true,
            minimap: {
                maxColumn: 80
            },
            lineNumbersMinChars: options.embedded ? 1 : 5
        });
        this.outputEditor.addAction({
            id: 'viewsource',
            label: 'Scroll to source',
            keybindings: [monaco.KeyMod.CtrlCmd | monaco.KeyCode.F10],
            keybindingContext: null,
            contextMenuGroupId: 'navigation',
            contextMenuOrder: 1.5,
            run: function (ed) {
                var desiredLine = ed.getPosition().lineNumber - 1;
                var source = self.assembly[desiredLine].source;
                if (source.file === null) {
                    // a null file means it was the user's source
                    self.eventHub.emit('editorSetDecoration', self.sourceEditorId, source.line, true);
                } else {
                    // TODO: some indication this asm statement came from elsewhere
                    self.eventHub.emit('editorSetDecoration', self.sourceEditorId, -1, false);
                }
            }
        });

        this.outputEditor.addAction({
            id: 'viewasmdoc',
            label: 'View asm doc',
            keybindings: [monaco.KeyMod.CtrlCmd | monaco.KeyCode.F8],
            keybindingContext: null,
            contextMenuGroupId: 'help',
            contextMenuOrder: 1.5,
            run: _.bind(this.onAsmToolTip, this)
        });

        this.outputEditor.addAction({
            id: 'toggleColourisation',
            label: 'Toggle colourisation',
            keybindings: [monaco.KeyMod.CtrlCmd | monaco.KeyMod.Shift | monaco.KeyCode.F1],
            keybindingContext: null,
            run: _.bind(function () {
                this.eventHub.emit('modifySettings', {
                    colouriseAsm: !this.settings.colouriseAsm
                });
            }, this)
        });

        function clearEditorsLinkedLines() {
            self.eventHub.emit('editorSetDecoration', self.sourceEditorId, -1, false);
        }

        this.outputEditor.onMouseMove(function (e) {
            self.mouseMoveThrottledFunction(e);
            if (self.linkedFadeTimeoutId !== -1) {
                clearTimeout(self.linkedFadeTimeoutId);
                self.linkedFadeTimeoutId = -1;
            }
        });

        this.mouseMoveThrottledFunction = _.throttle(_.bind(this.onMouseMove, this), 250);

        this.outputEditor.onMouseLeave(function () {
            self.linkedFadeTimeoutId = setTimeout(function () {
                clearEditorsLinkedLines();
                self.linkedFadeTimeoutId = -1;
            }, 5000);
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
            self.outputEditor.dispose();
        }, this);
        container.on('resize', this.resize, this);
        container.on('shown', this.resize, this);
        container.on('open', function () {
            self.eventHub.emit('compilerOpen', self.id, self.sourceEditorId);
            self.updateFontScale();
        });
        this.eventHub.on('editorChange', this.onEditorChange, this);
        this.eventHub.on('editorClose', this.onEditorClose, this);
        this.eventHub.on('colours', this.onColours, this);
        this.eventHub.on('resendCompilation', this.onResendCompilation, this);
        this.eventHub.on('findCompilers', this.sendCompiler, this);
        this.eventHub.on('compilerSetDecorations', this.onCompilerSetDecorations, this);
        this.eventHub.on('settingsChange', this.onSettingsChange, this);
        this.eventHub.on('optViewClosed', this.onOptViewClosed, this);
        this.eventHub.on('astViewOpened', this.onAstViewOpened, this);
        this.eventHub.on('astViewClosed', this.onAstViewClosed, this);
        this.eventHub.on('optViewOpened', this.onOptViewOpened, this);
        this.eventHub.on('resize', this.resize, this);
        this.eventHub.emit('requestSettings');
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

        function createOptView() {
            return Components.getOptViewWith(self.id, self.source, self.lastResult.optOutput, self.getCompilerName(), self.sourceEditorId);
        }

        function createAstView() {
            return Components.getAstViewWith(self.id, self.source, self.lastResult.astOutput, self.getCompilerName(), self.sourceEditorId);
        }

        this.container.layoutManager.createDragSource(
            this.domRoot.find('.btn.add-compiler'), cloneComponent);

        this.domRoot.find('.btn.add-compiler').click(_.bind(function () {
            var insertPoint = hub.findParentRowOrColumn(this.container) ||
                this.container.layoutManager.root.contentItems[0];
            insertPoint.addChild(cloneComponent());
        }, this));

        this.container.layoutManager.createDragSource(
            this.optButton, createOptView.bind(this));

        this.optButton.click(_.bind(function () {
            this.wantOptInfo = true;
            var insertPoint = hub.findParentRowOrColumn(this.container) ||
                this.container.layoutManager.root.contentItems[0];
            insertPoint.addChild(createOptView());
            this.optButton.prop("disabled", true);
            this.compile();
        }, this));

        this.container.layoutManager.createDragSource(
            this.astButton, createAstView.bind(this));

        this.astButton.click(_.bind(function () {
            var insertPoint = hub.findParentRowOrColumn(this.container) ||
                this.container.layoutManager.root.contentItems[0];
            insertPoint.addChild(createAstView());
            this.compile();
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
        if (filters.execute && !this.compiler.supportsExecute) {
            delete filters.execute;
        }
        return filters;
    };


    Compiler.prototype.compile = function () {
        var options = {
            userArguments: this.options,
            compilerOptions: { produceAst: this.astViewOpen, produceOptInfo: this.wantOptInfo },
            filters: this.getEffectiveFilters()
        };
        this.compilerService.expand(this.source).then(_.bind(function (expanded) {
            var request = {
                source: expanded || "",
                compiler: this.compiler ? this.compiler.id : "",
                options: options
            };
            if (!this.compiler) {
                this.onCompileResponse(request, errorResult("<Please select a compiler>"), false);
            } else {
                this.sendCompile(request);
            }
        }, this));
    };

    Compiler.prototype.sendCompile = function (request) {
        var onCompilerResponse = _.bind(this.onCompileResponse, this);

        if (this.pendingRequestSentAt) {
            // If we have a request pending, then just store this request to do once the
            // previous request completes.
            this.nextRequest = request;
            return;
        }
        this.eventHub.emit('compiling', this.id, this.compiler);
        this.pendingRequestSentAt = Date.now();
        // After a short delay, give the user some indication that we're working on their
        // compilation.
        var progress = setTimeout(_.bind(function () {
            this.setAssembly(fakeAsm("<Compiling...>"));
        }, this), 500);
        this.compilerService.submit(request)
            .then(function (x) {
                clearTimeout(progress);
                onCompilerResponse(request, x.result, x.localCacheHit);
            })
            .catch(function (x) {
                clearTimeout(progress);
                onCompilerResponse(request, errorResult("<Remote compilation failed: " + x.error + ">"), false);
            });
    };

    Compiler.prototype.getBinaryForLine = function (line) {
        var obj = this.assembly[line - 1];
        if (!obj) return '<div class="address">????</div><div class="opcodes"><span class="opcode">????</span></div>';
        var address = obj.address ? obj.address.toString(16) : "";
        var opcodes = '<div class="opcodes" title="' + (obj.opcodes || []).join(" ") + '">';
        _.each(obj.opcodes, function (op) {
            opcodes += ('<span class="opcode">' + op + '</span>');
        });
        return '<div class="binary-side"><div class="address">' + address + '</div>' + opcodes + '</div></div>';
    };

    // TODO: use ContentWidgets? OverlayWidgets?
    // Use highlight providers? hover providers? highlight providers?
    Compiler.prototype.setAssembly = function (assembly) {
        this.assembly = assembly;
        if (this.outputEditor.getModel()) {
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
        }
    };

    function errorResult(text) {
        return {asm: fakeAsm(text), code: -1, stdout: "", stderr: ""};
    }

    function fakeAsm(text) {
        return [{text: text, source: null, fake: true}];
    }

    Compiler.prototype.onCompileResponse = function (request, result, cached) {
        // Delete trailing empty lines
        if ($.isArray(result.asm)) {
            var cutCount = 0;
            for (var i = result.asm.length - 1; i >= 0; i--) {
                if (result.asm[i].text) {
                    break;
                }
                cutCount++;
            }
            result.asm.splice(result.asm.length - cutCount, cutCount);
        }
        this.lastResult = result;
        var timeTaken = Math.max(0, Date.now() - this.pendingRequestSentAt);
        var wasRealReply = this.pendingRequestSentAt > 0;
        this.pendingRequestSentAt = 0;
        ga('send', {
            hitType: 'event',
            eventCategory: 'Compile',
            eventAction: request.compiler,
            eventLabel: request.options.userArguments,
            eventValue: cached ? 1 : 0
        });
        ga('send', {
            hitType: 'timing',
            timingCategory: 'Compile',
            timingVar: request.compiler,
            timingValue: timeTaken
        });

        this.setAssembly(result.asm || fakeAsm("<No output>"));
        if (request.options.filters.binary) {
            this.outputEditor.updateOptions({
                lineNumbers: _.bind(this.getBinaryForLine, this),
                // Until something comes out of Microsoft/monaco-editor/issues/515, this gets clamped to ten
                lineNumbersMinChars: 25,
                glyphMargin: false
            });
        } else {
            this.outputEditor.updateOptions({
                lineNumbers: true,
                lineNumbersMinChars: options.embedded ? 1 : 5,
                glyphMargin: true
            });
        }
        var status = this.domRoot.find(".status");
        var allText = _.pluck((result.stdout || []).concat(result.stderr || []), 'text').join("\n");
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

    Compiler.prototype.onEditorChange = function (editor, source) {
        if (editor === this.sourceEditorId) {
            this.source = source;
            this.compile();
        }
    };

    Compiler.prototype.onOptViewClosed = function (id) {
        if (this.id == id) {
            this.wantOptInfo = false;
            this.optButton.prop("disabled", false);
        }
    };

    Compiler.prototype.onAstViewOpened = function (id) {
        if (this.id == id) {
            this.astButton.prop("disabled", true);
            this.astViewOpen = true;
        }
    };

    Compiler.prototype.onAstViewClosed = function (id) {
        if (this.id == id) {
            this.astButton.prop('disabled', false);
            this.astViewOpen = false;
        }
    };
    Compiler.prototype.onOptViewOpened = function (id) {
        if (this.id == id) {
            this.optButton.prop("disabled", true);
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
        this.domRoot.find("[data-bind='execute']")
            .toggleClass("disabled", !this.compiler.supportsExecute);
        // Disable any of the options which don't make sense in binary mode.
        var filtersDisabled = !!filters.binary && !this.compiler.supportsFiltersInBinary;
        this.domRoot.find('.nonbinary').toggleClass("disabled", filtersDisabled);
        this.optButton.prop("disabled", !this.compiler.supportsOptOutput);
    };

    Compiler.prototype.onOptionsChange = function (options) {
        this.options = options;
        this.saveState();
        this.compile();
        this.updateButtons();
        this.sendCompiler();
    };

    Compiler.prototype.onCompilerChange = function (value) {
        this.compiler = this.compilerService.getCompilerById(value);
        this.saveState();
        this.compile();
        this.updateButtons();
        this.updateCompilerName();
        this.sendCompiler();
    };

    Compiler.prototype.sendCompiler = function () {
        this.eventHub.emit('compiler', this.id, this.compiler, this.options, this.sourceEditorId);
    };

    Compiler.prototype.onEditorClose = function (editor) {
        if (editor === this.sourceEditorId) {
            // We can't immediately close as an outer loop somewhere in GoldenLayout is iterating over
            // the hierarchy. We can't modify while it's being iterated over.
            _.defer(function (self) {
                self.container.close();
            }, this);
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
            source: this.sourceEditorId,
            options: this.options,
            // NB must *not* be effective filters
            filters: this.filters.get(),
            wantOptInfo: this.wantOptInfo
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

    Compiler.prototype.onColours = function (editor, colours, scheme) {
        if (editor == this.sourceEditorId) {
            var asmColours = {};
            _.each(this.assembly, function (x, index) {
                if (x.source && x.source.file === null) {
                    asmColours[index] = colours[x.source.line - 1];
                }
            });
            this.colours = colour.applyColours(this.outputEditor, asmColours, scheme, this.colours);
        }
    };

    Compiler.prototype.getCompilerName = function () {
        return this.compiler ? this.compiler.name : "no compiler set";
    };

    Compiler.prototype.updateCompilerName = function () {
        var compilerName = this.getCompilerName();
        var compilerVersion = this.compiler ? this.compiler.version : "";
        this.container.setTitle(compilerName + " (Editor #" + this.sourceEditorId + ", Compiler #" + this.id + ")");
        this.domRoot.find(".full-compiler-name").text(compilerVersion);
    };

    Compiler.prototype.onResendCompilation = function (id) {
        if (id == this.id && !$.isEmptyObject(this.lastResult)) {
            this.eventHub.emit('compileResult', this.id, this.compiler, this.lastResult);
        }
    };

    Compiler.prototype.updateDecorations = function () {
        this.prevDecorations = this.outputEditor.deltaDecorations(
            this.prevDecorations, _.flatten(_.values(this.decorations), true));
    };

    Compiler.prototype.onCompilerSetDecorations = function (id, lineNums, revealLine) {
        if (id == this.id) {
            if (revealLine && lineNums[0])
                this.outputEditor.revealLineInCenter(lineNums[0]);
            this.decorations.linkedCode = _.map(lineNums, function (line) {
                return {
                    range: new monaco.Range(line, 1, line, 1),
                    options: {
                        isWholeLine: true,
                        linesDecorationsClassName: 'linked-code-decoration-margin',
                        inlineClassName: 'linked-code-decoration-inline'
                    }
                };
            });
            this.updateDecorations();
        }
    };

    Compiler.prototype.onSettingsChange = function (newSettings) {
        var before = this.settings;
        this.settings = _.clone(newSettings);
        if (!before.lastHoverShowSource && this.settings.hoverShowSource) {
            this.onCompilerSetDecorations(this.id, []);
        }
        this.outputEditor.updateOptions({
            contextmenu: this.settings.useCustomContextMenu,
            minimap: {
                enabled: this.settings.showMinimap && !options.embedded
            }
        });
    };

    var hexLike = /^(#?[$]|0x)([0-9a-fA-F]+)$/;
    var hexLike2 = /^(#?)([0-9a-fA-F]+)H$/;
    var decimalLike = /^(#?)(-?[0-9]+)$/;

    function getNumericToolTip(value) {
        var match = hexLike.exec(value) || hexLike2.exec(value);
        if (match) {
            return value + ' = ' + bigInt(match[2], 16).toString(10);
        }
        match = decimalLike.exec(value);
        if (match) {
            var asBig = bigInt(match[2]);
            if (asBig.isNegative()) {
                asBig = bigInt("ffffffffffffffff", 16).and(asBig);
            }
            return value + ' = 0x' + asBig.toString(16).toUpperCase();
        }

        return null;
    }

    var opcodeLike = /^[a-zA-Z][a-zA-Z0-9_.]+$/; // at least two characters
    var getAsmInfo = function (opcode) {
        if (!opcodeLike.exec(opcode)) {
            return Promise.resolve(null);
        }
        var cacheName = "asm/" + opcode;
        var cached = OpcodeCache.get(cacheName);
        if (cached) {
            return Promise.resolve(cached.found ? cached.result : null);
        }
        return new Promise(function (resolve, reject) {
            $.ajax({
                type: 'GET',
                url: 'api/asm/' + opcode,
                dataType: 'json',  // Expected,
                contentType: 'text/plain',  // Sent
                success: function (result) {
                    OpcodeCache.set(cacheName, result);
                    resolve(result.found ? result.result : null);
                },
                error: function (result) {
                    reject(result);
                },
                cache: true
            });
        });
    };

    Compiler.prototype.onMouseMove = function (e) {
        if (e === null || e.target === null || e.target.position === null ||
            e.target.position.lineNumber > this.outputEditor.getModel().getLineCount()) return;
        if (this.settings.hoverShowSource === true && this.assembly) {
            var desiredLine = e.target.position.lineNumber - 1;
            if (this.assembly[desiredLine]) {
                // We check that we actually have something to show at this point!
                this.eventHub.emit('editorSetDecoration', this.sourceEditorId, this.assembly[desiredLine].source, false);
            }
        }
        var currentWord = this.outputEditor.getModel().getWordAtPosition(e.target.position);
        if (currentWord && currentWord.word) {
            var word = currentWord.word;
            currentWord.range = new monaco.Range(e.target.position.lineNumber, currentWord.startColumn, e.target.position.lineNumber, currentWord.endColumn);
            // Hacky workaround to check for negative numbers. c.f. https://github.com/mattgodbolt/compiler-explorer/issues/434
            var lineContent = this.outputEditor.getModel().getLineContent(e.target.position.lineNumber);
            if (lineContent[currentWord.startColumn - 2] === '-') {
                word = '-' + word;
                currentWord.range.startColumn -= 1;
            }
            var numericToolTip = getNumericToolTip(word);
            if (numericToolTip) {
                this.decorations.numericToolTip = {
                    range: currentWord.range,
                    options: {isWholeLine: false, hoverMessage: ['`' + numericToolTip + '`']}
                };
                this.updateDecorations();
            }

            if (this.settings.hoverShowAsmDoc === true) {
                getAsmInfo(currentWord.word).then(_.bind(function (response) {
                    if (response) {
                        this.decorations.asmToolTip = {
                            range: currentWord.range,
                            options: {
                                isWholeLine: false,
                                hoverMessage: [response.tooltip + "\n\nMore information available in the context menu."]
                            }
                        };
                        this.updateDecorations();
                    }
                }, this));
            }
        }
    };

    Compiler.prototype.onAsmToolTip = function (ed) {
        var pos = ed.getPosition();
        var word = ed.getModel().getWordAtPosition(pos);
        if (!word || !word.word) return;
        var opcode = word.word.toUpperCase();
        getAsmInfo(word.word).then(
            _.bind(function (asmHelp) {
                if (asmHelp) {
                    new Alert().alert(opcode + " help", asmHelp.html +
                        '<br><br>For more information, visit <a href="' + asmHelp.url + '" target="_blank" rel="noopener noreferrer">the ' +
                        opcode + ' documentation <sup><small class="glyphicon glyphicon-new-window" width="16px" height="16px" title="Opens in a new window"></small></sup></a>.',
                        function () {
                            ed.focus();
                            ed.setPosition(pos);
                        }
                    );
                } else {
                    new Alert().notify('This token was not found in the documentation.<br>Only <i>most</i> <b>Intel x86</b> opcodes supported for now.', {
                        group: "notokenindocs",
                        alertClass: "notification-error",
                        dismissTime: 3000
                    });
                }
            }), function (rejection) {
                new Alert().notify('There was an error fetching the documentation for this opcode (' + rejection + ').', {
                    group: "notokenindocs",
                    alertClass: "notification-error",
                    dismissTime: 3000
                });
            }
        );
    };

    return {
        Compiler: Compiler
    };
});
