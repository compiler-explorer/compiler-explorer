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
    'use strict';
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

    function patchOldFilters(filters) {
        if (filters === undefined) return undefined;
        // Filters are of the form {filter: true|false¸ ...}. In older versions, we used
        // to suppress the {filter:false} form. This means we can't distinguish between
        // "filter not on" and "filter not present". In the latter case we want to default
        // the filter. In the former case we want the filter off. Filters now don't suppress
        // but there are plenty of permanlinks out there with no filters set at all. Here
        // we manually set any missing filters to 'false' to recover the old behaviour of
        // "if it's not here, it's off".
        _.each(['binary', 'labels', 'directives', 'commentOnly', 'trim', 'intel'], function (oldFilter) {
            if (filters[oldFilter] === undefined) filters[oldFilter] = false;
        });
        return filters;
    }

    function Compiler(hub, container, state) {
        this.container = container;
        this.eventHub = hub.createEventHub();
        this.compilerService = hub.compilerService;
        this.domRoot = container.getElement();
        this.domRoot.html($('#compiler').html());
        this.id = state.id || hub.nextCompilerId();
        this.sourceEditorId = state.source || 1;
        this.compiler = this.compilerService.getCompilerById(state.compiler) ||
            this.compilerService.getCompilerById(options.defaultCompiler);
        this.deferCompiles = hub.deferred;
        this.needsCompile = false;
        this.options = state.options || options.compileOptions;
        this.filters = new Toggles(this.domRoot.find('.filters'), patchOldFilters(state.filters));
        this.source = '';
        this.assembly = [];
        this.colours = [];
        this.lastResult = {};
        this.pendingRequestSentAt = 0;
        this.nextRequest = null;
        this.settings = {};
        this.optViewOpen = false;
        this.cfgViewOpen = false;
        this.wantOptInfo = state.wantOptInfo;
        this.compilerSupportsCfg = false;
        this.decorations = {};
        this.prevDecorations = [];
        this.optButton = this.domRoot.find('.btn.view-optimization');
        this.astButton = this.domRoot.find('.btn.view-ast');
        this.gccDumpButton = this.domRoot.find('.btn.view-gccdump');
        this.cfgButton = this.domRoot.find('.btn.view-cfg');
        this.libsButton = this.domRoot.find('.btn.show-libs');

        this.availableLibs = $.extend(true, {}, options.libs);

        this.compileTimeLabel = this.domRoot.find('.compile-time');
        this.compileClearCache = this.domRoot.find('.clear-cache');

        _.each(state.libs, _.bind(function (lib) {
            if (this.availableLibs[lib.name] && this.availableLibs[lib.name].versions &&
                this.availableLibs[lib.name].versions[lib.ver]) {
                this.availableLibs[lib.name].versions[lib.ver].used = true;
            }
        }, this));

        this.linkedFadeTimeoutId = -1;

        this.domRoot.find('.compiler-picker').selectize({
            sortField: 'name',
            valueField: 'id',
            labelField: 'name',
            searchField: ['name'],
            options: options.compilers,
            items: this.compiler ? [this.compiler.id] : []
        }).on('change', _.bind(function (e) {
            var val = $(e.target).val();
            ga('send', {
                hitType: 'event',
                eventCategory: 'SelectCompiler',
                eventAction: val
            });
            this.onCompilerChange(val);
        }, this));
        var optionsChange = _.debounce(_.bind(function (e) {
            this.onOptionsChange($(e.target).val());
        }, this), 800);
        this.optionsField = this.domRoot.find('.options');
        this.optionsField
            .val(this.options)
            .on('change', optionsChange)
            .on('keyup', optionsChange);

        // Hide the binary option if the global options has it disabled.
        this.domRoot.find('[data-bind=\'binary\']').toggle(options.supportsBinary);
        this.domRoot.find('[data-bind=\'execute\']').toggle(options.supportsExecute);

        this.outputEditor = monaco.editor.create(this.domRoot.find('.monaco-placeholder')[0], {
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
            run: _.bind(function (ed) {
                var desiredLine = ed.getPosition().lineNumber - 1;
                var source = this.assembly[desiredLine].source;
                if (source.file === null) {
                    // a null file means it was the user's source
                    this.eventHub.emit('editorSetDecoration', this.sourceEditorId, source.line, true);
                } else {
                    // TODO: some indication this asm statement came from elsewhere
                    this.eventHub.emit('editorSetDecoration', this.sourceEditorId, -1, false);
                }
            }, this)
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

        var clearEditorsLinkedLines = _.bind(function () {
            this.eventHub.emit('editorSetDecoration', this.sourceEditorId, -1, false);
        }, this);

        this.outputEditor.onMouseMove(_.bind(function (e) {
            this.mouseMoveThrottledFunction(e);
            if (this.linkedFadeTimeoutId !== -1) {
                clearTimeout(this.linkedFadeTimeoutId);
                this.linkedFadeTimeoutId = -1;
            }
        }, this));

        this.mouseMoveThrottledFunction = _.throttle(_.bind(this.onMouseMove, this), 250);

        this.outputEditor.onMouseLeave(_.bind(function () {
            this.linkedFadeTimeoutId = setTimeout(_.bind(function () {
                clearEditorsLinkedLines();
                this.linkedFadeTimeoutId = -1;
            }, this), 5000);
        }, this));

        this.fontScale = new FontScale(this.domRoot, state, this.outputEditor);
        this.fontScale.on('change', _.bind(function () {
            this.saveState();
            this.updateFontScale();
        }, this));

        this.filters.on('change', _.bind(this.onFilterChange, this));

        container.on('destroy', function () {
            this.eventHub.unsubscribe();
            this.eventHub.emit('compilerClose', this.id);
            this.outputEditor.dispose();
        }, this);
        container.on('resize', this.resize, this);
        container.on('shown', this.resize, this);
        container.on('open', function () {
            this.eventHub.emit('compilerOpen', this.id, this.sourceEditorId);
            this.updateFontScale();
        }, this);
        this.eventHub.on('editorChange', this.onEditorChange, this);
        this.eventHub.on('editorClose', this.onEditorClose, this);
        this.eventHub.on('colours', this.onColours, this);
        this.eventHub.on('resendCompilation', this.onResendCompilation, this);
        this.eventHub.on('findCompilers', this.sendCompiler, this);
        this.eventHub.on('compilerSetDecorations', this.onCompilerSetDecorations, this);
        this.eventHub.on('settingsChange', this.onSettingsChange, this);
        this.eventHub.on('optViewOpened', this.onOptViewOpened, this);
        this.eventHub.on('optViewClosed', this.onOptViewClosed, this);
        this.eventHub.on('astViewOpened', this.onAstViewOpened, this);
        this.eventHub.on('astViewClosed', this.onAstViewClosed, this);

        this.eventHub.on('gccDumpPassSelected', this.onGccDumpPassSelected, this);
        this.eventHub.on('gccDumpFiltersChanged', this.onGccDumpFiltersChanged, this);
        this.eventHub.on('gccDumpViewOpened', this.onGccDumpViewOpened, this);
        this.eventHub.on('gccDumpViewClosed', this.onGccDumpViewClosed, this);
        this.eventHub.on('gccDumpUIInit', this.onGccDumpUIInit, this);

        this.eventHub.on('cfgViewOpened', this.onCfgViewOpened, this);
        this.eventHub.on('cfgViewClosed', this.onCfgViewClosed, this);
        this.eventHub.on('resize', this.resize, this);
        this.eventHub.on('requestFilters', function (id) {
            if (id === this.id) {
                this.eventHub.emit('filtersChange', this.id, this.getEffectiveFilters());
            }
        }, this);
        this.eventHub.emit('requestSettings');
        this.sendCompiler();
        this.updateCompilerName();
        this.updateButtons();

        var outputConfig = _.bind(function () {
            return Components.getOutput(this.id, this.sourceEditorId);
        }, this);

        this.container.layoutManager.createDragSource(this.domRoot.find('.status').parent(), outputConfig);
        this.domRoot.find('.status').parent().click(_.bind(function () {
            var insertPoint = hub.findParentRowOrColumn(this.container) ||
                this.container.layoutManager.root.contentItems[0];
            insertPoint.addChild(outputConfig);
        }, this));

        var cloneComponent = _.bind(function () {
            return {
                type: 'component',
                componentName: 'compiler',
                componentState: this.currentState()
            };
        }, this);

        var createOptView = _.bind(function () {
            return Components.getOptViewWith(this.id, this.source, this.lastResult.optOutput, this.getCompilerName(), this.sourceEditorId);
        }, this);

        var createAstView = _.bind(function () {
            return Components.getAstViewWith(this.id, this.source, this.lastResult.astOutput, this.getCompilerName(), this.sourceEditorId);
        }, this);

        var createGccDumpView = _.bind(function () {
            return Components.getGccDumpViewWith(this.id, this.getCompilerName(), this.sourceEditorId, this.lastResult.gccDumpOutput);
        }, this);

        var createCfgView = _.bind(function () {
            return Components.getCfgViewWith(this.id, this.getCompilerName(), this.sourceEditorId);
        }, this);

        this.container.layoutManager.createDragSource(
            this.domRoot.find('.btn.add-compiler'), cloneComponent);

        this.domRoot.find('.btn.add-compiler').click(_.bind(function () {
            var insertPoint = hub.findParentRowOrColumn(this.container) ||
                this.container.layoutManager.root.contentItems[0];
            insertPoint.addChild(cloneComponent);
        }, this));

        this.container.layoutManager.createDragSource(
            this.optButton, createOptView);

        this.optButton.click(_.bind(function () {
            var insertPoint = hub.findParentRowOrColumn(this.container) ||
                this.container.layoutManager.root.contentItems[0];
            insertPoint.addChild(createOptView);
        }, this));

        this.container.layoutManager.createDragSource(
            this.astButton, createAstView);

        this.astButton.click(_.bind(function () {
            var insertPoint = hub.findParentRowOrColumn(this.container) ||
                this.container.layoutManager.root.contentItems[0];
            insertPoint.addChild(createAstView);
        }, this));

        this.container.layoutManager.createDragSource(
            this.gccDumpButton, createGccDumpView);

        this.gccDumpButton.click(_.bind(function () {
            var insertPoint = hub.findParentRowOrColumn(this.container) ||
                this.container.layoutManager.root.contentItems[0];
            insertPoint.addChild(createGccDumpView);
        }, this));

        this.container.layoutManager.createDragSource(
            this.cfgButton, createCfgView);

        this.cfgButton.click(_.bind(function () {
            var insertPoint = hub.findParentRowOrColumn(this.container) ||
                this.container.layoutManager.root.contentItems[0];
            insertPoint.addChild(createCfgView);
        }, this));


        var updateLibsUsed = _.bind(function () {
            var libsCount = Object.keys(this.availableLibs).length;
            if (libsCount === 0) {
                return $('<p></p>')
                    .text('No libs configured for this language yet. ')
                    .append($('<a></a>')
                        .attr('target', '_blank')
                        .attr('rel', 'noopener noreferrer')
                        .attr('href', 'https://github.com/mattgodbolt/compiler-explorer/issues/new')
                        .text('You can suggest us one at any time ')
                        .append($('<sup></sup>')
                            .addClass('glyphicon glyphicon-new-window')
                            .width('16px')
                            .height('16px')
                            .attr('title', 'Opens in a new window')
                        )
                    );
            }
            var columnCount = Math.ceil(libsCount / 5);
            var currentLibIndex = -1;

            var libLists = [];
            for (var i = 0; i < columnCount; i++) {
                libLists.push($('<ul></ul>').addClass('lib-list'));
            }

            // Utility function so we can iterate indefinetly over our lists
            var getNextList = function () {
                currentLibIndex = (currentLibIndex + 1) % columnCount;
                return libLists[currentLibIndex];
            };

            var onChecked = _.bind(function (e) {
                var elem = $(e.target);
                // Uncheck every lib checkbox with the same name if we're checking the target
                if (elem.prop('checked')) {
                    var others = $.find('input[name=\'' + elem.prop('name') + '\']');
                    _.each(others, function (other) {
                        $(other).prop('checked', false);
                    });
                    // Recheck the targeted one
                    elem.prop('checked', true);
                }
                // And now do the same with the availableLibs object
                _.each(this.availableLibs[elem.prop('data-lib')].versions, function (version) {
                    version.used = false;
                });
                this.availableLibs[elem.prop('data-lib')].versions[elem.prop('data-version')].used = elem.prop('checked');
                this.saveState();
                this.compile();
            }, this);

            _.each(this.availableLibs, function (lib, libKey) {
                var libsList = getNextList();
                var libCat = $('<li></li>')
                    .append($('<span></span>')
                        .text(lib.name)
                        .addClass('lib-header')
                    )
                    .addClass('lib-item');

                var libGroup = $('<div></div>');

                if (libsList.children().length > 0)
                    libsList.append($('<hr>').addClass('lib-separator'));

                _.each(lib.versions, function (version, vKey) {
                    libGroup.append($('<div></div>')
                        .append($('<input type="checkbox">')
                            .addClass('lib-checkbox')
                            .prop('data-lib', libKey)
                            .prop('data-version', vKey)
                            .prop('checked', version.used)
                            .prop('name', libKey)
                            .on('change', onChecked)
                        ).append($('<label></label>')
                            .addClass('lib-label')
                            .text(lib.name + ' ' + version.version)
                            .on('click', function () {
                                $(this).parent().find('.lib-checkbox').trigger('click');
                            })
                        )
                    );
                });
                libGroup.appendTo(libCat);
                libCat.appendTo(libsList);
            });
            return $('<div></div>').addClass('libs-container').append(libLists);
        }, this);

        this.libsButton.popover({
            container: 'body',
            content: updateLibsUsed(),
            html: true,
            placement: 'bottom',
            trigger: 'manual'
        }).click(_.bind(function () {
            this.libsButton.popover('show');
        }, this)).on('inserted.bs.popover', function (e) {
            $(e.target).content = updateLibsUsed().html();
        }).on('show.bs.popover', function () {
            $(this).data('bs.popover').tip().css('max-width', '100%').css('width', 'auto');
        });

        this.compileClearCache.on('click', _.bind(function () {
            this.compilerService.cache.reset();
            this.compile();
        }, this));

        // Dismiss the popover on escape.
        $(document).on('keyup.editable', _.bind(function (e) {
            if (e.which === 27) {
                this.libsButton.popover('hide');
            }
        }, this));

        // Dismiss on any click that isn't either in the opening element, inside
        // the popover or on any alert
        $(document).on('click', _.bind(function (e) {
            var elem = this.libsButton;
            var target = $(e.target);
            if (!target.is(elem) && elem.has(target).length === 0 && target.closest('.popover').length === 0) {
                elem.popover('hide');
            }
        }, this));

        this.eventHub.on('initialised', this.undefer, this);

        this.saveState();
    }

    Compiler.prototype.undefer = function () {
        this.deferCompiles = false;
        if (this.needsCompile) this.compile();
    };

    // TODO: need to call resize if either .top-bar or .bottom-bar resizes, which needs some work.
    // Issue manifests if you make a window where one compiler is small enough that the buttons spill onto two lines:
    // reload the page and the bottom-bar is off the bottom until you scroll a tiny bit.
    Compiler.prototype.resize = function () {
        var topBarHeight = this.domRoot.find('.top-bar').outerHeight(true);
        var bottomBarHeight = this.domRoot.find('.bottom-bar').outerHeight(true);
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
        if (this.deferCompiles) {
            this.needsCompile = true;
            return;
        }
        this.needsCompile = false;
        var options = {
            userArguments: this.options,
            compilerOptions: {
                produceAst: this.astViewOpen,
                produceGccDump: {
                    opened: this.gccDumpViewOpen,
                    pass: this.gccDumpPassSelected,
                    treeDump: this.treeDumpEnabled,
                    rtlDump: this.rtlDumpEnabled
                },
                produceOptInfo: this.wantOptInfo
            },
            filters: this.getEffectiveFilters()
        };
        _.each(this.availableLibs, function (lib) {
            _.each(lib.versions, function (version) {
                if (version.used) {
                    _.each(version.path, function (path) {
                        options.userArguments += ' -I' + path;
                    });
                }
            });
        });
        this.compilerService.expand(this.source).then(_.bind(function (expanded) {
            var request = {
                source: expanded || '',
                compiler: this.compiler ? this.compiler.id : '',
                options: options
            };
            if (!this.compiler) {
                this.onCompileResponse(request, errorResult('<Please select a compiler>'), false);
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
            this.setAssembly(fakeAsm('<Compiling...>'));
        }, this), 500);
        this.compilerService.submit(request)
            .then(function (x) {
                clearTimeout(progress);
                onCompilerResponse(request, x.result, x.localCacheHit);
            })
            .catch(function (x) {
                clearTimeout(progress);
                onCompilerResponse(request, errorResult('<Remote compilation failed: ' + x.error + '>'), false);
            });
    };

    Compiler.prototype.getBinaryForLine = function (line) {
        var obj = this.assembly[line - 1];
        if (!obj) return '<div class="address">????</div><div class="opcodes"><span class="opcode">????</span></div>';
        var address = obj.address ? obj.address.toString(16) : '';
        var opcodes = '<div class="opcodes" title="' + (obj.opcodes || []).join(' ') + '">';
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
            this.outputEditor.getModel().setValue(_.pluck(assembly, 'text').join('\n'));
            var addrToAddrDiv = {};
            var decorations = [];
            _.each(this.assembly, _.bind(function (obj, line) {
                var address = obj.address ? obj.address.toString(16) : '';
                //     var div = $("<div class='address cm-number'>" + address + "</div>");
                addrToAddrDiv[address] = {div: 'moo', line: line};
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
        return {asm: fakeAsm(text), code: -1, stdout: '', stderr: ''};
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

        this.setAssembly(result.asm || fakeAsm('<No output>'));
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
        var status = this.domRoot.find('.status');
        var allText = _.pluck((result.stdout || []).concat(result.stderr || []), 'text').join('\n');
        var failed = result.code !== 0;
        var warns = !failed && !!allText;
        status.toggleClass('error', failed);
        status.toggleClass('warning', warns);
        status.parent().attr('title', allText);
        if (cached) {
            this.compileTimeLabel.text(' - cached');
        } else if (wasRealReply) {
            this.compileTimeLabel.text(' - ' + timeTaken + 'ms');
        } else {
            this.compileTimeLabel.text('');
        }
        this.compilerSupportsCfg = result.supportsCfg;
        this.eventHub.emit('compileResult', this.id, this.compiler, result);
        this.updateButtons();

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
            this.optViewOpen = false;
            this.optButton.prop('disabled', this.optViewOpen);
        }
    };

    Compiler.prototype.onAstViewOpened = function (id) {
        if (this.id == id) {
            this.astButton.prop('disabled', true);
            this.astViewOpen = true;
            this.compile();
        }
    };

    Compiler.prototype.onAstViewClosed = function (id) {
        if (this.id === id) {
            this.astButton.prop('disabled', false);
            this.astViewOpen = false;
        }
    };

    Compiler.prototype.onGccDumpUIInit = function (id) {
        if (this.id === id) {
            this.compile();
        }
    };

    Compiler.prototype.onGccDumpFiltersChanged = function (id, filters, reqCompile) {
        if (this.id === id) {
            this.treeDumpEnabled = (filters.treeDump !== false);
            this.rtlDumpEnabled = (filters.rtlDump !== false);

            if (reqCompile) {
                this.compile();
            }
        }
    };

    Compiler.prototype.onGccDumpPassSelected = function (id, passId, reqCompile) {
        if (this.id === id) {
            this.gccDumpPassSelected = passId;

            if (reqCompile && passId !== '') {
                this.compile();
            }
        }
    };

    Compiler.prototype.onGccDumpViewOpened = function (id) {
        if (this.id === id) {
            this.gccDumpButton.prop('disabled', true);
            this.gccDumpViewOpen = true;
        }
    };

    Compiler.prototype.onGccDumpViewClosed = function (id) {
        if (this.id === id) {
            this.gccDumpButton.prop('disabled', !this.compiler.supportsGccDump);
            this.gccDumpViewOpen = false;

            delete this.gccDumpPassSelected;
            delete this.treeDumpEnabled;
            delete this.rtlDumpEnabled;
        }
    };

    Compiler.prototype.onOptViewOpened = function (id) {
        if (this.id === id) {
            this.optViewOpen = true;
            this.wantOptInfo = true;
            this.optButton.prop('disabled', this.optViewOpen);
            this.compile();
        }
    };

    Compiler.prototype.onCfgViewOpened = function (id) {
        if (this.id == id) {
            this.cfgButton.prop('disabled', true);
            this.cfgViewOpen = true;
            this.compile();
        }
    };

    Compiler.prototype.onCfgViewClosed = function (id) {
        if (this.id === id) {
            this.cfgViewOpen = false;
            this.cfgButton.prop('disabled', this.getEffectiveFilters().binary);
        }
    };


    Compiler.prototype.updateButtons = function () {
        if (!this.compiler) return;
        var filters = this.getEffectiveFilters();
        // We can support intel output if the compiler supports it, or if we're compiling
        // to binary (as we can disassemble it however we like).
        var intelAsm = this.compiler.supportsIntel || filters.binary;
        this.domRoot.find('[data-bind=\'intel\']').toggleClass('disabled', !intelAsm);
        // Disable binary support on compilers that don't work with it.
        this.domRoot.find('[data-bind=\'binary\']')
            .toggleClass('disabled', !this.compiler.supportsBinary);
        this.domRoot.find('[data-bind=\'execute\']')
            .toggleClass('disabled', !this.compiler.supportsExecute);
        // Disable demangle for compilers where we can't access it
        this.domRoot.find('[data-bind=\'demangle\']')
            .toggleClass('disabled', !this.compiler.demangler);
        // Disable any of the options which don't make sense in binary mode.
        var filtersDisabled = !!filters.binary && !this.compiler.supportsFiltersInBinary;
        this.domRoot.find('.nonbinary').toggleClass('disabled', filtersDisabled);
        // If its already open, we should turn the it off.
        // The pane will update with error text
        // Other wise we just disable the button.
        if (!this.optViewOpen) {
            this.optButton.prop('disabled', !this.compiler.supportsOptOutput);
        } else {
            this.optButton.prop('disabled', true);
        }

        if (!this.cfgViewOpen) {
            this.cfgButton.prop('disabled', !this.compilerSupportsCfg);
        } else {
            this.cfgButton.prop('disabled', true);
        }

        if (!this.gccDumpViewOpen) {
            this.gccDumpButton.prop('disabled', !this.compiler.supportsGccDump);
        } else {
            this.gccDumpButton.prop('disabled', true);
        }
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
        this.eventHub.emit('filtersChange', this.id, this.getEffectiveFilters());
        this.saveState();
        this.compile();
        this.updateButtons();
    };

    Compiler.prototype.currentState = function () {
        var libs = [];
        _.each(this.availableLibs, function (library, name) {
            _.each(library.versions, function (version, ver) {
                if (library.versions[ver].used) {
                    libs.push({name: name, ver: ver});
                }
            });
        });
        var state = {
            compiler: this.compiler ? this.compiler.id : '',
            source: this.sourceEditorId,
            options: this.options,
            // NB must *not* be effective filters
            filters: this.filters.get(),
            wantOptInfo: this.wantOptInfo,
            libs: libs
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
        return this.compiler ? this.compiler.name : 'no compiler set';
    };

    Compiler.prototype.updateCompilerName = function () {
        var compilerName = this.getCompilerName();
        var compilerVersion = this.compiler ? this.compiler.version : '';
        this.container.setTitle(compilerName + ' (Editor #' + this.sourceEditorId + ', Compiler #' + this.id + ')');
        this.domRoot.find('.full-compiler-name').text(compilerVersion);
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
                asBig = bigInt('ffffffffffffffff', 16).and(asBig);
            }
            return value + ' = 0x' + asBig.toString(16).toUpperCase();
        }

        return null;
    }


    var getAsmInfo = function (opcode) {
        var cacheName = 'asm/' + opcode;
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
        if (e === null || e.target === null || e.target.position === null) return;
        if (this.settings.hoverShowSource === true && this.assembly) {
            var hoverAsm = this.assembly[e.target.position.lineNumber - 1];
            if (hoverAsm) {
                // We check that we actually have something to show at this point!
                this.eventHub.emit('editorSetDecoration', this.sourceEditorId, hoverAsm.source && !hoverAsm.source.file ? hoverAsm.source.line : -1, false);
            }
        }
        var currentWord = this.outputEditor.getModel().getWordAtPosition(e.target.position);
        if (currentWord && currentWord.word) {
            var word = currentWord.word;
            currentWord.range = new monaco.Range(e.target.position.lineNumber, currentWord.startColumn, e.target.position.lineNumber, currentWord.endColumn);
            // Avoid throwing an exception if somehow (How?) we have a non existent lineNumber. c.f. https://sentry.io/matt-godbolt/compiler-explorer/issues/285270358/
            if (e.target.position.lineNumber < this.outputEditor.getModel().getLineCount()) {
                // Hacky workaround to check for negative numbers. c.f. https://github.com/mattgodbolt/compiler-explorer/issues/434
                var lineContent = this.outputEditor.getModel().getLineContent(e.target.position.lineNumber);
                if (lineContent[currentWord.startColumn - 2] === '-') {
                    word = '-' + word;
                    currentWord.range.startColumn -= 1;
                }
            }
            var numericToolTip = getNumericToolTip(word);
            if (numericToolTip) {
                this.decorations.numericToolTip = {
                    range: currentWord.range,
                    options: {isWholeLine: false, hoverMessage: ['`' + numericToolTip + '`']}
                };
                this.updateDecorations();
            }

            var getTokensForLine = function (model, line) {
                //Force line's state to be accurate
                if (line > model.getLineCount()) return [];
                model.getLineTokens(line, /*inaccurateTokensAcceptable*/false);
                // Get the tokenization state at the beginning of this line
                var state = model._lines[line - 1].getState();
                if (!state) return [];
                var freshState = model._lines[line - 1].getState().clone();
                // Get the human readable tokens on this line
                return model._tokenizationSupport.tokenize(model.getLineContent(line), freshState, 0).tokens;
            };

            if (this.settings.hoverShowAsmDoc === true &&
                _.some(getTokensForLine(this.outputEditor.getModel(), currentWord.range.startLineNumber), function (t) {
                    return t.offset + 1 === currentWord.startColumn && t.type === 'keyword.asm';
                })) {
                getAsmInfo(currentWord.word).then(_.bind(function (response) {
                    if (response) {
                        this.decorations.asmToolTip = {
                            range: currentWord.range,
                            options: {
                                isWholeLine: false,
                                hoverMessage: [response.tooltip + '\n\nMore information available in the context menu.']
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
                    new Alert().alert(opcode + ' help', asmHelp.html +
                        '<br><br>For more information, visit <a href="' + asmHelp.url + '" target="_blank" rel="noopener noreferrer">the ' +
                        opcode + ' documentation <sup><small class="glyphicon glyphicon-new-window" width="16px" height="16px" title="Opens in a new window"></small></sup></a>.',
                        function () {
                            ed.focus();
                            ed.setPosition(pos);
                        }
                    );
                } else {
                    new Alert().notify('This token was not found in the documentation.<br>Only <i>most</i> <b>Intel x86</b> opcodes supported for now.', {
                        group: 'notokenindocs',
                        alertClass: 'notification-error',
                        dismissTime: 3000
                    });
                }
            }), function (rejection) {
                new Alert().notify('There was an error fetching the documentation for this opcode (' + rejection + ').', {
                    group: 'notokenindocs',
                    alertClass: 'notification-error',
                    dismissTime: 3000
                });
            }
        );
    };

    return {
        Compiler: Compiler
    };
});
