// Copyright (c) 2012-2018, Rubén Rincón
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

    var options = require('options');
    var _ = require('underscore');
    var $ = require('jquery');

    require('selectize');

    function Conformance(hub, container, state) {
        this.container = container;
        this.eventHub = hub.createEventHub();
        this.compilerService = hub.compilerService;
        this.domRoot = container.getElement();
        this.domRoot.html($('#conformance').html());
        this.selectorList = this.domRoot.find('.compiler-list');
        this.addCompilerButton = this.domRoot.find('.add-compiler');
        this.selectorTemplate = $('#compiler-selector .compiler-row');
        this.editorId = state.editorid;
        this.nextSelectorId = 0;
        this.maxCompilations = options.cvCompilerCountMax || 6;
        this.langId = state.langId || _.keys(options.languages)[0];
        this.source = state.source || "";

        this.status = {
            allowCompile: false,
            allowAdd: true
        };
        this.stateByLang = {};

        this.container.on('destroy', this.close, this);

        this.container.on('open', function () {
            this.eventHub.emit("conformanceViewOpen", this.editorId);
        }, this);

        this.eventHub.on('resize', this.resize, this);
        this.container.on('resize', this.resize, this);

        this.container.on('shown', this.resize, this);

        this.eventHub.on('editorChange', this.onEditorChange, this);
        this.eventHub.on('editorClose', this.onEditorClose, this);
        this.eventHub.on('languageChange', this.onLanguageChange, this);

        this.addCompilerButton.on('click', _.bind(function () {
            this.addCompilerSelector();
            this.saveState();
        }, this));

        this.initFromState(state);
        this.handleToolbarUI();
    }

    Conformance.prototype.setTitle = function (compilerCount) {
        this.container.setTitle(options.languages[this.langId].name + " Conformance viewer (Editor #" + this.editorId + ") " + (
            compilerCount !== 0 ? (compilerCount + "/" + this.maxCompilations) : ""
        ));
    };

    Conformance.prototype.addCompilerSelector = function (config) {
        if (!config) {
            config = {
                // Code we have
                cv: this.nextSelectorId,
                // Compiler id which is being used
                compilerId: "",
                // Options which are in use
                options: "",
            };
            this.nextSelectorId++;
        }

        config.cv = Number(config.cv);

        var newEntry = this.selectorTemplate.clone();

        newEntry.attr("data-cv", config.cv);

        var onOptionsChange = _.debounce(_.bind(function () {
            this.saveState();
            this.compileAll();
        }, this), 800);

        newEntry.find('.options')
            .attr("data-cv", config.cv)
            .val(config.options)
            .on("click", onOptionsChange)
            .on("keyup", onOptionsChange);

        newEntry.find('.close')
            .attr("data-cv", config.cv)
            .on("click", _.bind(function () {
                this.removeCompilerSelector(config.cv);
            }, this));

        this.selectorList.append(newEntry);

        var status = newEntry.find('.status').attr("data-cv", config.cv);
        var langId = this.langId;
        var isVisible = function (compiler) {
            return compiler.lang === langId;
        };

        newEntry.find('.compiler-picker')
            .attr("data-cv", config.cv)
            .selectize({
                sortField: 'name',
                valueField: 'id',
                labelField: 'name',
                searchField: ['name'],
                options: _.filter(options.compilers, isVisible),
                items: config.compilerId ? [config.compilerId] : []
            })
            .on('change', _.bind(function () {
                // Hide the results button when a new compiler is selected
                this.handleStatusIcon(status, {code: 0, text: ""});
                // We could narrow the compilation to only this compiler!
                this.compileAll();
                // We're not saving state here. It's done after compiling
            }, this));
        this.handleStatusIcon(status, {code: 0, text: ""});
        this.handleToolbarUI();
        this.saveState();
    };

    Conformance.prototype.removeCompilerSelector = function (cv) {
        _.each(this.selectorList.children(), function (row) {
            var child = $(row);
            if (child.attr("data-cv") == cv) {
                child.remove();
            }
        }, this);
        this.handleToolbarUI();
        this.saveState();
    };

    Conformance.prototype.onEditorChange = function (editorId, newSource, langId) {
        if (editorId == this.editorId) {
            this.langId = langId;
            this.source = newSource;
            this.compileAll();
        }
    };

    Conformance.prototype.onEditorClose = function (editorId) {
        if (editorId === this.editorId) {
            this.close();
            _.defer(function (self) {
                self.container.close();
            }, this);
        }
    };

    Conformance.prototype.onCompileResponse = function (cv, result) {
        var allText = _.pluck((result.stdout || []).concat(result.stderr || []), 'text').join("\n");
        var failed = result.code !== 0;
        var warns = !failed && !!allText;
        var status = {
            text: allText.replace(/\x1b\\[[0-9;]*m/, ''),
            code: failed ? 3 : (warns ? 2 : 1)
        };
        this.handleStatusIcon(this.selectorList.find('[data-cv="' + cv + '"] .status'), status);
        this.saveState();
    };

    Conformance.prototype.compileAll = function () {
        if (!this.source) return;
        // Hide previous status icons
        this.selectorList.find('.status').css("visibility", "hidden");
        this.compilerService.expand(this.source).then(_.bind(function (expanded) {
            var compileCount = 0;
            _.each(this.selectorList.children(), _.bind(function (child) {
                var picker = $(child).find('.compiler-picker');
                // We make sure we are not over our limit
                if (picker && compileCount < this.maxCompilations) {
                    compileCount++;
                    if (picker.val()) {
                        var cv = Number(picker.attr("data-cv"));
                        var request = {
                            source: expanded || "",
                            compiler: picker.val(),
                            options: {
                                userArguments: $(child).find(".options[data-cv='" + cv + "']").val(),
                                filters: {},
                                compilerOptions: {produceAst: false, produceOptInfo: false}
                            }
                        };
                        // This error function ensures that the user will know we had a problem (As we don't save asm)
                        this.compilerService.submit(request)
                            .then(_.bind(function (x) {
                                this.onCompileResponse(cv, x.result);
                            }, this))
                            .catch(_.bind(function (x) {
                                this.onCompileResponse(cv, {
                                    asm: "",
                                    code: -1,
                                    stdout: "",
                                    stderr: x.error
                                });
                            }, this));
                    }
                }
            }, this));
        }, this));
    };

    Conformance.prototype.handleToolbarUI = function () {
        var compilerCount = this.selectorList.children().length;

        // Only allow new compilers if we allow for more
        this.addCompilerButton.attr("disabled", compilerCount >= this.maxCompilations);

        this.setTitle(compilerCount);
    };

    Conformance.prototype.handleStatusIcon = function (element, status) {
        if (!element) return;
        element.attr("class", "status glyphicon glyphicon-" + (status.code === 3 ? "remove-sign" : (status.code === 2 ? "info-sign" : "ok-sign")))
            .css("visibility", status.code === 0 ? "hidden" : "visible")
            .css("color", status.code === 3 ? "red" : (status.code === 2 ? "yellow" : "green"))
            .attr("title", status.text)
            .attr("aria-label", status.code === 3 ? "Compilation failed!" : (status.code === 2 ? "Compiled with warnings" : "Compiled without warnings"))
            .attr("data-status", status.code);
    };

    Conformance.prototype.currentState = function () {
        var state = {
            editorid: this.editorId,
            langId: this.langId,
            compilers: []
        };
        _.each(this.selectorList.children(), _.bind(function (child) {
            state.compilers.push({
                // Code we have
                cv: $(child).attr("data-cv"),
                // Compiler which is being used
                compilerId: $(child).find('.compiler-picker').val(),
                // Options which are in use
                options: $(child).find(".options").val(),
            });
        }, this));
        return state;
    };

    Conformance.prototype.saveState = function () {
        this.container.setState(this.currentState());
    };

    Conformance.prototype.resize = function () {
        this.selectorList.css("height", this.domRoot.height() - this.domRoot.find('.top-bar').outerHeight(true));
    };

    Conformance.prototype.onLanguageChange = function (editorId, newLangId) {
        if (editorId === this.editorId && this.langId !== newLangId) {
            var oldLangId = this.langId;
            this.stateByLang[oldLangId] = this.currentState();

            this.langId = newLangId;
            this.selectorList.children().remove();
            this.nextSelectorId = 0;
            if (this.stateByLang[newLangId]) {
                this.initFromState(this.stateByLang[newLangId]);
            }
            this.handleToolbarUI();
            this.saveState();
        }
    };

    Conformance.prototype.initFromState = function (state) {
        if (state.compilers) {
            _.each(state.compilers, _.bind(function (config) {
                config.cv = this.nextSelectorId;
                this.nextSelectorId++;
                this.addCompilerSelector(config);
            }, this));
        }
    };

    Conformance.prototype.close = function () {
        this.eventHub.unsubscribe();
        this.eventHub.emit("conformanceViewClose", this.editorId);
    };

    return {
        Conformance: Conformance
    };
});
