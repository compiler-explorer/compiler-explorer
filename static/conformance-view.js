// Copyright (c) 2012-2017, Rubén Rincón
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

    require('selectize');

    var Compiler = require('./compiler').Compiler;

    var compilers = options.compilers;
    var compilersById = {};
    _.forEach(compilers, function (compiler) {
        compilersById[compiler.id] = compiler;
        if (compiler.alias) compilersById[compiler.alias] = compiler;
    });

    function Conformance(hub, container, state) {
        this.container = container;
        this.eventHub = hub.createEventHub();
        this.domRoot = container.getElement();
        this.domRoot.html($('#conformance').html());
        this.selectorList = this.domRoot.find('.compiler-list');
        this.addCompilerButton = this.domRoot.find('.add-compiler');
        this.selectorTemplate = $('#compiler-selector .compiler-row');
        this.editorId = state.editorid;
        this.source = state.source || "";
        this.nextSelectorId = 0;
        this.maxCompilations = options.cvCompilerCountMax || 6;

        this.status = {
            allowCompile: false,
            allowAdd: true
        };

        this.container.on('destroy', function () {
            this.eventHub.emit("conformanceViewClose", this.editorId);
            this.eventHub.unsubscribe();
        }, this);

        this.container.on('open', function () {
            this.eventHub.emit("conformanceViewOpen", this.editorId);
        }, this);

        this.container.on('resize', this.resize, this);

        this.container.on('shown', this.resize, this);

        this.eventHub.on('editorChange', this.onEditorChange, this);
        this.eventHub.on('editorClose', this.onEditorClose, this);

        this.addCompilerButton.on('click', _.bind(function() {
            this.addCompilerSelector();
            this.saveState();
        }, this));

        if (state.compilers) {
            _.each(state.compilers, _.bind(function (config) {
                config.cv = this.nextSelectorId;
                this.nextSelectorId++;
                this.addCompilerSelector(config);
            }, this));
        }

        this.handleToolbarUI();
    }

    Conformance.prototype.setTitle = function (compilerCount) {
        this.container.setTitle("Conformance viewer (Editor #" + this.editorId + ") " + (
                compilerCount !== 0 ? (compilerCount + "/" + this.maxCompilations) : ""
            ));
    };

    Conformance.prototype.addCompilerSelector = function(config) {
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

        var status = newEntry.find('.status')
            .attr("data-cv", config.cv)
            .on('change', _.bind(function() {
                this.saveState();
            }, this));

        newEntry.find('.options')
            .attr("data-cv", config.cv)
            .val(config.options);

        newEntry.find('.close')
            .attr("data-cv", config.cv)
            .on("click", _.bind(function() {
                this.removeCompilerSelector(config.cv);
            }, this));

        this.selectorList.append(newEntry);

        newEntry.find('.compiler-picker')
            .attr("data-cv", config.cv)
            .selectize({
                sortField: 'name',
                valueField: 'id',
                labelField: 'name',
                searchField: ['name'],
                options: compilers,
                items: config.compilerId ? [config.compilerId] : []
            }).on('change', _.bind(function() {
                // Hide the results button when a new compiler is selected
                this.handleStatusIcon(status, {code: 0, text: ""});
                // We could narrow the compilation to only this compiler!
                this.compileAll();
                // We're not saving state here. It's done after compiling
            }, this));
        this.handleStatusIcon(status, {code: 0, text: ""});
        this.handleToolbarUI();
    };

    Conformance.prototype.removeCompilerSelector = function(cv) {
        _.each(this.selectorList.children(), function(row) {
            var child = $(row);
            if (child.attr("data-cv") == cv) {
                child.remove();
            }
        }, this);
        this.handleToolbarUI();
        this.saveState();
    };

    Conformance.prototype.onEditorChange = function(editorId, newSource) {
        if (editorId == this.editorId) {
            this.source = newSource;
            this.compileAll();
        }
    };

    Conformance.prototype.onEditorClose = function(editorId) {
        if (editorId == this.editorId) {
            _.defer(function (self) {
                self.container.close();
            }, this);
        }
    };

    Conformance.prototype.onCompileResponse = function (request, result, cached) {
        var allText = _.pluck((result.stdout || []).concat(result.stderr || []), 'text').join("\n");
        var failed = result.code !== 0;
        var warns = !failed && !!allText;
        var status = {
            text: allText,
            code: failed ? 3 : (warns ? 2 : 1)
        };
        this.handleStatusIcon(this.selectorList.find('[data-cv="' + request.extras.cv + '"] .status'), status);
        this.saveState();
    };

    Conformance.prototype.compileAll = function() {
        // Hide previous status icons
        this.selectorList.find('.status').css("visibility", "hidden");
        Compiler.prototype.expand(this.source).then(_.bind(function (expanded) {
            var compileCount = 0;
            _.each(this.selectorList.children(), _.bind(function(child) {
                var picker = $(child).find('.compiler-picker');
                // We make sure we are not over our limit
                if (picker && compileCount < this.maxCompilations) {
                    compileCount++;
                    if (picker.val()) {
                        var cv = Number(picker.attr("data-cv"));
                        var request = {
                            source: expanded || "",
                            compiler: picker.val(),
                            options: $(child).find(".options[data-cv='" + cv + "']").val(),
                            backendOptions: {produceAst: false},
                            filters: {},
                            extras: {
                                cv: cv,
                                storeAsm: false,
                                emitCompilingEvent: false,
                                ignorePendingRequest: true
                            }
                        };
                        // This error function ensures that the user will know we had a problem (As we don't save asm)
                        Compiler.prototype.sendCompile(request, _.bind(this.onCompileResponse, this), function(text) {
                                return {asm: "", code: -1, stdout: "", stderr: text};
                            }
                        );
                    }
                }
            }, this));
        }, this));
    };

    Conformance.prototype.handleToolbarUI = function() {
        var compilerCount = this.selectorList.children().length;

        // Only allow new compilers if we allow for more
        this.addCompilerButton.attr("disabled", compilerCount >= this.maxCompilations);

        this.setTitle(compilerCount);
    };

    Conformance.prototype.handleStatusIcon = function(element, status) {
        if (!element) return;
        element.attr("class", "status glyphicon glyphicon-" + (status.code === 3 ? "remove-sign" : (status.code === 2 ? "info-sign" : "ok-sign")))
            .css("visibility", status.code === 0 ? "hidden" : "visible")
            .css("color",  status.code === 3 ? "red" : (status.code === 2 ? "yellow" : "green"))
            .attr("title", status.text)
            .attr("aria-label", status.code === 3 ? "Compilation failed!" : (status.code === 2 ? "Compiled with warnings" : "Compiled without warnings"))
            .attr("data-status", status.code);
    };

    Conformance.prototype.currentState = function () {
        var state = {
            editorid: this.editorId,
            source: this.source,
            compilers: []
        };
        _.each(this.selectorList.children(), _.bind(function(child) {
            var status = $(child).find('.status');
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

    };
    

    return {
        Conformance: Conformance
    };
});
