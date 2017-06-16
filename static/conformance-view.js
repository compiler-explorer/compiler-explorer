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

    var Compiler = require('./compiler');

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
        this.compileButton = this.domRoot.find('.compile');
        this.editorId = state.editorid;
        this.source = state.source || "";
        this.currentExpandedSource = "";
        this.nextSelectorId = 0;

        this.maxCompilations = 6;

        this.status = {
            allowCompile: false,
            allowAdd: true
        };

        this.container.on('destroy', function () {
            this.eventHub.emit("conformanceViewClose", this.editorId);
            this.eventHub.unsubscribe();
        }, this);

        this.eventHub.on('editorChange', this.onEditorChange, this);
        this.eventHub.on('editorClose', this.onEditorClose, this);
        this.eventHub.emit("conformanceViewOpen", this.editorId);

        this.addCompilerButton.on('click', _.bind(function() {
            this.addCompilerSelector();
        }, this));

        this.compileButton.on('click', _.bind(function() {
            Compiler.Compiler.prototype.expand(this.source).then(_.bind(function (expanded) {
                this.currentExpandedSource = expanded;
                // And trun off every icon
                this.selectorList.find('.status').css("visibility", "hidden");
                this.compileAll();
            }, this));
        }, this));

        this.handleUiStatus();

        this.setTitle();
    }

    Conformance.prototype.setTitle = function () {
        this.container.setTitle("Conformance viewer (Editor #" + this.editorId + ")");
    };

    Conformance.prototype.addCompilerSelector = function() {
        var self = this;
        var selectorsCount = this.nextSelectorId;
        this.nextSelectorId++;

        var newSelector = $("<select>")
            .attr("class", "compiler-picker")
            .attr("placeholder", "Select a compiler...")
            .attr("data-cv", selectorsCount);

        this.selectorList.append($("<tr>")
            .attr("data-cv", selectorsCount)
            .append($("<td>")
                .append($("<span>")
                    .attr("class", "status glyphicon glyphicon-signal")
                    .css("visibility", "hidden")
                )
            ).append($("<td>")
                .append(newSelector)
            ).append($("<td>")
                .append($("<input>")
                    .attr("class", "options form-control")
                    .attr("type", "text")
                    .attr("size","256")
                    .attr("placeholder", "Compiler options...")
                    .attr("data-cv", selectorsCount)
                )
            ).append($("<td>")
                .append($("<button>")
                    .attr("class", "close")
                    .attr("aria-label", "Close")
                    .attr("data-cv", selectorsCount)
                    .append($("<span>")
                        .html("&times;")
                        .attr("aria-hidden", "true")
                    )
                    .on("click", function() {
                        self.removeCompilerSelector(selectorsCount);
                    })
                )
            )
        );

        newSelector.selectize({
            sortField: 'name',
            valueField: 'id',
            labelField: 'name',
            searchField: ['name'],
            options: compilers,
            items: []
        }).on('change', function() {
            // Hide the results button when a new compiler is selected
            self.selectorList.find('[data-cv="' + selectorsCount + '"] .status').css("visibility", "hidden");
        });

        this.handleUiStatus();
    };

    Conformance.prototype.removeCompilerSelector = function(cv) {
        _.each(this.selectorList.children(), function(row) {
            var child = $(row);
            if (Number(child.attr("data-cv")) === cv) {
                child.remove();
            }
        }, this);

        this.handleUiStatus();
    };

    // TODO: Maybe use premises
    Conformance.prototype.compile = function(selector) {
        if (!selector) return;
        var val = selector.val();
        if (val) {
            var cv = Number(selector.attr("data-cv"));
            var request = {
                source: this.currentExpandedSource || "",
                compiler: val,
                options: this.selectorList.find(".options[data-cv='" + cv + "']").val(),
                backendOptions: {produceAst: false},
                filters: {},
                extras: {
                    cv: cv,
                    storeAsm: false,
                    emitCompilingEvent: false,
                    ignorePendingRequest: true
                }
            };
            Compiler.Compiler.prototype.sendCompile(request, _.bind(this.onCompileResponse, this));
        }
    };

    Conformance.prototype.onEditorChange = function(editorId, newSource) {
        if (editorId == this.editorId) this.source = newSource;
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
        var style = {
            color: failed ? "red" : (warns ? "yellow" : "green"),
            glyph: failed ? "remove-sign" : (warns ? "info-sign" : "ok-sign"),
            aria: failed ? "Compilation failed" : (warns ? "Compiled with warnings" : "Compiled without warnings")
        };
        var statusIcon = this.selectorList.find('[data-cv="' + request.extras.cv + '"] .status');

        statusIcon.attr("class", "status glyphicon glyphicon-" + style.glyph)
            .css("visibility", "visible")
            .css("color", style.color)
            .attr("title", allText)
            .attr("aria-label", style.aria);
    };

    Conformance.prototype.compileAll = function() {
        var compileCount = 0;
        _.each(this.selectorList.children(), _.bind(function(child) {
            var picker = $(child).find('.compiler-picker');
            if (picker && compileCount < this.maxCompilations) {
                compileCount++;
                this.compile(picker);
            }
        }, this));
    };

    Conformance.prototype.handleUiStatus = function() {
        this.status.allowCompile = this.selectorList.children().length > 0;
        this.status.allowAdd = this.selectorList.children().length < this.maxCompilations;

        this.compileButton.attr("disabled", !this.status.allowCompile);
        this.addCompilerButton.attr("disabled", !this.status.allowAdd);
    };

    return {
        Conformance: Conformance
    };
});
