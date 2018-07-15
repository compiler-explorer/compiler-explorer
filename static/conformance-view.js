// Copyright (c) 2017, Rubén Rincón
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

"use strict";

var options = require('options');
var _ = require('underscore');
var $ = require('jquery');
var Promise = require('es6-promise').Promise;
var ga = require('./analytics');

require('selectize');

function Conformance(hub, container, state) {
    this.container = container;
    this.eventHub = hub.createEventHub();
    this.compilerService = hub.compilerService;
    this.domRoot = container.getElement();
    this.domRoot.html($('#conformance').html());
    this.editorId = state.editorid;
    this.maxCompilations = options.cvCompilerCountMax || 6;
    this.langId = state.langId || _.keys(options.languages)[0];
    this.source = state.source || "";
    this.sourceNeedsExpanding = true;
    this.expandedSource = null;

    this.status = {
        allowCompile: false,
        allowAdd: true
    };
    this.stateByLang = {};

    this.initButtons();
    this.initCallbacks();
    this.initFromState(state);
    this.handleToolbarUI();
    ga.proxy('send', {
        hitType: 'event',
        eventCategory: 'ViewPane',
        eventAction: 'Open',
        eventValue: 'ConfView'
    });
}

Conformance.prototype.initButtons = function () {
    this.selectorList = this.domRoot.find('.compiler-list');
    this.addCompilerButton = this.domRoot.find('.add-compiler');
    this.selectorTemplate = $('#compiler-selector').find('.compiler-row');
    this.topBar = this.domRoot.find('.top-bar');

    this.hideable = this.domRoot.find('.hideable');
};

Conformance.prototype.initCallbacks = function () {
    this.container.on('destroy', function () {
        this.eventHub.unsubscribe();
        this.eventHub.emit("conformanceViewClose", this.editorId);
    }, this);

    this.container.on('destroy', this.close, this);
    this.container.on('open', function () {
        this.eventHub.emit("conformanceViewOpen", this.editorId);
    }, this);

    this.container.on('resize', this.resize, this);
    this.container.on('shown', this.resize, this);
    this.eventHub.on('resize', this.resize, this);
    this.eventHub.on('editorChange', this.onEditorChange, this);
    this.eventHub.on('editorClose', this.onEditorClose, this);
    this.eventHub.on('languageChange', this.onLanguageChange, this);

    this.addCompilerButton.on('click', _.bind(function () {
        this.addCompilerSelector();
        this.saveState();
    }, this));
};

Conformance.prototype.setTitle = function (compilerCount) {
    this.container.setTitle("Conformance viewer (Editor #" + this.editorId + ") " + (
        compilerCount !== 0 ? (compilerCount + "/" + this.maxCompilations) : ""
    ));
};

Conformance.prototype.getGroupsInUse = function () {
    var currentLangCompilers = _.map(this.compilerService.getCompilersForLang(this.langId), _.identity);
    return _.map(_.uniq(currentLangCompilers, false, function (compiler) {
        return compiler.group;
    }), function (compiler) {
        return {value: compiler.group, label: compiler.groupName || compiler.group};
    });
};

Conformance.prototype.addCompilerSelector = function (config) {
    if (!config) {
        config = {
            // Compiler id which is being used
            compilerId: "",
            // Options which are in use
            options: ""
        };
    }

    var newEntry = this.selectorTemplate.clone();

    var onOptionsChange = _.debounce(_.bind(function () {
        this.saveState();
        this.compileChild(newEntry);
    }, this), 800);

    newEntry.find('.options')
        .val(config.options)
        .on("change", onOptionsChange)
        .on("keyup", onOptionsChange);

    newEntry.find('.close')
        .on("click", _.bind(function () {
            this.removeCompilerSelector(newEntry);
        }, this));

    this.selectorList.append(newEntry);

    var status = newEntry.find('.status');
    var prependOptions = newEntry.find('.prepend-options');
    var langId = this.langId;
    var isVisible = function (compiler) {
        return compiler.lang === langId;
    };

    var onCompilerChange = _.bind(function (compilerId) {
        // Hide the results icon when a new compiler is selected
        this.handleStatusIcon(status, {code: 0, text: ""});
        var compiler = this.compilerService.findCompiler(langId, compilerId);
        if (compiler) {
            prependOptions.prop('title', compiler.options);
            prependOptions.toggle(!!compiler.options);
        } else {
            prependOptions.hide();
        }
    }, this);

    newEntry.find('.compiler-picker')
        .selectize({
            sortField: [
                {field: '$order'},
                {field: 'name'}
            ],
            valueField: 'id',
            labelField: 'name',
            searchField: ['name'],
            optgroupField: 'group',
            optgroups: this.getGroupsInUse(),
            options: _.filter(options.compilers, isVisible),
            items: config.compilerId ? [config.compilerId] : []
        })
        .on('change', _.bind(function (e) {
            onCompilerChange($(e.target).val());
            this.compileChild(newEntry);
        }, this));
    onCompilerChange(config.compilerId);
    this.handleToolbarUI();
    this.saveState();
};

Conformance.prototype.removeCompilerSelector = function (element) {
    if (element) element.remove();
    this.handleToolbarUI();
    this.saveState();
};

Conformance.prototype.expandSource = function () {
    if (this.sourceNeedsExpanding || !this.expandedSource) {
        return this.compilerService.expand(this.source).then(_.bind(function (expandedSource) {
            this.expandedSource = expandedSource;
            this.sourceNeedsExpanding = false;
            return expandedSource;
        }, this));
    }
    return Promise.resolve(this.expandedSource);
};

Conformance.prototype.onEditorChange = function (editorId, newSource, langId) {
    if (editorId === this.editorId) {
        this.langId = langId;
        this.source = newSource;
        this.sourceNeedsExpanding = true;
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

Conformance.prototype.onCompileResponse = function (child, result) {
    var allText = _.pluck((result.stdout || []).concat(result.stderr || []), 'text').join("\n");
    var failed = result.code !== 0;
    var warns = !failed && !!allText;
    var status = {
        text: allText.replace(/\x1b\\[[0-9;]*m/, ''),
        code: failed ? 3 : (warns ? 2 : 1)
    };

    child.find('.prepend-options')
        .toggle(!!result.compilationOptions)
        .prop('title', result.compilationOptions ? result.compilationOptions.join(' ') : '');

    this.handleStatusIcon(child.find('.status'), status);
    this.saveState();
};

Conformance.prototype.compileChild = function (child) {
    // Hide previous status icons
    var picker = child.find('.compiler-picker');

    if (!picker || !picker.val()) return;
    this.handleStatusIcon(child.find('.status'), {
        code: 4, // Compiling code
        text: "Compiling"
    });
    this.expandSource().then(_.bind(function (expandedSource) {
        var request = {
            source: expandedSource,
            compiler: picker.val(),
            options: {
                userArguments: child.find(".options").val(),
                filters: {},
                compilerOptions: {produceAst: false, produceOptInfo: false}
            }
        };
        // This error function ensures that the user will know we had a problem (As we don't save asm)
        this.compilerService.submit(request)
            .then(_.bind(function (x) {
                this.onCompileResponse(child, x.result);
            }, this))
            .catch(_.bind(function (x) {
                this.onCompileResponse(child, {
                    asm: "",
                    code: -1,
                    stdout: "",
                    stderr: x.error
                });
            }, this));
    }, this));
};

Conformance.prototype.compileAll = function () {
    _.each(this.selectorList.children(), _.bind(function (child) {
        this.compileChild($(child));
    }, this));
};

Conformance.prototype.handleToolbarUI = function () {
    var compilerCount = this.selectorList.children().length;

    // Only allow new compilers if we allow for more
    this.addCompilerButton.prop("disabled", compilerCount >= this.maxCompilations);

    this.setTitle(compilerCount);
};

Conformance.prototype.handleStatusIcon = function (element, status) {
    if (!element) return;

    function ariaLabel(code) {
        if (code === 4) return "Compiling";
        if (code === 3) return "Compilation failed";
        if (code === 2) return "Compiled with warnings";
        return "Compiled without warnings";
    }

    function color(code) {
        if (code === 4) return "black";
        if (code === 3) return "red";
        if (code === 2) return "yellow";
        return "green";
    }

    element
        .toggleClass('status', true)
        .css("color", color(status.code))
        .toggle(status.code !== 0)
        .prop("title", status.text)
        .prop("aria-label", ariaLabel(status.code))
        .prop("data-status", status.code);

    element.toggleClass('glyphicon-tasks', status.code === 4);
    element.toggleClass('glyphicon-remove-sign', status.code === 3);
    element.toggleClass('glyphicon-info-sign', status.code === 2);
    element.toggleClass('glyphicon-ok-sign', status.code === 1);
};

Conformance.prototype.currentState = function () {
    var state = {
        editorid: this.editorId,
        langId: this.langId,
        compilers: []
    };
    _.each(this.selectorList.children(), _.bind(function (child) {
        child = $(child);
        state.compilers.push({
            // Compiler which is being used
            compilerId: child.find('.compiler-picker').val() || "",
            // Options which are in use
            options: child.find(".options").val() || ""
        });
    }, this));
    return state;
};

Conformance.prototype.saveState = function () {
    this.container.setState(this.currentState());
};

Conformance.prototype.resize = function () {
    this.updateHideables();
    this.selectorList.css("height", this.domRoot.height() - this.topBar.outerHeight(true));
};

Conformance.prototype.updateHideables = function () {
    this.hideable.toggle(this.domRoot.width() > this.addCompilerButton.width());
};

Conformance.prototype.onLanguageChange = function (editorId, newLangId) {
    if (editorId === this.editorId && this.langId !== newLangId) {
        var oldLangId = this.langId;
        this.stateByLang[oldLangId] = this.currentState();

        this.langId = newLangId;
        this.selectorList.children().remove();
        if (this.stateByLang[newLangId]) {
            this.initFromState(this.stateByLang[newLangId]);
        }
        this.handleToolbarUI();
        this.saveState();
    }
};

Conformance.prototype.close = function () {
    this.eventHub.unsubscribe();
    this.eventHub.emit("conformanceViewClose", this.editorId);
};

Conformance.prototype.initFromState = function (state) {
    if (state.compilers) {
        _.each(state.compilers, _.bind(this.addCompilerSelector, this));
    }
};
module.exports = {
    Conformance: Conformance
};
