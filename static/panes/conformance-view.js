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

'use strict';

var options = require('../options');
var _ = require('underscore');
var $ = require('jquery');
var Promise = require('es6-promise').Promise;
var ga = require('../analytics');
var Components = require('../components');
var Libraries = require('../libs-widget-ext');

require('selectize');

function Conformance(hub, container, state) {
    this.hub = hub;
    this.container = container;
    this.eventHub = hub.createEventHub();
    this.compilerService = hub.compilerService;
    this.domRoot = container.getElement();
    this.domRoot.html($('#conformance').html());
    this.editorId = state.editorid;
    this.maxCompilations = options.cvCompilerCountMax || 6;
    this.langId = state.langId || _.keys(options.languages)[0];
    this.source = state.source || '';
    this.sourceNeedsExpanding = true;
    this.expandedSource = null;

    this.status = {
        allowCompile: false,
        allowAdd: true,
    };
    this.stateByLang = {};

    this.initButtons();
    this.initLibraries(state);
    this.initCallbacks();
    this.initFromState(state);
    this.handleToolbarUI();
    ga.proxy('send', {
        hitType: 'event',
        eventCategory: 'OpenViewPane',
        eventAction: 'Conformance',
    });

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
}

Conformance.prototype.onLibsChanged = function () {
    this.saveState();
    this.compileAll();
};

Conformance.prototype.initLibraries = function (state) {
    this.libsWidget = new Libraries.Widget(this.langId, null,
        this.libsButton, state, _.bind(this.onLibsChanged, this));
};

Conformance.prototype.initButtons = function () {
    this.selectorList = this.domRoot.find('.compiler-list');
    this.addCompilerButton = this.domRoot.find('.add-compiler');
    this.selectorTemplate = $('#compiler-selector').find('.form-row');
    this.topBar = this.domRoot.find('.top-bar');
    this.libsButton = this.topBar.find('.show-libs');
    this.hideable = this.domRoot.find('.hideable');
};

Conformance.prototype.initCallbacks = function () {
    this.container.on('destroy', function () {
        this.eventHub.unsubscribe();
        this.eventHub.emit('conformanceViewClose', this.editorId);
    }, this);

    this.container.on('destroy', this.close, this);
    this.container.on('open', function () {
        this.eventHub.emit('conformanceViewOpen', this.editorId);
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
    this.container.setTitle('Conformance viewer (Editor #' + this.editorId + ') ' + (
        compilerCount !== 0 ? (compilerCount + '/' + this.maxCompilations) : ''
    ));
};

Conformance.prototype.addCompilerSelector = function (config) {
    if (!config) {
        config = {
            // Compiler id which is being used
            compilerId: '',
            // Options which are in use
            options: options.compileOptions[this.langId],
        };
    }

    var newEntry = this.selectorTemplate.clone();

    var onOptionsChange = _.debounce(_.bind(function () {
        this.saveState();
        this.compileChild(newEntry);
    }, this), 800);

    var optionsField = newEntry.find('.options')
        .val(config.options)
        .on('change', onOptionsChange)
        .on('keyup', onOptionsChange);

    newEntry.find('.close').not('.extract-compiler')
        .on('click', _.bind(function () {
            this.removeCompilerSelector(newEntry);
        }, this));

    this.selectorList.append(newEntry);

    var status = newEntry.find('.status-icon');
    var prependOptions = newEntry.find('.prepend-options');
    var popCompilerButton = newEntry.find('.extract-compiler');

    var onCompilerChange = _.bind(function (compilerId) {
        popCompilerButton.toggleClass('d-none', !compilerId);
        // Hide the results icon when a new compiler is selected
        this.handleStatusIcon(status, {code: 0});
        var compiler = this.compilerService.findCompiler(this.langId, compilerId);
        if (compiler) this.setCompilationOptionsPopover(prependOptions, compiler.options);
        this.updateLibraries();
    }, this);

    var compilerPicker = newEntry.find('.compiler-picker').selectize({
        sortField: this.compilerService.getSelectizerOrder(),
        valueField: 'id',
        labelField: 'name',
        searchField: ['name'],
        optgroupField: 'group',
        optgroups: this.compilerService.getGroupsInUse(this.langId),
        lockOptgroupOrder: true,
        options: _.filter(this.getCurrentLangCompilers(), function (e) {
            return !e.hidden || e.id === config.compilerId;
        }),
        items: config.compilerId ? [config.compilerId] : [],
        dropdownParent: 'body',
        closeAfterSelect: true,
    }).on('change', _.bind(function (e) {
        onCompilerChange($(e.target).val());
        this.compileChild(newEntry);
    }, this));


    var getCompilerConfig = _.bind(function () {
        return Components.getCompilerWith(
            this.editorId, undefined, optionsField.val(), compilerPicker.val(), this.langId, this.lastState.libs);
    }, this);

    this.container.layoutManager
        .createDragSource(popCompilerButton, getCompilerConfig);

    popCompilerButton.click(_.bind(function () {
        var insertPoint = this.hub.findParentRowOrColumn(this.container) ||
            this.container.layoutManager.root.contentItems[0];
        insertPoint.addChild(getCompilerConfig);
    }, this));

    this.handleToolbarUI();
    this.saveState();
};

Conformance.prototype.getCurrentLangCompilers = function () {
    return this.compilerService.getCompilersForLang(this.langId);
};

Conformance.prototype.setCompilationOptionsPopover = function (element, content) {
    element.popover('dispose');
    element.popover({
        content: content || 'No options in use',
        template: '<div class="popover' +
            (content ? ' compiler-options-popover' : '') +
            '" role="tooltip"><div class="arrow"></div>' +
            '<h3 class="popover-header"></h3><div class="popover-body"></div></div>',
    });
};

Conformance.prototype.removeCompilerSelector = function (element) {
    if (element) element.remove();
    this.updateLibraries();
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
    var allText = _.pluck((result.stdout || []).concat(result.stderr || []), 'text').join('\n');
    var failed = result.code !== 0;
    var warns = !failed && !!allText;

    this.setCompilationOptionsPopover(child.find('.prepend-options'),
        result.compilationOptions ? result.compilationOptions.join(' ') : '');
    child.find('.compiler-out')
        .prop('title', allText.replace(/\x1b\[[0-9;]*m(.\[K)?/g, ''))
        .toggleClass('d-none', !allText);
    this.handleStatusIcon(child.find('.status-icon'), {code: failed ? 3 : (warns ? 2 : 1), compilerOut: result.code});
    this.saveState();
};

Conformance.prototype.compileChild = function (child) {
    // Hide previous status icons
    var picker = child.find('.compiler-picker');

    if (!picker || !picker.val()) return;

    this.handleStatusIcon(child.find('.status-icon'), {code: 4});

    this.expandSource().then(_.bind(function (expandedSource) {
        var request = {
            source: expandedSource,
            compiler: picker.val(),
            options: {
                userArguments: child.find('.options').val() || '',
                filters: {},
                compilerOptions: {produceAst: false, produceOptInfo: false},
                libraries: [],
                skipAsm: true,
            },
            lang: this.langId,
        };

        _.each(this.libsWidget.getLibsInUse(), function (item) {
            request.options.libraries.push({
                id: item.libId,
                version: item.versionId,
            });
        });

        // This error function ensures that the user will know we had a problem (As we don't save asm)
        this.compilerService.submit(request)
            .then(_.bind(function (x) {
                this.onCompileResponse(child, x.result);
            }, this))
            .catch(_.bind(function (x) {
                this.onCompileResponse(child, {
                    asm: '',
                    code: -1,
                    stdout: '',
                    stderr: x.error,
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
    this.addCompilerButton.prop('disabled', compilerCount >= this.maxCompilations);

    this.setTitle(compilerCount);
};

Conformance.prototype.handleStatusIcon = function (element, status) {
    if (!element) return;

    function ariaLabel() {
        // Compiling...
        if (status.code === 4) return 'Compiling';
        if (status.compilerOut === 0) {
            // StdErr.length > 0
            if (status.code === 3) return 'Compilation succeeded with errors';
            // StdOut.length > 0
            if (status.code === 2) return 'Compilation succeeded with warnings';
            return 'Compilation succeeded';
        } else {
            // StdErr.length > 0
            if (status.code === 3) return 'Compilation failed with errors';
            // StdOut.length > 0
            if (status.code === 2) return 'Compilation failed with warnings';
            return 'Compilation failed';
        }
    }

    function color() {
        // Compiling...
        if (status.code === 4) return 'black';
        if (status.compilerOut === 0) {
            // StdErr.length > 0
            if (status.code === 3) return '#FF6645';
            // StdOut.length > 0
            if (status.code === 2) return '#FF6500';
            return '#12BB12';
        } else {
            // StdErr.length > 0
            if (status.code === 3) return '#FF1212';
            // StdOut.length > 0
            if (status.code === 2) return '#BB8700';
            return '#FF6645';
        }
    }

    element
        .removeClass()
        .addClass('status-icon fas')
        .css('color', color())
        .toggle(status.code !== 0)
        .prop('aria-label', ariaLabel())
        .prop('data-status', status.code)
        .toggleClass('fa-spinner', status.code === 4)
        .toggleClass('fa-times-circle', status.code === 3)
        .toggleClass('fa-check-circle', status.code === 1 || status.code === 2);
};

Conformance.prototype.currentState = function () {
    var compilers = _.map(this.selectorList.children(), function (child) {
        child = $(child);
        return {
            compilerId: child.find('.compiler-picker').val() || '',
            options: child.find('.options').val() || '',
        };
    });
    return {
        editorid: this.editorId,
        langId: this.langId,
        compilers: compilers,
        libs: this.libsWidget.get(),
    };
};

Conformance.prototype.saveState = function () {
    this.lastState = this.currentState();
    this.container.setState(this.lastState);
};

Conformance.prototype.resize = function () {
    this.updateHideables();
    this.selectorList.css('height', this.domRoot.height() - this.topBar.outerHeight(true));
};

Conformance.prototype.updateHideables = function () {
    this.hideable.toggle(this.domRoot.width() > this.addCompilerButton.width());
};

Conformance.prototype.getOverlappingLibraries = function (compilerIds) {
    var compilers = _.map(compilerIds, _.bind(function (compilerId) {
        return this.compilerService.findCompiler(this.langId, compilerId);
    }, this));

    var libraries = {};
    var first = true;
    _.forEach(compilers, function (compiler) {
        if (compiler) {
            if (first) {
                libraries = _.extend({}, compiler.libs);
                first = false;
            } else {
                var libsInCommon = _.intersection(_.keys(libraries),
                    _.keys(compiler.libs));
    
                _.forEach(libraries, function (lib, libkey) {
                    if (libsInCommon.includes(libkey)) {
                        var versionsInCommon = _.intersection(_.keys(lib.versions),
                            _.keys(compiler.libs[libkey].versions));

                        libraries[libkey].versions = _.pick(lib.versions,
                            function (version, versionkey) {
                                return versionsInCommon.includes(versionkey);
                            });
                    } else {
                        libraries[libkey] = false;
                    }
                });
    
                libraries = _.omit(libraries, function (lib) {
                    return !lib || _.isEmpty(lib.versions);
                });
            }
        }
    });

    return libraries;
};

Conformance.prototype.updateLibraries = function () {
    var compilerIds = _.uniq(
        _.filter(
            _.map(this.selectorList.children(), function (child) {
                return $(child).find('.compiler-picker').val();
            })
            , function (compilerId) {
                return compilerId !== '';
            })
    );

    var libraries = this.getOverlappingLibraries(compilerIds);

    this.libsWidget.setNewLangId(this.langId, compilerIds.join('|'), libraries);
};

Conformance.prototype.onLanguageChange = function (editorId, newLangId) {
    if (editorId === this.editorId && this.langId !== newLangId) {
        var oldLangId = this.langId;
        this.stateByLang[oldLangId] = this.currentState();

        this.langId = newLangId;
        this.selectorList.children().remove();
        var langState = this.stateByLang[newLangId];
        this.initFromState(langState);
        this.updateLibraries();
        this.handleToolbarUI();
        this.saveState();
    }
};

Conformance.prototype.close = function () {
    this.eventHub.unsubscribe();
    this.eventHub.emit('conformanceViewClose', this.editorId);
};

Conformance.prototype.initFromState = function (state) {
    if (state && state.compilers) {
        this.lastState = state;
        _.each(state.compilers, _.bind(this.addCompilerSelector, this));
    } else {
        this.lastState = this.currentState();
    }
};

module.exports = {
    Conformance: Conformance,
};
