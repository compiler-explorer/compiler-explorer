// Copyright (c) 2017, Compiler Explorer Authors
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
var CompilerPicker = require('../compiler-picker');
var utils = require('../utils');

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
    this.compilerPickers = [];
    this.currentLibs = [];

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
    var newLibs = this.libsWidget.get();
    if (newLibs !== this.currentLibs) {
        this.currentLibs = newLibs;
        this.saveState();
        this.compileAll();
    }
};

Conformance.prototype.initLibraries = function (state) {
    this.libsWidget = new Libraries.Widget(this.langId, null, this.libsButton, state, _.bind(this.onLibsChanged, this));
    // No callback is done on initialization, so make sure we store the current libs
    this.currentLibs = this.libsWidget.get();
};

Conformance.prototype.initButtons = function () {
    this.conformanceContentRoot = this.domRoot.find('.conformance-wrapper');
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
        this.addCompilerPicker();
        this.saveState();
    }, this));
};

Conformance.prototype.getPaneName = function () {
    return 'Conformance Viewer (Editor #' + this.editorId + ')';
};

Conformance.prototype.setTitle = function (compilerCount) {
    var compilerText = '';
    if (compilerCount !== 0) {
        compilerText = ' ' + compilerCount + '/' + this.maxCompilations;
    }
    this.container.setTitle(this.getPaneName() + compilerText);
};

Conformance.prototype.addCompilerPicker = function (config) {
    if (!config) {
        config = {
            // Compiler id which is being used
            compilerId: '',
            // Options which are in use
            options: options.compileOptions[this.langId],
        };
    }
    var newSelector = this.selectorTemplate.clone();
    var newCompilerEntry = {
        parent: newSelector,
        picker: null,
        optionsField: null,
        statusIcon: null,
        prependOptions: null,
    };

    var onOptionsChange = _.debounce(_.bind(function () {
        this.saveState();
        this.compileChild(newCompilerEntry);
    }, this), 800);

    newCompilerEntry.optionsField = newSelector.find('.conformance-options')
        .val(config.options)
        .on('change', onOptionsChange)
        .on('keyup', onOptionsChange);

    newSelector.find('.close').not('.extract-compiler')
        .on('click', _.bind(function () {
            this.removeCompilerPicker(newCompilerEntry);
        }, this));

    newCompilerEntry.statusIcon = newSelector.find('.status-icon');
    newCompilerEntry.prependOptions = newSelector.find('.prepend-options');
    var popCompilerButton = newSelector.find('.extract-compiler');

    var onCompilerChange = _.bind(function (compilerId) {
        popCompilerButton.toggleClass('d-none', !compilerId);
        this.saveState();
        // Hide the results icon when a new compiler is selected
        this.handleStatusIcon(newCompilerEntry.statusIcon, {code: 0});
        var compiler = this.compilerService.findCompiler(this.langId, compilerId);
        if (compiler) this.setCompilationOptionsPopover(newCompilerEntry.prependOptions, compiler.options);
        this.updateLibraries();
        this.compileChild(newCompilerEntry);
    }, this);

    newCompilerEntry.picker = new CompilerPicker(
        $(newSelector[0]), this.hub, this.langId,
        config.compilerId, _.bind(onCompilerChange, this)
    );

    var getCompilerConfig = _.bind(function () {
        return Components.getCompilerWith(
            this.editorId, undefined, newCompilerEntry.optionsField.val(),
            newCompilerEntry.picker.lastCompilerId, this.langId, this.lastState.libs
        );
    }, this);

    this.container.layoutManager.createDragSource(popCompilerButton, getCompilerConfig);

    popCompilerButton.click(_.bind(function () {
        var insertPoint = this.hub.findParentRowOrColumn(this.container) ||
            this.container.layoutManager.root.contentItems[0];
        insertPoint.addChild(getCompilerConfig);
    }, this));

    this.selectorList.append(newSelector);
    this.compilerPickers.push(newCompilerEntry);

    this.handleToolbarUI();
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

Conformance.prototype.removeCompilerPicker = function (compilerEntry) {
    this.compilerPickers = _.reject(this.compilerPickers, function (entry) {
        return compilerEntry.picker.id === entry.picker.id;
    });
    compilerEntry.picker.tomSelect.close();
    compilerEntry.parent.remove();

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

function hasResultAnyOutput(result) {
    return (result.stdout || []).length > 0 || (result.stderr || []).length > 0;
}

Conformance.prototype.handleCompileOutIcon = function (element, result) {
    var hasOutput = hasResultAnyOutput(result);
    element.toggleClass('d-none', !hasOutput);
    if (hasOutput) {
        this.compilerService.handleOutputButtonTitle(element, result);
    }
};

Conformance.prototype.onCompileResponse = function (compilerEntry, result) {
    var compilationOptions = '';
    if (result.compilationOptions) {
        compilationOptions = result.compilationOptions.join(' ');
    }

    this.setCompilationOptionsPopover(compilerEntry.prependOptions, compilationOptions);

    this.handleCompileOutIcon(compilerEntry.parent.find('.compiler-out'), result);

    this.handleStatusIcon(compilerEntry.statusIcon, this.compilerService.calculateStatusIcon(result));
    this.saveState();
};

function getCompilerId(compilerEntry) {
    if (compilerEntry && compilerEntry.picker && compilerEntry.picker.tomSelect) {
        return compilerEntry.picker.tomSelect.getValue();
    }
    return '';
}

Conformance.prototype.compileChild = function (compilerEntry) {
    var compilerId = getCompilerId(compilerEntry);
    if (compilerId === '') return;
    // Hide previous status icons
    this.handleStatusIcon(compilerEntry.statusIcon, {code: 4});

    this.expandSource().then(_.bind(function (expandedSource) {
        var request = {
            source: expandedSource,
            compiler: compilerId,
            options: {
                userArguments: compilerEntry.optionsField.val() || '',
                filters: {},
                compilerOptions: {produceAst: false, produceOptInfo: false, skipAsm: true},
                libraries: [],
            },
            lang: this.langId,
            files: [],
        };

        _.each(this.currentLibs, function (item) {
            request.options.libraries.push({
                id: item.name,
                version: item.ver,
            });
        });

        // This error function ensures that the user will know we had a problem (As we don't save asm)
        this.compilerService.submit(request)
            .then(_.bind(function (x) {
                this.onCompileResponse(compilerEntry, x.result);
            }, this))
            .catch(_.bind(function (x) {
                this.onCompileResponse(compilerEntry, {
                    asm: '',
                    code: -1,
                    stdout: '',
                    stderr: x.error,
                });
            }, this));
    }, this));
};

Conformance.prototype.compileAll = function () {
    _.each(this.compilerPickers, _.bind(function (compilerEntry) {
        this.compileChild(compilerEntry);
    }, this));
};

Conformance.prototype.handleToolbarUI = function () {
    var compilerCount = this.compilerPickers.length;

    // Only allow new compilers if we allow for more
    this.addCompilerButton.prop('disabled', compilerCount >= this.maxCompilations);

    this.setTitle(compilerCount);
};

Conformance.prototype.handleStatusIcon = function (statusIcon, status) {
    this.compilerService.handleCompilationStatus(null, statusIcon, status);
};

Conformance.prototype.currentState = function () {
    var compilers = _.map(this.compilerPickers, function (compilerEntry) {
        return {
            compilerId: getCompilerId(compilerEntry),
            options: compilerEntry.optionsField.val() || '',
        };
    });
    return {
        editorid: this.editorId,
        langId: this.langId,
        compilers: compilers,
        libs: this.currentLibs,
    };
};

Conformance.prototype.saveState = function () {
    this.lastState = this.currentState();
    this.container.setState(this.lastState);
};

Conformance.prototype.resize = function () {
    // The pane becomes unusable long before this hides the icons
    // Added either way just in case we ever add more icons to this pane
    var topBarHeight = utils.updateAndCalcTopBarHeight(this.domRoot, this.topBar, this.hideable);
    this.conformanceContentRoot.outerHeight(this.domRoot.height() - topBarHeight);
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
            _.map(this.compilerPickers, function (compilerEntry) {
                return getCompilerId(compilerEntry);
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
        _.each(this.compilerPickers, function (compilerEntry) {
            compilerEntry.picker.tomSelect.close();
            compilerEntry.parent.remove();
        });
        this.compilerPickers = [];
        var langState = this.stateByLang[newLangId];
        this.initFromState(langState);
        this.updateLibraries();
        this.handleToolbarUI();
        this.saveState();
    }
};

Conformance.prototype.close = function () {
    this.eventHub.unsubscribe();
    _.each(this.compilerPickers, function (compilerEntry) {
        compilerEntry.picker.tomSelect.close();
        compilerEntry.parent.remove();
    });
    this.eventHub.emit('conformanceViewClose', this.editorId);
};

Conformance.prototype.initFromState = function (state) {
    if (state && state.compilers) {
        this.lastState = state;
        _.each(state.compilers, _.bind(this.addCompilerPicker, this));
    } else {
        this.lastState = this.currentState();
    }
};

module.exports = {
    Conformance: Conformance,
};
