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

var options = require('../options');
var _ = require('underscore');
var $ = require('jquery');
var Promise = require('es6-promise').Promise;
var ga = require('../analytics');
var Components = require('../components');

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
    this.source = state.source || "";
    this.sourceNeedsExpanding = true;
    this.expandedSource = null;

    this.status = {
        allowCompile: false,
        allowAdd: true
    };
    this.stateByLang = {};

    this.initButtons();
    this.initLibraries(state);
    this.updateLibsDropdown();
    this.initCallbacks();
    this.initFromState(state);
    this.handleToolbarUI();
    ga.proxy('send', {
        hitType: 'event',
        eventCategory: 'OpenViewPane',
        eventAction: 'Conformance'
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

Conformance.prototype.initLibraries = function (state) {
    this.availableLibs = {};
    this.updateAvailableLibs();
    _.each(state.libs, _.bind(function (lib) {
        this.markLibraryAsUsed(lib.name, lib.ver);
    }, this));
};

Conformance.prototype.updateAvailableLibs = function () {
    if (!this.availableLibs[this.langId]) {
        this.availableLibs[this.langId] = $.extend(true, {}, options.libs[this.langId]);
        this.initLangDefaultLibs();
    }
};

Conformance.prototype.initLangDefaultLibs = function () {
    var defaultLibs = options.defaultLibs[this.langId];
    if (!defaultLibs) return;
    _.each(defaultLibs.split(':'), _.bind(function (libPair) {
        var pairSplits = libPair.split('.');
        if (pairSplits.length === 2) {
            var lib = pairSplits[0];
            var ver = pairSplits[1];
            this.markLibraryAsUsed(lib, ver);
        }
    }, this));
};

Conformance.prototype.updateLibsDropdown = function () {
    this.updateAvailableLibs();
    this.libsButton.popover({
        container: 'body',
        content: _.bind(function () {
            var libsCount = _.keys(this.availableLibs[this.langId]).length;
            if (libsCount === 0) return this.noLibsPanel;
            var columnCount = Math.ceil(libsCount / 5);
            var currentLibIndex = -1;

            var libLists = [];
            for (var i = 0; i < columnCount; i++) {
                libLists.push($('<ul></ul>').addClass('lib-list'));
            }

            // Utility function so we can iterate indefinitely over our lists
            var getNextList = function () {
                currentLibIndex = (currentLibIndex + 1) % columnCount;
                return libLists[currentLibIndex];
            };

            var handleArrow = function (libGroup, libArrow) {
                var anyInUse = _.any(libGroup.children().children('input'), function (element) {
                    return $(element).prop('checked');
                });
                var isVisible = libGroup.is(":visible");

                libArrow.toggleClass('lib-arrow-up', isVisible);
                libArrow.toggleClass('lib-arrow-down', !isVisible);
                libArrow.toggleClass('lib-arrow-used', anyInUse);
            };

            var onChecked = _.bind(function (e) {
                var elem = $(e.target);
                // Uncheck every lib checkbox with the same name if we're checking the target
                if (elem.prop('checked')) {
                    _.each(e.data.group.children().children('input'), function (other) {
                        $(other).prop('checked', false);
                    });
                    // Recheck the targeted one
                    elem.prop('checked', true);
                }
                // And now do the same with the availableLibs object
                _.each(this.availableLibs[this.langId][elem.prop('data-lib')].versions, function (version) {
                    version.used = false;
                });
                this.availableLibs[this.langId][elem.prop('data-lib')]
                    .versions[elem.prop('data-version')].used = elem.prop('checked');

                handleArrow(e.data.group, e.data.arrow);

                this.saveState();
                this.compileAll();
            }, this);

            _.each(this.availableLibs[this.langId], function (lib, libKey) {
                var libsList = getNextList();
                var libArrow = $('<div></div>').addClass('lib-arrow');
                var libName = $('<span></span>').text(lib.name);
                var libHeader = $('<span></span>')
                    .addClass('lib-header')
                    .append(libArrow)
                    .append(libName);
                if (lib.url && lib.url.length > 0) {
                    libHeader.append($('<a></a>')
                        .css("float", "right")
                        .addClass('opens-new-window')
                        .prop('href', lib.url)
                        .prop('target', '_blank')
                        .prop('rel', 'noopener noreferrer')
                        .append($('<sup></sup>')
                            .addClass('fas fa-external-link-alt ')
                        )
                    );
                }
                if (lib.description && lib.description.length > 0) {
                    libName
                        .addClass('lib-described')
                        .prop('title', lib.description);
                }
                var libCat = $('<li></li>')
                    .append(libHeader)
                    .addClass('lib-item');

                var libGroup = $('<div></div>');

                if (libsList.children().length > 0)
                    libsList.append($('<hr>').addClass('lib-separator'));

                _.each(lib.versions, function (version, vKey) {
                    var verCheckbox = $('<input type="checkbox">')
                        .addClass('lib-checkbox')
                        .prop('data-lib', libKey)
                        .prop('data-version', vKey)
                        .prop('checked', version.used)
                        .prop('name', libKey)
                        .on('change', {arrow: libArrow, group: libGroup}, onChecked);
                    libGroup
                        .append($('<div></div>')
                            .append(verCheckbox)
                            .append($('<label></label>')
                                .addClass('lib-label')
                                .text(version.version)
                                .on('click', function () {
                                    verCheckbox.trigger('click');
                                })
                            )
                        );
                });

                libGroup.hide();
                handleArrow(libGroup, libArrow);

                libHeader.on('click', function () {
                    libGroup.toggle();
                    handleArrow(libGroup, libArrow);
                });

                libGroup.appendTo(libCat);
                libCat.appendTo(libsList);
            });
            return $('<div></div>').addClass('libs-container').append(libLists);
        }, this),
        html: true,
        placement: 'bottom',
        trigger: 'manual'
    }).click(_.bind(function () {
        this.libsButton.popover('show');
    }, this)).on('show.bs.popover', function () {
    });
};

Conformance.prototype.markLibraryAsUsed = function (name, version) {
    if (this.availableLibs[this.langId] &&
        this.availableLibs[this.langId][name] &&
        this.availableLibs[this.langId][name].versions[version]) {

        this.availableLibs[this.langId][name].versions[version].used = true;
    }
};

Conformance.prototype.initButtons = function () {
    this.selectorList = this.domRoot.find('.compiler-list');
    this.addCompilerButton = this.domRoot.find('.add-compiler');
    this.selectorTemplate = $('#compiler-selector').find('.form-row');
    this.topBar = this.domRoot.find('.top-bar');
    this.libsButton = this.topBar.find('.show-libs');
    this.libsTemplates = $('.template #libs-dropdown');
    this.noLibsPanel = this.libsTemplates.children('.no-libs');
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
    return _.chain(this.compilerService.getCompilersForLang(this.langId))
        .map()
        .uniq(false, function (compiler) {
            return compiler.group;
        })
        .map(function (compiler) {
            return {value: compiler.group, label: compiler.groupName || compiler.group};
        })
        .sortBy('label')
        .value();
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

    var optionsField = newEntry.find('.options')
        .val(config.options)
        .on("change", onOptionsChange)
        .on("keyup", onOptionsChange);

    newEntry.find('.close').not('.extract-compiler')
        .on("click", _.bind(function () {
            this.removeCompilerSelector(newEntry);
        }, this));

    this.selectorList.append(newEntry);

    var status = newEntry.find('.status-icon');
    var prependOptions = newEntry.find('.prepend-options');
    var langId = this.langId;
    var isVisible = function (compiler) {
        return compiler.lang === langId;
    };

    var popCompilerButton = newEntry.find('.extract-compiler');

    var onCompilerChange = _.bind(function (compilerId) {
        popCompilerButton.toggleClass('d-none', !compilerId);
        // Hide the results icon when a new compiler is selected
        this.handleStatusIcon(status, {code: 0});
        var compiler = this.compilerService.findCompiler(langId, compilerId);
        if (compiler) this.setCompilationOptionsPopover(prependOptions, compiler.options);
    }, this);

    var compilerPicker = newEntry.find('.compiler-picker')
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
            items: config.compilerId ? [config.compilerId] : [],
            dropdownParent: 'body'
        })
        .on('change', _.bind(function (e) {
            onCompilerChange($(e.target).val());
            this.compileChild(newEntry);
        }, this));
    onCompilerChange(config.compilerId);


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

Conformance.prototype.setCompilationOptionsPopover = function (element, content) {
    element.popover('dispose');
    element.popover({
        content: content || 'No options in use',
        template: '<div class="popover' +
            (content ? ' compiler-options-popover' : '')  +
            '" role="tooltip"><div class="arrow"></div>' +
            '<h3 class="popover-header"></h3><div class="popover-body"></div></div>'
    });
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

    this.setCompilationOptionsPopover(child.find('.prepend-options'),
        result.compilationOptions ? result.compilationOptions.join(' ') : '');
    child.find('.compiler-out')
        .prop('title', allText.replace(/\x1b\[[0-9;]*m(.\[K)?/g, ''))
        .toggleClass('d-none', !allText);
    this.handleStatusIcon(child.find('.status-icon'), {code: failed ? 3 : (warns ? 2 : 1)});
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
                userArguments: child.find(".options").val() || "",
                filters: {},
                compilerOptions: {produceAst: false, produceOptInfo: false}
            }
        };
        var compiler = this.compilerService.findCompiler(this.langId, picker.val());
        var includeFlag = compiler ? compiler.includeFlag : '-I';
        _.each(this.availableLibs[this.langId], function (lib) {
            _.each(lib.versions, function (version) {
                if (version.used) {
                    _.each(version.path, function (path) {
                        request.options.userArguments += ' ' + includeFlag + path;
                    });
                }
            });
        });
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
        if (code === 2) return "#BBBB00";
        return "green";
    }

    element
        .removeClass()
        .addClass('status-icon fas fa-minus-circle')
        .css("color", color(status.code))
        .prop("aria-label", ariaLabel(status.code))
        .prop("data-status", status.code)
        .toggle(status.code !== 0)
        .toggleClass('fa-spinner', status.code === 4)
        .toggleClass('fa-minus-circle', status.code === 3)
        .toggleClass('fa-exclamation-circle', status.code === 2)
        .toggleClass('fa-check-circle', status.code === 1);
};

Conformance.prototype.currentState = function () {
    var libs = [];
    _.each(this.availableLibs[this.langId], function (library, name) {
        _.each(library.versions, function (version, ver) {
            if (library.versions[ver].used) {
                libs.push({name: name, ver: ver});
            }
        });
    });
    var compilers = _.map(this.selectorList.children(), function (child) {
        child = $(child);
        return {
            compilerId: child.find('.compiler-picker').val() || "",
            options: child.find(".options").val() || ""
        };
    });
    return {
        editorid: this.editorId,
        langId: this.langId,
        compilers: compilers,
        libs: libs
    };
};

Conformance.prototype.saveState = function () {
    this.lastState = this.currentState();
    this.container.setState(this.lastState);
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
        var langState = this.stateByLang[newLangId];
        this.initFromState(langState);
        this.updateLibsDropdown();
        this.handleToolbarUI();
        this.saveState();
    }
};

Conformance.prototype.close = function () {
    this.eventHub.unsubscribe();
    this.eventHub.emit("conformanceViewClose", this.editorId);
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
    Conformance: Conformance
};
