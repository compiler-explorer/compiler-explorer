// Copyright (c) 2018, Compiler Explorer Authors
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

var options = require('options'),
    _ = require('underscore'),
    $ = require('jquery');

function LibsWidget(langId, compiler, dropdownButton, state, onChangeCallback) {
    this.dropdownButton = dropdownButton;
    var possibleLibs = false;
    if (compiler) {
        this.currentCompilerId = compiler.id;
        possibleLibs = compiler.libs;
    } else {
        this.currentCompilerId = '_default_';
    }
    this.currentLangId = langId;
    this.initButtons();
    this.domRoot = null;
    this.onChangeCallback = onChangeCallback;
    this.availableLibs = {};
    this.updateAvailableLibs(possibleLibs);
    _.each(state.libs, _.bind(function (lib) {
        this.markLibrary(lib.name, lib.ver, true);
    }, this));
}

LibsWidget.prototype.initButtons = function () {
    this.noLibsPanel = $('#libs-dropdown .no-libs');
    this.libsEntry = $('#libs-entry .input-group');
};

LibsWidget.prototype.initLangDefaultLibs = function () {
    var defaultLibs = options.defaultLibs[this.currentLangId];
    if (!defaultLibs) return;
    _.each(defaultLibs.split(':'), _.bind(function (libPair) {
        var pairSplits = libPair.split('.');
        if (pairSplits.length === 2) {
            var lib = pairSplits[0];
            var ver = pairSplits[1];
            this.markLibrary(lib, ver, true);
        }
    }, this));
};

LibsWidget.prototype.updateAvailableLibs = function (possibleLibs) {
    if (!this.availableLibs[this.currentLangId]) {
        this.availableLibs[this.currentLangId] = {};
    }

    if (!this.availableLibs[this.currentLangId][this.currentCompilerId]) {
        if (this.currentCompilerId === '_default_') {
            this.availableLibs[this.currentLangId][this.currentCompilerId] =
                $.extend(true, {}, options.libs[this.currentLangId]);
        } else {
            this.availableLibs[this.currentLangId][this.currentCompilerId] =
                $.extend(true, {}, possibleLibs);
        }
    }

    this.initLangDefaultLibs();
    this.updateLibsDropdown();
};

LibsWidget.prototype.setNewLangId = function (langId, compilerId, possibleLibs) {
    var libsInUse = this.listUsedLibs();

    this.currentLangId = langId;

    if (compilerId) {
        this.currentCompilerId = compilerId;
    } else {
        this.currentCompilerId = '_default_';
    }

    // Clear the dom Root so it gets rebuilt with the new language libraries
    this.domRoot = null;
    this.updateAvailableLibs(possibleLibs);

    _.forEach(libsInUse, _.bind(function (version, lib) {
        this.markLibrary(lib, version, true);
    }, this));
};

LibsWidget.prototype.lazyDropdownLoad = function () {
    var libsCount = _.keys(this.availableLibs[this.currentLangId][this.currentCompilerId]).length;
    if (libsCount === 0) {
        return this.noLibsPanel;
    }
    if (this.domRoot === null) {
        var MAX_COLUMNS = 3;
        var currentColumn = null;
        var currentColumnItemCount = 0;
        var libsKeys = _.keys(this.availableLibs[this.currentLangId][this.currentCompilerId]).sort();
        var itemsPerColumn = Math.ceil(libsKeys.length / MAX_COLUMNS);
        this.domRoot = $('<div></div>');
        var libsPanel = $('<div></div>')
            .addClass('card-columns');
        var getOrCreateNextColumn = function () {
            if (currentColumn === null || currentColumnItemCount >= itemsPerColumn) {
                currentColumn = $('<div></div>').addClass('card');
                libsPanel.append(currentColumn);
                currentColumnItemCount = 0;
            }
            return currentColumn;
        };
        var addLibCardToColumn = function (libCard) {
            var column = getOrCreateNextColumn();
            column.append(libCard);
            currentColumnItemCount++;
        };
        var libsInUse = this.listUsedLibs();

        _.each(libsKeys, _.bind(function (id) {
            var libEntry = this.availableLibs[this.currentLangId][this.currentCompilerId][id];
            var newLibCard = this.libsEntry.clone();
            var label = newLibCard.find('.input-group-prepend label')
                .text(libEntry.name)
                .prop('title', libEntry.description || '')
                .prop('for', id);
            var select = newLibCard.find('select')
                .prop('id', id)
                .append($('<option>', {
                    value: '-',
                    text: '-'
                }));
            _.each(libEntry.versions, _.bind(function (version, versionId) {
                select.append($('<option>', {
                    value: versionId,
                    text: version.version,
                    selected: libsInUse[id] && libsInUse[id] === versionId
                }));
            }, this));
            label.toggleClass('bg-success text-white', select.val() !== '-');
            select.on('change', _.bind(function () {
                var newVal = select.val();
                label.toggleClass('bg-success text-white', newVal !== '-');
                // Disable every version for this lib
                _.each(libEntry.versions, _.bind(function (version, verId) {
                    this.markLibrary(id, verId, false);
                }, this));
                if (newVal !== '-') {
                    this.markLibrary(id, newVal, true);
                }
                this.onChangeCallback();
            }, this));
            addLibCardToColumn(newLibCard);
        }, this));
        this.domRoot.append(libsPanel);
        return this.domRoot;
    }
    return this.domRoot;
};

LibsWidget.prototype.updateLibsDropdown = function () {
    this.dropdownButton.popover('dispose');
    this.dropdownButton.popover({
        container: 'body',
        content: _.bind(this.lazyDropdownLoad, this),
        html: true,
        placement: 'bottom',
        trigger: 'click',
        template: '<div class="popover libs-popover" role="tooltip"><div class="arrow"></div>' +
            '<h3 class="popover-header"></h3>' +
            '<div class="popover-body"></div></div>'
    });
};

LibsWidget.prototype.markLibrary = function (name, version, used) {
    if (this.availableLibs[this.currentLangId] &&
        this.availableLibs[this.currentLangId][this.currentCompilerId] &&
        this.availableLibs[this.currentLangId][this.currentCompilerId][name] &&
        this.availableLibs[this.currentLangId][this.currentCompilerId][name].versions[version]) {

        this.availableLibs[this.currentLangId][this.currentCompilerId][name].versions[version].used = used;
    }
};

LibsWidget.prototype.get = function () {
    return _.map(this.listUsedLibs(), function (item, libId) {
        return {name: libId, ver: item};
    });
};

LibsWidget.prototype.listUsedLibs = function () {
    var libs = {};
    _.each(this.availableLibs[this.currentLangId][this.currentCompilerId], function (library, libId) {
        _.each(library.versions, function (version, ver) {
            if (library.versions[ver].used) {
                // We trust the invariant of only 1 used version at any given time per lib
                libs[libId] = ver;
            }
        });
    });
    return libs;
};

LibsWidget.prototype.getLibsInUse = function () {
    var libs = [];
    _.each(this.availableLibs[this.currentLangId][this.currentCompilerId], function (library, libId) {
        _.each(library.versions, function (version, ver) {
            if (library.versions[ver].used) {
                var libVer = Object.assign({libId: libId, versionId: ver}, library.versions[ver]);
                libs.push(libVer);
            }
        });
    });
    return libs;
};

module.exports = {
    Widget: LibsWidget
};
