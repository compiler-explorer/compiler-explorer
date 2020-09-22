// Copyright (c) 2020, Compiler Explorer Authors
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

function LibsWidgetExt(langId, compiler, dropdownButton, state, onChangeCallback) {
    this.dropdownButton = dropdownButton;
    var possibleLibs = false;
    if (compiler) {
        this.currentCompilerId = compiler.id;
        possibleLibs = compiler.libs;
    } else {
        this.currentCompilerId = '_default_';
    }
    this.currentLangId = langId;
    this.domRoot = $('#library-selection');
    this.onChangeCallback = onChangeCallback;
    this.availableLibs = {};
    this.updateAvailableLibs(possibleLibs);
    _.each(state.libs, _.bind(function (lib) {
        this.markLibrary(lib.name, lib.ver, true);
    }, this));
    this.showSelectedLibs();

    this.domRoot.find('.lib-search-input').on('keypress', _.bind(function (e) {
        if(e.which === 13) {
            this.startSearching();
        }
    }, this));

    this.domRoot.find('.lib-search-button').on('click', _.bind(function () {
        this.startSearching();
    }, this));
}

LibsWidgetExt.prototype.getAndEmptySearchResults = function () {
    var searchResults = this.domRoot.find('.lib-results-items');
    searchResults.html('');
    return searchResults;
};

LibsWidgetExt.prototype.newSelectedLibDiv = function (libId, versionId, lib, version) {
    var template = $('#lib-selected-tpl');

    var libDiv = $(template.children()[0].cloneNode(true));

    var detailsButton = libDiv.find('.lib-name-and-version');
    detailsButton.html(lib.name + ' ' + version.version);
    detailsButton.on('click', _.bind(function () {
        var searchResults = this.getAndEmptySearchResults();
        this.addSearchResult(lib, searchResults);
    }, this));

    var deleteButton = libDiv.find('.lib-remove');
    deleteButton.on('click', _.bind(function () {
        this.markLibrary(libId, versionId, false);
        libDiv.remove();
        this.onChangeCallback();
    }, this));

    return libDiv;
};

LibsWidgetExt.prototype.newSearchResult = function (lib) {
    var template = $('#lib-search-result-tpl');

    var result = $(template.children()[0].cloneNode(true));
    result.find('.lib-name').html(lib.name);
    result.find('.lib-description').html(lib.description ? lib.description : '&nbsp;');
    result.find('.lib-website-link').attr('href', lib.url ? lib.url : '#');

    result.find('.lib-fav-button').removeClass('fas').addClass('far');

    var versions = result.find('.lib-version-select');
    versions.html('');
    versions.append($('<option>-</option>'));
    _.each(lib.versions, function (version, versionId) {
        var option = $('<option>');
        if (version.used) {
            option.attr('selected','selected');
        }
        option.attr('value', versionId);
        option.html(version.version);
        versions.append(option);
    });
    return result;
};

LibsWidgetExt.prototype.addSearchResult = function (library, searchResults) {
    var card = this.newSearchResult(library);
    searchResults.append(card);
};

LibsWidgetExt.prototype.startSearching = function () {
    var searchtext = this.domRoot.find('.lib-search-input').val();
    var lcSearchtext = searchtext.toLowerCase();

    var searchResults = this.getAndEmptySearchResults();

    _.each(this.availableLibs[this.currentLangId][this.currentCompilerId], _.bind(function (library) {
        if (library.versions && library.versions.autodetect) return;

        if (library.name) {
            if (library.name.toLowerCase().includes(lcSearchtext)) {
                this.addSearchResult(library, searchResults);
                return;
            }
        }

        if (library.description) {
            if (library.description.toLowerCase().includes(lcSearchtext)) {
                this.addSearchResult(library, searchResults);
            }
        }
    }, this));
};

LibsWidgetExt.prototype.showSelectedLibs = function () {
    var items = this.domRoot.find('.libs-selected-items');
    items.html('');

    var selectedLibs = this.listUsedLibs();
    _.each(selectedLibs, _.bind(function (versionId, libId) {
        var lib = this.availableLibs[this.currentLangId][this.currentCompilerId][libId];
        var version = lib.versions[versionId];

        var libDiv = this.newSelectedLibDiv(libId, versionId, lib, version);
        items.append(libDiv);
    }, this));

    var searchResults = this.getAndEmptySearchResults();

    _.each(this.availableLibs[this.currentLangId][this.currentCompilerId], _.bind(function (library) {
        if (library.versions && library.versions.autodetect) return;

        var card = this.newSearchResult(library);
        searchResults.append(card);
    }, this));
};

LibsWidgetExt.prototype.initLangDefaultLibs = function () {
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

LibsWidgetExt.prototype.updateAvailableLibs = function (possibleLibs) {
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
};

LibsWidgetExt.prototype.setNewLangId = function (langId, compilerId, possibleLibs) {
    var libsInUse = this.listUsedLibs();

    this.currentLangId = langId;

    if (compilerId) {
        this.currentCompilerId = compilerId;
    } else {
        this.currentCompilerId = '_default_';
    }

    // Clear the dom Root so it gets rebuilt with the new language libraries
    this.updateAvailableLibs(possibleLibs);

    _.forEach(libsInUse, _.bind(function (version, lib) {
        this.markLibrary(lib, version, true);
    }, this));

    this.showSelectedLibs();
};

LibsWidgetExt.prototype.getVersionOrAlias = function (name, version) {
    if (this.availableLibs[this.currentLangId] &&
        this.availableLibs[this.currentLangId][this.currentCompilerId] &&
        this.availableLibs[this.currentLangId][this.currentCompilerId][name]) {
        if (this.availableLibs[this.currentLangId][this.currentCompilerId][name].versions[version]) {
            return version;
        } else {
            return _.findKey(
                this.availableLibs[this.currentLangId][this.currentCompilerId][name].versions,
                function (ver) {
                    return ver.alias && ver.alias.includes(version);
                });
        }
    }
};

LibsWidgetExt.prototype.markLibrary = function (name, version, used) {
    var actualVersion = this.getVersionOrAlias(name, version);

    if (this.availableLibs[this.currentLangId] &&
        this.availableLibs[this.currentLangId][this.currentCompilerId] &&
        this.availableLibs[this.currentLangId][this.currentCompilerId][name] &&
        this.availableLibs[this.currentLangId][this.currentCompilerId][name].versions[actualVersion]) {
        this.availableLibs[this.currentLangId][this.currentCompilerId][name].versions[actualVersion].used = used;
    }
};

LibsWidgetExt.prototype.get = function () {
    return _.map(this.listUsedLibs(), function (item, libId) {
        return {name: libId, ver: item};
    });
};

LibsWidgetExt.prototype.listUsedLibs = function () {
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

LibsWidgetExt.prototype.getLibsInUse = function () {
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
    Widget: LibsWidgetExt,
};
