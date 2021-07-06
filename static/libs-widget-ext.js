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

var options = require('options'),
    _ = require('underscore'),
    local = require('./local'),
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
    this.domRoot = $('#library-selection').clone(true);
    this.onChangeCallback = function () {
        this.updateButton();
        onChangeCallback();
    };
    this.availableLibs = {};
    this.updateAvailableLibs(possibleLibs);
    _.each(state.libs, _.bind(function (lib) {
        if (lib.name && lib.ver) {
            this.markLibrary(lib.name, lib.ver, true);
        } else {
            this.markLibrary(lib.id, lib.version, true);
        }
    }, this));

    this.fullRefresh();

    var searchInput = this.domRoot.find('.lib-search-input');

    if (window.compilerExplorerOptions.mobileViewer) {
        this.domRoot.addClass('mobile');
    }

    this.domRoot.on('shown.bs.modal', function () {
        searchInput.trigger('focus');
    });

    searchInput.on('input', _.bind(function () {
        this.startSearching();
    }, this));

    this.domRoot.find('.lib-search-button').on('click', _.bind(function () {
        this.startSearching();
    }, this));

    this.dropdownButton.on('click', _.bind(function () {
        this.domRoot.modal({});
    }, this));

    this.updateButton();
}

LibsWidgetExt.prototype.fullRefresh = function () {
    this.showSelectedLibs();
    this.showSelectedLibsAsSearchResults();
    this.showFavorites();
};

LibsWidgetExt.prototype.updateButton = function () {
    var selectedLibs = this.get();
    if (selectedLibs.length > 0) {
        this.dropdownButton.addClass('btn-success').removeClass('btn-light');
    } else {
        this.dropdownButton.removeClass('btn-success').addClass('btn-light');
    }
};

LibsWidgetExt.prototype.getFavorites = function () {
    var storkey = 'favlibs';

    return JSON.parse(local.get(storkey, '{}'));
};

LibsWidgetExt.prototype.setFavorites = function (faves) {
    var storkey = 'favlibs';

    local.set(storkey, JSON.stringify(faves));
};

LibsWidgetExt.prototype.isAFavorite = function (libId, versionId) {
    var faves = this.getFavorites();
    if (faves[libId]) {
        return faves[libId].includes(versionId);
    }

    return false;
};

LibsWidgetExt.prototype.addToFavorites = function (libId, versionId) {
    var faves = this.getFavorites();
    if (faves[libId]) {
        faves[libId].push(versionId);
    } else {
        faves[libId] = [];
        faves[libId].push(versionId);
    }

    this.setFavorites(faves);
};

LibsWidgetExt.prototype.removeFromFavorites = function (libId, versionId) {
    var faves = this.getFavorites();
    if (faves[libId]) {
        faves[libId] = _.filter(faves[libId], function (v) {
            return (v !== versionId);
        });
    }

    this.setFavorites(faves);
};

LibsWidgetExt.prototype.newFavoriteLibDiv = function (libId, versionId, lib, version) {
    var template = $('#lib-favorite-tpl');

    var libDiv = $(template.children()[0].cloneNode(true));

    var quickSelectButton = libDiv.find('.lib-name-and-version');
    quickSelectButton.html(lib.name + ' ' + version.version);
    quickSelectButton.on('click', _.bind(function () {
        this.selectLibAndVersion(libId, versionId);
        this.showSelectedLibs();
        this.onChangeCallback();
    }, this));

    return libDiv;
};

LibsWidgetExt.prototype.showFavorites = function () {
    var favoritesDiv = this.domRoot.find('.lib-favorites');
    favoritesDiv.html('');

    var faves = this.getFavorites();
    _.each(faves, _.bind(function (versionArr, libId) {
        _.each(versionArr, _.bind(function (versionId) {
            var lib = this.getLibInfoById(libId);
            if (lib) {
                var version = lib.versions[versionId];
                if (version) {
                    var div = this.newFavoriteLibDiv(libId, versionId, lib, version);
                    favoritesDiv.append(div);
                }
            }
        }, this));
    }, this));
};

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
        this.addSearchResult(libId, lib, searchResults);
    }, this));

    var deleteButton = libDiv.find('.lib-remove');
    deleteButton.on('click', _.bind(function () {
        this.markLibrary(libId, versionId, false);
        libDiv.remove();
        this.showSelectedLibs();
        this.onChangeCallback();
        // We need to refresh the library lists, or the selector will still show up with the old library version
        this.startSearching();
    }, this));

    return libDiv;
};

LibsWidgetExt.prototype.conjureUpExamples = function (result, lib) {
    var examples = result.find('.lib-examples');
    if (lib.examples && lib.examples.length > 0) {
        var examplesHeader = $('<b>Examples</b>');
        var examplesList = $('<ul />');
        _.each(lib.examples, function (exampleId) {
            var li = $('<li />');
            examplesList.append(li);
            var exampleLink = $('<a>Example</a>');
            exampleLink.attr('href', window.httpRoot + 'z/' + exampleId);
            exampleLink.attr('target', '_blank');
            exampleLink.attr('rel', 'noopener');
            li.append(exampleLink);
        });

        examples.append(examplesHeader);
        examples.append(examplesList);
    }
};

LibsWidgetExt.prototype.newSearchResult = function (libId, lib) {
    var template = $('#lib-search-result-tpl');

    var result = $(template.children()[0].cloneNode(true));
    result.find('.lib-name').html(lib.name);
    if (!lib.description) {
        result.find('.lib-description').hide();
    } else {
        result.find('.lib-description').html(lib.description);
    }
    result.find('.lib-website-link').attr('href', lib.url ? lib.url : '#');

    this.conjureUpExamples(result, lib);

    var faveButton = result.find('.lib-fav-button');
    var faveStar = faveButton.find('.lib-fav-btn-icon');
    faveButton.hide();

    var versions = result.find('.lib-version-select');
    versions.html('');
    versions.append($('<option value="">-</option>'));
    _.each(lib.versions, _.bind(function (version, versionId) {
        var option = $('<option>');
        if (version.used) {
            option.attr('selected','selected');

            if (this.isAFavorite(libId, versionId)) {
                faveStar.removeClass('far').addClass('fas');
            }

            faveButton.show();
        }
        option.attr('value', versionId);
        option.html(version.version);
        versions.append(option);
    }, this));

    faveButton.on('click', _.bind(function () {
        var option = versions.find('option:selected');
        var verId = option.attr('value');
        if (this.isAFavorite(libId, verId)) {
            this.removeFromFavorites(libId, verId);
            faveStar.removeClass('fas').addClass('far');
        } else {
            this.addToFavorites(libId, verId);
            faveStar.removeClass('far').addClass('fas');
        }
        this.showFavorites();
    }, this));

    versions.on('change', _.bind(function () {
        var option = versions.find('option:selected');
        var verId = option.attr('value');

        this.selectLibAndVersion(libId, verId);
        this.showSelectedLibs();

        if (this.isAFavorite(libId, verId)) {
            faveStar.removeClass('far').addClass('fas');
        } else {
            faveStar.removeClass('fas').addClass('far');
        }

        if (verId) {
            faveButton.show();
        } else {
            faveButton.hide();
        }

        this.onChangeCallback();
    }, this));

    return result;
};

LibsWidgetExt.prototype.addSearchResult = function (libId, library, searchResults) {
    var card = this.newSearchResult(libId, library);
    searchResults.append(card);
};

LibsWidgetExt.prototype.startSearching = function () {
    var searchtext = this.domRoot.find('.lib-search-input').val();
    var lcSearchtext = searchtext.toLowerCase();

    var searchResults = this.getAndEmptySearchResults();

    if (Object.keys(this.availableLibs[this.currentLangId][this.currentCompilerId]).length === 0) {
        var nolibsMessage = $($('#libs-dropdown').children()[0].cloneNode(true));
        searchResults.append(nolibsMessage);
        return;
    }

    var descriptionSearchResults = [];

    _.each(this.availableLibs[this.currentLangId][this.currentCompilerId], _.bind(function (library, libId) {
        if (library.versions && library.versions.autodetect) return;

        if (library.name) {
            if (library.name.toLowerCase().includes(lcSearchtext)) {
                this.addSearchResult(libId, library, searchResults);
                return;
            }
        }

        if (library.description) {
            if (library.description.toLowerCase().includes(lcSearchtext)) {

                descriptionSearchResults.push({
                    libId: libId,
                    library: library,
                    searchResults: searchResults,
                });
            }
        }
    }, this));

    _.each(descriptionSearchResults, _.bind(function (res) {
        this.addSearchResult(res.libId, res.library, res.searchResults);
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
};

LibsWidgetExt.prototype.showSelectedLibsAsSearchResults = function () {
    var searchResults = this.getAndEmptySearchResults();

    if (Object.keys(this.availableLibs[this.currentLangId][this.currentCompilerId]).length === 0) {
        var nolibsMessage = $($('#libs-dropdown').children()[0].cloneNode(true));
        searchResults.append(nolibsMessage);
        return;
    }

    _.each(this.availableLibs[this.currentLangId][this.currentCompilerId], _.bind(function (library, libId) {
        if (library.versions && library.versions.autodetect) return;

        var card = this.newSearchResult(libId, library);
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

    this.fullRefresh();
    this.onChangeCallback();
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

LibsWidgetExt.prototype.getLibInfoById = function (libId) {
    if (this.availableLibs[this.currentLangId] &&
        this.availableLibs[this.currentLangId][this.currentCompilerId] &&
        this.availableLibs[this.currentLangId][this.currentCompilerId][libId]) {
        return this.availableLibs[this.currentLangId][this.currentCompilerId][libId];
    }

    return false;
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

LibsWidgetExt.prototype.selectLibAndVersion = function (libId, versionId) {
    if (this.availableLibs[this.currentLangId] &&
        this.availableLibs[this.currentLangId][this.currentCompilerId] &&
        this.availableLibs[this.currentLangId][this.currentCompilerId][libId]) {

        _.each(
            this.availableLibs[this.currentLangId][this.currentCompilerId][libId].versions,
            function (curver, curverId) {
                curver.used = curverId === versionId;
            });
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
