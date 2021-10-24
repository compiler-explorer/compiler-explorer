// Copyright (c) 2021, Compiler Explorer Authors
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

import { options } from './options'
import * as local from './local'

const _ = require('underscore');

export class Widget {
    private domRoot: JQuery<HTMLElement>;

    private currentLangId: string;
    private currentCompilerId: string;

    private dropdownButton: JQuery<HTMLElement>;
    private searchResults: any;

    private onChangeCallback: () => void;

    private availableLibs: any;


    constructor(langId, compiler, dropdownButton, state, onChangeCallback, possibleLibs) {
        this.dropdownButton = dropdownButton;
        if (compiler) {
            this.currentCompilerId = compiler.id;
        } else {
            this.currentCompilerId = '_default_';
        }
        this.currentLangId = langId;
        this.domRoot = $('#library-selection').clone(true);
        this.initButtons();
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

        const searchInput = this.domRoot.find('.lib-search-input');

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

    initButtons() {
        this.searchResults = this.domRoot.find('.lib-results-items');
    }

    fullRefresh() {
        this.showSelectedLibs();
        this.showSelectedLibsAsSearchResults();
        this.showFavorites();
    }

    updateButton () {
        const selectedLibs = this.get();
        let text = ' Libraries';
        if (selectedLibs.length > 0) {
            this.dropdownButton
                .addClass('btn-success')
                .removeClass('btn-light')
                .prop('title', 'Current libraries:\n' + _.map(selectedLibs, function (lib) {
                    return '- ' + lib.name;
                }).join('\n'));
            text = text + ' (' + selectedLibs.length + ')';
        } else {
            this.dropdownButton
                .removeClass('btn-success')
                .addClass('btn-light')
                .prop('title', 'Include libs');
        }

        this.dropdownButton.find('.dp-text').text(text);
    }

    getFavorites() {
        const storkey = 'favlibs';

        return JSON.parse(local.get(storkey, '{}'));
    }

    setFavorites(faves) {
        const storkey = 'favlibs';

        local.set(storkey, JSON.stringify(faves));
    }

    isAFavorite(libId, versionId) {
        const faves = this.getFavorites();
        if (faves[libId]) {
            return faves[libId].includes(versionId);
        }

        return false;
    }

    addToFavorites(libId, versionId) {
        const faves = this.getFavorites();
        if (faves[libId]) {
            faves[libId].push(versionId);
        } else {
            faves[libId] = [];
            faves[libId].push(versionId);
        }

        this.setFavorites(faves);
    }

    removeFromFavorites(libId, versionId) {
        const faves = this.getFavorites();
        if (faves[libId]) {
            faves[libId] = _.filter(faves[libId], function (v) {
                return (v !== versionId);
            });
        }

        this.setFavorites(faves);
    }

    newFavoriteLibDiv = function (libId, versionId, lib, version) {
        const template = $('#lib-favorite-tpl');

        const libDiv = $(template.children()[0].cloneNode(true));

        const quickSelectButton = libDiv.find('.lib-name-and-version');
        quickSelectButton.html(lib.name + ' ' + version.version);
        quickSelectButton.on('click', _.bind(function () {
            this.selectLibAndVersion(libId, versionId);
            this.showSelectedLibs();
            this.onChangeCallback();
        }, this));

        return libDiv;
    }

    showFavorites() {
        const favoritesDiv = this.domRoot.find('.lib-favorites');
        favoritesDiv.html('');

        const faves = this.getFavorites();
        _.each(faves, _.bind(function (versionArr, libId) {
            _.each(versionArr, _.bind(function (versionId) {
                const lib = this.getLibInfoById(libId);
                if (lib) {
                    const version = lib.versions[versionId];
                    if (version) {
                        const div = this.newFavoriteLibDiv(libId, versionId, lib, version);
                        favoritesDiv.append(div);
                    }
                }
            }, this));
        }, this));
    }

    emptySearchResults() {
        this.searchResults.html('');
    }

    newSelectedLibDiv(libId, versionId, lib, version) {
        const template = $('#lib-selected-tpl');

        const libDiv = $(template.children()[0].cloneNode(true));

        const detailsButton = libDiv.find('.lib-name-and-version');
        detailsButton.html(lib.name + ' ' + version.version);
        detailsButton.on('click', _.bind(function () {
            this.emptySearchResults()
            this.addSearchResult(libId, lib);
        }, this));

        const deleteButton = libDiv.find('.lib-remove');
        deleteButton.on('click', _.bind(function () {
            this.markLibrary(libId, versionId, false);
            libDiv.remove();
            this.showSelectedLibs();
            this.onChangeCallback();
            // We need to refresh the library lists, or the selector will still show up with the old library version
            this.startSearching();
        }, this));

        return libDiv;
    }

    conjureUpExamples(result, lib) {
        const examples = result.find('.lib-examples');
        if (lib.examples && lib.examples.length > 0) {
            const examplesHeader = $('<b>Examples</b>');
            const examplesList = $('<ul />');
            _.each(lib.examples, function (exampleId) {
                const li = $('<li />');
                examplesList.append(li);
                const exampleLink = $('<a>Example</a>');
                exampleLink.attr('href', window.httpRoot + 'z/' + exampleId);
                exampleLink.attr('target', '_blank');
                exampleLink.attr('rel', 'noopener');
                li.append(exampleLink);
            });

            examples.append(examplesHeader);
            examples.append(examplesList);
        }
    }

    newSearchResult(libId, lib) {
        const template = $('#lib-search-result-tpl');

        const result = $($(template.children()[0].cloneNode(true)));
        result.find('.lib-name').html(lib.name);
        if (!lib.description) {
            result.find('.lib-description').hide();
        } else {
            result.find('.lib-description').html(lib.description);
        }
        result.find('.lib-website-link').attr('href', lib.url ? lib.url : '#');

        this.conjureUpExamples(result, lib);

        const faveButton = result.find('.lib-fav-button');
        const faveStar = faveButton.find('.lib-fav-btn-icon');
        faveButton.hide();

        const versions = result.find('.lib-version-select');
        versions.html('');
        const noVersionSelectedOption = $('<option value="">-</option>');
        versions.append(noVersionSelectedOption);
        let hasVisibleVersions = false;

        _.each(lib.versions, _.bind(function (version, versionId) {
            const option = $('<option>');
            if (version.used) {
                option.attr('selected','selected');

                if (this.isAFavorite(libId, versionId)) {
                    faveStar.removeClass('far').addClass('fas');
                }

                faveButton.show();
            }
            option.attr('value', versionId);
            option.html(version.version);
            if (version.used || !version.hidden) {
                hasVisibleVersions = true;
                versions.append(option);
            }
        }, this));

        if (!hasVisibleVersions) {
            noVersionSelectedOption.text('No available versions');
            versions.prop('disabled', true);
        }

        faveButton.on('click', _.bind(function () {
            const option = versions.find('option:selected');
            const verId = option.attr('value');
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
            const option = versions.find('option:selected');
            const verId = option.attr('value');

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
    }

    addSearchResult(libId, library) {
        this.searchResults.append(this.newSearchResult(libId, library));
    }

    startSearching() {
        const searchtext = this.domRoot.find('.lib-search-input').val().toString();
        const lcSearchtext = searchtext.toLowerCase();

        this.emptySearchResults();

        if (Object.keys(this.availableLibs[this.currentLangId][this.currentCompilerId]).length === 0) {
            const nolibsMessage = $($('#libs-dropdown').children()[0].cloneNode(true));
            this.searchResults.append(nolibsMessage);
            return;
        }

        _.each(this.availableLibs[this.currentLangId][this.currentCompilerId], _.bind(function (library, libId) {
            if (library.versions && library.versions.autodetect) return;

            if (library.name) {
                if (library.name.toLowerCase().includes(lcSearchtext)) {
                    this.addSearchResult(libId, library);
                    return;
                }
            }

            if (library.description) {
                if (library.description.toLowerCase().includes(lcSearchtext)) {
                    this.addSearchResult(libId, library);
                }
            }
        }, this));
    }

    showSelectedLibs() {
        const items = this.domRoot.find('.libs-selected-items');
        items.html('');

        const selectedLibs = this.listUsedLibs();
        _.each(selectedLibs, _.bind(function (versionId, libId) {
            const lib = this.availableLibs[this.currentLangId][this.currentCompilerId][libId];
            const version = lib.versions[versionId];

            const libDiv = this.newSelectedLibDiv(libId, versionId, lib, version);
            items.append(libDiv);
        }, this));
    }

    showSelectedLibsAsSearchResults() {
        this.emptySearchResults();

        if (Object.keys(this.availableLibs[this.currentLangId][this.currentCompilerId]).length === 0) {
            const nolibsMessage = $($('#libs-dropdown').children()[0].cloneNode(true));
            this.searchResults.append(nolibsMessage);
            return;
        }

        _.each(this.availableLibs[this.currentLangId][this.currentCompilerId], _.bind(function (library, libId) {
            if (library.versions && library.versions.autodetect) return;

            const card = this.newSearchResult(libId, library);
            this.searchResults.append(card);
        }, this));
    }

    initLangDefaultLibs() {
        const defaultLibs = options.defaultLibs[this.currentLangId];
        if (!defaultLibs) return;
        _.each(defaultLibs.split(':'), _.bind(function (libPair) {
            const pairSplits = libPair.split('.');
            if (pairSplits.length === 2) {
                const lib = pairSplits[0];
                const ver = pairSplits[1];
                this.markLibrary(lib, ver, true);
            }
        }, this));
    }

    updateAvailableLibs(possibleLibs) {
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
    }

    setNewLangId(langId, compilerId, possibleLibs) {
        const libsInUse = this.listUsedLibs();

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
    }

    getVersionOrAlias(name, version) {
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
    }

    getLibInfoById(libId) {
        if (this.availableLibs[this.currentLangId] &&
            this.availableLibs[this.currentLangId][this.currentCompilerId] &&
            this.availableLibs[this.currentLangId][this.currentCompilerId][libId]) {
            return this.availableLibs[this.currentLangId][this.currentCompilerId][libId];
        }

        return false;
    };

    markLibrary(name, version, used) {
        const actualVersion = this.getVersionOrAlias(name, version);

        if (this.availableLibs[this.currentLangId] &&
            this.availableLibs[this.currentLangId][this.currentCompilerId] &&
            this.availableLibs[this.currentLangId][this.currentCompilerId][name] &&
            this.availableLibs[this.currentLangId][this.currentCompilerId][name].versions[actualVersion]) {
            this.availableLibs[this.currentLangId][this.currentCompilerId][name].versions[actualVersion].used = used;
        }
    };

    selectLibAndVersion(libId, versionId) {
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

    get() {
        return _.map(this.listUsedLibs(), function (item, libId) {
            return {name: libId, ver: item};
        });
    };

    listUsedLibs() {
        const libs = {};
        _.each(this.availableLibs[this.currentLangId][this.currentCompilerId], function (library, libId) {
            _.each(library.versions, function (version, ver) {
                if (library.versions[ver].used) {
                    // We trust the inconstiant of only 1 used version at any given time per lib
                    libs[libId] = ver;
                }
            });
        });
        return libs;
    };

    getLibsInUse() {
        const libs = [];
        _.each(this.availableLibs[this.currentLangId][this.currentCompilerId], function (library, libId) {
            _.each(library.versions, function (version, ver) {
                if (library.versions[ver].used) {
                    const libVer = Object.assign({libId: libId, versionId: ver}, library.versions[ver]);
                    libs.push(libVer);
                }
            });
        });
        return libs;
    };
}
