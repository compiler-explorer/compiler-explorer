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

import $ from 'jquery';
import {options} from '../options.js';
import {Library, LibraryVersion} from '../options.interfaces.js';
import {Lib, WidgetState} from './libs-widget.interfaces.js';
import {unwrapString} from '../assert.js';
import {localStorage} from '../local.js';
import {Alert} from './alert';
import {SentryCapture} from '../sentry.js';

const FAV_LIBS_STORE_KEY = 'favlibs';
const c_default_compiler_non_id = '_default_';

export type CompilerLibs = Record<string, Library>;
type LangLibs = Record<string, CompilerLibs>;
type AvailableLibs = Record<string, LangLibs>;
type LibInUse = {libId: string; versionId: string} & LibraryVersion;

type FavLibraries = Record<string, string[]>;

type PopupAlertFilter = (compiler: string, langId: string) => {title: string; content: string} | null;

type LibraryAnnotation = {
    commithash?: string;
    cxx11?: boolean;
    machine?: string;
    osabi?: string;
    error?: string;
};

type LibraryBuildInfo = {
    arch?: string;
    compilerId?: string;
    compilerType?: string;
    libcxx?: string;
    os?: string;
    error?: string;
};

type LibraryAnnotationDetail = {
    buildhash: string;
    annotation: LibraryAnnotation;
    buildinfo: LibraryBuildInfo;
};

class LibraryAnnotations {
    private all: Record<string, LibraryAnnotationDetail[]> = {};
    constructor() {}

    private async get(library: string, version: string): Promise<LibraryAnnotationDetail[]> {
        const libver = `${library}/${version}`;
        if (!Object.keys(this.all).includes(libver)) {
            const response = await fetch(`https://conan.compiler-explorer.com/annotations/${libver}`);
            this.all[libver] = (await response.json()) as LibraryAnnotationDetail[];
        }
        return this.all[libver];
    }

    async getForCompiler(library: string, version: string, compilerId: string): Promise<LibraryAnnotationDetail[]> {
        const result: LibraryAnnotationDetail[] = [];

        try {
            const details = await this.get(library, version);
            for (const detail of details) {
                if (detail.buildinfo.compilerId === compilerId) {
                    result.push(detail);
                } else if (detail.buildinfo.compilerId === 'cshared') {
                    result.push(detail);
                }
            }
        } catch (e) {
            SentryCapture(e, `getForCompiler(${library}, ${version}, ${compilerId})`);
        }

        return result;
    }
}

function getCompilerName(compilerId: string): string {
    if (compilerId === c_default_compiler_non_id) {
        return 'compiler';
    }

    const compilerName = '';
    for (const compiler of options.compilers) {
        if (compiler.id === compilerId) {
            return compiler.name;
        }
    }

    return compilerName;
}

function shortenMachineName(name: string): string {
    if (name === 'Advanced Micro Devices X86-64') {
        return 'amd64';
    } else if (name === 'Intel 80386') {
        return '386';
    } else if (name === '') {
        return 'amd64';
    }

    return name;
}

const lib_annotations = new LibraryAnnotations();

export class LibsWidget {
    private domRoot: JQuery;

    private currentLangId: string;
    private currentCompilerId: string;

    private dropdownButton: JQuery;
    private searchResults: JQuery;

    private readonly onChangeCallback: () => void;

    private readonly availableLibs: AvailableLibs;
    private readonly filters: PopupAlertFilter[] = [
        (compilerId, langId) => {
            if (langId === 'rust' && ['beta', 'nightly'].includes(compilerId)) {
                return {
                    title: 'Missing library support for rustc nightly/beta',
                    content:
                        'Compiler Explorer does not yet support libraries for the rustc nightly/beta compilers.' +
                        'For library support, please use the stable compiler. Please see tracking issue' +
                        '<a href="https://github.com/compiler-explorer/compiler-explorer/issues/3766">' +
                        'compiler-explorer/compiler-explorer#3766</a> for more information.',
                };
            }
            return null;
        },
    ];

    constructor(
        langId: string,
        compiler: any,
        dropdownButton: JQuery,
        state: WidgetState,
        onChangeCallback: () => void,
        possibleLibs: CompilerLibs,
    ) {
        this.dropdownButton = dropdownButton;
        if (compiler) {
            this.currentCompilerId = compiler.id;
        } else {
            this.currentCompilerId = c_default_compiler_non_id;
        }
        this.currentLangId = langId;
        this.domRoot = $('#library-selection').clone(true);
        this.initButtons();
        this.onChangeCallback = onChangeCallback;
        this.availableLibs = {};
        this.updateAvailableLibs(possibleLibs, true);
        this.loadState(state);

        this.fullRefresh();

        const searchInput = this.domRoot.find('.lib-search-input');

        if (window.compilerExplorerOptions.mobileViewer) {
            this.domRoot.addClass('mobile');
        }

        this.domRoot
            .on('shown.bs.modal', () => {
                searchInput.trigger('focus');

                for (const filter of this.filters) {
                    const filterResult = filter(this.currentCompilerId, this.currentLangId);
                    if (filterResult !== null) {
                        const alertSystem = new Alert();
                        alertSystem.notify(`${filterResult.title}: ${filterResult.content}`, {
                            group: 'libs',
                            alertClass: 'notification-error',
                        });
                        break;
                    }
                }
            })
            .on('hide.bs.modal', () => {
                this.hidePopups();
            });

        searchInput.on('input', this.startSearching.bind(this));

        this.domRoot.find('.lib-search-button').on('click', this.startSearching.bind(this));

        this.dropdownButton.on('click', () => {
            this.domRoot.modal({});
        });

        this.updateButton();
    }

    onChange() {
        this.updateButton();
        this.onChangeCallback();
    }

    loadState(state: WidgetState) {
        // If state exists, clear previously selected libraries.
        if (state.libs !== undefined) {
            const libsInUse = this.listUsedLibs();
            for (const libId in libsInUse) {
                this.markLibrary(libId, libsInUse[libId], false);
            }
        }

        for (const lib of state.libs ?? []) {
            if (lib.name && lib.ver) {
                this.markLibrary(lib.name, lib.ver, true);
            } else if (lib.id && lib.version) {
                this.markLibrary(lib.id, lib.version, true);
            }
        }
    }

    initButtons() {
        this.searchResults = this.domRoot.find('.lib-results-items');
    }

    fullRefresh() {
        this.showSelectedLibs();
        this.showSelectedLibsAsSearchResults();
        this.showFavorites();
    }

    updateButton() {
        const selectedLibs = this.get();
        let text = 'Libraries';
        if (selectedLibs.length > 0) {
            this.dropdownButton
                .addClass('btn-success')
                .removeClass('btn-light')
                .prop('title', 'Current libraries:\n' + selectedLibs.map(lib => '- ' + lib.name).join('\n'));
            text += ' (' + selectedLibs.length + ')';
        } else {
            this.dropdownButton.removeClass('btn-success').addClass('btn-light').prop('title', 'Include libs');
        }

        this.dropdownButton.find('.dp-text').text(text);
    }

    getFavorites(): FavLibraries {
        return JSON.parse(localStorage.get(FAV_LIBS_STORE_KEY, '{}'));
    }

    setFavorites(faves: FavLibraries) {
        localStorage.set(FAV_LIBS_STORE_KEY, JSON.stringify(faves));
    }

    isAFavorite(libId: string, versionId: string): boolean {
        const faves = this.getFavorites();
        if (libId in faves) {
            return faves[libId].includes(versionId);
        }

        return false;
    }

    addToFavorites(libId: string, versionId: string) {
        const faves = this.getFavorites();
        if (libId in faves) {
            faves[libId].push(versionId);
        } else {
            faves[libId] = [];
            faves[libId].push(versionId);
        }

        this.setFavorites(faves);
    }

    removeFromFavorites(libId: string, versionId: string) {
        const faves = this.getFavorites();
        if (libId in faves) {
            faves[libId] = faves[libId].filter(v => v !== versionId);
        }

        this.setFavorites(faves);
    }

    newFavoriteLibDiv(libId: string, versionId: string, lib: Library, version: LibraryVersion): JQuery<Node> {
        const template = $('#lib-favorite-tpl');

        const libDiv = $(template.children()[0].cloneNode(true));

        const quickSelectButton = libDiv.find('.lib-name-and-version');
        quickSelectButton.html(lib.name + ' ' + version.version);
        quickSelectButton.on('click', () => {
            this.selectLibAndVersion(libId, versionId);
            this.showSelectedLibs();
            this.onChange();
        });

        return libDiv;
    }

    showFavorites() {
        const favoritesDiv = this.domRoot.find('.lib-favorites');
        favoritesDiv.html('');

        const faves = this.getFavorites();
        for (const libId in faves) {
            const versionArr = faves[libId];
            for (const versionId of versionArr) {
                const lib = this.getLibInfoById(libId);
                if (lib) {
                    if (versionId in lib.versions) {
                        const version = lib.versions[versionId];
                        const div: any = this.newFavoriteLibDiv(libId, versionId, lib, version);
                        favoritesDiv.append(div);
                    }
                }
            }
        }
    }

    hidePopups() {
        this.searchResults.find('.lib-info-button').popover('hide');
    }

    clearSearchResults() {
        this.searchResults.find('.lib-info-button').popover('dispose');
        this.searchResults.html('');
    }

    newSelectedLibDiv(libId: string, versionId: string, lib: Library, version: LibraryVersion): JQuery<Node> {
        const template = $('#lib-selected-tpl');

        const libDiv = $(template.children()[0].cloneNode(true));

        const detailsButton = libDiv.find('.lib-name-and-version');
        detailsButton.html(lib.name + ' ' + version.version);
        detailsButton.on('click', () => {
            this.clearSearchResults();
            this.addSearchResult(libId, lib);
        });

        const deleteButton = libDiv.find('.lib-remove');
        deleteButton.on('click', () => {
            this.markLibrary(libId, versionId, false);
            libDiv.remove();
            this.showSelectedLibs();
            this.onChange();
            // We need to refresh the library lists, or the selector will still show up with the old library version
            this.startSearching();
        });

        return libDiv;
    }

    conjureUpExamples(result: JQuery<Node>, lib: Library) {
        const examples = result.find('.lib-examples');
        if (lib.examples && lib.examples.length > 0) {
            examples.append($('<b>Examples</b>'));
            const examplesList = $('<ul />');
            for (const exampleId of lib.examples) {
                const li = $('<li />');
                examplesList.append(li);
                const exampleLink = $('<a>Example</a>');
                exampleLink.attr('href', `${window.httpRoot}z/${exampleId}`);
                exampleLink.attr('target', '_blank');
                exampleLink.attr('rel', 'noopener');
                li.append(exampleLink);
            }
            examples.append(examplesList);
        }
    }

    async getBuildInfoAsHtml(libId: string, semver: string, url?: string): Promise<string> {
        const details = await lib_annotations.getForCompiler(libId, semver, this.currentCompilerId);

        let libInfoText = '';
        for (const info of details) {
            if (info.annotation.commithash) {
                const machineName = shortenMachineName(info.annotation.machine || '');
                const stdlib = info.buildinfo.libcxx;
                if (url && url.startsWith('https://github.com/')) {
                    // this is a bit of a hack because we don't store the git repo in our properties files
                    libInfoText +=
                        `<li>Binary for ${machineName} (${stdlib}) based on commit: ` +
                        `<a href="${url}/commit/${info.annotation.commithash}" target="_blank">` +
                        info.annotation.commithash +
                        `</a></li>`;
                } else {
                    libInfoText +=
                        `<li>Binary for ${machineName} (${stdlib}) based on commit: ` +
                        info.annotation.commithash +
                        `</li>`;
                }
            }
        }

        if (!libInfoText) {
            libInfoText = 'No binaries available';
        } else {
            libInfoText = '<ul>' + libInfoText + '</ul>';
        }

        return libInfoText;
    }

    async loadBuildInfoIntoPopup(popupId: string, libId: string, semver: string, url?: string) {
        const libInfoText = await this.getBuildInfoAsHtml(libId, semver, url);
        $('#' + popupId).html(libInfoText);
    }

    newSearchResult(libId: string, lib: Library): JQuery<Node> {
        const template = $('#lib-search-result-tpl');

        const result = $($(template.children()[0].cloneNode(true)));
        result.find('.lib-name').html(lib.name || libId);
        if (!lib.description) {
            result.find('.lib-description').hide();
        } else {
            result.find('.lib-description').html(lib.description);
        }
        result.find('.lib-website-link').attr('href', lib.url ?? '#');

        this.conjureUpExamples(result, lib);

        const faveButton = result.find('.lib-fav-button');
        const faveStar = faveButton.find('.lib-fav-btn-icon');
        const infoButton = result.find('.lib-info-button');
        faveButton.hide();
        infoButton.hide();

        const versions = result.find('.lib-version-select');
        versions.html('');
        const noVersionSelectedOption = $('<option value="">-</option>');
        versions.append(noVersionSelectedOption);
        let hasVisibleVersions = false;

        const versionsArr = Object.keys(lib.versions).map(id => {
            return {id: id, order: lib.versions[id]['$order']};
        });
        versionsArr.sort((a, b) => b.order - a.order);

        for (const libVersion of versionsArr) {
            const versionId = libVersion.id;
            const version = lib.versions[versionId];
            const option = $('<option>');
            if (version.used) {
                option.attr('selected', 'selected');

                if (this.isAFavorite(libId, versionId)) {
                    faveStar.removeClass('far').addClass('fas');
                }

                faveButton.show();
                infoButton.show();
            }
            option.attr('value', versionId);
            option.html(version.version || versionId);
            if (version.used || !version.hidden) {
                hasVisibleVersions = true;
                versions.append(option);
            }
        }

        if (!hasVisibleVersions) {
            noVersionSelectedOption.text('No available versions');
            versions.prop('disabled', true);
        }

        const popoverTemplate =
            '<div class="popover" role="tooltip">' +
            '<div class="arrow"></div>' +
            '<h3 class="popover-header"></h3><div class="popover-body"></div>' +
            '</div>';
        infoButton.popover({
            html: true,
            title: 'Build info for ' + getCompilerName(this.currentCompilerId),
            content: () => {
                const nowts = Math.round(+new Date() / 1000);
                const popupId = `build-info-content-${nowts}`;
                const option = versions.find('option:selected');
                const semver = option.html();
                if (semver !== '-') {
                    this.loadBuildInfoIntoPopup(popupId, libId, semver, lib.url);
                    return `<div id="${popupId}">Loading...</div>`;
                } else {
                    return `<div id="${popupId}">No version selected</div>`;
                }
            },
            template: popoverTemplate,
            customClass: 'library-info-popover',
        });

        faveButton.on('click', () => {
            const option = versions.find('option:selected');
            const verId = option.attr('value') as string;
            if (this.isAFavorite(libId, verId)) {
                this.removeFromFavorites(libId, verId);
                faveStar.removeClass('fas').addClass('far');
            } else {
                this.addToFavorites(libId, verId);
                faveStar.removeClass('far').addClass('fas');
            }
            this.showFavorites();
        });

        versions.on('change', () => {
            const option = versions.find('option:selected');
            const verId = option.attr('value') as string;

            this.selectLibAndVersion(libId, verId);
            this.showSelectedLibs();

            if (this.isAFavorite(libId, verId)) {
                faveStar.removeClass('far').addClass('fas');
            } else {
                faveStar.removeClass('fas').addClass('far');
            }

            // Is this the "No selection" option?
            if (verId.length > 0) {
                faveButton.show();
                infoButton.show();
            } else {
                faveButton.hide();
                infoButton.hide();
            }

            this.onChange();
        });

        return result;
    }

    addSearchResult(libId: string, library: Library) {
        // FIXME: Type mismatch.
        // The any here stops TS from complaining
        const result: any = this.newSearchResult(libId, library);
        this.searchResults.append(result);
    }

    static _libVersionMatchesQuery(library: Library, searchText: string): boolean {
        const text = searchText.toLowerCase();
        return library.name?.toLowerCase().includes(text) || library.description?.toLowerCase().includes(text) || false;
    }

    startSearching() {
        const searchText = unwrapString(this.domRoot.find('.lib-search-input').val());

        this.clearSearchResults();

        const currentAvailableLibs = this.availableLibs[this.currentLangId][this.currentCompilerId];
        if (Object.keys(currentAvailableLibs).length === 0) {
            const nolibsMessage: any = $($('#libs-dropdown').children()[0].cloneNode(true));
            this.searchResults.append(nolibsMessage);
            return;
        }

        for (const libId in currentAvailableLibs) {
            const library = currentAvailableLibs[libId];

            if ('autodetect' in library.versions) continue;

            if (LibsWidget._libVersionMatchesQuery(library, searchText)) {
                this.addSearchResult(libId, library);
            }
        }
    }

    showSelectedLibs() {
        const items = this.domRoot.find('.libs-selected-items');
        items.html('');

        const selectedLibs = this.listUsedLibs();
        for (const libId in selectedLibs) {
            const versionId = selectedLibs[libId];

            const lib = this.availableLibs[this.currentLangId][this.currentCompilerId][libId];
            const version = lib.versions[versionId];

            const libDiv: any = this.newSelectedLibDiv(libId, versionId, lib, version);
            items.append(libDiv);
        }
    }

    showSelectedLibsAsSearchResults() {
        this.clearSearchResults();

        const currentAvailableLibs = this.availableLibs[this.currentLangId][this.currentCompilerId];
        if (Object.keys(currentAvailableLibs).length === 0) {
            const nolibsMessage: any = $($('#libs-dropdown').children()[0].cloneNode(true));
            this.searchResults.append(nolibsMessage);
            return;
        }

        for (const libId in currentAvailableLibs) {
            const library = currentAvailableLibs[libId];

            if ('autodetect' in library.versions) continue;

            const card: any = this.newSearchResult(libId, library);
            this.searchResults.append(card);
        }
    }

    initLangDefaultLibs() {
        const defaultLibs = options.defaultLibs[this.currentLangId];
        if (!defaultLibs) return;
        for (const libPair of defaultLibs.split(':')) {
            const pairSplits = libPair.split('.');
            if (pairSplits.length === 2) {
                const lib = pairSplits[0];
                const ver = pairSplits[1];
                this.markLibrary(lib, ver, true);
            }
        }
    }

    updateAvailableLibs(possibleLibs: CompilerLibs, isLangChanged: boolean) {
        if (!(this.currentLangId in this.availableLibs)) {
            this.availableLibs[this.currentLangId] = {};
        }

        if (!(this.currentCompilerId in this.availableLibs[this.currentLangId])) {
            if (this.currentCompilerId === '_default_') {
                this.availableLibs[this.currentLangId][this.currentCompilerId] = $.extend(
                    true,
                    {},
                    options.libs[this.currentLangId],
                );
            } else {
                this.availableLibs[this.currentLangId][this.currentCompilerId] = $.extend(true, {}, possibleLibs);
            }
        }

        if (isLangChanged) {
            this.initLangDefaultLibs();
        }
    }

    setNewLangId(langId: string, compilerId: string, possibleLibs: CompilerLibs) {
        const libsInUse = this.listUsedLibs();

        const isLangChanged = this.currentLangId !== langId;

        this.currentLangId = langId;

        if (compilerId) {
            this.currentCompilerId = compilerId;
        } else {
            this.currentCompilerId = '_default_';
        }

        // Clear the dom Root so it gets rebuilt with the new language libraries
        this.updateAvailableLibs(possibleLibs, isLangChanged);

        for (const libId in libsInUse) {
            this.markLibrary(libId, libsInUse[libId], true);
        }

        this.fullRefresh();
        this.onChange();
    }

    getVersionOrAlias(name: string, versionId: string): string | null {
        const lib = this.getLibInfoById(name);
        if (!lib) return null;
        // If it's already a key, return it directly
        if (versionId in lib.versions) {
            return versionId;
        } else {
            // Else, look in each version and see if it has the id as an alias
            for (const verId in lib.versions) {
                const version = lib.versions[verId];
                if (version.alias.includes(versionId)) {
                    return verId;
                }
            }
            return null;
        }
    }

    getLibInfoById(libId: string): Library | undefined {
        if (
            this.currentLangId in this.availableLibs &&
            this.currentCompilerId in this.availableLibs[this.currentLangId] &&
            libId in this.availableLibs[this.currentLangId][this.currentCompilerId]
        ) {
            return this.availableLibs[this.currentLangId][this.currentCompilerId][libId];
        } else {
            return undefined;
        }
    }

    markLibrary(name: string, versionId: string, used: boolean) {
        const actualId = this.getVersionOrAlias(name, versionId);
        if (actualId != null) {
            const v = this.getLibInfoById(name)?.versions[actualId];
            if (v != null) {
                v.used = used;
            }
        }
    }

    selectLibAndVersion(libId: string, versionId: string) {
        const actualId = this.getVersionOrAlias(libId, versionId);
        const libInfo = this.getLibInfoById(libId);
        if (libInfo) {
            for (const v in libInfo.versions) {
                const version = libInfo.versions[v];
                version.used = v === actualId;
            }
        }
    }

    get(): Lib[] {
        const result: Lib[] = [];
        const usedLibs = this.listUsedLibs();
        for (const libId in usedLibs) {
            result.push({name: libId, ver: usedLibs[libId]});
        }
        return result;
    }

    listUsedLibs(): Record<string, string> {
        const libs: Record<string, string> = {};
        const currentAvailableLibs = this.availableLibs[this.currentLangId][this.currentCompilerId];
        for (const libId in currentAvailableLibs) {
            const library = currentAvailableLibs[libId];
            for (const verId in library.versions) {
                if (library.versions[verId].used) {
                    libs[libId] = verId;
                }
            }
        }
        return libs;
    }

    getLibsInUse(): LibInUse[] {
        const libs: LibInUse[] = [];
        const currentAvailableLibs = this.availableLibs[this.currentLangId][this.currentCompilerId];
        for (const libId in currentAvailableLibs) {
            const library = currentAvailableLibs[libId];
            for (const verId in library.versions) {
                if (library.versions[verId].used) {
                    libs.push({...library.versions[verId], libId: libId, versionId: verId});
                }
            }
        }
        return libs;
    }
}
