// Copyright (c) 2022, Compiler Explorer Authors
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
import TomSelect from 'tom-select';

import {ga} from '../analytics.js';
import * as local from '../local.js';
import {EventHub} from '../event-hub.js';
import {Hub} from '../hub.js';
import {CompilerService} from '../compiler-service.js';

type Favourites = {
    [compilerId: string]: boolean;
};

export class CompilerPicker {
    static readonly favoriteGroupName = '__favorites__';
    static readonly favoriteStoreKey = 'favCompilerIds';
    static nextSelectorId = 1;
    domRoot: JQuery;
    domNode: HTMLSelectElement;
    eventHub: EventHub;
    id: number;
    compilerService: CompilerService;
    onCompilerChange: (x: string) => any;
    tomSelect: TomSelect | null;
    lastLangId: string;
    lastCompilerId: string;
    compilerIsVisible: (any) => any; // TODO => bool probably
    constructor(
        domRoot: JQuery,
        hub: Hub,
        langId: string,
        compilerId: string,
        onCompilerChange: (x: string) => any,
        compilerIsVisible?: (x: any) => any,
    ) {
        this.eventHub = hub.createEventHub();
        this.id = CompilerPicker.nextSelectorId++;
        const compilerPicker = domRoot.find('.compiler-picker')[0];
        if (!(compilerPicker instanceof HTMLSelectElement)) {
            throw new Error('.compiler-picker is not an HTMLSelectElement');
        }
        this.domNode = compilerPicker;
        this.compilerService = hub.compilerService;
        this.onCompilerChange = onCompilerChange;
        this.eventHub.on('compilerFavoriteChange', this.onCompilerFavoriteChange, this);
        this.tomSelect = null;
        if (compilerIsVisible) {
            this.compilerIsVisible = compilerIsVisible;
        } else {
            this.compilerIsVisible = () => true;
        }

        this.initialize(langId, compilerId);
    }

    close() {
        // Quick note while I'm here: This function is never called. It probably should be. The conformance view
        // bypasses this function and does compilerEntry.picker.tomSelect.close(); manually. This function is the
        // only time this.tomSelect can be null, might be nice if we can get rid of that.
        this.eventHub.unsubscribe();
        if (this.tomSelect) this.tomSelect.destroy();
        this.tomSelect = null;
    }

    initialize(langId: string, compilerId: string) {
        this.lastLangId = langId;
        this.lastCompilerId = compilerId;

        this.tomSelect = new TomSelect(this.domNode, {
            sortField: CompilerService.getSelectizerOrder(),
            valueField: 'id',
            labelField: 'name',
            searchField: ['name'],
            placeholder: '🔍 Select a compiler...',
            optgroupField: '$groups',
            optgroups: this.getGroups(langId),
            lockOptgroupOrder: true,
            options: this.getOptions(langId, compilerId),
            items: compilerId ? [compilerId] : [],
            dropdownParent: 'body',
            closeAfterSelect: true,
            plugins: ['dropdown_input'],
            maxOptions: 1000,
            onChange: val => {
                // TODO(jeremy-rifkin) I don't think this can be undefined.
                // Typing here needs improvement later anyway.
                /* eslint-disable-next-line @typescript-eslint/no-unnecessary-condition */
                if (val) {
                    ga.proxy('send', {
                        hitType: 'event',
                        eventCategory: 'SelectCompiler',
                        eventAction: val,
                    });
                    const str = val as any as string;
                    this.onCompilerChange(str);
                    this.lastCompilerId = str;
                }
            },
            duplicates: true,
            render: <any>{
                option: (data, escape) => {
                    const isFavoriteGroup = data.$groups.indexOf(CompilerPicker.favoriteGroupName) !== -1;
                    const extraClasses = isFavoriteGroup ? 'fas fa-star fav' : 'far fa-star';
                    return (
                        '<div class="d-flex"><div>' +
                        escape(data.name) +
                        '</div>' +
                        '<div title="Click to mark or unmark as a favorite" class="ml-auto toggle-fav">' +
                        '<i class="' +
                        extraClasses +
                        '"></i>' +
                        '</div>' +
                        '</div>'
                    );
                },
            },
        });

        $(this.tomSelect.dropdown_content).on('click', '.toggle-fav', evt => {
            evt.preventDefault();
            evt.stopPropagation();

            if (this.tomSelect) {
                let optionElement = evt.currentTarget.closest('.option');
                const clickedGroup = optionElement.parentElement.dataset.group;
                const value = optionElement.dataset.value;
                const data = this.tomSelect.options[value];
                const isAddingNewFavorite = data.$groups.indexOf(CompilerPicker.favoriteGroupName) === -1;
                const elemTop = optionElement.offsetTop;

                if (isAddingNewFavorite) {
                    data.$groups.push(CompilerPicker.favoriteGroupName);
                    this.addToFavorites(data.id);
                } else {
                    data.$groups.splice(data.group.indexOf(CompilerPicker.favoriteGroupName), 1);
                    this.removeFromFavorites(data.id);
                }

                this.tomSelect.updateOption(value, data);
                this.tomSelect.refreshOptions(false);

                if (clickedGroup !== CompilerPicker.favoriteGroupName) {
                    // If the user clicked on an option that wasn't in the top "Favorite" group, then we just added
                    // or removed a bunch of controls way up in the list. Find the new element top and adjust the scroll
                    // so the element that was just clicked is back under the mouse.
                    optionElement = this.tomSelect.getOption(value);
                    const previousSmooth = this.tomSelect.dropdown_content.style.scrollBehavior;
                    this.tomSelect.dropdown_content.style.scrollBehavior = 'auto';
                    this.tomSelect.dropdown_content.scrollTop += optionElement.offsetTop - elemTop;
                    this.tomSelect.dropdown_content.style.scrollBehavior = previousSmooth;
                }
            }
        });
    }

    getOptions(langId: string, compilerId: string) {
        const favorites = this.getFavorites();
        return Object.values(this.compilerService.getCompilersForLang(langId) ?? {})
            .filter(e => (this.compilerIsVisible(e) && !e.hidden) || e.id === compilerId)
            .map(e => {
                e.$groups = [e.group];
                if (favorites[e.id]) e.$groups.unshift(CompilerPicker.favoriteGroupName);
                return e;
            });
    }

    getGroups(langId: string) {
        const optgroups = this.compilerService.getGroupsInUse(langId);
        optgroups.unshift({
            value: CompilerPicker.favoriteGroupName,
            label: 'Favorites',
        });
        return optgroups;
    }

    update(langId: string, compilerId: string) {
        this.tomSelect?.destroy();
        this.initialize(langId, compilerId);
    }

    onCompilerFavoriteChange(id: number) {
        if (this.id !== id) {
            // Rebuild the rest of compiler pickers so they can properly show the new fav status
            this.update(this.lastLangId, this.lastCompilerId);
        }
    }

    getFavorites(): Favourites {
        return JSON.parse(local.get(CompilerPicker.favoriteStoreKey, '{}'));
    }

    setFavorites(faves: Favourites) {
        local.set(CompilerPicker.favoriteStoreKey, JSON.stringify(faves));
    }

    isAFavorite(compilerId: string) {
        return compilerId in this.getFavorites();
    }

    addToFavorites(compilerId: string) {
        const faves = this.getFavorites();
        faves[compilerId] = true;
        this.setFavorites(faves);
        this.eventHub.emit('compilerFavoriteChange', this.id);
    }

    removeFromFavorites(compilerId: string) {
        const faves = this.getFavorites();
        delete faves[compilerId];
        this.setFavorites(faves);
        this.eventHub.emit('compilerFavoriteChange', this.id);
    }
}
