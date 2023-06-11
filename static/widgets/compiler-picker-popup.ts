// Copyright (c) 2023, Compiler Explorer Authors
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
import _ from 'underscore';

import * as sifter from '@orchidjs/sifter';

import {CompilerInfo} from '../../types/compiler.interfaces';
import {intersection, remove, unique} from '../../lib/common-utils';
import {unwrap, unwrapString} from '../assert';
import {CompilerPicker} from './compiler-picker';
import {CompilerService} from '../compiler-service';
import {highlight} from '../highlight';

export class CompilerPickerPopup {
    modal: JQuery<HTMLElement>;
    searchBar: JQuery<HTMLElement>;
    architectures: JQuery<HTMLElement>;
    compilerTypes: JQuery<HTMLElement>;
    compilersContainer: JQuery<HTMLElement>;
    resultsContainer: JQuery<HTMLElement>;
    favoritesContainer: JQuery<HTMLElement>;

    groups: {value: string; label: string}[];
    options: (CompilerInfo & {$groups: string[]})[];
    langId: string;

    isaFilters: string[];
    categoryFilters: string[];

    sifter: sifter.Sifter;
    searchResults: ReturnType<sifter.Sifter['search']> | undefined;

    constructor(private readonly compilerPicker: CompilerPicker) {
        this.modal = $('#compiler-picker-modal').clone(true);
        this.searchBar = this.modal.find('.compiler-search');
        this.architectures = this.modal.find('.architectures');
        this.compilerTypes = this.modal.find('.compiler-types');
        this.compilersContainer = this.modal.find('.compilers-row');
        this.resultsContainer = this.modal.find('.compilers');
        this.favoritesContainer = this.modal.find('.favorites');

        this.modal.on('shown.bs.modal', () => {
            this.searchBar[0].focus();
        });
    }

    setLang(groups: {value: string; label: string}[], options: (CompilerInfo & {$groups: string[]})[], langId: string) {
        this.groups = groups;
        this.options = options;
        this.langId = langId;
        this.setupFilters();
        this.sifter = new sifter.Sifter(options, {
            diacritics: false,
        });
    }

    setupFilters() {
        // get available instruction sets
        const compilers = Object.values(this.compilerPicker.compilerService.getCompilersForLang(this.langId) ?? {});
        // If instructionSet is '', just label it unknown
        const instruction_sets = compilers.map(compiler => compiler.instructionSet || 'other');
        this.architectures.empty();
        this.architectures.append(
            ...unique(instruction_sets)
                .sort()
                .map(isa => `<span class="architecture" data-value=${_.escape(isa)}>${_.escape(isa)}</span>`),
        );
        // get available compiler types
        const compilerTypes = compilers.map(compiler => compiler.compilerCategories ?? ['other']).flat();
        this.compilerTypes.empty();
        this.compilerTypes.append(
            ...unique(compilerTypes)
                .sort()
                .map(type => `<span class="compiler-type" data-value=${_.escape(type)}>${_.escape(type)}</span>`),
        );

        // search box
        this.searchBar.on('input', () => {
            const query = unwrapString(this.searchBar.val()).trim();
            if (query === '') {
                this.searchResults = undefined;
            } else {
                this.searchResults = this.sifter.search(query, {
                    fields: ['name'],
                    conjunction: 'and',
                    sort: CompilerService.getSelectizerOrder(),
                });
            }
            this.fillCompilers();
        });

        // isa filters
        $(this.architectures)
            .find('.architecture')
            .on('click', e => this.onFilterClick(e, this.isaFilters));

        // category filters
        $(this.compilerTypes)
            .find('.compiler-type')
            .on('click', e => this.onFilterClick(e, this.categoryFilters));
    }

    onFilterClick(e: JQuery.ClickEvent, filtersArray: string[]) {
        e.preventDefault();
        const elem = $(e.currentTarget);
        elem.toggleClass('active');
        const filterValue = unwrap(elem.attr('data-value'));
        if (filtersArray.includes(filterValue)) {
            // This is pretty much the best way to filter an array in-place
            filtersArray.splice(0, filtersArray.length, ...filtersArray.filter(v => v !== filterValue));
        } else {
            filtersArray.push(filterValue);
        }
        this.fillCompilers();
    }

    fillCompilers() {
        const filteredIndices = this.searchResults
            ? new Set(this.searchResults.items.map(item => item.id as number))
            : undefined;
        const filteredCompilers = this.options.filter((compiler, i) => {
            if (this.isaFilters.length > 0) {
                if (!this.isaFilters.includes(compiler.instructionSet || 'other')) {
                    return false;
                }
            }
            if (this.categoryFilters.length > 0) {
                const categories = compiler.compilerCategories ?? ['other'];
                if (intersection(this.categoryFilters, categories).length === 0) {
                    return false;
                }
            }
            if (filteredIndices) {
                if (!filteredIndices.has(i)) {
                    return false;
                }
            }
            return true;
        });
        const searchRegexes = this.searchResults
            ? remove(
                  this.searchResults.tokens.map(token => token.regex),
                  null,
              )
            : undefined;
        // place compilers into groups
        const groupMap: Record<string, {elem: JQuery; order: number}[]> = {};
        for (const group of this.groups) {
            groupMap[group.value] = [];
        }
        for (const compiler of filteredCompilers) {
            const isFavorited = compiler.$groups.includes(CompilerPicker.favoriteGroupName);
            const extraClasses = isFavorited ? 'fas fa-star fav' : 'far fa-star';
            for (const group of compiler.$groups) {
                // TODO(jeremy-rifkin): At the moment none of our compiler names should contain html special characters.
                // This is just a good measure to take. If a compiler is ever added that does have special characters in
                // its name it could interfere with the highlighting (e.g. if your text search is for "<" that won't
                // highlight). I'm going to defer handling that to a future PR though.
                const name = _.escape(compiler.name);
                const compiler_elem = $(
                    `
                    <div class="compiler d-flex" data-value="${compiler.id}">
                        <div>${searchRegexes ? highlight(name, searchRegexes) : name}</div>
                        <div title="Click to mark or unmark as a favorite" class="ml-auto toggle-fav">
                            <i class="${extraClasses}"></i>
                        </div>
                    </div>
                    `,
                );
                if (compiler.id === this.compilerPicker.lastCompilerId) {
                    compiler_elem.addClass('selected');
                }
                groupMap[group].push({
                    elem: compiler_elem,
                    order: compiler.$order,
                });
            }
        }
        // sort compilers before placing them into groups
        for (const [_, compilers] of Object.entries(groupMap)) {
            compilers.sort((a, b) => a.order - b.order);
        }
        // add groups and compilers
        this.resultsContainer.empty();
        this.favoritesContainer.empty();
        for (const group of this.groups) {
            if (groupMap[group.value].length > 0 || group.value === CompilerPicker.favoriteGroupName) {
                const groupWrapper = $(
                    `
                    <div class="group-wrapper">
                        <div class="group">
                            <div class="label">${_.escape(group.label)}</div>
                        </div>
                    </div>
                    `,
                );
                if (group.value === CompilerPicker.favoriteGroupName) {
                    groupWrapper.appendTo(this.favoritesContainer);
                } else {
                    groupWrapper.appendTo(this.resultsContainer);
                }
                const groupElem = groupWrapper.find('.group');
                for (const compiler of groupMap[group.value]) {
                    compiler.elem.appendTo(groupElem);
                }
            }
        }
        // if there can only ever be one column, don't bother with room for 2
        this.resultsContainer.toggleClass(
            'one-col',
            this.groups.filter(group => group.value !== CompilerPicker.favoriteGroupName).length <= 1,
        );
        // group header click events
        this.compilersContainer.find('.group').append('<div class="folded">&#8943;</div>');
        this.compilersContainer.find('.group > .label').on('click', e => {
            $(e.currentTarget).closest('.group').toggleClass('collapsed');
        });
        // compiler click events
        this.compilersContainer.find('.compiler').on('click', e => {
            this.compilerPicker.selectCompiler(unwrap(e.currentTarget.getAttribute('data-value')));
            this.hide();
        });
        // favorite stars
        this.compilersContainer.find('.compiler .toggle-fav').on('click', e => {
            const compilerId = unwrap($(e.currentTarget).closest('.compiler').attr('data-value'));
            const data = filteredCompilers.filter(c => c.id === compilerId)[0];
            const isAddingNewFavorite = !data.$groups.includes(CompilerPicker.favoriteGroupName);
            if (isAddingNewFavorite) {
                data.$groups.push(CompilerPicker.favoriteGroupName);
                this.compilerPicker.addToFavorites(data.id);
            } else {
                data.$groups.splice(data.$groups.indexOf(CompilerPicker.favoriteGroupName), 1);
                this.compilerPicker.removeFromFavorites(data.id);
            }
            this.fillCompilers();
        });
    }

    show() {
        // reflow the compilers to get any new favorites from the compiler picker dropdown and reset filters and whatnot
        this.isaFilters = [];
        this.categoryFilters = [];
        this.searchBar.val('');
        this.searchBar.trigger('input');
        this.modal.find('.architectures .active, .compiler-types .active').toggleClass('active');
        this.fillCompilers();
        this.modal.modal({});
    }

    hide() {
        this.modal.modal('hide');
    }
}
