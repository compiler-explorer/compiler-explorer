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

import * as sifter from '@orchidjs/sifter';

import {CompilerInfo} from '../../types/compiler.interfaces';
import {intersection, remove, unique} from '../../lib/common-utils';
import {unwrap, unwrapString} from '../assert';
import {CompilerPicker} from './compiler-picker';
import {CompilerService} from '../compiler-service';
import {highlight} from '../highlight';

export class CompilerPickerPopup {
    modal: JQuery<HTMLElement>;
    modalSearch: JQuery<HTMLElement>;
    modalArchitectures: JQuery<HTMLElement>;
    modalCompilerTypes: JQuery<HTMLElement>;
    modalCompilers: JQuery<HTMLElement>;

    groups: {value: string; label: string}[];
    options: (CompilerInfo & {$groups: string[]})[];
    langId: string;

    isaFilters: string[];
    categoryFilters: string[];

    sifter: sifter.Sifter;
    searchResults: ReturnType<sifter.Sifter['search']> | undefined;

    constructor(private readonly compilerPicker: CompilerPicker) {
        this.modal = $('#compiler-picker-modal').clone(true);
        this.modalSearch = this.modal.find('.compiler-search');
        this.modalArchitectures = this.modal.find('.architectures');
        this.modalCompilerTypes = this.modal.find('.compiler-types');
        this.modalCompilers = this.modal.find('.compilers');
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
        this.modalArchitectures.empty();
        this.modalArchitectures.append(
            ...unique(instruction_sets)
                .sort()
                .map(isa => `<span class="architecture" data-value=${isa}>${isa}</span>`),
        );
        // get available compiler types
        const compilerTypes = compilers.map(compiler => compiler.compilerCategories ?? ['other']).flat();
        this.modalCompilerTypes.empty();
        this.modalCompilerTypes.append(
            ...unique(compilerTypes)
                .sort()
                .map(type => `<span class="compiler-type" data-value=${type}>${type}</span>`),
        );

        // search box
        this.modalSearch.on('input', () => {
            const query = unwrapString(this.modalSearch.val()).trim();
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
        $(this.modalArchitectures)
            .find('.architecture')
            .on('click', e => {
                e.preventDefault();
                const elem = $(e.currentTarget);
                elem.toggleClass('active');
                const isa = unwrap(elem.attr('data-value'));
                if (this.isaFilters.includes(isa)) {
                    this.isaFilters = this.isaFilters.filter(v => v !== isa);
                } else {
                    this.isaFilters.push(isa);
                }
                this.fillCompilers();
            });

        // category filters
        $(this.modalCompilerTypes)
            .find('.compiler-type')
            .on('click', e => {
                e.preventDefault();
                const elem = $(e.currentTarget);
                elem.toggleClass('active');
                const category = unwrap(elem.attr('data-value'));
                if (this.categoryFilters.includes(category)) {
                    this.categoryFilters = this.categoryFilters.filter(v => v !== category);
                } else {
                    this.categoryFilters.push(category);
                }
                this.fillCompilers();
            });
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
        // figure out if there are any empty groups, these will be ignored
        const groupCounts: Partial<Record<string, number>> = {};
        for (const compiler of filteredCompilers) {
            for (const group of compiler.$groups) {
                groupCounts[group] = (groupCounts[group] ?? 0) + 1;
            }
        }
        // add the compiler entries / group headers themselves
        this.modalCompilers.empty();
        const groupMap: Record<string, JQuery> = {};
        for (const group of this.groups) {
            if ((groupCounts[group.value] ?? 0) > 0) {
                const group_elem = $(
                    `
                    <div class="group-wrapper">
                        <div class="group">
                            <div class="label">${group.label}</div>
                        </div>
                    </div>
                    `,
                );
                group_elem.appendTo(this.modalCompilers);
                groupMap[group.value] = group_elem.find('.group');
            }
        }
        const searchRegexes = this.searchResults
            ? remove(
                  this.searchResults.tokens.map(token => token.regex),
                  null,
              )
            : undefined;
        for (const compiler of filteredCompilers) {
            const isFavorited = compiler.$groups.includes(CompilerPicker.favoriteGroupName);
            const extraClasses = isFavorited ? 'fas fa-star fav' : 'far fa-star';
            for (const group of compiler.$groups) {
                const compiler_elem = $(
                    `
                    <div class="compiler d-flex" data-value="${compiler.id}">
                        <div>${searchRegexes ? highlight(compiler.name, searchRegexes) : compiler.name}</div>
                        <div title="Click to mark or unmark as a favorite" class="ml-auto toggle-fav">
                            <i class="${extraClasses}"></i>
                        </div>
                    </div>
                    `,
                );
                compiler_elem.appendTo(groupMap[group]);
            }
        }
        // group header click events
        this.modalCompilers.find('.group').append('<div class="folded">&#8943;</div>');
        this.modalCompilers.find('.group > .label').on('click', e => {
            $(e.currentTarget).closest('.group').toggleClass('collapsed');
        });
        // favorite stars
        this.modalCompilers.find('.compiler .toggle-fav').on('click', e => {
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
        this.modalSearch.val('');
        this.modalSearch.trigger('input');
        this.modal.find('.architectures .active, .compiler-types .active').toggleClass('active');
        this.fillCompilers();
        this.modal.modal({});
        //console.log(this.modalSearch[0]);
        //window.setTimeout(() => {
        //    console.log("go");
        //    this.modalSearch[0].focus();
        //}, 200);
    }
}
