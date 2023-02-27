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
import TomSelect from 'tom-select';

import {ga} from '../analytics';
import * as local from '../local';
import {EventHub} from '../event-hub';
import {Hub} from '../hub';
import {CompilerService} from '../compiler-service';
import {CompilerInfo} from '../../types/compiler.interfaces';
import {unique} from '../../lib/common-utils';
import {unwrap} from '../assert';
import {CompilerPicker} from './compiler-picker';

export class CompilerPickerPopup {
    modal: JQuery<HTMLElement>;
    modalArchitectures: JQuery<HTMLElement>;
    modalCompilerTypes: JQuery<HTMLElement>;
    modalCompilers: JQuery<HTMLElement>;
    constructor(private readonly compilerPicker: CompilerPicker) {
        this.modal = $('#compiler-picker-modal').clone(true);
        this.modalArchitectures = this.modal.find('.architectures');
        this.modalCompilerTypes = this.modal.find('.compiler-types');
        this.modalCompilers = this.modal.find('.compilers');
    }

    setLang(
        groups: {value: string; label: string}[],
        options: (CompilerInfo & {$groups: string[]})[],
        langId: string,
        compilerId: string,
    ) {
        // setup modal / button
        // text filter
        // instructionset filters
        const compilers = Object.values(this.compilerPicker.compilerService.getCompilersForLang(langId) ?? {});
        const instruction_sets = compilers.map(compiler => compiler.instructionSet);
        this.modalArchitectures.empty();
        this.modalArchitectures.append(
            ...unique(instruction_sets.map(isa => `<span class="architecture" data-value=${isa}>${isa}</span>`)).sort(),
        );
        //
        const compilerTypes = compilers.map(compiler => compiler.compilerCategory ?? 'other');
        this.modalCompilerTypes.empty();
        this.modalCompilerTypes.append(
            ...unique(
                compilerTypes.map(type => `<span class="compiler-type" data-value=${type}>${type}</span>`),
            ).sort(),
        );
        let isaFilters: string[] = [];
        let categoryFilters: string[] = [];
        const doCompilers = () => {
            const filteredCompilers = options.filter(compiler => {
                if (isaFilters.length > 0) {
                    if (!isaFilters.includes(compiler.instructionSet)) {
                        return false;
                    }
                }
                if (categoryFilters.length > 0) {
                    if (!categoryFilters.includes(compiler.compilerCategory ?? 'other')) {
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
            for (const group of groups) {
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
            for (const compiler of filteredCompilers) {
                const isFavorited = compiler.$groups.includes(CompilerPicker.favoriteGroupName);
                const extraClasses = isFavorited ? 'fas fa-star fav' : 'far fa-star';
                for (const group of compiler.$groups) {
                    const compiler_elem = $(
                        `
                        <div class="compiler d-flex" data-value="${compiler.id}">
                            <div>${compiler.name}</div>
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
                doCompilers();
            });
        };
        doCompilers();

        // isa click events
        $(this.modalArchitectures)
            .find('.architecture')
            .on('click', e => {
                e.preventDefault();
                const elem = $(e.currentTarget);
                elem.toggleClass('active');
                const isa = unwrap(elem.attr('data-value'));
                if (isaFilters.includes(isa)) {
                    isaFilters = isaFilters.filter(v => v !== isa);
                } else {
                    isaFilters.push(isa);
                }
                doCompilers();
            });

        // do category filters
        $(this.modalCompilerTypes)
            .find('.compiler-type')
            .on('click', e => {
                e.preventDefault();
                const elem = $(e.currentTarget);
                elem.toggleClass('active');
                const category = unwrap(elem.attr('data-value'));
                if (categoryFilters.includes(category)) {
                    categoryFilters = categoryFilters.filter(v => v !== category);
                } else {
                    categoryFilters.push(category);
                }
                doCompilers();
            });

        // TODO:
        // - text search
        // - filter isa
        // - filter category
        // - collapse group headers
        // - filter special forks?
        // - save collapsed groups?
    }

    show() {
        this.modal.modal({});
    }
}
