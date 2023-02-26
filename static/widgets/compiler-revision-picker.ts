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
import _ from 'underscore';

import {searchCompilerRepository} from '../api/api';
import {ga} from '../analytics';

export class CompilerRevisionPicker {
    static nextSelectorId = 1;
    domRoot: JQuery;
    domNode: HTMLElement;
    id: number;
    onCompilerRevisionChange: (x: string) => any;
    tomSelect: TomSelect | null;
    lastCompilerId: string;
    lastQuery: string;
    constructor(domRoot: JQuery, onCompilerRevisionChange: (x: string) => any) {
        this.domNode = domRoot.find('.compiler-revision-picker')[0];

        this.onCompilerRevisionChange = onCompilerRevisionChange;
        this.tomSelect = null;
        this.lastQuery = '';

        $(this.domNode).on('click', '.compiler-revision-prev', () => {
            this.onNavigate(1);
        });
        $(this.domNode).on('click', '.compiler-revision-next', async () => {
            this.onNavigate(-1);
        });
    }

    /**
     * Navigate n revisions forward/backwards.
     */
    async onNavigate(distance: number) {
        if (this.tomSelect === null) {
            return;
        }

        const item = this.tomSelect.items[0];
        if (typeof item === 'undefined') {
            return;
        }

        const option = this.tomSelect.options[item];
        const nextOffset = option.offset + distance;

        if (nextOffset < 0) {
            // We know this offset is invalid.
            return;
        }

        // TODO: Check whether we have this offset locally.
        const res = await searchCompilerRepository({
            compilerId: this.lastCompilerId,
            offset: option.offset + distance,
            // Request two revisions. We can detect whether we've exhaused the
            // list of revisions by checking whether we only get 1 in response.
            limit: 2,
            query: this.lastQuery,
        });
        const data = await res.json();
        if (data.results.length === 0) {
            // Offset is past the end of the list.
            return;
        }
        if (nextOffset === 0) {
            // TODO: Disable prev button
        }
        if (data.results.length === 1) {
            // TODO: Disable next button
        }

        const nextRevision = data.results[0];

        // Insert the option into the picker and set it as active.
        if (Object.prototype.hasOwnProperty.call(this.tomSelect.options, 'nextRevision')) {
            this.tomSelect.addOptions([
                {
                    revision: nextRevision,
                    offset: nextOffset,
                },
            ]);
        }
        this.tomSelect.removeItem(item);
        this.tomSelect.addItem(nextRevision);
    }

    initialize(compilerId: string) {
        this.lastCompilerId = compilerId;
        this.lastQuery = '';

        // eslint-disable-next-line @typescript-eslint/no-this-alias
        const self = this;
        this.tomSelect = new TomSelect($(this.domNode).find('select')[0], {
            valueField: 'revision',
            labelField: 'revision',
            searchField: ['revision'],
            placeholder: '🔍 Select revision...',
            optgroupField: '$groups',
            lockOptgroupOrder: true,
            items: compilerId ? [compilerId] : [],
            dropdownParent: 'body',
            closeAfterSelect: true,
            plugins: ['dropdown_input', 'virtual_scroll'],
            maxOptions: 1000,
            firstUrl: function (query) {
                return '0';
            },
            load: async function (query, callback) {
                // Stash the query so we can navigate the result list
                self.lastQuery = query;
                const offset = parseInt((this as any).getUrl(query));
                const res = await searchCompilerRepository({
                    compilerId: self.lastCompilerId,
                    query,
                    offset,
                });
                const data = await res.json();
                const items = data.results.map((revision, i) => ({
                    revision,
                    // Track the offset of the revision in the backend result set.
                    offset: offset + i,
                    // Group is the first dash-delimited substring, which is
                    // YYYYMMDD for manyclangs.
                    $groups: [revision.split('-')[0]],
                }));
                const groups = _.uniq(items.map(x => x.$groups[0])).map(x => ({
                    label: self.formatGroupLabel(x),
                    value: x,
                }));
                if (data.offset + data.results.length < data.total) {
                    (this as any).setNextUrl(query, (data.offset + data.results.length).toString());
                }
                callback(items, groups);
                if ((this as any).items.length === 0 && items.length > 0) {
                    (this as any).addItem(items[0].revision);
                }
            },
            shouldLoad: function (query) {
                return true;
            },
            onChange: val => {
                ga.proxy('send', {
                    hitType: 'event',
                    eventCategory: 'SelectCompilerRevision',
                    eventAction: val,
                });
                const str = val as any as string;
                this.onCompilerRevisionChange(str);
            },
            render: <any>{
                option: (data, escape) => {
                    return '<div>' + escape(data.revision) + '</div>';
                },
                item: (data, escape) => {
                    return '<div>' + escape(data.revision) + '</div>';
                },
                loading_more: function (data, escape) {
                    return `<div class="loading-more-results">Loading more results... </div>`;
                },
            },
        });

        this.tomSelect.preload();
    }

    /**
     * Formats YYYYMMDD strings as YYYY/MM/DD.
     */
    formatGroupLabel(label: string): string {
        if (label.length !== 8) {
            return label;
        }

        const dateLike = `${label.slice(0, 4)}/${label.slice(4, 6)}/${label.slice(6, 8)}`;
        if (!isNaN(new Date(dateLike).valueOf())) {
            return dateLike;
        }
        return dateLike;
    }

    update(compilerId: string | null) {
        this.tomSelect?.destroy();
        this.tomSelect = null;
        $(this.domNode).toggle(compilerId !== null);
        if (compilerId !== null) {
            this.initialize(compilerId);
        }
    }
}
