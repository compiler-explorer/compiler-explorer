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
import {pluck} from 'underscore';
import {sortedList, HistoryEntry, EditorSource} from '../history.js';
import {editor} from 'monaco-editor';
import ITextModel = editor.ITextModel;
import {unwrap} from '../assert.js';

type Entry = {dt: number; name: string; load: () => void};

export class HistoryWidget {
    private modal: JQuery | null = null;
    private srcDisplay: editor.ICodeEditor | null = null;
    private model: ITextModel = editor.createModel('', 'c++');
    private currentList: HistoryEntry[] = [];
    private onLoadCallback: (data: HistoryEntry) => void = () => {};

    private initializeIfNeeded() {
        if (this.modal === null) {
            this.modal = $('#history');

            const placeholder = this.modal.find('.monaco-placeholder');
            this.srcDisplay = editor.create(placeholder[0], {
                fontFamily: 'Consolas, "Liberation Mono", Courier, monospace',
                scrollBeyondLastLine: true,
                readOnly: true,
                minimap: {
                    enabled: false,
                },
            });

            this.srcDisplay.setModel(this.model);
        }
    }

    private static getLanguagesFromHistoryEntry(entry: HistoryEntry) {
        return pluck(entry.sources, 'lang');
    }

    private populateFromLocalStorage() {
        this.currentList = sortedList();
        this.populate(
            unwrap(this.modal).find('.historiccode'),
            this.currentList.map((data): Entry => {
                const dt = new Date(data.dt).toString();
                const languages = HistoryWidget.getLanguagesFromHistoryEntry(data).join(', ');
                return {
                    dt: data.dt,
                    name: `${dt.replace(/\s\(.*\)/, '')} (${languages})`,
                    load: () => {
                        this.onLoad(data);
                        this.modal?.modal('hide');
                    },
                };
            }),
        );
    }

    private getContent(src: EditorSource[]) {
        return src.map(val => `/****** ${val.lang} ******/\n${val.source}`).join('\n');
    }

    private showPreview() {
        const root = unwrap(this.modal).find('.historiccode');
        const elements = root.find('li:not(.template)');

        for (const elem of elements) {
            const li = $(elem);
            const dt = li.data('dt');

            const preview = li.find('.preview');

            if (preview.prop('checked')) {
                const item = this.currentList.find(item => item.dt === dt);
                const text: string = this.getContent(item!.sources);
                this.model.setValue(text);

                // Syntax-highlight by the language of the 1st source
                let firstLang = HistoryWidget.getLanguagesFromHistoryEntry(item!)[0];
                firstLang = firstLang === 'c++' ? 'cpp' : firstLang;
                editor.setModelLanguage(this.model, firstLang);
            }
        }
    }

    private populate(root: JQuery, list: Entry[]) {
        root.find('li:not(.template)').remove();
        const template = root.find('.template');

        for (const elem of list) {
            const li = template.clone().removeClass('template').appendTo(root);

            li.data('dt', elem.dt);

            const preview = li.find('.preview');

            preview.on('click', () => this.showPreview());
            li.find('a').text(elem.name).on('click', elem.load);
        }
    }

    private resizeLayout() {
        const content = unwrap(this.modal).find('div.src-content');
        this.srcDisplay?.layout({
            width: unwrap(content.width()),
            height: unwrap(content.height()) - 20,
        });
    }

    private onLoad(data: HistoryEntry) {
        this.onLoadCallback(data);
    }

    run(onLoad: (data: HistoryEntry) => void) {
        this.initializeIfNeeded();
        this.populateFromLocalStorage();
        this.onLoadCallback = onLoad;

        // It can't tell that we initialize modal on initializeIfNeeded, so it sticks to the possibility of it being null
        unwrap(this.modal).on('shown.bs.modal', () => this.resizeLayout());
        unwrap(this.modal).modal();
    }
}
