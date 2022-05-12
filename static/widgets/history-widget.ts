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

import {pluck} from 'underscore';
import {ga} from '../analytics';
import {sortedList, HistoryEntry, EditorSource} from '../history';
import {editor} from 'monaco-editor';

import IStandaloneDiffEditor = editor.IStandaloneDiffEditor;
import ITextModel = editor.ITextModel;

export class HistoryDiffState {
    public model: ITextModel;
    private result: EditorSource[];

    constructor(model: ITextModel) {
        this.model = model;
        this.result = [];
    }

    update(result: HistoryEntry) {
        this.result = result.sources;
        this.refresh();

        return true;
    }

    private refresh() {
        const output = this.result;
        const content = output.map(val => `/****** ${val.lang} ******/\n${val.source}`).join('\n');

        this.model.setValue(content);
    }
}

type Entry = {dt: number; name: string; load: () => void};

export class HistoryWidget {
    private modal: JQuery | null;
    private diffEditor: IStandaloneDiffEditor | null;
    private lhs: HistoryDiffState | null;
    private rhs: HistoryDiffState | null;
    private currentList: HistoryEntry[];
    private onLoadCallback: (data: HistoryEntry) => void;

    constructor() {
        this.modal = null;
        this.diffEditor = null;
        this.lhs = null;
        this.rhs = null;
        this.currentList = [];
        this.onLoadCallback = () => {};
    }

    private initializeIfNeeded() {
        if (this.modal === null) {
            this.modal = $('#history');

            const placeholder = this.modal.find('.monaco-placeholder');
            this.diffEditor = editor.createDiffEditor(placeholder[0], {
                fontFamily: 'Consolas, "Liberation Mono", Courier, monospace',
                scrollBeyondLastLine: true,
                readOnly: true,
                // language: 'c++',
                minimap: {
                    enabled: true,
                },
            });

            this.lhs = new HistoryDiffState(editor.createModel('', 'c++'));
            this.rhs = new HistoryDiffState(editor.createModel('', 'c++'));
            this.diffEditor.setModel({original: this.lhs.model, modified: this.rhs.model});

            this.modal.find('.inline-diff-checkbox').on('click', event => {
                const inline = $(event.target).prop('checked');
                this.diffEditor?.updateOptions({
                    renderSideBySide: !inline,
                });
                this.resizeLayout();
            });
        }
    }

    private static getLanguagesFromHistoryEntry(entry: HistoryEntry) {
        return pluck(entry.sources, 'lang');
    }

    private populateFromLocalStorage() {
        this.currentList = sortedList();
        this.populate(
            // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
            this.modal!.find('.historiccode'),
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
            })
        );
    }

    private hideRadiosAndSetDiff() {
        // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
        const root = this.modal!.find('.historiccode');
        const items = root.find('li:not(.template)');

        let foundbase = false;
        let foundcomp = false;

        for (const elem of items) {
            const li = $(elem);
            const dt = li.data('dt');

            const base = li.find('.base');
            const comp = li.find('.comp');

            let baseShouldBeVisible = true;
            let compShouldBeVisible = true;

            if (comp.prop('checked')) {
                foundcomp = true;
                baseShouldBeVisible = false;

                const itemRight = this.currentList.find(item => item.dt === dt);
                if (itemRight) {
                    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
                    this.rhs!.update(itemRight);
                }
            } else if (base.prop('checked')) {
                foundbase = true;

                const itemLeft = this.currentList.find(item => item.dt === dt);
                if (itemLeft) {
                    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
                    this.lhs!.update(itemLeft);
                }
            }

            if (foundbase && foundcomp) {
                compShouldBeVisible = false;
            } else if (!foundbase && !foundcomp) {
                baseShouldBeVisible = false;
            }

            if (compShouldBeVisible) {
                comp.css('visibility', '');
            } else {
                comp.css('visibility', 'hidden');
            }

            if (baseShouldBeVisible) {
                base.css('visibility', '');
            } else {
                base.css('visibility', 'hidden');
            }
        }
    }

    private populate(root: JQuery, list: Entry[]) {
        root.find('li:not(.template)').remove();
        const template = root.find('.template');

        let baseMarked = false;
        let compMarked = false;

        for (const elem of list) {
            const li = template.clone().removeClass('template').appendTo(root);

            li.data('dt', elem.dt);

            const base = li.find('.base');
            const comp = li.find('.comp');

            if (!compMarked) {
                comp.prop('checked', 'checked');
                compMarked = true;
            } else if (!baseMarked) {
                base.prop('checked', 'checked');
                baseMarked = true;
            }

            base.on('click', () => this.hideRadiosAndSetDiff());
            comp.on('click', () => this.hideRadiosAndSetDiff());

            li.find('a').text(elem.name).on('click', elem.load);
        }

        this.hideRadiosAndSetDiff();
    }

    private resizeLayout() {
        // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
        const tabcontent = this.modal!.find('div.tab-content');
        this.diffEditor?.layout({
            width: tabcontent.width() as number,
            height: (tabcontent.height() as number) - 20,
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
        // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
        this.modal!.on('shown.bs.modal', () => this.resizeLayout());
        // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
        this.modal!.modal();

        ga.proxy('send', {
            hitType: 'event',
            eventCategory: 'OpenModalPane',
            eventAction: 'History',
        });
    }
}
