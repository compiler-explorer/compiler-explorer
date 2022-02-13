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
import _ from 'underscore';
import { Tab } from 'golden-layout';
import { LoadSave } from './load-save';

const Alert = require('alert').Alert;

// TODO: Does not quite work for: tool-input-view.js

export class PaneRenaming {
    private pane: any;
    private alertSystem: any;
    private key: string;

    constructor(pane: any, key?: string) {
        this.pane = pane;
        this.alertSystem = new Alert();
        this.key = key ?? this.pane.getPaneName();

        this.restoreSavedPaneState();
        this.registerCallbacks();
    }

    private registerCallbacks(): void {
        this.pane.container.on('tab', this.addRenameButton.bind(this));
        this.pane.container.on('destroy', this.cleanLocalStorage.bind(this));
    }

    private restoreSavedPaneState(): void {
        this.pane.paneName = LoadSave.getLocalPanes()[this.key];
        this.pane.updateTitle();
    }

    private cleanLocalStorage(parent: Tab): void {
        LoadSave.delLocalPane(this.key);
    }

    private addRenameButton(parent: Tab): void {
        // Add little pencil icon next to close tab
        const btn = $(`<div class="lm_modify_tab_title"></div>`);
        parent.element.prepend(btn);
        parent.titleElement.css('margin-right', '15px');

        // Open modal for entering new pane name
        btn.on('click', () => {
            this.alertSystem.enterSomething('Rename pane', 'Please enter new pane name', this.pane.getPaneName(), {
                yes: (value: string) => {
                    // Update title and save to local storage
                    this.pane.paneName = value;
                    this.pane.updateTitle();
                    LoadSave.setLocalPane(this.key, value);
                },
                no: () => {
                    this.alertSystem.resolve(false);
                },
                yesClass: 'btn btn-primary',
                yesHtml: 'Rename',
                noClass: 'btn-outline-info',
                noHtml: 'Cancel',
            });
        });
    }
}
