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

const Alert = require('alert').Alert;

export class PaneRenaming {
    public static registerCallback(pane: any): void {
        const alertSystem = new Alert();
        const addRenameButton = function (parentTab: Tab) {
            return PaneRenaming.addRenameButton.call(this, parentTab, pane, alertSystem);
        };
        pane.container.on('tab', addRenameButton);
    }

    public static addRenameButton(parent: Tab, pane: any, alertSystem: any): void {
        // Add little pencil icon next to close tab
        const btn = $(`<div class="lm_modify_tab_title"></div>`);
        parent.element.prepend(btn);
        parent.titleElement.css('margin-right', '15px');

        btn.on('click', () => {
            alertSystem.enterSomething('Rename pane', 'Please enter new pane name', pane.getPaneName(), {
                yes: (value: string) => {
                    pane.paneName = value;
                    pane.updateTitle();
                },
                no: () => {
                    alertSystem.resolve(false);
                },
                yesClass: 'btn btn-primary',
                yesHtml: 'Rename',
                noClass: 'btn-outline-info',
                noHtml: 'Cancel',
            });
        });
    }
}
