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
import _ from 'underscore';
import { Tab } from 'golden-layout';

export class PaneRenaming {
    public static registerCallback(pane: any): void {
        var addRenameButton = function (parentTab: any) {
            return PaneRenaming.addRenameButton.call(this, parentTab, pane);
        };
        pane.container.on('tab', addRenameButton);
    }

    public static addRenameButton(parent: Tab, pane: any): void {
        // Add little pencil icon next to close tab
        const btn = $(`<div class="lm_modify_tab_title"></div>`);
        parent.element.prepend(btn);
        parent.titleElement.css('margin-right', '15px');

        btn.on('click', () => {
            const modal = $('#renamepanemodal');
            modal.modal('show');

            const textField = $('#renamepaneinput');
            PaneRenaming.toggleEventListener(modal, 'shown.bs.modal', () => {
                textField.trigger('focus');
            });

            textField.on('keyup', (event: JQuery.Event) => {
                if (event.key === 'Enter') {
                    PaneRenaming.saveChanges(pane, modal);
                }
            });

            const saveChangesBtn = $('#renamepanesubmit');
            PaneRenaming.toggleEventListener(saveChangesBtn, 'click', () => {
                PaneRenaming.saveChanges(pane, modal);
            });
        });
    }

    private static saveChanges(pane: any, modal: JQuery): void {
        pane.paneName = PaneRenaming.getInputData();
        pane.updateTitle();

        modal.modal('hide');
    }

    private static getInputData(): string {
        return $('#renamepaneinput').val().toString();
    }

    private static toggleEventListener(element: JQuery, eventName: string, callback: any): void {
        element.on(eventName, (event) => {
            callback(event);
            // Unsuscribe the event listener so we don´t have more than one.
            element.off(eventName);
        });
    }
}
