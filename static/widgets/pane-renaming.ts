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

import _ from 'underscore';
import {Tab} from 'golden-layout';
import {EventEmitter} from 'events';
import {Alert} from '../alert';

export class PaneRenaming extends EventEmitter.EventEmitter {
    private pane: any;
    private alertSystem: any;
    private state: any;

    constructor(pane: any, state: any) {
        super();
        this.pane = pane;
        this.alertSystem = this.pane.alertSystem ?? new Alert();
        this.state = state;

        this.loadSavedPaneName();
        this.registerCallbacks();
    }

    public addState(state: any) {
        state.paneName = this.pane.paneName;
    }

    private loadSavedPaneName() {
        this.pane.paneName = this.state.paneName;
        this.pane.updateTitle();
    }

    private registerCallbacks(): void {
        this.pane.container.on('tab', this.addRenameButton.bind(this));
    }

    private addRenameButton(parent: Tab): void {
        // Add little pencil icon next to close tab
        const btn = $(`<div class="lm_modify_tab_title"></div>`);
        parent.element.prepend(btn);
        parent.titleElement.css('margin-right', '15px');

        // Open modal for entering new pane name
        btn.on('click', () => {
            const modalTextPlaceholder = this.pane.paneName || this.pane.getPaneName();
            this.alertSystem.enterSomething('Rename pane', 'Please enter new pane name', modalTextPlaceholder, {
                yes: (value: string) => {
                    // Update title and emit event to save it into the state
                    this.pane.paneName = value;
                    this.pane.updateTitle();
                    this.emit('renamePane');
                },
                yesClass: 'btn btn-primary',
                yesHtml: 'Rename',
                noClass: 'btn-outline-info',
                noHtml: 'Cancel',
            });
        });
    }
}
