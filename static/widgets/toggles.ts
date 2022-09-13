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

import {EventEmitter} from 'events';

import $ from 'jquery';

const settings = {
    on: {
        icon: 'far fa-check-square',
    },
    off: {
        icon: 'far fa-square',
    },
};

export class Toggles extends EventEmitter {
    private readonly buttons: JQuery;
    private readonly state: Record<string, boolean> = {};

    constructor(root: JQuery, state: Record<string, boolean> | undefined) {
        super();
        this.buttons = root.find('.button-checkbox');

        for (const element of this.buttons) {
            const widget = $(element);
            const button = widget.find('button');
            const checkbox = widget.find('input:checkbox');
            const bind = button.data('bind');

            // copy relevant parts of the state
            this.state[bind] = state && bind in state ? state[bind] : checkbox.is(':checked');

            // Event Handlers
            button.on('click', e => {
                checkbox.prop('checked', !checkbox.is(':checked'));
                checkbox.triggerHandler('change');
                e.stopPropagation();
            });
            checkbox.on('change', () => {
                this.updateButtonDisplay(button, checkbox);
            });

            this.updateButtonDisplay(button, checkbox, this.state[bind]);

            // Inject the icon if applicable
            if (button.find('.state-icon').length === 0) {
                button.prepend('<i class="state-icon ' + settings[button.data('state')].icon + '"></i> ');
            }
        }
    }

    private updateButtonDisplay(button: JQuery, checkbox: JQuery, forcedState?: boolean) {
        if (forcedState !== undefined) {
            checkbox.prop('checked', forcedState);
        }
        const isChecked = checkbox.is(':checked');

        // Set the button's state
        button.data('state', isChecked ? 'on' : 'off');

        // Set the button's icon
        button
            .find('.state-icon')
            .removeClass()
            .addClass(`state-icon ${settings[button.data('state')].icon}`);

        // Update the button's color
        button.toggleClass('active', isChecked);
        if (forcedState === undefined) {
            this.set(button.data('bind'), isChecked);
        }
    }

    get() {
        return {...this.state};
    }

    set(key: string, value: boolean) {
        const before = this.get();
        this.state[key] = value;
        this.emit('change', before, this.get());
    }

    enableToggle(key: string, enable: boolean) {
        for (const element of this.buttons) {
            const widget = $(element);
            const button = widget.find('button');
            const bind = button.data('bind');
            if (bind === key) {
                button.prop('disabled', !enable);
            }
        }
    }
}
