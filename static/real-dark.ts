// Copyright (c) 2024, Compiler Explorer Authors
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

import {Hub} from './hub.js';
import * as local from './local.js';
import {Settings} from './settings.js';

const localKey = 'aprilfools2024';

function toggleButton() {
    const theme = Settings.getStoredSettings().theme;
    const date = new Date();
    // month is 0-index and date is 1-indexed, because obviously that makes sense
    const is_april_1 = date.getMonth() === 3 && date.getDate() === 1;
    $('#true-dark .content').toggle(
        is_april_1 && theme !== 'real-dark' && local.localStorage.get(localKey, '') !== 'hidden',
    );
}

export function takeUsersOutOfRealDark() {
    // take user out of real-dark in case they got stuck previously
    if (Settings.getStoredSettings().theme === 'real-dark') {
        const settings = Settings.getStoredSettings();
        settings.theme = 'dark';
        Settings.setStoredSettings(settings);
    }
}

export function setupRealDark(hub: Hub) {
    const overlay = $('#real-dark');
    let overlay_on = false;
    const toggleOverlay = () => {
        const theme = Settings.getStoredSettings().theme;
        overlay_on = theme === 'real-dark';
        overlay.toggle(overlay_on);
    };

    const eventHub = hub.createEventHub();
    eventHub.on('settingsChange', () => {
        toggleButton();
        toggleOverlay();
    });
    toggleButton();
    toggleOverlay();
    $('#true-dark .content').on('click', e => {
        if (e.target.classList.contains('content')) {
            // A little bit of a hack:
            $('#settings .theme').val('real-dark').trigger('change');
        }
    });
    $('#true-dark .content .dark-close').on('click', _e => {
        local.localStorage.set(localKey, 'hidden');
        toggleButton();
        toggleOverlay();
    });

    window.addEventListener(
        'mousemove',
        e => {
            if (overlay_on) {
                overlay.css({top: e.pageY - window.innerHeight, left: e.pageX - window.innerWidth});
            }
        },
        false,
    );
}
