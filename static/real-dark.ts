import $ from 'jquery';

import {Hub} from './hub.js';
import {Settings} from './settings.js';

import * as local from './local.js';

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
    $('#true-dark .content .close').on('click', e => {
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
