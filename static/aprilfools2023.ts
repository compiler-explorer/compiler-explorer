import $ from 'jquery';

import {Settings, SiteSettings} from './settings.js';
import {Hub} from './hub.js';
import {Themes} from './themes.js';

import * as local from './local.js';

const localKey = 'aprilfools2023';

function toggleButton() {
    const theme = Settings.getStoredSettings().theme;
    $('#aprilfools2023 .content').toggle(theme !== 'pink' && local.get(localKey, '') !== 'hidden');
}

export function aprilfools2023(hub: Hub) {
    const eventHub = hub.createEventHub();
    eventHub.on('settingsChange', () => {
        toggleButton();
    });
    toggleButton();
    $('#aprilfools2023 .content').on('click', e => {
        if (e.target.classList.contains('content')) {
            // A little bit of a hack:
            $('#settings .theme').val('pink').trigger('change');
            //Settings.setTheme('pink');
        }
    });
    $('#aprilfools2023 .content .close').on('click', e => {
        local.set(localKey, 'hidden');
        toggleButton();
    });
}
