import $ from 'jquery';

import {Settings, SiteSettings} from './settings.js';
import {Hub} from './hub.js';
import {Themes} from './themes.js';

function toggleButton() {
    const theme = Settings.getStoredSettings().theme;
    if (theme === 'pink') {
        $('#aprilfools2023 .content').hide();
    } else {
        $('#aprilfools2023 .content').show();
    }
}

export function aprilfools2023(hub: Hub) {
    const eventHub = hub.createEventHub();
    eventHub.on('settingsChange', () => {
        toggleButton();
    });
    toggleButton();
    $('#aprilfools2023 .content').on('click', () => {
        // A little bit of a hack:
        $('#settings .theme').val('pink').trigger('change');
        //Settings.setTheme('pink');
    });
}
