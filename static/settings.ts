// Copyright (c) 2016, Compiler Explorer Authors
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

import { SiteSettings } from './settings.interfaces';
import { options } from './options';
import * as colour from './colour';
import * as local from './local';

const themes = require('./themes').themes

class ISetting {
    constructor(public elem: JQuery, public name: string) {}

    getUi(): any {
        return this.elem.val();
    }

    putUi(value): void {
        this.elem.val(value);
    }
}

class Checkbox extends ISetting {
    override getUi(): any {
        return !!this.elem.prop('checked');
    }

    override putUi(value) {
        this.elem.prop('checked', !!value);
    }
}

class Select extends ISetting {
    constructor(elem, name, populate) {
        super(elem, name);

        elem.empty();
        for (let e of populate) {
            elem.append($(`<option value="${e.label}">${e.desc}</option>`))
        }
    }

    override putUi(value) {
        this.elem.val(value.toString());
    }
}

class Slider extends ISetting {
    constructor(elem, name, sliderSettings) {
        super(elem, name);

        elem
            .prop('max', sliderSettings.max || 100)
            .prop('min', sliderSettings.min || 1)
            .prop('step', sliderSettings.step || 1);
    }

    override getUi(): number {
        return parseInt(this.elem.val().toString())
    }
}

class Textbox extends ISetting {}

class Numeric extends ISetting {
    private readonly min: number;
    private readonly max: number;

    constructor(elem, name, params) {
        super(elem, name);

        this.min = params.min;
        this.max = params.max;

        elem.attr('min', this.min)
            .attr('max', this.max);
    }

    override getUi(): any {
        return this.clampValue(parseInt(this.elem.val().toString()));
    }

    override putUi(value) {
        this.elem.val(this.clampValue(value))
    }

    private clampValue(value) {
        return Math.min(Math.max(value, this.min), this.max);
    }
}

// Ignore max statements, there's no limit as to how many settings we need
// eslint-disable-next-line max-statements
export function setupSettings(root: JQuery, settings, onChange: (SiteSettings) => void, subLangId: string | null) {
    settings = settings || {};
    // Ensure the default language is not "null" but undefined. Temporary patch for a previous bug :(
    settings.defaultLanguage = settings.defaultLanguage === null ? undefined : settings.defaultLanguage;
    const settingsObjs: ISetting[] = [];

    var currentSettings = settings;

    function onUiChange() {
        var settings = {};
        for (let s of settingsObjs) {
            settings[s.name] = s.getUi();
        }
        currentSettings = settings;
        onChange(settings);
    }

    function onSettingsChange(settings) {
        for (let s of settingsObjs) {
            s.putUi(settings[s.name]);
        }
    }

    // Don't forget to edit the settings.interfaces.ts file if you add/modify
    // a setting!
    function add<Type extends ISetting>(setting: Type, defaultValue: any) {
        const key = setting.name;
        if (settings[key] === undefined)
            settings[key] = defaultValue;
        settingsObjs.push(setting);
        setting.elem.change(onUiChange);
    }

    add(new Checkbox(root.find('.colourise'), 'colouriseAsm'), true);
    add(new Checkbox(root.find('.autoCloseBrackets'), 'autoCloseBrackets'), true);
    var colourSchemeSelect = root.find('.colourScheme');
    add(new Select(colourSchemeSelect, 'colourScheme',
            colour.schemes.map(scheme => {
                return {label: scheme.name, desc: scheme.desc}
            })),
        colour.schemes[0].name,
    );
    // Handle older settings
    if (settings.delayAfterChange === 0) {
        settings.delayAfterChange = 750;
        settings.compileOnChange = false;
    }
    add(new Checkbox(root.find('.compileOnChange'), 'compileOnChange'), true);
    add(new Slider(root.find('.delay'), 'delayAfterChange', {
        max: 3000,
        step: 250,
        min: 250,
        formatter: x => (x / 1000.0).toFixed(2) + 's',
    }),
        750);
    add(new Checkbox(root.find('.enableCommunityAds'), 'enableCommunityAds'), true);
    add(new Checkbox(root.find('.hoverShowSource'), 'hoverShowSource'), true);
    add(new Checkbox(root.find('.hoverShowAsmDoc'), 'hoverShowAsmDoc'), true);

    var themeSelect = root.find('.theme');

    var defaultThemeId = themes.default.id;
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
        defaultThemeId = themes.dark.id;
    }
    function keys(obj): string[] {
        return Object.keys(obj);
    }

    add(new Select(themeSelect, 'theme',
            keys(themes).map(theme => {
                return {label: themes[theme].id, desc: themes[theme].name}
            })),
        defaultThemeId
    );
    add(new Checkbox(root.find('.showQuickSuggestions'), 'showQuickSuggestions'), false);
    add(new Checkbox(root.find('.useCustomContextMenu'), 'useCustomContextMenu'), true);
    add(new Checkbox(root.find('.showMinimap'), 'showMinimap'), true);

    var enableAllSchemesCheckbox = root.find('.alwaysEnableAllSchemes');
    add(new Checkbox(enableAllSchemesCheckbox, 'alwaysEnableAllSchemes'), false);

    function handleThemes() {
        var newTheme = themeSelect.val() as colour.AppTheme;
        // Store the scheme of the old theme
        $.data(themeSelect, 'theme-' + $.data(themeSelect, 'last-theme'), colourSchemeSelect.val());
        // Get the scheme of the new theme
        var newThemeStoredScheme = $.data(themeSelect, 'theme-' + newTheme);
        var isStoredUsable = false;
        colourSchemeSelect.empty();
        for (let scheme of colour.schemes) {
            if (currentSettings.alwaysEnableAllSchemes
                || !scheme.themes || scheme.themes.length === 0
                || scheme.themes.includes(newTheme) || scheme.themes.includes('all')) {

                colourSchemeSelect.append($('<option value="' + scheme.name + '">' + scheme.desc + '</option>'));
                if (newThemeStoredScheme === scheme.name) {
                    isStoredUsable = true;
                }
            }
        }
        if (colourSchemeSelect.children().length >= 1) {
            colourSchemeSelect.val(isStoredUsable ? newThemeStoredScheme : colourSchemeSelect.first().val());
        } else {
            // This should never happen. In case it does, lets use the default one
            colourSchemeSelect.append(
                $('<option value="' + colour.schemes[0].name + '">' + colour.schemes[0].desc + '</option>'));
            colourSchemeSelect.val(colourSchemeSelect.first().val());
        }
        colourSchemeSelect.trigger('change');
    }

    var langs = options.languages;

    var defaultLanguageSelector = root.find('.defaultLanguage');
    var defLang = settings.defaultLanguage || keys(langs)[0] || 'c++';
    add(new Select(defaultLanguageSelector, 'defaultLanguage',
            keys(langs).map(lang => {
                return {label: langs[lang].id, desc: langs[lang].name};
            })),
        defLang
    );
    if (subLangId) {
        defaultLanguageSelector
            .prop('disabled', true)
            .prop('title', 'Default language inherited from subdomain')
            .css('cursor', 'not-allowed');
    }

    add(new Checkbox(root.find('.newEditorLastLang'), 'newEditorLastLang'), true);

    var formats = ['Google', 'LLVM', 'Mozilla', 'Chromium', 'WebKit', 'Microsoft', 'GNU'];
    add(new Select(root.find('.formatBase'), 'formatBase', formats.map(format => {
            return {label: format, desc: format};
        })),
        formats[0]);
    //add(root.find('.formatOverrides'), 'formatOverrides', "", TextAreaInput);
    add(new Checkbox(root.find('.wordWrap'), 'wordWrap'), false);

    function setSettings(settings) {
        onSettingsChange(settings);
        onChange(settings);
    }
    add(new Checkbox(root.find('.useSpaces'), 'useSpaces'), true);
    add(new Numeric(root.find('.tabWidth'), 'tabWidth', {min: 1, max: 80}), 4);
    // note: this is the ctrl+s "Save option"
    add(new Checkbox(root.find('.enableCtrlS'), 'enableCtrlS'), true);
    add(new Checkbox(root.find('.enableCtrlStree'), 'enableCtrlStree'), true);
    add(new Textbox(root.find('.editorsFFont'), 'editorsFFont'), 'Consolas, "Liberation Mono", Courier, monospace');
    add(new Checkbox(root.find('.editorsFLigatures'), 'editorsFLigatures'), false);
    add(new Checkbox(root.find('.allowStoreCodeDebug'), 'allowStoreCodeDebug'), true);
    add(new Checkbox(root.find('.useVim'), 'useVim'), false);
    add(new Checkbox(root.find('.autoIndent'), 'autoIndent'), true);
    add(new Checkbox(root.find('.keepSourcesOnLangChange'), 'keepSourcesOnLangChange'), false);
    add(new Checkbox(root.find('.enableCodeLens'), 'enableCodeLens'), true);

    setSettings(settings);
    handleThemes();
    themeSelect.change(function () {
        handleThemes();
        $.data(themeSelect, 'last-theme', themeSelect.val());
    });
    enableAllSchemesCheckbox.change(handleThemes);

    $.data(themeSelect, 'last-theme', themeSelect.val());
    return setSettings;
}

export function getStoredSettings(): SiteSettings {
    return JSON.parse(local.get('settings', '{}'));
}
