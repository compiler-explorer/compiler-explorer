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
import {options} from './options.js';
import * as colour from './colour.js';
import * as local from './local.js';
import {themes, Themes} from './themes.js';
import {AppTheme, ColourScheme, ColourSchemeInfo} from './colour.js';
import {Hub} from './hub.js';
import {EventHub} from './event-hub.js';
import {keys, isString} from '../lib/common-utils.js';
import {assert, unwrapString} from './assert.js';

import {LanguageKey} from '../types/languages.interfaces.js';

export type FormatBase = 'Google' | 'LLVM' | 'Mozilla' | 'Chromium' | 'WebKit' | 'Microsoft' | 'GNU';

export interface SiteSettings {
    autoCloseBrackets: boolean;
    autoCloseQuotes: boolean;
    autoSurround: boolean;
    autoIndent: boolean;
    allowStoreCodeDebug: boolean;
    alwaysEnableAllSchemes: boolean;
    colouriseAsm: boolean;
    colourScheme: ColourScheme;
    compileOnChange: boolean;
    defaultLanguage?: LanguageKey;
    delayAfterChange: number;
    enableCodeLens: boolean;
    enableCommunityAds: boolean;
    enableCtrlS: string;
    enableSharingPopover: boolean;
    enableCtrlStree: boolean;
    editorsFFont: string;
    editorsFLigatures: boolean;
    executorCompileOnChange: boolean;
    defaultFontScale?: number; // the font scale widget can check this setting before the default has been populated
    formatBase: FormatBase;
    formatOnCompile: boolean;
    hoverShowAsmDoc: boolean;
    hoverShowSource: boolean;
    indefiniteLineHighlight: boolean;
    keepMultipleTabs: boolean;
    keepSourcesOnLangChange: boolean;
    newEditorLastLang: boolean;
    showMinimap: boolean;
    showQuickSuggestions: boolean;
    tabWidth: number;
    theme: Themes | undefined;
    useCustomContextMenu: boolean;
    useSpaces: boolean;
    useVim: boolean;
    wordWrap: boolean;
}

class BaseSetting {
    constructor(public elem: JQuery, public name: string) {}

    // Can be undefined if the element doesn't exist which is the case in embed mode
    protected val(): string | number | string[] | undefined {
        return this.elem.val();
    }

    getUi(): any {
        return this.val();
    }

    putUi(value: any): void {
        this.elem.val(value);
    }
}

class Checkbox extends BaseSetting {
    override getUi(): boolean {
        return !!this.elem.prop('checked');
    }

    override putUi(value: any) {
        this.elem.prop('checked', !!value);
    }
}

class Select extends BaseSetting {
    constructor(elem: JQuery, name: string, populate: {label: string; desc: string}[]) {
        super(elem, name);

        elem.empty();
        for (const e of populate) {
            elem.append($(`<option value="${e.label}">${e.desc}</option>`));
        }
    }

    override putUi(value: string | number | boolean | null) {
        this.elem.val(value?.toString() ?? '');
    }
}

class NumericSelect extends Select {
    constructor(elem: JQuery, name: string, populate: {label: string; desc: string}[]) {
        super(elem, name, populate);
    }
    override getUi(): number {
        return Number(this.val());
    }
}

interface SliderSettings {
    min: number;
    max: number;
    step: number;
    display: JQuery;
    formatter: (number) => string;
}

class Slider extends BaseSetting {
    private readonly formatter: (number) => string;
    private display: JQuery;
    private max: number;
    private min: number;

    constructor(elem: JQuery, name: string, sliderSettings: SliderSettings) {
        super(elem, name);

        this.formatter = sliderSettings.formatter;
        this.display = sliderSettings.display;

        this.max = sliderSettings.max || 100;
        this.min = sliderSettings.min || 1;

        elem.prop('max', this.max)
            .prop('min', this.min)
            .prop('step', sliderSettings.step || 1);

        elem.on('change', this.updateDisplay.bind(this));
    }

    override putUi(value: number) {
        this.elem.val(value);
        this.updateDisplay();
    }

    override getUi(): number {
        return parseInt(this.val()?.toString() ?? '0');
    }

    private updateDisplay() {
        this.display.text(this.formatter(this.getUi()));
    }
}

class Textbox extends BaseSetting {}

class Numeric extends BaseSetting {
    private readonly min: number;
    private readonly max: number;

    constructor(elem: JQuery, name: string, params: Record<'min' | 'max', number>) {
        super(elem, name);

        this.min = params.min;
        this.max = params.max;

        elem.attr('min', this.min).attr('max', this.max);
    }

    override getUi(): number {
        return this.clampValue(parseInt(this.val()?.toString() ?? '0'));
    }

    override putUi(value: number) {
        this.elem.val(this.clampValue(value));
    }

    private clampValue(value: number): number {
        return Math.min(Math.max(value, this.min), this.max);
    }
}

export class Settings {
    private readonly settingsObjs: BaseSetting[];
    private eventHub: EventHub;

    constructor(
        hub: Hub,
        private root: JQuery,
        private settings: SiteSettings,
        private onChange: (siteSettings: SiteSettings) => void,
        private subLangId: string | undefined,
    ) {
        this.eventHub = hub.createEventHub();
        this.settings = settings;
        this.settingsObjs = [];

        this.addCheckboxes();
        this.addSelectors();
        this.addSliders();
        this.addNumerics();
        this.addTextBoxes();

        this.setSettings(this.settings);
        this.handleThemes();
    }

    public static getStoredSettings(): SiteSettings {
        return JSON.parse(local.get('settings', '{}'));
    }

    public setSettings(newSettings: SiteSettings) {
        this.onSettingsChange(newSettings);
        this.onChange(newSettings);
    }

    private onUiChange() {
        for (const setting of this.settingsObjs) {
            this.settings[setting.name] = setting.getUi();
        }
        this.onChange(this.settings);
    }

    private onSettingsChange(settings: SiteSettings) {
        this.settings = settings;
        for (const setting of this.settingsObjs) {
            setting.putUi(this.settings[setting.name]);
        }
    }

    private add<T extends BaseSetting>(setting: T, defaultValue: any) {
        const key = setting.name;
        if (this.settings[key] === undefined) this.settings[key] = defaultValue;
        this.settingsObjs.push(setting);
        setting.elem.on('change', this.onUiChange.bind(this));
    }

    private addCheckboxes() {
        // Known checkbox options in order [selector, key, defaultValue]
        const checkboxes: [string, keyof SiteSettings, boolean][] = [
            ['.allowStoreCodeDebug', 'allowStoreCodeDebug', true],
            ['.alwaysEnableAllSchemes', 'alwaysEnableAllSchemes', false],
            ['.autoCloseBrackets', 'autoCloseBrackets', true],
            ['.autoCloseQuotes', 'autoCloseQuotes', true],
            ['.autoSurround', 'autoSurround', true],
            ['.autoIndent', 'autoIndent', true],
            ['.colourise', 'colouriseAsm', true],
            ['.compileOnChange', 'compileOnChange', true],
            ['.editorsFLigatures', 'editorsFLigatures', false],
            ['.enableCodeLens', 'enableCodeLens', true],
            ['.enableCommunityAds', 'enableCommunityAds', true],
            ['.enableCtrlStree', 'enableCtrlStree', true],
            ['.enableSharingPopover', 'enableSharingPopover', true],
            ['.executorCompileOnChange', 'executorCompileOnChange', true],
            ['.formatOnCompile', 'formatOnCompile', false],
            ['.hoverShowAsmDoc', 'hoverShowAsmDoc', true],
            ['.hoverShowSource', 'hoverShowSource', true],
            ['.indefiniteLineHighlight', 'indefiniteLineHighlight', false],
            ['.keepMultipleTabs', 'keepMultipleTabs', false],
            ['.keepSourcesOnLangChange', 'keepSourcesOnLangChange', false],
            ['.newEditorLastLang', 'newEditorLastLang', true],
            ['.showMinimap', 'showMinimap', true],
            ['.showQuickSuggestions', 'showQuickSuggestions', false],
            ['.useCustomContextMenu', 'useCustomContextMenu', true],
            ['.useSpaces', 'useSpaces', true],
            ['.useVim', 'useVim', false],
            ['.wordWrap', 'wordWrap', false],
        ];

        for (const [selector, name, defaultValue] of checkboxes) {
            this.add(new Checkbox(this.root.find(selector), name), defaultValue);
        }
    }

    private addSelectors() {
        const addSelector = <Name extends keyof SiteSettings>(
            selector: string,
            name: Name,
            populate: {label: string; desc: string}[],
            defaultValue: SiteSettings[Name],
            component = Select,
        ) => {
            const instance = new component(this.root.find(selector), name, populate);
            this.add(instance, defaultValue);
            return instance;
        };

        // We need theme data to populate the colour schemes; We don't add the selector until later
        const themesData = keys(themes).map((theme: Themes) => {
            return {label: themes[theme].id, desc: themes[theme].name};
        });
        const defaultThemeId = themes.system.id;

        const colourSchemesData = colour.schemes
            .filter(scheme => this.isSchemeUsable(scheme, defaultThemeId))
            .map(scheme => ({label: scheme.name, desc: scheme.desc}));
        let defaultColourScheme = colour.schemes[0].name;
        if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
            defaultColourScheme = 'gray-shade';
        }
        addSelector('.colourScheme', 'colourScheme', colourSchemesData, defaultColourScheme);

        // Now add the theme selector
        addSelector('.theme', 'theme', themesData, defaultThemeId);

        const langs = options.languages;
        const defaultLanguageSelector = this.root.find('.defaultLanguage');
        const defLang = this.settings.defaultLanguage || Object.keys(langs)[0] || 'c++';

        const defaultLanguageData = Object.keys(langs).map(lang => {
            return {label: langs[lang].id, desc: langs[lang].name};
        });
        addSelector('.defaultLanguage', 'defaultLanguage', defaultLanguageData, defLang as LanguageKey);

        if (this.subLangId) {
            defaultLanguageSelector
                .prop('disabled', true)
                .prop('title', 'Default language inherited from subdomain')
                .css('cursor', 'not-allowed');
        }

        const defaultFontScale = options.defaultFontScale;
        const fontScales: {label: string; desc: string}[] = [];
        for (let i = 8; i <= 30; i++) {
            fontScales.push({label: i.toString(), desc: i.toString()});
        }
        const defaultFontScaleSelector = addSelector(
            '.defaultFontScale',
            'defaultFontScale',
            fontScales,
            defaultFontScale,
            NumericSelect,
        ).elem;
        defaultFontScaleSelector.on('change', e => {
            assert(e.target instanceof HTMLSelectElement);
            this.eventHub.emit('broadcastFontScale', parseInt(e.target.value));
        });

        const formats: FormatBase[] = ['Google', 'LLVM', 'Mozilla', 'Chromium', 'WebKit', 'Microsoft', 'GNU'];
        const formatsData = formats.map(format => {
            return {label: format, desc: format};
        });
        addSelector('.formatBase', 'formatBase', formatsData, formats[0]);

        const enableCtrlSData = [
            {label: 'true', desc: 'Save To Local File'},
            {label: 'false', desc: 'Create Short Link'},
            {label: '2', desc: 'Reformat code'},
            {label: '3', desc: 'Do nothing'},
        ];
        addSelector('.enableCtrlS', 'enableCtrlS', enableCtrlSData, 'true');
    }

    private addSliders() {
        // Handle older settings
        if (this.settings.delayAfterChange === 0) {
            this.settings.delayAfterChange = 750;
            this.settings.compileOnChange = false;
        }

        const delayAfterChangeSettings: SliderSettings = {
            max: 3000,
            step: 250,
            min: 250,
            display: this.root.find('.delay-current-value'),
            formatter: x => (x / 1000.0).toFixed(2) + 's',
        };
        this.add(new Slider(this.root.find('.delay'), 'delayAfterChange', delayAfterChangeSettings), 750);
    }

    private addNumerics() {
        this.add(
            new Numeric(this.root.find('.tabWidth'), 'tabWidth', {
                min: 1,
                max: 80,
            }),
            4,
        );
    }

    private addTextBoxes() {
        this.add(
            new Textbox(this.root.find('.editorsFFont'), 'editorsFFont'),
            'Consolas, "Liberation Mono", Courier, monospace',
        );
    }

    private handleThemes() {
        const themeSelect = this.root.find('.theme');
        themeSelect.on('change', () => {
            this.onThemeChange();
            $.data(themeSelect, 'last-theme', unwrapString(themeSelect.val()));
        });

        const colourSchemeSelect = this.root.find('.colourScheme');
        colourSchemeSelect.on('change', e => {
            const currentTheme = this.settings.theme;
            $.data(themeSelect, 'theme-' + currentTheme, unwrapString<ColourScheme>(colourSchemeSelect.val()));
        });

        const enableAllSchemesCheckbox = this.root.find('.alwaysEnableAllSchemes');
        enableAllSchemesCheckbox.on('change', this.onThemeChange.bind(this));

        // In embed mode themeSelect.length can be zero and thus themeSelect.val() isn't a string
        // TODO(jeremy-rifkin) Is last-theme ever read? Can it just be removed?
        $.data(themeSelect, 'last-theme', themeSelect.val() ?? '');
        this.onThemeChange();
    }

    private fillColourSchemeSelector(colourSchemeSelect: JQuery, newTheme?: AppTheme) {
        colourSchemeSelect.empty();
        if (newTheme === 'system') {
            if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
                newTheme = themes.dark.id;
            } else {
                newTheme = themes.default.id;
            }
        }
        for (const scheme of colour.schemes) {
            if (this.isSchemeUsable(scheme, newTheme)) {
                colourSchemeSelect.append($(`<option value="${scheme.name}">${scheme.desc}</option>`));
            }
        }
    }

    private isSchemeUsable(scheme: ColourSchemeInfo, newTheme?: AppTheme): boolean {
        return (
            this.settings.alwaysEnableAllSchemes ||
            scheme.themes.length === 0 ||
            (newTheme && scheme.themes.includes(newTheme)) ||
            scheme.themes.includes('all')
        );
    }

    private selectorHasOption(selector: JQuery, option: string): boolean {
        return selector.children(`[value=${option}]`).length > 0;
    }

    private onThemeChange() {
        // We can be called when:
        // Site is initializing (settings and dropdowns are already done)
        // "Make all colour schemes available" changes
        // Selected theme changes
        const themeSelect = this.root.find('.theme');
        const colourSchemeSelect = this.root.find('.colourScheme');

        const oldScheme = colourSchemeSelect.val() as colour.AppTheme | undefined;
        const newTheme = themeSelect.val() as colour.AppTheme | undefined;

        // Small check to make sure we aren't getting something completely unexpected, like a string[] or number
        assert(
            // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
            isString(oldScheme) || oldScheme === undefined || oldScheme == null,
            'Unexpected value received from colourSchemeSelect.val()',
        );
        assert(
            // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
            isString(newTheme) || newTheme === undefined || newTheme == null,
            'Unexpected value received from colourSchemeSelect.val()',
        );

        this.fillColourSchemeSelector(colourSchemeSelect, newTheme);
        const newThemeStoredScheme = $.data(themeSelect, 'theme-' + newTheme) as colour.AppTheme | undefined;

        // If nothing else, set the new scheme to the first of the available ones
        let newScheme = colourSchemeSelect.first().val() as colour.AppTheme | undefined;
        // If we have one old one stored, check if it's still valid and set it if so
        if (newThemeStoredScheme && this.selectorHasOption(colourSchemeSelect, newThemeStoredScheme)) {
            newScheme = newThemeStoredScheme;
        } else if (isString(oldScheme) && this.selectorHasOption(colourSchemeSelect, oldScheme)) {
            newScheme = oldScheme;
        }

        if (newScheme) {
            colourSchemeSelect.val(newScheme);
        }

        colourSchemeSelect.trigger('change');
    }
}
