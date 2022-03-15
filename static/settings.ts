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

import { options } from './options';
import * as colour from './colour';
import * as local from './local';
import { themes, Themes } from './themes';
import { AppTheme, ColourSchemeInfo } from './colour';

type ColourScheme =
    | 'rainbow'
    | 'rainbow2'
    | 'earth'
    | 'green-blue'
    | 'gray-shade'
    | 'rainbow-dark';

export type FormatBase =
    | 'Google'
    | 'LLVM'
    | 'Mozilla'
    | 'Chromium'
    | 'WebKit'
    | 'Microsoft'
    | 'GNU';

export interface SiteSettings {
    autoCloseBrackets: boolean;
    autoIndent: boolean;
    allowStoreCodeDebug: boolean;
    alwaysEnableAllSchemes: boolean;
    colouriseAsm: boolean;
    colourScheme: ColourScheme;
    compileOnChange: boolean;
    // TODO(supergrecko): make this more precise
    defaultLanguage?: string;
    delayAfterChange: number;
    enableCodeLens: boolean;
    enableCommunityAds: boolean
    enableCtrlS: string;
    enableSharingPopover: boolean;
    enableCtrlStree: boolean;
    editorsFFont: string
    editorsFLigatures: boolean;
    formatBase: FormatBase;
    formatOnCompile: boolean;
    hoverShowAsmDoc: boolean;
    hoverShowSource: boolean;
    keepSourcesOnLangChange: boolean;
    newEditorLastLang: boolean;
    showMinimap: boolean;
    showQuickSuggestions: boolean;
    tabWidth: number;
    theme: Themes;
    useCustomContextMenu: boolean;
    useSpaces: boolean;
    useVim: boolean;
    wordWrap: boolean;
}

class BaseSetting {
    constructor(public elem: JQuery, public name: string) {}

    protected val(): string | number | string[] {
        //  If it's undefined, something went wrong, so the following exception is helpful
        // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
        return this.elem.val()!;
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
    constructor(elem: JQuery, name: string, populate: {label: string, desc: string}[]) {
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

        elem
            .prop('max', this.max)
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

        elem.attr('min', this.min)
            .attr('max', this.max);
    }

    override getUi(): number {
        return this.clampValue(parseInt(this.val().toString()));
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

    constructor(private root: JQuery,
                private settings: SiteSettings,
                private onChange: (SiteSettings) => void,
                private subLangId: string | null) {

        this.settings = settings;
        this.settings.defaultLanguage = this.settings.defaultLanguage === null ? undefined : settings.defaultLanguage;
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
        const checkboxes: [string, keyof SiteSettings, boolean][]= [
            ['.allowStoreCodeDebug', 'allowStoreCodeDebug', true],
            ['.alwaysEnableAllSchemes', 'alwaysEnableAllSchemes', false],
            ['.autoCloseBrackets', 'autoCloseBrackets', true],
            ['.autoIndent', 'autoIndent', true],
            ['.colourise', 'colouriseAsm', true],
            ['.compileOnChange', 'compileOnChange', true],
            ['.editorsFLigatures', 'editorsFLigatures', false],
            ['.enableCodeLens', 'enableCodeLens', true],
            ['.enableCommunityAds', 'enableCommunityAds', true],
            ['.enableCtrlStree', 'enableCtrlStree', true],
            ['.enableSharingPopover', 'enableSharingPopover', true],
            ['.formatOnCompile', 'formatOnCompile', false],
            ['.hoverShowAsmDoc', 'hoverShowAsmDoc', true],
            ['.hoverShowSource', 'hoverShowSource', true],
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
        const addSelector = (
            selector: string,
            name: keyof SiteSettings,
            populate: {label: string, desc: string}[],
            defaultValue: string
        ) => {
            this.add(new Select(this.root.find(selector), name, populate), defaultValue);
        };

        const colourSchemesData = colour.schemes.map(scheme => {
            return {label: scheme.name, desc: scheme.desc};
        });
        addSelector('.colourScheme', 'colourScheme', colourSchemesData, colour.schemes[0].name);

        // keys(themes) is Themes[] but TS does not realize without help
        const themesData = (Object.keys(themes) as Themes[]).map((theme: Themes) => {
            return {label: themes[theme].id, desc: themes[theme].name};
        });
        let defaultThemeId = themes.default.id;
        if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
            defaultThemeId = themes.dark.id;
        }
        addSelector('.theme', 'theme', themesData, defaultThemeId);

        const langs = options.languages;
        const defaultLanguageSelector = this.root.find('.defaultLanguage');
        const defLang = this.settings.defaultLanguage || Object.keys(langs)[0] || 'c++';

        const defaultLanguageData = Object.keys(langs).map(lang => {
            return {label: langs[lang].id, desc: langs[lang].name};
        });
        addSelector('.defaultLanguage', 'defaultLanguage', defaultLanguageData, defLang);

        if (this.subLangId) {
            defaultLanguageSelector
                .prop('disabled', true)
                .prop('title', 'Default language inherited from subdomain')
                .css('cursor', 'not-allowed');
        }

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
        this.add(new Numeric(this.root.find('.tabWidth'), 'tabWidth', {min: 1, max: 80}), 4);
    }

    private addTextBoxes() {
        this.add(
            new Textbox(this.root.find('.editorsFFont'), 'editorsFFont'),
            'Consolas, "Liberation Mono", Courier, monospace'
        );
    }

    private handleThemes() {
        const themeSelect = this.root.find('.theme');
        themeSelect.on('change', () => {
            this.onThemeChange();
            $.data(themeSelect, 'last-theme', themeSelect.val() as string);
        });

        const colourSchemeSelect = this.root.find('.colourScheme');
        colourSchemeSelect.on('change', (e) => {
            const currentTheme = this.settings.theme;
            $.data(themeSelect, 'theme-' + currentTheme, colourSchemeSelect.val() as ColourScheme);
        });

        const enableAllSchemesCheckbox = this.root.find('.alwaysEnableAllSchemes');
        enableAllSchemesCheckbox.on('change', this.onThemeChange.bind(this));

        $.data(themeSelect, 'last-theme', themeSelect.val() as string);
    }

    private fillThemeSelector(colourSchemeSelect: JQuery, newTheme?: AppTheme) {
        for (const scheme of colour.schemes) {
            if (this.isSchemeUsable(scheme, newTheme)) {
                colourSchemeSelect.append($(`<option value="${scheme.name}">${scheme.desc}</option>`));
            }
        }
    }

    private isSchemeUsable(scheme: ColourSchemeInfo, newTheme?: AppTheme): boolean {
        return this.settings.alwaysEnableAllSchemes
            || !scheme.themes || scheme.themes.length === 0
            || (newTheme && scheme.themes.includes(newTheme)) || scheme.themes.includes('all');
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

        const oldScheme = colourSchemeSelect.val() as string;
        const newTheme = themeSelect.val() as colour.AppTheme;

        colourSchemeSelect.empty();
        this.fillThemeSelector(colourSchemeSelect, newTheme);
        const newThemeStoredScheme = $.data(themeSelect, 'theme-' + newTheme) as colour.AppTheme | undefined;

        // If nothing else, set the new scheme to the first of the available ones
        let newScheme = colourSchemeSelect.first().val() as string;
        // If we have one old one stored, check if it's still valid and set it if so
        if (newThemeStoredScheme && this.selectorHasOption(colourSchemeSelect, newThemeStoredScheme)) {
            newScheme = newThemeStoredScheme;
        } else if (this.selectorHasOption(colourSchemeSelect, oldScheme)) {
            newScheme = oldScheme;
        }

        colourSchemeSelect.val(newScheme);

        colourSchemeSelect.trigger('change');
    }
}
