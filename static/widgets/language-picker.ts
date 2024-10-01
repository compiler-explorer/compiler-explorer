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

import {Hub} from '../hub.js';
import {assert, unwrap} from '../assert.js';
import TomSelect from 'tom-select';
import {options} from '../options.js';
import {Language} from '../../types/languages.interfaces.js';
import type {escape_html} from 'tom-select/dist/types/utils';

export type LanguageSelectData = Language & {
    logoData?: string;
    logoDataDark?: string;
};

const languages = options.languages;

export class LanguagePicker {
    domNode: HTMLSelectElement;
    tomSelect: TomSelect | null;

    public currentLanguage?: Language;

    constructor(
        domRoot: JQuery,
        hub: Hub,
        onLanguageChange: (newLang: Language, oldLang?: Language) => any,
        initialLanguage?: Language,
    ) {
        const languagePicker = domRoot.find('.change-language')[0];
        assert(languagePicker instanceof HTMLSelectElement);
        this.domNode = languagePicker;

        const usableLanguages = Object.values(languages).filter(language => {
            return hub.compilerService.getCompilersForLang(language.id);
        });

        this.currentLanguage = initialLanguage;

        this.initialize(usableLanguages, onLanguageChange);
    }

    private initialize(usableLanguages: Language[], onLanguageChange: (newLang: Language, oldLang?: Language) => any) {
        this.tomSelect = new TomSelect(this.domNode, {
            sortField: 'name',
            valueField: 'id',
            labelField: 'name',
            searchField: ['name'],
            placeholder: '🔍 Select a language...',
            options: [...usableLanguages],
            items: this.currentLanguage?.id ? [this.currentLanguage.id] : [],
            dropdownParent: 'body',
            plugins: ['dropdown_input'],
            maxOptions: 1000,
            onChange: langId => {
                if (!(langId in languages)) return;
                const oldLang = this.currentLanguage;
                this.currentLanguage = languages[langId];
                onLanguageChange(unwrap(this.currentLanguage), oldLang);
            },
            closeAfterSelect: true,
            render: {
                option: this.renderSelectizeOption.bind(this),
                item: this.renderSelectizeItem.bind(this),
            },
        });
        this.tomSelect.on('dropdown_close', () => {
            // scroll back to the selection on the next open
            const selection = unwrap(this.tomSelect).getOption(this.currentLanguage?.id ?? '');
            unwrap(this.tomSelect).setActiveOption(selection);
        });
    }

    getSelectizeRenderHtml(
        data: LanguageSelectData,
        escape: typeof escape_html,
        width: number,
        height: number,
    ): string {
        let result =
            '<div class="d-flex" style="align-items: center">' +
            '<div class="mr-1 d-flex" style="align-items: center">' +
            '<img src="' +
            (data.logoData ? data.logoData : '') +
            '" class="' +
            (data.logoDataDark ? 'theme-light-only' : '') +
            '" width="' +
            width +
            '" style="max-height: ' +
            height +
            'px"/>';
        if (data.logoDataDark) {
            result +=
                '<img src="' +
                data.logoDataDark +
                '" class="theme-dark-only" width="' +
                width +
                '" style="max-height: ' +
                height +
                'px"/>';
        }

        result += '</div><div';
        if (data.tooltip) {
            result += ' title="' + data.tooltip + '"';
        }
        result += '>' + escape(data.name) + '</div></div>';
        return result;
    }

    renderSelectizeOption(data: LanguageSelectData, escape: typeof escape_html) {
        return this.getSelectizeRenderHtml(data, escape, 23, 23);
    }

    renderSelectizeItem(data: LanguageSelectData, escape: typeof escape_html) {
        return this.getSelectizeRenderHtml(data, escape, 20, 20);
    }

    changeLanguage(newLang: string): void {
        if (!this.tomSelect) {
            // In some initialization flows we get here before creating this.selectize
            setTimeout(() => this.changeLanguage(newLang), 0);
        } else {
            if (newLang === 'cmake') {
                this.tomSelect.addOption(unwrap(languages.cmake));
            }
            this.tomSelect.setValue(newLang);
        }
    }
}
