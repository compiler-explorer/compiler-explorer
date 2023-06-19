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
import {editor} from 'monaco-editor';
import {SiteSettings} from './settings.js';
import GoldenLayout from 'golden-layout';
import {isString} from '../lib/common-utils.js';

export type Themes = 'default' | 'dark' | 'darkplus' | 'pink' | 'system';

export type Theme = {
    path: string;
    id: Themes;
    name: string;
    mainColor: string;
    monaco: string;
};

export const themes: Record<Themes, Theme> = {
    default: {
        path: 'default',
        id: 'default',
        name: 'Light',
        mainColor: '#f2f2f2',
        monaco: 'ce',
    },
    dark: {
        path: 'dark',
        id: 'dark',
        name: 'Dark',
        mainColor: '#333333',
        monaco: 'ce-dark',
    },
    darkplus: {
        path: 'dark',
        id: 'darkplus',
        name: 'Dark+',
        mainColor: '#333333',
        monaco: 'ce-dark-plus',
    },
    pink: {
        path: 'pink',
        id: 'pink',
        name: 'Pink',
        mainColor: '#333333',
        monaco: 'ce-pink',
    },
    system: {
        id: 'system',
        name: 'Same as system',
        path: 'default',
        mainColor: '#f2f2f2',
        monaco: 'ce',
    },
};

editor.defineTheme('ce', {
    base: 'vs',
    inherit: true,
    rules: [
        {
            token: 'identifier.definition.herb',
            foreground: '008a00',
            fontStyle: 'bold',
        },
        {token: 'keyword.identifier.definition.herb', fontStyle: 'bold'},
    ],
    colors: {
        // There seems to be a monaco bug when switching between themes with the minimap's background not updating
        'editor.background': '#FFFFFE',
    },
});

editor.defineTheme('ce-dark', {
    base: 'vs-dark',
    inherit: true,
    rules: [
        {
            token: 'identifier.definition.herb',
            foreground: '7c9c7c',
            fontStyle: 'bold',
        },
        {token: 'keyword.identifier.definition.herb', fontStyle: 'bold'},
    ],
    colors: {},
});

editor.defineTheme('ce-dark-plus', {
    base: 'vs-dark',
    inherit: true,
    rules: [
        {
            token: 'identifier.definition.herb',
            foreground: '7c9c7c',
            fontStyle: 'bold',
        },
        {token: 'keyword.identifier.definition.herb', fontStyle: 'bold'},
        {token: 'keyword.if.cpp', foreground: 'b66bb0'},
        {token: 'keyword.else.cpp', foreground: 'b66bb0'},
        {token: 'keyword.while.cpp', foreground: 'b66bb0'},
        {token: 'keyword.for.cpp', foreground: 'b66bb0'},
        {token: 'keyword.return.cpp', foreground: 'b66bb0'},
        {token: 'keyword.break.cpp', foreground: 'b66bb0'},
        {token: 'keyword.continue.cpp', foreground: 'b66bb0'},
        {token: 'keyword.goto.cpp', foreground: 'b66bb0'},
        {token: 'keyword.directive.cpp', foreground: 'b66bb0'},
        {token: 'keyword.directive.include.cpp', foreground: 'b66bb0'},
        {token: 'keyword.directive.include.begin.cpp', foreground: 'ce9178'},
        {token: 'keyword.directive.include.end.cpp', foreground: 'ce9178'},
        {token: 'keyword.new.cpp', foreground: 'b66bb0'},
        {token: 'keyword.using.cpp', foreground: 'b66bb0'},
        {token: 'string.escape.cpp', foreground: 'd7ba7d'},
    ],
    colors: {},
});

editor.defineTheme('ce-pink', {
    base: 'vs',
    inherit: true,
    rules: [
        ...(() => {
            // based on https://github.com/huacat1017/huacat.pink-theme/blob/master/themes/pink-theme-color-theme.json
            const base: {
                name?: string;
                scope: string | string[];
                settings: {foreground?: string; fontStyle?: string};
            }[] = [
                {
                    name: 'Punctuation',
                    scope: ['punctuation.separator', 'punctuation.terminator'],
                    settings: {
                        foreground: '#dd71b9',
                    },
                },
                {
                    name: 'Comment',
                    scope: ['comment.line', 'comment.block', 'comment', 'punctuation.definition.comment'],
                    settings: {
                        foreground: '#978c94',
                        fontStyle: 'italic',
                    },
                },
                {
                    name: 'Class',
                    scope: ['entity.name.type', 'entity.other.inherited-class', 'support.class'],
                    settings: {
                        foreground: '#e2b213',
                    },
                },
                {
                    name: 'Function',
                    scope: ['entity.name.function', 'support.function', 'meta.function-call.generic'],
                    settings: {
                        foreground: '#a770db',
                    },
                },
                {
                    name: 'Parameter',
                    scope: 'variable.parameter',
                    settings: {
                        foreground: '#e45d1e',
                    },
                },
                {
                    name: 'Keyword',
                    scope: ['keyword', 'storage'],
                    settings: {
                        foreground: '#5782df',
                        fontStyle: 'italic',
                    },
                },
                {
                    name: 'Type',
                    scope: ['storage.type', 'support.type'],
                    settings: {
                        foreground: '#62ad44',
                        fontStyle: '',
                    },
                },
                {
                    name: 'Number',
                    scope: [
                        'constant.numeric',
                        'constant.language',
                        'constant.character',
                        'constant',
                        'number',
                        'keyword.other',
                    ],
                    settings: {
                        foreground: '#db6363',
                        fontStyle: '',
                    },
                },
                {
                    name: 'Operator',
                    scope: 'keyword.operator',
                    settings: {
                        foreground: '#dba15e',
                        fontStyle: '',
                    },
                },
                {
                    name: 'Delimiter',
                    scope: ['delimiter'],
                    settings: {
                        foreground: '#72696F',
                        fontStyle: '',
                    },
                },
                {
                    name: 'String, Symbols',
                    scope: ['string'],
                    settings: {
                        foreground: '#64afa9',
                    },
                },
                {
                    name: 'Variable',
                    scope: ['support.variable', 'variable'],
                    settings: {
                        foreground: '#72696f',
                    },
                },
                {
                    name: 'Strings: Escape Sequences',
                    scope: 'constant.character.escape',
                    settings: {
                        foreground: '#559fe4',
                    },
                },
                {
                    name: 'HTML CSS XML name',
                    scope: ['entity.other.attribute-name'],
                    settings: {
                        foreground: '#db6363',
                    },
                },
                {
                    name: 'HTML XML tag outer',
                    scope: ['meta.tag'],
                    settings: {
                        foreground: '#559fe4',
                    },
                },
                {
                    name: 'HTML XML tag inner',
                    scope: ['entity.name.tag'],
                    settings: {
                        foreground: '#5782df',
                    },
                },
                {
                    name: 'CSS class',
                    scope: ['entity.other.attribute-name.class'],
                    settings: {
                        foreground: '#dba15e',
                    },
                },
                {
                    name: 'CSS id',
                    scope: ['entity.other.attribute-name.id'],
                    settings: {
                        foreground: '#9045d6',
                    },
                },
                {
                    name: 'CSS Selector',
                    scope: ['support.constant.property-value'],
                    settings: {
                        foreground: '#62ad44',
                    },
                },
                {
                    name: 'Markdown heading',
                    scope: ['markup.heading', 'markup.heading entity.name'],
                    settings: {
                        foreground: '#5782df',
                    },
                },
                {
                    name: 'Markdown list',
                    scope: ['markup.list punctuation.definition.list.begin'],
                    settings: {
                        foreground: '#dd71b9',
                    },
                },
                {
                    name: 'Markdown link',
                    scope: ['markup.underline.link'],
                    settings: {
                        foreground: '#a770db',
                    },
                },
                {
                    name: 'Markdown bold',
                    scope: ['markup.bold'],
                    settings: {
                        foreground: '#db6363',
                        fontStyle: 'bold',
                    },
                },
                {
                    name: 'Markdown italic',
                    scope: ['markup.italic'],
                    settings: {
                        foreground: '#db6363',
                        fontStyle: 'italic',
                    },
                },
                {
                    name: 'Markdown italic bold',
                    scope: ['markup.italic markup.bold', 'markup.bold markup.italic'],
                    settings: {
                        fontStyle: 'italic bold',
                    },
                },
                {
                    name: 'Markdown code',
                    scope: ['text.html.markdown markup.inline.raw'],
                    settings: {
                        foreground: '#62ad44',
                    },
                },
                {
                    name: 'Markdown quote',
                    scope: ['markup.quote'],
                    settings: {
                        foreground: '#3d8b1c',
                        fontStyle: 'italic',
                    },
                },
            ];
            const monacoRules: editor.ITokenThemeRule[] = [];
            for (const rule of base) {
                if (isString(rule.scope)) {
                    monacoRules.push({
                        token: rule.scope,
                        ...rule.settings,
                    });
                } else {
                    for (const scope of rule.scope) {
                        monacoRules.push({
                            token: scope,
                            ...rule.settings,
                        });
                    }
                }
            }
            return monacoRules;
        })(),
        {token: 'keyword.if.cpp', foreground: 'b66bb0'},
        {token: 'keyword.else.cpp', foreground: 'b66bb0'},
        {token: 'keyword.while.cpp', foreground: 'b66bb0'},
        {token: 'keyword.for.cpp', foreground: 'b66bb0'},
        {token: 'keyword.return.cpp', foreground: 'b66bb0'},
        {token: 'keyword.break.cpp', foreground: 'b66bb0'},
        {token: 'keyword.continue.cpp', foreground: 'b66bb0'},
        {token: 'keyword.goto.cpp', foreground: 'b66bb0'},
        {token: 'keyword.directive.cpp', foreground: 'b66bb0'},
        {token: 'keyword.directive.include.cpp', foreground: 'b66bb0'},
        {token: 'keyword.new.cpp', foreground: 'b66bb0'},
        {token: 'keyword.using.cpp', foreground: 'b66bb0'},
        {token: 'keyword.directive.include.begin.cpp', foreground: 'ce9178'},
        {token: 'keyword.directive.include.end.cpp', foreground: 'ce9178'},
        {token: 'string.escape.cpp', foreground: 'df945a'},
    ],
    colors: {
        'editor.background': '#fae1fa',
        'editor.foreground': '#72696f',
        'editor.lineHighlightBorder': '#e3a5e3',
        'editor.wordHighlightTextBackground': '#e3a5e3',
        'editorHoverWidget.background': '#fae1fa',
        'editorLineNumber.foreground': '#e787e7',
        'editorLineNumber.activeForeground': '#d12cd1',
        'editor.selectionBackground': '#e3a5e3',
        'editor.inactiveSelectionBackground': '#f0bcf0',
        'minimap.selectionHighlight': '#e3a5e3',
    },
});

export class Themer {
    private currentTheme: Theme | null = null;

    constructor(private eventHub: GoldenLayout.EventEmitter, initialSettings: SiteSettings) {
        this.onSettingsChange(initialSettings);

        this.eventHub.on('settingsChange', this.onSettingsChange, this);

        this.eventHub.on(
            'requestTheme',
            () => {
                this.eventHub.emit('themeChange', this.currentTheme);
            },
            this,
        );
    }

    public setTheme(theme: Theme) {
        if (this.currentTheme === theme) return;
        if (theme.id === 'system') {
            if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
                theme = themes.dark;
            } else {
                theme = themes.default;
            }
        }
        $('html').attr('data-theme', theme.path);
        $('#meta-theme').prop('content', theme.mainColor);
        editor.setTheme(theme.monaco);
        this.eventHub.emit('resize');
        this.currentTheme = theme;
    }

    private onSettingsChange(newSettings: SiteSettings) {
        const newTheme = newSettings.theme && newSettings.theme in themes ? themes[newSettings.theme] : themes.default;
        if (!newTheme.monaco) newTheme.monaco = 'vs';
        this.setTheme(newTheme);
    }
}
