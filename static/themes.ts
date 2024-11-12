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

import GoldenLayout from 'golden-layout';
import $ from 'jquery';
import {editor} from 'monaco-editor';
import {isString} from '../shared/common-utils.js';
import {options} from './options.js';
import {SiteSettings} from './settings.js';

export type Themes = 'default' | 'dark' | 'darkplus' | 'pink' | 'onedark' | 'real-dark' | 'system';

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
    'real-dark': {
        path: 'dark',
        id: 'real-dark',
        name: 'Real dark',
        mainColor: '#333333',
        monaco: 'ce-dark',
    },
    onedark: {
        path: 'one-dark',
        id: 'onedark',
        name: 'One Dark',
        mainColor: '#282c34',
        monaco: 'ce-one-dark',
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

editor.defineTheme('ce-one-dark', {
    base: 'vs-dark',
    inherit: true,
    rules: [
        {
            token: 'identifier.definition.herb',
            foreground: '61afef',
            fontStyle: 'bold',
        },
        // General Identifiers
        {token: 'identifier.cpp', foreground: 'abb2bf'},
        // Annotations (includes items in attribute list)
        {token: 'annotation.cpp', foreground: 'abb2bf'},
        // Keywords
        {token: 'keyword.auto.cpp', foreground: 'c678dd'},
        {token: 'keyword.break.cpp', foreground: 'c678dd'},
        {token: 'keyword.case.cpp', foreground: 'c678dd'},
        {token: 'keyword.catch.cpp', foreground: 'c678dd'},
        {token: 'keyword.class.cpp', foreground: 'c678dd'},
        {token: 'keyword.constexpr.cpp', foreground: 'c678dd'},
        {token: 'keyword.const.cpp', foreground: 'c678dd'},
        {token: 'keyword.continue.cpp', foreground: 'c678dd'},
        {token: 'keyword.default.cpp', foreground: 'c678dd'},
        {token: 'keyword.delete.cpp', foreground: 'c678dd'},
        {token: 'keyword.do.cpp', foreground: 'c678dd'},
        {token: 'keyword.else.cpp', foreground: 'c678dd'},
        {token: 'keyword.enum.cpp', foreground: 'c678dd'},
        {token: 'keyword.explicit.cpp', foreground: 'c678dd'},
        {token: 'keyword.export.cpp', foreground: 'c678dd'},
        {token: 'keyword.extern.cpp', foreground: 'c678dd'},
        {token: 'keyword.final.cpp', foreground: 'c678dd'},
        {token: 'keyword.for.cpp', foreground: 'c678dd'},
        {token: 'keyword.friend.cpp', foreground: 'c678dd'},
        {token: 'keyword.goto.cpp', foreground: 'c678dd'},
        {token: 'keyword.if.cpp', foreground: 'c678dd'},
        {token: 'keyword.inline.cpp', foreground: 'c678dd'},
        {token: 'keyword.mutable.cpp', foreground: 'c678dd'},
        {token: 'keyword.namespace.cpp', foreground: 'c678dd'},
        {token: 'keyword.new.cpp', foreground: 'c678dd'},
        {token: 'keyword.noexcept.cpp', foreground: 'c678dd'},
        {token: 'keyword.operator.cpp', foreground: 'c678dd'},
        {token: 'keyword.override.cpp', foreground: 'c678dd'},
        {token: 'keyword.private.cpp', foreground: 'c678dd'},
        {token: 'keyword.protected.cpp', foreground: 'c678dd'},
        {token: 'keyword.public.cpp', foreground: 'c678dd'},
        {token: 'keyword.return.cpp', foreground: 'c678dd'},
        {token: 'keyword.static.cpp', foreground: 'c678dd'},
        {token: 'keyword.struct.cpp', foreground: 'c678dd'},
        {token: 'keyword.switch.cpp', foreground: 'c678dd'},
        {token: 'keyword.template.cpp', foreground: 'c678dd'},
        {token: 'keyword.thread-local.cpp', foreground: 'c678dd'},
        {token: 'keyword.throw.cpp', foreground: 'c678dd'},
        {token: 'keyword.try.cpp', foreground: 'c678dd'},
        {token: 'keyword.typedef.cpp', foreground: 'c678dd'},
        {token: 'keyword.typename.cpp', foreground: 'c678dd'},
        {token: 'keyword.union.cpp', foreground: 'c678dd'},
        {token: 'keyword.unsigned.cpp', foreground: 'c678dd'},
        {token: 'keyword.using.cpp', foreground: 'c678dd'},
        {token: 'keyword.virtual.cpp', foreground: 'c678dd'},
        {token: 'keyword.while.cpp', foreground: 'c678dd'},
        {token: 'keyword.-asm.cpp', foreground: 'c678dd'},
        {token: 'keyword.and.cpp', foreground: 'c678dd'},
        {token: 'keyword.and-eq.cpp', foreground: 'c678dd'},
        {token: 'keyword.bitand.cpp', foreground: 'c678dd'},
        {token: 'keyword.bitor.cpp', foreground: 'c678dd'},
        {token: 'keyword.compl.cpp', foreground: 'c678dd'},
        {token: 'keyword.concept.cpp', foreground: 'c678dd'},
        {token: 'keyword.co-await.cpp', foreground: 'c678dd'},
        {token: 'keyword.co-return.cpp', foreground: 'c678dd'},
        {token: 'keyword.co-yield.cpp', foreground: 'c678dd'},
        {token: 'keyword.export.cpp', foreground: 'c678dd'},
        {token: 'keyword.import.cpp', foreground: 'c678dd'},
        {token: 'keyword.module.cpp', foreground: 'c678dd'},
        {token: 'keyword.not.cpp', foreground: 'c678dd'},
        {token: 'keyword.not-eq.cpp', foreground: 'c678dd'},
        {token: 'keyword.or.cpp', foreground: 'c678dd'},
        {token: 'keyword.or-eq.cpp', foreground: 'c678dd'},
        {token: 'keyword.requires.cpp', foreground: 'c678dd'},
        {token: 'keyword.xor.cpp', foreground: 'c678dd'},
        {token: 'keyword.xor-eq.cpp', foreground: 'c678dd'},
        // Function-like keywords
        {token: 'keyword.static-assert.cpp', foreground: '61afef'},
        {token: 'keyword.alignof.cpp', foreground: '61afef'},
        {token: 'keyword.typeid.cpp', foreground: '61afef'},
        {token: 'keyword.static-cast.cpp', foreground: '61afef'},
        {token: 'keyword.sizeof.cpp', foreground: '61afef'},
        {token: 'keyword.reinterpret-cast.cpp', foreground: '61afef'},
        {token: 'keyword.dynamic-cast.cpp', foreground: '61afef'},
        {token: 'keyword.decltype.cpp', foreground: '61afef'},
        {token: 'keyword.asm.cpp', foreground: '61afef'},
        // Constants and Literals
        {token: 'keyword.true.cpp', foreground: 'e5c07b'},
        {token: 'keyword.false.cpp', foreground: 'e5c07b'},
        {token: 'keyword.nullptr.cpp', foreground: 'e5c07b'},
        {token: 'number.cpp', foreground: 'e5c07b'},
        {token: 'number.hex.cpp', foreground: 'e5c07b'},
        {token: 'number.float.cpp', foreground: 'e5c07b'},
        {token: 'number.octal.cpp', foreground: 'e5c07b'},
        {token: 'number.binary.cpp', foreground: 'e5c07b'},
        {token: 'string.cpp', foreground: '98c379'},
        // Other
        {token: 'keyword.this.cpp', foreground: 'e06c75'},
        {token: 'keyword.directive.include.cpp', foreground: 'c678dd'},
        {token: 'keyword.directive.include.begin.cpp', foreground: '98c379'},
        {token: 'keyword.directive.include.end.cpp', foreground: '98c379'},
        {token: 'entity.name.type.cpp', foreground: 'e5c07b'},
        {token: 'string.escape.cpp', foreground: '56b6c2'},
        {token: 'string.include.identifier.cpp', foreground: '98c379'},
    ],
    colors: {
        'editor.background': '#282c34',
        'editor.foreground': '#abb2bf',
        // 'editor.lineHighlightBorder': '#e3a5e3',
        'editor.lineHighlightBackground': '#2c313c',
        'editor.wordHighlightBackground': '#484e5b',
        'editor.wordHighlightBorder': '#7f848e',
        'editor.wordHighlightStrongBackground': '#abb2bf26',
        'editor.wordHighlightStrongBorder': '#7f848e',
        'editorHoverWidget.background': '#21252b',
        'editorLineNumber.foreground': '#495162',
        'editor.selectionBackground': '#67769660',
        'editor.inactiveSelectionBackground': '#3a3f4b',
        'minimap.selectionHighlight': '#abb2bf',
    },
});

export class Themer {
    private currentTheme: Theme | null = null;

    constructor(
        private eventHub: GoldenLayout.EventEmitter,
        initialSettings: SiteSettings,
    ) {
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

    public getCurrentTheme() {
        return this.currentTheme;
    }

    private onSettingsChange(newSettings: SiteSettings) {
        if (options.mobileViewer && newSettings.theme === 'real-dark') {
            newSettings.theme = 'darkplus';
        }
        const newTheme = newSettings.theme && newSettings.theme in themes ? themes[newSettings.theme] : themes.default;
        if (!newTheme.monaco) newTheme.monaco = 'vs';
        this.setTheme(newTheme);
    }
}
