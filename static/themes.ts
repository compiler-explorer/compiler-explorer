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
import {assert} from './assert.js';
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
    ],
    colors: {},
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
        {token: 'keyword.if.cpp', foreground: 'c586c0'},
        {token: 'keyword.else.cpp', foreground: 'c586c0'},
        {token: 'keyword.while.cpp', foreground: 'c586c0'},
        {token: 'keyword.for.cpp', foreground: 'c586c0'},
        {token: 'keyword.return.cpp', foreground: 'c586c0'},
        {token: 'keyword.break.cpp', foreground: 'c586c0'},
        {token: 'keyword.continue.cpp', foreground: 'c586c0'},
        {token: 'keyword.goto.cpp', foreground: 'c586c0'},
        {token: 'keyword.directive.cpp', foreground: 'c586c0'},
        {token: 'keyword.directive.include.cpp', foreground: 'c586c0'},
        {token: 'keyword.directive.include.begin.cpp', foreground: 'ce9178'},
        {token: 'keyword.directive.include.end.cpp', foreground: 'ce9178'},
        {token: 'keyword.new.cpp', foreground: 'c586c0'},
        {token: 'keyword.using.cpp', foreground: 'c586c0'},
        {token: 'string.escape.cpp', foreground: 'd7ba7d'},
    ],
    colors: {},
});

editor.defineTheme('ce-pink', {
    base: 'vs-dark',
    inherit: true,
    rules: [
        ...(() => {
            // based on https://github.com/nopjmp/vscode-fairyfloss/blob/master/themes/fairyfloss-color-theme.json
            const base: {
                name?: string;
                scope: string | string[];
                settings: {foreground?: string; fontStyle?: string};
            }[] = [
                {
                    name: 'Comment',
                    scope: 'comment',
                    settings: {
                        foreground: '#E6C000',
                    },
                },
                {
                    name: 'String',
                    scope: 'string',
                    settings: {
                        foreground: '#FFEA00',
                    },
                },
                {
                    name: 'Number',
                    scope: 'constant.numeric',
                    settings: {
                        foreground: '#C5A3FF',
                    },
                },
                {
                    name: 'Built-in constant',
                    scope: 'constant.language',
                    settings: {
                        foreground: '#C5A3FF',
                    },
                },
                {
                    name: 'User-defined constant',
                    scope: ['constant.character', 'constant.other'],
                    settings: {
                        foreground: '#C5A3FF',
                    },
                },
                {
                    name: 'Variable',
                    scope: 'variable',
                    settings: {
                        fontStyle: '',
                    },
                },
                {
                    name: 'Keyword',
                    scope: 'keyword',
                    settings: {
                        foreground: '#FFB8D1',
                    },
                },
                {
                    name: 'Storage',
                    scope: 'storage',
                    settings: {
                        fontStyle: '',
                        foreground: '#FFB8D1',
                    },
                },
                {
                    name: 'Storage type',
                    scope: 'storage.type',
                    settings: {
                        fontStyle: 'italic',
                        foreground: '#C2FFDF',
                    },
                },
                {
                    name: 'Class name',
                    scope: 'entity.name.class',
                    settings: {
                        fontStyle: 'underline',
                        foreground: '#FFF352',
                    },
                },
                {
                    name: 'This',
                    scope: ['variable.language.this'],
                    settings: {
                        foreground: '#d5c6f0',
                    },
                },
                {
                    name: 'Inherited class',
                    scope: 'entity.other.inherited-class',
                    settings: {
                        fontStyle: 'italic underline',
                        foreground: '#FFF352',
                    },
                },
                {
                    name: 'Function name',
                    scope: 'entity.name.function',
                    settings: {
                        fontStyle: '',
                        foreground: '#FFF352',
                    },
                },
                {
                    name: 'Function argument',
                    scope: 'variable.parameter',
                    settings: {
                        fontStyle: 'italic',
                        foreground: '#FF857F',
                    },
                },
                {
                    name: 'Tag name',
                    scope: 'entity.name.tag',
                    settings: {
                        fontStyle: '',
                        foreground: '#FFB8D1',
                    },
                },
                {
                    name: 'Tag attribute',
                    scope: 'entity.other.attribute-name',
                    settings: {
                        fontStyle: '',
                        foreground: '#FFF352',
                    },
                },
                {
                    name: 'Library function',
                    scope: 'support.function',
                    settings: {
                        fontStyle: '',
                        foreground: '#C2FFDF',
                    },
                },
                {
                    name: 'Library constant',
                    scope: 'support.constant',
                    settings: {
                        fontStyle: '',
                        foreground: '#C2FFDF',
                    },
                },
                {
                    name: 'Library class/type',
                    scope: ['support.type', 'support.class'],
                    settings: {
                        fontStyle: 'italic',
                        foreground: '#C2FFDF',
                    },
                },
                {
                    name: 'Library variable',
                    scope: 'support.other.variable',
                    settings: {
                        fontStyle: '',
                    },
                },
                {
                    name: 'Invalid',
                    scope: 'invalid',
                    settings: {
                        fontStyle: '',
                        foreground: '#ffb0b0',
                    },
                },
                {
                    name: 'Invalid deprecated',
                    scope: 'invalid.deprecated',
                    settings: {
                        foreground: '#dfdfcb',
                    },
                },
                {
                    name: 'diff: header',
                    scope: ['meta.diff', 'meta.diff.header'],
                    settings: {
                        fontStyle: 'italic',
                        foreground: '#FFF352',
                    },
                },
                {
                    name: 'diff: deleted',
                    scope: 'markup.deleted',
                    settings: {
                        foreground: '#fa736e',
                    },
                },
                {
                    name: 'diff: changed',
                    scope: 'markup.changed',
                    settings: {
                        foreground: '#fa9d99',
                    },
                },
                {
                    name: 'diff: inserted',
                    scope: 'markup.inserted',
                    settings: {
                        foreground: '#afecad',
                    },
                },
                {
                    name: 'Markup Quote',
                    scope: 'markup.quote',
                    settings: {
                        foreground: '#FF857F',
                    },
                },
                {
                    name: 'Markup Styling Bold',
                    scope: 'markup.bold',
                    settings: {
                        fontStyle: 'bold',
                        foreground: '#C2FFDF',
                    },
                },
                {
                    name: 'Markup Styling Italic',
                    scope: 'markup.italic',
                    settings: {
                        fontStyle: 'italic',
                        foreground: '#C2FFDF',
                    },
                },
                {
                    name: 'Markup Inline',
                    scope: 'markup.inline.raw',
                    settings: {
                        foreground: '#FFF352',
                    },
                },
                {
                    name: 'Markup Header',
                    scope: 'markup.heading.markdown',
                    settings: {
                        fontStyle: 'bold',
                        foreground: '#ddbb88',
                    },
                },
                {
                    name: 'Source PHP Embedded Begin/End',
                    scope: [
                        'text.html.php punctuation.section.embedded.begin.php',
                        'text.html.php punctuation.section.embedded.end.php',
                    ],
                    settings: {
                        foreground: '#d5c6f0',
                    },
                },
                {
                    name: 'Source Java Storage Modifier Import',
                    scope: ['source.java storage.modifier.import', 'source.java storage.modifier.package'],
                    settings: {
                        fontStyle: '',
                        foreground: '#dda0b4',
                    },
                },
                {
                    name: 'Javadoc keywords',
                    scope: 'source.java keyword.other.documentation.javadoc.java',
                    settings: {
                        foreground: '#9966b8',
                    },
                },
                {
                    name: 'Clojure locals',
                    scope: 'source.clojure meta.symbol',
                    settings: {
                        foreground: '#beb0cc',
                    },
                },
                {
                    name: 'Clojure namespace',
                    scope: ['source.clojure entity.global', 'source.clojure meta.definition.global'],
                    settings: {
                        foreground: '#df7e9f',
                    },
                },
                {
                    name: 'Clojure keyword',
                    scope: 'source.clojure constant.keyword',
                    settings: {
                        foreground: '#c19fd8',
                    },
                },
                {
                    name: 'Clojure namespace',
                    scope: ['source.clojure meta.symbol meta.expression', 'source.clojure meta.vector'],
                    settings: {
                        foreground: '#df7e9f',
                    },
                },
                {
                    name: 'Python cls, self',
                    scope: [
                        'source.python variable.language.special.self.python',
                        'source.python variable.language.special.cls.python',
                    ],
                    settings: {
                        foreground: '#FF857F',
                    },
                },
                {
                    name: 'Python docstring',
                    scope: ['source.python string.quoted.docstring.multi.python'],
                    settings: {
                        foreground: '#E6C000',
                    },
                },
                {
                    name: 'Python function name that matches builtins',
                    scope: ['source.python meta.function.python support.function.builtin.python'],
                    settings: {
                        foreground: '#FFF352',
                    },
                },
                {
                    name: 'Python logical operator',
                    scope: [
                        'source.python keyword.operator.logical.python',
                        'source.python keyword.operator.comparison.python',
                    ],
                    settings: {
                        foreground: '#C5A3FF',
                    },
                },
                {
                    name: 'Python decorators',
                    scope: [
                        'source.python meta.function.decorator.python',
                        'source.python entity.name.function.decorator.python',
                    ],
                    settings: {
                        fontStyle: 'italic',
                        foreground: '#C2FFDF',
                    },
                },
                {
                    name: 'Python function parameter annotation',
                    scope: [
                        'source.python meta.function.parameters.python',
                        'source.python keyword.operator.unpacking.parameter.python',
                    ],
                    settings: {
                        fontStyle: 'italic',
                        foreground: '#C2FFDF',
                    },
                },
                // Infra specific changes from ardenasasvc
                {
                    scope: ['string.unquoted.plain.out.yaml'],
                    settings: {
                        foreground: '#dec0e2',
                    },
                },
                {
                    scope: ['string.quoted.double.yaml'],
                    settings: {
                        foreground: '#C2FFDF',
                    },
                },
                {
                    scope: ['comment.line.number-sign.yaml'],
                    settings: {
                        foreground: '#a186cf',
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
        {
            token: 'identifier.definition.herb',
            foreground: '7c9c7c',
            fontStyle: 'bold',
        },
        {token: 'keyword.if.cpp', foreground: 'c586c0'},
        {token: 'keyword.else.cpp', foreground: 'c586c0'},
        {token: 'keyword.while.cpp', foreground: 'c586c0'},
        {token: 'keyword.for.cpp', foreground: 'c586c0'},
        {token: 'keyword.return.cpp', foreground: 'c586c0'},
        {token: 'keyword.break.cpp', foreground: 'c586c0'},
        {token: 'keyword.continue.cpp', foreground: 'c586c0'},
        {token: 'keyword.goto.cpp', foreground: 'c586c0'},
        {token: 'keyword.directive.cpp', foreground: 'c586c0'},
        {token: 'keyword.directive.include.cpp', foreground: 'c586c0'},
        {token: 'keyword.directive.include.begin.cpp', foreground: 'ce9178'},
        {token: 'keyword.directive.include.end.cpp', foreground: 'ce9178'},
        {token: 'keyword.new.cpp', foreground: 'c586c0'},
        {token: 'keyword.using.cpp', foreground: 'c586c0'},
        {token: 'string.escape.cpp', foreground: 'd7ba7d'},
    ],
    colors: {
        'editor.background': '#544C6E',
        'editor.lineHighlightBorder': '#5C5378',
        'editor.wordHighlightTextBackground': '#5C5378',
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
