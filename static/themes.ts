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

import { editor } from 'monaco-editor';
import { SiteSettings } from './settings';

export type Themes = 'default' | 'dark' | 'darkplus';

export interface Theme {
    path: string;
    id: string;
    name: string;
    mainColor: string;
    monaco: string;
}

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
};

editor.defineTheme('ce', {
    base: 'vs',
    inherit: true,
    rules: [
        {token: 'identifier.definition.cppx-blue', foreground: '008a00', fontStyle: 'bold'},
    ],
    colors: {},
});

editor.defineTheme('ce-dark', {
    base: 'vs-dark',
    inherit: true,
    rules: [
        {token: 'identifier.definition.cppx-blue', foreground: '7c9c7c', fontStyle: 'bold'},
    ],
    colors: {},
});

editor.defineTheme('ce-dark-plus', {
    base: 'vs-dark',
    inherit: true,
    rules: [
        {token: 'identifier.definition.cppx-blue', foreground: '7c9c7c', fontStyle: 'bold'},
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

export class Themer {
    private currentTheme: Theme | null = null;

    constructor(private eventHub: any, initialSettings: SiteSettings) {
        this.onSettingsChange(initialSettings);

        this.eventHub.on('settingsChange', this.onSettingsChange, this);

        this.eventHub.on('requestTheme', () => {
            this.eventHub.emit('themeChange', this.currentTheme);
        }, this);
    }

    public setTheme(theme: Theme) {
        if (this.currentTheme === theme) return;
        $('html').attr('data-theme', theme.path);
        $('#meta-theme').prop('content', theme.mainColor);
        editor.setTheme(theme.monaco);
        this.eventHub.emit('resize');
        this.currentTheme = theme;
    }

    private onSettingsChange(newSettings: SiteSettings) {
        const newTheme = newSettings.theme in themes ? themes[newSettings.theme] : themes.default;
        if (!newTheme.monaco)
            newTheme.monaco = 'vs';
        this.setTheme(newTheme);
    }
}
