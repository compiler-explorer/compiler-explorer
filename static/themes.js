// Copyright (c) 2017, Matt Godbolt & Rubén Rincón
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

'use strict';
var $ = require('jquery');
var monaco = require('monaco-editor');

var themes = {
    default: {
        path: 'default',
        id: 'default',
        name: 'Light',
        'main-color': '#f2f2f2',
        monaco: 'ce',
    },
    dark: {
        path: 'dark',
        id: 'dark',
        name: 'Dark',
        'main-color': '#333333',
        monaco: 'ce-dark',
    },
};

monaco.editor.defineTheme('ce', {
    base: 'vs',
    inherit: true,
    rules: [
        { token: 'identifier.definition.cppx-blue', foreground: '004400', fontStyle: 'bold' },
    ],
});

monaco.editor.defineTheme('ce-dark', {
    base: 'vs-dark',
    inherit: true,
    rules: [
        { token: 'identifier.definition.cppx-blue', foreground: 'bbccbb', fontStyle: 'bold' },
    ],
});

function Themer(eventHub, initialSettings) {
    this.currentTheme = null;
    this.eventHub = eventHub;

    this.setTheme = function (theme) {
        if (this.currentTheme === theme) return;
        $('html').attr('data-theme', theme.path);
        $('#meta-theme').prop('content', theme['main-color']);
        monaco.editor.setTheme(theme.monaco);
        this.eventHub.emit('resize');
        this.currentTheme = theme;
    };

    this.onSettingsChange = function (newSettings) {
        var newTheme = themes[newSettings.theme] || themes.default;
        if (!newTheme.monaco)
            newTheme.monaco = 'vs';
        this.setTheme(newTheme);

        // This line is used to set thet codelens font
        // Official support using the IEditorOptions.codeLensFontFamily property has landed in vscode
        // It should be removed once a downstream release of monaco is cut
        document.querySelector(':root').style.setProperty('--user-selected-font-stack', newSettings.editorsFFont);
    };
    this.onSettingsChange(initialSettings);

    this.eventHub.on('settingsChange', this.onSettingsChange, this);

    this.eventHub.on('requestTheme', function () {
        this.eventHub.emit('themeChange', this.currentTheme);
    }, this);
}

module.exports = {
    themes: themes,
    Themer: Themer,
};
