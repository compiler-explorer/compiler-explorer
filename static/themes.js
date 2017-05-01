// Copyright (c) 2012-2017, Matt Godbolt & Rubén Rincón
//
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

define(function (require) {
    "use strict";
    var $ = require('jquery');
    var themes = {
        default: {
            path: "./themes/explorer-default.css",
            id: "default",
            name: "Default",
            monaco: "vs" // Optional field
        },
        dark: {
            path: "./themes/dark/theme-dark.css",
            id: "dark",
            name: "Dark",
            monaco: "vs-dark"
        }
    };

    function Themer(eventHub, initialSettings) {
        this.currentTheme = null;
        this.eventHub = eventHub;
        this.root = root;

        this.setTheme = function (theme) {
            if (this.currentTheme === theme) return;
            $.get(require.toUrl(theme.path), function (cssData) {
                $('#theme').html(cssData);
                eventHub.emit('themeChange', theme);
            });
            this.currentTheme = theme;
        };

        this.onSettingsChange = function (newSettings) {
            var newTheme = themes[newSettings.theme] || themes.default;
            if (!newTheme.monaco)
                newTheme.monaco = "vs";
            this.setTheme(newTheme);
        };
        this.onSettingsChange(initialSettings);

        this.eventHub.on('settingsChange', this.onSettingsChange, this);

        this.eventHub.on('requestTheme', function () {
            this.eventHub.emit('themeChange', this.currentTheme);
        }, this);
    }

    return {
        themes: themes,
        Themer: Themer
    };
});
