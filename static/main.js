// Copyright (c) 2012-2017, Matt Godbolt
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

require.config({
    paths: {
        bootstrap: 'ext/bootstrap/dist/js/bootstrap',
        jquery: 'ext/jquery/dist/jquery',
        underscore: 'ext/underscore/underscore',
        goldenlayout: 'ext/golden-layout/dist/goldenlayout',
        selectize: 'ext/selectize/dist/js/selectize',
        sifter: 'ext/sifter/sifter',
        microplugin: 'ext/microplugin/src/microplugin',
        events: 'ext/eventEmitter/EventEmitter',
        lzstring: 'ext/lz-string/libs/lz-string',
        clipboard: 'ext/clipboard/dist/clipboard',
        'raven-js': 'ext/raven-js/dist/raven',
        'es6-promise': 'ext/es6-promise/es6-promise',
        'lru-cache': 'ext/lru-cache/lib/lru-cache',
        'vs': "ext/monaco-editor/min/vs",
        'bootstrap-slider': 'ext/seiyria-bootstrap-slider/dist/bootstrap-slider'
    },
    shim: {
        underscore: {exports: '_'},
        'lru-cache': {exports: 'LRUCache'},
        bootstrap: ['jquery'],
        'bootstrap-slider': ['bootstrap']
    }
});

define(function (require) {
    "use strict";
    require('bootstrap');
    require('bootstrap-slider');
    var analytics = require('analytics');
    var sharing = require('sharing');
    var _ = require('underscore');
    var $ = require('jquery');
    var GoldenLayout = require('goldenlayout');
    var Components = require('components');
    var url = require('url');
    var clipboard = require('clipboard');
    var Hub = require('hub');
    var Raven = require('raven-js');
    var settings = require('./settings');
    var local = require('./local');
    var Alert = require('./alert');
    var themer = require('./themes');

    function setupSettings(eventHub) {
        var currentSettings = JSON.parse(local.get('settings', '{}'));

        function onChange(settings) {
            currentSettings = settings;
            local.set('settings', JSON.stringify(settings));
            eventHub.emit('settingsChange', settings);
        }

        new themer.Themer(eventHub);
        themer.setupThemes();
        eventHub.on('requestSettings', function () {
            eventHub.emit('settingsChange', currentSettings);
        });

        var setSettings = settings($('#settings'), currentSettings, onChange);
        eventHub.on('modifySettings', function (newSettings) {
            setSettings(_.extend(currentSettings, newSettings));
        });
    }

    function start() {
        analytics.initialise();

        var options = require('options');

        var defaultSrc = $('.template .lang').text().trim();
        var defaultConfig = {
            settings: {showPopoutIcon: false},
            content: [{type: 'row', content: [Components.getEditor(1), Components.getCompiler(1)]}]
        };

        $(window).bind('hashchange', function () {
            // punt on hash events and just reload the page if there's a hash
            if (window.location.hash.substr(1))
                window.location.reload();
        });

        var config;
        if (!options.embedded) {
            config = url.deserialiseState(window.location.hash.substr(1));
            if (config) {
                // replace anything in the default config with that from the hash
                config = _.extend(defaultConfig, config);
            }

            if (!config) {
                var savedState = local.get('gl', null);
                config = savedState !== null ? JSON.parse(savedState) : defaultConfig;
            }
        } else {
            config = _.extend(defaultConfig,
                {
                    settings: {
                        showMaximiseIcon: false,
                        showCloseIcon: false,
                        hasHeaders: false
                    }
                },
                sharing.configFromEmbedded(window.location.hash.substr(1)));
        }

        var root = $("#root");

        var layout;
        var hub;
        try {
            layout = new GoldenLayout(config, root);
            hub = new Hub(layout, defaultSrc);
        } catch (e) {
            Raven.captureException(e);
            layout = new GoldenLayout(defaultConfig, root);
            hub = new Hub(layout, defaultSrc);
        }
        layout.on('stateChanged', function () {
            var config = layout.toConfig();
            // Only preserve state in localStorage in non-embedded mode.
            if (!options.embedded) {
                local.set('gl', JSON.stringify(config));
            } else {
                var strippedToLast = window.location.pathname;
                strippedToLast = strippedToLast.substr(0,
                    strippedToLast.lastIndexOf('/') + 1);
                $('a.link').attr('href', strippedToLast + '#' + url.serialiseState(config));
            }
        });

        function sizeRoot() {
            var height = $(window).height() - root.position().top;
            root.height(height);
            layout.updateSize();
        }

        $(window).resize(sizeRoot);
        sizeRoot();

        new clipboard('.btn.clippy');

        setupSettings(layout.eventHub);

        sharing.initShareButton($('#share'), layout);

        function setupAdd(thing, func) {
            layout.createDragSource(thing, func);
            thing.click(function () {
                hub.addAtRoot(func());
            });
        }

        setupAdd($('#add-diff'), function () {
            return Components.getDiff();
        });
        setupAdd($('#add-editor'), function () {
            return Components.getEditor();
        });

        $('#ui-reset').click(function () {
            local.remove('gl');
            window.location.reload();
        });
        $('#thanks-to').click(function () {
            $.get('thanks.html', function (result) {
                new Alert().alert("Special thanks to", $(result));
            });
        });
    }

    $(start);
});
