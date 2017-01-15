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
        'lru-cache': 'ext/lru-cache/lib/lru-cache'
    },
    packages: [{
        name: "codemirror",
        location: "ext/codemirror",
        main: "lib/codemirror"
    }],
    shim: {
        underscore: {exports: '_'},
        'lru-cache': {exports: 'LRUCache'},
        bootstrap: ['jquery']
    }
});

define(function (require) {
    "use strict";
    require('bootstrap');
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
    var Alert = require('alert');

    function start() {
        analytics.initialise();
        sharing.initialise();

        var options = require('options');
        $('.language-name').text(options.language);
        var alert = new Alert();

        var safeLang = options.language.toLowerCase().replace(/[^a-z_]+/g, '');
        var defaultSrc = $('.template .lang.' + safeLang).text().trim();
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
            var serializedState = window.location.hash.substr(1);
            if (serializedState) {
                try {
                    config = url.deserialiseState(serializedState);
                } catch (exception) {
                    alert.alert("Unable to parse URL",
                        "<div>Compiler Explorer was unable to parse the URL hash. " +
                        "Please check it and try again.</div>" +
                        "<div class='url-parse-info'>" + exception + "</div>");
                }
            }
            if (config) {
                // replace anything in the default config with that from the hash
                config = _.extend(defaultConfig, config);
            }

            if (!config) {
                var savedState = null;
                try {
                    savedState = window.localStorage.getItem('gl');
                } catch (e) {
                    // Some browsers in secure modes can throw exceptions here...
                }
                config = savedState !== null ? JSON.parse(savedState) : defaultConfig;
            }
        } else {
            config = _.extend(defaultConfig,
                {
                    settings: {
                        showMaximiseIcon: false,
                        showCloseIcon: false,
                        hasHeaders: false
                    },
                    content: sharing.contentFromEmbedded(window.location.hash.substr(1))
                });
        }

        var root = $("#root");

        var layout;
        try {
            layout = new GoldenLayout(config, root);
            new Hub(layout, defaultSrc);
        } catch (e) {
            Raven.captureException(e);
            layout = new GoldenLayout(defaultConfig, root);
            new Hub(layout, defaultSrc);
        }
        layout.on('stateChanged', function () {
            var config = layout.toConfig();
            // Only preserve state in localStorage in non-embedded mode.
            if (!options.embedded) {
                var state = JSON.stringify(config);
                try {
                    window.localStorage.setItem('gl', state);
                } catch (e) {
                    // Some browsers in secure modes may throw
                }
            } else {
                $('a.link').attr('href', '/#' + url.serialiseState(config));
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

        sharing.initShareButton($('#share'), layout);
        $('#ui-reset').click(function () {
            window.localStorage.removeItem('gl');
            window.location.reload();
        });
    }

    $(start);
});
