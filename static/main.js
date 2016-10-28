// Copyright (c) 2012-2016, Matt Godbolt
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
        'es6-promise': 'ext/es6-promise/es6-promise'
    },
    packages: [{
        name: "codemirror",
        location: "ext/codemirror",
        main: "lib/codemirror"
    }],
    shim: {
        underscore: {exports: '_'},
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
    var compiler = require('compiler');
    var editor = require('editor');
    var url = require('url');
    var clipboard = require('clipboard');
    var Hub = require('hub');
    var shortenURL = require('urlshorten-google');
    var Raven = require('raven-js');

    function contentFromEmbedded(embeddedUrl) {
        var params = url.unrisonify(embeddedUrl);
        var filters = _.chain((params.filters || "").split(','))
            .map(function (o) {
                return [o, true];
            })
            .object()
            .value();
        return [
            {
                type: 'row',
                content: [
                    editor.getComponentWith(1, params.source, filters),
                    compiler.getComponentWith(1, filters, params.options, params.compiler)
                ]
            }
        ];
    }

    function getItemsByComponent(layout, component) {
        return layout.root.getItemsByFilter(function (o) {
            return o.type === "component" && o.componentName === component;
        });
    }

    function getEmbeddedUrl(layout) {
        window.layout = layout;
        var source = "";
        var filters = {};
        var compilerName = "";
        var options = "";
        _.each(getItemsByComponent(layout, editor.getComponent().componentName),
            function (editor) {
                var state = editor.config.componentState;
                source = state.source;
                filters = _.extend(filters, state.options);
            });
        _.each(getItemsByComponent(layout, compiler.getComponent().componentName),
            function (compiler) {
                var state = compiler.config.componentState;
                compilerName = state.compiler;
                options = state.options;
                filters = _.extend(filters, state.filters);
            });
        if (!filters.compileOnChange)
            filters.readOnly = true;
        return window.location.origin + '/e#' + url.risonify({
                filters: _.keys(filters).join(","),
                source: source,
                compiler: compilerName,
                options: options
            });
    }

    function start() {
        analytics.initialise();
        sharing.initialise();

        var options = require('options');
        $('.language-name').text(options.language);

        var safeLang = options.language.toLowerCase().replace(/[^a-z_]+/g, '');
        var defaultSrc = $('.template .lang.' + safeLang).text().trim();
        var defaultConfig = {
            settings: {showPopoutIcon: false},
            content: [{type: 'row', content: [editor.getComponent(1), compiler.getComponent(1)]}]
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
                    content: contentFromEmbedded(window.location.hash.substr(1))
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

        function initPopover(getLink, provider) {
            var html = $('.template .urls').html();

            getLink.popover({
                container: 'body',
                content: html,
                html: true,
                placement: 'bottom',
                trigger: 'manual'
            }).click(function () {
                getLink.popover('show');
            }).on('inserted.bs.popover', function () {
                provider(function (url) {
                    $(".permalink:visible").val(url);
                });
            });

            // Dismiss the popover on escape.
            $(document).on('keyup.editable', function (e) {
                if (e.which === 27) {
                    getLink.popover("hide");
                }
            });

            // Dismiss on any click that isn't either on the opening element, or inside
            // the popover.
            $(document).on('click.editable', function (e) {
                var target = $(e.target);
                if (!target.is(getLink) && target.closest('.popover').length === 0)
                    getLink.popover("hide");
            });
        }

        function permalink() {
            var config = layout.toConfig();
            return window.location.href.split('#')[0] + '#' + url.serialiseState(config);
        }

        initPopover($("#get-full-link"), function (done) {
            done(permalink);
        });
        initPopover($("#get-short-link"), function (done) {
            shortenURL(permalink(), done);
        });
        initPopover($("#get-embed-link"), function (done) {
            done(function () {
                return '<iframe width="800px" height="200px" src="' +
                    getEmbeddedUrl(layout) + '"></iframe>';
            });
        });
    }

    $(start);
});
