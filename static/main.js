// Copyright (c) 2016, Matt Godbolt
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
// POSSIBILITY OF SUCH DAMAGE
"use strict";

require("monaco-loader")().then(function () {

    require('bootstrap');
    require('bootstrap-slider');
    require('cookieconsent');

    var analytics = require('analytics');
    var sharing = require('sharing');
    var _ = require('underscore');
    var $ = require('jquery');
    var GoldenLayout = require('goldenlayout');
    var Components = require('components');
    var url = require('./url');
    var clipboard = require('clipboard');
    var Hub = require('hub');
    var Raven = require('raven-js');
    var settings = require('./settings');
    var local = require('./local');
    var Alert = require('./alert');
    var themer = require('./themes');

    //css
    require("bootstrap/dist/css/bootstrap.min.css");
    require("goldenlayout/src/css/goldenlayout-base.css");
    require("selectize/dist/css/selectize.bootstrap2.css");
    require("bootstrap-slider/dist/css/bootstrap-slider.css");
    require("./colours.css");
    require("./explorer.css");

    function setupSettings(hub) {
        var eventHub = hub.layout.eventHub;
        var defaultSettings = {
            defaultLanguage: hub.subdomainLangId ? hub.subdomainLangId : undefined
        };
        var currentSettings = JSON.parse(local.get('settings', null)) || defaultSettings;

        function onChange(settings) {
            currentSettings = settings;
            local.set('settings', JSON.stringify(settings));
            eventHub.emit('settingsChange', settings);
        }

        new themer.Themer(eventHub, currentSettings);

        eventHub.on('requestSettings', function () {
            eventHub.emit('settingsChange', currentSettings);
        });

        var setSettings = settings($('#settings'), currentSettings, onChange, hub.subdomainLangId);
        eventHub.on('modifySettings', function (newSettings) {
            setSettings(_.extend(currentSettings, newSettings));
        });
    }

    function setupButtons(options) {
        var alertSystem = new Alert();

        var cookiemodal = null;
        var getCookieTitle = function () {
            return 'Cookies & related technologies policy<br><p>Current consent status: <span style="color:' +
                (cookieconsent.hasConsented() ? 'green':'red') +  '">' +
                (cookieconsent.hasConsented() ? 'Granted' : 'Denied') + '</span></p>';
        };
        var openCookiePolicy = function () {
            cookiemodal = alertSystem.ask(getCookieTitle(), $(require('./cookies.html')), {
                yes: _.bind(cookieconsent.doConsent, cookieconsent),
                yesHtml: 'Consent',
                no: _.bind(cookieconsent.doOppose, cookieconsent),
                noHtml: 'Do NOT consent',
                onClose: function () {
                    cookiemodal = null;
                }
            });
        };
        window.cookieconsent.status.allow = options.hashs.cookies;
        window.cookieconsent.status.dismiss = window.cookieconsent.status.deny;
        var cookieconsent = window.cookieconsent.initialise({
            palette: {
                popup: {
                    background: "#eaf7f7",
                    text: "#5c7291"
                },
                button: {
                    background: "#56cbdb",
                    text: "#ffffff"
                }
            },
            theme: "edgeless",
            type: "opt-in",
            // We handle the revoking elsewhere
            revokable: false,
            forceRevokable: false,
            content: {
                // We use onClick handlers to open the popup without reloading
                link: 'Check our cookie policy',
                href: "#",
                message: "Compiler Explorer uses cookies & related technologies.",
                dismiss: 'Do NOT allow nonessential cookies'
            },
            elements: {
                messagelink: '<span id="cookieconsent:desc" class="cc-message">' +
                '{{message}} <a aria-label="learn more about cookies" tabindex="0" class="cc-link cookies" ' +
                'href="{{href}}">{{link}}</a></span>',
                link: '<a aria-label="learn more about cookies" tabindex="0" class="cc-link cookies" href="{{href}}">' +
                '{{link}}</a>'
            },
            onStatusChange: function () {
                if (cookiemodal) {
                    cookiemodal.find('.modal-title').html(getCookieTitle());
                }
                if (this.hasConsented()) {
                    // enable cookies
                }
            },
            onInitialise: function () {
                if (this.hasConsented()) {
                    // enable cookies
                }
            }
        });

        $('#ui-reset').click(function () {
            local.remove('gl');
            window.location.reload();
        });
        $('#thanks-to').click(function () {
            alertSystem.alert("Special thanks to", $(require('./thanks.html')));
        });
        $('#changes').click(function () {
            alertSystem.alert("Changelog", $(require('./changelog.html')));
        });

        $('#cookies').click(openCookiePolicy);
        $('.cookies').click(openCookiePolicy);
        $('#privacy').click(function () {
            alertSystem.alert("Privacy policy", $(require('./privacy.html')));
        });
    }

    function start() {
        analytics.initialise();

        var options = require('options');

        var subdomainPart = window.location.hostname.split('.')[0];
        var langBySubdomain = _.find(options.languages, function (lang) {
            return lang.id === subdomainPart || lang.alias.indexOf(subdomainPart) >= 0;
        });
        var subLangId = langBySubdomain ? langBySubdomain.id : undefined;

        var defaultConfig = {
            settings: {showPopoutIcon: false},
            content: [{
                type: 'row',
                content: [
                    Components.getEditor(1, subLangId),
                    Components.getCompiler(1, subLangId)
                ]
            }]
        };

        $(window).bind('hashchange', function () {
            // punt on hash events and just reload the page if there's a hash
            if (window.location.hash.substr(1))
                window.location.reload();
        });

        var hashPart = null;
        // Which buttons act as a linkable popup
        var linkablePopups = ['#thanks-to', '#changes', '#cookies', '#setting', '#privacy'];
        if (linkablePopups.indexOf(window.location.hash) > -1) {
            hashPart = window.location.hash;
            window.location.hash = "";
        }

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
            config = _.extend(defaultConfig, {settings: {
                showMaximiseIcon: false,
                showCloseIcon: false,
                hasHeaders: false
            }}, sharing.configFromEmbedded(window.location.hash.substr(1)));
        }

        var root = $("#root");

        var layout;
        var hub;
        try {
            layout = new GoldenLayout(config, root);
            hub = new Hub(layout, subLangId);
        } catch (e) {
            Raven.captureException(e);
            layout = new GoldenLayout(defaultConfig, root);
            hub = new Hub(layout, subLangId);
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
            var height = $(window).height() - (root.position().top || 0);
            root.height(height);
            layout.updateSize();
        }

        $(window).resize(sizeRoot);
        sizeRoot();

        new clipboard('.btn.clippy');

        setupSettings(hub);

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

        setupButtons(options);

        if (hashPart) {
            var element = $(hashPart);
            if (element) element.click();
        }
    }

    $(start);
});

