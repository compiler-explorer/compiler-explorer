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

    var analytics = require('./analytics');
    var sharing = require('./sharing');
    var _ = require('underscore');
    var $ = require('jquery');
    var GoldenLayout = require('goldenlayout');
    var Components = require('./components');
    var url = require('./url');
    var clipboard = require('clipboard');
    var Hub = require('./hub');
    var Raven = require('raven-js');
    var settings = require('./settings');
    var local = require('./local');
    var Alert = require('./alert');
    var themer = require('./themes');
    var motd = require('./motd');
    //css
    require("bootstrap/dist/css/bootstrap.min.css");
    require("goldenlayout/src/css/goldenlayout-base.css");
    require("selectize/dist/css/selectize.bootstrap2.css");
    require("bootstrap-slider/dist/css/bootstrap-slider.css");
    require("./colours.css");
    require("./explorer.css");

    // Check to see if the current unload is a UI reset.
    // Forgive me the global usage here
    var hasUIBeenReset = false;

    function setupSettings(hub) {
        var eventHub = hub.layout.eventHub;
        var defaultSettings = {
            defaultLanguage: hub.subdomainLangId ? hub.subdomainLangId : undefined
        };
        var currentSettings = JSON.parse(local.get('settings', null)) || defaultSettings;

        function onChange(newSettings) {
            if (currentSettings.theme !== newSettings.theme) {
                analytics.proxy('send', {
                    hitType: 'event',
                    eventCategory: 'ThemeChange',
                    eventAction: newSettings.theme
                });
            }
            if (currentSettings.colourScheme !== newSettings.colourScheme) {
                analytics.proxy('send', {
                    hitType: 'event',
                    eventCategory: 'ColourSchemeChange',
                    eventAction: newSettings.colourScheme
                });
            }
            currentSettings = newSettings;
            local.set('settings', JSON.stringify(newSettings));
            eventHub.emit('settingsChange', newSettings);
        }

        new themer.Themer(eventHub, currentSettings);

        eventHub.on('requestSettings', function () {
            eventHub.emit('settingsChange', currentSettings);
        });

        var setSettings = settings($('#settings'), currentSettings, onChange, hub.subdomainLangId);
        eventHub.on('modifySettings', function (newSettings) {
            setSettings(_.extend(currentSettings, newSettings));
        });
        return currentSettings;
    }

    function setupButtons(options) {
        var alertSystem = new Alert();

        if (options.policies.cookies.enabled) {
            var cookiemodal = null;

            var getCookieTitle = function () {
                return 'Cookies & related technologies policy<br><p>Current consent status: <span style="color:' +
                    (cookieconsent.hasConsented() ? 'green' : 'red') + '">' +
                    (cookieconsent.hasConsented() ? 'Granted' : 'Denied') + '</span></p>';
            };
            var openCookiePolicy = function () {
                cookiemodal = alertSystem.ask(getCookieTitle(), $(require('./policies/cookies.html')), {
                    yes: function () {
                        cookieconsent.doConsent.apply(cookieconsent);
                        analytics.toggle(true);
                    },
                    yesHtml: 'Consent',
                    no: function () {
                        analytics.toggle(false);
                        cookieconsent.doOppose.apply(cookieconsent);
                    },
                    noHtml: 'Do NOT consent',
                    onClose: function () {
                        // Remove modal ref so we don't try to update its title if the consent changes
                        cookiemodal = null;
                    }
                });
            };
            window.cookieconsent.status.allow = options.policies.cookies.hash;
            window.cookieconsent.status.dismiss = window.cookieconsent.status.deny;
            window.cookieconsent.hasTransition = false;
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
                position: "bottom",
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
                    link: '<a aria-label="learn more about cookies" tabindex="0" ' +
                        'class="cc-link cookies" href="{{href}}">{{link}}</a>'
                },
                onStatusChange: function () {
                    if (cookiemodal) {
                        // Change the title based on the current consent status
                        cookiemodal.find('.modal-title').html(getCookieTitle());
                    }
                    // Toggle GA if the user consents, disable if not.
                    // analytics.toggle has internal checks for when we initialize without being turned off first
                    analytics.toggle(this.hasConsented());
                },
                onInitialise: function () {
                    // Toggle GA on if the user has already consented
                    analytics.toggle(this.hasConsented());
                },
                onPopupClose: function () {
                    $(window).resize();
                }
            });


            $('#cookies').click(openCookiePolicy);
            $('.cookies').click(openCookiePolicy);
        }

        // I'd like for this to be the only function used, but it gets messy to pass the callback function around,
        // so we instead trigger a click here when we want it to open with this effect. Sorry!
        if (options.policies.privacy.enabled) {
            $('#privacy').click(function (event, data) {
                alertSystem.alert(
                    data && data.title ? data.title : "Privacy policy",
                    require('./policies/privacy.html')
                );
                // I can't remember why this check is here as it seems superfluous
                if (options.policies.privacy.enabled) {
                    local.set(options.policies.privacy.key, options.policies.privacy.hash);
                }
            });
        }

        $('#ui-reset').click(function () {
            local.remove('gl');
            hasUIBeenReset = true;
            window.history.replaceState(null, null, window.httpRoot);
            window.location.reload();
        });
        $('#thanks-to').click(function () {
            alertSystem.alert("Special thanks to", $(require('./thanks.html')));
        });
        $('#changes').click(function () {
            alertSystem.alert("Changelog", $(require('./changelog.html')));
        });
    }

    function findConfig(defaultConfig, options) {
        var config = null;
        if (!options.embedded) {
            if (options.config) {
                config = options.config;
            } else {
                config = url.deserialiseState(window.location.hash.substr(1));
            }

            if (config) {
                // replace anything in the default config with that from the hash
                config = _.extend(defaultConfig, config);
            }
            if (!config) {
                var savedState = local.get('gl', null);
                config = savedState !== null ? JSON.parse(savedState) : defaultConfig;
            }
        } else {
            config = _.extend(defaultConfig, {
                settings: {
                    showMaximiseIcon: false,
                    showCloseIcon: false,
                    hasHeaders: false
                }
            }, sharing.configFromEmbedded(window.location.hash.substr(1)));
        }
        return config;
    }

    function initializeResetLayoutLink() {
        var currentUrl = document.URL;
        if (currentUrl.includes("/z/")) {
            $("#ui-brokenlink").attr("href", currentUrl.replace("/z/", "/resetlayout/"));
            $("#ui-brokenlink").show();
        } else {
            $("#ui-brokenlink").hide();
        }
    }

    function start() {
        initializeResetLayoutLink();

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
            if (window.location.hash.substr(1)) window.location.reload();
        });

        // Which buttons act as a linkable popup
        var linkablePopups = ['#thanks-to', '#changes', '#cookies', '#setting', '#privacy'];
        var hashPart = linkablePopups.indexOf(window.location.hash) > -1 ? window.location.hash : null;
        if (hashPart) {
            window.location.hash = "";
        }

        var config = findConfig(defaultConfig, options);

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

        var lastState = null;
        var storedPaths = {};  // TODO maybe make this an LRU cache?

        layout.on('stateChanged', function () {
            var config = layout.toConfig();
            var stringifiedConfig = JSON.stringify(config);
            if (stringifiedConfig !== lastState) {
                if (storedPaths[config]) {
                    window.history.replaceState(null, null, storedPaths[stringifiedConfig]);
                } else if (window.location.pathname !== window.httpRoot) {
                    window.history.replaceState(null, null, window.httpRoot);
                    // TODO: Add this state to storedPaths, but with a upper bound on the stored state count
                }
                lastState = stringifiedConfig;
            }
            if (options.embedded) {
                var strippedToLast = window.location.pathname;
                strippedToLast = strippedToLast.substr(0, strippedToLast.lastIndexOf('/') + 1);
                $('a.link').attr('href', strippedToLast + '#' + url.serialiseState(config));
            }
        });

        function sizeRoot() {
            var height = $(window).height() - (root.position().top || 0) - ($('.cc-window:visible').height() || 0);
            root.height(height);
            layout.updateSize();
        }

        $(window)
            .resize(sizeRoot)
            .on('beforeunload', function () {
                // Only preserve state in localStorage in non-embedded mode.
                if (!options.embedded && !hasUIBeenReset) {
                    local.set('gl', JSON.stringify(layout.toConfig()));
                }
            });

        new clipboard('.btn.clippy');

        var settings = setupSettings(hub);

        // We assume no consent for embed users
        if (!options.embedded) {
            setupButtons(options);
        }

        sharing.initShareButton($('#share'), layout, function (config, extra) {
            window.history.pushState(null, null, extra);
            storedPaths[JSON.stringify(config)] = extra;
        });

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

        if (hashPart) {
            var element = $(hashPart);
            if (element) element.click();
        }
        // Ensure old cookies are removed, to avoid user confusion
        document.cookie = 'fs_uid=;expires=Thu, 01 Jan 1970 00:00:01 GMT;';
        if (options.policies.privacy.enabled &&
            options.policies.privacy.hash !== local.get(options.policies.privacy.key)) {
            $('#privacy').trigger('click', {
                title: 'New Privacy Policy. Please take a moment to read it'
            });
        }
        var onHide = function () {
            hub.layout.eventHub.emit('modifySettings', {
                enableCommunityAds: false
            });
        };
        motd.initialise(options.motdUrl, $('#motd'), subLangId, settings.enableCommunityAds, onHide);
        sizeRoot();
        lastState = JSON.stringify(layout.toConfig());
    }

    $(start);
});

