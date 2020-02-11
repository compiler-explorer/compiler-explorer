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

// setup analytics before anything else so we can capture any future errors in sentry
var analytics = require('./analytics');

require('popper.js');
require('bootstrap');
require('bootstrap-slider');

var sharing = require('./sharing');
var _ = require('underscore');
var $ = require('jquery');
var GoldenLayout = require('golden-layout');
var Components = require('./components');
var url = require('./url');
var clipboard = require('clipboard');
var Hub = require('./hub');
var Sentry = require('@sentry/browser');
var settings = require('./settings');
var local = require('./local');
var Alert = require('./alert');
var themer = require('./themes');
var motd = require('./motd');
var jsCookie = require('js-cookie');
var SimpleCook = require('./simplecook');
var History = require('./history');
var HistoryWidget = require('./history-widget').HistoryWidget;

//css
require("bootstrap/dist/css/bootstrap.min.css");
require("golden-layout/src/css/goldenlayout-base.css");
require("selectize/dist/css/selectize.bootstrap2.css");
require("bootstrap-slider/dist/css/bootstrap-slider.css");
require("./colours.css");
require("./explorer.css");

// Check to see if the current unload is a UI reset.
// Forgive me the global usage here
var hasUIBeenReset = false;
var simpleCooks = new SimpleCook();
var historyWidget = new HistoryWidget();

// Polyfill includes for IE11 - From MDN
if (!String.prototype.includes) {
    String.prototype.includes = function (search, start) {
        if (search instanceof RegExp) {
            throw TypeError('first argument must not be a RegExp');
        }
        if (start === undefined) { start = 0; }
        return this.indexOf(search, start) !== -1;
    };
}

function setupSettings(hub) {
    var eventHub = hub.layout.eventHub;
    var defaultSettings = {
        defaultLanguage: hub.defaultLangId
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

function hasCookieConsented(options) {
    return jsCookie.get(options.policies.cookies.key) === options.policies.cookies.hash;
}

function setupButtons(options) {
    var alertSystem = new Alert();

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
                jsCookie.set(options.policies.privacy.key, options.policies.privacy.hash, {expires: 365});
            }
        });
    }

    if (options.policies.cookies.enabled) {
        var getCookieTitle = function () {
            return 'Cookies & related technologies policy<br><p>Current consent status: <span style="color:' +
                (hasCookieConsented(options) ? 'green' : 'red') + '">' +
                (hasCookieConsented(options) ? 'Granted' : 'Denied') + '</span></p>';
        };
        $('#cookies').click(function () {
            alertSystem.ask(getCookieTitle(), $(require('./policies/cookies.html')), {
                yes: function () {
                    simpleCooks.callDoConsent.apply(simpleCooks);
                },
                yesHtml: 'Consent',
                no: function () {
                    simpleCooks.callDontConsent.apply(simpleCooks);
                },
                noHtml: 'Do NOT consent'
            });
        });
    }

    $('#ui-reset').click(function () {
        local.remove('gl');
        hasUIBeenReset = true;
        window.history.replaceState(null, null, window.httpRoot);
        window.location.reload();
    });

    $('#ui-duplicate').click(function () {
        window.open('/', '_blank');
    });

    $('#thanks-to').click(function () {
        alertSystem.alert("Special thanks to", $(require('./thanks.html')));
    });
    $('#changes').click(function () {
        alertSystem.alert("Changelog", $(require('./changelog.html')));
    });

    $('#ui-history').click(function () {
        historyWidget.run(function (data) {
            local.set('gl', JSON.stringify(data.config));
            hasUIBeenReset = true;
            window.history.replaceState(null, null, window.httpRoot);
            window.location.reload();
        });

        $('#history').modal();
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

function initPolicies(options) {
    // Ensure old cookies are removed, to avoid user confusion

    jsCookie.remove('fs_uid');
    jsCookie.remove('cookieconsent_status');
    if (options.policies.privacy.enabled &&
        options.policies.privacy.hash !== jsCookie.get(options.policies.privacy.key)) {
        $('#privacy').trigger('click', {
            title: 'New Privacy Policy. Please take a moment to read it'
        });
    }
    simpleCooks.onDoConsent = function () {
        jsCookie.set(options.policies.cookies.key, options.policies.cookies.hash, {expires: 365});
        analytics.toggle(true);
    };
    simpleCooks.onDontConsent = function () {
        analytics.toggle(false);
        jsCookie.set(options.policies.cookies.key, '');
    };
    simpleCooks.onHide = function () {
        $(window).trigger('resize');
    };
    // '' means no consent. Hash match means consent of old. Null means new user!
    var storedCookieConsent = jsCookie.get(options.policies.cookies.key);
    if (options.policies.cookies.enabled && storedCookieConsent !== '' &&
        options.policies.cookies.hash !== storedCookieConsent) {
        simpleCooks.show();
    } else if (options.policies.cookies.enabled && hasCookieConsented(options)) {
        analytics.initialise();
    }
}

// eslint-disable-next-line max-statements
function start() {
    initializeResetLayoutLink();

    var options = require('options');

    var hostnameParts = window.location.hostname.split('.');
    var subLangId = undefined;
    // Only set the subdomain lang id if it makes sense to do so
    if (hostnameParts.length > 0) {
        var subdomainPart = hostnameParts[0];
        var langBySubdomain = _.find(options.languages, function (lang) {
            return lang.id === subdomainPart || lang.alias.indexOf(subdomainPart) !== -1;
        });
        if (langBySubdomain) {
            subLangId = langBySubdomain.id;
        }
    }
    var defaultLangId = subLangId;
    if (!defaultLangId) {
        if (options.languages["c++"]) {
            defaultLangId = "c++";
        } else {
            defaultLangId = _.keys(options.languages)[0];
        }
    }

    // Cookie domains are matched as a RE against the window location. This allows a flexible
    // way that works across multiple domains (e.g. godbolt.org and compiler-explorer.com).
    // We allow this to be configurable so that (for example), gcc.godbolt.org and d.godbolt.org
    // share the same cookie domain for some settings.
    var cookieDomain = new RegExp(options.cookieDomainRe).exec(window.location.hostname);
    if (cookieDomain && cookieDomain[0]) {
        cookieDomain = cookieDomain[0];
        jsCookie.defaults.domain = cookieDomain;
    }

    var defaultConfig = {
        settings: {showPopoutIcon: false},
        content: [{
            type: 'row',
            content: [
                Components.getEditor(1, defaultLangId),
                Components.getCompiler(1, defaultLangId)
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
        hub = new Hub(layout, subLangId, defaultLangId);
    } catch (e) {
        Sentry.captureException(e);

        if (document.URL.includes("/z/")) {
            document.location = document.URL.replace("/z/", "/resetlayout/");
        }

        layout = new GoldenLayout(defaultConfig, root);
        hub = new Hub(layout, subLangId, defaultLangId);
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

            History.push(stringifiedConfig);
        }
        if (options.embedded) {
            var strippedToLast = window.location.pathname;
            strippedToLast = strippedToLast.substr(0, strippedToLast.lastIndexOf('/') + 1);
            $('a.link').attr('href', strippedToLast + '#' + url.serialiseState(config));
        }
    });

    function sizeRoot() {
        var height = $(window).height() - (root.position().top || 0) - ($('#simplecook:visible').height() || 0);
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

    setupAdd($('#add-editor'), function () {
        return Components.getEditor();
    });
    setupAdd($('#add-diff'), function () {
        return Components.getDiff();
    });

    if (hashPart) {
        var element = $(hashPart);
        if (element) element.click();
    }
    initPolicies(options);

    // Skip some steps if using embedded mode
    if (!options.embedded) {
        // Only fetch MOTD when not embedded.
        motd.initialise(options.motdUrl, $('#motd'), subLangId, settings.enableCommunityAds,
            function (data) {
                var sendMotd = function () {
                    hub.layout.eventHub.emit('motd', data);
                };
                hub.layout.eventHub.on('requestMotd', sendMotd);
                sendMotd();
            },
            function () {
                hub.layout.eventHub.emit('modifySettings', {
                    enableCommunityAds: false
                });
            });

        // Don't try to update Version tree link
        var release = window.compilerExplorerOptions.gitReleaseCommit;
        var versionLink = 'https://github.com/mattgodbolt/compiler-explorer/';
        if (release) {
            versionLink += 'tree/' + release;
        }
        $('#version-tree').prop('href', versionLink);
    }

    if (options.hideEditorToolbars) {
        $('[name="editor-btn-toolbar"]').addClass("d-none");
    }

    sizeRoot();
    lastState = JSON.stringify(layout.toConfig());
}

$(start);
