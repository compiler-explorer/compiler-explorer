// Copyright (c) 2016, Compiler Explorer Authors
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

// setup analytics before anything else so we can capture any future errors in sentry
var analytics = require('./analytics').ga;

require('whatwg-fetch');
// eslint-disable-next-line requirejs/no-js-extension
require('popper.js');
require('bootstrap');

var Sharing = require('./sharing').Sharing;
var _ = require('underscore');
var $ = require('jquery');
var GoldenLayout = require('golden-layout');
var Components = require('./components');
var url = require('./url');
var clipboard = require('clipboard');
var Hub = require('./hub').Hub;
var Sentry = require('@sentry/browser');
var Settings = require('./settings').Settings;
var local = require('./local');
var Alert = require('./alert').Alert;
var themer = require('./themes');
var motd = require('./motd');
var jsCookie = require('js-cookie');
var SimpleCook = require('./widgets/simplecook').SimpleCook;
var HistoryWidget = require('./widgets/history-widget').HistoryWidget;
var History = require('./history');
var Presentation = require('./presentation').Presentation;
var setupSiteTemplateWidgetButton = require('./widgets/site-templates-widget').setupSiteTemplateWidgetButton;

var logos = require.context('../views/resources/logos', false, /\.(png|svg)$/);

var siteTemplateScreenshots = require.context('../views/resources/template_screenshots', false, /\.png$/);

if (!window.PRODUCTION) {
    require('./tests/_all');
}

//css
require('bootstrap/dist/css/bootstrap.min.css');
require('golden-layout/src/css/goldenlayout-base.css');
require('tom-select/dist/css/tom-select.bootstrap4.css');
require('./colours.scss');
require('./explorer.scss');

// Check to see if the current unload is a UI reset.
// Forgive me the global usage here
var hasUIBeenReset = false;
var simpleCooks = new SimpleCook();
var historyWidget = new HistoryWidget();

var policyDocuments = {
    cookies: require('./generated/cookies.pug').default,
    privacy: require('./generated/privacy.pug').default,
};

function setupSettings(hub) {
    var eventHub = hub.layout.eventHub;
    var defaultSettings = {
        defaultLanguage: hub.defaultLangId,
    };
    var currentSettings = JSON.parse(local.get('settings', null)) || defaultSettings;

    function onChange(newSettings) {
        if (currentSettings.theme !== newSettings.theme) {
            analytics.proxy('send', {
                hitType: 'event',
                eventCategory: 'ThemeChange',
                eventAction: newSettings.theme,
            });
        }
        if (currentSettings.colourScheme !== newSettings.colourScheme) {
            analytics.proxy('send', {
                hitType: 'event',
                eventCategory: 'ColourSchemeChange',
                eventAction: newSettings.colourScheme,
            });
        }
        $('#settings').find('.editorsFFont').css('font-family', newSettings.editorsFFont);
        currentSettings = newSettings;
        local.set('settings', JSON.stringify(newSettings));
        eventHub.emit('settingsChange', newSettings);
    }

    new themer.Themer(eventHub, currentSettings);

    eventHub.on('requestSettings', function () {
        eventHub.emit('settingsChange', currentSettings);
    });

    var SettingsObject = new Settings(hub, $('#settings'), currentSettings, onChange, hub.subdomainLangId);
    eventHub.on('modifySettings', function (newSettings) {
        SettingsObject.setSettings(_.extend(currentSettings, newSettings));
    });
    return currentSettings;
}

function hasCookieConsented(options) {
    return jsCookie.get(options.policies.cookies.key) === policyDocuments.cookies.hash;
}

function isMobileViewer() {
    return window.compilerExplorerOptions.mobileViewer;
}

function calcLocaleChangedDate(policyModal) {
    var timestamp = policyModal.find('#changed-date');
    timestamp.text(new Date(timestamp.attr('datetime')).toLocaleString());
}

function setupButtons(options, hub) {
    var eventHub = hub.createEventHub();
    var alertSystem = new Alert();

    // I'd like for this to be the only function used, but it gets messy to pass the callback function around,
    // so we instead trigger a click here when we want it to open with this effect. Sorry!
    if (options.policies.privacy.enabled) {
        $('#privacy').on('click', function (event, data) {
            var modal = alertSystem.alert(
                data && data.title ? data.title : 'Privacy policy',
                policyDocuments.privacy.text
            );
            calcLocaleChangedDate(modal);
            // I can't remember why this check is here as it seems superfluous
            if (options.policies.privacy.enabled) {
                jsCookie.set(options.policies.privacy.key, policyDocuments.privacy.hash, {
                    expires: 365,
                    sameSite: 'strict',
                });
            }
        });
    }

    if (options.policies.cookies.enabled) {
        var getCookieTitle = function () {
            return (
                'Cookies &amp; related technologies policy<br><p>Current consent status: <span style="color:' +
                (hasCookieConsented(options) ? 'green' : 'red') +
                '">' +
                (hasCookieConsented(options) ? 'Granted' : 'Denied') +
                '</span></p>'
            );
        };
        $('#cookies').on('click', function () {
            var modal = alertSystem.ask(getCookieTitle(), policyDocuments.cookies.text, {
                yes: function () {
                    simpleCooks.callDoConsent.apply(simpleCooks);
                },
                yesHtml: 'Consent',
                no: function () {
                    simpleCooks.callDontConsent.apply(simpleCooks);
                },
                noHtml: 'Do NOT consent',
            });
            calcLocaleChangedDate(modal);
        });
    }

    $('#ui-reset').on('click', function () {
        local.remove('gl');
        hasUIBeenReset = true;
        window.history.replaceState(null, null, window.httpRoot);
        window.location.reload();
    });

    $('#ui-duplicate').on('click', function () {
        window.open('/', '_blank');
    });

    $('#changes').on('click', function () {
        alertSystem.alert('Changelog', $(require('./generated/changelog.pug').default.text));
    });

    $('#ces').on('click', function () {
        $.get(window.location.origin + window.httpRoot + 'bits/sponsors.html')
            .done(function (data) {
                alertSystem.alert('Compiler Explorer Sponsors', data);
                analytics.proxy('send', {
                    hitType: 'event',
                    eventCategory: 'Sponsors',
                    eventAction: 'open',
                });
            })
            .fail(function (err) {
                var result = err.responseText || JSON.stringify(err);
                alertSystem.alert(
                    'Compiler Explorer Sponsors',
                    '<div>Unable to fetch sponsors:</div><div>' + result + '</div>'
                );
            });
    });

    $('#ui-history').on('click', function () {
        historyWidget.run(function (data) {
            local.set('gl', JSON.stringify(data.config));
            hasUIBeenReset = true;
            window.history.replaceState(null, null, window.httpRoot);
            window.location.reload();
        });

        $('#history').modal();
    });

    $('#ui-apply-default-font-scale').on('click', function () {
        var defaultFontScale = Settings.getStoredSettings().defaultFontScale;
        if (defaultFontScale !== undefined) {
            eventHub.emit('broadcastFontScale', defaultFontScale);
        }
    });
}

function configFromEmbedded(embeddedUrl) {
    // Old-style link?
    var params;
    try {
        params = url.unrisonify(embeddedUrl);
    } catch (e) {
        // Ignore this, it's not a problem
    }
    if (params && params.source && params.compiler) {
        var filters = _.chain((params.filters || '').split(','))
            .map(function (o) {
                return [o, true];
            })
            .object()
            .value();
        return {
            content: [
                {
                    type: 'row',
                    content: [
                        Components.getEditorWith(1, params.source, filters),
                        Components.getCompilerWith(1, filters, params.options, params.compiler),
                    ],
                },
            ],
        };
    } else {
        return url.deserialiseState(embeddedUrl);
    }
}

function fixBugsInConfig(config) {
    if (config.activeItemIndex && config.activeItemIndex >= config.content.length) {
        config.activeItemIndex = config.content.length - 1;
    }

    _.each(config.content, function (item) {
        fixBugsInConfig(item);
    });
}

function findConfig(defaultConfig, options) {
    var config;
    if (!options.embedded) {
        if (options.slides) {
            var presentation = new Presentation(window.compilerExplorerOptions.slides.length);
            var currentSlide = presentation.currentSlide;
            if (currentSlide < options.slides.length) {
                config = options.slides[currentSlide];
            } else {
                presentation.currentSlide = 0;
                config = options.slides[0];
            }
            if (
                isMobileViewer() &&
                window.compilerExplorerOptions.slides &&
                window.compilerExplorerOptions.slides.length > 1
            ) {
                $('#share').remove();
                $('.ui-presentation-control').removeClass('d-none');
                $('.ui-presentation-first').on('click', presentation.first.bind(presentation));
                $('.ui-presentation-prev').on('click', presentation.previous.bind(presentation));
                $('.ui-presentation-next').on('click', presentation.next.bind(presentation));
            }
        } else {
            if (options.config) {
                config = options.config;
            } else {
                try {
                    config = url.deserialiseState(window.location.hash.substring(1));
                } catch (e) {
                    // #3518 Alert the user that the url is invalid
                    var alertSystem = new Alert();
                    alertSystem.notify(
                        'Unable to load custom configuration from URL,\
                     the last locally saved configuration will be used if present.',
                        {
                            alertClass: 'notification-error',
                            dismissTime: 5000,
                        }
                    );
                }
            }

            if (config) {
                // replace anything in the default config with that from the hash
                config = _.extend(defaultConfig, config);
            }
            if (!config) {
                var savedState = local.get('gl', null);
                config = savedState !== null ? JSON.parse(savedState) : defaultConfig;
            }
        }
    } else {
        config = _.extend(
            defaultConfig,
            {
                settings: {
                    showMaximiseIcon: false,
                    showCloseIcon: false,
                    hasHeaders: false,
                },
            },
            configFromEmbedded(window.location.hash.substr(1))
        );
    }

    removeOrphanedMaximisedItemFromConfig(config);
    fixBugsInConfig(config);

    return config;
}

function initializeResetLayoutLink() {
    var currentUrl = document.URL;
    if (currentUrl.includes('/z/')) {
        $('#ui-brokenlink').attr('href', currentUrl.replace('/z/', '/resetlayout/')).show();
    } else {
        $('#ui-brokenlink').hide();
    }
}

function initPolicies(options) {
    if (options.policies.privacy.enabled) {
        if (jsCookie.get(options.policies.privacy.key) == null) {
            $('#privacy').trigger('click', {
                title: 'New Privacy Policy. Please take a moment to read it',
            });
        } else if (policyDocuments.privacy.hash !== jsCookie.get(options.policies.privacy.key)) {
            // When the user has already accepted the privacy, just show a pretty notification.
            var ppolicyBellNotification = $('#policyBellNotification');
            var pprivacyBellNotification = $('#privacyBellNotification');
            var pcookiesBellNotification = $('#cookiesBellNotification');
            ppolicyBellNotification.removeClass('d-none');
            pprivacyBellNotification.removeClass('d-none');
            $('#privacy').on('click', function () {
                // Only hide if the other policy does not also have a bell
                if (pcookiesBellNotification.hasClass('d-none')) {
                    ppolicyBellNotification.addClass('d-none');
                }
                pprivacyBellNotification.addClass('d-none');
            });
        }
    }
    simpleCooks.setOnDoConsent(function () {
        jsCookie.set(options.policies.cookies.key, policyDocuments.cookies.hash, {
            expires: 365,
            sameSite: 'strict',
        });
        analytics.toggle(true);
    });
    simpleCooks.setOnDontConsent(function () {
        analytics.toggle(false);
        jsCookie.set(options.policies.cookies.key, '', {
            sameSite: 'strict',
        });
    });
    simpleCooks.setOnHide(function () {
        var spolicyBellNotification = $('#policyBellNotification');
        var sprivacyBellNotification = $('#privacyBellNotification');
        var scookiesBellNotification = $('#cookiesBellNotification');
        // Only hide if the other policy does not also have a bell
        if (sprivacyBellNotification.hasClass('d-none')) {
            spolicyBellNotification.addClass('d-none');
        }
        scookiesBellNotification.addClass('d-none');
        $(window).trigger('resize');
    });
    // '' means no consent. Hash match means consent of old. Null means new user!
    var storedCookieConsent = jsCookie.get(options.policies.cookies.key);
    if (options.policies.cookies.enabled) {
        if (storedCookieConsent !== '' && policyDocuments.cookies.hash !== storedCookieConsent) {
            simpleCooks.show();
            var cpolicyBellNotification = $('#policyBellNotification');
            var cprivacyBellNotification = $('#privacyBellNotification');
            var ccookiesBellNotification = $('#cookiesBellNotification');
            cpolicyBellNotification.removeClass('d-none');
            ccookiesBellNotification.removeClass('d-none');
            $('#cookies').on('click', function () {
                if (cprivacyBellNotification.hasClass('d-none')) {
                    cpolicyBellNotification.addClass('d-none');
                }
                ccookiesBellNotification.addClass('d-none');
            });
        } else if (hasCookieConsented(options)) {
            analytics.initialise();
        }
    }
}

/*
 * this nonsense works around a bug in goldenlayout where a config can be generated
 * that contains a flag indicating there is a maximized item which does not correspond
 * to any items that actually exist in the config.
 *
 * See https://github.com/compiler-explorer/compiler-explorer/issues/2056
 */
function removeOrphanedMaximisedItemFromConfig(config) {
    // nothing to do if the maximised item id is not set
    if (config.maximisedItemId !== '__glMaximised') return;

    var found = false;

    function impl(component) {
        if (component.id === '__glMaximised') {
            found = true;
            return;
        }

        if (component.content) {
            for (var i = 0; i < component.content.length; i++) {
                impl(component.content[i]);
                if (found) return;
            }
        }
    }

    impl(config);

    if (!found) {
        config.maximisedItemId = null;
    }
}

function setupLanguageLogos(languages) {
    _.each(
        languages,
        function (lang) {
            try {
                lang.logoData = logos('./' + lang.logoUrl);
                if (lang.logoUrlDark) {
                    lang.logoDataDark = logos('./' + lang.logoUrlDark);
                }
            } catch (ignored) {
                lang.logoData = '';
            }
        },
        this
    );
}

function earlyGetDefaultLangSetting() {
    // returns string | undefined
    return Settings.getStoredSettings().defaultLanguage;
}

// eslint-disable-next-line max-statements
function start() {
    initializeResetLayoutLink();
    setupSiteTemplateWidgetButton(siteTemplateScreenshots);

    var options = require('options').options;

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
        var defaultLangSetting = earlyGetDefaultLangSetting();
        if (defaultLangSetting) {
            defaultLangId = defaultLangSetting;
        } else if (options.languages['c++']) {
            defaultLangId = 'c++';
        } else {
            defaultLangId = _.keys(options.languages)[0];
        }
    }

    setupLanguageLogos(options.languages);

    // Cookie domains are matched as a RE against the window location. This allows a flexible
    // way that works across multiple domains (e.g. godbolt.org and compiler-explorer.com).
    // We allow this to be configurable so that (for example), gcc.godbolt.org and d.godbolt.org
    // share the same cookie domain for some settings.
    var cookieDomain = new RegExp(options.cookieDomainRe).exec(window.location.hostname);
    if (cookieDomain && cookieDomain[0]) {
        jsCookie = jsCookie.withAttributes({domain: cookieDomain[0]});
    }

    var defaultConfig = {
        settings: {showPopoutIcon: false},
        content: [
            {
                type: 'row',
                content: [Components.getEditor(1, defaultLangId), Components.getCompiler(1, defaultLangId)],
            },
        ],
    };

    $(window).on('hashchange', function () {
        // punt on hash events and just reload the page if there's a hash
        if (window.location.hash.substr(1)) window.location.reload();
    });

    // Which buttons act as a linkable popup
    var linkablePopups = ['#ces', '#sponsors', '#changes', '#cookies', '#setting', '#privacy'];
    var hashPart = linkablePopups.indexOf(window.location.hash) > -1 ? window.location.hash : null;
    if (hashPart) {
        window.location.hash = '';
        // Handle the time we renamed sponsors to ces to work around issues with blockers.
        if (hashPart === '#sponsors') hashPart = '#ces';
    }

    var config = findConfig(defaultConfig, options);

    var root = $('#root');

    var layout;
    var hub;
    try {
        layout = new GoldenLayout(config, root);
        hub = new Hub(layout, subLangId, defaultLangId);
    } catch (e) {
        Sentry.captureException(e);

        if (document.URL.includes('/z/')) {
            document.location = document.URL.replace('/z/', '/resetlayout/');
        }

        layout = new GoldenLayout(defaultConfig, root);
        hub = new Hub(layout, subLangId, defaultLangId);
    }

    if (hub.hasTree()) {
        $('#add-tree').prop('disabled', true);
    }

    function sizeRoot() {
        var height = $(window).height() - (root.position().top || 0) - ($('#simplecook:visible').height() || 0);
        root.height(height);
        layout.updateSize();
    }

    $(window)
        .on('resize', sizeRoot)
        .on('beforeunload', function () {
            // Only preserve state in localStorage in non-embedded mode.
            var shouldSave = !window.hasUIBeenReset && !hasUIBeenReset;
            if (!options.embedded && !isMobileViewer() && shouldSave) {
                local.set('gl', JSON.stringify(layout.toConfig()));
            }
        });

    new clipboard('.btn.clippy');

    var settings = setupSettings(hub);

    // We assume no consent for embed users
    if (!options.embedded) {
        setupButtons(options, hub);
    }

    var addDropdown = $('#addDropdown');

    function setupAdd(thing, func) {
        layout.createDragSource(thing, func)._dragListener.on('dragStart', function () {
            addDropdown.dropdown('toggle');
        });

        thing.on('click', function () {
            if (hub.hasTree()) {
                hub.addInEditorStackIfPossible(func());
            } else {
                hub.addAtRoot(func());
            }
        });
    }

    setupAdd($('#add-editor'), function () {
        return Components.getEditor();
    });
    setupAdd($('#add-diff'), function () {
        return Components.getDiff();
    });
    setupAdd($('#add-tree'), function () {
        $('#add-tree').prop('disabled', true);
        return Components.getTree();
    });

    if (hashPart) {
        var element = $(hashPart);
        if (element) element.trigger('click');
    }
    initPolicies(options);

    // Skip some steps if using embedded mode
    if (!options.embedded) {
        // Only fetch MOTD when not embedded.
        motd.initialise(
            options.motdUrl,
            $('#motd'),
            subLangId,
            settings.enableCommunityAds,
            function (data) {
                var sendMotd = function () {
                    hub.layout.eventHub.emit('motd', data);
                };
                hub.layout.eventHub.on('requestMotd', sendMotd);
                sendMotd();
            },
            function () {
                hub.layout.eventHub.emit('modifySettings', {
                    enableCommunityAds: false,
                });
            }
        );

        // Don't try to update Version tree link
        var release = window.compilerExplorerOptions.gitReleaseCommit;
        var versionLink = 'https://github.com/compiler-explorer/compiler-explorer/';
        if (release) {
            versionLink += 'tree/' + release;
        }
        $('#version-tree').prop('href', versionLink);
    }

    if (options.hideEditorToolbars) {
        $('[name="editor-btn-toolbar"]').addClass('d-none');
    }

    window.onSponsorClick = function (sponsor) {
        analytics.proxy('send', {
            hitType: 'event',
            eventCategory: 'Sponsors',
            eventAction: 'click',
            eventLabel: sponsor.url,
            transport: 'beacon',
        });
        window.open(sponsor.url);
    };

    if (options.pageloadUrl) {
        setTimeout(function () {
            var visibleIcons = $('.ces-icon:visible')
                .map(function (index, value) {
                    return value.dataset.statsid;
                })
                .get()
                .join(',');
            $.post(options.pageloadUrl + '?icons=' + encodeURIComponent(visibleIcons));
        }, 5000);
    }

    sizeRoot();

    History.trackHistory(layout);
    new Sharing(layout);
}

$(start);
