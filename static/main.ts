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

// setup analytics before anything else so we can capture any future errors in sentry
import {ga as analytics} from './analytics.js';

import 'whatwg-fetch';
import 'popper.js'; // eslint-disable-line requirejs/no-js-extension
import 'bootstrap';

import $ from 'jquery';
import _ from 'underscore';

import GoldenLayout from 'golden-layout';
import JsCookie from 'js-cookie';
import clipboard from 'clipboard';
import * as Sentry from '@sentry/browser';

// We re-assign this
let jsCookie = JsCookie;

import {Sharing} from './sharing.js';
import * as Components from './components.js';
import * as url from './url.js';
import {Hub} from './hub.js';
import {Settings, SiteSettings} from './settings.js';
import * as local from './local.js';
import {Alert} from './widgets/alert.js';
import * as themer from './themes.js';
import * as motd from './motd.js';
import {SimpleCook} from './widgets/simplecook.js';
import {HistoryWidget} from './widgets/history-widget.js';
import * as History from './history.js';
import {Presentation} from './presentation.js';
import {setupSiteTemplateWidgetButton} from './widgets/site-templates-widget.js';
import {options} from './options.js';
import {unwrap} from './assert.js';

import {Language, LanguageKey} from '../types/languages.interfaces.js';
import {CompilerExplorerOptions} from './global.js';
import {ComponentConfig, EmptyCompilerState, StateWithId, StateWithLanguage} from './components.interfaces.js';

import * as utils from '../lib/common-utils.js';

const logos = require.context('../views/resources/logos', false, /\.(png|svg)$/);

const siteTemplateScreenshots = require.context('../views/resources/template_screenshots', false, /\.png$/);

if (!window.PRODUCTION && !options.embedded) {
    require('./tests/_all');
}

//css
require('bootstrap/dist/css/bootstrap.min.css');
require('golden-layout/src/css/goldenlayout-base.css');
require('tom-select/dist/css/tom-select.bootstrap4.css');
require('./colours.scss');
require('./styles/explorer.scss');

// Check to see if the current unload is a UI reset.
// Forgive me the global usage here
let hasUIBeenReset = false;
const simpleCooks = new SimpleCook();
const historyWidget = new HistoryWidget();

const policyDocuments = {
    cookies: require('./generated/cookies.pug').default,
    privacy: require('./generated/privacy.pug').default,
};

function setupSettings(hub: Hub) {
    const eventHub = hub.layout.eventHub;
    const defaultSettings = {
        defaultLanguage: hub.defaultLangId,
    };
    let currentSettings: SiteSettings = JSON.parse(local.get('settings', 'null')) || defaultSettings;

    function onChange(newSettings: SiteSettings) {
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

    eventHub.on('requestSettings', () => {
        eventHub.emit('settingsChange', currentSettings);
    });

    const SettingsObject = new Settings(hub, $('#settings'), currentSettings, onChange, hub.subdomainLangId);
    eventHub.on('modifySettings', (newSettings: Partial<SiteSettings>) => {
        SettingsObject.setSettings(_.extend(currentSettings, newSettings));
    });
    return currentSettings;
}

function hasCookieConsented(options: CompilerExplorerOptions) {
    return jsCookie.get(options.policies.cookies.key) === policyDocuments.cookies.hash;
}

function isMobileViewer() {
    return window.compilerExplorerOptions.mobileViewer;
}

function calcLocaleChangedDate(policyModal: JQuery) {
    const timestamp = policyModal.find('#changed-date');
    timestamp.text(new Date(unwrap(timestamp.attr('datetime'))).toLocaleString());
}

function setupButtons(options: CompilerExplorerOptions, hub: Hub) {
    const eventHub = hub.createEventHub();
    const alertSystem = new Alert();

    // I'd like for this to be the only function used, but it gets messy to pass the callback function around,
    // so we instead trigger a click here when we want it to open with this effect. Sorry!
    if (options.policies.privacy.enabled) {
        $('#privacy').on('click', (event, data) => {
            const modal = alertSystem.alert(
                data && data.title ? data.title : 'Privacy policy',
                policyDocuments.privacy.text,
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
        const getCookieTitle = () => {
            return (
                'Cookies &amp; related technologies policy<br><p>Current consent status: <span style="color:' +
                (hasCookieConsented(options) ? 'green' : 'red') +
                '">' +
                (hasCookieConsented(options) ? 'Granted' : 'Denied') +
                '</span></p>'
            );
        };
        $('#cookies').on('click', () => {
            const modal = alertSystem.ask(getCookieTitle(), policyDocuments.cookies.text, {
                yes: () => {
                    simpleCooks.callDoConsent.apply(simpleCooks);
                },
                yesHtml: 'Consent',
                no: () => {
                    simpleCooks.callDontConsent.apply(simpleCooks);
                },
                noHtml: 'Do NOT consent',
            });
            calcLocaleChangedDate(modal);
        });
    }

    $('#ui-reset').on('click', () => {
        local.remove('gl');
        hasUIBeenReset = true;
        window.history.replaceState(null, '', window.httpRoot);
        window.location.reload();
    });

    $('#ui-duplicate').on('click', () => {
        window.open('/', '_blank');
    });

    $('#changes').on('click', () => {
        // TODO(jeremy-rifkin): Fix types
        alertSystem.alert('Changelog', $(require('./generated/changelog.pug').default.text) as any);
    });

    $.get(window.location.origin + window.httpRoot + 'bits/icons.html')
        .done(data => {
            $('#ces .ces-icons').html(data);
        })
        .fail(err => {
            Sentry.captureException(err);
        });

    $('#ces').on('click', () => {
        $.get(window.location.origin + window.httpRoot + 'bits/sponsors.html')
            .done(data => {
                alertSystem.alert('Compiler Explorer Sponsors', data);
                analytics.proxy('send', {
                    hitType: 'event',
                    eventCategory: 'Sponsors',
                    eventAction: 'open',
                });
            })
            .fail(err => {
                const result = err.responseText || JSON.stringify(err);
                alertSystem.alert(
                    'Compiler Explorer Sponsors',
                    '<div>Unable to fetch sponsors:</div><div>' + result + '</div>',
                );
            });
    });

    $('#ui-history').on('click', () => {
        historyWidget.run(data => {
            local.set('gl', JSON.stringify(data.config));
            hasUIBeenReset = true;
            window.history.replaceState(null, '', window.httpRoot);
            window.location.reload();
        });

        $('#history').modal();
    });

    $('#ui-apply-default-font-scale').on('click', () => {
        const defaultFontScale = Settings.getStoredSettings().defaultFontScale;
        if (defaultFontScale !== undefined) {
            eventHub.emit('broadcastFontScale', defaultFontScale);
        }
    });
}

function configFromEmbedded(embeddedUrl: string, defaultLangId: string) {
    // Old-style link?
    let params;
    try {
        params = url.unrisonify(embeddedUrl);
    } catch (e) {
        document.write(
            '<div style="padding: 10px; background: #fa564e; color: black;">' +
                "An error was encountered while decoding the URL for this embed. Make sure the URL hasn't been " +
                'truncated, otherwise if you believe your URL is valid please let us know on ' +
                '<a href="https://github.com/compiler-explorer/compiler-explorer/issues" style="color: black;">' +
                'our github' +
                '</a>.' +
                '</div>',
        );
        throw new Error('Embed url decode error');
    }
    if (params && params.source && params.compiler) {
        const filters = Object.fromEntries(((params.filters as string) || '').split(',').map(o => [o, true]));
        // TODO(jeremy-rifkin): Fix types
        return {
            content: [
                {
                    type: 'row',
                    content: [
                        Components.getEditorWith(1, params.source, filters as any, defaultLangId),
                        Components.getCompilerWith(1, filters as any, params.options, params.compiler),
                    ],
                },
            ],
        };
    } else {
        return url.deserialiseState(embeddedUrl);
    }
}

// TODO(jeremy-rifkin): Unsure of the type, just typing enough for `content` at the moment
function fixBugsInConfig(config: Record<string, any> & {content?: any[]}) {
    if (config.activeItemIndex && config.activeItemIndex >= unwrap(config.content).length) {
        config.activeItemIndex = unwrap(config.content).length - 1;
    }

    if (config.content) {
        for (const item of config.content) {
            fixBugsInConfig(item);
        }
    }
}

type ConfigType = {
    settings: {
        showPopoutIcon: boolean;
    };
    content: {
        type: string;
        content: (ComponentConfig<Partial<StateWithId & StateWithLanguage>> | ComponentConfig<EmptyCompilerState>)[];
    }[];
};

function findConfig(defaultConfig: ConfigType, options: CompilerExplorerOptions, defaultLangId: string) {
    let config;
    if (!options.embedded) {
        if (options.slides) {
            const presentation = new Presentation(unwrap(window.compilerExplorerOptions.slides).length);
            const currentSlide = presentation.currentSlide;
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
                    const alertSystem = new Alert();
                    alertSystem.alert(
                        'Decode Error',
                        'An error was encountered while decoding the URL, the last locally saved configuration will ' +
                            "be used if present.<br/><br/>Make sure the URL you're using hasn't been truncated, " +
                            'otherwise if you believe your URL is valid please let us know on ' +
                            '<a href="https://github.com/compiler-explorer/compiler-explorer/issues">our github</a>.',
                        {isError: true},
                    );
                }
            }

            if (config) {
                // replace anything in the default config with that from the hash
                config = _.extend(defaultConfig, config);
            }
            if (!config) {
                const savedState = local.get('gl', null);
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
            configFromEmbedded(window.location.hash.substring(1), defaultLangId),
        );
    }

    removeOrphanedMaximisedItemFromConfig(config);
    fixBugsInConfig(config);

    return config;
}

function initializeResetLayoutLink() {
    const currentUrl = document.URL;
    if (currentUrl.includes('/z/')) {
        $('#ui-brokenlink').attr('href', currentUrl.replace('/z/', '/resetlayout/')).show();
    } else {
        $('#ui-brokenlink').hide();
    }
}

function initPolicies(options: CompilerExplorerOptions) {
    if (options.policies.privacy.enabled) {
        if (jsCookie.get(options.policies.privacy.key) == null) {
            $('#privacy').trigger('click', {
                title: 'New Privacy Policy. Please take a moment to read it',
            });
        } else if (policyDocuments.privacy.hash !== jsCookie.get(options.policies.privacy.key)) {
            // When the user has already accepted the privacy, just show a pretty notification.
            const ppolicyBellNotification = $('#policyBellNotification');
            const pprivacyBellNotification = $('#privacyBellNotification');
            const pcookiesBellNotification = $('#cookiesBellNotification');
            ppolicyBellNotification.removeClass('d-none');
            pprivacyBellNotification.removeClass('d-none');
            $('#privacy').on('click', () => {
                // Only hide if the other policy does not also have a bell
                if (pcookiesBellNotification.hasClass('d-none')) {
                    ppolicyBellNotification.addClass('d-none');
                }
                pprivacyBellNotification.addClass('d-none');
            });
        }
    }
    simpleCooks.setOnDoConsent(() => {
        jsCookie.set(options.policies.cookies.key, policyDocuments.cookies.hash, {
            expires: 365,
            sameSite: 'strict',
        });
        analytics.toggle(true);
    });
    simpleCooks.setOnDontConsent(() => {
        analytics.toggle(false);
        jsCookie.set(options.policies.cookies.key, '', {
            sameSite: 'strict',
        });
    });
    simpleCooks.setOnHide(() => {
        const spolicyBellNotification = $('#policyBellNotification');
        const sprivacyBellNotification = $('#privacyBellNotification');
        const scookiesBellNotification = $('#cookiesBellNotification');
        // Only hide if the other policy does not also have a bell
        if (sprivacyBellNotification.hasClass('d-none')) {
            spolicyBellNotification.addClass('d-none');
        }
        scookiesBellNotification.addClass('d-none');
        $(window).trigger('resize');
    });
    // '' means no consent. Hash match means consent of old. Null means new user!
    const storedCookieConsent = jsCookie.get(options.policies.cookies.key);
    if (options.policies.cookies.enabled) {
        if (storedCookieConsent !== '' && policyDocuments.cookies.hash !== storedCookieConsent) {
            simpleCooks.show();
            const cpolicyBellNotification = $('#policyBellNotification');
            const cprivacyBellNotification = $('#privacyBellNotification');
            const ccookiesBellNotification = $('#cookiesBellNotification');
            cpolicyBellNotification.removeClass('d-none');
            ccookiesBellNotification.removeClass('d-none');
            $('#cookies').on('click', () => {
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

    let found = false as boolean;

    function impl(component) {
        if (component.id === '__glMaximised') {
            found = true;
            return;
        }

        if (component.content) {
            for (let i = 0; i < component.content.length; i++) {
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

function setupLanguageLogos(languages: Partial<Record<LanguageKey, Language>>) {
    for (const lang of Object.values(languages)) {
        try {
            if (lang.logoUrl !== null) {
                lang.logoData = logos('./' + lang.logoUrl);
                if (lang.logoUrlDark !== null) {
                    lang.logoDataDark = logos('./' + lang.logoUrlDark);
                }
            }
        } catch (ignored) {
            lang.logoData = '';
        }
    }
}

function earlyGetDefaultLangSetting() {
    return Settings.getStoredSettings().defaultLanguage;
}

function getDefaultLangId(subLangId: LanguageKey | undefined, options: CompilerExplorerOptions) {
    let defaultLangId = subLangId;
    if (!defaultLangId) {
        const defaultLangSetting = earlyGetDefaultLangSetting();
        if (defaultLangSetting && defaultLangSetting in options.languages) {
            defaultLangId = defaultLangSetting;
        } else if ('c++' in options.languages) {
            defaultLangId = 'c++';
        } else {
            defaultLangId = utils.keys(options.languages)[0];
        }
    }
    return defaultLangId;
}

// eslint-disable-next-line max-statements
function start() {
    initializeResetLayoutLink();
    setupSiteTemplateWidgetButton(siteTemplateScreenshots);

    const hostnameParts = window.location.hostname.split('.');
    let subLangId: LanguageKey | undefined = undefined;
    // Only set the subdomain lang id if it makes sense to do so
    if (hostnameParts.length > 0) {
        const subdomainPart = hostnameParts[0];
        const langBySubdomain = Object.values(options.languages).find(
            lang => lang.id === subdomainPart || lang.alias.includes(subdomainPart),
        );
        if (langBySubdomain) {
            subLangId = langBySubdomain.id;
        }
    }

    const defaultLangId = getDefaultLangId(subLangId, options);

    setupLanguageLogos(options.languages);

    // Cookie domains are matched as a RE against the window location. This allows a flexible
    // way that works across multiple domains (e.g. godbolt.org and compiler-explorer.com).
    // We allow this to be configurable so that (for example), gcc.godbolt.org and d.godbolt.org
    // share the same cookie domain for some settings.
    const cookieDomain = new RegExp(options.cookieDomainRe).exec(window.location.hostname);
    if (cookieDomain && cookieDomain[0]) {
        jsCookie = jsCookie.withAttributes({domain: cookieDomain[0]});
    }

    const defaultConfig = {
        settings: {showPopoutIcon: false},
        content: [
            {
                type: 'row',
                content: [Components.getEditor(defaultLangId, 1), Components.getCompiler(1, defaultLangId)],
            },
        ],
    };

    $(window).on('hashchange', () => {
        // punt on hash events and just reload the page if there's a hash
        if (window.location.hash.substring(1)) window.location.reload();
    });

    // Which buttons act as a linkable popup
    const linkablePopups = ['#ces', '#sponsors', '#changes', '#cookies', '#setting', '#privacy'];
    let hashPart = linkablePopups.includes(window.location.hash) ? window.location.hash : null;
    if (hashPart) {
        window.location.hash = '';
        // Handle the time we renamed sponsors to ces to work around issues with blockers.
        if (hashPart === '#sponsors') hashPart = '#ces';
    }

    const config = findConfig(defaultConfig, options, defaultLangId);

    const root = $('#root');

    let layout;
    let hub;
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
        const height = unwrap($(window).height()) - root.position().top - ($('#simplecook:visible').height() || 0);
        root.height(height);
        layout.updateSize();
    }

    $(window)
        .on('resize', sizeRoot)
        .on('beforeunload', () => {
            // Only preserve state in localStorage in non-embedded mode.
            const shouldSave = !window.hasUIBeenReset && !hasUIBeenReset;
            if (!options.embedded && !isMobileViewer() && shouldSave) {
                local.set('gl', JSON.stringify(layout.toConfig()));
            }
        });

    new clipboard('.btn.clippy');

    const settings = setupSettings(hub);

    // We assume no consent for embed users
    if (!options.embedded) {
        setupButtons(options, hub);
    }

    const addDropdown = $('#addDropdown');

    function setupAdd<C>(thing: JQuery, func: () => ComponentConfig<C>) {
        layout.createDragSource(thing, func)._dragListener.on('dragStart', () => {
            addDropdown.dropdown('toggle');
        });

        thing.on('click', () => {
            if (hub.hasTree()) {
                hub.addInEditorStackIfPossible(func());
            } else {
                hub.addAtRoot(func());
            }
        });
    }

    setupAdd($('#add-editor'), () => {
        return Components.getEditor(defaultLangId);
    });
    setupAdd($('#add-diff'), () => {
        return Components.getDiffView();
    });
    setupAdd($('#add-tree'), () => {
        $('#add-tree').prop('disabled', true);
        return Components.getTree();
    });

    if (hashPart) {
        const element = $(hashPart);
        element.trigger('click');
    }
    initPolicies(options);

    // Skip some steps if using embedded mode
    if (!options.embedded) {
        // Only fetch MOTD when not embedded.
        motd.initialise(
            options.motdUrl,
            $('#motd'),
            subLangId ?? '',
            settings.enableCommunityAds,
            data => {
                const sendMotd = () => {
                    hub.layout.eventHub.emit('motd', data);
                };
                hub.layout.eventHub.on('requestMotd', sendMotd);
                sendMotd();
            },
            () => {
                hub.layout.eventHub.emit('modifySettings', {
                    enableCommunityAds: false,
                });
            },
        );

        // Don't try to update Version tree link
        const release = window.compilerExplorerOptions.gitReleaseCommit;
        let versionLink = 'https://github.com/compiler-explorer/compiler-explorer/';
        if (release) {
            versionLink += 'tree/' + release;
        }
        $('#version-tree').prop('href', versionLink);
    }

    if (options.hideEditorToolbars) {
        $('[name="editor-btn-toolbar"]').addClass('d-none');
    }

    window.onSponsorClick = (sponsorUrl: string) => {
        analytics.proxy('send', {
            hitType: 'event',
            eventCategory: 'Sponsors',
            eventAction: 'click',
            eventLabel: sponsorUrl,
            transport: 'beacon',
        });
        window.open(sponsorUrl);
    };

    if (options.pageloadUrl) {
        setTimeout(() => {
            const visibleIcons = $('.ces-icon:visible')
                .map((_, value) => value.dataset.statsid)
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
