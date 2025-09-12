// Copyright (c) 2021, Compiler Explorer Authors
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

import {Modal, Tooltip} from 'bootstrap';
import ClipboardJS from 'clipboard';
import GoldenLayout from 'golden-layout';
import $ from 'jquery';
import _ from 'underscore';
import {unwrap} from './assert.js';
import * as BootstrapUtils from './bootstrap-utils.js';
import {sessionThenLocalStorage} from './local.js';
import {options} from './options.js';
import {SentryCapture} from './sentry.js';
import {Settings, SiteSettings} from './settings.js';
import * as url from './url.js';

import ClickEvent = JQuery.ClickEvent;

import cloneDeep from 'lodash.clonedeep';

enum LinkType {
    Short = 0,
    Full = 1,
    Embed = 2,
}

const shareServices = {
    twitter: {
        embedValid: false,
        logoClass: 'fab fa-twitter',
        cssClass: 'share-twitter',
        getLink: (title: string, url: string) => {
            return (
                'https://twitter.com/intent/tweet' +
                `?text=${encodeURIComponent(title)}` +
                `&url=${encodeURIComponent(url)}` +
                '&via=CompileExplore'
            );
        },
        text: 'Tweet',
    },
    bluesky: {
        embedValid: false,
        logoClass: 'fab fa-bluesky',
        cssClass: 'share-bluesky',
        getLink: (title: string, url: string) => {
            const text = `${title} ${url} via @compiler-explorer.com`;
            return `https://bsky.app/intent/compose?text=${encodeURIComponent(text)}`;
        },
        text: 'Share on Bluesky',
    },
    reddit: {
        embedValid: false,
        logoClass: 'fab fa-reddit',
        cssClass: 'share-reddit',
        getLink: (title: string, url: string) => {
            return (
                'http://www.reddit.com/submit' +
                `?url=${encodeURIComponent(url)}` +
                `&title=${encodeURIComponent(title)}`
            );
        },
        text: 'Share on Reddit',
    },
};

export class Sharing {
    private layout: GoldenLayout;
    private lastState: any;

    private readonly share: JQuery;
    private readonly shareTooltipTarget: JQuery;
    private readonly shareShort: JQuery;
    private readonly shareFull: JQuery;
    private readonly shareEmbed: JQuery;

    private settings: SiteSettings;

    private clippyButton: ClipboardJS | null;
    private readonly shareLinkDialog: HTMLElement;

    constructor(layout: any) {
        this.layout = layout;
        this.lastState = null;
        this.shareLinkDialog = unwrap(document.getElementById('sharelinkdialog'), 'Share modal element not found');

        this.share = $('#share');
        this.shareTooltipTarget = $('#share-tooltip-target');
        this.shareShort = $('#shareShort');
        this.shareFull = $('#shareFull');
        this.shareEmbed = $('#shareEmbed');

        [this.shareShort, this.shareFull, this.shareEmbed].forEach(el => {
            el.on('click', e => BootstrapUtils.showModal(this.shareLinkDialog, e.currentTarget));
        });
        this.settings = Settings.getStoredSettings();

        this.clippyButton = null;

        this.initButtons();
        this.initCallbacks();
    }

    private initCallbacks(): void {
        this.layout.eventHub.on('displaySharingPopover', () => {
            this.openShareModalForType(LinkType.Short);
        });
        this.layout.eventHub.on('copyShortLinkToClip', () => {
            this.copyLinkTypeToClipboard(LinkType.Short);
        });
        this.layout.on('stateChanged', this.onStateChanged.bind(this));

        this.shareLinkDialog.addEventListener('show.bs.modal', this.onOpenModalPane.bind(this));
        this.shareLinkDialog.addEventListener('hidden.bs.modal', this.onCloseModalPane.bind(this));

        this.layout.eventHub.on('settingsChange', (newSettings: SiteSettings) => {
            this.settings = newSettings;
        });

        $(window).on('blur', async () => {
            sessionThenLocalStorage.set('gl', JSON.stringify(this.layout.toConfig()));
            if (this.settings.keepMultipleTabs) {
                try {
                    const link = await this.getLinkOfType(LinkType.Full);
                    window.history.replaceState(null, '', link);
                } catch (e) {
                    // This is probably caused by a link that is too long
                    SentryCapture(e, 'url update');
                }
            }
        });
    }

    private onStateChanged(): void {
        const config = Sharing.filterComponentState(this.layout.toConfig());
        this.ensureUrlIsNotOutdated(config);
        if (options.embedded) {
            const strippedToLast = window.location.pathname.substring(0, window.location.pathname.lastIndexOf('/') + 1);
            $('a.link').prop('href', strippedToLast + '#' + url.serialiseState(config));
        }
    }

    private ensureUrlIsNotOutdated(config: any): void {
        const stringifiedConfig = JSON.stringify(config);
        if (stringifiedConfig !== this.lastState) {
            if (this.lastState != null && window.location.pathname !== window.httpRoot) {
                window.history.replaceState(null, '', window.httpRoot);
            }
            this.lastState = stringifiedConfig;
        }
    }

    private static bindToLinkType(bind: string): LinkType {
        switch (bind) {
            case 'Full':
                return LinkType.Full;
            case 'Short':
                return LinkType.Short;
            case 'Embed':
                return LinkType.Embed;
            default:
                return LinkType.Full;
        }
    }

    private onOpenModalPane(event: Event): void {
        const modalEvent = event as Modal.Event;
        if (!modalEvent.relatedTarget) {
            throw new Error('No relatedTarget found in modal event');
        }

        const button = $(modalEvent.relatedTarget);
        const bindStr = button.data('bind') as string;
        const currentBind = Sharing.bindToLinkType(bindStr);
        const modal = $(event.currentTarget as HTMLElement);
        const socialSharingElements = modal.find('.socialsharing');
        const permalink = modal.find('.permalink');
        const embedsettings = modal.find('#embedsettings');
        const clipboardButton = modal.find('.clippy');

        const updatePermaLink = () => {
            socialSharingElements.empty();
            const config = this.layout.toConfig();
            Sharing.getLinks(config, currentBind, (error: any, newUrl: string, extra: string, updateState: boolean) => {
                permalink.off('click');
                if (error || !newUrl) {
                    clipboardButton.prop('disabled', true);
                    permalink.val(error || 'Error providing URL');
                    SentryCapture(error, 'Error providing url');
                } else {
                    if (updateState) {
                        Sharing.storeCurrentConfig(config, extra);
                    }
                    clipboardButton.prop('disabled', false);
                    permalink.val(newUrl);
                    permalink.on('click', () => {
                        permalink.trigger('focus').trigger('select');
                    });
                    if (options.sharingEnabled) {
                        Sharing.updateShares(socialSharingElements, newUrl);
                        // Disable the links for every share item which does not support embed html as links
                        if (currentBind === LinkType.Embed) {
                            socialSharingElements.children('.share-no-embeddable').hide().on('click', false);
                        }
                    }
                }
            });
        };

        const clippyElement = modal.find('button.clippy').get(0);
        if (clippyElement != null) {
            this.clippyButton = new ClipboardJS(clippyElement);
            this.clippyButton.on('success', e => {
                this.displayTooltip(permalink, 'Link copied to clipboard');
                e.clearSelection();
            });
            this.clippyButton.on('error', _e => {
                this.displayTooltip(permalink, 'Error copying to clipboard');
            });
        }

        if (currentBind === LinkType.Embed) {
            embedsettings.show();
            embedsettings
                .find('input')
                // Off any prev click handlers to avoid multiple events triggering after opening the modal more than once
                .off('click')
                .on('click', () => updatePermaLink());
        } else {
            embedsettings.hide();
        }

        updatePermaLink();
    }

    private onCloseModalPane(): void {
        if (this.clippyButton) {
            this.clippyButton.destroy();
            this.clippyButton = null;
        }
    }

    private initButtons(): void {
        const shareShortCopyToClipBtn = this.shareShort.find('.clip-icon');
        const shareFullCopyToClipBtn = this.shareFull.find('.clip-icon');
        const shareEmbedCopyToClipBtn = this.shareEmbed.find('.clip-icon');

        shareShortCopyToClipBtn.on('click', e => this.onClipButtonPressed(e, LinkType.Short));
        shareFullCopyToClipBtn.on('click', e => this.onClipButtonPressed(e, LinkType.Full));
        shareEmbedCopyToClipBtn.on('click', e => this.onClipButtonPressed(e, LinkType.Embed));

        if (options.sharingEnabled) {
            Sharing.updateShares($('#socialshare'), window.location.protocol + '//' + window.location.hostname);
        }
    }

    private onClipButtonPressed(event: ClickEvent, type: LinkType): boolean {
        // Don't let the modal show up.
        // We need this because the button is a child of the dropdown-item with a data-bs-toggle=modal
        if (Sharing.isNavigatorClipboardAvailable()) {
            this.copyLinkTypeToClipboard(type);
            event.stopPropagation();
            // As we prevented bubbling, the dropdown won't close by itself.
            BootstrapUtils.hideDropdown(this.share);
        }
        return false;
    }

    private getLinkOfType(type: LinkType): Promise<string> {
        const config = this.layout.toConfig();
        return new Promise<string>((resolve, reject) => {
            Sharing.getLinks(config, type, (error: any, newUrl: string, extra: string, updateState: boolean) => {
                if (error || !newUrl) {
                    this.displayTooltip(this.shareTooltipTarget, 'Oops, something went wrong');
                    SentryCapture(error, 'Getting short link failed');
                    reject(
                        new Error(
                            error
                                ? `Getting short link failed: ${error}`
                                : 'Getting short link failed: no URL returned',
                        ),
                    );
                } else {
                    if (updateState) {
                        Sharing.storeCurrentConfig(config, extra);
                    }
                    resolve(newUrl);
                }
            });
        });
    }

    private copyLinkTypeToClipboard(type: LinkType): void {
        const config = this.layout.toConfig();
        Sharing.getLinks(config, type, (error: any, newUrl: string, extra: string, updateState: boolean) => {
            if (error || !newUrl) {
                this.displayTooltip(this.shareTooltipTarget, 'Oops, something went wrong');
                SentryCapture(error, 'Getting short link failed');
            } else {
                if (updateState) {
                    Sharing.storeCurrentConfig(config, extra);
                }
                this.doLinkCopyToClipboard(type, newUrl);
            }
        });
    }

    // TODO we can consider using bootstrap's "Toast" support in future.
    private displayTooltip(where: JQuery, message: string): void {
        // First dispose any existing tooltip
        const tooltipEl = where[0];
        if (!tooltipEl) return;

        const existingTooltip = Tooltip.getInstance(tooltipEl);
        if (existingTooltip) {
            existingTooltip.dispose();
        }

        // Create and show new tooltip
        try {
            const tooltip = new Tooltip(tooltipEl, {
                placement: 'bottom',
                trigger: 'manual',
                title: message,
            });

            tooltip.show();

            // Manual triggering of tooltips does not hide them automatically. This timeout ensures they do
            setTimeout(() => tooltip.hide(), 1500);
        } catch (e) {
            // If element doesn't exist, just silently fail
            console.warn('Could not show tooltip:', e);
        }
    }

    private openShareModalForType(type: LinkType): void {
        switch (type) {
            case LinkType.Short:
                this.shareShort.trigger('click');
                break;
            case LinkType.Full:
                this.shareFull.trigger('click');
                break;
            case LinkType.Embed:
                this.shareEmbed.trigger('click');
                break;
        }
    }

    private doLinkCopyToClipboard(type: LinkType, link: string): void {
        if (Sharing.isNavigatorClipboardAvailable()) {
            navigator.clipboard
                .writeText(link)
                .then(() => this.displayTooltip(this.shareTooltipTarget, 'Link copied to clipboard'))
                .catch(() => this.openShareModalForType(type));
        } else {
            this.openShareModalForType(type);
        }
    }

    public static getLinks(config: any, currentBind: LinkType, done: CallableFunction): void {
        const root = window.httpRoot;
        switch (currentBind) {
            case LinkType.Short:
                Sharing.getShortLink(config, root, done);
                return;
            case LinkType.Full:
                done(null, window.location.origin + root + '#' + url.serialiseState(config), false);
                return;
            case LinkType.Embed: {
                const options: Record<string, boolean> = {};
                $('#sharelinkdialog input:checked').each((i, element) => {
                    options[$(element).data('option')] = true;
                });
                done(null, Sharing.getEmbeddedHtml(config, root, false, options), false);
                return;
            }
            default:
                // Hmmm
                done('Unknown link type', null);
        }
    }

    private static getShortLink(config: any, root: string, done: CallableFunction): void {
        const useExternalShortener = options.urlShortenService !== 'default';
        const data = JSON.stringify({
            config: useExternalShortener ? url.serialiseState(config) : config,
        });
        $.ajax({
            type: 'POST',
            url: window.location.origin + root + 'api/shortener',
            dataType: 'json', // Expected
            contentType: 'application/json', // Sent
            data: data,
            success: (result: any) => {
                const pushState = useExternalShortener ? null : result.url;
                done(null, result.url, pushState, true);
            },
            error: err => {
                // Notify the user that we ran into trouble?
                done(err.statusText, null, false);
            },
            cache: true,
        });
    }

    private static getEmbeddedHtml(
        config: any,
        root: string,
        isReadOnly: boolean,
        extraOptions: Record<string, boolean>,
    ): string {
        const embedUrl = Sharing.getEmbeddedUrl(config, root, isReadOnly, extraOptions);
        // The attributes must be double quoted, the full url's rison contains single quotes
        return `<iframe width="800px" height="200px" src="${embedUrl}"></iframe>`;
    }

    private static getEmbeddedUrl(config: any, root: string, readOnly: boolean, extraOptions: object): string {
        const location = window.location.origin + root;
        const parameters = _.reduce(
            extraOptions,
            (total, value, key): string => {
                if (total === '') {
                    total = '?';
                } else {
                    total += '&';
                }

                return total + key + '=' + value;
            },
            '',
        );

        const path = (readOnly ? 'embed-ro' : 'e') + parameters + '#';

        return location + path + url.serialiseState(config);
    }

    private static storeCurrentConfig(config: any, extra: string): void {
        window.history.pushState(null, '', extra);
    }

    private static isNavigatorClipboardAvailable(): boolean {
        return (navigator.clipboard as Clipboard | undefined) !== undefined;
    }

    public static filterComponentState(config: any, keysToRemove: [string] = ['selection']): any {
        function filterComponentStateImpl(component: any) {
            if (component.content) {
                for (let i = 0; i < component.content.length; i++) {
                    filterComponentStateImpl(component.content[i]);
                }
            }

            if (component.componentState) {
                Object.keys(component.componentState)
                    .filter(e => keysToRemove.includes(e))
                    .forEach(key => {
                        delete component.componentState[key];
                    });
            }
        }

        config = cloneDeep(config);
        filterComponentStateImpl(config);
        return config;
    }

    private static updateShares(container: JQuery, url: string): void {
        const baseTemplate = $('#share-item');
        _.each(shareServices, (service, serviceName) => {
            const newElement = baseTemplate.children('a.share-item').clone();
            if (service.logoClass) {
                newElement.prepend(
                    $('<span>').addClass('dropdown-icon me-1').addClass(service.logoClass).prop('title', serviceName),
                );
            }
            if (service.text) {
                newElement.children('span.share-item-text').text(service.text);
            }
            newElement
                .prop('href', service.getLink('Compiler Explorer', url))
                .addClass(service.cssClass)
                .toggleClass('share-no-embeddable', !service.embedValid)
                .appendTo(container);
        });
    }
}
