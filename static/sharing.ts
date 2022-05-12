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

import * as Sentry from '@sentry/browser';
import GoldenLayout from 'golden-layout';
import _ from 'underscore';
import ClipboardJS from 'clipboard';

import ClickEvent = JQuery.ClickEvent;
import TriggeredEvent = JQuery.TriggeredEvent;

const ga = require('./analytics').ga;
const options = require('./options').options;
const url = require('./url');
const cloneDeep = require('lodash.clonedeep');

enum LinkType {
    Short,
    Full,
    Embed,
}

const shareServices = {
    twitter: {
        embedValid: false,
        logoClass: 'fab fa-twitter',
        cssClass: 'share-twitter',
        getLink: (title, url) => {
            return (
                'https://twitter.com/intent/tweet' +
                `?text=${encodeURIComponent(title)}` +
                `&url=${encodeURIComponent(url)}` +
                '&via=CompileExplore'
            );
        },
        text: 'Tweet',
    },
    reddit: {
        embedValid: false,
        logoClass: 'fab fa-reddit',
        cssClass: 'share-reddit',
        getLink: (title, url) => {
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

    private share: JQuery;
    private shareShort: JQuery;
    private shareFull: JQuery;
    private shareEmbed: JQuery;

    private clippyButton: ClipboardJS | null;

    constructor(layout: any) {
        this.layout = layout;
        this.lastState = null;

        this.share = $('#share');
        this.shareShort = $('#shareShort');
        this.shareFull = $('#shareFull');
        this.shareEmbed = $('#shareEmbed');

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

        $('#sharelinkdialog')
            .on('show.bs.modal', this.onOpenModalPane.bind(this))
            .on('hidden.bs.modal', this.onCloseModalPane.bind(this));
    }

    private onStateChanged(): void {
        const config = Sharing.filterComponentState(this.layout.toConfig());
        this.ensureUrlIsNotOutdated(config);
        if (options.embedded) {
            const strippedToLast = window.location.pathname.substr(0, window.location.pathname.lastIndexOf('/') + 1);
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

    private onOpenModalPane(event: TriggeredEvent<HTMLElement, undefined, HTMLElement, HTMLElement>): void {
        // @ts-ignore The property is added by bootstrap
        const button = $(event.relatedTarget);
        const currentBind = Sharing.bindToLinkType(button.data('bind'));
        const modal = $(event.currentTarget);
        const socialSharingElements = modal.find('.socialsharing');
        const permalink = modal.find('.permalink');
        const embedsettings = modal.find('#embedsettings');

        const updatePermaLink = () => {
            socialSharingElements.empty();
            const config = this.layout.toConfig();
            Sharing.getLinks(config, currentBind, (error: any, newUrl: string, extra: string, updateState: boolean) => {
                permalink.off('click');
                if (error || !newUrl) {
                    permalink.prop('disabled', true);
                    permalink.val(error || 'Error providing URL');
                    Sentry.captureException(error);
                } else {
                    if (updateState) {
                        Sharing.storeCurrentConfig(config, extra);
                    }
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
            this.clippyButton.on('error', e => {
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

        ga.proxy('send', {
            hitType: 'event',
            eventCategory: 'OpenModalPane',
            eventAction: 'Sharing',
        });
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

    private onClipButtonPressed(event: ClickEvent, type: LinkType): void {
        // Don't let the modal show up.
        // We need this because the button is a child of the dropdown-item with a data-toggle=modal
        if (Sharing.isNavigatorClipboardAvailable()) {
            event.stopPropagation();
            this.copyLinkTypeToClipboard(type);
            // As we prevented bubbling, the dropdown won't close by itself. We need to trigger it manually
            this.share.dropdown('hide');
        }
    }

    private copyLinkTypeToClipboard(type: LinkType): void {
        const config = this.layout.toConfig();
        Sharing.getLinks(config, type, (error: any, newUrl: string, extra: string, updateState: boolean) => {
            if (error || !newUrl) {
                this.displayTooltip(this.share, 'Oops, something went wrong');
                Sentry.captureException(error);
            } else {
                if (updateState) {
                    Sharing.storeCurrentConfig(config, extra);
                }
                this.doLinkCopyToClipboard(type, newUrl);
            }
        });
    }

    private displayTooltip(where: JQuery, message: string): void {
        where.tooltip('dispose');
        where.tooltip({
            placement: 'bottom',
            trigger: 'manual',
            title: message,
        });
        where.tooltip('show');
        // Manual triggering of tooltips does not hide them automatically. This timeout ensures they do
        setTimeout(() => where.tooltip('hide'), 1500);
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
                .then(() => this.displayTooltip(this.share, 'Link copied to clipboard'))
                .catch(() => this.openShareModalForType(type));
        } else {
            this.openShareModalForType(type);
        }
    }

    public static getLinks(config: any, currentBind: LinkType, done: CallableFunction): void {
        const root = window.httpRoot;
        ga.proxy('send', {
            hitType: 'event',
            eventCategory: 'CreateShareLink',
            eventAction: 'Sharing',
        });
        switch (currentBind) {
            case LinkType.Short:
                Sharing.getShortLink(config, root, done);
                return;
            case LinkType.Full:
                done(null, window.location.origin + root + '#' + url.serialiseState(config), false);
                return;
            case LinkType.Embed: {
                const options = {};
                $('#sharelinkdialog input:checked').each((i, element) => {
                    options[$(element).prop('class')] = true;
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

    private static getEmbeddedHtml(config, root, isReadOnly, extraOptions): string {
        const embedUrl = Sharing.getEmbeddedUrl(config, root, isReadOnly, extraOptions);
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
            ''
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
                    .forEach(key => delete component.componentState[key]);
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
                    $('<span>').addClass('dropdown-icon mr-1').addClass(service.logoClass).prop('title', serviceName)
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
