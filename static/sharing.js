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

'use strict';
var $ = require('jquery');
var _ = require('underscore');
var options = require('./options');
var url = require('./url');
var ga = require('./analytics');
var Sentry = require('@sentry/browser');
var cloneDeep = require('lodash.clonedeep');


var shareServices = {
    twitter: {
        embedValid: false,
        logoClass: 'fab fa-twitter',
        cssClass: 'share-twitter',
        getLink: function (title, url) {
            return 'https://twitter.com/intent/tweet?text=' +
                encodeURIComponent(title) + '&url=' + encodeURIComponent(url) + '&via=CompileExplore';
        },
        text: 'Tweet',
    },
    reddit: {
        embedValid: false,
        logoClass: 'fab fa-reddit',
        cssClass: 'share-reddit',
        getLink: function (title, url) {
            return 'http://www.reddit.com/submit?url=' +
                encodeURIComponent(url) + '&title=' + encodeURIComponent(title);
        },
        text: 'Share on Reddit',
    },
};

function filterComponentState(config, keysToRemove) {
    keysToRemove = keysToRemove || ['selection'];
    function filterComponentStateImpl(component) {
        if (component.content) {
            for (var i = 0; i < component.content.length; i++) {
                filterComponentStateImpl(component.content[i], keysToRemove);
            }
        }

        if (component.componentState) {
            Object.keys(component.componentState)
                .filter(function (key) {
                    return keysToRemove.includes(key);
                })
                .forEach(function (key) {
                    delete component.componentState[key];
                });
        }
    }

    config = cloneDeep(config);
    filterComponentStateImpl(config);
    return config;
}

function updateShares(container, url) {
    var baseTemplate = $('#share-item');
    _.each(shareServices, function (service, serviceName) {
        var newElement = baseTemplate.children('a.share-item').clone();
        if (service.logoClass) {
            newElement.prepend($('<span>')
                .addClass('dropdown-icon')
                .addClass(service.logoClass)
                .prop('title', serviceName)
            );
        }
        if (service.text) {
            newElement.children('span.share-item-text')
                .text(service.text);
        }
        newElement
            .prop('href', service.getLink('Compiler Explorer', url))
            .addClass(service.cssClass)
            .toggleClass('share-no-embeddable', !service.embedValid)
            .appendTo(container);
    });
}

function getEmbeddedUrl(config, root, readOnly, extraOptions) {
    var location = window.location.origin + root;
    var path = '';
    var parameters = '';

    _.forEach(extraOptions, function (value, key) {
        if (parameters === '') {
            parameters = '?';
        } else {
            parameters += '&';
        }

        parameters += key + '=' + value;
    });

    if (readOnly) {
        path = 'embed-ro' + parameters + '#';
    } else {
        path = 'e' + parameters + '#';
    }

    return location + path + url.serialiseState(config);
}

function getEmbeddedHtml(config, root, isReadOnly, extraOptions) {
    return '<iframe width="800px" height="200px" src="' +
        getEmbeddedUrl(config, root, isReadOnly, extraOptions) + '"></iframe>';
}

function getShortLink(config, root, done) {
    var useExternalShortener = options.urlShortenService !== 'default';
    var data = JSON.stringify({
        config: useExternalShortener ? url.serialiseState(config) : config,
    });
    $.ajax({
        type: 'POST',
        url: window.location.origin + root + 'api/shortener',
        dataType: 'json',  // Expected
        contentType: 'application/json',  // Sent
        data: data,
        success: _.bind(function (result) {
            var pushState = useExternalShortener ? null : result.url;
            done(null, result.url, pushState, true);
        }, this),
        error: _.bind(function (err) {
            // Notify the user that we ran into trouble?
            done(err.statusText, null, false);
        }, this),
        cache: true,
    });
}

function getLinks(config, currentBind, done) {
    var root = window.httpRoot;
    ga.proxy('send', {
        hitType: 'event',
        eventCategory: 'CreateShareLink',
        eventAction: 'Sharing',
    });
    switch (currentBind) {
        case 'Short':
            getShortLink(config, root, done);
            return;
        case 'Full':
            done(null, window.location.origin + root + '#' + url.serialiseState(config), false);
            return;
        default:
            if (currentBind.substr(0, 5) === 'Embed') {
                var options = {};
                $('#sharelinkdialog input:checked').each(function () {
                    options[$(this).prop('class')] = true;
                });
                done(null, getEmbeddedHtml(config, root, false, options), false);
                return;
            }
            // Hmmm
            done('Unknown link type', null);
    }
}

function Sharing(layout) {
    this.layout = layout;
    this.lastState = null;

    this.share = $('#share');
    this.shareShort = $('#shareShort');
    this.shareFull = $('#shareFull');
    this.shareEmbed = $('#shareEmbed');

    this.initButtons();
    this.initCallbacks();
}

Sharing.prototype.initButtons = function () {
    this.shareShortCopyToClipBtn = this.shareShort.find('.clip-icon');
    this.shareFullCopyToClipBtn = this.shareFull.find('.clip-icon');
    this.shareEmbedCopyToClipBtn = this.shareEmbed.find('.clip-icon');

    if (this.areClipboardOperationSupported()) {
        var onClipButtonPressed = _.bind(function (e, type) {
            // Dont let the modal show up.
            // We need this because the button is a child of the dropdown-item with a data-toggle=modal
            e.stopPropagation();
            this.copyLinkTypeToClipboard(type);
            // As we prevented bubbling, the dropdown won't close by itself. We need to trigger it manually
            this.share.dropdown('hide');
        }, this);

        this.shareShortCopyToClipBtn.on('click', _.bind(function (e) {
            onClipButtonPressed(e, 'Short');
        }, this));
        this.shareFullCopyToClipBtn.on('click', _.bind(function (e) {
            onClipButtonPressed(e, 'Full');
        }, this));
        this.shareEmbedCopyToClipBtn.on('click', _.bind(function (e) {
            onClipButtonPressed(e, 'Embed');
        }, this));
    } else {
        this.shareShortCopyToClipBtn.hide();
        this.shareFullCopyToClipBtn.hide();
        this.shareEmbedCopyToClipBtn.hide();
    }
};

Sharing.prototype.onOpenModalPane = function (event) {
    var button = $(event.relatedTarget);
    var currentBind = button.data('bind');
    var modal = $(event.currentTarget);
    var socialSharingElements = modal.find('.socialsharing');
    var permalink = modal.find('.permalink');
    var embedsettings = modal.find('#embedsettings');

    function updatePermaLink() {
        socialSharingElements.empty();
        var config = this.layout.toConfig();
        getLinks(config, currentBind, _.bind(function (error, newUrl, extra, updateState) {
            if (error || !newUrl) {
                permalink.prop('disabled', true);
                permalink.val(error || 'Error providing URL');
                Sentry.captureException(error);
            } else {
                if (updateState) {
                    this.storeCurrentConfig(config, extra);
                }
                permalink.val(newUrl);
                if (options.sharingEnabled) {
                    updateShares(socialSharingElements, newUrl);
                    // Disable the links for every share item which does not support embed html as links
                    if (currentBind === 'Embed') {
                        socialSharingElements.children('.share-no-embeddable')
                            .addClass('share-disabled')
                            .prop('title', 'Embed links are not supported in this service')
                            .on('click', false);
                    }
                }
            }
        }, this));
    }

    if (currentBind === 'Embed') {
        embedsettings.show();
        embedsettings.find('input')
            // Off any prev click handlers to avoid multiple events triggering after opening the modal more than once
            .off('click')
            .on('click', _.bind(function () {
                updatePermaLink.apply(this);
            }, this));
    } else {
        embedsettings.hide();
    }

    updatePermaLink.apply(this);

    ga.proxy('send', {
        hitType: 'event',
        eventCategory: 'OpenModalPane',
        eventAction: 'Sharing',
    });
};

Sharing.prototype.onStateChanged = function () {
    var config = filterComponentState(this.layout.toConfig());
    this.ensureUrlIsNotOutdated(config);
    if (options.embedded) {
        var strippedToLast = window.location.pathname;
        strippedToLast = strippedToLast.substr(0, strippedToLast.lastIndexOf('/') + 1);
        $('a.link').attr('href', strippedToLast + '#' + url.serialiseState(config));
    }
};

// This keeps the URL from displaying a short link of an outdated page state
Sharing.prototype.ensureUrlIsNotOutdated = function (config) {
    var stringifiedConfig = JSON.stringify(config);
    if (stringifiedConfig !== this.lastState) {
        if (this.lastState != null && window.location.pathname !== window.httpRoot) {
            window.history.replaceState(null, null, window.httpRoot);
        }
        this.lastState = stringifiedConfig;
    }
};

Sharing.prototype.initCallbacks = function () {
    this.layout.eventHub.on('displaySharingPopover', _.bind(function () {
        this.shareShort.trigger('click');
    }, this));
    this.layout.on('stateChanged', _.bind(this.onStateChanged, this));

    $('#sharelinkdialog').on('show.bs.modal', _.bind(this.onOpenModalPane, this));
};

Sharing.prototype.displayTooltip = function (message) {
    this.share.tooltip('dispose');
    this.share.tooltip({
        placement: 'bottom',
        trigger: 'manual',
        title: message,
    });
    this.share.tooltip('show');
    // Manual triggering of tooltips does not hide them automatically. This timeout ensures they do
    setTimeout(_.bind(function () {
        this.share.tooltip('hide');
    }, this), 1500);
};

Sharing.prototype.copyLinkTypeToClipboard = function (type) {
    var config = this.layout.toConfig();
    getLinks(config, type, _.bind(function (error, newUrl, extra, updateState) {
        if (error || !newUrl) {
            this.displayTooltip('Oops, something went wrong');
            Sentry.captureException(error);
        } else {
            if (updateState) {
                this.storeCurrentConfig(config, extra);
            }
            this.doLinkCopyToClipboard(newUrl);
        }
    }, this));
};

Sharing.prototype.doLinkCopyToClipboard = function (link) {
    // TODO: Add more ways for users to be able to copy the link text
    // Right now, only the newer navigator.clipboard is available, but more can be added
    if (this.isNavigatorClipboardAvailable()) {
        navigator.clipboard.writeText(link)
            .then(_.bind(function () {
                this.displayTooltip('Link copied to clipboard');
            }, this))
            .catch(_.bind(function () {
                this.displayTooltip('Error copying link to clipboard');
            }, this));
    }
};

Sharing.prototype.storeCurrentConfig = function (config, extra) {
    window.history.pushState(null, null, extra);
};

// True if there's at least one way to copy a link to the user's clipboard.
// Currently, only navigator.clipboard is supported
Sharing.prototype.areClipboardOperationSupported = function () {
    return this.isNavigatorClipboardAvailable();
};

Sharing.prototype.isNavigatorClipboardAvailable = function () {
    return navigator.clipboard != null;
};

module.exports = {
    Sharing: Sharing,
    filterComponentState: filterComponentState,
};
