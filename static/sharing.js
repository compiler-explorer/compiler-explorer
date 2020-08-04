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
// POSSIBILITY OF SUCH DAMAGE.

'use strict';
var $ = require('jquery');
var _ = require('underscore');
var options = require('./options');
var Components = require('./components');
var url = require('./url');
var ga = require('./analytics');

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

function initShareButton(getLink, layout, noteNewState) {
    var baseUrl = window.location.protocol + '//' + window.location.hostname;
    var html = $('.template .urls').html();
    var currentNode = null;
    // Explicit because webstorm gets confused about the type of this variable.
    /***
     * Current URL bind
     * @type {string}
     */
    var currentBind = '';
    var title = getLink.prop('title'); // preserve before popover/tooltip breaks it

    getLink.popover({
        container: 'body',
        content: html,
        html: true,
        placement: 'bottom',
        trigger: 'manual',
        sanitize: false,
    }).click(function () {
        getLink.popover('toggle');
    }).on('inserted.bs.popover', function () {
        ga.proxy('send', {
            hitType: 'event',
            eventCategory: 'OpenModalPane',
            eventAction: 'Sharing',
        });
        var popoverElement = $($(this).data('bs.popover').tip);
        var socialSharingElements = popoverElement.find('.socialsharing');
        var root = $('.urls-container:visible');
        var label = root.find('.current');
        var permalink = $('.permalink');
        var urls = {};
        if (!currentNode) currentNode = $(root.find('.sources button')[0]);
        if (!currentBind) currentBind = currentNode.data().bind;

        function setCurrent(node) {
            currentNode = node;
            currentBind = node.data().bind;
        }

        function setSocialSharing(element, sharedUrl) {
            if (options.sharingEnabled) {
                updateShares(element, sharedUrl);
                // Disable the links for every share item which does not support embed html as links
                if (currentBind !== 'Full' && currentBind !== 'Short') {
                    element.children('.share-no-embeddable')
                        .addClass('share-disabled')
                        .prop('title', 'Embed links are not supported in this service')
                        .on('click', false);
                }
            }
        }

        function onUpdate(socialSharing, config, bind, result) {
            if (result.updateState) {
                noteNewState(config, result.extra);
            }
            label.text(bind);
            permalink.val(result.url);
            setSocialSharing(socialSharing, result.url);
        }

        function getEmbeddedCacheLinkId() {
            if ($('#shareembedlink input:checked').length === 0) return 'Embed';

            return 'Embed|' + $('#shareembedlink input:checked').map(function () {
                return $(this).prop('class');
            })
                .get()
                .join();
        }

        function update() {
            var socialSharing = socialSharingElements;
            socialSharing.empty();
            if (!currentBind) return;
            permalink.prop('disabled', false);
            var config = layout.toConfig();
            var cacheLinkId = currentBind;
            if (currentBind === 'Embed') {
                cacheLinkId = getEmbeddedCacheLinkId();
            }
            if (!urls[cacheLinkId]) {
                label.text(currentNode.text());
                permalink.val('');
                getLinks(config, currentBind, function (error, newUrl, extra, updateState) {
                    if (error || !newUrl) {
                        permalink.prop('disabled', true);
                        permalink.val(error || 'Error providing URL');
                    } else {
                        urls[cacheLinkId] = {
                            updateState: updateState,
                            extra: extra,
                            url: newUrl,
                        };
                        onUpdate(socialSharing, config, currentBind, urls[cacheLinkId]);
                    }
                });
            } else {
                onUpdate(socialSharing, config, currentBind, urls[cacheLinkId]);
            }
        }

        root.find('.sources button').on('click', function () {
            setCurrent($(this));
            update();
        });

        var embeddedButton = $('.shareembed');
        embeddedButton.on('click', function () {
            setCurrent(embeddedButton);
            update();
            getLink.popover('hide');
        });

        $('#embedsettings input').off('click').on('click', function () {
            setCurrent(embeddedButton);
            update();
        });

        update();
    }).prop('title', title);

    // Dismiss the popover on escape.
    $(document).on('keyup.editable', function (e) {
        if (e.which === 27) {
            getLink.popover('hide');
        }
    });

    // Dismiss on any click that isn't either in the opening element, inside
    // the popover or on any alert
    $(document).on('mouseup', function (e) {
        var target = $(e.target);
        if (!target.is(getLink) && getLink.has(target).length === 0 && target.closest('.popover').length === 0)
            getLink.popover('hide');
    });

    // Opens the popup if asked to by the editor
    layout.eventHub.on('displaySharingPopover', function () {
        getLink.popover('show');
    });

    if (options.sharingEnabled) {
        updateShares($('#socialshare'), baseUrl);
    }
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
        config: useExternalShortener
            ? url.serialiseState(config)
            : config,
    });
    $.ajax({
        type: 'POST',
        url: window.location.origin + root + 'shortener',
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
                $('#shareembedlink input:checked').each(function () {
                    options[$(this).prop('class')] = true;
                });
                done(null, getEmbeddedHtml(config, root, false, options), false);
                return;
            }
            // Hmmm
            done('Unknown link type', null);
    }
}

module.exports = {
    initShareButton: initShareButton,
    configFromEmbedded: configFromEmbedded,
};
