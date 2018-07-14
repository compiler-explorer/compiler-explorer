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

"use strict";
var $ = require('jquery');
var _ = require('underscore');
var options = require('options');
var shortenURL = require('./urlshorten-' + options.urlShortenService);
var Components = require('components');
var url = require('./url');
var local = require('./local');

var shareServices = {
    twitter: {
        embedValid: false,
        cssClass: 'share-twitter',
        getLink: function (title, url) {
            return "https://twitter.com/intent/tweet?text=" +
                encodeURIComponent(title) + '&url=' + encodeURIComponent(url) + '&via=mattgodbolt';
        },
        text: 'Tweet'
    },
    reddit: {
        embedValid: false,
        cssClass: 'share-reddit',
        getLink: function (title, url) {
            return 'http://www.reddit.com/submit?url=' +
                encodeURIComponent(url) + '&title=' + encodeURIComponent(title);
        },
        text: 'Share on Reddit'
    }
};

/**
 * Has the user already seen the current version of the privacy policy? (Also true if the policy is disabled)
 * @returns {boolean}
 */
function hasUserSeenPrivacyPolicy() {
    return !options.policies.privacy.enabled ||
        options.policies.privacy.hash === local.get(options.policies.privacy.key);
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
        var filters = _.chain((params.filters || "").split(','))
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
                        Components.getCompilerWith(1, filters, params.options, params.compiler)
                    ]
                }
            ]
        };
    } else {
        return url.deserialiseState(embeddedUrl);
    }
}

function getEmbeddedUrl(layout, readOnly) {
    var location = window.location.origin + window.location.pathname;
    if (location[location.length - 1] !== '/') location += '/';
    var path = readOnly ? 'embed-ro#' : 'e#';
    return location + path + url.serialiseState(layout.toConfig());
}

function updateShares(container, url) {
    var baseTemplate = $('#share-item');
    _.each(shareServices, function (service, serviceName) {
        var newElement = baseTemplate.children('a.share-item').clone();
        var logoPath = baseTemplate.data('logo-' + serviceName);
        if (logoPath) {
            newElement.children('img.share-item-logo')
                .prop('src', logoPath);
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

function initShareButton(getLink, layout) {
    var html = $('.template .urls').html();
    var currentBind = '';
    var title = getLink.attr('title'); // preserve before popover/tooltip breaks it

    getLink.popover({
        container: 'body',
        content: html,
        html: true,
        placement: 'bottom',
        trigger: 'manual'
    }).click(function () {
        getLink.popover('show');
    }).on('inserted.bs.popover', function () {
        if (!hasUserSeenPrivacyPolicy()) {
            // Passs a slightly
            $('#privacy').trigger('click', {title: 'New Privacy Policy. Please take a moment to read it'});
        }
        var root = $('.urls-container:visible');
        var urls = {Short: 'Loading...'};
        if (!currentBind) currentBind = $(root.find('.sources a')[0]).data().bind;

        function update() {
            var url = currentBind ? urls[currentBind] || "" : "";
            if (options.sharingEnabled) {
                // Is there a better way to get the popup object? There's a field with the element in getLink, but
                // it's under a jQueryXXXXXX field which I guess changes name for each version?
                var popoverId = '#' + getLink.attr('aria-describedby');
                var socialSharing = $(popoverId).find('.socialsharing');
                socialSharing.empty();
                updateShares(socialSharing, url || "https://godbolt.org");
                // Disable the links for every share item which does not support embed html as links
                if (!(currentBind === 'Full' || currentBind === 'Short')) {
                    socialSharing.children('.share-no-embeddable')
                        .addClass('share-disabled')
                        .prop('title', 'Embed links are not supported in this service')
                        .on('click', false);
                }
            }
            if (!currentBind) return;
            root.find('.current').text(currentBind);
            $(".permalink:visible").val(url);
        }

        root.find('.sources a').on('click', function () {
            currentBind = $(this).data().bind;
            update();
        });

        getLinks(layout, function (theUrls) {
            urls = theUrls;
            if (!urls[currentBind])
                currentBind = 'Full';
            root.find('.sources a').each(function () {
                $(this).toggle(!!urls[$(this).data().bind]);
            });
            update();
        });
        update();
    }).attr('title', title);

    // Dismiss the popover on escape.
    $(document).on('keyup.editable', function (e) {
        if (e.which === 27) {
            getLink.popover("hide");
        }
    });

    // Dismiss on any click that isn't either in the opening element, inside
    // the popover or on any alert
    $(document).on('click.editable', function (e) {
        var target = $(e.target);
        if (!target.is(getLink) && getLink.has(target).length === 0 && target.closest('.popover').length === 0)
            getLink.popover("hide");
    });

    if (options.sharingEnabled) {
        updateShares($('#socialshare'), 'https://godbolt.org');
    }
}

function permalink(layout) {
    return window.location.href.split('#')[0] + '#' + url.serialiseState(layout.toConfig());
}

function getLinks(layout, done) {
    var result = {
        Full: permalink(layout),
        Embed: '<iframe width="800px" height="200px" src="' + getEmbeddedUrl(layout, false) + '"></iframe>',
        'Embed (RO)': '<iframe width="800px" height="200px" src="' + getEmbeddedUrl(layout, true) + '"></iframe>'
    };
    shortenURL(result.Full, function (shorter) {
        result.Short = shorter;
        done(result);
    });
}

module.exports = {
    initShareButton: initShareButton,
    configFromEmbedded: configFromEmbedded
};
