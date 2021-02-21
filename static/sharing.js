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

function initShareButtons(shortDomLink, fullDomLink, layout, noteNewState) {
    var baseUrl = window.location.protocol + '//' + window.location.hostname;
    var html = $('.template .urls').html();
    //TODO: Everything
    function init(domLink, currentBind) {
        domLink.popover({
            container: 'body',
            content: html,
            html: true,
            placement: 'top',
            sanitize: false,
        }).on('inserted.bs.popover', function () {
            ga.proxy('send', {
                hitType: 'event',
                eventCategory: 'OpenModalPane',
                eventAction: 'Sharing',
            });

            //var popoverElement = $($(this).data('bs.popover').tip);
            //var root = $('.urls-container:visible');
            var permalink = $('.permalink');

            permalink.prop('disabled', false);
            var config = layout.toConfig();
            permalink.val('');
            getLinks(config, currentBind, function (error, newUrl, extra, updateState) {
                if (error || !newUrl) {
                    permalink.prop('disabled', true);
                    permalink.val(error || 'Error providing URL');
                } else {
                    var result = {
                        updateState: updateState,
                        extra: extra,
                        url: newUrl,
                    };
                    if (result.updateState) {
                        noteNewState(config, result.extra);
                    }
                    permalink.val(result.url);
                }
            });
        });

        // Dismiss the popover on escape.
        $(document).on('keyup.editable', function (e) {
            if (e.which === 27) {
                domLink.popover('hide');
            }
        });

        // Dismiss on any click that isn't either in the opening element, inside
        // the popover or on any alert
        $(document).on('mouseup', function (e) {
            var target = $(e.target);
            if (!target.is(domLink) && domLink.has(target).length === 0 && target.closest('.popover').length === 0)
                domLink.popover('hide');
        });
    }

    init(fullDomLink, 'Full');
    init(shortDomLink, 'Short');

    // Opens the full link popup if asked to by the editor
    layout.eventHub.on('displaySharingPopover', function () {
        fullDomLink.popover('show');
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
    initShareButtons: initShareButtons,
    configFromEmbedded: configFromEmbedded,
};
