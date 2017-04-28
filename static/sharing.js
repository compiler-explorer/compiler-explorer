// Copyright (c) 2012-2017, Matt Godbolt
//
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

define(function (require) {
    "use strict";
    const $ = require('jquery');
    const _ = require('underscore');
    const options = require('options');
    const shortenURL = require('urlshorten-google');
    const Components = require('components');
    const url = require('url');

    function configFromEmbedded(embeddedUrl) {
        // Old-style link?
        let params;
        try {
            params = url.unrisonify(embeddedUrl);
        } catch (e) {
        }
        if (params && params.source && params.compiler) {
            const filters = _.chain((params.filters || "").split(','))
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

    /*function getItemsByComponent(layout, component) {
        return layout.root.getItemsByFilter(function (o) {
            return o.type === "component" && o.componentName === component;
        });
    }*/

    function getEmbeddedUrl(layout, readOnly) {
        let location = window.location.origin + window.location.pathname;
        if (location[location.length - 1] !== '/') location += '/';
        const path = readOnly ? 'embed-ro#' : 'e#';
        return location + path + url.serialiseState(layout.toConfig());
    }

    function initShareButton(getLink, layout) {
        const html = $('.template .urls').html();
        let currentBind = '';

        const title = getLink.attr('title'); // preserve before popover/tooltip breaks it

        getLink.popover({
            container: 'body',
            content: html,
            html: true,
            placement: 'bottom',
            trigger: 'manual'
        }).click(function () {
            getLink.popover('show');
        }).on('inserted.bs.popover', function () {
            const root = $('.urls-container:visible');
            let urls = {};
            if (!currentBind) currentBind = $(root.find('.sources a')[0]).data().bind;

            function update() {
                if (!currentBind) return;
                root.find('.current').text(currentBind);
                $(".permalink:visible").val(urls[currentBind] || "");
            }

            root.find('.sources a').on('click', function () {
                currentBind = $(this).data().bind;
                update();
            });
            getLinks(layout, function (theUrls) {
                urls = theUrls;
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
            const target = $(e.target);
            if (!target.is(getLink) && getLink.has(target).length === 0 && target.closest('.popover').length === 0)
                getLink.popover("hide");
        });
    }

    function permalink(layout) {
        return window.location.href.split('#')[0] + '#' + url.serialiseState(layout.toConfig());
    }

    function getLinks(layout, done) {
        const result = {
            Full: permalink(layout),
            Embed: '<iframe width="800px" height="200px" src="' + getEmbeddedUrl(layout, false) + '"></iframe>',
            'Embed (RO)': '<iframe width="800px" height="200px" src="' + getEmbeddedUrl(layout, true) + '"></iframe>'
        };
        if (!options.gapiKey) {
            done(result);
        } else {
            shortenURL(result.Full, function (shorter) {
                result.Short = shorter;
                done(result);
            });
        }
    }

    return {
        initShareButton: initShareButton,
        configFromEmbedded: configFromEmbedded
    };
});
