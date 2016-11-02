// Copyright (c) 2012-2016, Matt Godbolt
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
    var $ = require('jquery');
    var _ = require('underscore');
    var options = require('options');
    var shortenURL = require('urlshorten-google');
    var Components = require('components');
    var url = require('url');

    function contentFromEmbedded(embeddedUrl) {
        var params = url.unrisonify(embeddedUrl);
        var filters = _.chain((params.filters || "").split(','))
            .map(function (o) {
                return [o, true];
            })
            .object()
            .value();
        return [
            {
                type: 'row',
                content: [
                    Components.getEditorWith(1, params.source, filters),
                    Components.getCompilerWith(1, filters, params.options, params.compiler)
                ]
            }
        ];
    }

    function getItemsByComponent(layout, component) {
        return layout.root.getItemsByFilter(function (o) {
            return o.type === "component" && o.componentName === component;
        });
    }

    function getEmbeddedUrl(layout) {
        var source = "";
        var filters = {};
        var compilerName = "";
        var options = "";
        _.each(getItemsByComponent(layout, Components.getEditor().componentName),
            function (editor) {
                var state = editor.config.componentState;
                source = state.source;
                filters = _.extend(filters, state.options);
            });
        _.each(getItemsByComponent(layout, Components.getCompiler().componentName),
            function (compiler) {
                var state = compiler.config.componentState;
                compilerName = state.compiler;
                options = state.options;
                filters = _.extend(filters, state.filters);
            });
        if (!filters.compileOnChange)
            filters.readOnly = true;
        return window.location.origin + '/e#' + url.risonify({
                filters: _.keys(filters).join(","),
                source: source,
                compiler: compilerName,
                options: options
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
            var root = $('.urls-container:visible');
            var urls = {};
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

        // Dismiss on any click that isn't either in the opening element, or inside
        // the popover.
        $(document).on('click.editable', function (e) {
            var target = $(e.target);
            if (!target.is(getLink) && getLink.has(target).length === 0 && target.closest('.popover').length === 0)
                getLink.popover("hide");
        });
    }

    function permalink(layout) {
        return window.location.href.split('#')[0] + '#' + url.serialiseState(layout.toConfig());
    }

    function getLinks(layout, done) {
        var result = {
            Full: permalink(layout),
            Embed: '<iframe width="800px" height="200px" src="' + getEmbeddedUrl(layout) + '"></iframe>'
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

    function initialise() {
        if (!options.sharingEnabled)
            $('.if-share-enabled').remove();
        if (!options.githubEnabled)
            $('.if-github-enabled').remove();
        if (!options.gapiKey)
            $('.get-short-link').remove();
    }

    return {
        initialise: initialise,
        initShareButton: initShareButton,
        contentFromEmbedded: contentFromEmbedded
    };
});
