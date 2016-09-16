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

require.config({
    paths: {
        bootstrap: 'ext/bootstrap/dist/js/bootstrap.min',
        jquery: 'ext/jquery/dist/jquery.min',
        underscore: 'ext/underscore/underscore-min',
        goldenlayout: 'ext/golden-layout/dist/goldenlayout',
        selectize: 'ext/selectize/dist/js/selectize.min',
        sifter: 'ext/sifter/sifter.min',
        microplugin: 'ext/microplugin/src/microplugin',
        events: 'ext/eventEmitter/EventEmitter',
        lzstring: 'ext/lz-string/libs/lz-string',
        clipboard: 'ext/clipboard/dist/clipboard'
    },
    packages: [{
        name: "codemirror",
        location: "ext/codemirror",
        main: "lib/codemirror"
    }],
    shim: {
        underscore: {exports: '_'},
        bootstrap: ['jquery']
    }
});

define(function (require) {
    "use strict";
    require('bootstrap');
    var analytics = require('analytics');
    var sharing = require('sharing');
    var _ = require('underscore');
    var $ = require('jquery');
    var GoldenLayout = require('goldenlayout');
    var compiler = require('compiler');
    var editor = require('editor');
    var url = require('url');
    var clipboard = require('clipboard');
    var Hub = require('hub');
    var shortenURL = require('urlshorten-google');

    analytics.initialise();
    sharing.initialise();

    var options = require('options');
    $('.language-name').text(options.language);

    var safeLang = options.language.toLowerCase().replace(/[^a-z_]+/g, '');
    var defaultSrc = $('.template.lang.' + safeLang).text().trim();
    var defaultConfig = {
        settings: {showPopoutIcon: false},
        content: [{type: 'row', content: [editor.getComponent(1), compiler.getComponent(1)]}]
    };
    var root = $("#root");
    var config = url.deserialiseState(window.location.hash.substr(1));
    if (config) {
        // replace anything in the default config with that from the hash
        config = _.extend(defaultConfig, config);
    }
    $(window).bind('hashchange', function () {
        // punt on hash events and just reload the page if there's a hash
        if (window.location.hash.substr(1))
            window.location.reload();
    });

    if (!config) {
        // TODO: find old storage and convert
        var savedState = localStorage.getItem('gl');
        config = savedState !== null ? JSON.parse(savedState) : defaultConfig;
    }

    var layout = new GoldenLayout(config, root);
    layout.on('stateChanged', function () {
        var state = JSON.stringify(layout.toConfig());
        localStorage.setItem('gl', state);
    });

    var hub = new Hub(layout, defaultSrc);

    function sizeRoot() {
        var height = $(window).height() - root.position().top;
        root.height(height);
        layout.updateSize();
    }

    $(window).resize(sizeRoot);
    sizeRoot();

    new clipboard('.btn.clippy');

    function initPopover(getLink, provider) {
        var html = $('.urls.template').html();

        getLink.popover({
            container: 'body',
            content: html,
            html: true,
            placement: 'bottom',
            trigger: 'manual'
        }).click(function () {
            getLink.popover('show');
        }).on('inserted.bs.popover', function () {
            provider(function (url) {
                $(".permalink:visible").val(url);
            });
        });

        // Dismiss the popover on escape.
        $(document).on('keyup.editable', function (e) {
            if (e.which === 27) {
                getLink.popover("hide");
            }
        });

        // Dismiss on any click that isn't either on the opening element, or inside
        // the popover.
        $(document).on('click.editable', function (e) {
            var target = $(e.target);
            if (!target.is(getLink) && target.closest('.popover').length === 0)
                getLink.popover("hide");
        });
    }

    function permalink() {
        var config = layout.toConfig();
        return window.location.href.split('#')[0] + '#' + url.serialiseState(config);
    }

    initPopover($("#get-full-link"), function (done) {
        done(permalink)
    });
    initPopover($("#get-short-link"), function (done) {
        shortenURL(permalink(), done);
    });
});