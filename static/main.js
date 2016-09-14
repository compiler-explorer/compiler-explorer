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
        lzstring: 'ext/lz-string/libs/lz-string'
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
    require('bootstrap');
    var analytics = require('analytics');
    var sharing = require('sharing');
    var _ = require('underscore');
    var $ = require('jquery');
    var GoldenLayout = require('goldenlayout');
    var compiler = require('compiler');
    var editor = require('editor');
    var url = require('url');
    var Hub = require('hub');

    analytics.initialise();
    sharing.initialise();

    var options = require('options');
    $('.language-name').text(options.language);
    var safeLang = options.language.toLowerCase().replace(/[^a-z_]+/g, '');
    var defaultSrc = $('.template.lang.' + safeLang).text().trim();
    var defaultConfig = {
        showPopoutIcon: false,
        content: [{type: 'row', content: [editor.getComponent(1), compiler.getComponent(1)]}]
    };
    var root = $("#root");
    var config = url.deserialiseState(window.location.hash.substr(1));
    $(window).bind('hashchange', function () {
        window.location.reload();  // punt on hash events and just reload the page
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
});