require.config({
    paths: {
        bootstrap: 'ext/bootstrap/dist/js/bootstrap.min',
        jquery: 'ext/jquery/dist/jquery.min',
        underscore: 'ext/underscore/underscore-min',
        goldenlayout: 'ext/golden-layout/dist/goldenlayout',
        selectize: 'ext/selectize/dist/js/selectize.min',
        sifter: 'ext/sifter/sifter.min',
        microplugin: 'ext/microplugin/src/microplugin'
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
    var Hub = require('hub');

    analytics.initialise();
    sharing.initialise();

    var options = require('options');
    $('.language-name').text(options.language);
    var safeLang = options.language.toLowerCase().replace(/[^a-z_]+/g, '');
    var defaultSrc = $('.template.lang.' + safeLang).text().trim();
    var defaultConfig = {
        content: [{
            type: 'row',
            content: [
                {
                    type: 'component',
                    componentName: 'compilerOutput',
                    componentState: {source: 1}
                },
                {
                    type: 'component',
                    componentName: 'codeEditor',
                    componentState: {id: 1}
                },
                {
                    type: 'column', content: [
                    {
                        type: 'component',
                        componentName: 'compilerOutput',
                        componentState: {source: 1}
                    },
                    {
                        type: 'component',
                        componentName: 'compilerOutput',
                        componentState: {source: 1}
                    }
                ]
                }
            ]
        }]
    };
    var root = $("#root");
    // TODO: find old storage and convert
    var savedState = localStorage.getItem('gl');
    var config = savedState !== null ? JSON.parse(savedState) : defaultConfig;

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