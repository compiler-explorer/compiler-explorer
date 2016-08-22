require.config({
    paths: {
        bootstrap: 'ext/bootstrap/dist/js/bootstrap.min',
        jquery: 'ext/jquery/dist/jquery.min',
        underscore: 'ext/underscore/underscore-min',
        goldenlayout: 'ext/golden-layout/dist/goldenlayout.min',
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
    var config = {
        content: [{
            type: 'row',
            content: [{
                type: 'component',
                componentName: 'codeEditor',
                componentState: {}
            }, {
                type: 'component',
                componentName: 'compilerOutput',
                componentState: {}
            }]
        }]
    };
    var root = $("#root");
    var layout = new GoldenLayout(config, root);
    var hub = new Hub(layout);
    function sizeRoot() {
        var height = $(window).height() - root.position().top;
        root.height(height);
        layout.updateSize();
    }
    $(window).resize(sizeRoot);
    sizeRoot();
});