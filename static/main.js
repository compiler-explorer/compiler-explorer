require.config({
    paths: {
        bootstrap: 'ext/bootstrap/dist/js/bootstrap.min',
        jquery: 'ext/jquery/dist/jquery.min',
        underscore: 'ext/underscore/underscore-min',
        goldenlayout: 'ext/golden-layout/dist/goldenlayout.min'
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
    var Editor = require('editor');

    analytics.initialise();
    sharing.initialise();

    function codeEditorFactory(container, state) {
        var template = $('#codeEditor');
        var options = state.options;
        container.getElement().html(template.html());
        return new Editor(container, options.language);
    }

    var options = require('options');
    var config = {
        content: [{
            type: 'row',
            content: [{
                type: 'component',
                componentName: 'codeEditor',
                componentState: {options: options}
            }]
        }]
    };
    var myLayout = new GoldenLayout(config, $("#root")[0]);
    myLayout.registerComponent('codeEditor', codeEditorFactory);
    myLayout.init();
});