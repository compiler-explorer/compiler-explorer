define(function (require) {
    'use strict';
    var jquery = require('jquery');
    var monaco = require('monaco');
    var cpp = require('vs/basic-languages/src/cpp');

    // We need to create a new definition for cpp so we can remove invalid keywords

    function definition() {
        var cppp = jquery.extend(true, {}, cpp.language); // deep copy

        function removeKeyword(keyword) {
            var index = cppp.keywords.indexOf(keyword);
            if (index > -1) {
                cppp.keywords.splice(index, 1);
            }
        }

        removeKeyword("array");
        removeKeyword("in");
        removeKeyword("interface");

        return cppp;
    }

    monaco.languages.register({id: 'cppp'});
    monaco.languages.setMonarchTokensProvider('cppp', definition());
});
