define(function (require) {
    'use strict';
    var $ = require('jquery');
    var monaco = require('monaco');
    var php = require('vs/basic-languages/src/php');

    monaco.languages.register({id: 'php'});
    monaco.languages.setLanguageConfiguration('php', php.conf);
    monaco.languages.setMonarchTokensProvider('php', php.language);
});
