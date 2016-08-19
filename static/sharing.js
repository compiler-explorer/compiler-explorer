define(function (require, exports) {
    "use strict";
    var $ = require('jquery');
    var options = require('options');
    exports.initialise = function () {
        $(function () {
            if (!options.sharingEnabled)
                $('.if-share-enabled').remove();

            if (!options.githubEnabled)
                $('.if-github-enabled').remove();
        });

        // TODO: url shortening service choice here
        // was in analytics:\
        // create_script_element('urlshortener', 'urlshorten-' + options.urlshortener + ".js");

    };
});