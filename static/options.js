define(function (require) {
    "use strict";
    var $ = require('jquery');
    var options = $.ajax({type: "GET", url: 'client-options.json', async: false});
    return options.responseJSON;
});
