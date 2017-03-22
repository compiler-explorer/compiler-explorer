define(function (require) {
    "use strict";
    var child_process = require('child_process');

    function clangHandlder(req, res) {
        res.set('Content-Type', 'application/json');
        res.end(JSON.stringify(clangFormat(req.body.text)));
    }

    function clangFormat(text) {
        return child_process.execSync("clang-format").toString().trim();
    }

    return { 
        clangFormat: clangFormat,
        cppHandler: clangHandlder
    };
});
