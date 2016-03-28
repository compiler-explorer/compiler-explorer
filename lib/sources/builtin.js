// Copyright (c) 2012-2016, Matt Godbolt
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

(function () {

    var props = require('../properties.js'),
        path = require('path'),
        fs = require('fs');

    var sourcePath = props.get('builtin', 'sourcepath', './examples/c++');
    var sourceMatch = new RegExp(props.get('builtin', 'extensionRe', '.*\\.cpp$'));
    var examples = fs.readdirSync(sourcePath)
        .filter(function (file) {
            return file.match(sourceMatch);
        })
        .map(function (file) {
            var nicename = file.replace(/\.cpp$/, '');
            return {urlpart: nicename, name: nicename.replace(/_/g, ' '), path: path.join(sourcePath, file)};
        }).sort(function (x, y) {
            return x.name.localeCompare(y.name);
        });
    var byUrlpart = {};
    examples.forEach(function (e) {
        byUrlpart[e.urlpart] = e.path;
    });

    function load(filename, callback) {
        var path = byUrlpart[filename];
        if (!path) {
            callback("No such path");
            return;
        }
        fs.readFile(path, 'utf-8', function (err, res) {
            if (err) {
                callback(err);
                return;
            }
            callback(null, {file: res});
        });
    }

    function list(callback) {
        callback(null, examples.map(function (example) {
            return {urlpart: example.urlpart, name: example.name};
        }));
    }

    exports.load = load;
    exports.save = null;
    exports.list = list;
    exports.name = "Examples";
    exports.urlpart = "builtin";

}).call(this);
