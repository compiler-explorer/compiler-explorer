// Copyright (c) 2012, Matt Godbolt
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

const props = require('../properties.js'),
    path = require('path'),
    fs = require('fs'),
    _ = require('underscore');
const basePath = props.get('builtin', 'sourcePath', './examples/');
const replacer = new RegExp('_', 'g');
const examples = _.flatten(
    fs.readdirSync(basePath)
        .map(folder => {
            const folderPath = path.join(basePath, folder);
            return fs.readdirSync(folderPath)
                .map(file => {
                    const filePath = path.join(folderPath, file);
                    const fileName = path.parse(file).name;
                    return {lang: folder, name: fileName.replace(replacer, ' '), path: filePath, file: fileName};
                })
                .filter(descriptor => descriptor.name !== "default")
                .sort((x, y) => x.name.localeCompare(y.name));
        }));

function load(lang, filename) {
    const example = _.find(examples, example => example.lang === lang && example.file === filename);
    return new Promise(resolve => {
        if (!example) resolve({file: 'No path found'});
        fs.readFile(example.path, 'utf-8', (err, res) => {
            resolve({file: err ? 'Could not read file' : res});
        });
    });
}

function list() {
    return Promise.resolve(examples.map(example => {
        return {file: example.file, name: example.name, lang: example.lang};
    }));
}

module.exports.load = load;
module.exports.save = null;
module.exports.list = list;
module.exports.name = "Examples";
module.exports.urlpart = "builtin";
