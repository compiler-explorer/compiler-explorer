// Copyright (c) 2012-2017, Matt Godbolt
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

const _ = require('underscore-node');

const tabsRe = /\t/g;
const lineRe = /\r?\n/;

function splitLines(text) {
    const result = text.split(lineRe);
    if (result.length > 0 && result[result.length - 1] === '')
        return result.slice(0, result.length - 1);
    return result;
}

exports.splitLines = splitLines;

function eachLine(text, func, context) {
    return _.each(splitLines(text), func, context);
}

exports.eachLine = eachLine;

function expandTabs(line) {
    let extraChars = 0;
    return line.replace(tabsRe, function (match, offset) {
        const total = offset + extraChars;
        const spacesNeeded = (total + 8) & 7;
        extraChars += spacesNeeded - 1;
        return "        ".substr(spacesNeeded);
    });
}

exports.expandTabs = expandTabs;

function parseOutput(lines, inputFilename) {
    const re = /^\s*<source>[:(]([0-9]+)(:([0-9]+):)?[):]*\s*(.*)/;
    const result = [];
    eachLine(lines, function (line) {
        line = line.split('<stdin>').join('<source>');
        if (inputFilename) line = line.split(inputFilename).join('<source>');
        if (line !== "" && line.indexOf("fixme:") !== 0) {
            const lineObj = {text: line};
            const match = line.match(re);
            if (match) {
                lineObj.tag = {
                    line: parseInt(match[1]),
                    column: parseInt(match[3] || "0"),
                    text: match[4].trim()
                };
            }
            result.push(lineObj);
        }
    });
    return result;
}

exports.parseOutput = parseOutput;

function padRight(name, len) {
    while (name.length < len) name = name + ' ';
    return name;
}

exports.padRight = padRight;
