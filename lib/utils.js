// Copyright (c) 2016, Compiler Explorer Authors
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

import crypto from 'crypto';
import path from 'path';
import { fileURLToPath } from 'url';

import fs from 'fs-extra';
import quote from 'shell-quote';
import _ from 'underscore';

const tabsRe = /\t/g;
const lineRe = /\r?\n/;

/***
 *
 * @param {string} text
 * @returns {string[]}
 */
export function splitLines(text) {
    if (!text) return [];
    const result = text.split(lineRe);
    if (result.length > 0 && result[result.length - 1] === '')
        return result.slice(0, -1);
    return result;
}

/***
 * @callback eachLineFunc
 * @param {string} line
 * @returns {*}
 */

/***
 *
 * @param {string} text
 * @param {eachLineFunc} func
 * @param {*} [context]
 */
export function eachLine(text, func, context) {
    return _.each(splitLines(text), func, context);
}

/***
 *
 * @param {string} line
 * @returns {string}
 */
export function expandTabs(line) {
    let extraChars = 0;
    return line.replace(tabsRe, function (match, offset) {
        const total = offset + extraChars;
        const spacesNeeded = (total + 8) & 7;
        extraChars += spacesNeeded - 1;
        return '        '.substr(spacesNeeded);
    });
}

export function maskRootdir(filepath) {
    if (filepath) {
        // todo: make this compatible with local installations and windows etc
        return filepath.replace(/^\/tmp\/compiler-explorer-compiler[\w\d-.]*\//, '/app/').replace(/^\/app\//, '');
    } else {
        return filepath;
    }
}

/***
 * @typedef {Object} lineTag
 * @property {string} text
 * @property {number} line
 * @property {number} text
 */

/***
 * @typedef {Object} lineObj
 * @property {string} text
 * @property {lineTag} [tag]
 * @inner
 */

const ansiColoursRe = /\x1B\[[\d;]*[Km]/g;

function _parseOutputLine(line, inputFilename, pathPrefix) {
    line = line.split('<stdin>').join('<source>');
    if (pathPrefix) line = line.replace(pathPrefix, '');
    if (inputFilename) {
        line = line.split(inputFilename).join('<source>');

        if (inputFilename.indexOf('./') === 0) {
            line = line.split('/home/ubuntu/' + inputFilename.substr(2)).join('<source>');
            line = line.split('/home/ce/' + inputFilename.substr(2)).join('<source>');
        }
    }
    return line;
}

/***
 *
 * @param lines
 * @param inputFilename
 * @param pathPrefix
 * @returns {lineObj[]}
 */
export function parseOutput(lines, inputFilename, pathPrefix) {
    const re = /^\s*<source>[(:](\d+)(:?,?(\d+):?)?[):]*\s*(.*)/;
    const reWithFilename = /^\s*([\w.]*)[(:](\d+)(:?,?(\d+):?)?[):]*\s*(.*)/;
    const result = [];
    eachLine(lines, line => {
        line = _parseOutputLine(line, inputFilename, pathPrefix);
        if (!inputFilename) {
            line = maskRootdir(line);
        }
        if (line !== null) {
            const lineObj = {text: line};
            const filteredline = line.replace(ansiColoursRe, '');
            let match = filteredline.match(re);
            if (match) {
                lineObj.tag = {
                    line: parseInt(match[1]),
                    column: parseInt(match[3] || '0'),
                    text: match[4].trim(),
                };
            } else {
                match = filteredline.match(reWithFilename);
                if (match) {
                    lineObj.tag = {
                        file: match[1],
                        line: parseInt(match[2]),
                        column: parseInt(match[4] || '0'),
                        text: match[5].trim(),
                    };
                }
            }
            result.push(lineObj);
        }
    });
    return result;
}

/***
 *
 * @param lines
 * @param inputFilename
 * @param pathPrefix
 * @returns {lineObj[]}
 */
export function parseRustOutput(lines, inputFilename, pathPrefix) {
    const re = /^ --> <source>[(:](\d+)(:?,?(\d+):?)?[):]*\s*(.*)/;
    const result = [];
    eachLine(lines, line => {
        line = _parseOutputLine(line, inputFilename, pathPrefix);
        if (line !== null) {
            const lineObj = {text: line};
            const match = line.replace(ansiColoursRe, '').match(re);

            if (match) {
                const line = parseInt(match[1]);
                const column = parseInt(match[3] || '0');

                const previous = result.pop();
                previous.tag = {
                    line,
                    column,
                    text: previous.text.replace(ansiColoursRe, ''),
                };
                result.push(previous);

                lineObj.tag = {
                    line,
                    column,
                    text: '', // Left empty so that it does not show up in the editor
                };
            }
            result.push(lineObj);
        }
    });
    return result;
}

/***
 *
 * @param {string} name
 * @param {number} len
 * @returns {string}
 */
export function padRight(name, len) {
    while (name.length < len) name = name + ' ';
    return name;
}

/***
 *
 * @param {string} name
 * @returns {string}
 */
export function trimRight(name) {
    let l = name.length;
    while (l > 0 && name[l - 1] === ' ') l -= 1;
    return name.substr(0, l);
}

/***
 * Anonymizes given IP.
 * For IPv4, it removes the last octet
 * For IPv6, it removes the last three hextets
 *
 * @param {string} ip - IP string, of either type (localhost|IPv4|IPv6)
 * @returns {string} Anonymized IP
 */
export function anonymizeIp(ip) {
    if (ip.includes('localhost')) {
        return ip;
    } else if (ip.includes(':')) {
        // IPv6
        return ip.replace(/(?::[\dA-Fa-f]{0,4}){3}$/, ':0:0:0');
    } else {
        // IPv4
        return ip.replace(/\.\d{1,3}$/, '.0');
    }
}

/***
 *
 * @param {*} object
 * @returns {string}
 */
function objectToHashableString(object) {
    // See https://stackoverflow.com/questions/899574/which-is-best-to-use-typeof-or-instanceof/6625960#6625960
    return (typeof (object) === 'string') ? object : JSON.stringify(object);
}

const DefaultHash = 'Compiler Explorer Default Version 1';

/***
 * Gets the hash (as a binary buffer) of the given object
 *
 * Limitation: object shall not have circular references
 * @param {*} object - Object to get hash from
 * @param {string} [HashVersion=DefaultHash] - Hash "version" key
 * @returns {Buffer} - Hash of object
 */
export function getBinaryHash(object, HashVersion = DefaultHash) {
    return crypto.createHmac('sha256', HashVersion)
        .update(objectToHashableString(object))
        .digest();
}

/***
 * Gets the hash (as a hex string) of the given object
 *
 * Limitation: object shall not have circular references
 * @param {*} object - Object to get hash from
 * @param {string} [HashVersion=DefaultHash] - Hash "version" key
 * @returns {string} - Hash of object
 */
export function getHash(object, HashVersion = DefaultHash) {
    return crypto.createHmac('sha256', HashVersion)
        .update(objectToHashableString(object))
        .digest('hex');
}

/***
 * @typedef {Object} glEditorMainContent
 * @property {string} source - Editor content
 * @property {string} language - Editor syntax language
 * @inner
 */

/***
 * @typedef {Object} glCompilerMainContent
 * @property {string} compiler - Compiler id
 * @inner
 */

/***
 * @typedef {Object} glContents
 * @property {glEditorMainContent[]} editors
 * @property {glCompilerMainContent[]} compilers
 * @inner
 */

/***
 * Gets every (source, lang) & (compilerId) available
 * @param {Array} content - GoldenLayout config topmost content field
 * @returns {glContents}
 */
export function glGetMainContents(content) {
    let contents = {editors: [], compilers: []};
    _.each(content, element => {
        if (element.type === 'component') {
            if (element.componentName === 'codeEditor') {
                contents.editors.push({source: element.componentState.source, language: element.componentState.lang});
            } else if (element.componentName === 'compiler') {
                contents.compilers.push({compiler: element.componentState.compiler});
            }
        } else {
            const subComponents = glGetMainContents(element.content);
            contents.editors = contents.editors.concat(subComponents.editors);
            contents.compilers = contents.compilers.concat(subComponents.compilers);
        }
    });
    return contents;
}

/***
 *
 * @param {string} line
 * @param {boolean} [atStart=true]
 * @returns {string}
 */
export function squashHorizontalWhitespace(line, atStart) {
    if (atStart === undefined) atStart = true;
    if (line.trim().length === 0) {
        return '';
    }
    const splat = line.split(/\s+/);
    if (splat[0] === '' && atStart) {
        // An indented line: preserve a two-space indent (max)
        const intent = line[1] !== ' ' ? ' ' : '  ';
        return intent + splat.slice(1).join(' ');
    }
    return splat.join(' ');
}

/***
 *
 * @param {string} prop
 * @returns {boolean|number|string}
 */
export function toProperty(prop) {
    if (prop === 'true' || prop === 'yes') return true;
    if (prop === 'false' || prop === 'no') return false;
    if (/^-?(0|[1-9]\d*)$/.test(prop)) return parseInt(prop);
    if (/^-?\d*\.\d+$/.test(prop)) return parseFloat(prop);
    return prop;
}

/***
 * This function replaces all the "oldValues" in line with "newValue". It handles overlapping string replacement cases,
 * and is careful to return the exact same line object if there's no matches. This turns out to be super important for
 * performance.
 * @param {string} line
 * @param {string} oldValue
 * @param {string} newValue
 * @returns {string}
 */
export function replaceAll(line, oldValue, newValue) {
    if (oldValue.length === 0) return line;
    let startPoint = 0;
    for (; ;) {
        const index = line.indexOf(oldValue, startPoint);
        if (index === -1) break;
        line = line.substr(0, index) + newValue + line.substr(index + oldValue.length);
        startPoint = index + newValue.length;
    }
    return line;
}

// Initially based on http://philzimmermann.com/docs/human-oriented-base-32-encoding.txt
const BASE32_ALPHABET = '13456789EGKMPTWYabcdefhjnoqrsvxz';

/***
 *  A Base32 encoding implementation valid for our needs.
 *  Of importance, note that there's no padding
 * @param {Buffer} buffer
 * @returns {string}
 */
export function base32Encode(buffer) {
    let output = '';
    // This can grow up to 12 bits
    let digest = 0;
    // How many bits are we actually using from digest
    let bits = 0;

    function encodeNewWord() {
        // Get first 5 bits
        const word = digest & 0b11111;
        const character = BASE32_ALPHABET[word];
        output += character;
        bits -= 5;
        // Shift out the newly processed word
        digest = (digest >>> 5);
    }

    for (const byte of buffer) {
        // Append current byte to digest
        digest = (byte << bits) | digest;
        bits += 8;

        // Add characters until we run out of bits
        while (bits >= 5) {
            encodeNewWord();
        }
    }
    // Do we need a last pass?
    if (bits !== 0) {
        encodeNewWord();
    }

    return output;
}

export function splitArguments(options) {
    return _.chain(quote.parse(options || '')
        .map(x => typeof (x) === 'string' ? x : x.pattern))
        .compact()
        .value();
}

/***
 * Absolute path to the root of the application
 */
export const APP_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

export function resolvePathFromAppRoot(...args) {
    return path.resolve(APP_ROOT, ...args);
}

export async function fileExists(filename) {
    try {
        const stat = await fs.stat(filename);
        return stat.isFile();
    } catch {
        return false;
    }
}

export async function dirExists(dir) {
    try {
        const stat = await fs.stat(dir);
        return stat.isDirectory();
    } catch {
        return false;
    }
}
