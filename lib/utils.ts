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
import {fileURLToPath} from 'url';

import fs from 'fs-extra';
import {ComponentConfig, ItemConfigType} from 'golden-layout';
import semverParser from 'semver';
import {parse as quoteParse} from 'shell-quote';
import _ from 'underscore';

import {CacheableValue} from '../types/cache.interfaces';
import {ResultLine} from '../types/resultline/resultline.interfaces';

const tabsRe = /\t/g;
const lineRe = /\r?\n/;

export function splitLines(text: string): string[] {
    if (!text) return [];
    const result = text.split(lineRe);
    if (result.length > 0 && result[result.length - 1] === '') return result.slice(0, -1);
    return result;
}

export function eachLine(text: string, func: (line: string) => ResultLine | void): (ResultLine | void)[] {
    return splitLines(text).map(func);
}

export function expandTabs(line: string): string {
    let extraChars = 0;
    return line.replace(tabsRe, (match, offset) => {
        const total = offset + extraChars;
        const spacesNeeded = (total + 8) & 7;
        extraChars += spacesNeeded - 1;
        return '        '.substr(spacesNeeded);
    });
}

export function maskRootdir(filepath: string): string {
    if (filepath) {
        // todo: make this compatible with local installations and windows etc
        return filepath.replace(/^\/tmp\/compiler-explorer-compiler[\w\d-.]*\//, '/app/').replace(/^\/app\//, '');
    } else {
        return filepath;
    }
}

const ansiColoursRe = /\x1B\[[\d;]*[Km]/g;

function _parseOutputLine(line: string, inputFilename?: string, pathPrefix?: string) {
    line = line.split('<stdin>').join('<source>');
    if (pathPrefix) line = line.replace(pathPrefix, '');
    if (inputFilename) {
        line = line.split(inputFilename).join('<source>');

        if (inputFilename.indexOf('./') === 0) {
            line = line.split('/home/ubuntu/' + inputFilename.substring(2)).join('<source>');
            line = line.split('/home/ce/' + inputFilename.substring(2)).join('<source>');
        }
    }
    return line;
}

function parseSeverity(message: string): number {
    if (message.startsWith('warning')) return 2;
    if (message.startsWith('note')) return 1;
    return 3;
}

const SOURCE_RE = /^\s*<source>[(:](\d+)(:?,?(\d+):?)?[):]*\s*(.*)/;
const SOURCE_WITH_FILENAME = /^\s*([\w.]*)[(:](\d+)(:?,?(\d+):?)?[):]*\s*(.*)/;

export function parseOutput(lines: string, inputFilename?: string, pathPrefix?: string): ResultLine[] {
    const result: ResultLine[] = [];
    eachLine(lines, line => {
        line = _parseOutputLine(line, inputFilename, pathPrefix);
        if (!inputFilename) {
            line = maskRootdir(line);
        }
        if (line !== null) {
            const lineObj: ResultLine = {text: line};
            const filteredline = line.replace(ansiColoursRe, '');
            let match = filteredline.match(SOURCE_RE);
            if (match) {
                const message = match[4].trim();
                lineObj.tag = {
                    line: parseInt(match[1]),
                    column: parseInt(match[3] || '0'),
                    text: message,
                    severity: parseSeverity(message),
                    file: inputFilename ? path.basename(inputFilename) : undefined,
                };
            } else {
                match = filteredline.match(SOURCE_WITH_FILENAME);
                if (match) {
                    const message = match[5].trim();
                    lineObj.tag = {
                        file: match[1],
                        line: parseInt(match[2]),
                        column: parseInt(match[4] || '0'),
                        text: message,
                        severity: parseSeverity(message),
                    };
                }
            }
            result.push(lineObj);
        }
    });
    return result;
}

export function parseRustOutput(lines: string, inputFilename?: string, pathPrefix?: string) {
    const re = /^ --> <source>[(:](\d+)(:?,?(\d+):?)?[):]*\s*(.*)/;
    const result: ResultLine[] = [];
    eachLine(lines, line => {
        line = _parseOutputLine(line, inputFilename, pathPrefix);
        if (line !== null) {
            const lineObj: ResultLine = {text: line};
            const match = line.replace(ansiColoursRe, '').match(re);

            if (match) {
                const line = parseInt(match[1]);
                const column = parseInt(match[3] || '0');

                const previous = result.pop();
                if (previous !== undefined) {
                    const text = previous.text.replace(ansiColoursRe, '');
                    previous.tag = {
                        line,
                        column,
                        text,
                        severity: parseSeverity(text),
                    };
                    result.push(previous);
                }

                lineObj.tag = {
                    line,
                    column,
                    text: '', // Left empty so that it does not show up in the editor
                    severity: 3,
                };
            }
            result.push(lineObj);
        }
    });
    return result;
}

export function padRight(name: string, len: number): string {
    while (name.length < len) name = name + ' ';
    return name;
}

export function trimRight(name: string): string {
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
export function anonymizeIp(ip: string): string {
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
function objectToHashableString(object: CacheableValue): string {
    // See https://stackoverflow.com/questions/899574/which-is-best-to-use-typeof-or-instanceof/6625960#6625960
    return typeof object === 'string' ? object : JSON.stringify(object);
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
export function getBinaryHash(object: CacheableValue, HashVersion = DefaultHash): Buffer {
    return crypto.createHmac('sha256', HashVersion).update(objectToHashableString(object)).digest();
}

/***
 * Gets the hash (as a hex string) of the given object
 *
 * Limitation: object shall not have circular references
 * @param {*} object - Object to get hash from
 * @param {string} [HashVersion=DefaultHash] - Hash "version" key
 * @returns {string} - Hash of object
 */
export function getHash(object: CacheableValue, HashVersion = DefaultHash): string {
    return crypto.createHmac('sha256', HashVersion).update(objectToHashableString(object)).digest('hex');
}

interface glEditorMainContent {
    // Editor content
    source: string;
    // Editor syntax language
    language: string;
}

interface glCompilerMainContent {
    // Compiler id
    compiler: string;
}

interface glContents {
    editors: glEditorMainContent[];
    compilers: glCompilerMainContent[];
}

/***
 * Gets every (source, lang) & (compilerId) available
 * @param {Array} content - GoldenLayout config topmost content field
 * @returns {glContents}
 */
export function glGetMainContents(content: ItemConfigType[] = []): glContents {
    const contents: glContents = {editors: [], compilers: []};
    _.each(content, element => {
        if (element.type === 'component') {
            const component = element as ComponentConfig;
            if (component.componentName === 'codeEditor') {
                contents.editors.push({
                    source: component.componentState.source,
                    language: component.componentState.lang,
                });
            } else if (component.componentName === 'compiler') {
                contents.compilers.push({
                    compiler: component.componentState.compiler,
                });
            }
        } else {
            const subComponents = glGetMainContents(element.content);
            contents.editors = contents.editors.concat(subComponents.editors);
            contents.compilers = contents.compilers.concat(subComponents.compilers);
        }
    });
    return contents;
}

export function squashHorizontalWhitespace(line: string, atStart = true): string {
    if (line.trim().length === 0) {
        return '';
    }
    const splat = line.split(/\s+/);
    if (splat[0] === '' && atStart) {
        // An indented line: preserve a two-space indent (max)
        const intent = line[1] === ' ' ? '  ' : ' ';
        return intent + splat.slice(1).join(' ');
    }
    return splat.join(' ');
}

export function toProperty(prop: string): boolean | number | string {
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
export function replaceAll(line: string, oldValue: string, newValue: string): string {
    if (oldValue.length === 0) return line;
    let startPoint = 0;
    for (;;) {
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
export function base32Encode(buffer: Buffer): string {
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
        digest = digest >>> 5;
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

// Splits a : separated list into its own array, or to default if input is undefined
export function splitIntoArray(input?: string, defaultArray: string[] = []): string[] {
    if (input === undefined) {
        return defaultArray;
    } else {
        return input.split(':');
    }
}

export function splitArguments(options = ''): string[] {
    // escape hashes first, otherwise they're interpreted as comments
    const escapedOptions = options.replaceAll(/#/g, '\\#');
    return _.chain(quoteParse(escapedOptions).map((x: any) => (typeof x === 'string' ? x : (x.pattern as string))))
        .compact()
        .value();
}

/***
 * Absolute path to the root of the application
 */
export const APP_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

export function resolvePathFromAppRoot(...args: string[]) {
    return path.resolve(APP_ROOT, ...args);
}

export async function fileExists(filename: string): Promise<boolean> {
    try {
        const stat = await fs.stat(filename);
        return stat.isFile();
    } catch {
        return false;
    }
}

export async function dirExists(dir: string): Promise<boolean> {
    try {
        const stat = await fs.stat(dir);
        return stat.isDirectory();
    } catch {
        return false;
    }
}

export function countOccurrences<T>(collection: Iterable<T>, item: T): number {
    // _.reduce(collection, (total, value) => value === item ? total + 1 : total, 0) would work, but is probably slower
    let result = 0;
    for (const element of collection) {
        if (element === item) {
            result++;
        }
    }
    return result;
}

export function asSafeVer(semver: string | number | null | undefined) {
    if (semver != null) {
        if (typeof semver === 'number') {
            semver = `${semver}`;
        }
        const splits = semver.split(' ');
        if (splits.length > 0) {
            let interestingPart = splits[0];
            let dotCount = countOccurrences(interestingPart, '.');
            for (; dotCount < 2; dotCount++) {
                interestingPart += '.0';
            }
            const validated: string | null = semverParser.valid(interestingPart, true);
            if (validated != null) {
                return validated;
            }
        }
    }
    return '9999999.99999.999';
}
