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

import {Buffer} from 'buffer';
import crypto from 'crypto';
import os from 'os';
import path from 'path';
import {fileURLToPath} from 'url';

import fs from 'fs-extra';
import {ComponentConfig, ItemConfigType} from 'golden-layout';
import semverParser from 'semver';
import _ from 'underscore';

import type {CacheableValue} from '../types/cache.interfaces.js';
import {BasicExecutionResult, UnprocessedExecResult} from '../types/execution/execution.interfaces.js';
import {LanguageKey} from '../types/languages.interfaces.js';
import type {ResultLine} from '../types/resultline/resultline.interfaces.js';

const tabsRe = /\t/g;
const lineRe = /\r?\n/;

export const ce_temp_prefix = 'compiler-explorer-compiler';

export function splitLines(text: string): string[] {
    if (!text) return [];
    const result = text.split(lineRe);
    if (result.length > 0 && result[result.length - 1] === '') return result.slice(0, -1);
    return result;
}

/**
 * Applies a function to each line of text split by `splitLines`
 */
export function eachLine(text: string, func: (line: string) => void): void {
    for (const line of splitLines(text)) {
        func(line);
    }
}

export function expandTabs(line: string): string {
    let extraChars = 0;
    return line.replaceAll(tabsRe, (match, offset) => {
        const total = offset + extraChars;
        const spacesNeeded = (total + 8) & 7;
        extraChars += spacesNeeded - 1;
        return '        '.substring(spacesNeeded);
    });
}

function getRegexForTempdir(): RegExp {
    const tmp = os.tmpdir();
    return new RegExp(tmp.replaceAll('/', '\\/') + '\\/' + ce_temp_prefix + '[\\w\\d-.]*\\/');
}

/**
 * Removes the root dir from the given filepath, so that it will match to the user's filenames used
 *  note: will keep /app/ if instead of filepath something like '-I/tmp/path' is used
 */
export function maskRootdir(filepath: string): string {
    if (filepath) {
        if (process.platform === 'win32') {
            // todo: should also use temp_prefix here
            return filepath
                .replace(/^C:\/Users\/[\w\d-.]*\/AppData\/Local\/Temp\/compiler-explorer-compiler[\w\d-.]*\//, '/app/')
                .replace(/^\/app\//, '');
        } else {
            const re = getRegexForTempdir();
            return filepath.replace(re, '/app/').replace(/^\/app\//, '');
        }
    } else {
        return filepath;
    }
}

export function changeExtension(filename: string, newExtension: string): string {
    const lastDot = filename.lastIndexOf('.');
    if (lastDot === -1) return filename + newExtension;
    return filename.substring(0, lastDot) + newExtension;
}

const ansiColoursRe = /\x1B\[[\d;]*[Km]/g;
const terminalHyperlinkEscapeRe = /\x1B]8;;.*?(\x1B\\|\x07)(.*?)\x1B]8;;\1/g;

function filterEscapeSequences(line: string): string {
    return line.replaceAll(ansiColoursRe, '').replaceAll(terminalHyperlinkEscapeRe, '$2');
}

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
const SOURCE_WITH_FILENAME = /^\s*([\w.]+)[(:](\d+)(:?,?(\d+):?)?[):]*\s*(.*)/;
const ATFILELINE_RE = /\s*at ([\w-/.]+):(\d+)/;

export enum LineParseOption {
    SourceMasking,
    RootMasking,
    SourceWithLineMessage,
    FileWithLineMessage,
    AtFileLine,
}

export type LineParseOptions = LineParseOption[];

export const DefaultLineParseOptions = [
    LineParseOption.SourceMasking,
    LineParseOption.RootMasking,
    LineParseOption.SourceWithLineMessage,
    LineParseOption.FileWithLineMessage,
];

function applyParse_SourceWithLine(lineObj: ResultLine, filteredLine: string, inputFilename?: string) {
    const match = filteredLine.match(SOURCE_RE);
    if (match) {
        const message = match[4].trim();
        lineObj.tag = {
            line: parseInt(match[1]),
            column: parseInt(match[3] || '0'),
            text: message,
            severity: parseSeverity(message),
            file: inputFilename ? path.basename(inputFilename) : undefined,
        };
    }
}

function applyParse_FileWithLine(lineObj: ResultLine, filteredLine: string) {
    const match = filteredLine.match(SOURCE_WITH_FILENAME);
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

function applyParse_AtFileLine(lineObj: ResultLine, filteredLine: string) {
    const match = filteredLine.match(ATFILELINE_RE);
    if (match) {
        if (match[1].startsWith('/app/')) {
            lineObj.tag = {
                file: match[1].replace(/^\/app\//, ''),
                line: parseInt(match[2]),
                column: 0,
                text: filteredLine,
                severity: 3,
            };
        } else if (!match[1].startsWith('/')) {
            lineObj.tag = {
                file: match[1],
                line: parseInt(match[2]),
                column: 0,
                text: filteredLine,
                severity: 3,
            };
        }
    }
}

export function parseOutput(
    lines: string,
    inputFilename?: string,
    pathPrefix?: string,
    options: LineParseOptions = DefaultLineParseOptions,
): ResultLine[] {
    const result: ResultLine[] = [];
    eachLine(lines, line => {
        if (options.includes(LineParseOption.SourceMasking)) {
            line = _parseOutputLine(line, inputFilename, pathPrefix);
        }
        if (!inputFilename && options.includes(LineParseOption.RootMasking)) {
            line = maskRootdir(line);
        }
        if (line !== null) {
            const lineObj: ResultLine = {text: line};
            const filteredLine = filterEscapeSequences(line);

            if (options.includes(LineParseOption.SourceWithLineMessage))
                applyParse_SourceWithLine(lineObj, filteredLine, inputFilename);

            if (!lineObj.tag && options.includes(LineParseOption.FileWithLineMessage))
                applyParse_FileWithLine(lineObj, filteredLine);

            if (!lineObj.tag && options.includes(LineParseOption.AtFileLine))
                applyParse_AtFileLine(lineObj, filteredLine);

            result.push(lineObj);
        }
    });
    return result;
}

export function parseRustOutput(lines: string, inputFilename?: string, pathPrefix?: string) {
    const re = /^\s+-->\s+<source>[(:](\d+)(:?,?(\d+):?)?[):]*\s*(.*)/;
    const result: ResultLine[] = [];
    eachLine(lines, line => {
        line = _parseOutputLine(line, inputFilename, pathPrefix);
        if (line !== null) {
            const lineObj: ResultLine = {text: line};
            const match = filterEscapeSequences(line).match(re);

            if (match) {
                const line = parseInt(match[1]);
                const column = parseInt(match[3] || '0');

                const previous = result.pop();
                if (previous !== undefined) {
                    const text = filterEscapeSequences(previous.text);
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
    language: LanguageKey;
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
                    language: component.componentState.lang as LanguageKey,
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

export enum magic_semver {
    trunk = '99999999.99999.999',
    non_trunk = '99999998.99999.999',
}

export function asSafeVer(semver: string | number | null | undefined): string {
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

        if (semver.includes('trunk') || semver.includes('main')) {
            return magic_semver.trunk;
        }
    }
    return magic_semver.non_trunk;
}

export function processExecutionResult(input: UnprocessedExecResult, inputFilename?: string): BasicExecutionResult {
    const start = performance.now();
    const stdout = parseOutput(input.stdout, inputFilename);
    const stderr = parseOutput(input.stderr, inputFilename);
    const end = performance.now();
    return {
        ...input,
        stdout,
        stderr,
        processExecutionResultTime: end - start,
    };
}

export function getEmptyExecutionResult(): BasicExecutionResult {
    return {
        code: -1,
        okToCache: false,
        filenameTransform: x => x,
        stdout: [],
        stderr: [],
        execTime: 0,
        timedOut: false,
    };
}

export function deltaTimeNanoToMili(startTime: bigint, endTime: bigint): number {
    return Number((endTime - startTime) / BigInt(1_000_000));
}

/**
 * Sleep for a number of milliseconds.
 */
export async function sleep(ms: number) {
    return new Promise(resolve => setTimeout(resolve, ms));
}
