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

import path from 'path';

import {describe, expect, it} from 'vitest';

import {CC65AsmParser} from '../lib/parsers/asm-parser-cc65.js';
import {AsmEWAVRParser} from '../lib/parsers/asm-parser-ewavr.js';
import {SassAsmParser} from '../lib/parsers/asm-parser-sass.js';
import {VcAsmParser} from '../lib/parsers/asm-parser-vc.js';
import {AsmParser} from '../lib/parsers/asm-parser.js';
import {fakeProps} from '../lib/properties.js';
import {ParseFiltersAndOutputOptions} from '../types/features/filters.interfaces.js';

import {fs, resolvePathFromTestRoot} from './utils.js';

function processAsm(filename: string, filters: ParseFiltersAndOutputOptions) {
    const file = fs.readFileSync(filename, 'utf8');
    let parser: AsmParser;
    if (file.includes('Microsoft')) parser = new VcAsmParser();
    else if (filename.includes('sass-')) parser = new SassAsmParser();
    else if (filename.includes('cc65-')) parser = new CC65AsmParser(fakeProps({}));
    else if (filename.includes('ewarm-')) parser = new AsmEWAVRParser(fakeProps({}));
    else {
        parser = new AsmParser();
        parser.binaryHideFuncRe =
            /^(__.*|_(init|start|fini)|(de)?register_tm_clones|call_gmon_start|frame_dummy|\.plt.*|_dl_relocate_static_pie)$/;
    }
    return parser.process(file, filters);
}

const casesRoot = resolvePathFromTestRoot('filters-cases');
const files = fs.readdirSync(casesRoot);
const filesInCaseDir = files.map(x => resolvePathFromTestRoot('filters-cases', x));

const cases = filesInCaseDir.filter(x => x.endsWith('.asm'));

const recursivelyOrderKeys = (unordered: any): any => {
    if (unordered === null) {
        return null;
    }
    if (Array.isArray(unordered)) {
        return unordered.map(recursivelyOrderKeys);
    }
    if (typeof unordered === 'object') {
        const ordered: {[key: string]: any} = {};
        for (const key of Object.keys(unordered).sort()) {
            ordered[key] = recursivelyOrderKeys(unordered[key]);
        }
        return ordered;
    }
    return unordered;
};

const stringifyKeysInOrder = (data: any): string => {
    const sortedData = recursivelyOrderKeys(data);
    return JSON.stringify(sortedData, null, '  ');
};

function testFilter(filename: string, suffix: string, filters: ParseFiltersAndOutputOptions) {
    const testName = path.basename(filename + suffix);
    it(
        testName,
        () => {
            const result = processAsm(filename, filters);
            delete result.parsingTime;
            delete result.filteredCount;
            // TODO normalize line endings?
            expect(stringifyKeysInOrder(result)).toMatchFileSnapshot(path.join(casesRoot, testName + '.json'));
        },
        {timeout: 10000},
    ); // Bump the timeout a bit so that we don't fail for slow cases
}

describe('Filter test cases', () => {
    if (process.platform === 'win32') {
        it('should skip filter-tests on Windows', () => {
            expect(true).toBe(true);
        });
        return;
    }

    describe('No filters', () => {
        for (const x of cases) testFilter(x, '.none', {});
    });
    describe('Directive filters', () => {
        for (const x of cases) testFilter(x, '.directives', {directives: true});
    });
    describe('Directives and labels together', () => {
        for (const x of cases) testFilter(x, '.directives.labels', {directives: true, labels: true});
    });
    describe('Directives, labels and comments', () => {
        for (const x of cases) {
            testFilter(x, '.directives.labels.comments', {directives: true, labels: true, commentOnly: true});
        }
    });
    describe('Binary, directives, labels and comments', () => {
        if (process.platform !== 'win32') {
            for (const x of cases) {
                testFilter(x, '.binary.directives.labels.comments', {
                    binary: true,
                    directives: true,
                    labels: true,
                    commentOnly: true,
                });
            }
        }
    });
    describe('Binary, directives, labels, comments and library code', () => {
        if (process.platform !== 'win32') {
            for (const x of cases) {
                if (!x.endsWith('-bin.asm')) continue;

                testFilter(x, '.binary.directives.labels.comments.library', {
                    binary: true,
                    directives: true,
                    labels: true,
                    commentOnly: true,
                    libraryCode: true,
                });
            }
        }
    });
    describe('Binary, directives, labels, comments and library code with dontMaskFilenames', () => {
        if (process.platform !== 'win32') {
            for (const x of cases) {
                if (!x.endsWith('-bin.asm')) continue;

                testFilter(x, '.binary.directives.labels.comments.library.dontMaskFilenames', {
                    binary: true,
                    directives: true,
                    labels: true,
                    commentOnly: true,
                    libraryCode: true,
                    dontMaskFilenames: true,
                });
            }
        }
    });
    describe('Directives and comments', () => {
        for (const x of cases) testFilter(x, '.directives.comments', {directives: true, commentOnly: true});
    });
    describe('Directives and library code', () => {
        for (const x of cases) testFilter(x, '.directives.library', {directives: true, libraryCode: true});
    });
    describe('Directives, labels, comments and library code', () => {
        for (const x of cases) {
            testFilter(x, '.directives.labels.comments.library', {
                directives: true,
                labels: true,
                commentOnly: true,
                libraryCode: true,
            });
        }
    });
    describe('Directives, labels, comments and library code with dontMaskFilenames', () => {
        for (const x of cases) {
            testFilter(x, '.directives.labels.comments.library.dontMaskFilenames', {
                directives: true,
                labels: true,
                commentOnly: true,
                libraryCode: true,
                dontMaskFilenames: true,
            });
        }
    });
});
