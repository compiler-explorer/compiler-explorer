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

import approvals from 'approvals';

import { AsmParser } from '../lib/asm-parser';
import { SassAsmParser } from '../lib/asm-parser-sass';
import { VcAsmParser } from '../lib/asm-parser-vc';

import { fs, resolvePathFromTestRoot } from './utils';

approvals.mocha();

function processAsm(filename, filters) {
    const file = fs.readFileSync(filename, 'utf-8');
    let parser;
    if (file.includes('Microsoft'))
        parser = new VcAsmParser();
    else if (filename.includes('sass-'))
        parser = new SassAsmParser();
    else
        parser = new AsmParser();
    return parser.process(file, filters);
}

const casesRoot = resolvePathFromTestRoot('filters-cases');
const files = fs.readdirSync(casesRoot);
const filesInCaseDir = files.map(x => resolvePathFromTestRoot('filters-cases', x));

const cases = filesInCaseDir.filter(x => x.endsWith('.asm'));

const optionsOverride = {
    forceApproveAll: false, // set to true to automatically regenerate all the cases.
    blockUntilReporterExits: false,
    maxLaunches: 1,
    normalizeLineEndingsTo: process.platform === 'win32' ? '\r\n' : '\n',
    errorOnStaleApprovedFiles: process.platform !== 'win32',
};

function testFilter(filename, suffix, filters) {
    const testName = path.basename(filename + suffix);
    it(testName, () => {
        const result = processAsm(filename, filters);
        delete result.parsingTime;
        delete result.filteredCount;
        approvals.verifyAsJSON(
            casesRoot,
            testName,
            result,
            optionsOverride);
    }).timeout(5000); // Bump the timeout a bit so that we don't fail for slow cases
}

/*
    The before() hooks on mocha are for it()s - They don't execute before the describes!
    That's sad because then we can't have cases be loaded in a before() for every describe child to see.
 */
describe('Filter test cases', function () {

    describe('No filters', function () {
        for (const x of cases) testFilter(x, '.none', {});
    });
    describe('Directive filters', function () {
        for (const x of cases) testFilter(x, '.directives', {directives: true});
    });
    describe('Directives and labels together', function () {
        for (const x of cases) testFilter(x, '.directives.labels', {directives: true, labels: true});
    });
    describe('Directives, labels and comments', function () {
        for (const x of cases) {
            testFilter(x, '.directives.labels.comments', {directives: true, labels: true, commentOnly: true});
        }
    });
    describe('Binary, directives, labels and comments', function () {
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
    describe('Directives and comments', function () {
        for (const x of cases) testFilter(x, '.directives.comments', {directives: true, commentOnly: true});
    });
    describe('Directives and library code', function () {
        for (const x of cases) testFilter(x, '.directives.library', {directives: true, libraryCode: true});
    });
    describe('Directives, labels, comments and library code', function () {
        for (const x of cases) {
            testFilter(x, '.directives.labels.comments.library',
                {directives: true, labels: true, commentOnly: true, libraryCode: true});
        }
    });
});

describe('AsmParser tests', () => {
    const parser = new AsmParser();
    it('should identify generic opcodes', () => {
        parser.hasOpcode('  mov r0, #1').should.be.true;
        parser.hasOpcode('  ROL A').should.be.true;
    });
    it('should not identify non-opcodes as opcodes', () => {
        parser.hasOpcode('  ;mov r0, #1').should.be.false;
        parser.hasOpcode('').should.be.false;
        parser.hasOpcode('# moose').should.be.false;
    });
    it('should identify llvm opcodes', () => {
        parser.hasOpcode('  %i1 = phi i32 [ %i2, %.preheader ], [ 0, %bb ]').should.be.true;
    });
});

describe('forceApproveAll should be false', () => {
    it('should have forceApproveAll false', () => {
        optionsOverride.forceApproveAll.should.be.false;
    });
});
