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

import { AsmParser } from '../lib/asm-parser';
import { SassAsmParser } from '../lib/asm-parser-sass';
import { VcAsmParser } from '../lib/asm-parser-vc';
import * as utils from '../lib/utils';

import { fs, resolvePathFromTestRoot } from './utils';

// eslint-disable-next-line no-unused-vars
function bless(filename, output, filters) {
    const result = processAsm(resolvePathFromTestRoot(filename), filters);
    fs.writeFileSync(resolvePathFromTestRoot(output), JSON.stringify(result, null, 2));
}

// eslint-disable-next-line no-unused-vars
function dump(file) {
    for (const [i, element] of file.entries()) {
        console.log((i + 1) + ' : ' + JSON.stringify(element));
    }
}

// bless("filters-cases/mips5-square.asm", "filters-cases/mips5-square.asm.directives.labels.comments.json", {directives: true, labels: true, commentOnly: true});
// bless("filters-cases/gcc-sum.asm", "filters-cases/gcc-sum.asm.directives.labels.comments.json", {directives: true, labels: true, commentOnly: true});
// bless("filters-cases/gcc-arm-sum.asm", "filters-cases/gcc-arm-sum.asm.directives.labels.comments.json", {directives: true, labels: true, commentOnly: true});
// bless("filters-cases/gcc-avr-sum.asm", "filters-cases/gcc-avr-sum.asm.directives.labels.comments.json", {directives: true, labels: true, commentOnly: true});
// bless("filters-cases/cl-regex.asm", "filters-cases/cl-regex.asm.directives.labels.comments.json", {directives: true, labels: true, commentOnly: true});
// bless("filters-cases/cl-main-opt-out.asm", "filters-cases/cl-main-opt-out.directives.labels.comments.json", {directives: true, labels: true, commentOnly: true});
// bless("filters-cases/avr-loop.asm", "filters-cases/avr-loop.asm.directives.labels.comments.json", {directives: true, labels: true, commentOnly: true});
// bless("filters-cases/bug-192.asm", "filters-cases/bug-192.asm.directives.labels.comments.json", {directives: true, labels: true, commentOnly: true});
// bless("filters-cases/arm-moose.asm", "filters-cases/arm-moose.asm.directives.labels.comments.json", {directives: true, labels: true, commentOnly: true});
// bless("filters-cases/gcc-x86-vector.asm", "filters-cases/gcc-x86-vector.asm.directives.labels.comments.json", {directives: true, labels: true, commentOnly: true});
// bless("filters-cases/clang-on-mac.asm", "filters-cases/clang-on-mac.asm.directives.labels.comments.json", {directives: true, labels: true, commentOnly: true});
// bless("filters-cases/bug-349.asm", "filters-cases/bug-349.asm.directives.labels.comments.json", {directives: true, labels: true, commentOnly: true});
// bless("filters-cases/bug-348.asm", "filters-cases/bug-348.asm.directives.labels.comments.json", {directives: true, labels: true, commentOnly: true});
// bless("filters-cases/bug-660.asm", "filters-cases/bug-660.asm.directives.labels.comments.json", {directives: true, labels: true, commentOnly: true});
// bless("filters-cases/nvcc-example.asm", "filters-cases/nvcc-example.asm.directives.labels.comments.json", {directives: true, labels: true, commentOnly: true});
// bless("filters-cases/clang-cuda-example.asm", "filters-cases/clang-cuda.asm.directives.labels.comments.json", {directives: true, labels: true, commentOnly: true});
// bless("filters-cases/bug-995.asm", "filters-cases/bug-995.directives.labels.comments.json", {directives: true, labels: true, commentOnly: true});
// bless("filters-cases/arm-jump-table.asm", "filters-cases/arm-jump-table.asm.directives.labels.comments.json", {directives: true, labels: true, commentOnly: true});
// bless("filters-cases/bug-1179.asm", "filters-cases/bug-1179.asm.directives.labels.comments.json", {directives: true, labels: true, commentOnly: true});
// bless("filters-cases/6502-square.asm", "filters-cases/6502-square.asm.directives.labels.comments.json", {directives: true, labels: true, commentOnly: true});
// bless("filters-cases/sass-square.asm", "filters-cases/sass-square.asm.binary.directives.labels.comments.json", {binary: true, directives: true, labels: true, commentOnly: true});
// bless("filters-cases/sass-squarelabeled.asm", "filters-cases/sass-squarelabeled.asm.binary.directives.labels.comments.json", {binary: true, directives: true, labels: true, commentOnly: true});
// bless("filters-cases/bug-2164.asm", "filters-cases/bug-2164.asm.directives.labels.comments.json", {directives: true, labels: true, commentOnly: true});
// bless("filters-cases/bug-2164b.asm", "filters-cases/bug-2164b.asm.directives.labels.comments.json", {directives: true, labels: true, commentOnly: true});
// bless('filters-cases/diab.asm', 'filters-cases/diab.asm.directives.labels.comments.json', {directives: true, labels: true, commentOnly: true});
// bless("filters-cases/bintest-1.asm", "filters-cases/bintest-1.asm.binary.directives.labels.comments.json", {binary: true, directives: true, labels: true, commentOnly: true});
// bless("filters-cases/bintest-2.asm", "filters-cases/bintest-2.asm.binary.directives.labels.comments.json", {binary: true, directives: true, labels: true, commentOnly: true});
// bless("filters-cases/bintest-unicode-1.asm", "filters-cases/bintest-unicode-1.asm.binary.directives.labels.comments.json", {binary: true, directives: true, labels: true, commentOnly: true});
// describe('A test', function() {
//     it('should work', function(){
//         console.log(processAsm(resolvePathFromTestRoot('filters-cases/6502-square.asm'), {directives: true, labels: true, commentOnly: true}));
//     });
// });

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

const files = fs.readdirSync(resolvePathFromTestRoot('filters-cases'));
const filesInCaseDir = files.map(x => resolvePathFromTestRoot('filters-cases', x));

const cases = filesInCaseDir.filter(x => x.endsWith('.asm'));

function testFilter(filename, suffix, filters) {
    const expected = filename + suffix;
    const json = filesInCaseDir.includes(expected + '.json');

    let file;

    if (json) {
        file = fs.readFileSync(resolvePathFromTestRoot(expected + '.json'), 'utf-8');
    }
    else if (filesInCaseDir.includes(expected)) {
        file = fs.readFileSync(resolvePathFromTestRoot(expected), 'utf-8');
    }
    else {
        return;
    }
    const result = processAsm(resolvePathFromTestRoot(filename), filters);

    if (json) {
        file = JSON.parse(file);
    } else {
        file = utils.splitLines(file);
    }

    it(filename, () => {
        if (json) {
            result.should.deep.equal(file, `${filename} case error`);
        } else {
            result.asm.map(x => x.text).should.deep.equal(file, `${filename} case error`);
        }
    });
}

/*
    The before() hooks on mocha are for it()s - They don't execute before the describes!
    That's sad because then we can't have cases be loaded in a before() for every descirbe child to see
 */
describe('Filter test cases', function () {

    describe('No filters', function () {
        cases.forEach(x => testFilter(x, '.none', {}));
    });
    describe('Directive filters', function () {
        cases.forEach(x => testFilter(x, '.directives', {directives: true}));
    });
    describe('Directives and labels together', function () {
        cases.forEach(x => testFilter(x, '.directives.labels', {directives: true, labels: true}));
    });
    describe('Directives, labels and comments', function () {
        cases.forEach(function (x) {
            testFilter(x, '.directives.labels.comments', {directives: true, labels: true, commentOnly: true});
        });
    });
    describe('Binary, directives, labels and comments', function () {
        cases.forEach(function (x) {
            testFilter(x, '.binary.directives.labels.comments', {binary: true, directives: true, labels: true, commentOnly: true});
        });
    });
    describe('Directives and comments', function () {
        cases.forEach(x => testFilter(x, '.directives.comments', {directives: true, commentOnly: true}));
    });
    describe('Directives and library code', function () {
        cases.forEach(x => testFilter(x, '.directives.library', {directives: true, libraryCode: true}));
    });
    describe('Directives, labels, comments and library code', function () {
        cases.forEach(function (x) {
            testFilter(x, '.directives.labels.comments.library',
                {directives: true, labels: true, commentOnly: true, libraryCode: true});
        });
    });
});
