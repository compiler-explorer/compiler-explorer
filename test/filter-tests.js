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

import * as path from 'path';

import approvals from 'approvals';

import { SassAsmParser } from '../lib/asm-parser-sass';
import { VcAsmParser } from '../lib/asm-parser-vc';
import { BaseCompiler } from '../lib/base-compiler';
import { execute } from '../lib/exec';

import { fs, resolvePathFromTestRoot } from './utils';

approvals.mocha();

function processAsmTheNewWay(filename, filters) {
    if (filename.includes('sass-')) {
        return new Promise((resolve) => {
            const parser = new SassAsmParser();
            const file = fs.readFileSync(filename, 'utf-8');
            const result = parser.process(file, filters);
            resolve(result);
        });
    } else {
        const file = fs.readFileSync(filename, 'utf-8');
        if (file.includes('Microsoft')) {
            return new Promise((resolve) => {
                const parser = new VcAsmParser();
                const result = parser.process(file, filters);
                resolve(result);
            });
        }
    }

    const asmParserPath = process.env.ASMPARSER || '/opt/compiler-explorer/asm-parser/build/bin/asm-parser';
    const filterParams = BaseCompiler.getAsmParserParametersBasedOnFilter(filters, false);
    filterParams.push('-debugdump');
    filterParams.push(filename);

    return execute(asmParserPath, filterParams, {}).then(result => {
        result = Object.assign({}, JSON.parse(result.stdout));
        if (result.stderr) {
            throw result.stderr;
        }

        return result;
    });
}

const casesRoot = resolvePathFromTestRoot('filters-cases');

let optionsOverride = {
    forceApproveAll: false, // set to true to automatically regenerate all the cases.
    reporters: ['donothing'],
    errorOnStaleApprovedFiles: false,
};

if (process.env.REPORT === '1') {
    delete optionsOverride.reporters;
}

if (process.env.APPROVE === '1') {
    optionsOverride.forceApproveAll = true;
}

function testFilter(filename, suffix, filters) {
    const testName = path.basename(filename + suffix);
    it(testName, () => {
        const newResultPromise = processAsmTheNewWay(resolvePathFromTestRoot(path.join('filters-cases', filename)), filters);

        return newResultPromise.then(result => {
            delete result.parsingTime;
            approvals.verifyAsJSON(
                casesRoot,
                testName,
                result,
                optionsOverride);
        });
    });
}

function testAllForFile(filename) {
    // testFilter(filename, '.none', {});
    // testFilter(filename, '.directives', {directives: true});
    // testFilter(filename, '.directives.comments', {directives: true, commentOnly: true});
    // testFilter(filename, '.directives.labels', {directives: true, labels: true});
    testFilter(filename, '.directives.labels.comments', {directives: true, labels: true, commentOnly: true});
    // testFilter(filename, '.directives.labels.comments.library', {directives: true, labels: true, commentOnly: true, libraryCode: true});
    // testFilter(filename, '.directives.library', {directives: true, libraryCode: true});
}

describe('Filter test cases', function () {
    testAllForFile('6502-square.asm');
    //testAllForFile('arm-hellow.asm');
    // testAllForFile('arm-jump-table.asm');
    // testAllForFile('arm-moose.asm');
    // testAllForFile('arm-static.asm');
    // testAllForFile('avr-loop.asm');
    // testAllForFile('bintest-1.asm');
    // testAllForFile('bintest-2.asm');
    // testAllForFile('bintest-unicode-1.asm');
    // testAllForFile('bug-1179.asm');
    // testAllForFile('bug-1229.asm');
    // testAllForFile('bug-1285.asm');
    // testAllForFile('bug-1285b.asm');
    // testAllForFile('bug-1285c.asm');
    // testAllForFile('bug-1307.asm');
    // testAllForFile('bug-192.asm');
    // testAllForFile('bug-1989_alpha.asm');
    // testAllForFile('bug-1989_microblaze.asm');
    // testAllForFile('bug-1989_sparc.asm');
    // testAllForFile('bug-2032.asm');
    // testAllForFile('bug-2164.asm');
    // testAllForFile('bug-2164b.asm');
    // testAllForFile('bug-348.asm');
    // testAllForFile('bug-349.asm');
    // testAllForFile('bug-577_clang.asm');
    // testAllForFile('bug-577_gcc.asm');
    // testAllForFile('bug-577_icc.asm');
    // testAllForFile('bug-629.asm');
    // testAllForFile('bug-660.asm');
    // testAllForFile('bug-725_rust.asm');
    // testAllForFile('bug-995.asm');
    // testAllForFile('clang-cuda-example.asm');
    // testAllForFile('clang-hellow.asm');
    // testAllForFile('clang-maxArray.asm');
    // testAllForFile('clang-on-mac.asm');
    // testAllForFile('clang-static.asm');
    // testAllForFile('diab.asm');
    // testAllForFile('eigen-test.asm');
    // testAllForFile('gcc-arm-sum.asm');
    // testAllForFile('gcc-avr-sum.asm');
    // testAllForFile('gcc-sum.asm');
    // testAllForFile('gcc-x86-vector.asm');
    // testAllForFile('gcc4.6-hellow.asm');
    // testAllForFile('gcc4.6-static.asm');
    // testAllForFile('icc-static.asm');
    // testAllForFile('icc.hellow.asm');
    // testAllForFile('kalray-hellow.asm');
    // testAllForFile('mcore-square.asm');
    // testAllForFile('mips5-square.asm');
    // testAllForFile('nvcc-example.asm');
    // testAllForFile('rx-mas100-square.asm');
    // testAllForFile('sass-square.asm');
    // testAllForFile('sass-squarelabeled.asm');
    // testAllForFile('string-constant.asm');
    // testAllForFile('vc-main-opt-out.asm');
    // testAllForFile('vc-numbers.asm');
    // testAllForFile('vc-regex.asm');
    // testAllForFile('vc-threadlocalddef.asm');
});

// describe('forceApproveAll should be false', () => {
//     it('should have forceApproveAll false', () => {
//         optionsOverride.forceApproveAll.should.be.false;
//     });
// });
