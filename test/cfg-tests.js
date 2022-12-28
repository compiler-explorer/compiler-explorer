// Copyright (c) 2017, Najjar Chedy
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

import * as cfg from '../lib/cfg';

import {fs, path, resolvePathFromTestRoot} from './utils';

async function DoCfgTest(cfgArg, filename) {
    const contents = await fs.readJson(filename, 'utf8');
    const structure = cfg.generateStructure('', cfgArg, contents.asm);
    structure.should.deep.equal(contents.cfg);
}

describe('Cfg test cases', () => {
    const testcasespath = resolvePathFromTestRoot('cfg-cases');

    /*
     * NB: this readdir must *NOT* be async
     *
     * Mocha calls the function passed to `describe` synchronously
     * and expects the test suite to be fully configured upon return.
     *
     * If you pass an async function to describe and setup test cases
     * after an await there is no guarantee they will be found, and
     * if they are they will not end up in the expected suite.
     */
    const files = fs.readdirSync(testcasespath);

    describe('gcc', () => {
        for (const filename of files.filter(x => x.includes('gcc'))) {
            it(filename, async () => {
                await DoCfgTest('g++', path.join(testcasespath, filename));
            });
        }
    });

    describe('clang', () => {
        for (const filename of files.filter(x => x.includes('clang'))) {
            it(filename, async () => {
                await DoCfgTest('clang', path.join(testcasespath, filename));
            });
        }
    });
});
