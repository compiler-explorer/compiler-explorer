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

const chai = require('chai'),
    cfg = require('../lib/cfg'),
    fs = require('fs');

const cases = fs.readdirSync(__dirname + '/cases/')
    .filter(x => x.match(/cfg-\w*/))
    .map(x => __dirname + '/cases/' + x);

const assert = chai.assert;

function common(cases, filterArg, cfgArg) {
    cases.filter(x => x.includes(filterArg))
        .forEach(filename => {
            const file = fs.readFileSync(filename, 'utf-8');
            if (file) {
                const contents = JSON.parse(file);
                assert.deepEqual(cfg.generateStructure('', cfgArg, contents.asm), contents.cfg, `${filename}`);
            }
        });
}

describe('Cfg test cases', () => {
    it('works for gcc', () => {
        common(cases, "gcc", "g++");
    });

    it('works for clang', () => {
        common(cases, "clang", "clang");
    });
});
