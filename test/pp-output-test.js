// Copyright (c) 2022, Compiler Explorer Authors
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

import * as fs from 'fs';

import {BaseCompiler} from '../lib/base-compiler';
import * as properties from '../lib/properties';

import * as filterTests from './pp-output-cases/filter-tests';

//const makeFakeCompilerInfo = (id: string, lang: string, group: string, semver: string, isSemver: boolean) => {
const makeFakeCompilerInfo = (id, lang, group, semver, isSemver) => {
    return {
        id: id,
        exe: '/dev/null',
        name: id,
        lang: lang,
        group: group,
        isSemVer: isSemver,
        semver: semver,
        libsArr: [],
    };
};

describe('Preprocessor Output Handling', () => {
    it('correctly filters lines', () => {
        const compilerInfo = makeFakeCompilerInfo('g82', 'c++', 'cpp', '8.2', true);
        const env = {
            ceProps: properties.fakeProps({}),
            compilerProps: () => {},
        };
        const compiler = new BaseCompiler(compilerInfo, env);
        for (const testCase of filterTests.cases) {
            const output = compiler.filterPP(testCase.input)[1];
            output.trim().should.eql(testCase.output.trim());
        }
    });
});
