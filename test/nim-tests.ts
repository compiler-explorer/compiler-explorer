// Copyright (c) 2019, Bastien Penavayre
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

import {beforeAll, describe, expect, it} from 'vitest';

import {unwrap} from '../lib/assert.js';
import {NimCompiler} from '../lib/compilers/nim.js';
import {LanguageKey} from '../types/languages.interfaces.js';

import {makeCompilationEnvironment, makeFakeCompilerInfo} from './utils.js';

const languages = {
    nim: {id: 'nim' as LanguageKey},
};

describe('Nim', () => {
    let ce;
    const info = {
        exe: '/dev/null',
        remote: {
            target: 'foo',
            path: 'bar',
            cmakePath: 'cmake',
            basePath: '/',
        },
        lang: languages.nim.id,
    };

    beforeAll(() => {
        ce = makeCompilationEnvironment({languages});
    });

    it('Nim should not allow --run/-r parameter', () => {
        const compiler = new NimCompiler(makeFakeCompilerInfo(info), ce);
        expect(compiler.filterUserOptions(['c', '--run', '--something'])).toEqual(['c', '--something']);
        expect(compiler.filterUserOptions(['cpp', '-r', '--something'])).toEqual(['cpp', '--something']);
    });

    it('Nim compile to Cpp if not asked otherwise', () => {
        const compiler = new NimCompiler(makeFakeCompilerInfo(info), ce);
        expect(compiler.filterUserOptions([])).toEqual(['compile']);
        expect(compiler.filterUserOptions(['badoption'])).toEqual(['compile', 'badoption']);
        expect(compiler.filterUserOptions(['js'])).toEqual(['js']);
    });

    it('test getCacheFile from possible user-options', () => {
        const compiler = new NimCompiler(makeFakeCompilerInfo(info), ce),
            input = 'test.min',
            folder = path.join('/', 'tmp/'),
            expected = {
                cpp: folder + '@m' + input + '.cpp.o',
                c: folder + '@m' + input + '.c.o',
                objc: folder + '@m' + input + '.m.o',
            };

        for (const lang of ['cpp', 'c', 'objc']) {
            expect(unwrap(compiler.getCacheFile([lang], input, folder))).toEqual(expected[lang]);
        }

        expect(compiler.getCacheFile([], input, folder)).toBeNull();
        expect(compiler.getCacheFile(['js'], input, folder)).toBeNull();
    });
});
