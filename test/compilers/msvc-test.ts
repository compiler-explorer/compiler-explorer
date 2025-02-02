// Copyright (c) 2024, Compiler Explorer Authors
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

import {describe, expect, it} from 'vitest';

import {Win32VcCompiler} from '../../lib/compilers/index.js';
import {makeCompilationEnvironment} from '../utils.js';

describe('msvc tests', () => {
    const languages = {'c++': {id: 'c++'}};

    const info = {
        exe: 'foobar',
        remote: true,
        lang: 'c++',
        ldPath: [],
        options: '/EHsc /utf-8',
    };

    describe('utf8 replacement', async () => {
        const msvc = new Win32VcCompiler(info as any, makeCompilationEnvironment({languages}));
        it('should do the normal situation', async () => {
            const prepared: string[] = msvc.prepareArguments([], {}, {}, 'example.cpp', 'output.s', [], []);
            expect(prepared).toEqual([
                '/nologo',
                '/FA',
                '/c',
                '/Faoutput.s',
                '/Fooutput.s.obj',
                '/EHsc',
                '/utf-8',
                'example.cpp',
            ]);
        });

        it('should replace arguments', async () => {
            const prepared: string[] = msvc.prepareArguments(
                ['/O2', '/source-charset:windows-1252'],
                {},
                {},
                'example.cpp',
                'output.s',
                [],
                [],
            );
            expect(prepared).toEqual([
                '/nologo',
                '/FA',
                '/c',
                '/Faoutput.s',
                '/Fooutput.s.obj',
                '/EHsc',
                '/O2',
                '/source-charset:windows-1252',
                'example.cpp',
            ]);
        });
    });
});
