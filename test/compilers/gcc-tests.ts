// Copyright (c) 2025, Compiler Explorer Authors
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

import fs from 'node:fs/promises';
import path from 'node:path';

import {describe, expect, it} from 'vitest';

import {GCCCompiler} from '../../lib/compilers/index.js';
import {makeCompilationEnvironment} from '../utils.js';

describe('gcc tests', () => {
    const languages = {'c++': {id: 'c++'}};
    const info = {
        exe: 'foobar',
        remote: true,
        lang: 'c++',
        ldPath: [],
    };
    const gcc = new GCCCompiler(info as any, makeCompilationEnvironment({languages}));

    it('should process gcc opt output', async () => {
        const test = `<source>:5:9: optimized: loop with 1 iterations completely unrolled (header execution count 78082503)
<source>:3:6: note: ***** Analysis failed with vector mode V4SI
<source>:11:6: missed: splitting region at control altering definition _44 = std::basic_filebuf<char>::open (&fs._M_filebuf, "myfile", 16);
/opt/compiler-explorer/gcc-14.1.0/include/c++/14.1.0/bits/basic_ios.h:466:59: missed: statement clobbers memory: std::ios_base::ios_base (&MEM[(struct basic_ios *)&fs + 248B].D.46591);
`;

        const dirPath = await gcc.newTempDir();
        const optPath = path.join(dirPath, 'temp.out');
        await fs.writeFile(optPath, test);
        const dummyCompilationResult = {optPath: optPath, code: 0, stdout: [], stderr: [], timedOut: false};
        gcc.compiler.optArg = '-fopt-info-all=temp.out';
        expect(await gcc.processOptOutput(dummyCompilationResult)).toEqual([
            {
                Args: [],
                DebugLoc: {File: '<source>', Line: 5, Column: 9},
                Function: '',
                Name: '',
                Pass: '',
                displayString: 'loop with 1 iterations completely unrolled (header execution count 78082503)',
                optType: 'Passed',
            },
            {
                Args: [],
                DebugLoc: {File: '<source>', Line: 3, Column: 6},
                Function: '',
                Name: '',
                Pass: '',
                displayString: '***** Analysis failed with vector mode V4SI',
                optType: 'Analysis',
            },
            {
                Args: [],
                DebugLoc: {File: '<source>', Line: 11, Column: 6},
                Function: '',
                Name: '',
                Pass: '',
                displayString:
                    'splitting region at control altering definition _44 = std::basic_filebuf<char>::open (&fs._M_filebuf, "myfile", 16);',
                optType: 'Missed',
            },
        ]);
    });
});
