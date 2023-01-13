// Copyright (c) 2021, Compiler Explorer Authors
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

import {HookCompiler} from '../../lib/compilers';
import {chai, makeCompilationEnvironment} from '../utils';

const expect = chai.expect;

describe('Hook compiler', () => {
    it('should return correct key', () => {
        HookCompiler.key.should.equal('hook');
    });

    const info = {remote: true, lang: 'hook'};
    const languages = {hook: {id: 'hook'}};
    const hook = new HookCompiler(info, makeCompilationEnvironment({languages}));

    it('should return correct options for filter', () => {
        hook.optionsForFilter().should.deep.equal(['--dump']);
    });

    it('should return correct output filename', () => {
        const dirPath = '/tmp';
        hook.getOutputFilename(dirPath).should.equal('/tmp/example.out');
    });

    it('should process and return correct bytecode result', () => {
        const asm =
            '; main in /app/example.hk at 0x56554a556550\n' +
            '; 0 parameter(s), 0 non-local(s), 0 constant(s), 0 function(s)\n' +
            '  1         0 Int                       2\n' +
            '            3 Int                       2\n' +
            '            6 Multiply\n' +
            '  2         7 Load                      2\n' +
            '            9 Return\n' +
            '           10 ReturnNil\n' +
            '; 6 instruction(s)\n';
        const expected = {
            asm: [
                {
                    source: {
                        file: null,
                        line: undefined,
                    },
                    text: '; main in /app/example.hk at 0x56554a556550',
                },
                {
                    source: {
                        file: null,
                        line: undefined,
                    },
                    text: '; 0 parameter(s), 0 non-local(s), 0 constant(s), 0 function(s)',
                },
                {
                    source: {
                        file: null,
                        line: 1,
                    },
                    text: '  1         0 Int                       2',
                },
                {
                    source: {
                        file: null,
                        line: 1,
                    },
                    text: '            3 Int                       2',
                },
                {
                    source: {
                        file: null,
                        line: 1,
                    },
                    text: '            6 Multiply',
                },
                {
                    source: {
                        file: null,
                        line: 2,
                    },
                    text: '  2         7 Load                      2',
                },
                {
                    source: {
                        file: null,
                        line: 2,
                    },
                    text: '            9 Return',
                },
                {
                    source: {
                        file: null,
                        line: 2,
                    },
                    text: '           10 ReturnNil',
                },
                {
                    source: {
                        file: null,
                        line: undefined,
                    },
                    text: '; 6 instruction(s)',
                },
                {
                    source: {
                        file: null,
                        line: undefined,
                    },
                    text: '',
                },
            ],
        };
        const result = hook.processAsm({asm: asm});
        result.should.deep.equal(expected);
    });
});
