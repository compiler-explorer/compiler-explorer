// Copyright (c) 2026, Compiler Explorer Authors
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

import {beforeAll, describe, expect, it} from 'vitest';

import {CompilationEnvironment} from '../lib/compilation-env.js';
import {VyperCompiler} from '../lib/compilers/vyper.js';
import {LanguageKey} from '../types/languages.interfaces.js';
import {makeCompilationEnvironment, makeFakeCompilerInfo} from './utils.js';

const languages = {
    vyper: {id: 'vyper' as LanguageKey},
};

describe('VyperCompiler processAsm', () => {
    let ce: CompilationEnvironment;
    let compiler: VyperCompiler;

    beforeAll(() => {
        ce = makeCompilationEnvironment({languages});
        compiler = new VyperCompiler(
            makeFakeCompilerInfo({
                exe: '/fake/test/vyper/bin/vyper',
                remote: {
                    target: 'foo',
                    path: 'bar',
                    cmakePath: 'cmake',
                    basePath: '/',
                },
                lang: languages.vyper.id,
            }),
            ce,
        );
    });

    it('maps opcodes to addresses and source positions', async () => {
        const output = [
            'PUSH2 0x0028 PUSH1 0x00 CODECOPY PUSH0 CALLDATALOAD',
            JSON.stringify({pc_pos_map: {'0': [1, 0, 1, 21], '6': [4, 4, 5, 28]}}),
            '',
        ].join('\n');

        const {asm} = await compiler.processAsm({code: 0, asm: output});

        expect(asm.map(line => line.text)).toEqual(['PUSH2 0x0028', 'PUSH1 0x00', 'CODECOPY', 'PUSH0', 'CALLDATALOAD']);
        expect(asm.map(line => line.address)).toEqual([0, 3, 5, 6, 7]);
        expect(asm[0].source).toEqual({file: null, line: 1, column: 0, mainsource: true});
        expect(asm[1].source).toBeNull();
        expect(asm[3].source).toEqual({file: null, line: 4, column: 4, mainsource: true});
    });

    // Options like --help, --version or a different -f leave the output file missing or in another format.
    it.each([
        ['missing output file', '<No output file>'],
        ['bytecode only', '0x6100286100\n'],
        ['source map only', JSON.stringify({pc_pos_map: {'0': [1, 0, 1, 21]}}) + '\n'],
        ['source map without pc_pos_map', 'PUSH1 0x00\n{"breakpoints": []}\n'],
        ['non-object JSON', 'PUSH1 0x00\nnull\n'],
    ])('shows unexpected output verbatim instead of throwing (%s)', async (_name, output) => {
        const {asm} = await compiler.processAsm({code: 0, asm: output});
        expect(asm).toEqual([{text: output}]);
    });
});
