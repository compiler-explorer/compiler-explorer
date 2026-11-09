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

import {NexiumCompiler} from '../lib/compilers/nexium.js';
import {LanguageKey} from '../types/languages.interfaces.js';
import {makeCompilationEnvironment, makeFakeCompilerInfo, makeFakeParseFiltersAndOutputOptions} from './utils.js';

describe('Nexium C view filters', () => {
    let nexium: NexiumCompiler;
    const emitted = [
        '#include <stdint.h>',
        'static void nx_runtime_helper(void) {}',
        '/* ---- imported C headers ---- */',
        '#include <string.h>',
        '/* ---- functions ---- */',
        'static int nx_square(int n) {',
        '  return n * n;',
        '}',
    ].join('\n');
    const view = async (filters: Record<string, boolean>) =>
        (await nexium.processAsm({asm: emitted}, makeFakeParseFiltersAndOutputOptions(filters), [])).asm.map(
            (line: {text: string}) => line.text,
        );

    beforeAll(() => {
        const ce = makeCompilationEnvironment({languages: {nexium: {id: 'nexium' as LanguageKey}}});
        nexium = new NexiumCompiler(makeFakeCompilerInfo({exe: '/dev/null', lang: 'nexium' as LanguageKey}), ce);
    });

    it('shows everything with no filter on', async () => {
        expect(await view({})).toEqual(emitted.split('\n'));
    });

    it('hides the inlined runtime with the library functions filter', async () => {
        expect((await view({libraryCode: true}))[0]).toEqual('/* ---- imported C headers ---- */');
    });

    it('hides comments with the comments filter', async () => {
        expect(await view({commentOnly: true})).not.toContain('/* ---- functions ---- */');
    });

    it('hides preprocessor lines with the directives filter', async () => {
        expect((await view({directives: true})).filter((line: string) => line.startsWith('#'))).toEqual([]);
    });
});
