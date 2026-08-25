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

import {describe, expect, it} from 'vitest';

import type {CompilerInfo} from '../../../types/compiler.interfaces.js';
import {DEFAULT_COMPILER_KEY, toCompilerKey} from '../../widgets/libs-widget.js';

const fakeCompiler = {id: 'g142', name: 'x86-64 gcc 14.2'} as unknown as CompilerInfo;

describe('LibsWidget compiler key', () => {
    it('reads the id off a compiler object, as the compiler and executor panes pass', () => {
        expect(toCompilerKey(fakeCompiler)).toBe('g142');
    });

    it('takes an id string as-is, as the conformance view passes', () => {
        expect(toCompilerKey('g142|clang1500')).toBe('g142|clang1500');
    });

    it('agrees with itself across the two call shapes for the same compiler', () => {
        expect(toCompilerKey('g142')).toBe(toCompilerKey(fakeCompiler));
    });

    it('falls back to the default key for every empty shape', () => {
        expect(toCompilerKey(undefined)).toBe(DEFAULT_COMPILER_KEY);
        expect(toCompilerKey(null)).toBe(DEFAULT_COMPILER_KEY);
        expect(toCompilerKey('')).toBe(DEFAULT_COMPILER_KEY);
    });

    it('never yields undefined, which is what keyed the conformance view into a dead bucket', () => {
        for (const shape of [fakeCompiler, 'g142|clang1500', '', null, undefined]) {
            expect(typeof toCompilerKey(shape)).toBe('string');
        }
    });
});
