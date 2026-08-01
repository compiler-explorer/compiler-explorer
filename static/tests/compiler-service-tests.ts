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

import {CompilerService} from '../../static/compiler-service.js';
import {CompilationResult} from '../../types/compilation/compilation.interfaces.js';

/** A minimally-populated successful result; override just the bits a test cares about. */
function makeResult(overrides: Partial<CompilationResult> = {}): CompilationResult {
    return {code: 0, timedOut: false, stdout: [], stderr: [], ...overrides};
}

describe('CompilerService.getAsmAsText', () => {
    // Chained panes take an upstream compiler's output as their own source. These cases decide
    // what a downstream pane ends up compiling, so a failed upstream must yield nothing at all
    // rather than propagating placeholder or partial assembly down the chain.
    describe('results that yield no text', () => {
        it('handles a missing result', () => {
            // becomeEditor() passes lastResult, which is null until the pane has compiled once.
            expect(CompilerService.getAsmAsText(null)).toEqual('');
            expect(CompilerService.getAsmAsText(undefined)).toEqual('');
        });

        it('ignores assembly from a failed compilation', () => {
            // A failed compile still comes back with asm (often a synthetic "<Compilation
            // failed>" line); that must not be handed downstream as if it were real source.
            const result = makeResult({code: 1, asm: [{text: '<Compilation failed>'}]});
            expect(CompilerService.getAsmAsText(result)).toEqual('');
        });

        it('ignores assembly from a timed-out compilation', () => {
            // timedOut is checked separately from the exit code: a timeout can report code 0.
            const result = makeResult({timedOut: true, asm: [{text: 'mov eax, eax'}]});
            expect(CompilerService.getAsmAsText(result)).toEqual('');
        });

        it('handles a successful compilation with no assembly', () => {
            expect(CompilerService.getAsmAsText(makeResult({asm: undefined}))).toEqual('');
        });

        it('handles empty string assembly', () => {
            expect(CompilerService.getAsmAsText(makeResult({asm: ''}))).toEqual('');
        });

        it('handles an empty assembly array', () => {
            // Distinct path from the cases above: [] is truthy, so this is converted rather
            // than rejected, and joining no lines gives the empty string.
            expect(CompilerService.getAsmAsText(makeResult({asm: []}))).toEqual('');
        });
    });

    describe('successful results', () => {
        it('passes string assembly through unchanged', () => {
            const result = makeResult({asm: '  mov eax, 42\n  ret\n'});
            expect(CompilerService.getAsmAsText(result)).toEqual('  mov eax, 42\n  ret\n');
        });

        it('joins parsed assembly lines with newlines', () => {
            const result = makeResult({asm: [{text: 'square(int):'}, {text: '  imul eax, eax'}]});
            expect(CompilerService.getAsmAsText(result)).toEqual('square(int):\n  imul eax, eax');
        });

        it('does not append a trailing newline', () => {
            // Assemblers warn about a final line with no newline ("end of file not at end of a
            // line"). Pinned deliberately: adding one should be a conscious change.
            const result = makeResult({asm: [{text: 'ret'}]});
            expect(CompilerService.getAsmAsText(result)).toEqual('ret');
        });

        it('preserves blank lines between instructions', () => {
            const result = makeResult({asm: [{text: 'a'}, {text: ''}, {text: 'b'}]});
            expect(CompilerService.getAsmAsText(result)).toEqual('a\n\nb');
        });

        it('keeps only the text of each line', () => {
            // Source mappings, labels and opcodes are dropped; only text reaches the downstream.
            const result = makeResult({
                asm: [
                    {text: 'main:', source: {file: 'example.cpp', line: 1}, labels: []},
                    {text: '  ret', opcodes: ['c3'], address: 4198400},
                ],
            });
            expect(CompilerService.getAsmAsText(result)).toEqual('main:\n  ret');
        });
    });
});
