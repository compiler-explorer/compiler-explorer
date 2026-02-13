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

import {describe, expect, it} from 'vitest';

import {NumbaPassDumpParser} from '../lib/parsers/numba-pass-dump-parser.js';
import type {ResultLine} from '../types/resultline/resultline.interfaces.js';

function lines(...texts: string[]): ResultLine[] {
    return texts.map(text => ({text}));
}

describe('numba-pass-dump-parser', () => {
    const parser = new NumbaPassDumpParser();

    it('should parse a single pass for one function', () => {
        const output = lines(
            '------__main__.example: nopython: AFTER translate_bytecode------',
            'label 0:',
            "    arr = arg(0, name=arr)               ['arr']",
            '    jump 10                              []',
            'label 10:',
            '    return arr',
        );

        const dumps = parser.breakdownOutputIntoPassDumps(output);

        expect(dumps).toHaveLength(1);
        expect(dumps[0].functionName).toBe('__main__.example');
        expect(dumps[0].passName).toBe('translate_bytecode');
        expect(dumps[0].lines).toEqual(output.slice(1));
    });

    it('should parse multiple passes for one function', () => {
        const output = lines(
            '------__main__.example: nopython: AFTER translate_bytecode------',
            'label 0:',
            "    arr = arg(0, name=arr)               ['arr']",
            '------__main__.example: nopython: AFTER fixup_args------',
            'label 0:',
            "    arr = arg(0, name=arr)               ['arr']",
            '------__main__.example: nopython: AFTER ir_processing------',
            'label 0:',
            "    arr = arg(0, name=arr, optimized=True) ['arr']",
        );

        const dumps = parser.breakdownOutputIntoPassDumps(output);

        expect(dumps).toHaveLength(3);
        expect(dumps[0].passName).toBe('translate_bytecode');
        expect(dumps[1].passName).toBe('fixup_args');
        expect(dumps[2].passName).toBe('ir_processing');
    });

    it('should group passes by function name', () => {
        const output = lines(
            '------__main__.foo: nopython: AFTER translate_bytecode------',
            'label 0:',
            '    return 1',
            '------__main__.foo: nopython: AFTER fixup_args------',
            'label 0:',
            '    return 1',
            '------numba.np.arrayobj.impl: nopython: AFTER translate_bytecode------',
            'label 0:',
            '    return arr',
        );

        const dumps = parser.breakdownOutputIntoPassDumps(output);
        const grouped = parser.associatePassDumpsWithGroups(dumps);

        expect(Object.keys(grouped)).toEqual(['__main__.foo', 'numba.np.arrayobj.impl']);
        expect(grouped['__main__.foo']).toHaveLength(2);
        expect(grouped['numba.np.arrayobj.impl']).toHaveLength(1);
    });

    it('should detect irChanged as false when IR is identical between passes', () => {
        const output = lines(
            '------__main__.foo: nopython: AFTER translate_bytecode------',
            'label 0:',
            '    return 1',
            '------__main__.foo: nopython: AFTER fixup_args------',
            'label 0:',
            '    return 1',
        );

        const results = parser.process(output, {} as any, {} as any);

        expect(results['__main__.foo']).toHaveLength(2);
        // First pass: before is empty (no previous), after has content → changed
        expect(results['__main__.foo'][0].irChanged).toBe(true);
        // Second pass: IR identical to first → not changed
        expect(results['__main__.foo'][1].irChanged).toBe(false);
    });

    it('should detect irChanged as true when IR differs between passes', () => {
        const output = lines(
            '------__main__.foo: nopython: AFTER translate_bytecode------',
            'label 0:',
            "    x = arg(0, name=x)     ['x']",
            '    return x',
            '------__main__.foo: nopython: AFTER dead_branch_prune------',
            'label 0:',
            '    return x',
        );

        const results = parser.process(output, {} as any, {} as any);

        expect(results['__main__.foo'][1].irChanged).toBe(true);
        expect(results['__main__.foo'][1].before).toHaveLength(3);
        expect(results['__main__.foo'][1].after).toHaveLength(2);
    });

    it('should handle variable-width dash padding', () => {
        const output = lines(
            '------------------------------__main__.complex_example: nopython: AFTER translate_bytecode------------------------------',
            'label 0:',
            '    return 1',
            '--numba.np.arrayobj.ol_np_empty_like.<locals>.impl: nopython: AFTER translate_bytecode--',
            'label 0:',
            '    return arr',
        );

        const dumps = parser.breakdownOutputIntoPassDumps(output);

        expect(dumps).toHaveLength(2);
        expect(dumps[0].functionName).toBe('__main__.complex_example');
        expect(dumps[1].functionName).toBe('numba.np.arrayobj.ol_np_empty_like.<locals>.impl');
    });

    it('should set machine to false for all passes', () => {
        const output = lines(
            '------__main__.foo: nopython: AFTER translate_bytecode------',
            'label 0:',
            '    return 1',
        );

        const results = parser.process(output, {} as any, {} as any);

        for (const pass of results['__main__.foo']) {
            expect(pass.machine).toBe(false);
        }
    });

    it('should produce before/after pairs correctly', () => {
        const output = lines(
            '------__main__.foo: nopython: AFTER pass_a------',
            'line a1',
            'line a2',
            '------__main__.foo: nopython: AFTER pass_b------',
            'line b1',
            '------__main__.foo: nopython: AFTER pass_c------',
            'line c1',
            'line c2',
            'line c3',
        );

        const results = parser.process(output, {} as any, {} as any);
        const passes = results['__main__.foo'];

        // pass_a: before=[] (first), after=[a1,a2]
        expect(passes[0].name).toBe('pass_a');
        expect(passes[0].before).toHaveLength(0);
        expect(passes[0].after).toHaveLength(2);

        // pass_b: before=[a1,a2], after=[b1]
        expect(passes[1].name).toBe('pass_b');
        expect(passes[1].before).toHaveLength(2);
        expect(passes[1].after).toHaveLength(1);

        // pass_c: before=[b1], after=[c1,c2,c3]
        expect(passes[2].name).toBe('pass_c');
        expect(passes[2].before).toHaveLength(1);
        expect(passes[2].after).toHaveLength(3);
    });

    it('should ignore lines before the first header', () => {
        const output = lines(
            'some random preamble output',
            'another line of noise',
            '------__main__.foo: nopython: AFTER translate_bytecode------',
            'label 0:',
            '    return 1',
        );

        const dumps = parser.breakdownOutputIntoPassDumps(output);

        expect(dumps).toHaveLength(1);
        expect(dumps[0].lines).toHaveLength(2);
    });

    it('should handle object mode headers', () => {
        const output = lines('------__main__.foo: object: AFTER translate_bytecode------', 'label 0:', '    return 1');

        const dumps = parser.breakdownOutputIntoPassDumps(output);

        expect(dumps).toHaveLength(1);
        expect(dumps[0].functionName).toBe('__main__.foo');
    });
});
