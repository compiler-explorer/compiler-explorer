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

function centred(text: string, width = 120): string {
    if (text.length >= width) return text;
    const pad = width - text.length;
    const left = Math.floor(pad / 2);
    return `${'-'.repeat(left)}${text}${'-'.repeat(pad - left)}`;
}

describe('numba-pass-dump-parser', () => {
    const parser = new NumbaPassDumpParser();

    it('should parse a header centred to 120 columns', () => {
        const output = lines(
            centred('<dynamic>.square: nopython: AFTER translate_bytecode'),
            'label 0:',
            '    return num',
            centred('<dynamic>.square: nopython: AFTER Rewrite dynamic raises'),
            'label 0:',
            '    return num',
        );

        const results = parser.process(output);
        const passes = results['<dynamic>.square'];

        expect(passes).toHaveLength(2);
        expect(passes[0].name).toBe('translate_bytecode');
        expect(passes[1].name).toBe('Rewrite dynamic raises');
        expect(passes[0].after.map(line => line.text)).toEqual(['label 0:', '    return num']);
        expect(passes[0].irChanged).toBe(true);
        expect(passes[1].irChanged).toBe(false);
        expect(passes[1].before).toBe(passes[0].after);
    });

    it('should parse headers that str.center leaves with no dashes', () => {
        const name = 'n'.repeat(80);
        const output = lines(
            `<dynamic>.${name}: nopython: AFTER translate_bytecode`,
            'label 0:',
            `<dynamic>.${name}: nopython: AFTER rewrite_semantic_constants`,
            'label 1:',
        );

        const results = parser.process(output);

        expect(Object.keys(results)).toEqual([`<dynamic>.${name}`]);
        expect(results[`<dynamic>.${name}`].map(pass => pass.name)).toEqual([
            'translate_bytecode',
            'rewrite_semantic_constants',
        ]);
    });

    it('should parse a header with a single leading dash', () => {
        const output = lines('-__main__.foo: nopython: AFTER dead_branch_prune-', 'label 0:');

        const results = parser.process(output);

        expect(results['__main__.foo'][0].name).toBe('dead_branch_prune');
        expect(results['__main__.foo'][0].after).toEqual([{text: 'label 0:'}]);
    });

    it('should ignore lines before the first header', () => {
        const output = lines('noise', centred('__main__.foo: object: AFTER translate_bytecode'), 'label 0:');

        const results = parser.process(output);

        expect(Object.keys(results)).toEqual(['__main__.foo']);
        expect(results['__main__.foo'][0].after).toEqual([{text: 'label 0:'}]);
    });

    it('should split a second compilation of the same function', () => {
        const output = lines(
            centred('<dynamic>.square: nopython: AFTER translate_bytecode'),
            'label 0:',
            centred('<dynamic>.square: nopython: AFTER fixup_args'),
            'label 0:',
            centred('<dynamic>.square: nopython: AFTER translate_bytecode'),
            'label 1:',
        );

        const results = parser.process(output);

        expect(Object.keys(results)).toEqual(['<dynamic>.square', '<dynamic>.square [2]']);
        expect(results['<dynamic>.square'].map(pass => pass.name)).toEqual(['translate_bytecode', 'fixup_args']);
        expect(results['<dynamic>.square [2]']).toHaveLength(1);
        expect(results['<dynamic>.square [2]'][0].before).toEqual([]);
        expect(results['<dynamic>.square [2]'][0].after).toEqual([{text: 'label 1:'}]);
    });

    it('should keep one group when other functions are interleaved', () => {
        const output = lines(
            centred('__main__.foo: nopython: AFTER translate_bytecode'),
            'foo',
            centred('__main__.bar: nopython: AFTER translate_bytecode'),
            'bar',
            centred('__main__.foo: nopython: AFTER fixup_args'),
            'foo2',
        );

        const results = parser.process(output);

        expect(Object.keys(results)).toEqual(['__main__.foo', '__main__.bar']);
        expect(results['__main__.foo'].map(pass => pass.name)).toEqual(['translate_bytecode', 'fixup_args']);
        expect(results['__main__.foo'][1].before.map(line => line.text)).toEqual(['foo']);
        expect(results['__main__.foo'][1].after.map(line => line.text)).toEqual(['foo2']);
    });

    it('should accept function names that collide with Object.prototype', () => {
        for (const name of ['constructor', '__proto__', 'toString']) {
            const results = parser.process(lines(`${name}: nopython: AFTER translate_bytecode`, 'label 0:'));
            expect(results[name]).toHaveLength(1);
            expect(results[name][0].after).toEqual([{text: 'label 0:'}]);
        }
    });
});
