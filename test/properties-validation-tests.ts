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

import {parseCompilersList, parsePropertiesFileRaw, validateRawFile} from '../lib/properties-validator.js';

describe('Properties Validator', () => {
    describe('parseCompilersList', () => {
        it('should parse empty string', () => {
            expect(parseCompilersList('')).toEqual([]);
        });

        it('should parse single id', () => {
            expect(parseCompilersList('gcc')).toEqual(['gcc']);
        });

        it('should parse colon-separated ids', () => {
            expect(parseCompilersList('gcc:clang:msvc')).toEqual(['gcc', 'clang', 'msvc']);
        });

        it('should filter empty elements', () => {
            expect(parseCompilersList('gcc::clang')).toEqual(['gcc', 'clang']);
        });

        it('should handle group references', () => {
            expect(parseCompilersList('&mygroup:gcc')).toEqual(['&mygroup', 'gcc']);
        });

        it('should handle remote references', () => {
            expect(parseCompilersList('gcc@remote:clang')).toEqual(['gcc@remote', 'clang']);
        });
    });

    describe('parsePropertiesFileRaw', () => {
        it('should parse basic properties', () => {
            const content = `
foo=bar
baz=qux
`;
            const parsed = parsePropertiesFileRaw(content, 'test.properties');

            expect(parsed.filename).toBe('test.properties');
            expect(parsed.properties).toHaveLength(2);
            expect(parsed.properties[0]).toEqual({key: 'foo', value: 'bar', line: 2});
            expect(parsed.properties[1]).toEqual({key: 'baz', value: 'qux', line: 3});
        });

        it('should skip comments', () => {
            const content = `
# This is a comment
foo=bar
`;
            const parsed = parsePropertiesFileRaw(content, 'test.properties');

            expect(parsed.properties).toHaveLength(1);
            expect(parsed.properties[0].key).toBe('foo');
        });

        it('should parse Disabled: comments', () => {
            const content = `
# Disabled: orphan1 orphan2
foo=bar
`;
            const parsed = parsePropertiesFileRaw(content, 'test.properties');

            expect(parsed.disabledIds).toContain('orphan1');
            expect(parsed.disabledIds).toContain('orphan2');
        });

        it('should handle values with equals signs', () => {
            const content = `options=-O2 -DFOO=bar`;
            const parsed = parsePropertiesFileRaw(content, 'test.properties');

            expect(parsed.properties[0].value).toBe('-O2 -DFOO=bar');
        });
    });

    describe('duplicate key detection', () => {
        it('should report duplicate keys', () => {
            const content = `
foo=bar
foo=baz
`;
            const parsed = parsePropertiesFileRaw(content, 'test.properties');
            const result = validateRawFile(parsed);

            expect(result.duplicateKeys).toHaveLength(1);
            expect(result.duplicateKeys[0].id).toBe('foo');
        });

        it('should not report unique keys as duplicates', () => {
            const content = `
foo=bar
bar=baz
`;
            const parsed = parsePropertiesFileRaw(content, 'test.properties');
            const result = validateRawFile(parsed);

            expect(result.duplicateKeys).toHaveLength(0);
        });
    });
});
