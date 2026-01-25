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

    describe('empty list element detection', () => {
        it('should detect double colons in compilers list', () => {
            const content = `compilers=gcc::clang`;
            const parsed = parsePropertiesFileRaw(content, 'test.properties');
            const result = validateRawFile(parsed);

            expect(result.emptyListElements).toHaveLength(1);
        });

        it('should detect leading colons', () => {
            const content = `compilers=:gcc`;
            const parsed = parsePropertiesFileRaw(content, 'test.properties');
            const result = validateRawFile(parsed);

            expect(result.emptyListElements).toHaveLength(1);
        });

        it('should detect trailing colons', () => {
            const content = `compilers=gcc:`;
            const parsed = parsePropertiesFileRaw(content, 'test.properties');
            const result = validateRawFile(parsed);

            expect(result.emptyListElements).toHaveLength(1);
        });

        it('should not report valid compilers list', () => {
            const content = `compilers=gcc:clang:msvc`;
            const parsed = parsePropertiesFileRaw(content, 'test.properties');
            const result = validateRawFile(parsed);

            expect(result.emptyListElements).toHaveLength(0);
        });

        it('should detect empty elements in formatters list', () => {
            const content = `formatters=clangformat::rustfmt`;
            const parsed = parsePropertiesFileRaw(content, 'test.properties');
            const result = validateRawFile(parsed);

            expect(result.emptyListElements).toHaveLength(1);
        });

        it('should detect empty elements in tools list', () => {
            const content = `tools=readelf:nm:`;
            const parsed = parsePropertiesFileRaw(content, 'test.properties');
            const result = validateRawFile(parsed);

            expect(result.emptyListElements).toHaveLength(1);
        });
    });

    describe('typo detection', () => {
        it('should detect compilers. instead of compiler.', () => {
            const content = `compilers.gcc.exe=/path/to/gcc`;
            const parsed = parsePropertiesFileRaw(content, 'test.properties');
            const result = validateRawFile(parsed);

            expect(result.typoCompilers).toHaveLength(1);
            expect(result.typoCompilers[0].text).toContain('compilers.gcc');
        });

        it('should not flag valid compiler. properties', () => {
            const content = `compiler.gcc.exe=/path/to/gcc`;
            const parsed = parsePropertiesFileRaw(content, 'test.properties');
            const result = validateRawFile(parsed);

            expect(result.typoCompilers).toHaveLength(0);
        });

        it('should not flag compilers= list', () => {
            const content = `compilers=gcc:clang`;
            const parsed = parsePropertiesFileRaw(content, 'test.properties');
            const result = validateRawFile(parsed);

            expect(result.typoCompilers).toHaveLength(0);
        });
    });

    describe('suspicious path detection', () => {
        it('should flag paths outside standard locations', () => {
            const content = `compiler.gcc.exe=/usr/bin/gcc`;
            const parsed = parsePropertiesFileRaw(content, 'test.amazon.properties');
            const result = validateRawFile(parsed, {checkSuspiciousPaths: true});

            expect(result.suspiciousPaths).toHaveLength(1);
            expect(result.suspiciousPaths[0].text).toBe('/usr/bin/gcc');
        });

        it('should accept /opt/compiler-explorer paths', () => {
            const content = `compiler.gcc.exe=/opt/compiler-explorer/gcc-12/bin/gcc`;
            const parsed = parsePropertiesFileRaw(content, 'test.amazon.properties');
            const result = validateRawFile(parsed, {checkSuspiciousPaths: true});

            expect(result.suspiciousPaths).toHaveLength(0);
        });

        it('should accept Z:/compilers paths (Windows)', () => {
            const content = `compiler.msvc.exe=Z:/compilers/msvc/cl.exe`;
            const parsed = parsePropertiesFileRaw(content, 'test.amazon.properties');
            const result = validateRawFile(parsed, {checkSuspiciousPaths: true});

            expect(result.suspiciousPaths).toHaveLength(0);
        });

        it('should not check paths in .defaults.properties files', () => {
            const content = `compiler.gcc.exe=/usr/bin/gcc`;
            const parsed = parsePropertiesFileRaw(content, 'c.defaults.properties');
            const result = validateRawFile(parsed, {checkSuspiciousPaths: true});

            expect(result.suspiciousPaths).toHaveLength(0);
        });

        it('should not check paths in .local.properties files', () => {
            const content = `compiler.gcc.exe=/usr/bin/gcc`;
            const parsed = parsePropertiesFileRaw(content, 'c.local.properties');
            const result = validateRawFile(parsed, {checkSuspiciousPaths: true});

            expect(result.suspiciousPaths).toHaveLength(0);
        });

        it('should not check paths when option is disabled', () => {
            const content = `compiler.gcc.exe=/usr/bin/gcc`;
            const parsed = parsePropertiesFileRaw(content, 'test.amazon.properties');
            const result = validateRawFile(parsed, {checkSuspiciousPaths: false});

            expect(result.suspiciousPaths).toHaveLength(0);
        });

        it('should flag suspicious formatter paths', () => {
            const content = `formatter.clangformat.exe=/usr/bin/clang-format`;
            const parsed = parsePropertiesFileRaw(content, 'test.amazon.properties');
            const result = validateRawFile(parsed, {checkSuspiciousPaths: true});

            expect(result.suspiciousPaths).toHaveLength(1);
        });

        it('should flag suspicious tool paths', () => {
            const content = `tools.readelf.exe=/usr/bin/readelf`;
            const parsed = parsePropertiesFileRaw(content, 'test.amazon.properties');
            const result = validateRawFile(parsed, {checkSuspiciousPaths: true});

            expect(result.suspiciousPaths).toHaveLength(1);
        });
    });

    describe('orphaned compiler detection', () => {
        it('should report compilers listed but no .exe defined', () => {
            const content = `
compilers=gcc:clang
compiler.gcc.exe=/opt/compiler-explorer/gcc/bin/gcc
`;
            const parsed = parsePropertiesFileRaw(content, 'test.properties');
            const result = validateRawFile(parsed);

            expect(result.orphanedCompilerExe).toContainEqual(expect.objectContaining({id: 'clang'}));
        });

        it('should report compilers with .exe but not listed', () => {
            const content = `
compilers=gcc
compiler.gcc.exe=/opt/compiler-explorer/gcc/bin/gcc
compiler.clang.exe=/opt/compiler-explorer/clang/bin/clang
`;
            const parsed = parsePropertiesFileRaw(content, 'test.properties');
            const result = validateRawFile(parsed);

            expect(result.orphanedCompilerExe).toContainEqual(expect.objectContaining({id: 'clang'}));
        });

        it('should not report when compilers match', () => {
            const content = `
compilers=gcc:clang
compiler.gcc.exe=/opt/compiler-explorer/gcc/bin/gcc
compiler.clang.exe=/opt/compiler-explorer/clang/bin/clang
`;
            const parsed = parsePropertiesFileRaw(content, 'test.properties');
            const result = validateRawFile(parsed);

            expect(result.orphanedCompilerExe).toHaveLength(0);
        });

        it('should ignore remote compiler references (with @)', () => {
            const content = `
compilers=gcc:remote@host
compiler.gcc.exe=/opt/compiler-explorer/gcc/bin/gcc
`;
            const parsed = parsePropertiesFileRaw(content, 'test.properties');
            const result = validateRawFile(parsed);

            expect(result.orphanedCompilerExe).toHaveLength(0);
        });

        it('should handle alias expanding compilers', () => {
            const content = `
compilers=gcc:oldgcc
compiler.gcc.exe=/opt/compiler-explorer/gcc/bin/gcc
alias=oldgcc
`;
            const parsed = parsePropertiesFileRaw(content, 'test.properties');
            const result = validateRawFile(parsed);

            expect(result.orphanedCompilerExe).toHaveLength(0);
        });
    });

    describe('orphaned group detection', () => {
        it('should report groups referenced but not defined', () => {
            const content = `compilers=&mygroup`;
            const parsed = parsePropertiesFileRaw(content, 'test.properties');
            const result = validateRawFile(parsed);

            expect(result.orphanedGroups).toContainEqual(expect.objectContaining({id: 'mygroup'}));
        });

        it('should accept groups that are defined', () => {
            const content = `
compilers=&mygroup
group.mygroup.compilers=gcc
compiler.gcc.exe=/opt/compiler-explorer/gcc/bin/gcc
`;
            const parsed = parsePropertiesFileRaw(content, 'test.properties');
            const result = validateRawFile(parsed);

            expect(result.orphanedGroups).toHaveLength(0);
        });

        it('should report groups defined but not referenced', () => {
            const content = `
compilers=gcc
compiler.gcc.exe=/opt/compiler-explorer/gcc/bin/gcc
group.unused.compilers=clang
`;
            const parsed = parsePropertiesFileRaw(content, 'test.properties');
            const result = validateRawFile(parsed);

            expect(result.orphanedGroups).toContainEqual(expect.objectContaining({id: 'unused'}));
        });

        it('should handle nested group references', () => {
            const content = `
compilers=&outer
group.outer.compilers=&inner
group.inner.compilers=gcc
compiler.gcc.exe=/opt/compiler-explorer/gcc/bin/gcc
`;
            const parsed = parsePropertiesFileRaw(content, 'test.properties');
            const result = validateRawFile(parsed);

            expect(result.orphanedGroups).toHaveLength(0);
        });
    });

    describe('duplicated reference detection', () => {
        it('should detect duplicate compiler references in same list', () => {
            const content = `compilers=gcc:clang:gcc`;
            const parsed = parsePropertiesFileRaw(content, 'test.properties');
            const result = validateRawFile(parsed);

            expect(result.duplicatedCompilerRefs).toContainEqual(expect.objectContaining({id: 'gcc'}));
        });

        it('should detect duplicate group references', () => {
            const content = `compilers=&mygroup:&mygroup`;
            const parsed = parsePropertiesFileRaw(content, 'test.properties');
            const result = validateRawFile(parsed);

            expect(result.duplicatedGroupRefs).toContainEqual(expect.objectContaining({id: 'mygroup'}));
        });

        it('should not flag unique references', () => {
            const content = `compilers=gcc:clang:msvc`;
            const parsed = parsePropertiesFileRaw(content, 'test.properties');
            const result = validateRawFile(parsed);

            expect(result.duplicatedCompilerRefs).toHaveLength(0);
        });
    });
});
