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

import fs from 'node:fs';
import path from 'node:path';

import {afterAll, beforeAll, describe, expect, it} from 'vitest';

import {
    filterDisabled,
    parseCompilersList,
    parsePropertiesFileRaw,
    validateCrossFileCompilerIds,
    validateRawFile,
} from '../lib/properties-validator.js';

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

        it('should collect parse errors for invalid lines', () => {
            const content = `
foo=bar
this is not valid
baz=qux
`;
            const parsed = parsePropertiesFileRaw(content, 'test.properties');

            expect(parsed.parseErrors).toHaveLength(1);
            expect(parsed.parseErrors[0].line).toBe(3);
            expect(parsed.parseErrors[0].text).toBe('this is not valid');
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

    describe('invalid property format detection', () => {
        it('should report lines without equals sign', () => {
            const content = `
foo=bar
this is not valid
baz=qux
`;
            const parsed = parsePropertiesFileRaw(content, 'test.properties');
            const result = validateRawFile(parsed);

            expect(result.invalidPropertyFormat).toHaveLength(1);
            expect(result.invalidPropertyFormat[0].text).toBe('this is not valid');
        });

        it('should not report valid properties', () => {
            const content = `
foo=bar
baz=qux
`;
            const parsed = parsePropertiesFileRaw(content, 'test.properties');
            const result = validateRawFile(parsed);

            expect(result.invalidPropertyFormat).toHaveLength(0);
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

    describe('orphaned formatter detection', () => {
        it('should report formatters listed but not defined', () => {
            const content = `
formatters=clangformat:rustfmt
formatter.clangformat.exe=/opt/compiler-explorer/clang-format
`;
            const parsed = parsePropertiesFileRaw(content, 'test.properties');
            const result = validateRawFile(parsed);

            expect(result.orphanedFormatterExe).toContainEqual(expect.objectContaining({id: 'rustfmt'}));
        });

        it('should report formatters defined but not listed', () => {
            const content = `
formatters=clangformat
formatter.clangformat.exe=/opt/compiler-explorer/clang-format
formatter.rustfmt.exe=/opt/compiler-explorer/rustfmt
`;
            const parsed = parsePropertiesFileRaw(content, 'test.properties');
            const result = validateRawFile(parsed);

            expect(result.orphanedFormatterExe).toContainEqual(expect.objectContaining({id: 'rustfmt'}));
        });
    });

    describe('orphaned tool detection', () => {
        it('should report tools listed but not defined', () => {
            const content = `
tools=readelf:nm
tools.readelf.exe=/usr/bin/readelf
`;
            const parsed = parsePropertiesFileRaw(content, 'test.properties');
            const result = validateRawFile(parsed);

            expect(result.orphanedToolExe).toContainEqual(expect.objectContaining({id: 'nm'}));
        });

        it('should report tools defined but not listed', () => {
            const content = `
tools=readelf
tools.readelf.exe=/usr/bin/readelf
tools.nm.exe=/usr/bin/nm
`;
            const parsed = parsePropertiesFileRaw(content, 'test.properties');
            const result = validateRawFile(parsed);

            expect(result.orphanedToolExe).toContainEqual(expect.objectContaining({id: 'nm'}));
        });
    });

    describe('orphaned library detection', () => {
        it('should report libs listed but versions not defined', () => {
            const content = `
libs=boost:fmt
libs.boost.versions=1.80
libs.boost.versions.1.80.version=1.80.0
`;
            const parsed = parsePropertiesFileRaw(content, 'test.properties');
            const result = validateRawFile(parsed);

            expect(result.orphanedLibIds).toContainEqual(expect.objectContaining({id: 'fmt'}));
        });

        it('should report lib versions listed but not defined', () => {
            const content = `
libs=boost
libs.boost.versions=1.80:1.81
libs.boost.versions.1.80.version=1.80.0
`;
            const parsed = parsePropertiesFileRaw(content, 'test.properties');
            const result = validateRawFile(parsed);

            expect(result.orphanedLibVersions).toContainEqual(expect.objectContaining({id: 'boost 1.81'}));
        });
    });

    describe('invalid default compiler detection', () => {
        it('should report default compiler not in list', () => {
            const content = `
compilers=gcc:clang
defaultCompiler=msvc
compiler.gcc.exe=/opt/compiler-explorer/gcc
compiler.clang.exe=/opt/compiler-explorer/clang
`;
            const parsed = parsePropertiesFileRaw(content, 'test.properties');
            const result = validateRawFile(parsed);

            expect(result.invalidDefaultCompiler).toContainEqual(expect.objectContaining({id: 'msvc'}));
        });

        it('should accept valid default compiler', () => {
            const content = `
compilers=gcc:clang
defaultCompiler=gcc
compiler.gcc.exe=/opt/compiler-explorer/gcc
compiler.clang.exe=/opt/compiler-explorer/clang
`;
            const parsed = parsePropertiesFileRaw(content, 'test.properties');
            const result = validateRawFile(parsed);

            expect(result.invalidDefaultCompiler).toHaveLength(0);
        });

        it('should report default compiler when there is no compilers list', () => {
            const content = `
defaultCompiler=gcc
compiler.gcc.exe=/opt/compiler-explorer/gcc
`;
            const parsed = parsePropertiesFileRaw(content, 'test.properties');
            const result = validateRawFile(parsed);

            expect(result.invalidDefaultCompiler).toContainEqual(expect.objectContaining({id: 'gcc'}));
        });
    });

    describe('disabled allowlist', () => {
        it('should filter disabled IDs from orphaned compilers', () => {
            const content = `
# Disabled: orphan
compilers=gcc:orphan
compiler.gcc.exe=/opt/compiler-explorer/gcc
`;
            const parsed = parsePropertiesFileRaw(content, 'test.properties');
            const result = validateRawFile(parsed);
            const filtered = filterDisabled(result, parsed.disabledIds);

            expect(result.orphanedCompilerExe).toContainEqual(expect.objectContaining({id: 'orphan'}));
            expect(filtered.orphanedCompilerExe).not.toContainEqual(expect.objectContaining({id: 'orphan'}));
        });

        it('should handle multiple disabled IDs', () => {
            const content = `
# Disabled: orphan1 orphan2
compilers=gcc:orphan1:orphan2
compiler.gcc.exe=/opt/compiler-explorer/gcc
`;
            const parsed = parsePropertiesFileRaw(content, 'test.properties');
            const result = validateRawFile(parsed);
            const filtered = filterDisabled(result, parsed.disabledIds);

            expect(result.orphanedCompilerExe).toContainEqual(expect.objectContaining({id: 'orphan1'}));
            expect(result.orphanedCompilerExe).toContainEqual(expect.objectContaining({id: 'orphan2'}));
            expect(filtered.orphanedCompilerExe).not.toContainEqual(expect.objectContaining({id: 'orphan1'}));
            expect(filtered.orphanedCompilerExe).not.toContainEqual(expect.objectContaining({id: 'orphan2'}));
        });

        it('should support Disable: variant spelling', () => {
            const content = `
# Disable: orphan
compilers=orphan
`;
            const parsed = parsePropertiesFileRaw(content, 'test.properties');

            expect(parsed.disabledIds).toContain('orphan');
        });
    });

    describe('no compilers list detection', () => {
        it('should flag language files with compiler definitions but no compilers=', () => {
            const content = `
compiler.gcc.exe=/opt/compiler-explorer/gcc/bin/gcc
compiler.gcc.name=GCC
`;
            const parsed = parsePropertiesFileRaw(content, 'c++.amazon.properties');
            const result = validateRawFile(parsed);

            expect(result.noCompilersList).toBe(true);
        });

        it('should not flag files with compilers=', () => {
            const content = `
compilers=gcc
compiler.gcc.exe=/opt/compiler-explorer/gcc/bin/gcc
`;
            const parsed = parsePropertiesFileRaw(content, 'c++.amazon.properties');
            const result = validateRawFile(parsed);

            expect(result.noCompilersList).toBe(false);
        });

        it('should not flag files with group definitions', () => {
            const content = `
group.mygroup.compilers=gcc
compiler.gcc.exe=/opt/compiler-explorer/gcc/bin/gcc
`;
            const parsed = parsePropertiesFileRaw(content, 'c++.amazon.properties');
            const result = validateRawFile(parsed);

            expect(result.noCompilersList).toBe(false);
        });

        it('should not flag defaults files', () => {
            const content = `
compiler.gcc.exe=/opt/compiler-explorer/gcc/bin/gcc
`;
            const parsed = parsePropertiesFileRaw(content, 'c++.defaults.properties');
            const result = validateRawFile(parsed);

            expect(result.noCompilersList).toBe(false);
        });

        it('should not flag compiler-explorer config files', () => {
            const content = `
someOtherProperty=value
`;
            const parsed = parsePropertiesFileRaw(content, 'compiler-explorer.amazon.properties');
            const result = validateRawFile(parsed);

            expect(result.noCompilersList).toBe(false);
        });

        it('should not flag execution. files', () => {
            const content = `
compiler.gcc.exe=/opt/compiler-explorer/gcc/bin/gcc
`;
            const parsed = parsePropertiesFileRaw(content, 'execution.amazon.properties');
            const result = validateRawFile(parsed);

            expect(result.noCompilersList).toBe(false);
        });

        it('should not flag aws. files', () => {
            const content = `
someProperty=value
`;
            const parsed = parsePropertiesFileRaw(content, 'aws.properties');
            const result = validateRawFile(parsed);

            expect(result.noCompilersList).toBe(false);
        });

        it('should not flag asm-docs. files', () => {
            const content = `
someProperty=value
`;
            const parsed = parsePropertiesFileRaw(content, 'asm-docs.properties');
            const result = validateRawFile(parsed);

            expect(result.noCompilersList).toBe(false);
        });

        it('should not flag builtin. files', () => {
            const content = `
someProperty=value
`;
            const parsed = parsePropertiesFileRaw(content, 'builtin.amazon.properties');
            const result = validateRawFile(parsed);

            expect(result.noCompilersList).toBe(false);
        });

        it('should not flag files with no compiler definitions', () => {
            const content = `
someProperty=value
`;
            const parsed = parsePropertiesFileRaw(content, 'c++.amazon.properties');
            const result = validateRawFile(parsed);

            expect(result.noCompilersList).toBe(false);
        });
    });

    describe('cross-file compiler ID validation', () => {
        it('should detect compiler IDs defined in multiple files', () => {
            const file1Content = `
compiler.gcc.exe=/opt/compiler-explorer/gcc-12/bin/gcc
compiler.gcc.name=GCC 12
`;
            const file2Content = `
compiler.gcc.exe=/opt/compiler-explorer/gcc-13/bin/gcc
compiler.gcc.name=GCC 13
`;
            const parsed1 = parsePropertiesFileRaw(file1Content, 'c++.amazon.properties');
            const parsed2 = parsePropertiesFileRaw(file2Content, 'c.amazon.properties');

            const result = validateCrossFileCompilerIds([
                {filename: 'c++.amazon.properties', parsed: parsed1},
                {filename: 'c.amazon.properties', parsed: parsed2},
            ]);

            expect(result.duplicateCompilerIds.has('gcc')).toBe(true);
            expect(result.duplicateCompilerIds.get('gcc')).toHaveLength(2);
        });

        it('should not flag unique compiler IDs across files', () => {
            const file1Content = `compiler.gcc.exe=/opt/compiler-explorer/gcc/bin/gcc`;
            const file2Content = `compiler.clang.exe=/opt/compiler-explorer/clang/bin/clang`;

            const parsed1 = parsePropertiesFileRaw(file1Content, 'c++.amazon.properties');
            const parsed2 = parsePropertiesFileRaw(file2Content, 'c.amazon.properties');

            const result = validateCrossFileCompilerIds([
                {filename: 'c++.amazon.properties', parsed: parsed1},
                {filename: 'c.amazon.properties', parsed: parsed2},
            ]);

            expect(result.duplicateCompilerIds.size).toBe(0);
        });

        it('should not count same ID multiple times within single file', () => {
            const content = `
compiler.gcc.exe=/opt/compiler-explorer/gcc/bin/gcc
compiler.gcc.name=GCC
compiler.gcc.semver=12.0
`;
            const parsed = parsePropertiesFileRaw(content, 'c++.amazon.properties');

            const result = validateCrossFileCompilerIds([{filename: 'c++.amazon.properties', parsed}]);

            expect(result.duplicateCompilerIds.size).toBe(0);
        });
    });
});

describe('Real config validation', () => {
    const configDir = 'etc/config';
    const checkLocal = process.env.CHECK_LOCAL_PROPS === 'true';
    let propertyFiles: Array<{filename: string; parsed: ReturnType<typeof parsePropertiesFileRaw>}> = [];

    beforeAll(() => {
        const files = fs.readdirSync(configDir);
        propertyFiles = files
            .filter(f => {
                if (!f.endsWith('.properties')) return false;
                if (f.endsWith('.local.properties')) return checkLocal;
                return true; // All .properties files except .local.properties (unless CHECK_LOCAL_PROPS=true)
            })
            .map(filename => {
                const content = fs.readFileSync(path.join(configDir, filename), 'utf8');
                return {
                    filename,
                    parsed: parsePropertiesFileRaw(content, filename),
                };
            });
    });

    afterAll(() => {
        propertyFiles = [];
    });

    it('should have property files to validate', () => {
        expect(propertyFiles.length).toBeGreaterThan(0);
    });

    it('should have no duplicate keys in property files', () => {
        const filesWithDuplicates: Array<{file: string; duplicates: string[]}> = [];

        for (const {filename, parsed} of propertyFiles) {
            const result = validateRawFile(parsed);
            const filtered = filterDisabled(result, parsed.disabledIds);

            if (filtered.duplicateKeys.length > 0) {
                filesWithDuplicates.push({
                    file: filename,
                    duplicates: filtered.duplicateKeys.map(d => d.id ?? d.text),
                });
            }
        }

        expect(filesWithDuplicates, `Files with duplicate keys: ${JSON.stringify(filesWithDuplicates)}`).toEqual([]);
    });

    it('should have no empty list elements', () => {
        const filesWithEmpty: Array<{file: string; issues: string[]}> = [];

        for (const {filename, parsed} of propertyFiles) {
            const result = validateRawFile(parsed);

            if (result.emptyListElements.length > 0) {
                filesWithEmpty.push({
                    file: filename,
                    issues: result.emptyListElements.map(e => e.text),
                });
            }
        }

        expect(filesWithEmpty, `Files with empty list elements: ${JSON.stringify(filesWithEmpty)}`).toEqual([]);
    });

    it('should have no typo compilers (compilers. instead of compiler.)', () => {
        const filesWithTypos: Array<{file: string; typos: string[]}> = [];

        for (const {filename, parsed} of propertyFiles) {
            const result = validateRawFile(parsed);
            const filtered = filterDisabled(result, parsed.disabledIds);

            if (filtered.typoCompilers.length > 0) {
                filesWithTypos.push({
                    file: filename,
                    typos: filtered.typoCompilers.map(t => t.text),
                });
            }
        }

        expect(filesWithTypos, `Files with typo compilers: ${JSON.stringify(filesWithTypos)}`).toEqual([]);
    });

    it('should have no duplicate compiler IDs across amazon property files', () => {
        // Cross-file duplicate check only applies to amazon files
        // Local files are expected to override/mirror amazon config
        const amazonOnly = propertyFiles.filter(f => f.filename.includes('amazon'));
        const result = validateCrossFileCompilerIds(amazonOnly);

        if (result.duplicateCompilerIds.size > 0) {
            const duplicates = Object.fromEntries(result.duplicateCompilerIds);
            expect.fail(`Duplicate compiler IDs found: ${JSON.stringify(duplicates, null, 2)}`);
        }
    });

    it('should have no invalid property format errors', () => {
        const filesWithErrors: Array<{file: string; errors: string[]}> = [];

        for (const {filename, parsed} of propertyFiles) {
            const result = validateRawFile(parsed);

            if (result.invalidPropertyFormat.length > 0) {
                filesWithErrors.push({
                    file: filename,
                    errors: result.invalidPropertyFormat.map(e => `Line ${e.line}: ${e.text}`),
                });
            }
        }

        expect(filesWithErrors, `Files with invalid property format: ${JSON.stringify(filesWithErrors)}`).toEqual([]);
    });

    it('should have no language files missing compilers list', () => {
        const filesWithMissingList: string[] = [];

        for (const {filename, parsed} of propertyFiles) {
            const result = validateRawFile(parsed);

            if (result.noCompilersList) {
                filesWithMissingList.push(filename);
            }
        }

        expect(filesWithMissingList, `Files missing compilers list: ${filesWithMissingList.join(', ')}`).toEqual([]);
    });

    it('should have no orphaned compilers (exe)', () => {
        const filesWithOrphans: Array<{file: string; orphans: string[]}> = [];

        for (const {filename, parsed} of propertyFiles) {
            const result = validateRawFile(parsed);
            const filtered = filterDisabled(result, parsed.disabledIds);

            if (filtered.orphanedCompilerExe.length > 0) {
                filesWithOrphans.push({
                    file: filename,
                    orphans: filtered.orphanedCompilerExe.map(o => o.id ?? o.text),
                });
            }
        }

        expect(filesWithOrphans, `Files with orphaned compiler .exe: ${JSON.stringify(filesWithOrphans)}`).toEqual([]);
    });

    it('should have no orphaned compilers (ID)', () => {
        const filesWithOrphans: Array<{file: string; orphans: string[]}> = [];

        for (const {filename, parsed} of propertyFiles) {
            const result = validateRawFile(parsed);
            const filtered = filterDisabled(result, parsed.disabledIds);

            if (filtered.orphanedCompilerId.length > 0) {
                filesWithOrphans.push({
                    file: filename,
                    orphans: filtered.orphanedCompilerId.map(o => o.id ?? o.text),
                });
            }
        }

        expect(filesWithOrphans, `Files with orphaned compiler IDs: ${JSON.stringify(filesWithOrphans)}`).toEqual([]);
    });

    it('should have no orphaned groups', () => {
        const filesWithOrphans: Array<{file: string; orphans: string[]}> = [];

        for (const {filename, parsed} of propertyFiles) {
            const result = validateRawFile(parsed);
            const filtered = filterDisabled(result, parsed.disabledIds);

            if (filtered.orphanedGroups.length > 0) {
                filesWithOrphans.push({
                    file: filename,
                    orphans: filtered.orphanedGroups.map(o => o.id ?? o.text),
                });
            }
        }

        expect(filesWithOrphans, `Files with orphaned groups: ${JSON.stringify(filesWithOrphans)}`).toEqual([]);
    });

    it('should have no orphaned formatters (exe)', () => {
        const filesWithOrphans: Array<{file: string; orphans: string[]}> = [];

        for (const {filename, parsed} of propertyFiles) {
            const result = validateRawFile(parsed);
            const filtered = filterDisabled(result, parsed.disabledIds);

            if (filtered.orphanedFormatterExe.length > 0) {
                filesWithOrphans.push({
                    file: filename,
                    orphans: filtered.orphanedFormatterExe.map(o => o.id ?? o.text),
                });
            }
        }

        expect(filesWithOrphans, `Files with orphaned formatter .exe: ${JSON.stringify(filesWithOrphans)}`).toEqual([]);
    });

    it('should have no orphaned formatters (ID)', () => {
        const filesWithOrphans: Array<{file: string; orphans: string[]}> = [];

        for (const {filename, parsed} of propertyFiles) {
            const result = validateRawFile(parsed);
            const filtered = filterDisabled(result, parsed.disabledIds);

            if (filtered.orphanedFormatterId.length > 0) {
                filesWithOrphans.push({
                    file: filename,
                    orphans: filtered.orphanedFormatterId.map(o => o.id ?? o.text),
                });
            }
        }

        expect(filesWithOrphans, `Files with orphaned formatter IDs: ${JSON.stringify(filesWithOrphans)}`).toEqual([]);
    });

    it('should have no orphaned tools (exe)', () => {
        const filesWithOrphans: Array<{file: string; orphans: string[]}> = [];

        for (const {filename, parsed} of propertyFiles) {
            const result = validateRawFile(parsed);
            const filtered = filterDisabled(result, parsed.disabledIds);

            if (filtered.orphanedToolExe.length > 0) {
                filesWithOrphans.push({
                    file: filename,
                    orphans: filtered.orphanedToolExe.map(o => o.id ?? o.text),
                });
            }
        }

        expect(filesWithOrphans, `Files with orphaned tool .exe: ${JSON.stringify(filesWithOrphans)}`).toEqual([]);
    });

    it('should have no orphaned tools (ID)', () => {
        const filesWithOrphans: Array<{file: string; orphans: string[]}> = [];

        for (const {filename, parsed} of propertyFiles) {
            const result = validateRawFile(parsed);
            const filtered = filterDisabled(result, parsed.disabledIds);

            if (filtered.orphanedToolId.length > 0) {
                filesWithOrphans.push({
                    file: filename,
                    orphans: filtered.orphanedToolId.map(o => o.id ?? o.text),
                });
            }
        }

        expect(filesWithOrphans, `Files with orphaned tool IDs: ${JSON.stringify(filesWithOrphans)}`).toEqual([]);
    });

    it('should have no orphaned lib IDs', () => {
        const filesWithOrphans: Array<{file: string; orphans: string[]}> = [];

        for (const {filename, parsed} of propertyFiles) {
            const result = validateRawFile(parsed);
            const filtered = filterDisabled(result, parsed.disabledIds);

            if (filtered.orphanedLibIds.length > 0) {
                filesWithOrphans.push({
                    file: filename,
                    orphans: filtered.orphanedLibIds.map(o => o.id ?? o.text),
                });
            }
        }

        expect(filesWithOrphans, `Files with orphaned lib IDs: ${JSON.stringify(filesWithOrphans)}`).toEqual([]);
    });

    it('should have no orphaned lib versions', () => {
        const filesWithOrphans: Array<{file: string; orphans: string[]}> = [];

        for (const {filename, parsed} of propertyFiles) {
            const result = validateRawFile(parsed);
            const filtered = filterDisabled(result, parsed.disabledIds);

            if (filtered.orphanedLibVersions.length > 0) {
                filesWithOrphans.push({
                    file: filename,
                    orphans: filtered.orphanedLibVersions.map(o => o.id ?? o.text),
                });
            }
        }

        expect(filesWithOrphans, `Files with orphaned lib versions: ${JSON.stringify(filesWithOrphans)}`).toEqual([]);
    });

    it('should have no duplicated compiler references', () => {
        const filesWithDuplicates: Array<{file: string; duplicates: string[]}> = [];

        for (const {filename, parsed} of propertyFiles) {
            const result = validateRawFile(parsed);
            const filtered = filterDisabled(result, parsed.disabledIds);

            if (filtered.duplicatedCompilerRefs.length > 0) {
                filesWithDuplicates.push({
                    file: filename,
                    duplicates: filtered.duplicatedCompilerRefs.map(d => d.id ?? d.text),
                });
            }
        }

        expect(
            filesWithDuplicates,
            `Files with duplicated compiler references: ${JSON.stringify(filesWithDuplicates)}`,
        ).toEqual([]);
    });

    it('should have no duplicated group references', () => {
        const filesWithDuplicates: Array<{file: string; duplicates: string[]}> = [];

        for (const {filename, parsed} of propertyFiles) {
            const result = validateRawFile(parsed);
            const filtered = filterDisabled(result, parsed.disabledIds);

            if (filtered.duplicatedGroupRefs.length > 0) {
                filesWithDuplicates.push({
                    file: filename,
                    duplicates: filtered.duplicatedGroupRefs.map(d => d.id ?? d.text),
                });
            }
        }

        expect(
            filesWithDuplicates,
            `Files with duplicated group references: ${JSON.stringify(filesWithDuplicates)}`,
        ).toEqual([]);
    });

    it('should have no invalid default compilers', () => {
        const filesWithInvalid: Array<{file: string; invalids: string[]}> = [];

        for (const {filename, parsed} of propertyFiles) {
            const result = validateRawFile(parsed);
            const filtered = filterDisabled(result, parsed.disabledIds);

            if (filtered.invalidDefaultCompiler.length > 0) {
                filesWithInvalid.push({
                    file: filename,
                    invalids: filtered.invalidDefaultCompiler.map(i => i.id ?? i.text),
                });
            }
        }

        expect(filesWithInvalid, `Files with invalid default compiler: ${JSON.stringify(filesWithInvalid)}`).toEqual(
            [],
        );
    });

    it('should have no suspicious paths in amazon properties', () => {
        const filesWithSuspicious: Array<{file: string; paths: string[]}> = [];
        const amazonFiles = propertyFiles.filter(f => f.filename.includes('amazon'));

        for (const {filename, parsed} of amazonFiles) {
            const result = validateRawFile(parsed, {checkSuspiciousPaths: true});
            const filtered = filterDisabled(result, parsed.disabledIds);

            if (filtered.suspiciousPaths.length > 0) {
                filesWithSuspicious.push({
                    file: filename,
                    paths: filtered.suspiciousPaths.map(p => `${p.id}: ${p.text}`),
                });
            }
        }

        expect(filesWithSuspicious, `Files with suspicious paths: ${JSON.stringify(filesWithSuspicious)}`).toEqual([]);
    });
});
