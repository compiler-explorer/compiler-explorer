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
    type RawFileValidationResult,
    type RawValidatorOptions,
    validateCrossFileCompilerIds,
    validateRawFile,
} from '../lib/properties-validator.js';

function validate(content: string, filename = 'test.properties', options?: RawValidatorOptions) {
    return validateRawFile(parsePropertiesFileRaw(content, filename), options);
}

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
            const result = validate('foo=bar\nfoo=baz');
            expect(result.duplicateKeys).toHaveLength(1);
            expect(result.duplicateKeys[0].id).toBe('foo');
        });

        it('should not report unique keys as duplicates', () => {
            expect(validate('foo=bar\nbar=baz').duplicateKeys).toHaveLength(0);
        });
    });

    describe('empty list element detection', () => {
        it.each([
            ['double colons in compilers', 'compilers=gcc::clang'],
            ['leading colons', 'compilers=:gcc'],
            ['trailing colons', 'compilers=gcc:'],
            ['empty elements in formatters', 'formatters=clangformat::rustfmt'],
            ['empty elements in tools', 'tools=readelf:nm:'],
        ])('should detect %s', (_, content) => {
            expect(validate(content).emptyListElements).toHaveLength(1);
        });

        it('should not report valid compilers list', () => {
            expect(validate('compilers=gcc:clang:msvc').emptyListElements).toHaveLength(0);
        });
    });

    describe('invalid property format detection', () => {
        it('should report lines without equals sign', () => {
            const result = validate('foo=bar\nthis is not valid\nbaz=qux');
            expect(result.invalidPropertyFormat).toHaveLength(1);
            expect(result.invalidPropertyFormat[0].text).toBe('this is not valid');
        });

        it('should not report valid properties', () => {
            expect(validate('foo=bar\nbaz=qux').invalidPropertyFormat).toHaveLength(0);
        });
    });

    describe('typo detection', () => {
        it('should detect compilers. instead of compiler.', () => {
            const result = validate('compilers.gcc.exe=/path/to/gcc');
            expect(result.typoCompilers).toHaveLength(1);
            expect(result.typoCompilers[0].text).toContain('compilers.gcc');
        });

        it.each([
            ['valid compiler. properties', 'compiler.gcc.exe=/path/to/gcc'],
            ['compilers= list', 'compilers=gcc:clang'],
        ])('should not flag %s', (_, content) => {
            expect(validate(content).typoCompilers).toHaveLength(0);
        });
    });

    describe('suspicious path detection', () => {
        it('should flag paths outside standard locations', () => {
            const result = validate('compiler.gcc.exe=/usr/bin/gcc', 'test.amazon.properties', {
                checkSuspiciousPaths: true,
            });
            expect(result.suspiciousPaths).toHaveLength(1);
            expect(result.suspiciousPaths[0].text).toBe('/usr/bin/gcc');
        });

        it.each([
            ['/opt/compiler-explorer paths', 'compiler.gcc.exe=/opt/compiler-explorer/gcc-12/bin/gcc'],
            ['Z:/compilers paths (Windows)', 'compiler.msvc.exe=Z:/compilers/msvc/cl.exe'],
        ])('should accept %s', (_, content) => {
            expect(
                validate(content, 'test.amazon.properties', {checkSuspiciousPaths: true}).suspiciousPaths,
            ).toHaveLength(0);
        });

        it.each([
            ['.defaults.properties files', 'c.defaults.properties'],
            ['.local.properties files', 'c.local.properties'],
        ])('should not check paths in %s', (_, filename) => {
            expect(
                validate('compiler.gcc.exe=/usr/bin/gcc', filename, {checkSuspiciousPaths: true}).suspiciousPaths,
            ).toHaveLength(0);
        });

        it('should not check paths when option is disabled', () => {
            expect(
                validate('compiler.gcc.exe=/usr/bin/gcc', 'test.amazon.properties', {checkSuspiciousPaths: false})
                    .suspiciousPaths,
            ).toHaveLength(0);
        });

        it.each([
            ['formatter paths', 'formatter.clangformat.exe=/usr/bin/clang-format'],
            ['tool paths', 'tools.readelf.exe=/usr/bin/readelf'],
        ])('should flag suspicious %s', (_, content) => {
            expect(
                validate(content, 'test.amazon.properties', {checkSuspiciousPaths: true}).suspiciousPaths,
            ).toHaveLength(1);
        });
    });

    describe('orphaned compiler detection', () => {
        it('should report compilers listed but no .exe defined', () => {
            const result = validate(`compilers=gcc:clang\ncompiler.gcc.exe=/opt/compiler-explorer/gcc/bin/gcc`);
            expect(result.orphanedCompilerExe).toContainEqual(expect.objectContaining({id: 'clang'}));
        });

        it('should report compilers with .exe but not listed', () => {
            const result = validate(
                `compilers=gcc\ncompiler.gcc.exe=/opt/compiler-explorer/gcc/bin/gcc\ncompiler.clang.exe=/opt/compiler-explorer/clang/bin/clang`,
            );
            expect(result.orphanedCompilerExe).toContainEqual(expect.objectContaining({id: 'clang'}));
        });

        it('should not report when compilers match', () => {
            const result = validate(
                `compilers=gcc:clang\ncompiler.gcc.exe=/opt/compiler-explorer/gcc/bin/gcc\ncompiler.clang.exe=/opt/compiler-explorer/clang/bin/clang`,
            );
            expect(result.orphanedCompilerExe).toHaveLength(0);
        });

        it('should ignore remote compiler references (with @)', () => {
            const result = validate(`compilers=gcc:remote@host\ncompiler.gcc.exe=/opt/compiler-explorer/gcc/bin/gcc`);
            expect(result.orphanedCompilerExe).toHaveLength(0);
        });

        it('should handle alias expanding compilers', () => {
            const result = validate(
                `compilers=gcc:oldgcc\ncompiler.gcc.exe=/opt/compiler-explorer/gcc/bin/gcc\nalias=oldgcc`,
            );
            expect(result.orphanedCompilerExe).toHaveLength(0);
        });
    });

    describe('orphaned group detection', () => {
        it('should report groups referenced but not defined', () => {
            expect(validate('compilers=&mygroup').orphanedGroups).toContainEqual(
                expect.objectContaining({id: 'mygroup'}),
            );
        });

        it('should accept groups that are defined', () => {
            const result = validate(
                `compilers=&mygroup\ngroup.mygroup.compilers=gcc\ncompiler.gcc.exe=/opt/compiler-explorer/gcc/bin/gcc`,
            );
            expect(result.orphanedGroups).toHaveLength(0);
        });

        it('should report groups defined but not referenced', () => {
            const result = validate(
                `compilers=gcc\ncompiler.gcc.exe=/opt/compiler-explorer/gcc/bin/gcc\ngroup.unused.compilers=clang`,
            );
            expect(result.orphanedGroups).toContainEqual(expect.objectContaining({id: 'unused'}));
        });

        it('should handle nested group references', () => {
            const result = validate(
                `compilers=&outer\ngroup.outer.compilers=&inner\ngroup.inner.compilers=gcc\ncompiler.gcc.exe=/opt/compiler-explorer/gcc/bin/gcc`,
            );
            expect(result.orphanedGroups).toHaveLength(0);
        });
    });

    describe('duplicated reference detection', () => {
        it('should detect duplicate compiler references in same list', () => {
            expect(validate('compilers=gcc:clang:gcc').duplicatedCompilerRefs).toContainEqual(
                expect.objectContaining({id: 'gcc'}),
            );
        });

        it('should detect duplicate group references', () => {
            expect(validate('compilers=&mygroup:&mygroup').duplicatedGroupRefs).toContainEqual(
                expect.objectContaining({id: 'mygroup'}),
            );
        });

        it('should not flag unique references', () => {
            expect(validate('compilers=gcc:clang:msvc').duplicatedCompilerRefs).toHaveLength(0);
        });
    });

    describe('orphaned formatter detection', () => {
        it('should report formatters listed but not defined', () => {
            const result = validate(
                `formatters=clangformat:rustfmt\nformatter.clangformat.exe=/opt/compiler-explorer/clang-format`,
            );
            expect(result.orphanedFormatterExe).toContainEqual(expect.objectContaining({id: 'rustfmt'}));
        });

        it('should report formatters defined but not listed', () => {
            const result = validate(
                `formatters=clangformat\nformatter.clangformat.exe=/opt/compiler-explorer/clang-format\nformatter.rustfmt.exe=/opt/compiler-explorer/rustfmt`,
            );
            expect(result.orphanedFormatterExe).toContainEqual(expect.objectContaining({id: 'rustfmt'}));
        });
    });

    describe('orphaned tool detection', () => {
        it('should report tools listed but not defined', () => {
            const result = validate(`tools=readelf:nm\ntools.readelf.exe=/usr/bin/readelf`);
            expect(result.orphanedToolExe).toContainEqual(expect.objectContaining({id: 'nm'}));
        });

        it('should report tools defined but not listed', () => {
            const result = validate(`tools=readelf\ntools.readelf.exe=/usr/bin/readelf\ntools.nm.exe=/usr/bin/nm`);
            expect(result.orphanedToolExe).toContainEqual(expect.objectContaining({id: 'nm'}));
        });
    });

    describe('orphaned library detection', () => {
        it('should report libs listed but versions not defined', () => {
            const result = validate(
                `libs=boost:fmt\nlibs.boost.versions=1.80\nlibs.boost.versions.1.80.version=1.80.0`,
            );
            expect(result.orphanedLibIds).toContainEqual(expect.objectContaining({id: 'fmt'}));
        });

        it('should report lib versions listed but not defined', () => {
            const result = validate(
                `libs=boost\nlibs.boost.versions=1.80:1.81\nlibs.boost.versions.1.80.version=1.80.0`,
            );
            expect(result.orphanedLibVersions).toContainEqual(expect.objectContaining({id: 'boost 1.81'}));
        });
    });

    describe('invalid default compiler detection', () => {
        it('should report default compiler not in list', () => {
            const result = validate(
                `compilers=gcc:clang\ndefaultCompiler=msvc\ncompiler.gcc.exe=/opt/compiler-explorer/gcc\ncompiler.clang.exe=/opt/compiler-explorer/clang`,
            );
            expect(result.invalidDefaultCompiler).toContainEqual(expect.objectContaining({id: 'msvc'}));
        });

        it('should accept valid default compiler', () => {
            const result = validate(
                `compilers=gcc:clang\ndefaultCompiler=gcc\ncompiler.gcc.exe=/opt/compiler-explorer/gcc\ncompiler.clang.exe=/opt/compiler-explorer/clang`,
            );
            expect(result.invalidDefaultCompiler).toHaveLength(0);
        });

        it('should report default compiler when there is no compilers list', () => {
            const result = validate(`defaultCompiler=gcc\ncompiler.gcc.exe=/opt/compiler-explorer/gcc`);
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
            expect(
                validate(
                    `compiler.gcc.exe=/opt/compiler-explorer/gcc/bin/gcc\ncompiler.gcc.name=GCC`,
                    'c++.amazon.properties',
                ).noCompilersList,
            ).toBe(true);
        });

        it('should not flag files with compilers=', () => {
            expect(
                validate(`compilers=gcc\ncompiler.gcc.exe=/opt/compiler-explorer/gcc/bin/gcc`, 'c++.amazon.properties')
                    .noCompilersList,
            ).toBe(false);
        });

        it('should not flag files with group definitions', () => {
            expect(
                validate(
                    `group.mygroup.compilers=gcc\ncompiler.gcc.exe=/opt/compiler-explorer/gcc/bin/gcc`,
                    'c++.amazon.properties',
                ).noCompilersList,
            ).toBe(false);
        });

        it.each([
            ['defaults files', 'compiler.gcc.exe=/opt/compiler-explorer/gcc/bin/gcc', 'c++.defaults.properties'],
            ['compiler-explorer config files', 'someOtherProperty=value', 'compiler-explorer.amazon.properties'],
            ['execution. files', 'compiler.gcc.exe=/opt/compiler-explorer/gcc/bin/gcc', 'execution.amazon.properties'],
            ['aws. files', 'someProperty=value', 'aws.properties'],
            ['asm-docs. files', 'someProperty=value', 'asm-docs.properties'],
            ['builtin. files', 'someProperty=value', 'builtin.amazon.properties'],
            ['files with no compiler definitions', 'someProperty=value', 'c++.amazon.properties'],
        ])('should not flag %s', (_, content, filename) => {
            expect(validate(content, filename).noCompilersList).toBe(false);
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

    type ArrayFields = {
        [K in keyof RawFileValidationResult]: RawFileValidationResult[K] extends Array<unknown> ? K : never;
    }[keyof RawFileValidationResult];

    function collectIssues(
        files: typeof propertyFiles,
        field: ArrayFields,
        useFiltered = false,
        options?: RawValidatorOptions,
    ) {
        return files.flatMap(({filename, parsed}) => {
            const result = validateRawFile(parsed, options);
            const source = useFiltered ? filterDisabled(result, parsed.disabledIds) : result;
            const issues = source[field];
            if (issues.length > 0) {
                return [{file: filename, issues: issues.map(i => i.id ?? i.text)}];
            }
            return [];
        });
    }

    beforeAll(() => {
        const files = fs.readdirSync(configDir);
        propertyFiles = files
            .filter(f => {
                if (!f.endsWith('.properties')) return false;
                if (f.endsWith('.local.properties')) return checkLocal;
                return true;
            })
            .map(filename => {
                const content = fs.readFileSync(path.join(configDir, filename), 'utf8');
                return {filename, parsed: parsePropertiesFileRaw(content, filename)};
            });
    });

    afterAll(() => {
        propertyFiles = [];
    });

    it('should have property files to validate', () => {
        expect(propertyFiles.length).toBeGreaterThan(0);
    });

    const validationChecks: Array<{name: string; field: ArrayFields; useFiltered?: boolean}> = [
        {name: 'duplicate keys', field: 'duplicateKeys', useFiltered: true},
        {name: 'empty list elements', field: 'emptyListElements'},
        {name: 'typo compilers', field: 'typoCompilers', useFiltered: true},
        {name: 'invalid property format', field: 'invalidPropertyFormat'},
        {name: 'orphaned compilers (exe)', field: 'orphanedCompilerExe', useFiltered: true},
        {name: 'orphaned compilers (ID)', field: 'orphanedCompilerId', useFiltered: true},
        {name: 'orphaned groups', field: 'orphanedGroups', useFiltered: true},
        {name: 'orphaned formatters (exe)', field: 'orphanedFormatterExe', useFiltered: true},
        {name: 'orphaned formatters (ID)', field: 'orphanedFormatterId', useFiltered: true},
        {name: 'orphaned tools (exe)', field: 'orphanedToolExe', useFiltered: true},
        {name: 'orphaned tools (ID)', field: 'orphanedToolId', useFiltered: true},
        {name: 'orphaned lib IDs', field: 'orphanedLibIds', useFiltered: true},
        {name: 'orphaned lib versions', field: 'orphanedLibVersions', useFiltered: true},
        {name: 'duplicated compiler references', field: 'duplicatedCompilerRefs', useFiltered: true},
        {name: 'duplicated group references', field: 'duplicatedGroupRefs', useFiltered: true},
        {name: 'invalid default compilers', field: 'invalidDefaultCompiler', useFiltered: true},
    ];

    it.each(validationChecks)('should have no $name', ({field, useFiltered}) => {
        const filesWithIssues = collectIssues(propertyFiles, field, useFiltered);
        expect(filesWithIssues, `Files with ${field}`).toEqual([]);
    });

    it('should have no duplicate compiler IDs across amazon property files', () => {
        const amazonOnly = propertyFiles.filter(f => f.filename.includes('amazon'));
        const result = validateCrossFileCompilerIds(amazonOnly);
        if (result.duplicateCompilerIds.size > 0) {
            const duplicates = Object.fromEntries(result.duplicateCompilerIds);
            expect.fail(`Duplicate compiler IDs found: ${JSON.stringify(duplicates, null, 2)}`);
        }
    });

    it('should have no language files missing compilers list', () => {
        const filesWithMissingList = propertyFiles
            .filter(({parsed}) => validateRawFile(parsed).noCompilersList)
            .map(({filename}) => filename);
        expect(filesWithMissingList, `Files missing compilers list`).toEqual([]);
    });

    it('should have no suspicious paths in amazon properties', () => {
        const amazonFiles = propertyFiles.filter(f => f.filename.includes('amazon'));
        const filesWithIssues = collectIssues(amazonFiles, 'suspiciousPaths', true, {checkSuspiciousPaths: true});
        expect(filesWithIssues, 'Files with suspicious paths').toEqual([]);
    });
});
