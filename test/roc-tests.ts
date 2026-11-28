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

import path from 'node:path';

import {afterEach, describe, expect, it, vi} from 'vitest';

import {getCompilerTypeByKey} from '../lib/compilers/index.js';
import {RocCompiler} from '../lib/compilers/roc.js';
import type {ExecutionOptionsWithEnv} from '../types/compilation/compilation.interfaces.js';
import type {UnprocessedExecResult} from '../types/execution/execution.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../types/features/filters.interfaces.js';
import {
    makeCompilationEnvironment,
    makeFakeCompilerInfo,
    makeFakeParseFiltersAndOutputOptions,
    newTempDir,
} from './utils.js';

function makeCompiler() {
    const environment = makeCompilationEnvironment({languages: {roc: {id: 'roc'}}});
    return new RocCompiler(
        makeFakeCompilerInfo({
            id: 'roc',
            lang: 'roc',
            compilerType: 'roc',
            exe: '/opt/compiler-explorer/roc/roc',
            supportsBinary: true,
        }),
        environment,
    );
}

function makeExecResult(code = 0, stdout = '', stderr = ''): UnprocessedExecResult {
    return {
        code,
        okToCache: true,
        filenameTransform: filename => filename,
        stdout,
        stderr,
        execTime: 0,
        timedOut: false,
        truncated: false,
    };
}

afterEach(() => vi.restoreAllMocks());

describe('Roc compiler', () => {
    it('is registered as the roc compiler type', () => {
        expect(getCompilerTypeByKey('roc')).toBe(RocCompiler);
    });

    it('forces binary mode and builds with --no-cache and --output', () => {
        const compiler = makeCompiler();
        const filters: ParseFiltersAndOutputOptions = {};
        compiler.fixFiltersBeforeCacheKey(filters, [], []);
        expect(filters.binary).toBe(true);

        const outputPath = compiler.getOutputFilename('/tmp/ce-roc', 'output');
        expect(outputPath).toBe(path.join('/tmp/ce-roc', 'output'));

        const filterOpts: ParseFiltersAndOutputOptions = {};
        expect(compiler.optionsForFilter(filterOpts, outputPath)).toEqual([
            'build',
            '--no-cache',
            `--output=${outputPath}`,
        ]);
        expect(filterOpts.binary).toBe(true);
    });

    it('filters unsafe or conflicting CLI arguments while preserving build flags', () => {
        const compiler = makeCompiler();
        expect(
            compiler.filterUserOptions([
                'build',
                '--opt=dev',
                '--watch',
                '--output',
                '/tmp/evil',
                '--output=/tmp/evil2',
                '--replace-dep',
                'old',
                '/etc/passwd',
                '--specialize=no',
                '--',
                'extra.roc',
            ]),
        ).toEqual(['--opt=dev', '--specialize=no']);
    });

    it('sets writable HOME and XDG_CACHE_HOME inside the build directory', async () => {
        const compiler = makeCompiler();
        const workDir = newTempDir();
        const inputFilename = path.join(workDir, 'example.roc');
        const callerEnv = {PATH: '/usr/bin'};
        const execSpy = vi.spyOn(compiler, 'exec').mockImplementation(async (_exe, _args, execOptions) => {
            expect(execOptions.customCwd).toBe(workDir);
            expect(execOptions.env?.HOME).toBe(workDir);
            expect(execOptions.env?.XDG_CACHE_HOME).toBe(path.join(workDir, '.cache'));
            return makeExecResult(0, '0 errors and 0 warnings found');
        });

        const result = await compiler.runCompiler(
            '/opt/compiler-explorer/roc/roc',
            ['build', '--no-cache', `--output=${path.join(workDir, 'output')}`, inputFilename],
            inputFilename,
            {env: callerEnv} as ExecutionOptionsWithEnv,
        );

        expect(result.code).toBe(0);
        expect(execSpy).toHaveBeenCalledOnce();
        expect(callerEnv).toEqual({PATH: '/usr/bin'});
    });

    it('parses Roc error and warning headers into editor diagnostic tags', async () => {
        const compiler = makeCompiler();
        const workDir = newTempDir();
        const inputFilename = path.join(workDir, 'example.roc');
        const diagnostics = [
            `── ✗ does not exist ───────────────────────────────────── ${inputFilename}:4:14`,
            '',
            'Num.to_str does not exist.',
            '',
            '── ⚠ unused value ─────────────────────────────────────── ./example.roc:2:5',
        ].join('\n');
        vi.spyOn(compiler, 'exec').mockResolvedValue(makeExecResult(1, '', diagnostics));

        const result = await compiler.runCompiler(
            '/opt/compiler-explorer/roc/roc',
            ['build', inputFilename],
            inputFilename,
            {env: {PATH: '/usr/bin'}} as ExecutionOptionsWithEnv,
        );

        expect(result.code).toBe(1);
        expect(result.stderr[0].tag).toEqual({
            line: 4,
            column: 14,
            text: 'does not exist',
            severity: 3,
            file: 'example.roc',
        });
        expect(result.stderr[4].tag).toEqual({
            line: 2,
            column: 5,
            text: 'unused value',
            severity: 2,
            file: 'example.roc',
        });
    });

    it('filters host runtime symbols when libraryCode filter is enabled', async () => {
        const compiler = makeCompiler();
        const sampleObjdump = [
            'output:     file format elf64-x86-64',
            '',
            'Disassembly of section .text:',
            '',
            '0000000000201000 <roc__p1234567890abcdef>:',
            '  201000:\t0f af ff             \timul   %edi,%edi',
            '  201003:\t89 f8                \tmov    %edi,%eax',
            '  201005:\tc3                   \tret',
            '',
            '0000000000201010 <roc_llvm_rc_decref_1>:',
            '  201010:\tc3                   \tret',
            '',
            '0000000000201020 <roc_alloc>:',
            '  201020:\tc3                   \tret',
            '',
            '0000000000201030 <_start>:',
            '  201030:\tc3                   \tret',
        ].join('\n');

        const filtered = await compiler.processAsm(
            {asm: sampleObjdump},
            makeFakeParseFiltersAndOutputOptions({binary: true, libraryCode: true}),
            [],
        );
        const filteredText = filtered.asm.map((line: {text: string}) => line.text).join('\n');
        expect(filteredText).toContain('roc__p1234567890abcdef:');
        expect(filteredText).toContain('imul');
        expect(filteredText).not.toContain('roc_llvm_rc_decref_1');
        expect(filteredText).not.toContain('roc_alloc');
        expect(filteredText).not.toContain('_start');

        const unfiltered = await compiler.processAsm(
            {asm: sampleObjdump},
            makeFakeParseFiltersAndOutputOptions({binary: true, libraryCode: false}),
            [],
        );
        const unfilteredText = unfiltered.asm.map((line: {text: string}) => line.text).join('\n');
        expect(unfilteredText).toContain('roc__p1234567890abcdef:');
        expect(unfilteredText).toContain('roc_alloc:');
        expect(unfilteredText).toContain('_start:');
    });
});
