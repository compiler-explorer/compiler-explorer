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

import fs from 'node:fs/promises';
import path from 'node:path';

import {afterEach, describe, expect, it, vi} from 'vitest';

import {DucklingCompiler} from '../lib/compilers/duckling.js';
import {getCompilerTypeByKey} from '../lib/compilers/index.js';
import type {ClientOptionsType} from '../lib/options-handler.js';
import type {ExecutionOptionsWithEnv} from '../types/compilation/compilation.interfaces.js';
import type {UnprocessedExecResult} from '../types/execution/execution.interfaces.js';
import type {LanguageKey} from '../types/languages.interfaces.js';
import {makeCompilationEnvironment, makeFakeCompilerInfo, newTempDir} from './utils.js';

const languages = {duckling: {id: 'duckling' as LanguageKey}};
const environment = makeCompilationEnvironment({languages});

function makeCompiler() {
    return new DucklingCompiler(
        makeFakeCompilerInfo({
            id: 'duckling',
            lang: 'duckling' as LanguageKey,
            exe: '/duckling/bin/duckc',
            libsArr: [],
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

function makeExecOptions(): ExecutionOptionsWithEnv {
    return {env: {PATH: '/usr/bin'}} as ExecutionOptionsWithEnv;
}

afterEach(() => vi.restoreAllMocks());

describe('Duckling compiler', () => {
    it('is registered as the duckling compiler type', () => {
        expect(getCompilerTypeByKey('duckling')).toBe(DucklingCompiler);
    });

    it('keeps help-derived suggestions through initialisation and discovery-cache restoration', async () => {
        const compiler = makeCompiler();
        const clientOptions = {libs: {}} as ClientOptionsType;
        vi.spyOn(compiler, 'getVersion').mockResolvedValue({code: 0, stdout: 'Duckling compiler', stderr: ''});
        const help = [
            'Usage: duckc [GLOBAL OPTIONS] compile_module <module> [OPTIONS]',
            '',
            'Global options:',
            '  -h, --help               Print help and exit.',
            '',
            'Command options:',
            '  -O, --llvm-opt <level>   Set optimization level.',
            '                           Possible values are: 0, 1, 2, 3, s, z.',
            '                           See https://llvm.org/doxygen/classllvm_1_1OptimizationLevel.html',
            '  --print-ir <categories>  Print selected intermediate representations to stdout.',
            '                           Possible values are: hir, mir, lir, dbc.',
            '  --artifact-location <path>',
            '                           Path to compilation artifacts.',
            '',
        ].join('\n');
        const execCompilerCached = vi.spyOn(compiler, 'execCompilerCached').mockResolvedValue(makeExecResult(0, help));

        await expect(compiler.initialise(new Date(), clientOptions)).resolves.toBe(compiler);

        const suggestions = compiler.possibleArguments.getPopularArguments();
        expect(suggestions).toEqual({
            '-O 2': {
                description:
                    'Set optimization level. Possible values are: 0, 1, 2, 3, s, z. -O is short for --llvm-opt.',
                timesused: 0,
            },
            '--print-ir hir': {
                description:
                    'Print intermediate representations to Compiler Output. Values: hir, mir, lir (comma-separated).',
                timesused: 0,
            },
        });
        expect(execCompilerCached).toHaveBeenCalledExactlyOnceWith('/duckling/bin/duckc', ['compile_module', '--help']);

        const cachedCompiler = makeCompiler();
        cachedCompiler.compiler.cachedPossibleArguments = compiler.possibleArguments.possibleArguments;
        const cachedExec = vi.spyOn(cachedCompiler, 'execCompilerCached');
        await cachedCompiler.initialise(new Date(), clientOptions, true);

        expect(cachedCompiler.possibleArguments.getPopularArguments()).toEqual(suggestions);
        expect(cachedExec).not.toHaveBeenCalled();
    });

    it('remains available without suggestions when help fails', async () => {
        const compiler = makeCompiler();
        vi.spyOn(compiler, 'getVersion').mockResolvedValue({code: 0, stdout: 'Duckling compiler', stderr: ''});
        vi.spyOn(compiler, 'execCompilerCached').mockResolvedValue(makeExecResult(1, '', 'Help is unavailable.'));

        await expect(compiler.initialise(new Date(), {libs: {}} as ClientOptionsType)).resolves.toBe(compiler);
        expect(compiler.possibleArguments.getPopularArguments()).toEqual({});
    });

    it('puts the module path after the subcommand and preserves user options', () => {
        const compiler = makeCompiler();
        const inputFilename = '/tmp/example.dk';
        const options = (compiler as any).optionsForFilter({}, '/tmp/example.s');

        expect(options).toEqual(['compile_module', '--no-incremental', '--dump-ir', 'asm']);
        expect(
            (compiler as any).orderArguments(
                options,
                inputFilename,
                [],
                [],
                [],
                [],
                ['-O', '2', '--print-ir', 'hir,mir'],
                [],
            ),
        ).toEqual([
            'compile_module',
            inputFilename,
            '--no-incremental',
            '--dump-ir',
            'asm',
            '-O',
            '2',
            '--print-ir',
            'hir,mir',
        ]);
    });

    it('prepares Duckling artifacts and copies this module assembly to the CE output path', async () => {
        const compiler = makeCompiler();
        const workingDirectory = newTempDir();
        const inputFilename = path.join(workingDirectory, 'hello.dk');
        const assembly = '\t.text\nmain:\n\tretq\n';
        const exec = vi.spyOn(compiler, 'exec').mockImplementation(async (_exe, _args, execOptions) => {
            expect(execOptions.customCwd).toBe(workingDirectory);
            await expect(fs.stat(path.join(workingDirectory, 'duck_build'))).resolves.toBeDefined();
            const artifactsDirectory = path.join(workingDirectory, 'duck_build', 'duck_debug_artifacts');
            await fs.mkdir(artifactsDirectory, {recursive: true});
            await fs.writeFile(path.join(artifactsDirectory, 'random-id.hello.s'), assembly);
            return makeExecResult();
        });

        const result = await compiler.runCompiler(
            '/duckling/bin/duckc',
            ['compile_module', inputFilename, '--no-incremental', '--dump-ir', 'asm'],
            inputFilename,
            makeExecOptions(),
        );

        expect(result.code).toBe(0);
        expect(exec).toHaveBeenCalledOnce();
        await expect(fs.readFile(path.join(workingDirectory, 'example.s'), 'utf8')).resolves.toBe(assembly);
    });

    it('fails when Duckling exits successfully but does not write assembly for this module', async () => {
        const compiler = makeCompiler();
        const workingDirectory = newTempDir();
        const inputFilename = path.join(workingDirectory, 'hello.dk');
        vi.spyOn(compiler, 'exec').mockImplementation(async () => {
            const artifactsDirectory = path.join(workingDirectory, 'duck_build', 'duck_debug_artifacts');
            await fs.mkdir(artifactsDirectory, {recursive: true});
            await fs.writeFile(path.join(artifactsDirectory, 'random-id.other-module.s'), 'old assembly');
            return makeExecResult(0, '', 'Duckling diagnostic');
        });

        const result = await compiler.runCompiler(
            '/duckling/bin/duckc',
            ['compile_module', inputFilename, '--no-incremental', '--dump-ir', 'asm'],
            inputFilename,
            makeExecOptions(),
        );

        expect(result.code).toBe(1);
        expect(result.stderr.map(line => line.text).join('\n')).toContain('Duckling diagnostic');
        expect(result.stderr.map(line => line.text).join('\n')).toContain(
            'Duckling reported success but produced no assembly for hello.dk',
        );
        await expect(fs.stat(path.join(workingDirectory, 'example.s'))).rejects.toThrow();
    });

    it('preserves Duckling failure diagnostics', async () => {
        const compiler = makeCompiler();
        const workingDirectory = newTempDir();
        const inputFilename = path.join(workingDirectory, 'hello.dk');
        vi.spyOn(compiler, 'exec').mockResolvedValue(makeExecResult(1, '', 'Duckling compilation failed'));

        const result = await compiler.runCompiler(
            '/duckling/bin/duckc',
            ['compile_module', inputFilename, '--no-incremental', '--dump-ir', 'asm'],
            inputFilename,
            makeExecOptions(),
        );

        expect(result.code).toBe(1);
        expect(result.stderr.map(line => line.text).join('\n')).toContain('Duckling compilation failed');
        expect(result.stderr.map(line => line.text).join('\n')).not.toContain('produced no assembly');
    });
});
