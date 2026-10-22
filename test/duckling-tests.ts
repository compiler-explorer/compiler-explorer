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

import {BaseCompiler} from '../lib/base-compiler.js';
import {DucklingCompiler, DucklingDbcCompiler, DucklingScriptCompiler} from '../lib/compilers/duckling.js';
import {getCompilerTypeByKey} from '../lib/compilers/index.js';
import {languages as languageDefinitions} from '../lib/languages.js';
import type {ClientOptionsType} from '../lib/options-handler.js';
import type {CacheKey, ExecutionOptionsWithEnv} from '../types/compilation/compilation.interfaces.js';
import type {
    BasicExecutionResult,
    ExecutableExecutionOptions,
    UnprocessedExecResult,
} from '../types/execution/execution.interfaces.js';
import type {LanguageKey} from '../types/languages.interfaces.js';
import {makeCompilationEnvironment, makeFakeCompilerInfo, newTempDir} from './utils.js';

const languages = {
    duckling: {id: 'duckling' as LanguageKey},
    'duckling-script': {id: 'duckling-script' as LanguageKey},
};
const environment = makeCompilationEnvironment({languages});

function makeCompiler() {
    return new DucklingCompiler(
        makeFakeCompilerInfo({lang: 'duckling' as LanguageKey, exe: '/duckling/bin/duckc'}),
        environment,
    );
}

function makeScriptCompiler() {
    return new DucklingScriptCompiler(
        makeFakeCompilerInfo({
            id: 'duckcscript',
            lang: 'duckling-script' as LanguageKey,
            exe: 'duckc',
            interpreted: true,
            supportsExecute: true,
            supportsBinary: true,
        }),
        environment,
    );
}

function makeDbcCompiler() {
    const dbcEnvironment = makeCompilationEnvironment({
        languages,
        props: {'compiler.duckcdvm.dvmExe': 'test-VM'},
    });
    return new DucklingDbcCompiler(
        makeFakeCompilerInfo({
            id: 'duckcdvm',
            lang: 'duckling' as LanguageKey,
            exe: 'duckc',
        }),
        dbcEnvironment,
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

    it('registers Duckling Script as a separate language and compiler type', () => {
        const compiler = makeScriptCompiler();

        expect(languageDefinitions['duckling-script'].extensions).toEqual(['.dks']);
        expect(languageDefinitions['duckling-script'].monaco).toBe('duckling');
        expect(getCompilerTypeByKey('duckling-script')).toBe(DucklingScriptCompiler);
        expect(compiler.compiler.supportsBinary).toBe(true);
        expect((compiler as any).optionsForFilter({}, '/tmp/example.s')).toEqual([
            'compile_script',
            '--dump-ir',
            'asm',
        ]);
    });

    it('discovers compiler options from module and script help', async () => {
        const compiler = makeCompiler();
        const sharedOption = '  -O, --llvm-opt <level>   Set optimization level.\n';
        const execCompilerCached = vi
            .spyOn(compiler, 'execCompilerCached')
            .mockImplementation(async (_exe, args) =>
                makeExecResult(
                    0,
                    args[0] === 'compile_script' ? `${sharedOption}  --linker <path>   Linker path.\n` : sharedOption,
                ),
            );

        await (compiler as any).argParser.parse();

        expect(execCompilerCached).toHaveBeenNthCalledWith(1, '/duckling/bin/duckc', ['compile_module', '--help']);
        expect(execCompilerCached).toHaveBeenNthCalledWith(2, '/duckling/bin/duckc', ['compile_script', '--help']);
        expect(compiler.possibleArguments.possibleArguments['--llvm-opt <level>']?.description).toBe(
            'Set optimization level.',
        );
        expect(compiler.possibleArguments.possibleArguments['--linker <path>']?.description).toBe('Linker path.');
        expect(compiler.compiler.cachedPossibleArguments['--linker <path>']?.description).toBe('Linker path.');
    });

    it('restores discovered options after compiler initialization clears parser options', async () => {
        const compiler = makeCompiler();
        const cachedOptions = {
            '--llvm-opt <level>': {description: 'Set optimization level.', timesused: 0},
        };
        vi.spyOn(BaseCompiler.prototype, 'initialise').mockImplementation(async function (this: BaseCompiler) {
            this.compiler.cachedPossibleArguments = cachedOptions;
            this.possibleArguments.possibleArguments = {};
            return this;
        });

        await compiler.initialise(new Date(), {} as ClientOptionsType);

        expect(compiler.possibleArguments.possibleArguments).toEqual(cachedOptions);
        expect(compiler.compiler.cachedPossibleArguments).toBeUndefined();
    });

    it('compiles a Duckling Script source and selects its script assembly artifact', async () => {
        const compiler = makeScriptCompiler();
        const workingDirectory = newTempDir();
        const inputFilename = path.join(workingDirectory, 'hello.dks');
        const assembly = '\t.text\nscript_main:\n\tretq\n';
        await fs.writeFile(inputFilename, 'builtin_output_i64(42);');
        vi.spyOn(compiler, 'exec').mockImplementation(async (_exe, args, execOptions) => {
            expect(args).toEqual(['compile_script', inputFilename, '--dump-ir', 'asm']);
            expect(execOptions.customCwd).toBe(workingDirectory);
            const artifactsDirectory = path.join(workingDirectory, 'duck_build', 'duck_debug_artifacts');
            await fs.mkdir(artifactsDirectory, {recursive: true});
            await fs.writeFile(path.join(artifactsDirectory, 'random-id.hello.s'), 'module assembly');
            await fs.writeFile(path.join(artifactsDirectory, 'script_module_abc123.s'), assembly);
            return makeExecResult();
        });

        const result = await compiler.runCompiler(
            'duckc',
            ['compile_script', inputFilename, '--dump-ir', 'asm'],
            inputFilename,
            makeExecOptions(),
        );

        expect(result.code).toBe(0);
        await expect(fs.readFile(path.join(workingDirectory, 'example.s'), 'utf8')).resolves.toBe(assembly);
    });

    it('runs scripts through Duckling with the run subcommand and returns execution results', async () => {
        const compiler = makeScriptCompiler();
        const source = 'builtin_output_i64(42);';
        const runExecutable = vi.spyOn(compiler, 'runExecutable').mockResolvedValue({
            code: 0,
            okToCache: true,
            filenameTransform: filename => filename,
            stdout: [{text: '42'}],
            stderr: [],
            execTime: 1,
            timedOut: false,
        } as BasicExecutionResult);
        const executeParameters: ExecutableExecutionOptions = {
            args: ['first', 'second'],
            stdin: '',
            ldPath: [],
            env: {},
        };

        const result = await compiler.handleInterpreting({source, files: []} as unknown as CacheKey, executeParameters);
        const [executable, parameters, workingDirectory] = runExecutable.mock.calls[0];

        expect(executable).toBe('duckc');
        expect(parameters.args).toEqual(['run', path.join(workingDirectory, 'example.dks'), 'first', 'second']);
        expect(await fs.readFile(path.join(workingDirectory, 'example.dks'), 'utf8')).toBe(source);
        expect(result.didExecute).toBe(true);
        expect(result.code).toBe(0);
        expect(result.stdout).toEqual([{text: '42'}]);
        expect(result.buildResult?.executableFilename).toBe(path.join(workingDirectory, 'example.dks'));
    });

    it('uses the Duckling native executable for module execution', () => {
        const compiler = makeCompiler();

        expect((compiler as any).optionsForFilter({execute: true}, '/tmp/example.s')).toEqual([
            'compile_modules',
            '--dump-ir',
            'asm',
        ]);
        expect(compiler.getExecutableFilename('/tmp', 'example')).toBe('/tmp/duck_build/package_llvm.exe');
    });

    it('does not add generic library linker arguments when no libraries are selected', () => {
        const compiler = makeCompiler();

        const args = compiler.prepareArguments(
            [],
            {binary: true, execute: true},
            {},
            '/tmp/example.dk',
            '/tmp/duck_build/package_llvm.exe',
            [],
            [],
        );

        expect(args).toEqual(['compile_modules', '/tmp/example.dk', '--dump-ir', 'asm']);
    });

    it('selects native execution for Duckling Script Binary output on the same compiler profile', () => {
        const compiler = makeScriptCompiler();
        const interpretedKey = {filters: {binary: false}} as unknown as CacheKey;
        const binaryKey = {filters: {binary: true}} as unknown as CacheKey;

        expect((compiler as any).shouldInterpret(interpretedKey)).toBe(true);
        expect((compiler as any).shouldInterpret(binaryKey)).toBe(false);
        expect(compiler.getExecutableFilename('/tmp', 'example')).toBe('/tmp/example.dks');
        expect(compiler.getExecutableFilename('/tmp', 'example', binaryKey)).toBe('/tmp/duck_build/example.exe');
        expect((compiler as any).optionsForFilter(binaryKey.filters, '/tmp/example.s')).toEqual([
            'compile_script',
            '--dump-ir',
            'asm',
        ]);
    });

    it('registers a DVM-backed module compiler profile', () => {
        const compiler = makeDbcCompiler();

        expect(getCompilerTypeByKey('duckling-dbc')).toBe(DucklingDbcCompiler);
        expect((compiler as any).optionsForFilter({}, '/tmp/example.s')).toEqual([
            'compile_module',
            '--dvm-backend',
            '--dump-ir',
            'dbc',
        ]);
        expect((compiler as any).optionsForFilter({execute: true}, '/tmp/example.s')).toEqual([
            'compile_modules',
            '--dvm-backend',
            '--dump-ir',
            'dbc',
        ]);
        expect(compiler.getExecutableFilename('/tmp', 'example')).toBe('/tmp/duck_build/package_dvm.dbc');
        expect(compiler.compiler.supportsIrView).toBe(false);
        expect(compiler.compiler.supportsCfg).toBe(false);
    });

    it('copies the module DBC artifact and identifies its output as plaintext', async () => {
        const compiler = makeDbcCompiler();
        const workingDirectory = newTempDir();
        const inputFilename = path.join(workingDirectory, 'hello.dk');
        await fs.writeFile(inputFilename, 'fun main() -> i64 = { return 7; }');
        const dbc = 'module hello { function main() }';
        vi.spyOn(compiler, 'exec').mockImplementation(async () => {
            const artifactsDirectory = path.join(workingDirectory, 'duck_build', 'duck_debug_artifacts');
            await fs.mkdir(artifactsDirectory, {recursive: true});
            await fs.writeFile(path.join(artifactsDirectory, 'random-id.hello.dbc'), dbc);
            return makeExecResult();
        });

        const result = await compiler.runCompiler(
            'duckc',
            ['compile_module', inputFilename, '--dvm-backend', '--dump-ir', 'dbc'],
            inputFilename,
            makeExecOptions(),
        );

        expect(result.code).toBe(0);
        expect(result.languageId).toBe('plaintext');
        await expect(fs.readFile(path.join(workingDirectory, 'example.s'), 'utf8')).resolves.toBe(dbc);
    });

    it('invokes the configured DVM and passes program arguments as a comma-separated DVM option', async () => {
        const compiler = makeDbcCompiler();
        const runExecutable = vi
            .spyOn(BaseCompiler.prototype, 'runExecutable')
            .mockResolvedValue({} as BasicExecutionResult);
        const executeParameters: ExecutableExecutionOptions = {
            args: ['hello', 'world'],
            stdin: '',
            ldPath: [],
            env: {},
        };

        await compiler.runExecutable('/tmp/duck_build/package_dvm.dbc', executeParameters, '/tmp');

        expect(runExecutable).toHaveBeenCalledWith(
            'test-VM',
            {
                ...executeParameters,
                args: ['run', '/tmp/duck_build/package_dvm.dbc', '--args', 'hello,world'],
            },
            '/tmp',
        );
    });

    it('puts the module path after the subcommand and preserves user options', () => {
        const compiler = makeCompiler();
        const inputFilename = '/tmp/example.dk';
        const options = (compiler as any).optionsForFilter({}, '/tmp/example.s');

        expect(options).toEqual(['compile_module', '--dump-ir', 'asm']);
        expect(
            (compiler as any).orderArguments(options, inputFilename, [], [], [], [], ['--llvm-opt', '2'], []),
        ).toEqual(['compile_module', inputFilename, '--dump-ir', 'asm', '--llvm-opt', '2']);
    });

    it('prepares Duckling artifacts and copies this module assembly to the CE output path', async () => {
        const compiler = makeCompiler();
        const workingDirectory = newTempDir();
        const inputFilename = path.join(workingDirectory, 'hello.dk');
        const assembly = '\t.text\nmain:\n\tretq\n';
        await fs.writeFile(inputFilename, 'fun main() -> i64 = { return 7; }');
        const exec = vi.spyOn(compiler, 'exec').mockImplementation(async (_exe, args, execOptions) => {
            expect(args).toEqual(['compile_module', inputFilename, '--dump-ir', 'asm']);
            expect(execOptions.customCwd).toBe(workingDirectory);
            await expect(fs.stat(path.join(workingDirectory, 'duck_build'))).resolves.toBeDefined();
            const artifactsDirectory = path.join(workingDirectory, 'duck_build', 'duck_debug_artifacts');
            await fs.mkdir(artifactsDirectory, {recursive: true});
            await fs.writeFile(path.join(artifactsDirectory, 'random-id.hello.s'), assembly);
            return makeExecResult();
        });

        const result = await compiler.runCompiler(
            '/duckling/bin/duckc',
            ['compile_module', inputFilename, '--dump-ir', 'asm'],
            inputFilename,
            makeExecOptions(),
        );

        expect(result.code).toBe(0);
        expect(exec).toHaveBeenCalledOnce();
        await expect(fs.readFile(path.join(workingDirectory, 'example.s'), 'utf8')).resolves.toBe(assembly);
    });

    it('compiles sibling modules in deterministic order and excludes scripts and build files', async () => {
        const compiler = makeCompiler();
        const workingDirectory = newTempDir();
        const inputFilename = path.join(workingDirectory, 'main.dk');
        const modulesDirectory = path.join(workingDirectory, 'modules');
        const hiddenDirectory = path.join(workingDirectory, '.hidden');
        const buildDirectory = path.join(workingDirectory, 'duck_build');
        await fs.mkdir(modulesDirectory, {recursive: true});
        await fs.mkdir(hiddenDirectory, {recursive: true});
        await fs.mkdir(buildDirectory, {recursive: true});
        await fs.writeFile(inputFilename, 'import alpha;');
        await fs.writeFile(path.join(workingDirectory, 'zeta.dk'), 'fun zeta() = {}');
        await fs.writeFile(path.join(modulesDirectory, 'beta.dk'), 'fun beta() = {}');
        await fs.writeFile(path.join(modulesDirectory, 'alpha.dk'), 'fun alpha() = {}');
        await fs.writeFile(path.join(workingDirectory, 'script.dks'), 'fun script() = {}');
        await fs.writeFile(path.join(hiddenDirectory, 'hidden.dk'), 'fun hidden() = {}');
        await fs.writeFile(path.join(buildDirectory, 'stale.dk'), 'fun stale() = {}');

        const exec = vi.spyOn(compiler, 'exec').mockImplementation(async (_exe, args, execOptions) => {
            expect(args).toEqual([
                'compile_modules',
                inputFilename,
                path.join(modulesDirectory, 'alpha.dk'),
                path.join(modulesDirectory, 'beta.dk'),
                path.join(workingDirectory, 'zeta.dk'),
                '--dump-ir',
                'asm',
                '--no-executable',
                '--no-incremental',
            ]);
            expect(execOptions.customCwd).toBe(workingDirectory);
            const artifactsDirectory = path.join(workingDirectory, 'duck_build', 'duck_debug_artifacts');
            await fs.mkdir(artifactsDirectory, {recursive: true});
            await fs.writeFile(path.join(artifactsDirectory, 'random-id.main.s'), '\t.text\nmain:\n\tretq\n');
            return makeExecResult();
        });

        const result = await compiler.runCompiler(
            '/duckling/bin/duckc',
            ['compile_module', inputFilename, '--dump-ir', 'asm'],
            inputFilename,
            makeExecOptions(),
        );

        expect(result.code).toBe(0);
        expect(exec).toHaveBeenCalledOnce();
    });

    it('keeps executable linking enabled for multi-module execution', async () => {
        const compiler = makeCompiler();
        const workingDirectory = newTempDir();
        const inputFilename = path.join(workingDirectory, 'hello.dk');
        const helperFilename = path.join(workingDirectory, 'helper.dk');
        await fs.writeFile(inputFilename, 'import helper;');
        await fs.writeFile(helperFilename, 'fun value() -> i64 = { return 42; }');

        const exec = vi.spyOn(compiler, 'exec').mockImplementation(async (_exe, args) => {
            expect(args).toEqual([
                'compile_modules',
                inputFilename,
                helperFilename,
                '--dump-ir',
                'asm',
                '--no-incremental',
            ]);
            const artifactsDirectory = path.join(workingDirectory, 'duck_build', 'duck_debug_artifacts');
            await fs.mkdir(artifactsDirectory, {recursive: true});
            await fs.writeFile(path.join(artifactsDirectory, 'random-id.hello.s'), '\t.text\nmain:\n\tretq\n');
            return makeExecResult();
        });

        const result = await compiler.runCompiler(
            '/duckling/bin/duckc',
            ['compile_modules', inputFilename, '--dump-ir', 'asm'],
            inputFilename,
            makeExecOptions(),
            {binary: true, execute: true},
        );

        expect(result.code).toBe(0);
        expect(exec).toHaveBeenCalledOnce();
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
            ['compile_module', inputFilename, '--dump-ir', 'asm'],
            inputFilename,
            makeExecOptions(),
        );

        expect(result.code).toBe(1);
        expect(result.stderr.map(line => line.text).join('\n')).toContain('Duckling diagnostic');
        expect(result.stderr.map(line => line.text).join('\n')).toContain('Duckling produced no assembly for hello.dk');
        await expect(fs.stat(path.join(workingDirectory, 'example.s'))).rejects.toThrow();
    });

    it('maps Duckling diagnostic locations to the source editor', async () => {
        const compiler = makeCompiler();
        const workingDirectory = newTempDir();
        const inputFilename = path.join(workingDirectory, 'hello.dk');
        const diagnostic = [
            '\u001b[31merror[E0]: Call failed because no matching functions were found.\u001b[0m',
            '  > <source>:2:5',
            '  |',
        ].join('\n');
        vi.spyOn(compiler, 'exec').mockResolvedValue(makeExecResult(1, '', diagnostic));

        const result = await compiler.runCompiler(
            '/duckling/bin/duckc',
            ['compile_module', inputFilename, '--dump-ir', 'asm'],
            inputFilename,
            makeExecOptions(),
        );

        expect(result.stderr[0].tag).toEqual({
            file: 'hello.dk',
            line: 2,
            column: 5,
            text: 'error[E0]: Call failed because no matching functions were found.',
            severity: 3,
        });
        expect(result.stderr[1].tag).toEqual({file: 'hello.dk', line: 2, column: 5, text: '', severity: 3});
    });

    it('maps Duckling warning and informational severities', async () => {
        const compiler = makeCompiler();
        const workingDirectory = newTempDir();
        const inputFilename = path.join(workingDirectory, 'hello.dk');
        const diagnostic = [
            'warning[W1]: value is unused',
            '  > <source>:3:2',
            '  |',
            'note: value declared here',
            '  > <source>:1:1',
            '  |',
            'help: consider removing it',
            '  > <source>:4:5',
        ].join('\n');
        vi.spyOn(compiler, 'exec').mockResolvedValue(makeExecResult(1, '', diagnostic));

        const result = await compiler.runCompiler(
            '/duckling/bin/duckc',
            ['compile_module', inputFilename, '--dump-ir', 'asm'],
            inputFilename,
            makeExecOptions(),
        );

        expect(result.stderr.filter(line => line.tag).map(line => line.tag?.severity)).toEqual([2, 3, 1, 3, 1, 3]);
    });

    it('preserves Duckling failure diagnostics', async () => {
        const compiler = makeCompiler();
        const workingDirectory = newTempDir();
        const inputFilename = path.join(workingDirectory, 'hello.dk');
        vi.spyOn(compiler, 'exec').mockResolvedValue(makeExecResult(1, '', 'Duckling compilation failed'));

        const result = await compiler.runCompiler(
            '/duckling/bin/duckc',
            ['compile_module', inputFilename, '--dump-ir', 'asm'],
            inputFilename,
            makeExecOptions(),
        );

        expect(result.code).toBe(1);
        expect(result.stderr.map(line => line.text).join('\n')).toContain('Duckling compilation failed');
        expect(result.stderr.map(line => line.text).join('\n')).not.toContain('produced no assembly');
    });
});
