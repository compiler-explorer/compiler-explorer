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

import {GleamCompiler} from '../lib/compilers/gleam.js';
import type {CompilationResult, ExecutionOptionsWithEnv} from '../types/compilation/compilation.interfaces.js';
import type {CompilerInfo} from '../types/compiler.interfaces.js';
import type {UnprocessedExecResult} from '../types/execution/execution.interfaces.js';
import {makeCompilationEnvironment, makeFakeCompilerInfo, newTempDir} from './utils.js';

const languages = {gleam: {id: 'gleam'}};

const execResult = (stdout = ''): UnprocessedExecResult => ({
    code: 0,
    okToCache: true,
    filenameTransform: filename => filename,
    stdout,
    stderr: '',
    execTime: 0,
    timedOut: false,
    truncated: false,
});

function makeCompiler(target = 'erlang', runtime = 'erl') {
    const env = makeCompilationEnvironment({
        languages,
        props: {
            'compiler.gleam.stdlib': '/opt/gleam/stdlib',
            'compiler.gleam.runtime': runtime,
            'compiler.gleam.target': target,
        },
    });
    return new GleamCompiler(
        makeFakeCompilerInfo({
            id: 'gleam',
            exe: 'gleam',
            lang: 'gleam',
            ldPath: [],
            libPath: [],
        }) as CompilerInfo,
        env,
    );
}

describe('GleamCompiler', () => {
    afterEach(() => vi.restoreAllMocks());

    it('formats target-specific names using the discovered Gleam version', () => {
        expect(GleamCompiler.getDisplayName('1.18.1', 'erlang')).toBe('Gleam 1.18.1 (BEAM)');
        expect(GleamCompiler.getDisplayName('1.18.1', 'javascript')).toBe('Gleam 1.18.1 (JavaScript)');
    });

    it('rejects an unsupported compilation target', () => {
        expect(() => makeCompiler('native')).toThrow('Gleam compiler gleam has unsupported target "native"');
    });

    it('renders a failed compilation as an assembly placeholder', async () => {
        const result = await makeCompiler().processAsm({code: 1} as CompilationResult);
        expect(result.asm).toEqual([{text: '<Compilation failed>', source: null}]);
    });

    it('preserves an empty successful compilation output', async () => {
        const result = await makeCompiler().processAsm({code: 0} as CompilationResult);
        expect(result.asm).toEqual([]);
    });

    it('creates a fixed stdlib-only project and lists its generated Erlang modules', async () => {
        const compiler = makeCompiler('erlang', '/opt/otp/bin/erl');
        const projectDir = newTempDir();
        const inputFilename = path.join(projectDir, 'example.gleam');
        await fs.writeFile(inputFilename, 'pub fn main() { Nil }');
        const artefactsPath = path.join(projectDir, 'build/dev/erlang/compiler_explorer/_gleam_artefacts');
        await fs.mkdir(artefactsPath, {recursive: true});
        await fs.writeFile(path.join(artefactsPath, 'compiler_explorer.erl'), '');
        await fs.writeFile(path.join(artefactsPath, 'compiler_explorer@@main.erl'), '');

        const exec = vi.spyOn(compiler, 'exec').mockImplementation(async executable => {
            return execResult(
                executable === '/opt/otp/bin/erl'
                    ? [
                          '{module, compiler_explorer}.  %% version = 0',
                          '{call_ext,1,{extfunc,gleam_stdlib,println,1}}.',
                          "{'%',",
                          '  {var_info,',
                          '    {x,0}}}.',
                          '{move,{x,0},{x,1}}.',
                          '{return}.',
                          '{function, module_info, 0, 5}.',
                          '{label,5}.',
                          '{func_info,{atom,compiler_explorer},{atom,module_info},0}.',
                          '{function, useful_function, 0, 7}.',
                          '{label,7}.',
                      ].join('\n')
                    : '',
            );
        });

        const result = await compiler.runCompiler('gleam', [], inputFilename, {
            env: {PATH: '/compiler/bin'},
        } as ExecutionOptionsWithEnv);

        expect(await fs.readFile(path.join(projectDir, 'gleam.toml'), 'utf8')).toContain(
            'gleam_stdlib = { path = "/opt/gleam/stdlib" }',
        );
        await expect(fs.readFile(path.join(projectDir, 'src/compiler_explorer.gleam'), 'utf8')).resolves.toBe(
            'pub fn main() { Nil }',
        );
        await expect(fs.readFile(inputFilename, 'utf8')).resolves.toBe('pub fn main() { Nil }');
        expect(exec).toHaveBeenNthCalledWith(
            1,
            'gleam',
            ['build', '--no-print-progress'],
            expect.objectContaining({customCwd: projectDir, env: {PATH: '/opt/otp/bin:/compiler/bin'}}),
        );
        expect(exec).toHaveBeenNthCalledWith(
            2,
            '/opt/otp/bin/erl',
            expect.arrayContaining(['-input', path.join(artefactsPath, 'compiler_explorer.erl')]),
            expect.objectContaining({customCwd: projectDir}),
        );
        expect(exec).toHaveBeenCalledTimes(2);
        expect(result.asm).toEqual([
            {text: '{module, compiler_explorer}.  %% version = 0', source: null},
            {text: '{call_ext,1,{extfunc,gleam_stdlib,println,1}}.', source: null},
            {text: '{move,{x,0},{x,1}}.', source: null},
            {text: '{return}.', source: null},
            {text: '{function, useful_function, 0, 7}.', source: null},
            {text: '{label,7}.', source: null},
        ]);
    });

    it('exports an escript when building an executable', async () => {
        const compiler = makeCompiler('erlang', '/opt/otp/bin/erl');
        const projectDir = newTempDir();
        const inputFilename = path.join(projectDir, 'example.gleam');
        await fs.writeFile(inputFilename, 'pub fn main() { Nil }');
        const exec = vi.spyOn(compiler, 'exec').mockResolvedValue(execResult());

        await compiler.buildExecutable('gleam', [], inputFilename, {
            env: {PATH: '/compiler/bin'},
        } as ExecutionOptionsWithEnv);

        expect(exec).toHaveBeenCalledWith(
            'gleam',
            ['export', 'escript'],
            expect.objectContaining({customCwd: projectDir, env: {PATH: '/opt/otp/bin:/compiler/bin'}}),
        );
        expect(compiler.getExecutableFilename(projectDir)).toBe(path.join(projectDir, 'compiler_explorer'));
    });

    it('builds and runs JavaScript through a Node entry module', async () => {
        const compiler = makeCompiler('javascript');
        const projectDir = newTempDir();
        const inputFilename = path.join(projectDir, 'example.gleam');
        await fs.writeFile(inputFilename, 'pub fn main() { Nil }');
        const javascriptPath = path.join(projectDir, 'build/dev/javascript/compiler_explorer');
        await fs.mkdir(javascriptPath, {recursive: true});
        await fs.writeFile(path.join(javascriptPath, 'compiler_explorer.mjs'), 'export function main() {}\n');
        const executionProjectDir = newTempDir();
        const executionInputFilename = path.join(executionProjectDir, 'example.gleam');
        await fs.writeFile(executionInputFilename, 'pub fn main() { Nil }');
        const exec = vi.spyOn(compiler, 'exec').mockResolvedValue(execResult());

        const compileResult = await compiler.runCompiler('gleam', [], inputFilename, {
            env: {},
        } as ExecutionOptionsWithEnv);
        const executableResult = await compiler.buildExecutable('gleam', [], executionInputFilename, {
            env: {},
        } as ExecutionOptionsWithEnv);

        expect(exec).toHaveBeenNthCalledWith(
            1,
            'gleam',
            ['build', '--target', 'javascript', '--no-print-progress'],
            expect.objectContaining({customCwd: projectDir}),
        );
        expect(exec).toHaveBeenNthCalledWith(
            2,
            'gleam',
            ['build', '--target', 'javascript', '--no-print-progress'],
            expect.objectContaining({customCwd: executionProjectDir}),
        );
        expect(compileResult.asm).toEqual([{text: 'export function main() {}', source: null}]);
        expect(compileResult.languageId).toBe('typescript');
        expect(executableResult.code).toBe(0);
        await expect(fs.readFile(compiler.getExecutableFilename(executionProjectDir), 'utf8')).resolves.toBe(
            "import {main} from './build/dev/javascript/compiler_explorer/compiler_explorer.mjs';\n\nawait main();\n",
        );
    });

    it('does not accept user compiler options', () => {
        const compiler = makeCompiler();
        expect(compiler.filterUserOptions(['build', '--target', 'javascript'])).toEqual([]);
    });
});
