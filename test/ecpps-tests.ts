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

import {EcppsCompiler} from '../lib/compilers/ecpps.js';
import {getCompilerTypeByKey} from '../lib/compilers/index.js';
import type {ToolEnv} from '../lib/tooling/base-tool.interface.js';
import {EcppsIrTool} from '../lib/tooling/ecpps-ir.js';
import type {CompilationInfo} from '../types/compilation/compilation.interfaces.js';
import type {UnprocessedExecResult} from '../types/execution/execution.interfaces.js';
import type {LanguageKey} from '../types/languages.interfaces.js';
import type {ToolInfo} from '../types/tool.interfaces.js';
import {makeCompilationEnvironment, makeFakeCompilerInfo, newTempDir} from './utils.js';

const languages = {'c++': {id: 'c++' as LanguageKey}};
const environment = makeCompilationEnvironment({languages});

function makeCompiler() {
    return new EcppsCompiler(
        makeFakeCompilerInfo({lang: 'c++' as LanguageKey, exe: '/opt/compiler-explorer/ecpps-trunk/bin/ecpps'}),
        environment,
    );
}

describe('ECPPS compiler', () => {
    it('is registered as the ecpps compiler type', () => {
        expect(getCompilerTypeByKey('ecpps')).toBe(EcppsCompiler);
    });

    it('asks for the pseudo-instruction asm dump', () => {
        const options = (makeCompiler() as any).optionsForFilter({}, '/tmp/dir/output.s');
        expect(options).toEqual(['/verbose:pinst', '/asm', '/out:/tmp/dir/output.s']);
    });

    it('passes the source relative to its directory, since ECPPS reads a leading / as an option', () => {
        const args = (makeCompiler() as any).orderArguments(
            ['/asm'],
            '/tmp/dir/example.cpp',
            [],
            [],
            [],
            [],
            ['/WX'],
            [],
        );
        expect(args).toContain('./example.cpp');
        expect(args).not.toContain('/tmp/dir/example.cpp');
        expect(args.slice(0, 2)).toEqual(['/asm', '/WX']);
    });

    it('adds no shared library path arguments', () => {
        expect(makeCompiler().getSharedLibraryPathsAsArguments()).toEqual([]);
    });
});

describe('ECPPS IR tool', () => {
    it.each([
        [['IR'], 'ir'],
        [['Virtual', 'Instructions'], 'vinst'],
        [['AST'], 'ast'],
        [[], 'ir'],
        [['bogus'], 'ir'],
    ])('maps %j to the %s stage', (args, stage) => {
        expect(EcppsIrTool.stageFor(args)).toEqual(stage);
    });

    it('builds its own stage options and passes the source relative to its directory', () => {
        const args = EcppsIrTool.buildArguments(
            'vinst',
            '/tmp/dir/ecpps-vinst.txt',
            '/tmp/dir/example.cpp',
            ['/I:/opt/compiler-explorer/ecpps-trunk/include'],
            ['/verbose:pinst', '/asm', '/out:/tmp/dir/output.s', '/WX', '/tmp/dir/example.cpp', './example.cpp'],
        );
        expect(args).toEqual([
            '/verbose:vinst',
            '/asm',
            '/out:/tmp/dir/ecpps-vinst.txt',
            '/I:/opt/compiler-explorer/ecpps-trunk/include',
            '/WX',
            './example.cpp',
        ]);
    });

    describe('running', () => {
        afterEach(() => vi.restoreAllMocks());

        function makeTool(options: string[] = []) {
            return new EcppsIrTool(
                {id: 'ecppsir', name: 'ECPPS IR', exclude: [], options} as unknown as ToolInfo,
                {ceProps: (_key: string, def: any) => def, compilerProps: () => undefined} as unknown as ToolEnv,
            );
        }

        function execResult(stdout: string): UnprocessedExecResult {
            return {
                code: 0,
                okToCache: true,
                filenameTransform: (f: string) => f,
                stdout,
                stderr: '',
                execTime: 0,
                timedOut: false,
                truncated: false,
            };
        }

        const compilationInfo = {
            compiler: {exe: '/opt/compiler-explorer/ecpps-trunk/bin/ecpps', options: ''},
            options: [],
        } as unknown as CompilationInfo;

        it("runs in the source's directory and shows the stage dump", async () => {
            const dir = await newTempDir();
            const input = path.join(dir, 'example.cpp');
            const tool = makeTool();
            const exec = vi.spyOn(tool, 'exec').mockImplementation(async () => {
                await fs.writeFile(path.join(dir, 'ecpps-ast.txt'), 'line one\nline two\n');
                return execResult('ignored');
            });

            const result = await tool.runTool(compilationInfo, input, ['AST']);

            expect(exec.mock.calls[0][1]).toEqual([
                '/verbose:ast',
                '/asm',
                `/out:${path.join(dir, 'ecpps-ast.txt')}`,
                './example.cpp',
            ]);
            expect(exec.mock.calls[0][2].customCwd).toEqual(dir);
            expect(result.stdout.map(line => line.text)).toEqual(['line one', 'line two']);
        });

        it("passes the tool's configured options", async () => {
            const dir = await newTempDir();
            const tool = makeTool(['/warnings:all']);
            const exec = vi.spyOn(tool, 'exec').mockResolvedValue(execResult(''));

            await tool.runTool(compilationInfo, path.join(dir, 'example.cpp'), ['IR']);

            expect(exec.mock.calls[0][1]).toContain('/warnings:all');
        });

        it('shows the compiler output when there is no dump', async () => {
            const dir = await newTempDir();
            const tool = makeTool();
            vi.spyOn(tool, 'exec').mockResolvedValue(execResult('example.cpp:1:8 [Error] syntax-error'));

            const result = await tool.runTool(compilationInfo, path.join(dir, 'example.cpp'), ['IR']);

            expect(result.stdout.map(line => line.text).join('\n')).toContain('syntax-error');
        });
    });
});
