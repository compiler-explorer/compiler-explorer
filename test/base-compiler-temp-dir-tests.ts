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

import {afterEach, beforeEach, describe, expect, it, vi} from 'vitest';

import {BaseCompiler} from '../lib/base-compiler.js';
import {CompilationEnvironment} from '../lib/compilation-env.js';
import {ClangCompiler} from '../lib/compilers/clang.js';
import * as utils from '../lib/utils.js';
import {BypassCache, CompilationResult} from '../types/compilation/compilation.interfaces.js';
import {BasicExecutionResult} from '../types/execution/execution.interfaces.js';
import {makeCompilationEnvironment, makeFakeCompilerInfo} from './utils.js';

const languages = {'c++': {id: 'c++'}};

const executedFine: BasicExecutionResult = {
    code: 0,
    okToCache: true,
    filenameTransform: x => x,
    stdout: [],
    stderr: [],
    execTime: 1,
    timedOut: false,
};

function compiledResult(dirPath: string): CompilationResult {
    return {code: 0, okToCache: false, stdout: [], stderr: [], asm: '', timedOut: false, dirPath};
}

/** The directories a compiler made, in the order it made them, so a test can check what became of each. */
function trackTempDirs(compiler: BaseCompiler): string[] {
    const made: string[] = [];
    const original = compiler.newTempDir.bind(compiler);
    vi.spyOn(compiler, 'newTempDir').mockImplementation(async options => {
        const dir = await original(options);
        made.push(dir);
        return dir;
    });
    return made;
}

// The sweep that would otherwise reclaim these directories only runs when the compilation queue is idle.
describe('Per-request temp directory cleanup', () => {
    let ce: CompilationEnvironment;
    let compiler: BaseCompiler;

    beforeEach(() => {
        ce = makeCompilationEnvironment({languages, doCache: false});
        compiler = new BaseCompiler(
            makeFakeCompilerInfo({
                exe: 'fake-compiler',
                lang: 'c++',
                ldPath: [],
                libPath: [],
                supportsBinary: true,
                supportsExecute: true,
            }),
            ce,
        );
    });

    afterEach(() => vi.restoreAllMocks());

    describe('handleInterpreting()', () => {
        const run = () => {
            const key = compiler.getCacheKey('print(1)', [], {}, {}, [], [], []);
            return compiler.handleInterpreting(key, {args: [], stdin: '', ldPath: [], env: {}});
        };

        it('removes the directory it ran the source from', async () => {
            const dirs = trackTempDirs(compiler);
            let ranFrom: string | undefined;
            vi.spyOn(compiler, 'runExecutable').mockImplementation(async (_exe, _params, homeDir) => {
                ranFrom = homeDir;
                expect(await utils.dirExists(homeDir)).toBe(true);
                return executedFine;
            });

            const result = await run();

            expect(result.didExecute).toBe(true);
            expect(ranFrom).toBe(dirs[0]);
            expect(await utils.dirExists(dirs[0])).toBe(false);
        });

        it('removes the directory even when running the source throws', async () => {
            const dirs = trackTempDirs(compiler);
            vi.spyOn(compiler, 'runExecutable').mockRejectedValue(new Error('no interpreter'));

            await expect(run()).rejects.toThrow('no interpreter');

            expect(await utils.dirExists(dirs[0])).toBe(false);
        });
    });

    describe('buildProject()', () => {
        const parsedRequest = () => ({
            source: 'cmake_minimum_required(VERSION 3.5)',
            options: [],
            backendOptions: {},
            filters: {},
            bypassCache: BypassCache.Compilation,
            tools: [],
            executeParameters: {args: [], stdin: ''},
            libraries: [],
        });
        const files = [{filename: 'main.cpp', contents: 'int main() {}'}];

        const builtOk = (dirPath: string) => ({
            code: 0,
            timedOut: false,
            stdout: [],
            stderr: [],
            buildsteps: [],
            result: compiledResult(dirPath),
        });

        it('removes the project directory once the build has been post-processed', async () => {
            const dirs = trackTempDirs(compiler);
            vi.spyOn(compiler as any, 'runProjectBuild').mockImplementation(async (build: any) => {
                expect(await utils.dirExists(build.dirPath)).toBe(true);
                return builtOk(build.dirPath);
            });
            vi.spyOn(compiler, 'afterCmakeCompilation').mockImplementation(async fullResult => {
                expect(await utils.dirExists(dirs[0])).toBe(true);
                return fullResult;
            });

            const result = await compiler.cmake(files, parsedRequest(), BypassCache.Compilation);

            expect(result.code).toBe(0);
            expect(result.dirPath).toBeUndefined();
            expect(result.result?.dirPath).toBeUndefined();
            expect(dirs).toHaveLength(1);
            expect(await utils.dirExists(dirs[0])).toBe(false);
        });

        it('removes the project directory when the build throws', async () => {
            const dirs = trackTempDirs(compiler);
            vi.spyOn(compiler as any, 'runProjectBuild').mockRejectedValue(new Error('cmake exploded'));

            await expect(compiler.cmake(files, parsedRequest(), BypassCache.Compilation)).rejects.toThrow(
                'cmake exploded',
            );

            expect(dirs).toHaveLength(1);
            expect(await utils.dirExists(dirs[0])).toBe(false);
        });

        it('leaves the project directory for the sweep when delayCleanupTemp is set', async () => {
            const dirs = trackTempDirs(compiler);
            (compiler as any).delayCleanupTemp = true;
            vi.spyOn(compiler as any, 'runProjectBuild').mockImplementation(async (build: any) =>
                builtOk(build.dirPath),
            );

            await compiler.cmake(files, parsedRequest(), BypassCache.Compilation);

            expect(await utils.dirExists(dirs[0])).toBe(true);
            await fs.rm(dirs[0], {recursive: true, force: true});
        });
    });
});

describe('ClangCompiler.afterCompilation', () => {
    afterEach(() => vi.restoreAllMocks());

    const afterCompilation = (clang: ClangCompiler, result: CompilationResult, delayCaching: boolean) =>
        clang.afterCompilation(
            result,
            false,
            clang.getCacheKey('int main() {}', [], {}, {}, [], [], []),
            {args: [], stdin: '', ldPath: [], env: {}},
            [],
            {skipAsm: true},
            {} as any,
            [],
            undefined,
            undefined,
            BypassCache.Compilation,
            undefined,
            delayCaching,
        );

    let clang: ClangCompiler;
    let dir: string;

    beforeEach(async () => {
        const ce = makeCompilationEnvironment({languages, doCache: false});
        clang = new ClangCompiler(makeFakeCompilerInfo({exe: 'clang', lang: 'c++', ldPath: [], libPath: []}), ce);
        vi.spyOn(clang, 'addTimeTraceToResult').mockResolvedValue(undefined);
        dir = await clang.newTempDir();
    });

    afterEach(async () => {
        await fs.rm(dir, {recursive: true, force: true});
    });

    it('cleans up after itself when it is the whole of a compilation', async () => {
        await afterCompilation(clang, compiledResult(dir), false);

        expect(await utils.dirExists(dir)).toBe(false);
    });

    it('leaves the directory to the project build when told the caller is not finished with it', async () => {
        await afterCompilation(clang, compiledResult(dir), true);

        expect(await utils.dirExists(dir)).toBe(true);
    });
});
