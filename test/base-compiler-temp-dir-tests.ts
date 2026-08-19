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

import {afterEach, beforeEach, describe, expect, it, vi} from 'vitest';

import {BaseCompiler} from '../lib/base-compiler.js';
import {CompilationEnvironment} from '../lib/compilation-env.js';
import {LocalExecutionEnvironment} from '../lib/execution/base-execution-env.js';
import * as utils from '../lib/utils.js';
import {BypassCache, CompilationResult} from '../types/compilation/compilation.interfaces.js';
import {BasicExecutionResult} from '../types/execution/execution.interfaces.js';
import {makeCompilationEnvironment, makeFakeCompilerInfo} from './utils.js';

const languages = {'c++': {id: 'c++'}, javascript: {id: 'javascript'}};

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

// Each flow here makes a temporary directory, and is expected to have removed it by the time it answers: the sweep
// that would otherwise reclaim it only runs when the queue is idle.
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

    describe('compile()', () => {
        const compileSource = () => compiler.compile('int main() {}', [], {}, {}, BypassCache.None, [], {}, [], []);

        it('removes the directory when writing the source fails', async () => {
            const dirs = trackTempDirs(compiler);
            vi.spyOn(compiler as any, 'writeAllFiles').mockRejectedValue(new Error('disk full'));

            const result = await compileSource();

            expect(result.code).toBe(-1);
            expect(dirs).toHaveLength(1);
            expect(await utils.dirExists(dirs[0])).toBe(false);
        });

        it('removes the directory when compilation throws', async () => {
            const dirs = trackTempDirs(compiler);
            vi.spyOn(compiler, 'doCompilation').mockRejectedValue(new Error('compiler fell over'));

            await expect(compileSource()).rejects.toThrow('compiler fell over');

            expect(dirs).toHaveLength(1);
            expect(await utils.dirExists(dirs[0])).toBe(false);
        });

        it('removes the directory after a successful compilation', async () => {
            const dirs = trackTempDirs(compiler);
            vi.spyOn(compiler, 'doCompilation').mockImplementation(async (_input, dirPath) => [
                compiledResult(dirPath),
                [],
                [],
            ]);

            const result = await compileSource();

            expect(result.code).toBe(0);
            expect(result.dirPath).toBeUndefined();
            expect(dirs).toHaveLength(1);
            expect(await utils.dirExists(dirs[0])).toBe(false);
        });
    });

    describe('handleInterpreting()', () => {
        it('removes the directory it ran the source from', async () => {
            const dirs = trackTempDirs(compiler);
            let ranFrom: string | undefined;
            vi.spyOn(compiler, 'runExecutable').mockImplementation(async (_exe, _params, homeDir) => {
                ranFrom = homeDir;
                expect(await utils.dirExists(homeDir)).toBe(true);
                return executedFine;
            });

            const key = compiler.getCacheKey('print(1)', [], {}, {}, [], [], []);
            const result = await compiler.handleInterpreting(key, {args: [], stdin: '', ldPath: [], env: {}});

            expect(result.didExecute).toBe(true);
            expect(ranFrom).toBe(dirs[0]);
            expect(await utils.dirExists(dirs[0])).toBe(false);
        });

        it('removes the directory even when running the source throws', async () => {
            const dirs = trackTempDirs(compiler);
            vi.spyOn(compiler, 'runExecutable').mockRejectedValue(new Error('no interpreter'));

            const key = compiler.getCacheKey('print(1)', [], {}, {}, [], [], []);
            await expect(compiler.handleInterpreting(key, {args: [], stdin: '', ldPath: [], env: {}})).rejects.toThrow(
                'no interpreter',
            );

            expect(await utils.dirExists(dirs[0])).toBe(false);
        });
    });

    describe('getOrBuildExecutable()', () => {
        const executeParameters = () => ({args: [], stdin: '', ldPath: [], env: {}});
        const key = () => compiler.getCacheKey('int main() {}', [], {}, {}, [], [], []);

        it('hands back the directory of a failed build so the caller can remove it', async () => {
            const dirs = trackTempDirs(compiler);
            vi.spyOn(compiler, 'buildExecutableInFolder').mockResolvedValue({
                code: 1,
                okToCache: false,
                stdout: [],
                stderr: [{text: 'nope'}],
                timedOut: false,
                downloads: [],
                executableFilename: '',
                compilationOptions: [],
            });

            const buildResult = await compiler.getOrBuildExecutable(key(), BypassCache.Compilation, 'hash');

            expect(buildResult.code).toBe(1);
            expect(buildResult.dirPath).toBe(dirs[0]);
            expect(await utils.dirExists(dirs[0])).toBe(true);
        });

        it('removes the directory when the build throws past the user error handler', async () => {
            const dirs = trackTempDirs(compiler);
            vi.spyOn(compiler, 'buildExecutableInFolder').mockImplementation(async (_key, dirPath) => ({
                code: 0,
                okToCache: true,
                stdout: [],
                stderr: [],
                timedOut: false,
                downloads: [],
                executableFilename: `${dirPath}/output.s`,
                compilationOptions: [],
            }));
            vi.spyOn(compiler, 'storePackageWithExecutable').mockRejectedValue(new Error('no cache'));

            await expect(compiler.getOrBuildExecutable(key(), BypassCache.Compilation, 'hash')).rejects.toThrow(
                'no cache',
            );

            expect(await utils.dirExists(dirs[0])).toBe(false);
        });

        it('leaves a failed build for compile() to remove once it has reported it', async () => {
            const dirs = trackTempDirs(compiler);
            vi.spyOn(compiler, 'buildExecutableInFolder').mockResolvedValue({
                code: 1,
                okToCache: false,
                stdout: [],
                stderr: [{text: 'nope'}],
                timedOut: false,
                downloads: [],
                executableFilename: '',
                compilationOptions: [],
            });

            const result = await compiler.compile(
                'int main() {}',
                [],
                {executorRequest: true},
                {},
                BypassCache.Compilation,
                [],
                executeParameters(),
                [],
                [],
            );

            expect(result.didExecute).toBe(false);
            expect(result.buildResult?.dirPath).toBeUndefined();
            expect(dirs).toHaveLength(1);
            expect(await utils.dirExists(dirs[0])).toBe(false);
        });

        it('removes the build directory when running the executable throws', async () => {
            const dirs = trackTempDirs(compiler);
            vi.spyOn(compiler, 'buildExecutableInFolder').mockImplementation(async (_key, dirPath) => ({
                code: 0,
                okToCache: true,
                stdout: [],
                stderr: [],
                timedOut: false,
                downloads: [],
                executableFilename: `${dirPath}/output.s`,
                compilationOptions: [],
            }));
            vi.spyOn(compiler, 'storePackageWithExecutable').mockResolvedValue();
            vi.spyOn(utils, 'fileExists').mockResolvedValue(true);
            vi.spyOn(compiler, 'runExecutable').mockRejectedValue(new Error('sandbox missing'));

            await expect(compiler.doExecution(key(), executeParameters(), BypassCache.Compilation)).rejects.toThrow(
                'sandbox missing',
            );

            expect(await utils.dirExists(dirs[0])).toBe(false);
        });

        // Java, Kotlin and Cerberus run the built executable themselves from handleInterpreting, through this.
        it('removes the build directory when what withBuiltExecutable runs throws', async () => {
            const dirs = trackTempDirs(compiler);
            vi.spyOn(compiler, 'buildExecutableInFolder').mockResolvedValue({
                code: 0,
                okToCache: true,
                stdout: [],
                stderr: [],
                timedOut: false,
                downloads: [],
                executableFilename: '',
                compilationOptions: [],
            });
            vi.spyOn(compiler, 'storePackageWithExecutable').mockResolvedValue();

            await expect(
                (compiler as any).withBuiltExecutable(key(), BypassCache.Compilation, 'hash', async () => {
                    throw new Error('no main class');
                }),
            ).rejects.toThrow('no main class');

            expect(dirs).toHaveLength(1);
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

        it('removes the project directory once the build has been post-processed', async () => {
            const dirs = trackTempDirs(compiler);
            vi.spyOn(compiler as any, 'runProjectBuild').mockImplementation(async (build: any) => {
                expect(await utils.dirExists(build.dirPath)).toBe(true);
                return {
                    code: 0,
                    timedOut: false,
                    stdout: [],
                    stderr: [],
                    buildsteps: [],
                    result: compiledResult(build.dirPath),
                };
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
    });

    describe('execCompilerCached()', () => {
        it('releases and removes the scratch directory when the compiler throws', async () => {
            const dirs = trackTempDirs(compiler);
            (compiler as any).mtime = new Date();
            vi.spyOn(compiler, 'exec').mockRejectedValue(new Error('cannot spawn'));

            await expect(
                compiler.execCompilerCached('fake-compiler', ['--version'], {
                    ...compiler.getDefaultExecOptions(),
                    createAndUseTempDir: true,
                }),
            ).rejects.toThrow('cannot spawn');

            expect(dirs).toHaveLength(1);
            await vi.waitFor(async () => expect(await utils.dirExists(dirs[0])).toBe(false));
        });
    });
});

describe('LocalExecutionEnvironment cleanup', () => {
    let ce: CompilationEnvironment;

    beforeEach(() => {
        ce = makeCompilationEnvironment({languages, doCache: false});
    });

    afterEach(() => vi.restoreAllMocks());

    it('removes what it downloaded', async () => {
        const env = new LocalExecutionEnvironment(ce);
        let downloadedTo: string | undefined;
        vi.spyOn(env as any, 'loadPackageWithExecutable').mockImplementation(async (_hash, dirPath) => {
            downloadedTo = dirPath as string;
            return {code: 0, executableFilename: 'a.out'};
        });

        await env.downloadExecutablePackage('hash');
        expect(await utils.dirExists(downloadedTo!)).toBe(true);

        await env.cleanup();
        expect(await utils.dirExists(downloadedTo!)).toBe(false);
    });

    it('is harmless before anything was downloaded', async () => {
        const env = new LocalExecutionEnvironment(ce);
        await expect(env.cleanup()).resolves.toBeUndefined();
    });
});
