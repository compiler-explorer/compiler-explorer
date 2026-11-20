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

/**
 * Pins down how the compilation cache key is built today, ahead of ce-router#17 (checking the cache in the router
 * before queueing). A router-side key only helps if it matches the worker's byte for byte, so these tests record the
 * current behaviour, quirks included, rather than what it ought to be.
 *
 * The request bodies are the ones the frontend sent for each layout, captured from a dev instance running with
 * `logCompilationCacheAccesses=true`. Only the source text is replaced.
 */

import {afterEach, describe, expect, it, vi} from 'vitest';

import {BaseCompiler} from '../lib/base-compiler.js';
import {cmakeBuildSystem} from '../lib/build-systems/index.js';
import {BaseCache} from '../lib/cache/base.js';
import {CompilationEnvironment} from '../lib/compilation-env.js';
import {CompileHandler} from '../lib/handlers/compile.js';
import {ToolEnv} from '../lib/tooling/base-tool.interface.js';
import {BaseTool} from '../lib/tooling/base-tool.js';
import {BypassCache, CompilationInfo, FiledataPair} from '../types/compilation/compilation.interfaces.js';
import {CompilerInfo} from '../types/compiler.interfaces.js';
import {ToolInfo} from '../types/tool.interfaces.js';
import {makeCompilationEnvironment, makeFakeCompilerInfo} from './utils.js';

const languages = {'c++': {id: 'c++'}} as const;

const SOURCE = '#include <cstdio>\nint main() { std::puts("hi"); return 0; }\n';

// What the compiler pane sends with no extra panes open, in the order it sends it.
const UI_COMPILER_OPTIONS = {
    producePp: null,
    produceGccDump: {libraryFunctions: false},
    produceOptInfo: false,
    produceCfg: false,
    produceIr: null,
    produceClangir: null,
    produceOptPipeline: null,
    produceDevice: false,
    produceLeanC: null,
    produceYul: null,
    overrides: [],
    darkBackgroundColours: true,
};

// The compiler pane's filters, in the order it sends them.
const UI_FILTERS = {
    binaryObject: false,
    binary: false,
    execute: false,
    intel: true,
    demangle: true,
    labels: true,
    libraryCode: true,
    directives: true,
    commentOnly: true,
    trim: false,
    debugCalls: false,
};

function compilerPaneBody(changes: {
    compilerOptions?: object;
    filters?: object;
    userArguments?: string;
    tools?: any[];
}) {
    return {
        source: SOURCE,
        compiler: 'g161',
        options: {
            userArguments: changes.userArguments ?? '-O2',
            compilerOptions: {...UI_COMPILER_OPTIONS, ...changes.compilerOptions},
            filters: {...UI_FILTERS, ...changes.filters},
            tools: changes.tools ?? [],
            libraries: [],
            executeParameters: {args: [], stdin: ''},
        },
        lang: 'c++',
        bypassCache: 0,
        allowStoreCodeDebug: true,
    };
}

function executorPaneBody(args = '', stdin = '') {
    return {
        source: SOURCE,
        compiler: 'g161',
        options: {
            userArguments: '-O2',
            executeParameters: {args, stdin, runtimeTools: []},
            compilerOptions: {executorRequest: true, skipAsm: true, overrides: [], darkBackgroundColours: true},
            filters: {execute: true},
            tools: [],
            libraries: [],
        },
        lang: 'c++',
        allowStoreCodeDebug: true,
    };
}

// What /noscript/ posts: a form, so every value arrives as a string and only the ticked boxes are present.
const NOSCRIPT_FORM_BODY = {
    lang: 'c++',
    filterAnsi: 'true',
    intel: 'true',
    demangle: 'true',
    verboseDemangling: 'true',
    labels: 'true',
    libraryCode: 'true',
    directives: 'true',
    commentOnly: 'true',
    compiler: 'g161',
    userArguments: '-O2',
    source: SOURCE,
};

function makeCompiler(info: Partial<CompilerInfo> = {}) {
    const env = makeCompilationEnvironment({languages});
    const compiler = new BaseCompiler(
        makeFakeCompilerInfo({id: 'g161', exe: '/dev/null', lang: 'c++', options: '', supportsBinary: true, ...info}),
        env,
    );
    vi.spyOn(compiler as any, 'storeOversizedResult').mockResolvedValue(undefined);
    return {env, compiler};
}

type CacheAnswer = (key: unknown) => object | null;
const alwaysHit: CacheAnswer = () => ({code: 0, stdout: [], stderr: [], asm: []});

interface Recorded {
    key: any;
    hash: string;
    gets: unknown[];
    puts: unknown[];
}

// Runs a request the way the HTTP handler and the SQS worker both do: parse it, then compile() it. The cache is
// faked, so nothing is actually compiled; what matters is what compile() looked up and stored.
async function recordCacheKeys(
    body: any,
    opts: {
        isJson?: boolean;
        query?: any;
        files?: FiledataPair[];
        info?: Partial<CompilerInfo>;
        answer?: CacheAnswer;
        setup?: (compiler: BaseCompiler, env: CompilationEnvironment) => void;
    } = {},
): Promise<Recorded> {
    const {env, compiler} = makeCompiler(opts.info);
    const answer = opts.answer ?? alwaysHit;
    const cacheGet = vi.spyOn(env, 'cacheGet').mockImplementation(async key => answer(key));
    const cachePut = vi.spyOn(env, 'cachePut').mockResolvedValue(undefined as any);
    opts.setup?.(compiler, env);
    const parsed = CompileHandler.parseRequestReusable(opts.isJson ?? true, opts.query ?? {}, body, compiler);
    await compiler.compile(
        parsed.source,
        parsed.options,
        parsed.backendOptions,
        parsed.filters,
        parsed.bypassCache,
        parsed.tools,
        parsed.executeParameters,
        parsed.libraries,
        opts.files ?? [],
    );
    const key = cacheGet.mock.calls[0][0] as any;
    return {
        key,
        hash: BaseCache.hash(key),
        gets: cacheGet.mock.calls.map(call => call[0]),
        puts: cachePut.mock.calls.map(call => call[0]),
    };
}

describe('Compilation cache key', () => {
    afterEach(() => vi.restoreAllMocks());

    // ------------------------------------------------------------------------------------------------
    // The key built for what the frontend sends
    // ------------------------------------------------------------------------------------------------
    describe('for a compiler pane request', () => {
        it('holds the full CompilerInfo and the parsed request, with filters completed from the defaults', async () => {
            const {key} = await recordCacheKeys(compilerPaneBody({}));
            const {darkBackgroundColours, ...backendOptions} = UI_COMPILER_OPTIONS;
            expect(darkBackgroundColours).toBe(true);

            expect(Object.keys(key)).toEqual([
                'compiler',
                'source',
                'options',
                'backendOptions',
                'filters',
                'tools',
                'libraries',
                'files',
            ]);
            expect(key.compiler).toMatchObject({id: 'g161', exe: '/dev/null', lang: 'c++'});
            expect(key).toMatchObject({
                source: SOURCE,
                options: ['-O2'],
                backendOptions,
                tools: [],
                libraries: [],
                files: [],
            });
            // The compiler's default-filter order, not the order the frontend sent, plus optOutput which the
            // frontend never sends.
            expect(Object.entries(key.filters)).toEqual([
                ['binary', false],
                ['execute', false],
                ['demangle', true],
                ['intel', true],
                ['commentOnly', true],
                ['directives', true],
                ['labels', true],
                ['optOutput', false],
                ['libraryCode', true],
                ['trim', false],
                ['binaryObject', false],
                ['debugCalls', false],
            ]);
        });

        it('is the very CompilerInfo object the compiler holds, so any change to it is a new key', async () => {
            const {compiler, env} = makeCompiler();
            const cacheGet = vi.spyOn(env, 'cacheGet').mockResolvedValue(alwaysHit(null));
            await compiler.compile(SOURCE, ['-O2'], {}, {} as any, BypassCache.None, [], {} as any, [], []);
            expect((cacheGet.mock.calls[0][0] as any).compiler).toBe(compiler.getInfo());

            const base = await recordCacheKeys(compilerPaneBody({}));
            const otherVersion = await recordCacheKeys(compilerPaneBody({}), {info: {version: 'gcc 16.1.1'}});
            expect(otherVersion.hash).not.toEqual(base.hash);
        });

        it('does not depend on the order the frontend sends filters in', async () => {
            const reversed = Object.fromEntries(Object.entries(UI_FILTERS).reverse());
            const body = compilerPaneBody({});
            body.options.filters = reversed as typeof UI_FILTERS;
            expect((await recordCacheKeys(body)).hash).toEqual((await recordCacheKeys(compilerPaneBody({}))).hash);
        });

        it('does depend on the order the frontend sends compilerOptions in', async () => {
            const body = compilerPaneBody({});
            body.options.compilerOptions = Object.fromEntries(
                Object.entries(UI_COMPILER_OPTIONS).reverse(),
            ) as typeof UI_COMPILER_OPTIONS;
            expect((await recordCacheKeys(body)).hash).not.toEqual((await recordCacheKeys(compilerPaneBody({}))).hash);
        });

        it('ignores the fields that only steer the request: bypassCache, executeParameters, lang', async () => {
            const base = await recordCacheKeys(compilerPaneBody({}));
            const body: any = compilerPaneBody({});
            body.bypassCache = BypassCache.Execution;
            body.lang = 'c';
            body.allowStoreCodeDebug = false;
            body.options.executeParameters = {args: ['x'], stdin: 'y'};
            expect((await recordCacheKeys(body)).hash).toEqual(base.hash);
        });

        it('tells a pane that was opened and closed again from one that never was', async () => {
            // The frontend sends produceAst: undefined (so nothing) until the AST pane has been opened, then true,
            // then false once it's closed. The compiler's output is the same either side, the key is not.
            const neverOpened = await recordCacheKeys(compilerPaneBody({}));
            const open = await recordCacheKeys(compilerPaneBody({compilerOptions: {produceAst: true}}));
            const closed = await recordCacheKeys(compilerPaneBody({compilerOptions: {produceAst: false}}));

            expect(open.hash).not.toEqual(neverOpened.hash);
            expect(closed.hash).not.toEqual(neverOpened.hash);
            expect(closed.hash).not.toEqual(open.hash);
        });

        it('keeps a pane-specific option in the key even for a compiler that ignores it', async () => {
            const gccDump = {
                opened: true,
                treeDump: true,
                rtlDump: true,
                ipaDump: true,
                dumpFlags: {gimpleFe: false, address: false},
                libraryFunctions: false,
            };
            const {key} = await recordCacheKeys(compilerPaneBody({compilerOptions: {produceGccDump: gccDump}}), {
                info: {supportsGccDump: false},
            });
            expect(key.backendOptions.produceGccDump).toEqual(gccDump);
        });
    });

    // ------------------------------------------------------------------------------------------------
    // The same compile reached by different routes
    // ------------------------------------------------------------------------------------------------
    describe('across request routes', () => {
        it('is the same for an SQS message as for the HTTP request it was made from', async () => {
            const http = await recordCacheKeys(compilerPaneBody({}));
            const sqsMessage = {
                ...compilerPaneBody({}),
                guid: 'abc',
                compilerId: 'g161',
                headers: {'content-type': 'application/json'},
                queryStringParameters: {},
                queueTimeMs: 5,
            };
            expect((await recordCacheKeys(sqsMessage, {query: sqsMessage.queryStringParameters})).hash).toEqual(
                http.hash,
            );
        });

        it('differs for /noscript/, whose form sets other backend options', async () => {
            const json = await recordCacheKeys(compilerPaneBody({}));
            const noscript = await recordCacheKeys(NOSCRIPT_FORM_BODY, {isJson: false});

            expect(noscript.key.backendOptions).toEqual({filterAnsi: true, skipAsm: false});
            // Unticked boxes are false, and verboseDemangling, not being a default filter, is dropped.
            expect(noscript.key.filters).toEqual({...json.key.filters});
            expect(noscript.hash).not.toEqual(json.hash);
        });

        it('differs again for the plain-text REST API', async () => {
            const json = await recordCacheKeys(compilerPaneBody({}));
            const rest = await recordCacheKeys(SOURCE, {isJson: false, query: {options: '-O2'}});

            expect(rest.key.backendOptions).toEqual({skipAsm: false, skipPopArgs: false, filterAnsi: false});
            expect(rest.key.filters.libraryCode).toBe(false);
            expect(rest.hash).not.toEqual(json.hash);
        });
    });

    // ------------------------------------------------------------------------------------------------
    // Changes compile() makes before it builds the key
    // ------------------------------------------------------------------------------------------------
    describe('normalisation before the key is built', () => {
        it('turns every filter off for -E on C++', async () => {
            const {key} = await recordCacheKeys(compilerPaneBody({userArguments: '-E'}));
            expect(Object.values(key.filters).every(value => value === false)).toBe(true);
        });

        it('drops binary for a compiler that cannot produce one', async () => {
            const supported = await recordCacheKeys(compilerPaneBody({filters: {binary: true}}));
            const unsupported = await recordCacheKeys(compilerPaneBody({filters: {binary: true}}), {
                info: {supportsBinary: false},
            });
            expect(supported.key.filters.binary).toBe(true);
            expect(unsupported.key.filters).not.toHaveProperty('binary');
        });

        it('stops masking filenames when there are extra files', async () => {
            const {key} = await recordCacheKeys(compilerPaneBody({}), {
                files: [{filename: 'other.h', contents: ''}],
            });
            expect(key.filters.dontMaskFilenames).toBe(true);
            expect(key.files).toEqual([{filename: 'other.h', contents: ''}]);
        });
    });

    // ------------------------------------------------------------------------------------------------
    // Execution
    // ------------------------------------------------------------------------------------------------
    describe('with execution', () => {
        it('gives compile-and-execute a different compile key from compile only', async () => {
            const compileOnly = await recordCacheKeys(compilerPaneBody({}));
            const withExecute = await recordCacheKeys(compilerPaneBody({filters: {execute: true}}));
            expect(withExecute.key.filters.execute).toBe(true);
            expect(withExecute.hash).not.toEqual(compileOnly.hash);
        });

        it('looks up the run under a JSON string of the compile key and the execution parameters', async () => {
            const body = compilerPaneBody({filters: {execute: true}});
            body.options.executeParameters = {args: ['a', 'b'] as any, stdin: 'in'};
            const {key, gets} = await recordCacheKeys(body);

            expect(gets).toHaveLength(2);
            expect(gets[1]).toEqual(
                JSON.stringify({
                    key,
                    executeParameters: {args: ['a', 'b'], stdin: 'in', ldPath: [], env: {}, runtimeTools: []},
                }),
            );
        });

        it('builds an executor pane request a key of its own, which it looks up but never stores', async () => {
            const {key, gets, puts} = await recordCacheKeys(executorPaneBody(), {
                answer: () => null,
                setup: compiler =>
                    vi.spyOn(compiler, 'doExecution').mockResolvedValue({code: 0, stdout: [], stderr: []} as any),
            });

            expect(key.backendOptions).toEqual({executorRequest: true, skipAsm: true, overrides: []});
            // {execute: true} merged over the defaults, so libraryCode is false where the compiler pane sends true.
            expect(key.filters).toMatchObject({execute: true, libraryCode: false});
            expect(gets).toHaveLength(2);
            expect(typeof gets[1]).toBe('string');
            expect(puts).toEqual([gets[1]]);
        });

        it('shares an executor compile key across arguments and stdin, but not its execution key', async () => {
            const executorOnly = {setup: (c: BaseCompiler) => vi.spyOn(c, 'doExecution').mockResolvedValue({} as any)};
            const plain = await recordCacheKeys(executorPaneBody(), {answer: () => null, ...executorOnly});
            const withInput = await recordCacheKeys(executorPaneBody('a b', 'in'), {
                answer: () => null,
                ...executorOnly,
            });

            expect(withInput.hash).toEqual(plain.hash);
            expect(withInput.gets[1]).not.toEqual(plain.gets[1]);
        });
    });

    // ------------------------------------------------------------------------------------------------
    // Tools
    // ------------------------------------------------------------------------------------------------
    describe('with tools', () => {
        it("appends the source path to the request's own tool arguments, which the key also holds", async () => {
            // compile() passes key.tools[i].args straight to runTool. A tool that doesn't prepend its configured
            // options (clang-tidy, clang-query, PVS-Studio, ...) pushes the temporary source path onto that array,
            // so afterCompilation stores the result under a key that names a temp directory, and the key looked up
            // next time never matches it.
            const tool = new BaseTool(
                {id: 'tidy', exe: '/bin/true', options: []} as unknown as ToolInfo,
                {
                    ceProps: () => undefined,
                    compilerProps: () => undefined,
                } as unknown as ToolEnv,
            );
            (tool as any).addOptionsToToolArgs = false;
            vi.spyOn(tool, 'exec').mockResolvedValue({
                code: 0,
                stdout: '',
                stderr: '',
                filenameTransform: (f: string) => f,
            } as any);

            const requestArgs: string[] = [];
            const key = {tools: [{id: 'tidy', args: requestArgs, stdin: ''}]};
            const hashBefore = BaseCache.hash(key);

            await tool.runTool(
                {compiler: {lang: 'c++'}} as unknown as CompilationInfo,
                '/tmp/ce-xyz/example.cpp',
                requestArgs,
            );

            expect(key.tools[0].args).toEqual(['/tmp/ce-xyz/example.cpp']);
            expect(BaseCache.hash(key)).not.toEqual(hashBefore);
        });
    });

    // ------------------------------------------------------------------------------------------------
    // Project builds (CMake)
    // ------------------------------------------------------------------------------------------------
    describe('for a CMake build', () => {
        const CMAKE_FILES: FiledataPair[] = [{filename: 'main.cpp', contents: SOURCE}];
        const CMAKELISTS = 'cmake_minimum_required(VERSION 3.5)\nproject(hello)\nadd_executable(output.s main.cpp)\n';

        function cmakeBody(execute: boolean) {
            const body: any = compilerPaneBody({filters: {execute}});
            body.source = CMAKELISTS;
            body.files = CMAKE_FILES;
            body.options.compilerOptions = {...body.options.compilerOptions, cmakeArgs: '', customOutputFilename: ''};
            return body;
        }

        async function recordProjectBuild(body: any) {
            const {env, compiler} = makeCompiler();
            const cacheGet = vi.spyOn(env, 'cacheGet');
            const hashed: string[] = [];
            const getExecutableHash = env.getExecutableHash.bind(env);
            vi.spyOn(env, 'getExecutableHash').mockImplementation(key => {
                hashed.push(JSON.stringify(key));
                return getExecutableHash(key);
            });
            // Stop as soon as it would look for a previous build: everything the key depends on is settled by then.
            const stop = new Error('stop here');
            vi.spyOn(compiler, 'fetchExecutablePackage').mockRejectedValue(stop);

            const parsed = CompileHandler.parseRequestReusable(true, {}, body, compiler, cmakeBuildSystem);
            await expect(compiler.buildProject(cmakeBuildSystem, body.files, parsed, BypassCache.None)).rejects.toBe(
                stop,
            );
            return {cacheGet, key: JSON.parse(hashed[0]), hash: env.getExecutableHash(JSON.parse(hashed[0]))};
        }

        it('never reads the compilation cache, only the executable package cache', async () => {
            const {cacheGet} = await recordProjectBuild(cmakeBody(false));
            expect(cacheGet).not.toHaveBeenCalled();
        });

        it('keys on the CMakeLists.txt, the files and the build system, with execute removed', async () => {
            const {key} = await recordProjectBuild(cmakeBody(true));

            expect(Object.keys(key)).toEqual([
                'source',
                'options',
                'backendOptions',
                'filters',
                'libraries',
                'compiler',
                'files',
                'api',
            ]);
            expect(key).toMatchObject({source: CMAKELISTS, files: CMAKE_FILES, api: 'cmake', options: ['-O2']});
            expect(key.backendOptions).toMatchObject({cmakeArgs: '', customOutputFilename: ''});
            expect(key.backendOptions).not.toHaveProperty('darkBackgroundColours');
            expect(key.filters).not.toHaveProperty('execute');
            expect(key.filters).toMatchObject({binary: true, dontMaskFilenames: true});
        });

        it('is the same whether or not the build is to be run', async () => {
            expect((await recordProjectBuild(cmakeBody(true))).hash).toEqual(
                (await recordProjectBuild(cmakeBody(false))).hash,
            );
        });
    });
});
