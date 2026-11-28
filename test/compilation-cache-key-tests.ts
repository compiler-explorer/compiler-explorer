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
 * Pins down how the compilation cache key is built, for ce-router#17 (checking the cache in the router before
 * queueing). A router-side key only helps if it matches the worker's byte for byte. The key is the request as it
 * arrived plus the compiler's id, so it can be computed without knowing anything about the compiler: whatever compile()
 * does to adapt the request to a compiler (default filters, -E, binary support, library detection, the background
 * colour) happens to a copy, after the key is built.
 *
 * The cache stores a result under the HMAC-SHA256 of JSON.stringify(key), so the property order matters as much as the
 * values. expectKey() therefore checks the serialised form, not just deep equality.
 *
 * The request bodies are the ones the frontend sent for each layout, captured from a dev instance running with
 * `logCompilationCacheAccesses=true`. Only the source text is replaced.
 */

import fs from 'node:fs';
import path from 'node:path';

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

const COMPILER_INFO = {id: 'g161', exe: '/dev/null', lang: 'c++' as const, options: '', supportsBinary: true};

// All the key holds of the compiler.
const KEY_COMPILER = {id: 'g161'};

// What the compiler pane sends with no extra panes open, in the order it sends it. darkBackgroundColours is only sent
// for compilers with supportsDarkBackgroundColours, which none of these have.
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

// BaseCompiler.getDefaultFilters(), in its order. The routes that don't send JSON (/noscript/, the plain-text API)
// still build their filters from these.
const DEFAULT_FILTERS = {
    binary: false,
    execute: false,
    demangle: true,
    intel: true,
    commentOnly: true,
    directives: true,
    labels: true,
    optOutput: false,
    libraryCode: false,
    trim: false,
    binaryObject: false,
    debugCalls: false,
};

// The order the compiler pane builds compilerOptions in (static/panes/compiler.ts, compile()). It always builds the same
// object literal; a pane that has never been opened contributes undefined, which JSON.stringify leaves out.
const FRONTEND_COMPILER_OPTIONS_ORDER = [
    'producePp',
    'produceAst',
    'produceGccDump',
    'produceOptInfo',
    'produceStackUsageInfo',
    'produceCfg',
    'produceGnatDebugTree',
    'produceGnatDebug',
    'produceIr',
    'produceClangir',
    'produceOptPipeline',
    'produceDevice',
    'produceRustMir',
    'produceRustMacroExp',
    'produceRustHir',
    'produceHaskellCore',
    'produceHaskellStg',
    'produceHaskellCmm',
    'produceLeanC',
    'produceClojureMacroExp',
    'produceYul',
    'overrides',
    'darkBackgroundColours',
];

function inFrontendOrder(options: Record<string, unknown>) {
    const known = FRONTEND_COMPILER_OPTIONS_ORDER.filter(name => name in options).map(name => [name, options[name]]);
    const extra = Object.entries(options).filter(([name]) => !FRONTEND_COMPILER_OPTIONS_ORDER.includes(name));
    return Object.fromEntries([...known, ...extra]);
}

function compilerPaneBody(
    changes: {
        compilerOptions?: object;
        filters?: object;
        userArguments?: string;
        tools?: any[];
        libraries?: any[];
    } = {},
) {
    return {
        source: SOURCE,
        compiler: 'g161',
        options: {
            userArguments: changes.userArguments ?? '-O2',
            compilerOptions: inFrontendOrder({...UI_COMPILER_OPTIONS, ...changes.compilerOptions}),
            filters: {...UI_FILTERS, ...changes.filters},
            tools: changes.tools ?? [],
            libraries: changes.libraries ?? [],
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
            compilerOptions: {executorRequest: true, skipAsm: true, overrides: []},
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

// Checks the key both ways: deep equality for a readable diff, then the exact JSON, which is what gets hashed.
function expectKey(actual: unknown, expected: unknown) {
    expect(JSON.parse(JSON.stringify(actual))).toEqual(expected);
    expect(JSON.stringify(actual)).toBe(JSON.stringify(expected));
}

// The CompilerInfo a dev instance (production c++ config, --language c++ c rust) put into its cache keys when they still
// held all of it, as logged by logCompilationCacheAccesses. Its hash is in REAL_COMPILER_INFO_HASH.
function loadRealCompilerInfo(id: string): CompilerInfo {
    return JSON.parse(
        fs.readFileSync(path.join(__dirname, 'compilation-cache-key', `${id}.compiler-info.json`), 'utf8'),
    ) as CompilerInfo;
}

const REAL_COMPILER_INFO_HASH: Record<string, string> = {
    g161: '43d742db0d22f69563a858e33b726ae9b90142a90cf9bdd62303fdf08da4c8b1',
    clang2110: '4b2c78d6921530c14bf034bd38221050be98fc80e9c96c6cd99b38e0076380c4',
};

function makeCompiler(info: Partial<CompilerInfo> = {}, realCompilerId?: string) {
    const env = makeCompilationEnvironment({languages});
    const compiler = new BaseCompiler(makeFakeCompilerInfo({...COMPILER_INFO, ...info}), env);
    // Swapped in after construction, so the key holds exactly what the dev instance's key held.
    if (realCompilerId) (compiler as any).compiler = loadRealCompilerInfo(realCompilerId);
    vi.spyOn(compiler as any, 'storeOversizedResult').mockResolvedValue(undefined);
    // Only the keys are under test, and the real CompilerInfo names compilers that CI doesn't have.
    vi.spyOn(compiler, 'exec').mockRejectedValue(new Error('a cache key test must not run the compiler'));
    return {env, compiler};
}

type CacheAnswer = (key: unknown) => object | null;
const alwaysHit: CacheAnswer = () => ({code: 0, stdout: [], stderr: [], asm: []});

interface Recorded {
    key: any;
    // The request as compile() adapted it to the compiler: the CacheKey it hands on to compilation and execution.
    adapted: any;
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
        realCompilerId?: string;
        answer?: CacheAnswer;
        setup?: (compiler: BaseCompiler, env: CompilationEnvironment) => void;
    } = {},
): Promise<Recorded> {
    const {env, compiler} = makeCompiler(opts.info, opts.realCompilerId);
    const answer = opts.answer ?? alwaysHit;
    const cacheGet = vi.spyOn(env, 'cacheGet').mockImplementation(async key => answer(key));
    const cachePut = vi.spyOn(env, 'cachePut').mockResolvedValue(undefined as any);
    const built = vi.spyOn(compiler, 'getCacheKey');
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
        adapted: built.mock.results[1]?.value,
        hash: BaseCache.hash(key),
        gets: cacheGet.mock.calls.map(call => call[0]),
        puts: cachePut.mock.calls.map(call => call[0]),
    };
}

const noRunNeeded = (compiler: BaseCompiler) =>
    vi.spyOn(compiler, 'doExecution').mockResolvedValue({code: 0, stdout: [], stderr: []} as any);

describe('Compilation cache key', () => {
    afterEach(() => vi.restoreAllMocks());

    // ------------------------------------------------------------------------------------------------
    // The compiler part of the key
    // ------------------------------------------------------------------------------------------------
    describe('the compiler part of the key', () => {
        it('holds only the compiler id', async () => {
            const {key} = await recordCacheKeys(compilerPaneBody());
            expectKey(key.compiler, {id: 'g161'});
        });

        it('does not change with anything else in the CompilerInfo', async () => {
            const base = await recordCacheKeys(compilerPaneBody());
            const otherVersion = await recordCacheKeys(compilerPaneBody(), {info: {version: 'gcc 16.1.1'}});
            const otherExe = await recordCacheKeys(compilerPaneBody(), {info: {exe: '/opt/other/g++'}});
            expect(otherVersion.hash).toEqual(base.hash);
            expect(otherExe.hash).toEqual(base.hash);
        });

        it('leaves the tools and the output parsers the full CompilerInfo, from the compiler rather than the key', async () => {
            const {compiler} = makeCompiler();
            const {key} = await recordCacheKeys(compilerPaneBody());
            const info = compiler.getCompilationInfo(key, {code: 0, stdout: [], stderr: []} as any);
            expect(info.compiler).toBe(compiler.getInfo());
            expect(info.options).toEqual(['-O2']);
        });
    });

    // ------------------------------------------------------------------------------------------------
    // What the compiler pane sends
    // ------------------------------------------------------------------------------------------------
    describe('for a compiler pane request', () => {
        it('with no extra panes open', async () => {
            const {key} = await recordCacheKeys(compilerPaneBody());
            expectKey(key, {
                compiler: KEY_COMPILER,
                source: SOURCE,
                options: ['-O2'],
                backendOptions: UI_COMPILER_OPTIONS,
                filters: UI_FILTERS,
                tools: [],
                libraries: [],
                files: [],
            });
        });

        it('with a library selected', async () => {
            const {key} = await recordCacheKeys(compilerPaneBody({libraries: [{id: 'boost', version: '190'}]}));
            expectKey(key, {
                compiler: KEY_COMPILER,
                source: SOURCE,
                options: ['-O2'],
                backendOptions: UI_COMPILER_OPTIONS,
                filters: UI_FILTERS,
                tools: [],
                libraries: [{id: 'boost', version: '190'}],
                files: [],
            });
        });

        it('with a Release library: buildType is in the key, so Debug and Release results never share it', async () => {
            const {key} = await recordCacheKeys(
                compilerPaneBody({libraries: [{id: 'benchmark', version: '190', buildType: 'Release'}]}),
            );
            expectKey(key, {
                compiler: KEY_COMPILER,
                source: SOURCE,
                options: ['-O2'],
                backendOptions: UI_COMPILER_OPTIONS,
                filters: UI_FILTERS,
                tools: [],
                libraries: [{id: 'benchmark', version: '190', buildType: 'Release'}],
                files: [],
            });
            const {key: debugKey} = await recordCacheKeys(
                compilerPaneBody({libraries: [{id: 'benchmark', version: '190'}]}),
            );
            expect(JSON.stringify(debugKey)).not.toEqual(JSON.stringify(key));
        });

        it('with a Release library and execution: the run key differs from Debug too', async () => {
            const runKeyFor = async (library: Record<string, string>) => {
                const body = compilerPaneBody({filters: {execute: true}, libraries: [library]});
                body.options.executeParameters = {args: [] as any, stdin: ''};
                const {gets} = await recordCacheKeys(body);
                expect(gets).toHaveLength(2);
                return gets[1];
            };
            const release = await runKeyFor({id: 'benchmark', version: '190', buildType: 'Release'});
            const debug = await runKeyFor({id: 'benchmark', version: '190'});
            expect(release).not.toEqual(debug);
            expect(JSON.parse(release as string).key.libraries).toEqual([
                {id: 'benchmark', version: '190', buildType: 'Release'},
            ]);
        });

        it('with a tool added, as it is looked up (see "with tools" for how it is stored)', async () => {
            const {key} = await recordCacheKeys(
                compilerPaneBody({tools: [{id: 'clangtidytrunk', args: [], stdin: ''}]}),
            );
            expectKey(key, {
                compiler: KEY_COMPILER,
                source: SOURCE,
                options: ['-O2'],
                backendOptions: UI_COMPILER_OPTIONS,
                filters: UI_FILTERS,
                tools: [{id: 'clangtidytrunk', args: [], stdin: ''}],
                libraries: [],
                files: [],
            });
        });

        it('with the AST pane open', async () => {
            const {key} = await recordCacheKeys(compilerPaneBody({compilerOptions: {produceAst: true}}));
            expectKey(key.backendOptions, {
                producePp: null,
                produceAst: true,
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
            });
        });

        it('with the opt remarks pane open', async () => {
            const {key} = await recordCacheKeys(compilerPaneBody({compilerOptions: {produceOptInfo: true}}));
            expectKey(key.backendOptions, {...UI_COMPILER_OPTIONS, produceOptInfo: true});
        });

        it('with the GCC dump pane open', async () => {
            const gccDump = {
                opened: true,
                treeDump: true,
                rtlDump: true,
                ipaDump: true,
                dumpFlags: {
                    gimpleFe: false,
                    address: false,
                    alias: false,
                    slim: false,
                    raw: false,
                    details: false,
                    stats: false,
                    blocks: false,
                    vops: false,
                    lineno: false,
                    uid: false,
                    all: false,
                },
                libraryFunctions: false,
            };
            // Kept even for a compiler that can't produce a GCC dump.
            const {key} = await recordCacheKeys(compilerPaneBody({compilerOptions: {produceGccDump: gccDump}}), {
                info: {supportsGccDump: false},
            });
            expectKey(key.backendOptions, {...UI_COMPILER_OPTIONS, produceGccDump: gccDump});
        });

        it('tells a pane that was opened and closed again from one that never was', async () => {
            // The frontend sends produceAst: undefined (so nothing) until the AST pane has been opened, then true,
            // then false once it's closed. The compiler's output is the same either side, the key is not.
            const neverOpened = await recordCacheKeys(compilerPaneBody());
            const closed = await recordCacheKeys(compilerPaneBody({compilerOptions: {produceAst: false}}));

            expect(neverOpened.key.backendOptions).not.toHaveProperty('produceAst');
            expect(closed.key.backendOptions.produceAst).toBe(false);
            expect(closed.hash).not.toEqual(neverOpened.hash);
        });

        it('keeps the order the frontend sends filters and compilerOptions in', async () => {
            const body = compilerPaneBody();
            body.options.filters = Object.fromEntries(Object.entries(UI_FILTERS).reverse()) as typeof UI_FILTERS;
            body.options.compilerOptions = Object.fromEntries(Object.entries(UI_COMPILER_OPTIONS).reverse());
            const reversed = await recordCacheKeys(body);

            expect(Object.keys(reversed.key.filters)).toEqual(Object.keys(UI_FILTERS).reverse());
            expect(Object.keys(reversed.key.backendOptions)).toEqual(Object.keys(UI_COMPILER_OPTIONS).reverse());
            expect(reversed.hash).not.toEqual((await recordCacheKeys(compilerPaneBody())).hash);
        });

        it('leaves out the fields that only steer the request: bypassCache, executeParameters, lang', async () => {
            const body: any = compilerPaneBody();
            body.bypassCache = BypassCache.Execution;
            body.lang = 'c';
            body.allowStoreCodeDebug = false;
            body.options.executeParameters = {args: ['x'], stdin: 'y'};
            const {key} = await recordCacheKeys(body);
            expectKey(key, {
                compiler: KEY_COMPILER,
                source: SOURCE,
                options: ['-O2'],
                backendOptions: UI_COMPILER_OPTIONS,
                filters: UI_FILTERS,
                tools: [],
                libraries: [],
                files: [],
            });
        });
    });

    // ------------------------------------------------------------------------------------------------
    // The same compile reached by different routes
    // ------------------------------------------------------------------------------------------------
    describe('across request routes', () => {
        it('is the same for an SQS message as for the HTTP request it was made from', async () => {
            const http = await recordCacheKeys(compilerPaneBody());
            const sqsMessage = {
                ...compilerPaneBody(),
                guid: 'abc',
                compilerId: 'g161',
                headers: {'content-type': 'application/json'},
                queryStringParameters: {},
                queueTimeMs: 5,
            };
            const sqs = await recordCacheKeys(sqsMessage, {query: sqsMessage.queryStringParameters});
            expectKey(sqs.key, JSON.parse(JSON.stringify(http.key)));
        });

        it('for /noscript/, whose form sets other backend options', async () => {
            const {key} = await recordCacheKeys(NOSCRIPT_FORM_BODY, {isJson: false});
            // Unticked boxes become false, and verboseDemangling, not being a default filter, is dropped.
            expectKey(key, {
                compiler: KEY_COMPILER,
                source: SOURCE,
                options: ['-O2'],
                backendOptions: {filterAnsi: true, skipAsm: false},
                filters: {...DEFAULT_FILTERS, libraryCode: true},
                tools: [],
                libraries: [],
                files: [],
            });
        });

        it('for the plain-text REST API', async () => {
            const {key} = await recordCacheKeys(SOURCE, {isJson: false, query: {options: '-O2'}});
            expectKey(key, {
                compiler: KEY_COMPILER,
                source: SOURCE,
                options: ['-O2'],
                backendOptions: {skipAsm: false, skipPopArgs: false, filterAnsi: false},
                filters: DEFAULT_FILTERS,
                tools: [],
                libraries: [],
                files: [],
            });
        });
    });

    // ------------------------------------------------------------------------------------------------
    // With a real CompilerInfo: the keys a dev instance computed
    // ------------------------------------------------------------------------------------------------
    describe('with a real CompilerInfo, matching what a dev instance computed', () => {
        it.each(Object.entries(REAL_COMPILER_INFO_HASH))('%s CompilerInfo hashes as it did there', (id, devHash) => {
            expect(BaseCache.hash(loadRealCompilerInfo(id) as any)).toBe(devHash);
        });

        it('the key holds only the id of the real compiler', async () => {
            const {key} = await recordCacheKeys(compilerPaneBody(), {realCompilerId: 'g161'});
            expectKey(key.compiler, {id: 'g161'});
        });

        it('ignores what used to make keys differ between instances: $order, tools, fullVersion', async () => {
            const base = await recordCacheKeys(compilerPaneBody(), {realCompilerId: 'g161'});
            const {env, compiler} = makeCompiler({}, 'g161');
            const info: any = compiler.getInfo();
            info.$order = -21;
            info.tools = Object.keys(info.tools);
            info.fullVersion = `${info.fullVersion} (rebuilt)`;
            const cacheGet = vi.spyOn(env, 'cacheGet').mockResolvedValue(alwaysHit(null));
            const parsed = CompileHandler.parseRequestReusable(true, {}, compilerPaneBody(), compiler);
            await compiler.compile(
                parsed.source,
                parsed.options,
                parsed.backendOptions,
                parsed.filters,
                parsed.bypassCache,
                parsed.tools,
                parsed.executeParameters,
                parsed.libraries,
                [],
            );
            expect(BaseCache.hash(cacheGet.mock.calls[0][0] as any)).toEqual(base.hash);
        });

        const realScenarios: [string, string, any, {isJson?: boolean; query?: any}, string][] = [
            [
                'compiler pane, no extra panes',
                'g161',
                compilerPaneBody(),
                {},
                '702296f4d606f556cf9fcad5485a25df9aa27b75e1a56f153dfd90ee7abdf5ec',
            ],
            [
                'compiler pane, "Execute the code" ticked',
                'g161',
                compilerPaneBody({filters: {execute: true}}),
                {},
                'ab2d0f1c3aec29b8e4450fdc2976815b66f3897a92bc222b5c627670b502e400',
            ],
            [
                'compiler pane, link to binary',
                'g161',
                compilerPaneBody({filters: {binary: true}}),
                {},
                '64a51f778dd49bef7f35d096877551a725529f26c0a9ea4e05abf246947985c7',
            ],
            [
                'compiler pane, -E',
                'g161',
                compilerPaneBody({userArguments: '-E'}),
                {},
                '9f9451b2793b0b1298d3e44f386dbef5f88268998e45ee9f4d4c16d432bfa28d',
            ],
            [
                'compiler pane, boost 1.90 selected',
                'g161',
                compilerPaneBody({libraries: [{id: 'boost', version: '190'}]}),
                {},
                'f23189ff996613bb559fcf45fbb0a062c6a09a31781a3aa8bca73d84d215dfb2',
            ],
            [
                'compiler pane, -lfmt (not autodetected)',
                'g161',
                compilerPaneBody({userArguments: '-O2 -lfmt'}),
                {},
                '82859c56709b75a3070ec095cc7ea5d240191d9191e93e1784c39ed90fb74433',
            ],
            [
                'compiler pane, clang-tidy added (looked up and stored under the same key)',
                'g161',
                compilerPaneBody({tools: [{id: 'clangtidytrunk', args: [], stdin: ''}]}),
                {},
                '6773b48257a8c278e99307f880f41045412ffcd88719aa74f968988dfadd3910',
            ],
            [
                'compiler pane, GCC dump pane open',
                'g161',
                compilerPaneBody({
                    compilerOptions: {
                        produceGccDump: {
                            opened: true,
                            treeDump: true,
                            rtlDump: true,
                            ipaDump: true,
                            dumpFlags: {
                                gimpleFe: false,
                                address: false,
                                alias: false,
                                slim: false,
                                raw: false,
                                details: false,
                                stats: false,
                                blocks: false,
                                vops: false,
                                lineno: false,
                                uid: false,
                                all: false,
                            },
                            libraryFunctions: false,
                        },
                    },
                }),
                {},
                '37d74eaefe6a682655bca8658a8b46cbba5d3cbcd304a19de47bf8cfa68f7008',
            ],
            [
                'executor pane',
                'g161',
                executorPaneBody(),
                {},
                '9510d0c4620e4d6eca5aa288cfff25fb15e659eba88550a786dd1b5917323a19',
            ],
            [
                '/noscript/ form',
                'g161',
                NOSCRIPT_FORM_BODY,
                {isJson: false},
                '5d3dbc4c4e7bbf3743df2b82d0fe2a1b664eeaf7849273c15d66dd4fe4d87a57',
            ],
            [
                'plain-text REST API',
                'g161',
                SOURCE,
                {isJson: false, query: {options: '-O2'}},
                '084000d4592bdb053a2a4b1a82afa6c76379ea92c2ad745d14aed796186cd7b8',
            ],
            [
                'compiler pane, no extra panes',
                'clang2110',
                compilerPaneBody(),
                {},
                '646e25d7753dc67c4bd20e76658a6d1fd771699dbe95bd7e6782cb4b16314728',
            ],
            [
                'compiler pane, AST pane open',
                'clang2110',
                compilerPaneBody({compilerOptions: {produceAst: true}}),
                {},
                'a41eb74f79c1791aba4ebad240a792e5204d40aa1a754d7ced8bb90b026edf42',
            ],
            [
                'compiler pane, AST pane opened and closed again',
                'clang2110',
                compilerPaneBody({compilerOptions: {produceAst: false}}),
                {},
                'a0fc73ad295d80fdba0e05cb587e4ce49fb08dd2632ce1b6b175fbcb671d2563',
            ],
            [
                'compiler pane, opt remarks pane open',
                'clang2110',
                compilerPaneBody({compilerOptions: {produceOptInfo: true}}),
                {},
                'e40c651051814b85bffd0cfa42389be1ed007d5a82521350caebd6c499ef8574',
            ],
        ];

        it.each(realScenarios)('%s on %s', async (_, compilerId, body, route, devHash) => {
            const {hash} = await recordCacheKeys(body, {...route, realCompilerId: compilerId});
            expect(hash).toBe(devHash);
        });

        it('the run keys for compile-and-execute and the executor pane', async () => {
            const compileAndRun = await recordCacheKeys(compilerPaneBody({filters: {execute: true}}), {
                realCompilerId: 'g161',
            });
            expect(BaseCache.hash(compileAndRun.gets[1] as any)).toBe(
                '8df165d1b7ebbaa6c78bedadc850c0a0f3c00746fe75965885f7589e0a9ad0e8',
            );

            const executor = await recordCacheKeys(executorPaneBody(), {
                realCompilerId: 'g161',
                answer: () => null,
                setup: noRunNeeded,
            });
            expect(BaseCache.hash(executor.gets[1] as any)).toBe(
                '7de58a551f7bd3ca50eb543d81e91395826fac1d7ad97328f93db45ea710ea52',
            );

            const executorWithInput = await recordCacheKeys(executorPaneBody('a b', 'in'), {
                realCompilerId: 'g161',
                answer: () => null,
                setup: noRunNeeded,
            });
            expect(BaseCache.hash(executorWithInput.gets[1] as any)).toBe(
                '68258d5ad9db4f119835ddbf6c4f0526c3521fa52b0cea5bbceec72d5e89500e',
            );
        });
    });

    // ------------------------------------------------------------------------------------------------
    // Changes compile() makes before it builds the key
    // ------------------------------------------------------------------------------------------------
    describe('adapting the request to the compiler, which happens after the key is built', () => {
        it('fills in the default filters, but only in what is compiled', async () => {
            const {key, adapted} = await recordCacheKeys(compilerPaneBody());
            expectKey(key.filters, UI_FILTERS);
            expect(adapted.filters).toMatchObject({optOutput: false});
        });

        it("doesn't let a compiler's own default filters reach the key", async () => {
            const base = await recordCacheKeys(compilerPaneBody());
            const ownDefaults = await recordCacheKeys(compilerPaneBody(), {
                setup: compiler =>
                    vi.spyOn(compiler, 'getDefaultFilters').mockReturnValue({...DEFAULT_FILTERS, optOutput: true}),
            });
            expect(ownDefaults.hash).toEqual(base.hash);
            expect(ownDefaults.adapted.filters.optOutput).toBe(true);
        });

        it('turns every filter off for -E on C++', async () => {
            const {key, adapted} = await recordCacheKeys(compilerPaneBody({userArguments: '-E'}));
            expect(key.options).toEqual(['-E']);
            expectKey(key.filters, UI_FILTERS);
            expect(Object.values(adapted.filters).every(value => value === false)).toBe(true);
        });

        it('drops binary for a compiler that cannot produce one', async () => {
            const {key, adapted} = await recordCacheKeys(compilerPaneBody({filters: {binary: true}}), {
                info: {supportsBinary: false},
            });
            expect(key.filters.binary).toBe(true);
            expect(adapted.filters).not.toHaveProperty('binary');
        });

        it('stops masking filenames when there are extra files', async () => {
            const {key, adapted} = await recordCacheKeys(compilerPaneBody(), {
                files: [{filename: 'other.h', contents: ''}],
            });
            expectKey(key.filters, UI_FILTERS);
            expectKey(key.files, [{filename: 'other.h', contents: ''}]);
            expect(adapted.filters.dontMaskFilenames).toBe(true);
        });

        it('moves libraries it detects from -l options, without changing the key', async () => {
            const {key, adapted} = await recordCacheKeys(compilerPaneBody({userArguments: '-O2 -lfmt'}), {
                setup: compiler =>
                    vi.spyOn(compiler, 'tryAutodetectLibraries').mockImplementation(libsAndOptions => {
                        libsAndOptions.options = ['-O2'];
                        libsAndOptions.libraries = [{id: 'fmt', version: '10'}];
                        return true;
                    }),
            });
            expectKey(key.options, ['-O2', '-lfmt']);
            expectKey(key.libraries, []);
            expectKey(adapted.options, ['-O2']);
            expectKey(adapted.libraries, [{id: 'fmt', version: '10'}]);
        });

        it('keeps the background colour the request sent, and tells only a compiler that colours for it', async () => {
            const dark = () => compilerPaneBody({compilerOptions: {darkBackgroundColours: true}});
            const ignoring = await recordCacheKeys(dark());
            const colouring = await recordCacheKeys(dark(), {info: {supportsDarkBackgroundColours: true}});

            expect(ignoring.key.backendOptions.darkBackgroundColours).toBe(true);
            expect(ignoring.hash).toEqual(colouring.hash);
            expect(ignoring.adapted.backendOptions).not.toHaveProperty('darkBackgroundColours');
            expect(colouring.adapted.backendOptions.darkBackgroundColours).toBe(true);
        });
    });

    // ------------------------------------------------------------------------------------------------
    // Execution
    // ------------------------------------------------------------------------------------------------
    describe('with execution', () => {
        it('with "Execute the code" ticked in the compiler pane: execute is part of the compile key', async () => {
            const {key} = await recordCacheKeys(compilerPaneBody({filters: {execute: true}}));
            expectKey(key, {
                compiler: KEY_COMPILER,
                source: SOURCE,
                options: ['-O2'],
                backendOptions: UI_COMPILER_OPTIONS,
                filters: {...UI_FILTERS, execute: true},
                tools: [],
                libraries: [],
                files: [],
            });
        });

        it('the run is looked up under a JSON string of the adapted request and the execution parameters', async () => {
            const body = compilerPaneBody({filters: {execute: true}});
            body.options.executeParameters = {args: ['a', 'b'] as any, stdin: 'in'};
            const {adapted, gets} = await recordCacheKeys(body);

            expect(gets).toHaveLength(2);
            expect(typeof gets[1]).toBe('string');
            expectKey(JSON.parse(gets[1] as string), {
                key: JSON.parse(JSON.stringify(adapted)),
                executeParameters: {args: ['a', 'b'], stdin: 'in', ldPath: [], env: {}, runtimeTools: []},
            });
        });

        it('for the executor pane: a compile key of its own, looked up but never stored', async () => {
            const {key, adapted, gets, puts} = await recordCacheKeys(executorPaneBody('a b', 'in'), {
                answer: () => null,
                setup: noRunNeeded,
            });

            const executorKey = {
                compiler: KEY_COMPILER,
                source: SOURCE,
                options: ['-O2'],
                backendOptions: {executorRequest: true, skipAsm: true, overrides: []},
                filters: {execute: true},
                tools: [],
                libraries: [],
                files: [],
            };
            expectKey(key, executorKey);
            expectKey(JSON.parse(gets[1] as string), {
                key: JSON.parse(JSON.stringify(adapted)),
                executeParameters: {args: ['a', 'b'], stdin: 'in', ldPath: [], env: {}, runtimeTools: []},
            });
            expect(puts).toEqual([gets[1]]);
        });

        it('for the executor pane: arguments and stdin change the run key, not the compile key', async () => {
            const plain = await recordCacheKeys(executorPaneBody(), {answer: () => null, setup: noRunNeeded});
            const withInput = await recordCacheKeys(executorPaneBody('a b', 'in'), {
                answer: () => null,
                setup: noRunNeeded,
            });

            expect(withInput.hash).toEqual(plain.hash);
            expect(withInput.gets[1]).not.toEqual(plain.gets[1]);
        });
    });

    // ------------------------------------------------------------------------------------------------
    // Tools
    // ------------------------------------------------------------------------------------------------
    describe('with tools', () => {
        it('leaves the tool arguments in the key as they were looked up', async () => {
            // compile() passes key.tools[i].args straight to runTool, so a tool adding to that array would change the
            // key between the lookup and the store, and the result would be stored where it is never found.
            const tool = new BaseTool(
                {id: 'clangtidytrunk', exe: '/bin/true', options: []} as unknown as ToolInfo,
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

            const {key} = await recordCacheKeys(
                compilerPaneBody({tools: [{id: 'clangtidytrunk', args: [], stdin: ''}]}),
            );
            await tool.runTool(
                {compiler: {lang: 'c++'}} as unknown as CompilationInfo,
                '/tmp/compiler-explorer-compilerP0VA1h/example.cpp',
                key.tools[0].args,
            );
            expectKey(key.tools, [{id: 'clangtidytrunk', args: [], stdin: ''}]);
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
            // The tree pane adds these after the compiler pane has built its options.
            body.options.compilerOptions = {...body.options.compilerOptions, cmakeArgs: '', customOutputFilename: ''};
            return body;
        }

        async function recordProjectBuild(body: any, realCompilerId?: string) {
            const {env, compiler} = makeCompiler({}, realCompilerId);
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
            const {execute, ...filtersWithoutExecute} = UI_FILTERS;
            expect(execute).toBe(false);
            // A project build still fills in the default filters before its key is built: they follow the request's
            // own, and it always disassembles a binary.
            expectKey(key, {
                source: CMAKELISTS,
                options: ['-O2'],
                backendOptions: {...UI_COMPILER_OPTIONS, cmakeArgs: '', customOutputFilename: ''},
                filters: {...filtersWithoutExecute, binary: true, optOutput: false, dontMaskFilenames: true},
                libraries: [],
                compiler: KEY_COMPILER,
                files: CMAKE_FILES,
                api: 'cmake',
            });
        });

        it('is the same whether or not the build is to be run', async () => {
            expect((await recordProjectBuild(cmakeBody(true))).hash).toEqual(
                (await recordProjectBuild(cmakeBody(false))).hash,
            );
        });

        // The hashes a dev instance stored these builds' results under (its compilation cache put). The executable
        // package it looked for first is the same hash with '_exec' appended.
        it.each([
            [
                'a compiler pane build',
                cmakeBody(false),
                '455bee88297e87c02f9333c0686666060fb95ed8ce267bbefc5214f73538fd31',
            ],
            [
                'a build with CMake arguments',
                (() => {
                    const body = cmakeBody(false);
                    body.options.compilerOptions.cmakeArgs = '-DFOO=1';
                    return body;
                })(),
                'c2bdfbc721cfd7a75b87a0556e73f815d439ca54cac7ced062cdcc62decb73d3',
            ],
            [
                'an executor pane build',
                {
                    source: CMAKELISTS,
                    files: CMAKE_FILES,
                    compiler: 'g161',
                    options: {
                        userArguments: '-O2',
                        executeParameters: {args: '', stdin: '', runtimeTools: []},
                        compilerOptions: {
                            executorRequest: true,
                            skipAsm: true,
                            overrides: [],
                            cmakeArgs: '',
                            customOutputFilename: '',
                        },
                        filters: {execute: true},
                        tools: [],
                        libraries: [],
                    },
                    lang: 'c++',
                    allowStoreCodeDebug: true,
                },
                'd2ead1220b3b2ad19d34ad8bf87593df54449d589f56b0aa9172191d7d385ab6',
            ],
        ])('with the real g161 CompilerInfo, %s matches the dev instance', async (_, body, devHash) => {
            const {hash} = await recordProjectBuild(body, 'g161');
            expect(hash).toBe(`${devHash}_exec`);
        });
    });
});
