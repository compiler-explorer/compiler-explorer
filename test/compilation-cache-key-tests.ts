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
import {ClangTidyTool} from '../lib/tooling/clang-tidy-tool.js';
import {BypassCache, CompilationInfo, FiledataPair} from '../types/compilation/compilation.interfaces.js';
import {CompilerInfo} from '../types/compiler.interfaces.js';
import {ToolInfo} from '../types/tool.interfaces.js';
import {makeCompilationEnvironment, makeFakeCompilerInfo} from './utils.js';

const languages = {'c++': {id: 'c++'}} as const;

const SOURCE = '#include <cstdio>\nint main() { std::puts("hi"); return 0; }\n';

// Stands in for the CompilerInfo; see 'the compiler part of the key' for what a real one holds.
const COMPILER_INFO = {id: 'g161', exe: '/dev/null', lang: 'c++' as const, options: '', supportsBinary: true};

// The BaseCompiler constructor fills in defaults on the CompilerInfo it is given, and the key holds the result.
const CONSTRUCTOR_DEFAULTS = {
    supportsIntel: false,
    optArg: '',
    supportsOptOutput: false,
    supportsVerboseAsm: false,
    disabledFilters: [],
    instructionSet: 'amd64',
};
const KEY_COMPILER_INFO = {...COMPILER_INFO, ...CONSTRUCTOR_DEFAULTS};

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

// The same, as it ends up in the key: darkBackgroundColours goes for a compiler that doesn't colour by background.
const KEY_BACKEND_OPTIONS = {
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

// The same, as they end up in the key: in the order of getDefaultFilters(), with optOutput added.
const KEY_FILTERS = {
    binary: false,
    execute: false,
    demangle: true,
    intel: true,
    commentOnly: true,
    directives: true,
    labels: true,
    optOutput: false,
    libraryCode: true,
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

// Checks the key both ways: deep equality for a readable diff, then the exact JSON, which is what gets hashed.
function expectKey(actual: unknown, expected: unknown) {
    expect(JSON.parse(JSON.stringify(actual))).toEqual(expected);
    expect(JSON.stringify(actual)).toBe(JSON.stringify(expected));
}

// The CompilerInfo a dev instance (production c++ config, --language c++ c rust) put into its cache keys, as logged by
// logCompilationCacheAccesses. Its hash is in REAL_COMPILER_INFO_HASH.
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
        realCompilerId?: string;
        answer?: CacheAnswer;
        setup?: (compiler: BaseCompiler, env: CompilationEnvironment) => void;
    } = {},
): Promise<Recorded> {
    const {env, compiler} = makeCompiler(opts.info, opts.realCompilerId);
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

const noRunNeeded = (compiler: BaseCompiler) =>
    vi.spyOn(compiler, 'doExecution').mockResolvedValue({code: 0, stdout: [], stderr: []} as any);

describe('Compilation cache key', () => {
    afterEach(() => vi.restoreAllMocks());

    // ------------------------------------------------------------------------------------------------
    // The compiler part of the key
    // ------------------------------------------------------------------------------------------------
    describe('the compiler part of the key', () => {
        it('is the very CompilerInfo object the compiler holds', async () => {
            const {compiler, env} = makeCompiler();
            const cacheGet = vi.spyOn(env, 'cacheGet').mockResolvedValue(alwaysHit(null));
            await compiler.compile(SOURCE, ['-O2'], {}, {} as any, BypassCache.None, [], {} as any, [], []);
            expect((cacheGet.mock.calls[0][0] as any).compiler).toBe(compiler.getInfo());
        });

        it('holds each configured tool as the serialised tool object, not just its id', async () => {
            // In production this is most of the difference between the CompilerInfo hashed into the key and the one
            // /api/compilers?fields=all returns: the API lists tool ids, the key carries the tool objects, with their
            // exe paths, configured options and sandbox type. (g161's real CompilerInfo serialises to 14.5KB, 4.3KB of
            // it tools; its other 76 fields match the API, in the same order. See the real-CompilerInfo tests below.)
            const toolInfo = {
                id: 'clangtidytrunk',
                name: 'clang-tidy (trunk)',
                type: 'independent',
                exe: '/opt/compiler-explorer/clang-trunk/bin/clang-tidy',
                exclude: ['cl19'],
                options: ['--gcc-toolchain=/opt/compiler-explorer/gcc-snapshot'],
                stdinHint: 'disabled',
                compilerLanguage: 'c++',
            } as unknown as ToolInfo;
            const toolEnv = {ceProps: () => undefined, compilerProps: () => undefined} as unknown as ToolEnv;
            const tools = {clangtidytrunk: new ClangTidyTool(toolInfo, toolEnv)} as any;

            const {key} = await recordCacheKeys(compilerPaneBody(), {info: {tools}});
            expectKey(key.compiler, {
                ...COMPILER_INFO,
                tools: {
                    clangtidytrunk: {
                        tool: toolInfo,
                        env: {},
                        addOptionsToToolArgs: false,
                        id: 'clangtidytrunk',
                        type: 'independent',
                        sandboxType: 'none',
                    },
                },
                ...CONSTRUCTOR_DEFAULTS,
            });
        });

        it('changes with any field of the CompilerInfo', async () => {
            const base = await recordCacheKeys(compilerPaneBody());
            const otherVersion = await recordCacheKeys(compilerPaneBody(), {info: {version: 'gcc 16.1.1'}});
            expect(otherVersion.hash).not.toEqual(base.hash);
        });
    });

    // ------------------------------------------------------------------------------------------------
    // What the compiler pane sends
    // ------------------------------------------------------------------------------------------------
    describe('for a compiler pane request', () => {
        it('with no extra panes open', async () => {
            const {key} = await recordCacheKeys(compilerPaneBody());
            expectKey(key, {
                compiler: KEY_COMPILER_INFO,
                source: SOURCE,
                options: ['-O2'],
                backendOptions: KEY_BACKEND_OPTIONS,
                filters: KEY_FILTERS,
                tools: [],
                libraries: [],
                files: [],
            });
        });

        it('with a library selected', async () => {
            const {key} = await recordCacheKeys(compilerPaneBody({libraries: [{id: 'boost', version: '190'}]}));
            expectKey(key, {
                compiler: KEY_COMPILER_INFO,
                source: SOURCE,
                options: ['-O2'],
                backendOptions: KEY_BACKEND_OPTIONS,
                filters: KEY_FILTERS,
                tools: [],
                libraries: [{id: 'boost', version: '190'}],
                files: [],
            });
        });

        it('with a tool added, as it is looked up (see "with tools" for how it is stored)', async () => {
            const {key} = await recordCacheKeys(
                compilerPaneBody({tools: [{id: 'clangtidytrunk', args: [], stdin: ''}]}),
            );
            expectKey(key, {
                compiler: KEY_COMPILER_INFO,
                source: SOURCE,
                options: ['-O2'],
                backendOptions: KEY_BACKEND_OPTIONS,
                filters: KEY_FILTERS,
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
            expectKey(key.backendOptions, {...KEY_BACKEND_OPTIONS, produceOptInfo: true});
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
            expectKey(key.backendOptions, {...KEY_BACKEND_OPTIONS, produceGccDump: gccDump});
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

        it('does not depend on the order the frontend sends filters in', async () => {
            const body = compilerPaneBody();
            body.options.filters = Object.fromEntries(Object.entries(UI_FILTERS).reverse()) as typeof UI_FILTERS;
            const {key} = await recordCacheKeys(body);
            expectKey(key.filters, KEY_FILTERS);
        });

        it('does depend on the order the frontend sends compilerOptions in', async () => {
            const body = compilerPaneBody();
            body.options.compilerOptions = Object.fromEntries(Object.entries(UI_COMPILER_OPTIONS).reverse());
            const reversed = await recordCacheKeys(body);

            expect(Object.keys(reversed.key.backendOptions)).toEqual(Object.keys(KEY_BACKEND_OPTIONS).reverse());
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
                compiler: KEY_COMPILER_INFO,
                source: SOURCE,
                options: ['-O2'],
                backendOptions: KEY_BACKEND_OPTIONS,
                filters: KEY_FILTERS,
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
                compiler: KEY_COMPILER_INFO,
                source: SOURCE,
                options: ['-O2'],
                backendOptions: {filterAnsi: true, skipAsm: false},
                filters: KEY_FILTERS,
                tools: [],
                libraries: [],
                files: [],
            });
        });

        it('for the plain-text REST API', async () => {
            const {key} = await recordCacheKeys(SOURCE, {isJson: false, query: {options: '-O2'}});
            expectKey(key, {
                compiler: KEY_COMPILER_INFO,
                source: SOURCE,
                options: ['-O2'],
                backendOptions: {skipAsm: false, skipPopArgs: false, filterAnsi: false},
                filters: {...KEY_FILTERS, libraryCode: false},
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

        it('the key is the CompilerInfo object itself, all of it', async () => {
            const {key} = await recordCacheKeys(compilerPaneBody(), {realCompilerId: 'g161'});
            expectKey(key.compiler, loadRealCompilerInfo('g161'));
            expect(BaseCache.hash(key.compiler)).toBe(REAL_COMPILER_INFO_HASH.g161);
        });

        it('carries settings the API does not show: tools are full objects, with exe paths and sandbox type', () => {
            const info: any = loadRealCompilerInfo('g161');
            expect(Object.keys(info.tools)).toContain('clangtidytrunk');
            expect(info.tools.clangtidytrunk).toEqual({
                tool: {
                    id: 'clangtidytrunk',
                    name: 'clang-tidy (trunk)',
                    type: 'independent',
                    exe: '/opt/compiler-explorer/clang-trunk/bin/clang-tidy',
                    exclude: ['cl19', 'cl_new'],
                    options: ['--gcc-toolchain=/opt/compiler-explorer/gcc-snapshot'],
                    stdinHint: 'disabled',
                    compilerLanguage: 'c++',
                },
                env: {},
                addOptionsToToolArgs: false,
                id: 'clangtidytrunk',
                type: 'independent',
                sandboxType: 'nsjail',
            });
        });

        const realScenarios: [string, string, any, {isJson?: boolean; query?: any}, string][] = [
            [
                'compiler pane, no extra panes',
                'g161',
                compilerPaneBody(),
                {},
                'f311171eb9e040e3fe3ea6760c5de69634a3e25c09e86dac3facb2f0641245c6',
            ],
            [
                'compiler pane, "Execute the code" ticked',
                'g161',
                compilerPaneBody({filters: {execute: true}}),
                {},
                '14c9d1509ca7571cad9f4c618debf068b9d96c72a2462c62a6f75823601d708f',
            ],
            [
                'compiler pane, link to binary',
                'g161',
                compilerPaneBody({filters: {binary: true}}),
                {},
                '2f03201c3f4074adfa8f219533749ba55d9006de7d64c0511dc4b046040da36d',
            ],
            [
                'compiler pane, -E',
                'g161',
                compilerPaneBody({userArguments: '-E'}),
                {},
                '5408347acddfe6f24fe46c601dc1acf49e1d02ab2afc19b0a58bd2ad29df663e',
            ],
            [
                'compiler pane, boost 1.90 selected',
                'g161',
                compilerPaneBody({libraries: [{id: 'boost', version: '190'}]}),
                {},
                '349533c11b9a213e8fc616b597f427e8e31cb7f049c1a4a7a08fe55159351978',
            ],
            [
                'compiler pane, -lfmt (not autodetected)',
                'g161',
                compilerPaneBody({userArguments: '-O2 -lfmt'}),
                {},
                '37021a805523adc75f33491d77edd3e4c1a7ad888d7684866cfa35340d6c6d2d',
            ],
            [
                'compiler pane, clang-tidy added (as looked up)',
                'g161',
                compilerPaneBody({tools: [{id: 'clangtidytrunk', args: [], stdin: ''}]}),
                {},
                '3f18e6ba777600c8b4ddd63a2940c356f6e53f6b5b86833e79189b1f9ce7f16e',
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
                'db1c37fc412a4511ab1fdf9282ff6b3543580dd93ace57745503205ba281f8eb',
            ],
            [
                'executor pane',
                'g161',
                executorPaneBody(),
                {},
                'e2c0e3d02682216ee297b63086d6dbfdfbd3ff9ffd51f68baeefe9d4714e3258',
            ],
            [
                '/noscript/ form',
                'g161',
                NOSCRIPT_FORM_BODY,
                {isJson: false},
                'e7a818f790c59f098f980c652fc676e3f68125c4731dcb63c1579756b6d225f7',
            ],
            [
                'plain-text REST API',
                'g161',
                SOURCE,
                {isJson: false, query: {options: '-O2'}},
                '1fc9ccbf058af56f451e9ce91b51f60959b24d0721575988afccdc7947cc2673',
            ],
            [
                'compiler pane, no extra panes',
                'clang2110',
                compilerPaneBody(),
                {},
                'd90e1c4ce7e26024538bd8ee3f99910af1d77259c21e74ba4ea9c43c47ece1a0',
            ],
            [
                'compiler pane, AST pane open',
                'clang2110',
                compilerPaneBody({compilerOptions: {produceAst: true}}),
                {},
                '3a889bb7c3817053048ccad95aba6d195348751522d600079feeb2ae2e5f696e',
            ],
            [
                'compiler pane, AST pane opened and closed again',
                'clang2110',
                compilerPaneBody({compilerOptions: {produceAst: false}}),
                {},
                'ee8a5e1adb80a05f1d222e4b7e03c19b3a3fb1f9826e00664920811c740db8fe',
            ],
            [
                'compiler pane, opt remarks pane open',
                'clang2110',
                compilerPaneBody({compilerOptions: {produceOptInfo: true}}),
                {},
                '01f8feafd4df201f634365e6b4ddc3861f78e9ef827fed8458f341c85d6c0234',
            ],
        ];

        it.each(realScenarios)('%s on %s', async (_, compilerId, body, route, devHash) => {
            const {hash} = await recordCacheKeys(body, {
                ...route,
                realCompilerId: compilerId,
                answer: () => null,
                setup: noRunNeeded,
            });
            expect(hash).toBe(devHash);
        });

        it('the run keys for compile-and-execute and the executor pane', async () => {
            const compileAndRun = await recordCacheKeys(compilerPaneBody({filters: {execute: true}}), {
                realCompilerId: 'g161',
            });
            expect(BaseCache.hash(compileAndRun.gets[1] as any)).toBe(
                '05ba130811c8ec7ea17a68838cec7340925226e386b1558246a5b1149a4c71b1',
            );

            const executor = await recordCacheKeys(executorPaneBody(), {
                realCompilerId: 'g161',
                answer: () => null,
                setup: noRunNeeded,
            });
            expect(BaseCache.hash(executor.gets[1] as any)).toBe(
                '68070f984694e4ae30222803d7ca0fdbdc7269ec100e6647649e8cd5dfc1e930',
            );

            const executorWithInput = await recordCacheKeys(executorPaneBody('a b', 'in'), {
                realCompilerId: 'g161',
                answer: () => null,
                setup: noRunNeeded,
            });
            expect(BaseCache.hash(executorWithInput.gets[1] as any)).toBe(
                '5686363c90875c6b9cb4d9e4c2f29db57d1328b7d02599152d0371d1c7fb1ecf',
            );
        });

        it('the key a clang-tidy compile was stored under, with the temp path in it', async () => {
            const {key} = await recordCacheKeys(
                compilerPaneBody({tools: [{id: 'clangtidytrunk', args: [], stdin: ''}]}),
                {realCompilerId: 'g161'},
            );
            key.tools[0].args.push('/tmp/compiler-explorer-compilerP0VA1h/example.cpp');
            expect(BaseCache.hash(key)).toBe('9fec0bc2f08c89df1f19afa5d8ba6888b71f2f44a12a0b7de7a05e079e9a49b8');
        });
    });

    // ------------------------------------------------------------------------------------------------
    // Changes compile() makes before it builds the key
    // ------------------------------------------------------------------------------------------------
    describe('normalisation before the key is built', () => {
        it('turns every filter off for -E on C++', async () => {
            const {key} = await recordCacheKeys(compilerPaneBody({userArguments: '-E'}));
            expect(key.options).toEqual(['-E']);
            expectKey(key.filters, {
                binary: false,
                execute: false,
                demangle: false,
                intel: false,
                commentOnly: false,
                directives: false,
                labels: false,
                optOutput: false,
                libraryCode: false,
                trim: false,
                binaryObject: false,
                debugCalls: false,
            });
        });

        it('drops binary for a compiler that cannot produce one', async () => {
            const {key} = await recordCacheKeys(compilerPaneBody({filters: {binary: true}}), {
                info: {supportsBinary: false},
            });
            const {binary, ...withoutBinary} = KEY_FILTERS;
            expect(binary).toBe(false);
            expectKey(key.filters, withoutBinary);
        });

        it('stops masking filenames when there are extra files', async () => {
            const {key} = await recordCacheKeys(compilerPaneBody(), {files: [{filename: 'other.h', contents: ''}]});
            expectKey(key.filters, {...KEY_FILTERS, dontMaskFilenames: true});
            expectKey(key.files, [{filename: 'other.h', contents: ''}]);
        });
    });

    // ------------------------------------------------------------------------------------------------
    // Execution
    // ------------------------------------------------------------------------------------------------
    describe('with execution', () => {
        it('with "Execute the code" ticked in the compiler pane: execute is part of the compile key', async () => {
            const {key} = await recordCacheKeys(compilerPaneBody({filters: {execute: true}}));
            expectKey(key, {
                compiler: KEY_COMPILER_INFO,
                source: SOURCE,
                options: ['-O2'],
                backendOptions: KEY_BACKEND_OPTIONS,
                filters: {...KEY_FILTERS, execute: true},
                tools: [],
                libraries: [],
                files: [],
            });
        });

        it('the run is looked up under a JSON string of the compile key and the execution parameters', async () => {
            const body = compilerPaneBody({filters: {execute: true}});
            body.options.executeParameters = {args: ['a', 'b'] as any, stdin: 'in'};
            const {key, gets} = await recordCacheKeys(body);

            expect(gets).toHaveLength(2);
            expect(typeof gets[1]).toBe('string');
            expectKey(JSON.parse(gets[1] as string), {
                key: JSON.parse(JSON.stringify(key)),
                executeParameters: {args: ['a', 'b'], stdin: 'in', ldPath: [], env: {}, runtimeTools: []},
            });
        });

        it('for the executor pane: a compile key of its own, looked up but never stored', async () => {
            const {key, gets, puts} = await recordCacheKeys(executorPaneBody('a b', 'in'), {
                answer: () => null,
                setup: noRunNeeded,
            });

            // {execute: true} merged over the defaults, so libraryCode is false where the compiler pane sends true.
            const executorKey = {
                compiler: KEY_COMPILER_INFO,
                source: SOURCE,
                options: ['-O2'],
                backendOptions: {executorRequest: true, skipAsm: true, overrides: []},
                filters: {...KEY_FILTERS, execute: true, libraryCode: false},
                tools: [],
                libraries: [],
                files: [],
            };
            expectKey(key, executorKey);
            expectKey(JSON.parse(gets[1] as string), {
                key: executorKey,
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
        it("stores under a key with the temp source path appended to the request's tool arguments", async () => {
            // compile() passes key.tools[i].args straight to runTool. A tool that doesn't prepend its configured
            // options (clang-tidy, clang-query, PVS-Studio, ...) pushes the temporary source path onto that array,
            // so afterCompilation stores the result under a key naming a temp directory, which is never looked up.
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
            expectKey(key.tools, [{id: 'clangtidytrunk', args: [], stdin: ''}]);

            await tool.runTool(
                {compiler: {lang: 'c++'}} as unknown as CompilationInfo,
                '/tmp/compiler-explorer-compilerP0VA1h/example.cpp',
                key.tools[0].args,
            );
            expectKey(key.tools, [
                {id: 'clangtidytrunk', args: ['/tmp/compiler-explorer-compilerP0VA1h/example.cpp'], stdin: ''},
            ]);
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
            const {execute, ...filtersWithoutExecute} = KEY_FILTERS;
            expect(execute).toBe(false);
            expectKey(key, {
                source: CMAKELISTS,
                options: ['-O2'],
                backendOptions: {...KEY_BACKEND_OPTIONS, cmakeArgs: '', customOutputFilename: ''},
                filters: {...filtersWithoutExecute, binary: true, dontMaskFilenames: true},
                libraries: [],
                compiler: KEY_COMPILER_INFO,
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
                '9cf2729ed4e102d4cce3b0ca5bc8d0b72d6092312f3e95d31dbb0f2b135d9e3c',
            ],
            [
                'a build with CMake arguments',
                (() => {
                    const body = cmakeBody(false);
                    body.options.compilerOptions.cmakeArgs = '-DFOO=1';
                    return body;
                })(),
                'f75ea9450d9429725f6db4ae63c3cafc93496d466f69a534db3882ecbf21ca8c',
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
                            darkBackgroundColours: true,
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
                '37e11449beaadaa4260ae65b6b705788d1e0a6ce271eb1d605e1830a2f1777d0',
            ],
        ])('with the real g161 CompilerInfo, %s matches the dev instance', async (_, body, devHash) => {
            const {hash} = await recordProjectBuild(body, 'g161');
            expect(hash).toBe(`${devHash}_exec`);
        });
    });
});
