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
 * Pins down how the compilation cache key is built, ahead of ce-router#17 (checking the cache in the router before
 * queueing). A router-side key only helps if it matches the worker's byte for byte, so these tests record the
 * behaviour, quirks included. The key names the compiler by id only; anything else about the compiler that turns out
 * to change the output is to be added back to it one field at a time.
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
                compiler: KEY_COMPILER,
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
                compiler: KEY_COMPILER,
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
                compiler: KEY_COMPILER,
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
                compiler: KEY_COMPILER,
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
                compiler: KEY_COMPILER,
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
                'd1475dc01d3c9464b477821ffea245539a292d0478dcbe3f1f8f517f805893d8',
            ],
            [
                'compiler pane, "Execute the code" ticked',
                'g161',
                compilerPaneBody({filters: {execute: true}}),
                {},
                'd3a59d576e0d50623784a97011c1fc4d315ac1b0c155edc2aa4e55f5e1c85ea9',
            ],
            [
                'compiler pane, link to binary',
                'g161',
                compilerPaneBody({filters: {binary: true}}),
                {},
                'a43b06ea8fd13727d46cfe71cabbd02e8d3c0a0c13feafaff8d43912526539da',
            ],
            [
                'compiler pane, -E',
                'g161',
                compilerPaneBody({userArguments: '-E'}),
                {},
                '505cc8384f90f5f434bd628483613128381095980dd54cad06456e59477e7a87',
            ],
            [
                'compiler pane, boost 1.90 selected',
                'g161',
                compilerPaneBody({libraries: [{id: 'boost', version: '190'}]}),
                {},
                '058e8098a15485dad8295dd34e7c5ec9697e19952e16679465f5e5c4ae04c3a7',
            ],
            [
                'compiler pane, -lfmt (not autodetected)',
                'g161',
                compilerPaneBody({userArguments: '-O2 -lfmt'}),
                {},
                '6c08a5184a8aa0cb337917011c2d99918bdbd7b0895022b29c5818937b8225f4',
            ],
            [
                'compiler pane, clang-tidy added (looked up and stored under the same key)',
                'g161',
                compilerPaneBody({tools: [{id: 'clangtidytrunk', args: [], stdin: ''}]}),
                {},
                '4ec44f8c605e15f34702a85d68aa4c1e5fe9a5d8b1da6586df02ff894cee111a',
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
                '5450e7c946202591105a091ccf7eb4952ab592ec574ba8625d93e6afa7d85a8b',
            ],
            [
                'executor pane',
                'g161',
                executorPaneBody(),
                {},
                '36284587a2736bf0b9fe81e86bddc755399875160d64b5880b5207ad3b455cdb',
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
                '72c523d77a096efcf60d1bf5aa849b3876d1c5df967c92a830e9bd3af03fbea0',
            ],
            [
                'compiler pane, AST pane open',
                'clang2110',
                compilerPaneBody({compilerOptions: {produceAst: true}}),
                {},
                '00f33696122179a820d7d167bcc78785ab5373f623bb506bacb62225442a89a4',
            ],
            [
                'compiler pane, AST pane opened and closed again',
                'clang2110',
                compilerPaneBody({compilerOptions: {produceAst: false}}),
                {},
                '83cbfc662abcf3a98d092b6a7330d9d341ceecf9b39823a146fcd121a46134bb',
            ],
            [
                'compiler pane, opt remarks pane open',
                'clang2110',
                compilerPaneBody({compilerOptions: {produceOptInfo: true}}),
                {},
                'b89026aaae5ba7ce2b95eefd72628012824535fe85114f891fe3a6cc791e8abe',
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
                compiler: KEY_COMPILER,
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
                compiler: KEY_COMPILER,
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
            const {execute, ...filtersWithoutExecute} = KEY_FILTERS;
            expect(execute).toBe(false);
            expectKey(key, {
                source: CMAKELISTS,
                options: ['-O2'],
                backendOptions: {...KEY_BACKEND_OPTIONS, cmakeArgs: '', customOutputFilename: ''},
                filters: {...filtersWithoutExecute, binary: true, dontMaskFilenames: true},
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
                '67ebef00b3f72436f58de8b7f3fb6d5bde11f11bb808ff5bcd2d4966927190fb',
            ],
            [
                'a build with CMake arguments',
                (() => {
                    const body = cmakeBody(false);
                    body.options.compilerOptions.cmakeArgs = '-DFOO=1';
                    return body;
                })(),
                '5a373887668b81f20fa6e0badc63cfa9a8f81b4295e4138335943eb22c87280c',
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
                'd2ead1220b3b2ad19d34ad8bf87593df54449d589f56b0aa9172191d7d385ab6',
            ],
        ])('with the real g161 CompilerInfo, %s matches the dev instance', async (_, body, devHash) => {
            const {hash} = await recordProjectBuild(body, 'g161');
            expect(hash).toBe(`${devHash}_exec`);
        });
    });
});
