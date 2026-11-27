// Copyright (c) 2026, Compiler Explorer Authors
// Portions derived from Quick Bench (https://github.com/FredTingaud/quick-bench-back-end,
// https://github.com/FredTingaud/quick-bench-front-end, https://github.com/FredTingaud/quick-bench-docker),
// Copyright (c) 2017, Fred Tingaud, used under the BSD 2-Clause License.
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

import {describe, expect, it} from 'vitest';

import {ClientState} from '../../lib/clientstate.js';
import {goldenLayoutFromClientState} from '../../lib/clientstate-normalizer.js';
import type {QuickBenchCandidates} from '../../lib/quick-bench/compilers.js';
import {
    convertQuickBenchHash,
    convertStoredQuickBench,
    quickBenchCompilerOptions,
} from '../../lib/quick-bench/convert.js';
import {decodeQuickBenchId} from '../../lib/quick-bench/id.js';
import {
    stripQuickBenchHashBoilerplate,
    unwrapQuickBenchCode,
    wrapQuickBenchCode,
} from '../../lib/quick-bench/source.js';

// Fixtures are Quick Bench's own (quick-bench-back-end test/*.test.js), so these pin the behaviour of every stored
// format it ever wrote.

const START_CODE = `static void BM_StringCreation(benchmark::State& state) {
  while (state.KeepRunning())
    std::string empty_string;
}
// Register the function as a benchmark
BENCHMARK(BM_StringCreation);

static void BM_StringCopy(benchmark::State& state) {
  std::string x = "hello";
  while (state.KeepRunning())
    std::string copy(x);
}
BENCHMARK(BM_StringCopy);
`;

const CODE_V1 = `#include <benchmark/benchmark_api.h>
${START_CODE}

static void Noop(benchmark::State& state) {
  while (state.KeepRunning());
}
BENCHMARK(Noop);
BENCHMARK_MAIN()`;

const CODE_V2 = `#include <benchmark/benchmark.h>
${START_CODE}

static void Noop(benchmark::State& state) {
  for (auto _ : state) benchmark::DoNotOptimize(0);
}
BENCHMARK(Noop);
BENCHMARK_MAIN()`;

const CODE_V3 = `${CODE_V2};`;

const VISIBLE_SOURCE = CODE_V3;

const OPT_V1 = '{"protocolVersion":1,"compiler":"clang++-3.8","optim":"1","cppVersion":"17"}';
const OPT_V2 = '{"protocolVersion":2,"compiler":"clang++-3.8","optim":"1","cppVersion":"17","isAnnotated":true}';
const OPT_V3 =
    '{"protocolVersion":3,"compiler":"clang-9.1","optim":"1","cppVersion":"17","isAnnotated":true,"lib":"llvm"}';
const OPT_V4 =
    '{"protocolVersion":4, "compiler":"clang-9.0","optim":"3","cppVersion":"17","lib":"llvm","isAnnotated":true}';
const OPT_V5 =
    '{"protocolVersion":5, "compiler":"clang-17.0","optim":"3","cppVersion":"c++17","lib":"llvm","flags":["-ftest"],"disassemblyOption":"att"}';

const CANDIDATES: QuickBenchCandidates = {
    gcc: [
        {id: 'g550', semver: '5.5'},
        {id: 'g132', semver: '13.2'},
    ],
    clang: [
        {id: 'clang380', semver: '3.8'},
        {id: 'clang381', semver: '3.8.1'},
        {id: 'clang900', semver: '9.0.0'},
        {id: 'clang901', semver: '9.0.1'},
        {id: 'clang1701', semver: '17.0.1'},
    ],
};

describe('Quick Bench ids', () => {
    // From quick-bench-back-end test/url-name.test.js.
    it.each([
        ['Yw05jL2OjJ12vT_xft47d6vkMC4', '630d398cbd8e8c9d76bd3fb17ede3b77abe4302e'],
        ['hbkmjh-0uSQKrYlQ-qNjLpgDEto', '85b9268e1ff4b9240aad8950fea3632e980312da'],
        ['hbkmjh_0uSQKrYlQ_qNjLpgDEto', '85b9268e1fb4b9240aad8950faa3632e980312da'],
    ])('decodes %s to %s', (id, sha1) => {
        expect(decodeQuickBenchId(id)).toEqual(sha1);
    });

    it('rejects things that are not ids', () => {
        expect(decodeQuickBenchId('')).toBeUndefined();
        expect(decodeQuickBenchId('Yw05jL2OjJ12vT_xft47d6vkMC')).toBeUndefined();
        expect(decodeQuickBenchId('Yw05jL2OjJ12vT_xft47d6vkMC4x')).toBeUndefined();
        expect(decodeQuickBenchId('Yw05jL2OjJ12vT/xft47d6vkMC4')).toBeUndefined();
        expect(decodeQuickBenchId('Yw05jL2OjJ12vT+xft47d6vkMC4')).toBeUndefined();
        // 27 characters, but the last one carries bits a 20-byte hash doesn't have.
        expect(decodeQuickBenchId('Yw05jL2OjJ12vT_xft47d6vkMC5')).toBeUndefined();
    });
});

describe('Quick Bench source wrappers', () => {
    // From quick-bench-back-end test/code.test.js.
    it.each([
        ['v1', CODE_V1],
        ['v2', CODE_V2],
        ['v3', CODE_V3],
    ])('unwraps %s code', (_, code) => {
        expect(unwrapQuickBenchCode(code)).toEqual(START_CODE);
    });

    it('leaves unwrapped code alone', () => {
        expect(unwrapQuickBenchCode(START_CODE)).toEqual(START_CODE);
    });

    it('wraps with the include, Noop and main, as Quick Bench last did', () => {
        expect(wrapQuickBenchCode(START_CODE)).toEqual(VISIBLE_SOURCE);
        expect(unwrapQuickBenchCode(wrapQuickBenchCode(START_CODE))).toEqual(START_CODE);
    });

    it('strips the boilerplate Quick Bench stripped from #fragment code', () => {
        const text = `#include <benchmark/benchmark.h>\n${START_CODE}\nBENCHMARK_MAIN();`;
        expect(stripQuickBenchHashBoilerplate(text)).toEqual(START_CODE);
    });
});

describe('Quick Bench compiler options', () => {
    it.each([
        [{compiler: 'gcc-13.2', optim: '0', cppVersion: '11'}, '-std=c++11 -O0'],
        [{compiler: 'gcc-13.2', optim: 'S', cppVersion: 'c++14'}, '-std=c++14 -Os'],
        [{compiler: 'gcc-13.2', optim: 'G', cppVersion: '2a'}, '-std=c++2a -Og'],
        [{compiler: 'gcc-13.2', optim: '2', cppVersion: '98'}, '-std=c++98 -O2'],
        [{compiler: 'clang-17.0', optim: 'F', cppVersion: 'c++20', lib: 'gnu'}, '-std=c++20 -Ofast'],
        [{compiler: 'clang-17.0', optim: '2', cppVersion: '17', lib: 'llvm'}, '-std=c++17 -O2 -stdlib=libc++'],
        [
            {compiler: 'clang-17.0', optim: '3', cppVersion: '23', flags: ['-fexperimental-library']},
            '-std=c++23 -O3 -fexperimental-library',
        ],
    ])('maps %j to %s', (options, expected) => {
        const warnings: string[] = [];
        expect(quickBenchCompilerOptions(options, warnings)).toEqual(expected);
        expect(warnings).toEqual([]);
    });

    it('drops anything it does not recognise, with a warning', () => {
        const warnings: string[] = [];
        const options = {
            compiler: 'gcc-13.2',
            optim: '9',
            cppVersion: '17 -fplugin=x.so',
            lib: 'llvm',
            flags: ['-fcoroutines', 'two words', 42 as unknown as string],
        };
        expect(quickBenchCompilerOptions(options, warnings)).toEqual('-fcoroutines');
        expect(warnings).toHaveLength(5);
    });
});

function onlyExecutor(state: ClientState) {
    expect(state.sessions).toHaveLength(1);
    const session = state.sessions[0];
    expect(session.language).toEqual('c++');
    expect(session.compilers).toEqual([]);
    expect(session.executors).toHaveLength(1);
    return {session, executor: session.executors[0]};
}

describe('Converting stored Quick Bench benchmarks', () => {
    // .opt contents from quick-bench-back-end test/getv1..getv5.test.js, which all store the v1 .cpp.
    it.each([
        ['v1', OPT_V1, 'clang381', '-std=c++17 -O1', 0],
        ['v2', OPT_V2, 'clang381', '-std=c++17 -O1', 0],
        // clang-9.1 never existed; the fixture is Quick Bench's own, so it shows the fallback.
        ['v3', OPT_V3, 'clang901', '-std=c++17 -O1 -stdlib=libc++', 1],
        ['v4', OPT_V4, 'clang901', '-std=c++17 -O3 -stdlib=libc++', 0],
        ['v5', OPT_V5, 'clang1701', '-std=c++17 -O3 -stdlib=libc++ -ftest', 0],
    ])('converts protocol %s', (_, opt, compilerId, options, warningCount) => {
        const {state, warnings} = convertStoredQuickBench(CODE_V1, opt, CANDIDATES);
        const {session, executor} = onlyExecutor(state);
        expect(session.source).toEqual(VISIBLE_SOURCE);
        expect(executor.compiler.id).toEqual(compilerId);
        expect(executor.compiler.options).toEqual(options);
        expect(executor.compiler.libs).toEqual([{name: 'benchmark', ver: '161'}]);
        expect(executor.compilerVisible).toBe(true);
        expect(warnings).toHaveLength(warningCount);
    });

    it('adds range-v3 when the code uses it', () => {
        const code = wrapQuickBenchCode('#include <range/v3/all.hpp>\n' + START_CODE);
        const {state} = convertStoredQuickBench(code, OPT_V5, CANDIDATES);
        expect(onlyExecutor(state).executor.compiler.libs).toEqual([
            {name: 'benchmark', ver: '161'},
            {name: 'rangesv3', ver: '030'},
        ]);
    });

    it('leaves the compiler for CE to choose when it has none of the family', () => {
        const {state, warnings} = convertStoredQuickBench(CODE_V1, OPT_V5, {gcc: [], clang: []});
        expect(onlyExecutor(state).executor.compiler.id).toEqual('');
        expect(warnings).toHaveLength(1);
    });

    it('rejects options with no compiler', () => {
        expect(() => convertStoredQuickBench(CODE_V1, '{"optim":"3"}', CANDIDATES)).toThrow();
        expect(() => convertStoredQuickBench(CODE_V1, 'not json', CANDIDATES)).toThrow();
    });

    it('produces state CE can lay out as an editor and an executor', () => {
        const {state} = convertStoredQuickBench(CODE_V1, OPT_V5, CANDIDATES);
        const roundTripped = new ClientState(JSON.parse(JSON.stringify(state)));
        const components = JSON.stringify(goldenLayoutFromClientState(roundTripped));
        expect(components).toContain('"componentName":"codeEditor"');
        expect(components).toContain('"componentName":"executor"');
        expect(components).toContain('"compiler":"clang1701"');
    });
});

function asciiHash(value: unknown) {
    // What CE's editor sends: JSON with non-ASCII escaped, base64 encoded.
    const json = JSON.stringify(value).replace(
        /[\u0080-￿]/g,
        c => `\\u${c.charCodeAt(0).toString(16).padStart(4, '0')}`,
    );
    return '#' + Buffer.from(json).toString('base64');
}

function utf8Hash(value: unknown) {
    // What Quick Bench, Build Bench and C++ Insights send: the UTF-8 bytes of the JSON, base64 encoded.
    return '#' + Buffer.from(JSON.stringify(value), 'utf8').toString('base64');
}

describe('Converting quick-bench.com/#... links', () => {
    const ceText = `#include <benchmark/benchmark.h>\n${START_CODE}\nBENCHMARK_MAIN();`;

    it('converts what CE sent', () => {
        const fragment = asciiHash({text: ceText, compiler: 'gcc-13.2', optim: '2', cppVersion: '17'});
        const result = convertQuickBenchHash(fragment, CANDIDATES);
        const {session, executor} = onlyExecutor(result!.state);
        expect(session.source).toEqual(VISIBLE_SOURCE);
        expect(executor.compiler.id).toEqual('g132');
        expect(executor.compiler.options).toEqual('-std=c++17 -O2');
        expect(result!.warnings).toEqual([]);
    });

    it('decodes UTF-8 and escaped non-ASCII alike', () => {
        const text = '// naïve → fast\n' + START_CODE;
        for (const fragment of [asciiHash({text}), utf8Hash({text}), encodeURIComponent(utf8Hash({text}))]) {
            const result = convertQuickBenchHash(fragment, CANDIDATES);
            expect(onlyExecutor(result!.state).session.source).toEqual(wrapQuickBenchCode(text));
        }
    });

    it("fills in Quick Bench's defaults", () => {
        const result = convertQuickBenchHash(utf8Hash({text: START_CODE}), CANDIDATES);
        const {executor} = onlyExecutor(result!.state);
        expect(executor.compiler.id).toEqual('clang1701');
        expect(executor.compiler.options).toEqual('-std=c++20 -O3');
    });

    it('takes the first entry with code from an array, as Quick Bench did', () => {
        const fragment = utf8Hash([
            {text: '', compiler: 'gcc-5.5'},
            {text: START_CODE, compiler: 'clang-3.8', optim: '0', cppVersion: '11', lib: 'llvm'},
            {text: START_CODE, compiler: 'gcc-5.5', optim: '1', cppVersion: '14'},
        ]);
        const {executor} = onlyExecutor(convertQuickBenchHash(fragment, CANDIDATES)!.state);
        expect(executor.compiler.id).toEqual('clang381');
        expect(executor.compiler.options).toEqual('-std=c++11 -O0 -stdlib=libc++');
    });

    it.each([
        ['an empty fragment', '#'],
        ['not base64 JSON', '#definitely-not-json'],
        ['JSON with no code', utf8Hash({compiler: 'gcc-13.2'})],
        ['a JSON scalar', utf8Hash(42)],
    ])('ignores %s', (_, fragment) => {
        expect(convertQuickBenchHash(fragment, CANDIDATES)).toBeUndefined();
    });
});
