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

// Quick Bench hid the Google Benchmark boilerplate: users wrote only their benchmark functions, and the server added
// an include, a `Noop` benchmark to normalise against, and the main function. Over the years it used three wrappers,
// and stored the wrapped source, so stored files can carry any of them. CE shows exactly what it compiles, so a
// migrated benchmark gets the wrapper back in plain sight.

const PREFIX_V1 = '#include <benchmark/benchmark_api.h>\n';
const SUFFIX_V1 = `

static void Noop(benchmark::State& state) {
  while (state.KeepRunning());
}
BENCHMARK(Noop);
BENCHMARK_MAIN()`;
const PREFIX_V2 = '#include <benchmark/benchmark.h>\n';
const NOOP_V2 = `

static void Noop(benchmark::State& state) {
  for (auto _ : state) benchmark::DoNotOptimize(0);
}
BENCHMARK(Noop);`;
const SUFFIX_V2 = `${NOOP_V2}
BENCHMARK_MAIN()`;
const SUFFIX_V3 = `${NOOP_V2}
BENCHMARK_MAIN();`;

// What Quick Bench's front end stripped from code handed to it by other sites, which usually send a whole file.
const HASH_INCLUDE = '#include <benchmark/benchmark.h>\n';
const HASH_MAIN = '\nBENCHMARK_MAIN();';

/** Strips any of Quick Bench's wrappers from a stored .cpp file, leaving the code the user typed. */
export function unwrapQuickBenchCode(code: string): string {
    // Same order and same tests as Quick Bench's `unwrapCode`, so a file unwraps exactly as it did there.
    if (code.startsWith(PREFIX_V1)) code = code.slice(PREFIX_V1.length);
    if (code.endsWith(SUFFIX_V1)) code = code.slice(0, -SUFFIX_V1.length);
    if (code.startsWith(PREFIX_V2)) code = code.slice(PREFIX_V2.length);
    if (code.endsWith(SUFFIX_V2)) code = code.slice(0, -SUFFIX_V2.length);
    if (code.endsWith(SUFFIX_V3)) code = code.slice(0, -SUFFIX_V3.length);
    return code;
}

/** Strips the include and main that Quick Bench's front end removed from code passed in a link's #fragment. */
export function stripQuickBenchHashBoilerplate(text: string): string {
    return text.replace(HASH_INCLUDE, '').replace(HASH_MAIN, '');
}

/** Wraps code exactly as Quick Bench's last protocol compiled it: the include, a Noop benchmark and BENCHMARK_MAIN(). */
export function wrapQuickBenchCode(code: string): string {
    return PREFIX_V2 + code + SUFFIX_V3;
}
