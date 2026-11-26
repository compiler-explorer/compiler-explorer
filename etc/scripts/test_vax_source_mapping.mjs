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

import assert from 'node:assert/strict';

import {CompilerFinder} from '../../lib/compiler-finder.js';
import * as properties from '../../lib/properties.js';

// Manual integration test: node --import=tsx etc/scripts/test_vax_source_mapping.mjs
// Kept outside the unit suite because it needs the deployed VAX toolchains and network access.
properties.initialize('etc/config', ['defaults', 'amazon']);
const compilerProps = new properties.CompilerProps({'c++': {id: 'c++'}}, properties.propsFor('compiler-explorer'));
const finder = new CompilerFinder({}, compilerProps, {}, {get: () => ({libs: {}, tools: {}})});
const compilers = await finder.recurseGetCompilers('c++', '&vax', finder.compilerProps);
assert.deepEqual(
    compilers.map(compiler => compiler.id),
    ['vaxg1240', 'vaxg1050', 'vaxg1040'],
);

for (const compiler of compilers) {
    for (const binary of [false, true]) {
        const source =
            'int square(int num) {\n return num * num;\n}' + (binary ? '\nint main() { return square(5); }' : '');
        const response = await fetch(`https://godbolt.org/api/compiler/${compiler.id}/compile`, {
            method: 'POST',
            signal: AbortSignal.timeout(60_000),
            headers: {Accept: 'application/json', 'Content-Type': 'application/json'},
            body: JSON.stringify({
                source,
                options: {
                    userArguments: `${compiler.options} -O0`,
                    filters: {binary, directives: true, labels: true, commentOnly: true},
                    compilerOptions: {},
                },
            }),
        });
        assert.equal(response.status, 200);
        const result = await response.json();
        const name = `${compiler.id} ${binary ? 'binary' : 'assembly'}`;
        assert.equal(result.code, 0, `${name}: ${JSON.stringify(result.stderr)}`);
        const multiply = result.asm.find(line => line.text.includes('mull3'));
        assert.ok(multiply, `${name}: multiplication is present`);
        assert.equal(multiply.source?.line, 2, `${name}: multiplication maps to the return statement`);
        assert.equal(multiply.source.file, null, `${name}: multiplication maps to the main source`);
        console.log(`${name}: multiplication maps to source line 2`);
    }
}
