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

import {describe, expect, it} from 'vitest';

import {getDeclaredFunctionNames, getFunctionDefinitions} from '../../modes/duckling-completions.js';

describe('Duckling source completions', () => {
    it('suggests function and function declaration names without duplicates', () => {
        expect(
            getDeclaredFunctionNames(
                `# fun comment()\nfun main() -> i64 = {\n    return helper();\n}\nfundecl helper() -> i64;\nfun main() -> i64 = {}`,
            ),
        ).toEqual(['main', 'helper']);
    });

    it('returns no names when the source has no function declarations', () => {
        expect(getDeclaredFunctionNames('var answer: i64 = 42i64;')).toEqual([]);
    });

    it('finds function definitions with one-based source positions and ignores declarations', () => {
        expect(
            getFunctionDefinitions(
                '# fun ignored()\nfundecl external() -> i64;\nfun main() -> i64 = {\n    return helper();\n}\n    fun helper() = {}',
            ),
        ).toEqual([
            {name: 'main', lineNumber: 3, column: 5},
            {name: 'helper', lineNumber: 6, column: 9},
        ]);
    });
});
