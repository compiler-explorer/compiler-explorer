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

import fs from 'node:fs';

import {beforeAll, describe, expect, it} from 'vitest';

import * as properties from '../lib/properties.js';
import {PythonAstParser} from '../lib/python-ast.js';
import * as utils from '../lib/utils.js';

const languages = {
    python: {id: 'python'},
};

function mockAstOutput(astLines: string[]) {
    return {stdout: astLines.map(l => ({text: l}))};
}

describe('python-ast', () => {
    let compilerProps;
    let astParser: any;

    let shortAstDump: string[];
    let longAstDump: string[];

    beforeAll(() => {
        const fakeProps = new properties.CompilerProps(
            languages,
            properties.fakeProps({
                maxLinesOfAst: 100,
            }),
        );
        compilerProps = (fakeProps.get as any).bind(fakeProps, 'python');
        astParser = new PythonAstParser(compilerProps);

        shortAstDump = utils.splitLines(fs.readFileSync('test/ast/python-short.ast').toString());
        longAstDump = utils.splitLines(fs.readFileSync('test/ast/python-long.ast').toString());
    });

    it('keeps original lines on short ast output', () => {
        const processedLines = astParser.processAst(mockAstOutput(shortAstDump));
        expect(processedLines.length).toBe(shortAstDump.length);
    });

    it('truncates long ast output', () => {
        const processedLines = astParser.processAst(mockAstOutput(longAstDump));
        expect(processedLines.length).toBe(100);
    });
});
