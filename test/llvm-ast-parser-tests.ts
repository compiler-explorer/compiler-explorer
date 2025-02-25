// Copyright (c) 2023, Compiler Explorer Authors
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

import cloneDeep from 'lodash.clonedeep';
import {beforeAll, describe, expect, it} from 'vitest';

import {LlvmAstParser} from '../lib/llvm-ast.js';
import * as properties from '../lib/properties.js';
import * as utils from '../lib/utils.js';

const languages = {
    'c++': {id: 'c++'},
};

function mockAstOutput(astLines) {
    return {stdout: astLines.map(l => ({text: l}))};
}

describe('llvm-ast', () => {
    let compilerProps;
    let astParser;
    let astDump;
    let compilerOutput;
    let astDumpWithCTime;
    let astDumpNestedDecl1346;

    beforeAll(() => {
        const fakeProps = new properties.CompilerProps(languages, properties.fakeProps({}));
        compilerProps = (fakeProps.get as any).bind(fakeProps, 'c++');

        astParser = new LlvmAstParser(compilerProps);
        astDump = utils.splitLines(fs.readFileSync('test/ast/square.ast').toString());
        compilerOutput = mockAstOutput(astDump);
        astDumpWithCTime = utils.splitLines(fs.readFileSync('test/ast/ctime.ast').toString());
        astDumpNestedDecl1346 = utils.splitLines(fs.readFileSync('test/ast/bug-1346-typedef-struct.ast').toString());
    });

    it('keeps fewer lines than the original', () => {
        const origHeight = astDump.length;
        const processed = astParser.processAst(cloneDeep(compilerOutput));
        expect(processed.length).toBeLessThan(origHeight);
    });

    it('removes invalid slocs', () => {
        expect(astDump.join('\n')).toMatch(/<invalid sloc>/);
        const processed = astParser.processAst(cloneDeep(compilerOutput));
        const fullText = processed.map(l => l.text).join('\n');
        expect(fullText).not.toMatch(/<invalid sloc>/);
    });

    it('keeps reasonable-sized output', () => {
        expect(astDumpWithCTime.length).toBeGreaterThan(200);

        const output = mockAstOutput(astDumpWithCTime);
        const processed = astParser.processAst(output);
        expect(processed.length).toBeLessThan(200);
    });

    it('links some source lines', () => {
        expect(compilerOutput.stdout.find(l => l.text.match(/col:21, line:4:1/))).toBeTruthy();
        expect(compilerOutput.stdout.find(l => l.text.match(/line:3:5, col:18/))).toBeTruthy();
        const processed = astParser.processAst(cloneDeep(compilerOutput));
        expect(processed.find(l => l.source && 0 < l.source.from.line)).toBeTruthy();
        expect(processed.find(l => l.text.match(/col:21, line:4:1/))).toMatchObject({
            source: {to: {line: 4, col: 1}, from: {line: 2, col: 21}},
        });
        expect(processed.find(l => l.text.match(/line:3:5, col:18/))).toMatchObject({
            source: {to: {line: 3, col: 18}, from: {line: 3, col: 5}},
        });
        // Here "from.line" is inherited from the parent "FunctionDecl <<source>:2:1, line:4:1>"
        expect(processed.find(l => l.text.match(/CompoundStmt.*<col:21, line:4:1>/))).toMatchObject({
            source: {from: {line: 2}},
        });
    });

    it('does not truncate nested declarations', () => {
        // See https://github.com/compiler-explorer/compiler-explorer/issues/1346
        const output = mockAstOutput(astDumpNestedDecl1346);
        const processed = astParser.processAst(output);
        expect(processed.length).toBeGreaterThan(2);
        expect(processed.find(l => l.text.match(/CXXRecordDecl.*struct x/))).toBeTruthy();
        expect(processed.find(l => l.text.match(/TypedefDecl.*struct x/))).toBeTruthy();
        expect(processed.find(l => l.text.match(/ElaboratedType/))).toBeTruthy();
        expect(processed.find(l => l.text.match(/RecordType/))).toBeTruthy();
        expect(processed.find(l => l.text.match(/CXXRecord/))).toBeTruthy();
    });
});

describe('llvm-ast bug-3849a', () => {
    let compilerProps;
    let astParser;
    let astDump;
    let compilerOutput;

    beforeAll(() => {
        const fakeProps = new properties.CompilerProps(languages, properties.fakeProps({}));
        compilerProps = (fakeProps.get as any).bind(fakeProps, 'c++');

        astParser = new LlvmAstParser(compilerProps);
        astDump = utils.splitLines(fs.readFileSync('test/ast/bug-3849a.ast').toString());
        compilerOutput = mockAstOutput(astDump);
    });

    it('should have more than 2 lines', () => {
        const processed = astParser.processAst(compilerOutput);
        expect(processed.length).toBeGreaterThan(2);
    });
});

describe('llvm-ast bug-3849b', () => {
    let compilerProps;
    let astParser;
    let astDump;
    let compilerOutput;

    beforeAll(() => {
        const fakeProps = new properties.CompilerProps(languages, properties.fakeProps({}));
        compilerProps = (fakeProps.get as any).bind(fakeProps, 'c++');

        astParser = new LlvmAstParser(compilerProps);
        astDump = utils.splitLines(fs.readFileSync('test/ast/bug-3849b.ast').toString());
        compilerOutput = mockAstOutput(astDump);
    });

    it('should have not too many lines', () => {
        const processed = astParser.processAst(compilerOutput);
        expect(processed.length).toBeGreaterThan(200);
        expect(processed.length).toBeLessThan(300);
    });
});

describe('llvm-ast bug-5889', () => {
    let compilerProps;
    let astParser;
    let astDump;
    let compilerOutput;

    beforeAll(() => {
        const fakeProps = new properties.CompilerProps(languages, properties.fakeProps({}));
        compilerProps = (fakeProps.get as any).bind(fakeProps, 'c++');

        astParser = new LlvmAstParser(compilerProps);
        astDump = utils.splitLines(fs.readFileSync('test/ast/bug-5889.ast').toString());
        compilerOutput = mockAstOutput(astDump);
    });

    it('should have not too many lines', () => {
        const processed = astParser.processAst(compilerOutput);
        expect(processed.length).toBeLessThan(50);
    });
});
