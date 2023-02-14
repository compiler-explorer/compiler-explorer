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

import cloneDeep from 'lodash.clonedeep';

import {LlvmAstParser} from '../lib/llvm-ast';
import * as properties from '../lib/properties';
import {PropertyValue} from '../lib/properties.interfaces';
import * as utils from '../lib/utils';

import {fs, should} from './utils';

const languages = {
    'c++': {id: 'c++'},
};

function mockAstOutput(astLines) {
    return {stdout: astLines.map(l => ({text: l}))};
}

describe('llvm-ast', function () {
    let compilerProps;
    let astParser;
    let astDump;
    let compilerOutput;
    let astDumpWithCTime;
    let astDumpNestedDecl1346;

    before(() => {
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
        processed.length.should.be.below(origHeight);
    });

    it('removes invalid slocs', () => {
        const processed = astParser.processAst(cloneDeep(compilerOutput));
        astDump.should.match(/<invalid sloc>/);
        const fullText = processed.map(l => l.text).join('\n');
        fullText.should.not.match(/<invalid sloc>/);
    });

    it('keeps reasonable-sized output', () => {
        astDumpWithCTime.length.should.be.above(200);

        const output = mockAstOutput(astDumpWithCTime);
        const processed = astParser.processAst(output);
        processed.length.should.be.below(200);
    });

    it('links some source lines', () => {
        should.exist(compilerOutput.stdout.find(l => l.text.match(/col:21, line:4:1/)));
        should.exist(compilerOutput.stdout.find(l => l.text.match(/line:3:5, col:18/)));
        const processed = astParser.processAst(cloneDeep(compilerOutput));
        should.exist(processed.find(l => l.source && 0 < l.source.from.line));
        processed.find(l => l.text.match(/col:21, line:4:1/)).source.to.line.should.equal(4);
        processed.find(l => l.text.match(/col:21, line:4:1/)).source.to.col.should.equal(1);
        processed.find(l => l.text.match(/col:21, line:4:1/)).source.from.col.should.equal(21);
        processed.find(l => l.text.match(/line:3:5, col:18/)).source.from.line.should.equal(3);
        processed.find(l => l.text.match(/line:3:5, col:18/)).source.from.col.should.equal(5);
        processed.find(l => l.text.match(/line:3:5, col:18/)).source.to.line.should.equal(3);
        processed.find(l => l.text.match(/line:3:5, col:18/)).source.to.col.should.equal(18);
        // Here "from.line" is inherited from the parent "FunctionDecl <<source>:2:1, line:4:1>"
        processed.find(l => l.text.match(/CompoundStmt.*<col:21, line:4:1>/)).source.from.line.should.equal(2);
    });

    it('does not truncate nested declarations', () => {
        // See https://github.com/compiler-explorer/compiler-explorer/issues/1346
        const output = mockAstOutput(astDumpNestedDecl1346);
        const processed = astParser.processAst(output);
        processed.length.should.be.above(2);
        should.exist(processed.find(l => l.text.match(/CXXRecordDecl.*struct x/)));
        should.exist(processed.find(l => l.text.match(/TypedefDecl.*struct x/)));
        should.exist(processed.find(l => l.text.match(/ElaboratedType/)));
        should.exist(processed.find(l => l.text.match(/RecordType/)));
        should.exist(processed.find(l => l.text.match(/CXXRecord/)));
    });
});

describe('llvm-ast bug-3849a', function () {
    let compilerProps;
    let astParser;
    let astDump;
    let compilerOutput;

    before(() => {
        const fakeProps = new properties.CompilerProps(languages, properties.fakeProps({}));
        compilerProps = (fakeProps.get as any).bind(fakeProps, 'c++');

        astParser = new LlvmAstParser(compilerProps);
        astDump = utils.splitLines(fs.readFileSync('test/ast/bug-3849a.ast').toString());
        compilerOutput = mockAstOutput(astDump);
    });

    it('should have more than 2 lines', () => {
        const processed = astParser.processAst(compilerOutput);
        processed.length.should.be.above(2);
    });
});

describe('llvm-ast bug-3849b', function () {
    let compilerProps;
    let astParser;
    let astDump;
    let compilerOutput;

    before(() => {
        const fakeProps = new properties.CompilerProps(languages, properties.fakeProps({}));
        compilerProps = (fakeProps.get as any).bind(fakeProps, 'c++');

        astParser = new LlvmAstParser(compilerProps);
        astDump = utils.splitLines(fs.readFileSync('test/ast/bug-3849b.ast').toString());
        compilerOutput = mockAstOutput(astDump);
    });

    it('should have not too many lines', () => {
        const processed = astParser.processAst(compilerOutput);
        processed.length.should.be.above(200);
        processed.length.should.be.below(300);
    });
});
