// Copyright (c) 2021, Compiler Explorer Authors
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

import { LlvmAstParser } from '../lib/llvm-ast';
import * as properties from '../lib/properties';
import * as utils from '../lib/utils';

import { fs, should } from './utils';

const languages = {
    'c++': {id: 'c++'},
};

function mockAstOutput(astLines) {
    return { stdout : astLines.map(l => ( { text : l } ))};
}

describe('llvm-ast', function () {
    let compilerProps;
    let astParser;
    let astDump;
    let compilerOutput;
    let astDumpWithCTime;

    before(() => {
        let fakeProps = new properties.CompilerProps(languages, properties.fakeProps({}));
        compilerProps = fakeProps.get.bind(fakeProps, 'c++');

        astParser = new LlvmAstParser(compilerProps);
        astDump = utils.splitLines(fs.readFileSync('test/ast/square.ast').toString());
        compilerOutput = mockAstOutput(astDump);
        astDumpWithCTime = utils.splitLines(fs.readFileSync('test/ast/ctime.ast').toString());
    });

    it('keeps fewer lines than the original', () => {
        let origHeight = astDump.length;
        let processed = astParser.processAst(cloneDeep(compilerOutput));
        processed.length.should.be.below(origHeight);
    });

    it('removes invalid slocs', () => {
        let processed = astParser.processAst(cloneDeep(compilerOutput));
        astDump.should.match(/<invalid sloc>/);
        let fullText = processed.map(l => l.text).join('\n');
        fullText.should.not.match(/<invalid sloc>/);
    });

    it('keeps reasonable-sized output', () => {
        astDumpWithCTime.length.should.be.above(100);

        let output = mockAstOutput(astDumpWithCTime);
        let processed = astParser.processAst(output);
        processed.length.should.be.below(100);
    });

    it('links some source lines', () => {
        should.exist(compilerOutput.stdout.find(l => l.text.match(/col:21, line:4:1/)));
        should.exist(compilerOutput.stdout.find(l => l.text.match(/line:3:5, col:18/)));
        let processed = astParser.processAst(cloneDeep(compilerOutput));
        should.exist(processed.find(l => l.source && 0 < l.source.from));
        processed.find(l => l.text.match(/col:21, line:4:1/)).source.to.should.equal(4);
        processed.find(l => l.text.match(/line:3:5, col:18/)).source.from.should.equal(3);
        processed.find(l => l.text.match(/line:3:5, col:18/)).source.to.should.equal(3);
    });
});
