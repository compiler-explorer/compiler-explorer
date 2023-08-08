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

import {LLVMIRDemangler} from '../lib/demangler/llvm.js';
import {LlvmIrParser} from '../lib/llvm-ir.js';
import * as properties from '../lib/properties.js';

import {chai} from './utils.js';

const expect = chai.expect;

const languages = {
    'c++': {id: 'c++'},
};

describe('llvm-ir parseMetaNode', function () {
    let llvmIrParser;
    let compilerProps;

    before(() => {
        const fakeProps = new properties.CompilerProps(languages, properties.fakeProps({}));
        compilerProps = (fakeProps.get as any).bind(fakeProps, 'c++');

        llvmIrParser = new LlvmIrParser(compilerProps, undefined as unknown as LLVMIRDemangler);
    });

    it('should parse DILocation node', function () {
        llvmIrParser.parseMetaNode('!60 = !DILocation(line: 9, column: 15, scope: !58)').should.deep.equal({
            metaType: 'Location',
            metaId: '!60',
            line: '9',
            column: '15',
            scope: '!58',
        });
    });

    it('should parse distinct DILexicalBlock', function () {
        llvmIrParser
            .parseMetaNode('!50 = distinct !DILexicalBlock(scope: !44, file: !1, line: 8, column: 5)')
            .should.deep.equal({
                metaType: 'LexicalBlock',
                metaId: '!50',
                scope: '!44',
                file: '!1',
                line: '8',
                column: '5',
            });
    });

    it('should parse all value types', function () {
        llvmIrParser
            .parseMetaNode(
                '!44 = distinct !DISubprogram(name: "func<int, int>", ' +
                    'scope: !1, line: 7, isLocal: false, isDefinition: true, flags: ' +
                    'DIFlagPrototyped, ceEmpty: "", ceTest: "a:b\\"c,d")',
            )
            .should.deep.equal({
                metaType: 'Subprogram',
                metaId: '!44',
                name: 'func<int, int>',
                line: '7',
                scope: '!1',
                isLocal: 'false',
                isDefinition: 'true',
                flags: 'DIFlagPrototyped',
                ceTest: 'a:b\\"c,d',
                ceEmpty: '',
            });
    });

    it('should parse distinct DILexicalBlock', function () {
        llvmIrParser
            .parseMetaNode('!1 = !DIFile(filename: "/tmp/example.cpp", directory: "/home/compiler-explorer")')
            .should.deep.equal({
                metaType: 'File',
                metaId: '!1',
                filename: '/tmp/example.cpp',
                directory: '/home/compiler-explorer',
            });
    });
});

describe('llvm-ir getSourceLineNumber', function () {
    let llvmIrParser;
    let compilerProps;

    before(() => {
        const fakeProps = new properties.CompilerProps(languages, properties.fakeProps({}));
        compilerProps = (fakeProps.get as any).bind(fakeProps, 'c++');

        llvmIrParser = new LlvmIrParser(compilerProps, undefined as unknown as LLVMIRDemangler);
    });

    const debugInfo = {
        '!10': {line: 10},
        '!20': {line: 20, scope: '!10'},
        '!11': {scope: '!10'},
        '!12': {line: 0, scope: '!10'},
        '!14': {},
        '!15': {scope: '!14'},
        '!16': {scope: '!42'},
    };

    it('should return a line number', function () {
        expect(llvmIrParser.getSourceLineNumber(debugInfo, '!10')).to.equal(10);
        expect(llvmIrParser.getSourceLineNumber(debugInfo, '!20')).to.equal(20);
    });

    it('should return the line number of its parent scope', function () {
        expect(llvmIrParser.getSourceLineNumber(debugInfo, '!11')).to.equal(10);
        expect(llvmIrParser.getSourceLineNumber(debugInfo, '!12')).to.equal(10);
    });

    it('should return null on non-existend node', function () {
        expect(llvmIrParser.getSourceLineNumber(debugInfo, '!16')).to.equal(null);
    });

    it('should return null if no higher scope has a line', function () {
        expect(llvmIrParser.getSourceLineNumber(debugInfo, '!14')).to.equal(null);
        expect(llvmIrParser.getSourceLineNumber(debugInfo, '!15')).to.equal(null);
    });
});

describe('llvm-ir getSourceColumn', function () {
    let llvmIrParser;
    let compilerProps;

    before(() => {
        const fakeProps = new properties.CompilerProps(languages, properties.fakeProps({}));
        compilerProps = (fakeProps.get as any).bind(fakeProps, 'c++');

        llvmIrParser = new LlvmIrParser(compilerProps, undefined as unknown as LLVMIRDemangler);
    });

    const debugInfo = {
        '!10': {column: 10},
        '!20': {column: 20, scope: '!10'},
        '!11': {scope: '!10'},
        '!12': {column: 0, scope: '!10'},
        '!14': {},
        '!15': {scope: '!14'},
        '!16': {scope: '!42'},
    };

    it('should return a column number', function () {
        expect(llvmIrParser.getSourceColumn(debugInfo, '!10')).to.equal(10);
        expect(llvmIrParser.getSourceColumn(debugInfo, '!20')).to.equal(20);
    });

    it('should return the column number of its parent scope', function () {
        expect(llvmIrParser.getSourceColumn(debugInfo, '!11')).to.equal(10);
        expect(llvmIrParser.getSourceColumn(debugInfo, '!12')).to.equal(10);
    });

    it('should return undefined on non-existend node', function () {
        expect(llvmIrParser.getSourceColumn(debugInfo, '!16')).to.equal(undefined);
        expect(llvmIrParser.getSourceColumn(debugInfo, '!30')).to.equal(undefined);
    });

    it('should return undefined if no higher scope has a column', function () {
        expect(llvmIrParser.getSourceColumn(debugInfo, '!14')).to.equal(undefined);
        expect(llvmIrParser.getSourceColumn(debugInfo, '!15')).to.equal(undefined);
    });
});

describe('llvm-ir getFileName', function () {
    let llvmIrParser;
    let compilerProps;

    before(() => {
        const fakeProps = new properties.CompilerProps(languages, properties.fakeProps({}));
        compilerProps = (fakeProps.get as any).bind(fakeProps, 'c++');

        llvmIrParser = new LlvmIrParser(compilerProps, undefined as unknown as LLVMIRDemangler);
    });
    const debugInfo = {
        '!10': {filename: '/test.cpp'},
        '!20': {filename: '/example.cpp'},
        '!11': {file: '!10'},
        '!21': {file: '!20'},
        '!12': {scope: '!11'},
        '!13': {scope: '!12'},
    };

    it('should return a filename', function () {
        expect(llvmIrParser.getFileName(debugInfo, '!10')).to.equal('/test.cpp');
        expect(llvmIrParser.getFileName(debugInfo, '!11')).to.equal('/test.cpp');
    });

    it('should return the filename of its parent scope', function () {
        expect(llvmIrParser.getFileName(debugInfo, '!12')).to.equal('/test.cpp');
        expect(llvmIrParser.getFileName(debugInfo, '!13')).to.equal('/test.cpp');
    });

    it('should return null on non-existend node', function () {
        expect(llvmIrParser.getFileName(debugInfo, '!42')).to.equal(null);
    });

    it('should not return source filename', function () {
        expect(llvmIrParser.getFileName(debugInfo, '!20')).to.equal(null);
        expect(llvmIrParser.getFileName(debugInfo, '!21')).to.equal(null);
    });
});
