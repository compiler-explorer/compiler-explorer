// Copyright (c) 2012-2018, Matt Godbolt
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

const chai = require('chai'),
    utils = require('../lib/utils');

chai.should();

describe('Splits lines', () => {
    it('handles empty input', () => {
        utils.splitLines('').should.deep.equals([]);
    });
    it('handles a single line with no newline', () => {
        utils.splitLines('A line').should.deep.equals(['A line']);
    });
    it('handles a single line with a newline', () => {
        utils.splitLines('A line\n').should.deep.equals(['A line']);
    });
    it('handles multiple lines', () => {
        utils.splitLines('A line\nAnother line\n').should.deep.equals(['A line', 'Another line']);
    });
    it('handles multiple lines ending on a non-newline', () => {
        utils.splitLines('A line\nAnother line\nLast line').should.deep.equals(
            ['A line', 'Another line', 'Last line']);
    });
    it('handles empty lines', () => {
        utils.splitLines('A line\n\nA line after an empty').should.deep.equals(
            ['A line', '', 'A line after an empty']);
    });
    it('handles a single empty line', () => {
        utils.splitLines('\n').should.deep.equals(['']);
    });
    it('handles multiple empty lines', () => {
        utils.splitLines('\n\n\n').should.deep.equals(['', '', '']);
    });
    it('handles \\r\\n lines', () => {
        utils.splitLines('Some\r\nLines\r\n').should.deep.equals(['Some', 'Lines']);
    });
});

describe('Expands tabs', () => {
    it('leaves non-tabs alone', () => {
        utils.expandTabs('This has no tabs at all').should.equals('This has no tabs at all');
    });
    it('at beginning of line', () => {
        utils.expandTabs('\tOne tab').should.equals('        One tab');
        utils.expandTabs('\t\tTwo tabs').should.equals('                Two tabs');
    });
    it('mid-line', () => {
        utils.expandTabs('0\t1234567A').should.equals('0       1234567A');
        utils.expandTabs('01\t234567A').should.equals('01      234567A');
        utils.expandTabs('012\t34567A').should.equals('012     34567A');
        utils.expandTabs('0123\t4567A').should.equals('0123    4567A');
        utils.expandTabs('01234\t567A').should.equals('01234   567A');
        utils.expandTabs('012345\t67A').should.equals('012345  67A');
        utils.expandTabs('0123456\t7A').should.equals('0123456 7A');
        utils.expandTabs('01234567\tA').should.equals('01234567        A');
    });
});

describe('Parses compiler output', () => {
    it('handles simple cases', () => {
        utils.parseOutput('Line one\nLine two', 'bob.cpp').should.deep.equals([
            {text: 'Line one'},
            {text: 'Line two'}
        ]);
        utils.parseOutput('Line one\nbob.cpp:1 Line two', 'bob.cpp').should.deep.equals([
            {text: 'Line one'},
            {
                tag: {column: 0, line: 1, text: "Line two"},
                text: '<source>:1 Line two'
            }
        ]);
        utils.parseOutput('Line one\nbob.cpp:1:5: Line two', 'bob.cpp').should.deep.equals([
            {text: 'Line one'},
            {
                tag: {column: 5, line: 1, text: "Line two"},
                text: '<source>:1:5: Line two'
            }
        ]);
    });
    it('handles windows output', () => {
        utils.parseOutput('bob.cpp(1) Oh noes', 'bob.cpp').should.deep.equals([
            {
                tag: {column: 0, line: 1, text: 'Oh noes'},
                text: '<source>(1) Oh noes'
            }
        ]);
    });
    it('replaces all references to input source', () => {
        utils.parseOutput('bob.cpp:1 error in bob.cpp', 'bob.cpp').should.deep.equals([
            {
                tag: {column: 0, line: 1, text: 'error in <source>'},
                text: '<source>:1 error in <source>'
            }
        ]);
    });
    it('treats <stdin> as if it were the compiler source', () => {
        utils.parseOutput('<stdin>:120:25: error: variable or field \'transform_data\' declared void', 'bob.cpp')
            .should.deep.equals([
            {
                tag: {
                    column: 25,
                    line: 120,
                    text: 'error: variable or field \'transform_data\' declared void'
                },
                text: '<source>:120:25: error: variable or field \'transform_data\' declared void'
            }
        ]);
    });
});

describe('Pads right', () => {
    it('works', () => {
        utils.padRight('abcd', 8).should.equal('abcd    ');
        utils.padRight('a', 8).should.equal('a       ');
        utils.padRight('', 8).should.equal('        ');
        utils.padRight('abcd', 4).should.equal('abcd');
        utils.padRight('abcd', 2).should.equal('abcd');
    });
});
