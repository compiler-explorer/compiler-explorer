// Copyright (c) 2017, Matt Godbolt
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
    utils = require('../lib/utils'),
    logger = require('../lib/logger').logger,
    fs = require('fs-extra');

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

describe('Pascal compiler output', () => {
    it('recognize fpc identifier not found error', () => {
        utils.parseOutput('output.pas(13,23) Error: Identifier not found "adsadasd"', 'output.pas').should.deep.equals([
            {
                tag: {
                    column: 23,
                    line: 13,
                    text: 'Error: Identifier not found "adsadasd"'
                },
                text: '<source>(13,23) Error: Identifier not found "adsadasd"'
            }
        ]);
    });

    it('recognize fpc exiting error', () => {
        utils.parseOutput('output.pas(17) Fatal: There were 1 errors compiling module, stopping', 'output.pas').should.deep.equals([
            {
                tag: {
                    column: 0,
                    line: 17,
                    text: 'Fatal: There were 1 errors compiling module, stopping'
                },
                text: '<source>(17) Fatal: There were 1 errors compiling module, stopping'
            }
        ]);
    });

    it('removes the temp path', () => {
        utils.parseOutput('Compiling /tmp/path/prog.dpr\noutput.pas(17) Fatal: There were 1 errors compiling module, stopping', 'output.pas', '/tmp/path/').should.deep.equals([
            {
                text: 'Compiling prog.dpr'
            },
            {
                tag: {
                    column: 0,
                    line: 17,
                    text: 'Fatal: There were 1 errors compiling module, stopping'
                },
                text: '<source>(17) Fatal: There were 1 errors compiling module, stopping'
            }
        ]);
    });
});

describe('Tool output', () => {
    it('removes the relative path', () => {
        utils.parseOutput('./example.cpp:1:1: Fatal: There were 1 errors compiling module, stopping', './example.cpp').should.deep.equals([
            {
                tag: {
                    column: 1,
                    line: 1,
                    text: 'Fatal: There were 1 errors compiling module, stopping'
                },
                text: '<source>:1:1: Fatal: There were 1 errors compiling module, stopping'
            }
        ]);
    });

    it('removes the jailed path', () => {
        utils.parseOutput('/home/ubuntu/example.cpp:1:1: Fatal: There were 1 errors compiling module, stopping', './example.cpp').should.deep.equals([
            {
                tag: {
                    column: 1,
                    line: 1,
                    text: 'Fatal: There were 1 errors compiling module, stopping'
                },
                text: '<source>:1:1: Fatal: There were 1 errors compiling module, stopping'
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

describe('Trim right', () => {
    it('works', () => {
        utils.trimRight('  ').should.equal('');
        utils.trimRight('').should.equal('');
        utils.trimRight(' ab ').should.equal(' ab');
        utils.trimRight(' a  b ').should.equal(' a  b');
        utils.trimRight('a    ').should.equal('a');
    });
});

describe('Anonymizes all kind of IPs', () => {
    it('Ignores localhost', () => {
        utils.anonymizeIp('localhost').should.equal('localhost');
        utils.anonymizeIp('localhost:42').should.equal('localhost:42');
    });
    it('Removes last octet from IPv4 addresses', () => {
        utils.anonymizeIp('127.0.0.0').should.equal('127.0.0.0');
        utils.anonymizeIp('127.0.0.10').should.equal('127.0.0.0');
        utils.anonymizeIp('127.0.0.255').should.equal('127.0.0.0');
    });
    it('Removes last 3 hextets from IPv6 addresses', () => {
        // Not necessarily valid addresses, we're interested in the format
        utils.anonymizeIp('ffff:aaaa:dead:beef').should.equal('ffff:0:0:0');
        utils.anonymizeIp('bad:c0de::').should.equal('bad:0:0:0');
        utils.anonymizeIp(':1d7e::c0fe').should.equal(':0:0:0');
    });
});

describe('Logger functionality', () => {
    it('has info stream with a write function', () => {
        logger.stream.write.should.a("function");
    });
    it('has warning stream with a write function', () => {
        logger.warnStream.write.should.a("function");
    });
    it('has error stream with a write function', () => {
        logger.errStream.write.should.a("function");
    });
});

describe('Hash interface', () => {
    it('correctly hashes strings', () => {
        const version = 'Compiler Explorer Tests Version 0';
        utils.getHash('cream cheese', version).should.equal('cfff2d1f7a213e314a67cce8399160ae884f794a3ee9d4a01cd37a8c22c67d94');
        utils.getHash('large eggs', version).should.equal('9144dec50b8df5bc5cc24ba008823cafd6616faf2f268af84daf49ac1d24feb0');
        utils.getHash('sugar', version).should.equal('afa3c89d0f6a61de6805314c9bd7c52d020425a3a3c7bbdfa7c0daec594e5ef1');
    });
    it('correctly hashes objects', () => {
        utils.getHash({
            toppings: [
                {name: 'raspberries', optional: false},
                {name: 'ground cinnamon', optional: true}
            ]
        }).should.equal('e205d63abd5db363086621fdc62c4c23a51b733bac5855985a8b56642d570491');
    });
});

describe('GoldenLayout utils', () => {
    it('finds every editor & compiler', () => {
        fs.readJson('test/example-states/default-state.json')
            .then(state => {
                const contents = utils.glGetMainContents(state.content);
                contents.should.deep.equal({
                    editors: [
                        {source: 'Editor 1', language: 'c++'},
                        {source: 'Editor 2', language: 'c++'},
                        {source: 'Editor 3', language: 'c++'},
                        {source: 'Editor 4', language: 'c++'}
                    ],
                    compilers: [
                        {compiler: 'clang_trunk'},
                        {compiler: 'gsnapshot'},
                        {compiler: 'clang_trunk'},
                        {compiler: 'gsnapshot'},
                        {compiler: 'rv32clang'}
                    ]
                });
            });
    });
});

describe('squashes horizontal whitespace', () => {
    it('handles empty input', () => {
        utils.squashHorizontalWhitespace('').should.equals('');
        utils.squashHorizontalWhitespace(' ').should.equals('');
        utils.squashHorizontalWhitespace('    ').should.equals('');
    });
    it('handles leading spaces', () => {
        utils.squashHorizontalWhitespace(' abc').should.equals(' abc');
        utils.squashHorizontalWhitespace('   abc').should.equals('  abc');
        utils.squashHorizontalWhitespace('       abc').should.equals('  abc');
    });
    it('handles interline spaces', () => {
        utils.squashHorizontalWhitespace('abc abc').should.equals('abc abc');
        utils.squashHorizontalWhitespace('abc   abc').should.equals('abc abc');
        utils.squashHorizontalWhitespace('abc     abc').should.equals('abc abc');
    });
    it('handles leading and interline spaces', () => {
        utils.squashHorizontalWhitespace(' abc  abc').should.equals(' abc abc');
        utils.squashHorizontalWhitespace('  abc abc').should.equals('  abc abc');
        utils.squashHorizontalWhitespace('  abc     abc').should.equals('  abc abc');
        utils.squashHorizontalWhitespace('    abc   abc').should.equals('  abc abc');
    });
});

describe('replaces all substrings', () => {
    it('works with no substitutions', () => {
        const string = "This is a line with no replacements";
        utils.replaceAll(string, "not present", "won't be substituted").should.equal(string);
    });
    it('handles odd cases', () => {
        utils.replaceAll("", "", "").should.equal("");
        utils.replaceAll("Hello", "", "").should.equal("Hello");
    });
    it('works with single replacement', () => {
        utils.replaceAll("This is a line with a mistook in it", "mistook", "mistake")
            .should.equal("This is a line with a mistake in it");
        utils.replaceAll("This is a line with a mistook", "mistook", "mistake")
            .should.equal("This is a line with a mistake");
        utils.replaceAll("Mistooks were made", "Mistooks", "Mistakes")
            .should.equal("Mistakes were made");
    });

    it('works with multiple replacements', () => {
        utils.replaceAll("A mistook is a mistook", "mistook", "mistake")
            .should.equal("A mistake is a mistake");
        utils.replaceAll("aaaaaaaaaaaaaaaaaaaaaaaaaaa", "a", "b")
            .should.equal("bbbbbbbbbbbbbbbbbbbbbbbbbbb");
    });

    it('works with overlapping replacements', () => {
        utils.replaceAll("aaaaaaaa", "a", "ba")
            .should.equal("babababababababa");
    });
});

describe('encodes in our version of base32', () => {
    function doTest(original, expected) {
        utils.base32Encode(Buffer.from(original)).should.equal(expected);
    }

    // Done by hand to check that they are valid

    it('works for empty strings', () => {
        doTest("", "");
    });

    it('works for lengths multiple of 5 bits', () => {
        doTest("aaaaa", "3Mn4ha7P");
    });

    it('works for lengths not multiple of 5 bits', () => {
        // 3
        doTest("a", "35");

        // 1
        doTest("aa", "3Mn1");

        // 4
        doTest("aaa", "3Mn48");

        // 2
        doTest("aaaa", "3Mn4ha3");
    });

    it('works for some random strings', () => {
        // I also calculated this ones so lets put them
        doTest("foo", "8rrx8");

        doTest("foobar", "8rrx8b7Pc5");
    });
});
