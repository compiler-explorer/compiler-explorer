// Copyright (c) 2017, Compiler Explorer Authors
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

import path from 'path';
import {fileURLToPath} from 'url';

import {describe, expect, it} from 'vitest';
import winston from 'winston';

import {makeLogStream} from '../lib/logger.js';
import * as utils from '../lib/utils.js';

import {fs} from './utils.js';

describe('Splits lines', () => {
    it('handles empty input', () => {
        expect(utils.splitLines('')).toEqual([]);
    });
    it('handles a single line with no newline', () => {
        expect(utils.splitLines('A line')).toEqual(['A line']);
    });
    it('handles a single line with a newline', () => {
        expect(utils.splitLines('A line\n')).toEqual(['A line']);
    });
    it('handles multiple lines', () => {
        expect(utils.splitLines('A line\nAnother line\n')).toEqual(['A line', 'Another line']);
    });
    it('handles multiple lines ending on a non-newline', () => {
        expect(utils.splitLines('A line\nAnother line\nLast line')).toEqual(['A line', 'Another line', 'Last line']);
    });
    it('handles empty lines', () => {
        expect(utils.splitLines('A line\n\nA line after an empty')).toEqual(['A line', '', 'A line after an empty']);
    });
    it('handles a single empty line', () => {
        expect(utils.splitLines('\n')).toEqual(['']);
    });
    it('handles multiple empty lines', () => {
        expect(utils.splitLines('\n\n\n')).toEqual(['', '', '']);
    });
    it('handles \\r\\n lines', () => {
        expect(utils.splitLines('Some\r\nLines\r\n')).toEqual(['Some', 'Lines']);
    });
});

describe('Expands tabs', () => {
    it('leaves non-tabs alone', () => {
        expect(utils.expandTabs('This has no tabs at all')).toEqual('This has no tabs at all');
    });
    it('at beginning of line', () => {
        expect(utils.expandTabs('\tOne tab')).toEqual('        One tab');
        expect(utils.expandTabs('\t\tTwo tabs')).toEqual('                Two tabs');
    });
    it('mid-line', () => {
        expect(utils.expandTabs('0\t1234567A')).toEqual('0       1234567A');
        expect(utils.expandTabs('01\t234567A')).toEqual('01      234567A');
        expect(utils.expandTabs('012\t34567A')).toEqual('012     34567A');
        expect(utils.expandTabs('0123\t4567A')).toEqual('0123    4567A');
        expect(utils.expandTabs('01234\t567A')).toEqual('01234   567A');
        expect(utils.expandTabs('012345\t67A')).toEqual('012345  67A');
        expect(utils.expandTabs('0123456\t7A')).toEqual('0123456 7A');
        expect(utils.expandTabs('01234567\tA')).toEqual('01234567        A');
    });
});

describe('Parses compiler output', () => {
    it('handles simple cases', () => {
        expect(utils.parseOutput('Line one\nLine two', 'bob.cpp')).toEqual([{text: 'Line one'}, {text: 'Line two'}]);
        expect(utils.parseOutput('Line one\nbob.cpp:1 Line two', 'bob.cpp')).toEqual([
            {text: 'Line one'},
            {
                tag: {column: 0, line: 1, text: 'Line two', severity: 3, file: 'bob.cpp'},
                text: '<source>:1 Line two',
            },
        ]);
        expect(utils.parseOutput('Line one\nbob.cpp:1:5: Line two', 'bob.cpp')).toEqual([
            {text: 'Line one'},
            {
                tag: {column: 5, line: 1, text: 'Line two', severity: 3, file: 'bob.cpp'},
                text: '<source>:1:5: Line two',
            },
        ]);
    });
    it('handles windows output', () => {
        expect(utils.parseOutput('bob.cpp(1) Oh noes', 'bob.cpp')).toEqual([
            {
                tag: {column: 0, line: 1, text: 'Oh noes', severity: 3, file: 'bob.cpp'},
                text: '<source>(1) Oh noes',
            },
        ]);
    });
    it('replaces all references to input source', () => {
        expect(utils.parseOutput('bob.cpp:1 error in bob.cpp', 'bob.cpp')).toEqual([
            {
                tag: {column: 0, line: 1, text: 'error in <source>', severity: 3, file: 'bob.cpp'},
                text: '<source>:1 error in <source>',
            },
        ]);
    });
    it('treats warnings and notes as the correct severity', () => {
        expect(utils.parseOutput('Line one\nbob.cpp:1:5: warning Line two', 'bob.cpp')).toEqual([
            {text: 'Line one'},
            {
                tag: {column: 5, line: 1, text: 'warning Line two', severity: 2, file: 'bob.cpp'},
                text: '<source>:1:5: warning Line two',
            },
        ]);
        expect(utils.parseOutput('Line one\nbob.cpp:1:5: note Line two', 'bob.cpp')).toEqual([
            {text: 'Line one'},
            {
                tag: {column: 5, line: 1, text: 'note Line two', severity: 1, file: 'bob.cpp'},
                text: '<source>:1:5: note Line two',
            },
        ]);
    });
    it('treats <stdin> as if it were the compiler source', () => {
        expect(
            utils.parseOutput("<stdin>:120:25: error: variable or field 'transform_data' declared void", 'bob.cpp'),
        ).toEqual([
            {
                tag: {
                    column: 25,
                    line: 120,
                    text: "error: variable or field 'transform_data' declared void",
                    severity: 3,
                    file: 'bob.cpp',
                },
                text: "<source>:120:25: error: variable or field 'transform_data' declared void",
            },
        ]);
    });

    it('parser error with full path', () => {
        expect(utils.parseOutput("/app/example.cl:5:30: error: use of undeclared identifier 'ad'")).toEqual([
            {
                tag: {
                    file: 'example.cl',
                    column: 30,
                    line: 5,
                    text: "error: use of undeclared identifier 'ad'",
                    severity: 3,
                },
                text: "example.cl:5:30: error: use of undeclared identifier 'ad'",
            },
        ]);
    });

    it('removes hyperlink escape sequences', () => {
        expect(
            utils.parseOutput(
                't.c:3:1: warning: control reaches end of non-void function [\x1B]8;;https://gcc.gnu.org/onlinedocs/gcc-14.2.0/gcc/Warning-Options.html#index-Wno-return-type\x1B\\-Wreturn-type\x1B]8;;\x1B\\]',
            ),
        ).toEqual([
            {
                tag: {
                    file: 't.c',
                    line: 3,
                    column: 1,
                    text: 'warning: control reaches end of non-void function [-Wreturn-type]',
                    severity: 2,
                },
                text: 't.c:3:1: warning: control reaches end of non-void function [\x1B]8;;https://gcc.gnu.org/onlinedocs/gcc-14.2.0/gcc/Warning-Options.html#index-Wno-return-type\x1B\\-Wreturn-type\x1B]8;;\x1B\\]',
            },
        ]);
    });
});

describe('Pascal compiler output', () => {
    it('recognize fpc identifier not found error', () => {
        expect(utils.parseOutput('output.pas(13,23) Error: Identifier not found "adsadasd"', 'output.pas')).toEqual([
            {
                tag: {
                    column: 23,
                    line: 13,
                    text: 'Error: Identifier not found "adsadasd"',
                    severity: 3,
                    file: 'output.pas',
                },
                text: '<source>(13,23) Error: Identifier not found "adsadasd"',
            },
        ]);
    });

    it('recognize fpc exiting error', () => {
        expect(
            utils.parseOutput('output.pas(17) Fatal: There were 1 errors compiling module, stopping', 'output.pas'),
        ).toEqual([
            {
                tag: {
                    column: 0,
                    line: 17,
                    text: 'Fatal: There were 1 errors compiling module, stopping',
                    severity: 3,
                    file: 'output.pas',
                },
                text: '<source>(17) Fatal: There were 1 errors compiling module, stopping',
            },
        ]);
    });

    it('removes the temp path', () => {
        expect(
            utils.parseOutput(
                'Compiling /tmp/path/prog.dpr\noutput.pas(17) Fatal: There were 1 errors compiling module, stopping',
                'output.pas',
                '/tmp/path/',
            ),
        ).toEqual([
            {
                text: 'Compiling prog.dpr',
            },
            {
                tag: {
                    column: 0,
                    line: 17,
                    text: 'Fatal: There were 1 errors compiling module, stopping',
                    severity: 3,
                    file: 'output.pas',
                },
                text: '<source>(17) Fatal: There were 1 errors compiling module, stopping',
            },
        ]);
    });
});

describe('Rust compiler output', () => {
    it('handles simple cases', () => {
        expect(utils.parseRustOutput('Line one\nLine two', 'bob.rs')).toEqual([{text: 'Line one'}, {text: 'Line two'}]);
        expect(utils.parseRustOutput('Unrelated\nLine one\n --> bob.rs:1\nUnrelated', 'bob.rs')).toEqual([
            {text: 'Unrelated'},
            {
                tag: {column: 0, line: 1, text: 'Line one', severity: 3},
                text: 'Line one',
            },
            {
                tag: {column: 0, line: 1, text: '', severity: 3},
                text: ' --> <source>:1',
            },
            {text: 'Unrelated'},
        ]);
        expect(utils.parseRustOutput('Line one\n --> bob.rs:1:5', 'bob.rs')).toEqual([
            {
                tag: {column: 5, line: 1, text: 'Line one', severity: 3},
                text: 'Line one',
            },
            {
                tag: {column: 5, line: 1, text: '', severity: 3},
                text: ' --> <source>:1:5',
            },
        ]);
        expect(utils.parseRustOutput('Multiple spaces\n   --> bob.rs:1:5', 'bob.rs')).toEqual([
            {
                tag: {column: 5, line: 1, text: 'Multiple spaces', severity: 3},
                text: 'Multiple spaces',
            },
            {
                tag: {column: 5, line: 1, text: '', severity: 3},
                text: '   --> <source>:1:5',
            },
        ]);
    });

    it('replaces all references to input source', () => {
        expect(utils.parseRustOutput('error: Error in bob.rs\n --> bob.rs:1', 'bob.rs')).toEqual([
            {
                tag: {column: 0, line: 1, text: 'error: Error in <source>', severity: 3},
                text: 'error: Error in <source>',
            },
            {
                tag: {column: 0, line: 1, text: '', severity: 3},
                text: ' --> <source>:1',
            },
        ]);
    });

    it('treats <stdin> as if it were the compiler source', () => {
        expect(utils.parseRustOutput('error: <stdin> is sad\n --> <stdin>:120:25', 'bob.rs')).toEqual([
            {
                tag: {column: 25, line: 120, text: 'error: <source> is sad', severity: 3},
                text: 'error: <source> is sad',
            },
            {
                tag: {column: 25, line: 120, text: '', severity: 3},
                text: ' --> <source>:120:25',
            },
        ]);
    });

    it('removes hyperlink escape sequences', () => {
        expect(
            utils.parseRustOutput(
                'error[\x1B]8;;https://doc.rust-lang.org/error_codes/E0425.html\x07E0425\x1B]8;;\x07]: cannot find value `x` in this scope\n --> <source>:42:27',
            ),
        ).toEqual([
            {
                tag: {
                    line: 42,
                    column: 27,
                    text: 'error[E0425]: cannot find value `x` in this scope',
                    severity: 3,
                },
                text: 'error[\x1B]8;;https://doc.rust-lang.org/error_codes/E0425.html\x07E0425\x1B]8;;\x07]: cannot find value `x` in this scope',
            },
            {
                tag: {
                    line: 42,
                    column: 27,
                    text: '',
                    severity: 3,
                },
                text: ' --> <source>:42:27',
            },
        ]);
    });
});

describe('Tool output', () => {
    it('removes the relative path', () => {
        expect(
            utils.parseOutput(
                './example.cpp:1:1: Fatal: There were 1 errors compiling module, stopping',
                './example.cpp',
            ),
        ).toEqual([
            {
                tag: {
                    column: 1,
                    line: 1,
                    text: 'Fatal: There were 1 errors compiling module, stopping',
                    severity: 3,
                    file: 'example.cpp',
                },
                text: '<source>:1:1: Fatal: There were 1 errors compiling module, stopping',
            },
        ]);
    });

    it('removes fortran relative path', () => {
        expect(
            utils.parseOutput("./example.f90:5:22: error: No explicit type declared for 'y'", './example.f90'),
        ).toEqual([
            {
                tag: {
                    column: 22,
                    line: 5,
                    text: "error: No explicit type declared for 'y'",
                    severity: 3,
                    file: 'example.f90',
                },
                text: "<source>:5:22: error: No explicit type declared for 'y'",
            },
        ]);
    });

    it('removes the jailed path', () => {
        expect(
            utils.parseOutput(
                '/home/ubuntu/example.cpp:1:1: Fatal: There were 1 errors compiling module, stopping',
                './example.cpp',
            ),
        ).toEqual([
            {
                tag: {
                    column: 1,
                    line: 1,
                    text: 'Fatal: There were 1 errors compiling module, stopping',
                    severity: 3,
                    file: 'example.cpp',
                },
                text: '<source>:1:1: Fatal: There were 1 errors compiling module, stopping',
            },
        ]);
    });
});

describe('Anonymizes all kind of IPs', () => {
    it('Ignores localhost', () => {
        expect(utils.anonymizeIp('localhost')).toEqual('localhost');
        expect(utils.anonymizeIp('localhost:42')).toEqual('localhost:42');
    });
    it('Removes last octet from IPv4 addresses', () => {
        expect(utils.anonymizeIp('127.0.0.0')).toEqual('127.0.0.0');
        expect(utils.anonymizeIp('127.0.0.10')).toEqual('127.0.0.0');
        expect(utils.anonymizeIp('127.0.0.255')).toEqual('127.0.0.0');
    });
    it('Removes last 3 hextets from IPv6 addresses', () => {
        // Not necessarily valid addresses, we're interested in the format
        expect(utils.anonymizeIp('ffff:aaaa:dead:beef')).toEqual('ffff:0:0:0');
        expect(utils.anonymizeIp('bad:c0de::')).toEqual('bad:0:0:0');
        expect(utils.anonymizeIp(':1d7e::c0fe')).toEqual(':0:0:0');
    });
});

describe('Logger functionality', () => {
    it('correctly logs streams split over lines', () => {
        const logs: {level: string; msg: string}[] = [];
        const fakeLog = {log: (level: string, msg: string) => logs.push({level, msg})} as any as winston.Logger;
        const infoStream = makeLogStream('info', fakeLog);
        infoStream.write('first\n');
        infoStream.write('part');
        infoStream.write('ial\n');
        expect(logs).toEqual([
            {
                level: 'info',
                msg: 'first',
            },
            {
                level: 'info',
                msg: 'partial',
            },
        ]);
    });
    it('correctly logs streams to the right destination', () => {
        const logs: {level: string; msg: string}[] = [];
        const fakeLog = {log: (level: string, msg: string) => logs.push({level, msg})} as any as winston.Logger;
        const infoStream = makeLogStream('warn', fakeLog);
        infoStream.write('ooh\n');
        expect(logs).toEqual([
            {
                level: 'warn',
                msg: 'ooh',
            },
        ]);
    });
});

describe('Hash interface', () => {
    it('correctly hashes strings', () => {
        const version = 'Compiler Explorer Tests Version 0';
        expect(utils.getHash('cream cheese', version)).toEqual(
            'cfff2d1f7a213e314a67cce8399160ae884f794a3ee9d4a01cd37a8c22c67d94',
        );
        expect(utils.getHash('large eggs', version)).toEqual(
            '9144dec50b8df5bc5cc24ba008823cafd6616faf2f268af84daf49ac1d24feb0',
        );
        expect(utils.getHash('sugar', version)).toEqual(
            'afa3c89d0f6a61de6805314c9bd7c52d020425a3a3c7bbdfa7c0daec594e5ef1',
        );
    });
    it('correctly hashes objects', () => {
        expect(
            utils.getHash({
                toppings: [
                    {name: 'raspberries', optional: false},
                    {name: 'ground cinnamon', optional: true},
                ],
            }),
        ).toEqual('e205d63abd5db363086621fdc62c4c23a51b733bac5855985a8b56642d570491');
    });
});

describe('GoldenLayout utils', () => {
    it('finds every editor & compiler', async () => {
        const state = await fs.readJson('test/example-states/default-state.json');
        const contents = utils.glGetMainContents(state.content);
        expect(contents).toEqual({
            editors: [
                {source: 'Editor 1', language: 'c++'},
                {source: 'Editor 2', language: 'c++'},
                {source: 'Editor 3', language: 'c++'},
                {source: 'Editor 4', language: 'c++'},
            ],
            compilers: [
                {compiler: 'clang_trunk'},
                {compiler: 'gsnapshot'},
                {compiler: 'clang_trunk'},
                {compiler: 'gsnapshot'},
                {compiler: 'rv32-clang'},
            ],
        });
    });
});

describe('squashes horizontal whitespace', () => {
    it('handles empty input', () => {
        expect(utils.squashHorizontalWhitespace('')).toEqual('');
        expect(utils.squashHorizontalWhitespace(' ')).toEqual('');
        expect(utils.squashHorizontalWhitespace('    ')).toEqual('');
    });
    it('handles leading spaces', () => {
        expect(utils.squashHorizontalWhitespace(' abc')).toEqual(' abc');
        expect(utils.squashHorizontalWhitespace('   abc')).toEqual('  abc');
        expect(utils.squashHorizontalWhitespace('       abc')).toEqual('  abc');
    });
    it('handles interline spaces', () => {
        expect(utils.squashHorizontalWhitespace('abc abc')).toEqual('abc abc');
        expect(utils.squashHorizontalWhitespace('abc   abc')).toEqual('abc abc');
        expect(utils.squashHorizontalWhitespace('abc     abc')).toEqual('abc abc');
    });
    it('handles leading and interline spaces', () => {
        expect(utils.squashHorizontalWhitespace(' abc  abc')).toEqual(' abc abc');
        expect(utils.squashHorizontalWhitespace('  abc abc')).toEqual('  abc abc');
        expect(utils.squashHorizontalWhitespace('  abc     abc')).toEqual('  abc abc');
        expect(utils.squashHorizontalWhitespace('    abc   abc')).toEqual('  abc abc');
    });
});

describe('encodes in our version of base32', () => {
    function doTest(original, expected) {
        expect(utils.base32Encode(Buffer.from(original))).toEqual(expected);
    }

    // Done by hand to check that they are valid

    it('works for empty strings', () => {
        doTest('', '');
    });

    it('works for lengths multiple of 5 bits', () => {
        doTest('aaaaa', '3Mn4ha7P');
    });

    it('works for lengths not multiple of 5 bits', () => {
        // 3
        doTest('a', '35');

        // 1
        doTest('aa', '3Mn1');

        // 4
        doTest('aaa', '3Mn48');

        // 2
        doTest('aaaa', '3Mn4ha3');
    });

    it('works for some random strings', () => {
        // I also calculated this ones so lets put them
        doTest('foo', '8rrx8');

        doTest('foobar', '8rrx8b7Pc5');
    });
});

describe('fileExists', () => {
    it('Returns true for files that exists', async () => {
        await expect(utils.fileExists(fileURLToPath(import.meta.url))).resolves.toBe(true);
    });
    it("Returns false for files that don't exist", async () => {
        await expect(utils.fileExists('./ABC-FileThatDoesNotExist.extension')).resolves.toBe(false);
    });
    it('Returns false for directories that exist', async () => {
        await expect(utils.fileExists(path.resolve(path.dirname(fileURLToPath(import.meta.url))))).resolves.toBe(false);
    });
});

describe('safe semver', () => {
    it('should understand most kinds of semvers', () => {
        expect(utils.asSafeVer('0')).toEqual('0.0.0');
        expect(utils.asSafeVer('1')).toEqual('1.0.0');

        expect(utils.asSafeVer('1.0')).toEqual('1.0.0');
        expect(utils.asSafeVer('1.1')).toEqual('1.1.0');

        expect(utils.asSafeVer('1.1.0')).toEqual('1.1.0');
        expect(utils.asSafeVer('1.1.1')).toEqual('1.1.1');

        expect(utils.asSafeVer('trunk')).toEqual(utils.magic_semver.trunk);
        expect(utils.asSafeVer('(trunk)')).toEqual(utils.magic_semver.trunk);
        expect(utils.asSafeVer('(123.456.789 test)')).toEqual(utils.magic_semver.non_trunk);

        expect(utils.asSafeVer('0..0')).toEqual(utils.magic_semver.non_trunk);
        expect(utils.asSafeVer('0.0.')).toEqual(utils.magic_semver.non_trunk);
        expect(utils.asSafeVer('0.')).toEqual(utils.magic_semver.non_trunk);
        expect(utils.asSafeVer('.0.0')).toEqual(utils.magic_semver.non_trunk);
        expect(utils.asSafeVer('.0..')).toEqual(utils.magic_semver.non_trunk);
        expect(utils.asSafeVer('0..')).toEqual(utils.magic_semver.non_trunk);

        expect(utils.asSafeVer('123 TEXT')).toEqual('123.0.0');
        expect(utils.asSafeVer('123.456 TEXT')).toEqual('123.456.0');
        expect(utils.asSafeVer('123.456.789 TEXT')).toEqual('123.456.789');
    });
});
