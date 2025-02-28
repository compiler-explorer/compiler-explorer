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

import {describe, expect, it} from 'vitest';

import {Filter} from '../static/ansi-to-html.js';

describe('ansi-to-html', () => {
    const filterOpts = {
        fg: '#333333',
        bg: '#f5f5f5',
        stream: true,
        escapeXML: true,
    };
    it('Should leave non-ansi colours alone', () => {
        const filter = new Filter(filterOpts);
        expect(filter.toHtml('I am a boring old string')).toEqual('I am a boring old string');
    });
    it('Should handle simple cases', () => {
        const filter = new Filter(filterOpts);
        expect(filter.toHtml('\x1B[38;5;99mTest')).toEqual('<span style="color:#875fff">Test</span>');
    });
    it('Should handle nasty edge cases', () => {
        const filter = new Filter(filterOpts);
        // See #1666, this used to cause catastrophic backtracking.
        expect(
            filter.toHtml(
                '\x1B[38;5;9999999999999999999999999999999999999999999999999999999999999999999999999999999' +
                    '99999999999999999999"mTest',
            ),
        ).toEqual(
            '5;9999999999999999999999999999999999999999999999999999999999999' +
                '99999999999999999999999999999999999999"mTest',
        );
    });

    // With thanks to https://github.com/rburns/ansi-to-html/pull/84/files
    it('renders xterm foreground 256 sequences', () => {
        const filter = new Filter(filterOpts);
        expect(filter.toHtml('\x1B[38;5;196mhello')).toEqual('<span style="color:#ff0000">hello</span>');
    });
    it('renders xterm background 256 sequences', () => {
        const filter = new Filter(filterOpts);
        expect(filter.toHtml('\x1B[48;5;196mhello')).toEqual('<span style="background-color:#ff0000">hello</span>');
    });

    it('should ignore reverse video', () => {
        const filter = new Filter(filterOpts);
        expect(filter.toHtml('\x1B[7mhello')).toEqual('hello');
    });

    // tests for #3659
    it('should stream', () => {
        const filter = new Filter(filterOpts);
        filter.toHtml('\x1B[38;5;99mfoo');
        expect(filter.toHtml('bar')).toEqual('<span style="color:#875fff">bar</span>');
    });
    it('should handle stream reset', () => {
        const filter = new Filter(filterOpts);
        filter.toHtml('\x1B[38;5;99mfoo');
        filter.reset();
        expect(filter.toHtml('bar')).toEqual('bar');
    });

    // rgb test
    it('should process rgb colors', () => {
        const filter = new Filter(filterOpts);
        expect(filter.toHtml('\x1B[38;2;57;170;243mfoo\x1B[48;2;100;100;100mbar')).toEqual(
            '<span style="color:#39aaf3">foo<span style="background-color:#646464">bar</span></span>',
        );
    });

    // hyperlinks
    it('should parse terminal hyperlinks', () => {
        const filter = new Filter(filterOpts);
        expect(
            filter.toHtml(
                'error[\x1B]8;;https://doc.rust-lang.org/error_codes/E0425.html\x07E0425\x1B]8;;\x07]: cannot find value `x` in this scope',
            ),
        ).toEqual(
            'error[<a class="diagnostic-url" target="_blank" rel="noreferrer" href=https://doc.rust-lang.org/error_codes/E0425.html>E0425</a>]: cannot find value `x` in this scope',
        );
        expect(
            filter.toHtml(
                't.c:3:1: warning: control reaches end of non-void function [\x1B]8;;https://gcc.gnu.org/onlinedocs/gcc-14.2.0/gcc/Warning-Options.html#index-Wno-return-type\x1B\\-Wreturn-type\x1B]8;;\x1B\\]',
            ),
        ).toEqual(
            't.c:3:1: warning: control reaches end of non-void function [<a class="diagnostic-url" target="_blank" rel="noreferrer" href=https://gcc.gnu.org/onlinedocs/gcc-14.2.0/gcc/Warning-Options.html#index-Wno-return-type>-Wreturn-type</a>]',
        );
    });

    it('should properly escape hyperlinks', () => {
        const filter = new Filter(filterOpts);
        expect(
            filter.toHtml('\x1B]8;;https://example.org/</a><b>bold</b><a>link\x07<i>italic</i>\x1B]8;;\x07'),
        ).toEqual(
            '<a class="diagnostic-url" target="_blank" rel="noreferrer" href=https://example.org/%3C/a%3E%3Cb%3Ebold%3C/b%3E%3Ca%3Elink>&lt;i&gt;italic&lt;/i&gt;</a>',
        );
    });
});
