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

import {beforeAll, describe, expect, it} from 'vitest';

import {RacketPassDumpParser} from '../lib/parsers/racket-pass-dump-parser.js';
import * as properties from '../lib/properties.js';

const languages = {
    racket: {id: 'racket'},
};

function deepCopy(obj) {
    return JSON.parse(JSON.stringify(obj));
}

describe('racket-pass-dump-parser', () => {
    let racketPassDumpParser;

    beforeAll(() => {
        const fakeProps = new properties.CompilerProps(languages, properties.fakeProps({}));
        const compilerProps = (fakeProps.get as any).bind(fakeProps, 'racket');
        racketPassDumpParser = new RacketPassDumpParser(compilerProps);
    });

    it('should recognize step', () => {
        const output = [
            {text: ';; compile-linklet: phase: 0'},
            {text: ';; compile-linklet: module: example'},
            {text: ';; compile-linklet: name: module'},
            {text: ';; compile-linklet: step: linklet'},
            {text: ';; ---------------------'},
            {text: '(linklet ((.get-syntax-literal!) (.set-transformer!)) (square)'},
            {text: '  (void)'},
            {text: '  (define-values (square)'},
            {text: '    (#%name square (lambda (num_1) (* num_1 num_1))))'},
            {text: '  (void))'},
        ];

        const brokenDown = racketPassDumpParser.breakdownOutputIntoPassDumps(deepCopy(output), {});

        expect(brokenDown).toEqual([
            {
                group: 'module: example, linklet: module, phase: 0',
                header: 'linklet',
                lines: output.slice(5),
                machine: false,
            },
        ]);
    });

    it('should recognize pass', () => {
        const output = [
            {text: ';; compile-linklet: module: (phases configure-runtime)'},
            {text: ';; compile-linklet: name: decl'},
            {text: ';; compile-linklet: passes: all'},
            {text: ';; ---------------------'},
            {text: 'output of cpnanopass:'},
            {text: '(case-lambda          '},
            {text: '    [clause'},
            {text: '    ()'},
            {text: '    0'},
            {text: '    (case-lambda'},
            {text: '        [clause'},
            {text: '        (instance-variable-reference.22'},
            {text: '        .get-syntax-literal!1.23'},
            {text: '        .set-transformer!2.24'},
            {text: '        configure3.25)'},
            {text: '        4'},
            {text: '        (begin'},
            {text: '        ((#[primref.a0xltlrcpeygsahopkplcn-2 $top-level-value 263393 #f]'},
            {text: "            '1/print-as-expression.rkt-io.sls-1/print-as-expression-0)"},
            {text: "            '#t)"},
            {text: "        '#<void>)])])"},
        ];

        const brokenDown = racketPassDumpParser.breakdownOutputIntoPassDumps(deepCopy(output), {});

        expect(brokenDown).toEqual([
            {
                group: 'module: (phases configure-runtime), linklet: decl',
                header: 'cpnanopass',
                lines: output.slice(5),
                machine: false,
            },
        ]);
    });
});
