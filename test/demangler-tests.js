// Copyright (c) 2018, Compiler Explorer Authors
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

import { CppDemangler, Win32Demangler } from '../lib/demangler';
import * as exec from '../lib/exec';
import { SymbolStore } from '../lib/symbol-store';
import * as utils from '../lib/utils';

import { fs, path, resolvePathFromTestRoot } from './utils';

const cppfiltpath = 'c++filt';

class DummyCompiler {
    exec(command, args, options) {
        return exec.execute(command, args, options);
    }
}

const catchCppfiltNonexistence = err => {
    if (!err.message.startsWith('spawn c++filt')) {
        throw err;
    }
};

describe('Basic demangling', function () {
    it('One line of asm', function () {
        const result = {
            asm: [{text: 'Hello, World!'}],
        };

        const demangler = new CppDemangler(cppfiltpath, new DummyCompiler());
        demangler.demanglerArguments = ['-n'];

        return Promise.all([
            demangler.process(result).then((output) => {
                output.asm[0].text.should.equal('Hello, World!');
            }),
        ]);
    });

    it('One label and some asm', function () {
        const result = {};
        result.asm = [
            {text: '_Z6squarei:'},
            {text: '  ret'},
        ];

        const demangler = new CppDemangler(cppfiltpath, new DummyCompiler());
        demangler.demanglerArguments = ['-n'];

        return Promise.all([
            demangler.process(result)
                .then((output) => {
                    output.asm[0].text.should.equal('square(int):');
                    output.asm[1].text.should.equal('  ret');
                })
                .catch(catchCppfiltNonexistence),
        ]);
    });

    it('One label and use of a label', function () {
        const result = {};
        result.asm = [
            {text: '_Z6squarei:'},
            {text: '  mov eax, $_Z6squarei'},
        ];

        const demangler = new CppDemangler(cppfiltpath, new DummyCompiler());
        demangler.demanglerArguments = ['-n'];

        return Promise.all([
            demangler.process(result).then((output) => {
                output.asm[0].text.should.equal('square(int):');
                output.asm[1].text.should.equal('  mov eax, $square(int)');
            })
                .catch(catchCppfiltNonexistence),
        ]);
    });

    it('Two destructors', function () {
        const result = {};
        result.asm = [
            {text: '_ZN6NormalD0Ev:'},
            {text: '  callq _ZdlPv'},
            {text: '_Z7caller1v:'},
            {text: '  rep ret'},
            {text: '_Z7caller2P6Normal:'},
            {text: '  cmp rax, OFFSET FLAT:_ZN6NormalD0Ev'},
            {text: '  jmp _ZdlPvm'},
            {text: '_ZN6NormalD2Ev:'},
            {text: '  rep ret'},
        ];

        const demangler = new CppDemangler(cppfiltpath, new DummyCompiler());
        demangler.demanglerArguments = ['-n'];

        return demangler.process(result)
            .then((output) => {
                output.asm[0].text.should.equal('Normal::~Normal() [deleting destructor]:');
                output.asm[1].text.should.equal('  callq operator delete(void*)');
                output.asm[6].text.should.equal('  jmp operator delete(void*, unsigned long)');
            })
            .catch(catchCppfiltNonexistence);
    });

    it('Should ignore comments (CL)', function () {
        const result = {};
        result.asm = [
            {text: '        call     ??3@YAXPEAX_K@Z                ; operator delete'},
        ];

        const demangler = new Win32Demangler(cppfiltpath, new DummyCompiler());
        demangler.result = result;
        demangler.symbolstore = new SymbolStore();
        demangler.collectLabels();

        const output = demangler.win32RawSymbols;
        output.should.deep.equal(
            [
                '??3@YAXPEAX_K@Z',
            ],
        );
    });

    it('Should ignore comments (CPP)', function () {
        const result = {};
        result.asm = [
            {text: '        call     hello                ; operator delete'},
        ];

        const demangler = new CppDemangler(cppfiltpath, new DummyCompiler());
        demangler.demanglerArguments = ['-n'];
        demangler.result = result;
        demangler.symbolstore = new SymbolStore();
        demangler.collectLabels();

        const output = demangler.othersymbols.listSymbols();
        output.should.deep.equal(
            [
                'hello',
            ],
        );
    });

    it('Should also support ARM branch instructions', () => {
        const result = {};
        result.asm = [
            {text: '   bl _ZN3FooC1Ev'},
        ];

        const demangler = new CppDemangler(cppfiltpath, new DummyCompiler());
        demangler.demanglerArguments = ['-n'];
        demangler.result = result;
        demangler.symbolstore = new SymbolStore();
        demangler.collectLabels();

        const output = demangler.othersymbols.listSymbols();
        output.should.deep.equal(
            [
                '_ZN3FooC1Ev',
            ],
        );
    });

    it('Should NOT handle undecorated labels', () => {
        const result = {};
        result.asm = [
            {text: '$LN3@caller2:'},
        ];

        const demangler = new Win32Demangler(cppfiltpath, new DummyCompiler());
        demangler.result = result;
        demangler.symbolstore = new SymbolStore();
        demangler.collectLabels();

        const output = demangler.win32RawSymbols;
        output.should.deep.equal(
            [],
        );
    });

    it('Should ignore comments after jmps', function () {
        const result = {};
        result.asm = [
            {text: '  jmp _Z1fP6mytype # TAILCALL'},
        ];

        const demangler = new CppDemangler(cppfiltpath, new DummyCompiler());
        demangler.demanglerArguments = ['-n'];
        demangler.result = result;
        demangler.symbolstore = new SymbolStore();
        demangler.collectLabels();

        const output = demangler.othersymbols.listSymbols();
        output.should.deep.equal(
            [
                '_Z1fP6mytype',
            ],
        );
    });

    it('Should still work with normal jmps', function () {
        const result = {};
        result.asm = [
            {text: '  jmp _Z1fP6mytype'},
        ];

        const demangler = new CppDemangler(cppfiltpath, new DummyCompiler());
        demangler.demanglerArguments = ['-n'];
        demangler.result = result;
        demangler.symbolstore = new SymbolStore();
        demangler.collectLabels();

        const output = demangler.othersymbols.listSymbols();
        output.should.deep.equal(
            [
                '_Z1fP6mytype',
            ],
        );
    });
});

async function readResultFile(filename) {
    const data = await fs.readFile(filename);
    const asm = utils.splitLines(data.toString()).map(line => {
        return {text: line};
    });

    return {asm};
}

async function DoDemangleTest(filename) {
    const resultIn = await readResultFile(filename);
    const resultOut = await readResultFile(filename + '.demangle');

    const demangler = new CppDemangler(cppfiltpath, new DummyCompiler());
    demangler.demanglerArguments = ['-n'];
    await demangler.process(resultIn).should.eventually.deep.equal(resultOut);
}

describe('File demangling', () => {
    const testcasespath = resolvePathFromTestRoot('demangle-cases');

    /*
     * NB: this readdir must *NOT* be async
     *
     * Mocha calls the function passed to `describe` synchronously
     * and expects the test suite to be fully configured upon return.
     *
     * If you pass an async function to describe and setup test cases
     * after an await there is no guarantee they will be found, and
     * if they are they will not end up in the expected suite.
     */
    const files = fs.readdirSync(testcasespath);

    for (const filename of files) {
        if (filename.endsWith('.asm')) {
            it(filename, async () => {
                await DoDemangleTest(path.join(testcasespath, filename));
            });
        }
    }
});
