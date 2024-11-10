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

import {describe, expect, it} from 'vitest';

import {unwrap} from '../lib/assert.js';
import {BaseCompiler} from '../lib/base-compiler.js';
import {CompilationEnvironment} from '../lib/compilation-env.js';
import {CppDemangler, Win32Demangler} from '../lib/demangler/index.js';
import {LLVMIRDemangler} from '../lib/demangler/llvm.js';
import {PrefixTree} from '../lib/demangler/prefix-tree.js';
import * as exec from '../lib/exec.js';
import * as properties from '../lib/properties.js';
import {SymbolStore} from '../lib/symbol-store.js';
import * as utils from '../lib/utils.js';

import {fs, makeFakeCompilerInfo, path, resolvePathFromTestRoot} from './utils.js';

const cppfiltpath = 'c++filt';

class DummyCompiler extends BaseCompiler {
    constructor() {
        const env = {
            ceProps: properties.fakeProps({}),
            getCompilerPropsForLanguage: () => {
                return (prop, def) => def;
            },
        } as unknown as CompilationEnvironment;

        // using c++ as the compiler needs at least one language
        const compiler = makeFakeCompilerInfo({lang: 'c++'});

        super(compiler, env);
    }
    override exec(command, args, options) {
        return exec.execute(command, args, options);
    }
}

class DummyCppDemangler extends CppDemangler {
    public override collectLabels = super.collectLabels;
}

class DummyLlvmDemangler extends LLVMIRDemangler {
    public override collectLabels = super.collectLabels;
}

class DummyWin32Demangler extends Win32Demangler {
    public override collectLabels = super.collectLabels;
}

const catchCppfiltNonexistence = err => {
    if (!err.message.startsWith('spawn c++filt')) {
        throw err;
    }
};

describe('Basic demangling', () => {
    it('One line of asm', () => {
        const result = {
            asm: [{text: 'Hello, World!'}],
        };

        const demangler = new DummyCppDemangler(cppfiltpath, new DummyCompiler(), ['-n']);

        return Promise.all([
            demangler.process(result).then(output => {
                expect(output.asm[0].text).toEqual('Hello, World!');
            }),
        ]);
    });

    it('One label and some asm', () => {
        const result = {asm: [{text: '_Z6squarei:'}, {text: '  ret'}]};

        const demangler = new DummyCppDemangler(cppfiltpath, new DummyCompiler(), ['-n']);

        return Promise.all([
            demangler
                .process(result)
                .then(output => {
                    expect(output.asm[0].text).toEqual('square(int):');
                    expect(output.asm[1].text).toEqual('  ret');
                })
                .catch(catchCppfiltNonexistence),
        ]);
    });

    it('One label and use of a label', () => {
        const result = {asm: [{text: '_Z6squarei:'}, {text: '  mov eax, $_Z6squarei'}]};

        const demangler = new DummyCppDemangler(cppfiltpath, new DummyCompiler(), ['-n']);

        return Promise.all([
            demangler
                .process(result)
                .then(output => {
                    expect(output.asm[0].text).toEqual('square(int):');
                    expect(output.asm[1].text).toEqual('  mov eax, $square(int)');
                })
                .catch(catchCppfiltNonexistence),
        ]);
    });

    it('Mov with OFFSET FLAT', () => {
        // regression test for https://github.com/compiler-explorer/compiler-explorer/issues/6348
        const result = {asm: [{text: 'mov     eax, OFFSET FLAT:_ZN1a1gEi'}]};

        const demangler = new DummyCppDemangler(cppfiltpath, new DummyCompiler(), ['-n']);

        return Promise.all([
            demangler
                .process(result)
                .then(output => {
                    expect(output.asm[0].text).toEqual('mov     eax, OFFSET FLAT:a::g(int)');
                })
                .catch(catchCppfiltNonexistence),
        ]);
    });

    it('rip-relative jump', () => {
        // regression test for https://github.com/compiler-explorer/compiler-explorer/issues/6348
        const result = {
            asm: [
                {
                    text: 'jmp     qword ptr [rip + _ZN4core3fmt3num3imp54_$LT$impl$u20$core..fmt..Display$u20$for$u20$usize$GT$3fmt17h7bbbd896a38dcccaE@GOTPCREL]',
                },
            ],
        };

        const demangler = new DummyCppDemangler(cppfiltpath, new DummyCompiler(), ['-n']);

        return Promise.all([
            demangler
                .process(result)
                .then(output => {
                    if (process.platform === 'win32') {
                        expect(output.asm[0].text).toEqual(
                            'jmp     qword ptr [rip + core::fmt::num::imp::<impl core::fmt::Display for usize>::fmt@GOTPCREL]',
                        );
                    } else {
                        expect(output.asm[0].text).toEqual(
                            'jmp     qword ptr [rip + core::fmt::num::imp::<impl core::fmt::Display for usize>::fmt::h7bbbd896a38dccca@GOTPCREL]',
                        );
                    }
                })
                .catch(catchCppfiltNonexistence),
        ]);
    });

    it('Two destructors', () => {
        const result = {
            asm: [
                {text: '_ZN6NormalD0Ev:'},
                {text: '  callq _ZdlPv'},
                {text: '_Z7caller1v:'},
                {text: '  rep ret'},
                {text: '_Z7caller2P6Normal:'},
                {text: '  cmp rax, OFFSET FLAT:_ZN6NormalD0Ev'},
                {text: '  jmp _ZdlPvm'},
                {text: '_ZN6NormalD2Ev:'},
                {text: '  rep ret'},
            ],
        };

        const demangler = new DummyCppDemangler(cppfiltpath, new DummyCompiler(), ['-n']);

        return demangler
            .process(result)
            .then(output => {
                expect(output.asm[0].text).toEqual('Normal::~Normal() [deleting destructor]:');
                expect(output.asm[1].text).toEqual('  callq operator delete(void*)');
                expect(output.asm[6].text).toEqual('  jmp operator delete(void*, unsigned long)');
            })
            .catch(catchCppfiltNonexistence);
    });

    it('Should ignore comments (CL)', () => {
        const result = {asm: [{text: '        call     ??3@YAXPEAX_K@Z                ; operator delete'}]};

        const demangler = new DummyWin32Demangler(cppfiltpath, new DummyCompiler());
        demangler.result = result;
        demangler.symbolstore = new SymbolStore();
        demangler.collectLabels();

        const output = demangler.win32RawSymbols;
        expect(unwrap(output)).toEqual(['??3@YAXPEAX_K@Z']);
    });

    it('Should ignore comments (CPP)', () => {
        const result = {asm: [{text: '        call     hello                ; operator delete'}]};

        const demangler = new DummyCppDemangler(cppfiltpath, new DummyCompiler(), ['-n']);

        demangler.result = result;
        demangler.symbolstore = new SymbolStore();
        demangler.collectLabels();

        const output = demangler.othersymbols.listSymbols();
        expect(output).toEqual(['hello']);
    });

    it('Should also support ARM branch instructions', () => {
        const result = {asm: [{text: '   bl _ZN3FooC1Ev'}]};

        const demangler = new DummyCppDemangler(cppfiltpath, new DummyCompiler(), ['-n']);

        demangler.result = result;
        demangler.symbolstore = new SymbolStore();
        demangler.collectLabels();

        const output = demangler.othersymbols.listSymbols();
        expect(output).toEqual(['_ZN3FooC1Ev']);
    });

    it('Should NOT handle undecorated labels', () => {
        const result = {asm: [{text: '$LN3@caller2:'}]};

        const demangler = new DummyWin32Demangler(cppfiltpath, new DummyCompiler());
        demangler.result = result;
        demangler.symbolstore = new SymbolStore();
        demangler.collectLabels();

        const output = demangler.win32RawSymbols;
        expect(output).toEqual([]);
    });

    it('Should ignore comments after jmps', () => {
        const result = {asm: [{text: '  jmp _Z1fP6mytype # TAILCALL'}]};

        const demangler = new DummyCppDemangler(cppfiltpath, new DummyCompiler(), ['-n']);

        demangler.result = result;
        demangler.symbolstore = new SymbolStore();
        demangler.collectLabels();

        const output = demangler.othersymbols.listSymbols();
        expect(output).toEqual(['_Z1fP6mytype']);
    });

    it('Should still work with normal jmps', () => {
        const result = {asm: [{text: '  jmp _Z1fP6mytype'}]};

        const demangler = new DummyCppDemangler(cppfiltpath, new DummyCompiler(), ['-n']);

        demangler.result = result;
        demangler.symbolstore = new SymbolStore();
        demangler.collectLabels();

        const output = demangler.othersymbols.listSymbols();
        expect(output).toEqual(['_Z1fP6mytype']);
    });

    it('Should support CUDA PTX', () => {
        const result = {
            asm: [
                {text: '  .visible .entry _Z6squarePii('},
                {text: '  .param .u64 _Z6squarePii_param_0,'},
                {text: '  ld.param.u64    %rd1, [_Z6squarePii_param_0];'},
                {text: '  .func  (.param .b32 func_retval0) _Z4cubePii('},
                {text: '.global .attribute(.managed) .align 4 .b8 _ZN2ns9mymanagedE[16];'},
                {text: '.global .texref _ZN2ns6texRefE;'},
                {text: '.const .align 8 .u64 _ZN2ns5mystrE = generic($str);'},
            ],
        };

        const demangler = new DummyCppDemangler(cppfiltpath, new DummyCompiler(), ['-n']);

        return Promise.all([
            demangler
                .process(result)
                .then(output => {
                    expect(output.asm[0].text).toEqual('  .visible .entry square(int*, int)(');
                    expect(output.asm[1].text).toEqual('  .param .u64 square(int*, int)_param_0,');
                    expect(output.asm[2].text).toEqual('  ld.param.u64    %rd1, [square(int*, int)_param_0];');
                    expect(output.asm[3].text).toEqual('  .func  (.param .b32 func_retval0) cube(int*, int)(');
                    expect(output.asm[4].text).toEqual('.global .attribute(.managed) .align 4 .b8 ns::mymanaged[16];');
                    expect(output.asm[5].text).toEqual('.global .texref ns::texRef;');
                    expect(output.asm[6].text).toEqual('.const .align 8 .u64 ns::mystr = generic($str);');
                })
                .catch(catchCppfiltNonexistence),
        ]);
    });
});

async function readResultFile(filename: string) {
    const data = await fs.readFile(filename);
    const asm = utils.splitLines(data.toString()).map(line => {
        return {text: line};
    });

    return {asm};
}

async function DoDemangleTest(filename: string) {
    const resultIn = await readResultFile(filename);
    const resultOut = await readResultFile(filename + '.demangle');

    const demangler = new DummyCppDemangler(cppfiltpath, new DummyCompiler(), ['-n']);

    await expect(demangler.process(resultIn)).resolves.toEqual(resultOut);
}

if (process.platform === 'linux') {
    describe('File demangling', () => {
        const testcasespath = resolvePathFromTestRoot('demangle-cases');

        // For backwards compatability reasons, we have a sync readdir here. For details, see
        // the git blame of this file.
        // TODO: Consider replacing with https://github.com/vitest-dev/vitest/issues/703
        const files = fs.readdirSync(testcasespath);

        for (const filename of files) {
            if (filename.endsWith('.asm')) {
                it(filename, async () => {
                    await DoDemangleTest(path.join(testcasespath, filename));
                });
            }
        }
    });
}

describe('Demangler prefix tree', () => {
    const replacements = new PrefixTree([]);
    replacements.add('a', 'short_a');
    replacements.add('aa', 'long_a');
    replacements.add('aa_shouldnotmatch', 'ERROR');
    it('should replace a short match', () => {
        expect(replacements.replaceAll('a').newText).toEqual('short_a');
    });
    it('should replace using the longest match', () => {
        expect(replacements.replaceAll('aa').newText).toEqual('long_a');
    });
    it('should replace using both', () => {
        expect(replacements.replaceAll('aaa').newText).toEqual('long_ashort_a');
    });
    it('should replace using both', () => {
        expect(replacements.replaceAll('a aa a aa').newText).toEqual('short_a long_a short_a long_a');
    });
    it('should work with empty replacements', () => {
        expect(new PrefixTree([]).replaceAll('Testing 123').newText).toEqual('Testing 123');
    });
    it('should leave unmatching text alone', () => {
        expect(
            replacements.replaceAll('Some text with none of the first letter of the ordered letter list').newText,
        ).toEqual('Some text with none of the first letter of the ordered letter list');
    });
    it('should handle a mixture', () => {
        expect(replacements.replaceAll('Everyone loves an aardvark').newText).toEqual(
            'Everyone loves short_an long_ardvshort_ark',
        );
    });
    it('should find exact matches', () => {
        expect(unwrap(replacements.findExact('a'))).toEqual('short_a');
        expect(unwrap(replacements.findExact('aa'))).toEqual('long_a');
        expect(unwrap(replacements.findExact('aa_shouldnotmatch'))).toEqual('ERROR');
    });
    it('should find not find mismatches', () => {
        expect(replacements.findExact('aaa')).toBeNull();
        expect(replacements.findExact(' aa')).toBeNull();
        expect(replacements.findExact(' a')).toBeNull();
        expect(replacements.findExact('Oh noes')).toBeNull();
        expect(replacements.findExact('')).toBeNull();
    });
});

// FIXME: The `c++filt` installed on `windows-2019` runners is so old that it produces
// different output, so we skip this test on Windows for now.
describe.skipIf(process.platform === 'win32')('LLVM IR demangler', () => {
    it('demangles normal identifiers', () => {
        const result = {
            asm: [
                {text: 'define dso_local noundef i32 @_Z6squarei(i32 noundef %num)'},
                {text: 'define i32 @_ZN7example6square17hf2a64558a18ed1c1E(i32 %num) unnamed_addr'},
            ],
        };

        const demangler = new DummyLlvmDemangler(cppfiltpath, new DummyCompiler(), ['-n']);

        return Promise.all([
            demangler
                .process(result)
                .then(output => {
                    expect(output.asm[0].text).toEqual('define dso_local noundef i32 @square(int)(i32 noundef %num)');
                    expect(output.asm[1].text).toEqual(
                        'define i32 @example::square::hf2a64558a18ed1c1(i32 %num) unnamed_addr',
                    );
                })
                .catch(catchCppfiltNonexistence),
        ]);
    });

    it('demangles quoted identifiers', () => {
        const result = {
            asm: [
                {
                    text: '  invoke void @"_ZN4core3ptr53drop_in_place$LT$alloc..raw_vec..RawVec$LT$u8$GT$$GT$17h2e3e5a8e7287bb5aE"(ptr align 8 %_1) #17',
                },
            ],
        };

        const demangler = new DummyLlvmDemangler(cppfiltpath, new DummyCompiler(), ['-n']);

        return Promise.all([
            demangler
                .process(result)
                .then(output => {
                    expect(output.asm[0].text).toEqual(
                        '  invoke void @"core::ptr::drop_in_place<alloc::raw_vec::RawVec<u8>>::h2e3e5a8e7287bb5a"(ptr align 8 %_1) #17',
                    );
                })
                .catch(catchCppfiltNonexistence),
        ]);
    });
});
