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

import * as fsSync from 'node:fs';
import fs from 'node:fs/promises';
import path from 'node:path';

import {afterEach, describe, expect, it, vi} from 'vitest';

import {unwrap} from '../lib/assert.js';
import {BaseCompiler} from '../lib/base-compiler.js';
import {CFG, generateStructure} from '../lib/cfg/cfg.js';
import {CompilationEnvironment} from '../lib/compilation-env.js';
import {CppDemangler, LLVMWin32Demangler, Win32Demangler} from '../lib/demangler/index.js';
import {LLVMIRDemangler} from '../lib/demangler/llvm.js';
import {PrefixTree} from '../lib/demangler/prefix-tree.js';
import * as exec from '../lib/exec.js';
import * as properties from '../lib/properties.js';
import {SymbolStore} from '../lib/symbol-store.js';
import * as utils from '../lib/utils.js';
import {makeFakeCompilerInfo, processAsm, resolvePathFromTestRoot} from './utils.js';

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
        const compiler = makeFakeCompilerInfo({lang: 'c++', exe: 'gcc'});

        super(compiler, env);
    }
    override exec(command, args, options) {
        return exec.execute(command, args, options);
    }
}

class DummyCppDemangler extends CppDemangler {
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

    it('One quoted label and some asm', () => {
        const result = {asm: [{text: '"_Z6squarei":'}, {text: '  ret'}]};

        const demangler = new DummyCppDemangler(cppfiltpath, new DummyCompiler(), ['-n']);

        return Promise.all([
            demangler
                .process(result)
                .then(output => {
                    expect(output.asm[0].text).toEqual('"square(int)":');
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

    it('One quoted label and use of a label', () => {
        const result = {asm: [{text: '"_Z6squarei":'}, {text: '  mov eax, $"_Z6squarei"'}]};

        const demangler = new DummyCppDemangler(cppfiltpath, new DummyCompiler(), ['-n']);

        return Promise.all([
            demangler
                .process(result)
                .then(output => {
                    expect(output.asm[0].text).toEqual('"square(int)":');
                    expect(output.asm[1].text).toEqual('  mov eax, $"square(int)"');
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
                    expect(output.asm[0].text).toEqual(
                        'jmp     qword ptr [rip + core::fmt::num::imp::<impl core::fmt::Display for usize>::fmt::h7bbbd896a38dccca@GOTPCREL]',
                    );
                })
                .catch(catchCppfiltNonexistence),
        ]);
    });

    it('AArch64 branch with dotted symbol', () => {
        const result = {
            asm: [
                {
                    text: 'b       _ZN4core3fmt3num3imp52_$LT$impl$u20$core..fmt..Display$u20$for$u20$i32$GT$3fmt17h0feee90717706137E',
                },
            ],
        };

        const demangler = new DummyCppDemangler(cppfiltpath, new DummyCompiler(), ['-n']);

        return Promise.all([
            demangler
                .process(result)
                .then(output => {
                    expect(output.asm[0].text).toEqual(
                        'b       core::fmt::num::imp::<impl core::fmt::Display for i32>::fmt::h0feee90717706137',
                    );
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
        const result = {
            asm: [
                {text: '   bl _ZN3FooC1Ev'},
                {
                    text: 'b       _ZN4core3fmt3num3imp52_$LT$impl$u20$core..fmt..Display$u20$for$u20$i32$GT$3fmt17h0feee90717706137E',
                },
            ],
        };

        const demangler = new DummyCppDemangler(cppfiltpath, new DummyCompiler(), ['-n']);

        demangler.result = result;
        demangler.symbolstore = new SymbolStore();
        demangler.collectLabels();

        const output = demangler.othersymbols.listSymbols();
        expect(output.sort()).toEqual(
            [
                '_ZN3FooC1Ev',
                '_ZN4core3fmt3num3imp52_$LT$impl$u20$core..fmt..Display$u20$for$u20$i32$GT$3fmt17h0feee90717706137E',
            ].sort(),
        );
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
    const asm = utils.splitLines(await fs.readFile(filename, 'utf-8')).map(line => {
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

async function DoDemangleTestWithLabels(filename: string) {
    const asm = processAsm(filename, {labels: true});
    delete asm.parsingTime;
    delete asm.filteredCount;

    const demangler = new DummyCppDemangler(cppfiltpath, new DummyCompiler(), ['-n']);
    await expect(demangler.process(asm)).resolves.toMatchFileSnapshot(filename + '.json');
}

if (process.platform === 'linux') {
    describe('File demangling', () => {
        const testcasespath = resolvePathFromTestRoot('demangle-cases');

        // For backwards compatability reasons, we have a sync readdir here. For details, see
        // the git blame of this file.
        // TODO: Consider replacing with https://github.com/vitest-dev/vitest/issues/703
        const files = fsSync.readdirSync(testcasespath);

        for (const filename of files) {
            if (filename.endsWith('.asm')) {
                it(filename, async () => {
                    await DoDemangleTest(path.join(testcasespath, filename));
                });
                if (filename !== 'bug-1336-first-20000-lines.asm') {
                    it(`demangles ${filename} with labels`, async () => {
                        await DoDemangleTestWithLabels(path.join(testcasespath, filename));
                    });
                }
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
    it('should match replaceAllText to replaceAll newText', () => {
        expect(replacements.replaceAllText('aaa')).toEqual(replacements.replaceAll('aaa').newText);
        expect(replacements.replaceAllText('Everyone loves an aardvark')).toEqual(
            replacements.replaceAll('Everyone loves an aardvark').newText,
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
    it('should only replace whole identifiers when given identifier characters', () => {
        const tree = new PrefixTree(
            [
                ['_Z3bari', 'bar(int)'],
                ['_Z3bari.cold', 'bar(int) [clone .cold]'],
            ],
            /[-\w$.]/,
        );
        expect(tree.replaceAll('call @_Z3bari(i32 %x)').newText).toEqual('call @bar(int)(i32 %x)');
        expect(tree.replaceAll('$_Z3bari = comdat any').newText).toEqual('$bar(int) = comdat any');
        expect(tree.replaceAll('call @_Z3bari.cold(i32 %x)').newText).toEqual('call @bar(int) [clone .cold](i32 %x)');
        expect(tree.replaceAll('_Z3bari.exit:   ; preds = %_Z3bari.exit8').newText).toEqual(
            '_Z3bari.exit:   ; preds = %_Z3bari.exit8',
        );
        expect(tree.replaceAllText('_Z3bari.exit:   ; preds = %_Z3bari.exit8')).toEqual(
            '_Z3bari.exit:   ; preds = %_Z3bari.exit8',
        );
        expect(tree.replaceAllText('ends with _Z3bari')).toEqual('ends with bar(int)');
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

        const baseDemangler = new DummyCppDemangler(cppfiltpath, new DummyCompiler(), ['-n']);
        const demangler = new LLVMIRDemangler(baseDemangler);

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

    it('leaves labels named after inlined functions alone, keeping the IR CFG intact', () => {
        // clang trunk -O2 IR, keeping value names, for:
        //   void ext(int); void ext2(int);
        //   int bar(int x) { if (x > 5) { ext(x); return 1; } ext2(x); return 0; }
        //   int foo(int y) { return bar(y) + bar(y + 1); }
        const ir = `define dso_local noundef range(i32 0, 2) i32 @_Z3bari(i32 noundef %x) local_unnamed_addr {
entry:
  %cmp = icmp sgt i32 %x, 5
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  tail call void @_Z3exti(i32 noundef %x)
  br label %return

if.end:                                           ; preds = %entry
  tail call void @_Z4ext2i(i32 noundef %x)
  br label %return

return:                                           ; preds = %if.end, %if.then
  %retval.0 = phi i32 [ 1, %if.then ], [ 0, %if.end ]
  ret i32 %retval.0
}

declare void @_Z3exti(i32 noundef) local_unnamed_addr #1

declare void @_Z4ext2i(i32 noundef) local_unnamed_addr #1

define dso_local noundef range(i32 0, 3) i32 @_Z3fooi(i32 noundef %y) local_unnamed_addr {
entry:
  %cmp.i = icmp sgt i32 %y, 5
  %add10 = add nsw i32 %y, 1
  br i1 %cmp.i, label %_Z3bari.exit.thread, label %_Z3bari.exit

_Z3bari.exit.thread:                              ; preds = %entry
  tail call void @_Z3exti(i32 noundef %y)
  br label %if.then.i7

_Z3bari.exit:                                     ; preds = %entry
  tail call void @_Z4ext2i(i32 noundef %y)
  %cmp.i4 = icmp eq i32 %y, 5
  br i1 %cmp.i4, label %if.then.i7, label %if.end.i5

if.then.i7:                                       ; preds = %_Z3bari.exit.thread, %_Z3bari.exit
  %add14 = phi i32 [ %add10, %_Z3bari.exit.thread ], [ 6, %_Z3bari.exit ]
  %retval.0.i13 = phi i32 [ 1, %_Z3bari.exit.thread ], [ 0, %_Z3bari.exit ]
  tail call void @_Z3exti(i32 noundef %add14)
  br label %_Z3bari.exit8

if.end.i5:                                        ; preds = %_Z3bari.exit
  tail call void @_Z4ext2i(i32 noundef %add10)
  br label %_Z3bari.exit8

_Z3bari.exit8:                                    ; preds = %if.then.i7, %if.end.i5
  %retval.0.i12 = phi i32 [ %retval.0.i13, %if.then.i7 ], [ 0, %if.end.i5 ]
  %retval.0.i6 = phi i32 [ 1, %if.then.i7 ], [ 0, %if.end.i5 ]
  %add2 = add nuw nsw i32 %retval.0.i6, %retval.0.i12
  ret i32 %add2
}`;
        const irLines = () => ir.split('\n').map(text => ({text}));
        const shape = ({nodes, edges}: CFG) => ({
            nodes: nodes.map(node => node.id),
            edges: edges.map(({from, to}) => [from, to]),
        });
        const compilerInfo = makeFakeCompilerInfo({compilerType: '', version: 'clang'});

        const baseDemangler = new DummyCppDemangler(cppfiltpath, new DummyCompiler(), ['-n']);
        const demangler = new LLVMIRDemangler(baseDemangler);

        return demangler
            .process({asm: irLines()})
            .then(async output => {
                const text = output.asm.map(line => line.text);
                expect(text).toContain('  tail call void @ext2(int)(i32 noundef %y)');
                expect(text).toContain('  br i1 %cmp.i, label %_Z3bari.exit.thread, label %_Z3bari.exit');
                expect(text).toContain('_Z3bari.exit:                                     ; preds = %entry');

                const mangledCfg = await generateStructure(compilerInfo, irLines(), true);
                const demangledCfg = await generateStructure(
                    compilerInfo,
                    output.asm.map(line => ({text: line.text})),
                    true,
                );
                expect(shape(mangledCfg._Z3fooi).nodes).toHaveLength(6);
                expect(shape(demangledCfg['foo(int)'])).toEqual(shape(mangledCfg._Z3fooi));
            })
            .catch(catchCppfiltNonexistence);
    });

    it('demangles quoted identifiers', () => {
        const result = {
            asm: [
                {
                    text: '  invoke void @"_ZN4core3ptr53drop_in_place$LT$alloc..raw_vec..RawVec$LT$u8$GT$$GT$17h2e3e5a8e7287bb5aE"(ptr align 8 %_1) #17',
                },
            ],
        };

        const baseDemangler = new DummyCppDemangler(cppfiltpath, new DummyCompiler(), ['-n']);
        const demangler = new LLVMIRDemangler(baseDemangler);

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

describe('LLVM IR demangler with Windows demanglers', () => {
    afterEach(() => {
        vi.restoreAllMocks();
    });

    function makeCompilerWithDemangler(stdoutFor: (input: string) => string) {
        const compiler = new DummyCompiler();
        vi.spyOn(compiler, 'getDefaultExecOptions').mockReturnValue({env: {}});
        const execSpy = vi.spyOn(compiler, 'exec').mockImplementation(async (_exe, _args, options) => ({
            code: 0,
            okToCache: true,
            filenameTransform: f => f,
            stdout: stdoutFor(options.input ?? ''),
            stderr: '',
            execTime: 0,
            timedOut: false,
            truncated: false,
        }));
        return {compiler, execSpy};
    }

    it('does not map the undname banner onto symbols', async () => {
        const {compiler, execSpy} = makeCompilerWithDemangler(() =>
            [
                'Microsoft (R) C++ Name Undecorator',
                'Copyright (C) Microsoft Corporation. All rights reserved.',
                '',
                'Undecoration of :- "?longFunctionName@@YA?AUCustomS@@U1@@Z"',
                'is :- "CustomS __cdecl longFunctionName(CustomS)"',
                '',
                'Undecoration of :- "llvm.memcpy.p0.p0.i64"',
                'is :- "llvm.memcpy.p0.p0.i64"',
                '',
            ].join('\n'),
        );
        const demangler = new LLVMIRDemangler(new Win32Demangler('undname.exe', compiler));

        const output = await demangler.process({
            asm: [
                {text: 'define dso_local i64 @"?longFunctionName@@YA?AUCustomS@@U1@@Z"(i64 %0) {'},
                {text: '  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %2, ptr align 4 %3, i64 8, i1 false)'},
            ],
        });

        expect(output.asm.map(line => line.text)).toEqual([
            'define dso_local i64 @"CustomS __cdecl longFunctionName(CustomS)"(i64 %0) {',
            '  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %2, ptr align 4 %3, i64 8, i1 false)',
        ]);
        expect(execSpy.mock.calls[0][1]).toContain('?longFunctionName@@YA?AUCustomS@@U1@@Z');
    });

    it('demangles clang-cl IR of a class with a vftable, despite symbols llvm-undname cannot demangle', async () => {
        // Real llvm-undname results. It can't demangle the other names it gets: the unnamed vftable `0`, and the
        // fragments `6B` and `8` that the unquoted symbol pattern picks out of quoted names.
        const demangled: Record<string, string> = {
            '??_R4S@@6B@': "const S::`RTTI Complete Object Locator'",
            '?f@S@@UEAAHXZ': 'virtual int S::f(void)',
            '??_R0?AUS@@@8': "struct S `RTTI Type Descriptor'",
            '??_7type_info@@6B@': "const type_info::`vftable'",
            '??_7S@@6B@': "const S::`vftable'",
            '?g@@YAHXZ': 'int g(void)',
            '??0S@@QEAA@XZ': 'S::S(void)',
        };
        // Like llvm-undname: echo each name, print its demangling only on success, then an empty line
        const {compiler} = makeCompilerWithDemangler(input =>
            utils
                .splitLines(input)
                .filter(Boolean)
                .map(name => (name in demangled ? `${name}\n${demangled[name]}\n\n` : `${name}\n\n`))
                .join(''),
        );
        const demangler = new LLVMIRDemangler(new LLVMWin32Demangler('llvm-undname.exe', compiler));

        // From clang-cl 18.1 for: struct S { virtual int f(); }; int S::f() { return 1; } int g() { S s; return s.f(); }
        const output = await demangler.process({
            asm: [
                {
                    text: '@0 = private unnamed_addr constant { [2 x ptr] } { [2 x ptr] [ptr @"??_R4S@@6B@", ptr @"?f@S@@UEAAHXZ"] }, comdat($"??_7S@@6B@")',
                },
                {
                    text: '@"??_R0?AUS@@@8" = linkonce_odr global %rtti.TypeDescriptor7 { ptr @"??_7type_info@@6B@", ptr null, [8 x i8] c".?AUS@@\\00" }, comdat',
                },
                {
                    text: '@"??_7S@@6B@" = unnamed_addr alias ptr, getelementptr inbounds ({ [2 x ptr] }, ptr @0, i32 0, i32 0, i32 1)',
                },
                {
                    text: 'define dso_local noundef i32 @"?f@S@@UEAAHXZ"(ptr noundef nonnull align 8 dereferenceable(8) %0) unnamed_addr align 2 {',
                },
                {text: 'define dso_local noundef i32 @"?g@@YAHXZ"() {'},
                {
                    text: '  %2 = call noundef ptr @"??0S@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %1) #2',
                },
                {
                    text: 'define linkonce_odr dso_local noundef ptr @"??0S@@QEAA@XZ"(ptr noundef nonnull returned align 8 dereferenceable(8) %0) unnamed_addr comdat align 2 {',
                },
                {text: '  store ptr @"??_7S@@6B@", ptr %3, align 8'},
            ],
        });

        expect(output.asm.map(line => line.text)).toEqual([
            '@0 = private unnamed_addr constant { [2 x ptr] } { [2 x ptr] [ptr @"const S::`RTTI Complete Object Locator\'", ptr @"virtual int S::f(void)"] }, comdat($"const S::`vftable\'")',
            '@"struct S `RTTI Type Descriptor\'" = linkonce_odr global %rtti.TypeDescriptor7 { ptr @"const type_info::`vftable\'", ptr null, [8 x i8] c".?AUS@@\\00" }, comdat',
            '@"const S::`vftable\'" = unnamed_addr alias ptr, getelementptr inbounds ({ [2 x ptr] }, ptr @0, i32 0, i32 0, i32 1)',
            'define dso_local noundef i32 @"virtual int S::f(void)"(ptr noundef nonnull align 8 dereferenceable(8) %0) unnamed_addr align 2 {',
            'define dso_local noundef i32 @"int g(void)"() {',
            '  %2 = call noundef ptr @"S::S(void)"(ptr noundef nonnull align 8 dereferenceable(8) %1) #2',
            'define linkonce_odr dso_local noundef ptr @"S::S(void)"(ptr noundef nonnull returned align 8 dereferenceable(8) %0) unnamed_addr comdat align 2 {',
            '  store ptr @"const S::`vftable\'", ptr %3, align 8',
        ]);
    });
});
