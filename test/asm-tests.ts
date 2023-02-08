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

import * as fs from 'fs';
import {fileURLToPath} from 'url';

import {AsmParser} from '../lib/parsers/asm-parser';
import {AsmParserTasking} from '../lib/parsers/asm-parser-tasking';
import {VcAsmParser} from '../lib/parsers/asm-parser-vc';
import {AsmRegex} from '../lib/parsers/asmregex';
import {elf_Parse} from '../lib/tooling/tasking-elfparse-tool';

import {makeFakeParseFiltersAndOutputOptions} from './utils';

describe('ASM CL parser', () => {
    it('should work for error documents', () => {
        const parser = new VcAsmParser();
        const result = parser.process('<Compilation failed>', {
            directives: true,
        });

        result.asm.should.deep.equal([
            {
                source: null,
                text: '<Compilation failed>',
            },
        ]);
    });
});

describe('ASM regex base class', () => {
    it('should leave unfiltered lines alone', () => {
        const line = '     this    is    a line';
        AsmRegex.filterAsmLine(line, makeFakeParseFiltersAndOutputOptions({})).should.equal(line);
    });
    it('should use up internal whitespace when asked', () => {
        AsmRegex.filterAsmLine(
            '     this    is    a line',
            makeFakeParseFiltersAndOutputOptions({trim: true}),
        ).should.equal('  this is a line');
        AsmRegex.filterAsmLine('this    is    a line', makeFakeParseFiltersAndOutputOptions({trim: true})).should.equal(
            'this is a line',
        );
    });
    it('should keep whitespace in strings', () => {
        AsmRegex.filterAsmLine(
            'equs     "this    string"',
            makeFakeParseFiltersAndOutputOptions({trim: true}),
        ).should.equal('equs "this    string"');
        AsmRegex.filterAsmLine(
            '     equs     "this    string"',
            makeFakeParseFiltersAndOutputOptions({trim: true}),
        ).should.equal('  equs "this    string"');
        AsmRegex.filterAsmLine(
            'equs     "this    \\"  string  \\""',
            makeFakeParseFiltersAndOutputOptions({trim: true}),
        ).should.equal('equs "this    \\"  string  \\""');
    });
    it('should not get upset by mismatched strings', () => {
        AsmRegex.filterAsmLine(
            'a   "string    \'yeah',
            makeFakeParseFiltersAndOutputOptions({trim: true}),
        ).should.equal('a "string \'yeah');
    });
});

describe('ASM parser base class', () => {
    let parser;
    const filters = {};

    before(() => {
        parser = new AsmParser();
    });

    it('should recognize source column numbers', () => {
        const asm = `
    .text
    .intel_syntax noprefix
    .file	"tmp.cpp"
    .file	1 "/usr/include" "stdlib.h"
    .file	2 "/usr/bin/../lib/gcc/x86_64-linux-gnu/9/../../../../include/c++/9/bits" "std_abs.h"
    .file	3 "/usr/bin/../lib/gcc/x86_64-linux-gnu/9/../../../../include/c++/9" "cstdlib"
    .file	4 "/usr/lib/llvm-11/lib/clang/11.0.0/include" "stddef.h"
    .file	5 "/usr/bin/../lib/gcc/x86_64-linux-gnu/9/../../../../include/c++/9" "stdlib.h"
    .globl	main                            # -- Begin function main
    .p2align	4, 0x90
    .type	main,@function
main:                                   # @main
    .Lfunc_begin0:
    .file	6 "/home/necto/proj/compiler-explorer" "tmp.cpp"
    .loc	6 3 0                           # tmp.cpp:3:0
    .cfi_startproc
# %bb.0:                                # %entry
    push	rbp
    .cfi_def_cfa_offset 16
    .cfi_offset rbp, -16
    mov1	rbp, rsp
    .cfi_def_cfa_register rbp
    sub	rsp, 48
    mov2	dword ptr [rbp - 4], 0
.Ltmp0:
    .loc	6 4 20 prologue_end             # tmp.cpp:4:20
    mov3	edi, 16
    call	malloc
    .loc	6 4 9 is_stmt 0                 # tmp.cpp:4:9
    mov4	qword ptr [rbp - 16], rax
`;
        const output = parser.process(asm, filters);
        const push_line = output.asm.find(line => line.text.trim().startsWith('push'));
        const mov1_line = output.asm.find(line => line.text.trim().startsWith('mov1'));
        const call_line = output.asm.find(line => line.text.trim().startsWith('call'));
        const mov4_line = output.asm.find(line => line.text.trim().startsWith('mov4'));
        push_line.source.should.not.have.ownProperty('column');
        mov1_line.source.should.not.have.ownProperty('column');
        call_line.source.column.should.equal(20);
        mov4_line.source.column.should.equal(9);
    });

    it('should parse line numbers when a column is not specified', () => {
        const asm = `
        .section .text
.LNDBG_TX:
# mark_description "Intel(R) C Intel(R) 64 Compiler XE for applications running on Intel(R) 64, Version 12.1 Build 20120410";
        .file "iccKTGaIssTdIn_"
        .text
..TXTST0:
# -- Begin  main
# mark_begin;
       .align    16,0x90
        .globl main
main:
..B1.1:                         # Preds ..B1.0
..___tag_value_main.2:                                          #
..LN0:
  .file   1 "-"
   .loc    1  2  is_stmt 1
        pushq     %rbp                                          #2.12
`;
        const output = parser.process(asm, filters);
        const pushq_line = output.asm.find(line => line.text.trim().startsWith('pushq'));
        pushq_line.source.should.not.have.ownProperty('column');
        pushq_line.source.line.should.equal(2);
    });
});

describe('Elf parse tooling', () => {
    let parser;
    const file = fileURLToPath(new URL('tasking\\cpp_demo.o', import.meta.url));

    before(() => {
        parser = new elf_Parse(file);
        parser._elf_examplepathc = 'C:\\Users\\QXZ3F7O\\Documents\\work\\compiler-explorer\\test\\tasking\\cpp_demo.c';
        parser._elf_examplepathcpp =
            'C:\\Users\\QXZ3F7O\\Documents\\work\\compiler-explorer\\test\\tasking\\cpp_demo.cpp';
        parser.start();
    });

    it('whether elf file', () => {
        const arr = parser._elf_header.e_ident.toString();
        const equal = Buffer.from([127, 69, 76, 70, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]).toString();
        arr.should.equal(equal);
    });

    it('section .debug_line', () => {
        //read file
        const file1 = fileURLToPath(new URL('tasking\\testdemo1', import.meta.url));
        const arr = fs.readFileSync(file1);
        const strs = arr.toString();
        const strarr = strs.split(' ');

        //0x
        for (let j = 0; j < strarr.length; j++) {
            strarr[j] = '0x' + strarr[j];
        }

        let i = 0;
        const sections = parser._elf_debugLine;
        for (const section of sections) {
            section.sh_offset.should.equal(Number.parseInt(strarr[i++]));
            section.sh_size.should.equal(Number.parseInt(strarr[i++]));
        }
    });

    //make sure opcode number
    it('opcode number test', () => {
        //read file
        const file2 = fileURLToPath(new URL('tasking\\testdemo2', import.meta.url));
        const arr = fs.readFileSync(file2);
        const strs = arr.toString();
        const strarr = strs.split(' ');
        const opcodearr = parser._elf_opcode_test;
        for (const [i, element] of opcodearr.entries()) {
            element.should.equal(Number.parseInt(strarr[i]));
        }
    });

    //make sure opcode number
    it('section group', () => {
        //read file
        const file3 = fileURLToPath(new URL('tasking\\testdemo3', import.meta.url));
        const arr = fs.readFileSync(file3);
        const strs = arr.toString();
        const strarr = strs.split(' ');
        for (let j = 0; j < strarr.length; j++) {
            strarr[j] = '0x' + strarr[j];
        }
        let i = 0;
        const elfgroup = parser._elf_group;
        for (const section of elfgroup) {
            section.sh_offset.should.equal(Number.parseInt(strarr[i++]));
            section.sh_size.should.equal(Number.parseInt(strarr[i++]));
        }
    });

    //make sure line number
    it('line number', () => {
        //map
        const map1 = parser._elf_debugLineMap.get('.text.cpp_demo._Z11printfhellov');
        const map2 = parser._elf_debugLineMap.get('.text.cpp_demo.main');

        map1.get('00000000').should.equal(5);
        map1.get('0000000c').should.equal(6);

        map2.get('00000000').should.equal(9);
        map2.get('00000004').should.equal(10);
        map2.get('00000006').should.equal(11);
        map2.get('0000000a').should.equal(12);
        map2.get('0000000e').should.equal(11);
        map2.get('00000016').should.equal(15);
        map2.get('0000001a').should.equal(17);
        map2.get('00000022').should.equal(18);
        //set
        parser._elf_debugLineSet.has('.text.cpp_demo._Z11printfhellov').should.equal(true);
        parser._elf_debugLineSet.has('.text.cpp_demo.main').should.equal(true);
    });
});

describe('Elf All Opcode', () => {
    let parser;
    const file = fileURLToPath(new URL('tasking\\cpp.o', import.meta.url));
    before(() => {
        parser = new elf_Parse(file);
    });

    it('All switch', () => {
        parser.start();
        parser._filecontent = parser._filecontent.slice(0, 2444);
        const buf = Buffer.from([
            11, 1, 2, 6, 3, 221, 159, 171, 198, 192, 249, 198, 192, 159, 125, 4, 1, 5, 163, 224, 212, 185, 191, 134, 2,
            6, 7, 8, 9, 0, 1, 0, 6, 1, 0, 6, 2, 6, 0, 0, 0, 0, 6, 4, 6,
        ]);
        parser._filecontent = Buffer.concat([parser._filecontent, buf]);
        parser._elf_section[25].sh_size = parser._filecontent.length - 2409;
        parser._elf_section[25].sh_end = parser._elf_section[25].sh_offset + parser._elf_section[25].sh_size - 1;
        parser.parse_debugLine(25, '.text.cpp.main');
        parser.begin.should.equal(2489);
    });
});

describe('Asm Parser tooling-tasking', () => {
    let parser;
    const file = fileURLToPath(new URL('tasking\\cpp_demo.o', import.meta.url));
    const filters = {
        trim: true,
        directives: true,
    };
    before(() => {
        parser = new AsmParserTasking();
        parser._elffilepath = file;
        parser.testcpppath = 'C:\\Users\\QXZ3F7O\\Documents\\work\\compiler-explorer\\test\\tasking\\cpp_demo.cpp';
    });

    it('output.s-text and address)', () => {
        const asm = `
        ---------- Section dump ----------

        .sdecl '.text.cpp_demo.main', CODE AT 0x0
        .sect  '.text.cpp_demo.main'
00000000 6d 00 00 00  printfhello():   call        printfhello()
00000004 82 08                         mov         d8,#0x0
00000006 82 00                         mov         d0,#0x0
00000008 3c 04                         jg          0x10
0000000a 8b 48 06 80                   add         d8,d8,#0x64
0000000e c2 10        ___Z11printfhellov_function_end:add         d0,#0x1       
00000010 da 0a                         mov         d15,#0xa
00000012 3f f0 fc 7f                   jlt         d0,d15,0xa
00000016 6d 00 00 00                   call        0x16
0000001a bb c0 e0 2a                   mov.u       d2,#0xae0c
0000001e e2 82                         mul         d2,d8
00000020 3c 01                         jg          0x22
00000022 00 90                         ret

        .sdecl '.text.cpp_demo._Z11printfhellov', CODE AT 0x0
        .sect  '.text.cpp_demo._Z11printfhellov'
00000000 91 00 00 40  printfhello():   movh.a      a4,#0x0
00000004 d9 44 00 00                   lea         a4,[a4]0x0
00000008 6d 00 00 00                   call        0x8
0000000c 00 90                         ret
`;
        filters.directives = false;
        const output = parser.process(asm, filters);
        const address_00 = output.asm.find(line => line.text.trim().startsWith('00000000'));
        const address_04 = output.asm.find(line => line.text.trim().startsWith('00000004'));
        const address_06 = output.asm.find(line => line.text.trim().startsWith('00000006'));
        address_00.source.line.should.equal(9);
        address_04.source.line.should.equal(10);
        address_06.source.line.should.equal(11);
    });

    it('output.s-replace address code with symbol', () => {
        const asm = `
        ---------- Section dump ----------

        .sdecl '.text.cpp_demo.main', CODE AT 0x0
        .sect  '.text.cpp_demo.main'
00000000 6d 00 00 00  printfhello():   call        printfhello()
00000004 82 08                         mov         d8,#0x0
00000006 82 00                         mov         d0,#0x0
00000008 3c 04                         jg          0x10
0000000a 8b 48 06 80                   add         d8,d8,#0x64
0000000e c2 10        ___Z11printfhellov_function_end:add         d0,#0x1       
00000010 da 0a                         mov         d15,#0xa
00000012 3f f0 fc 7f                   jlt         d0,d15,0xa
00000016 6d 00 00 00                   call        0x16
0000001a bb c0 e0 2a                   mov.u       d2,#0xae0c
0000001e e2 82                         mul         d2,d8
00000020 3c 01                         jg          0x22
00000022 00 90                         ret

        .sdecl '.text.cpp_demo._Z11printfhellov', CODE AT 0x0
        .sect  '.text.cpp_demo._Z11printfhellov'
00000000 91 00 00 40  printfhello():   movh.a      a4,#0x0
00000004 d9 44 00 00                   lea         a4,[a4]0x0
00000008 6d 00 00 00                   call        0x8
0000000c 00 90                         ret
`;
        filters.directives = true;
        const output = parser.process(asm, filters);
        const address_00 = output.asm.find(line => line.text.trim().startsWith('call'));
        const address_04 = output.asm.find(line => line.text.trim().startsWith('mov'));

        address_00.text.should.equal('  call _main');
        address_04.text.should.equal('  mov d8,#0x0');
    });
});

describe('Function buttons', () => {
    let parser;
    const file = fileURLToPath(new URL('tasking\\example.o', import.meta.url));
    const elfparser = new AsmParserTasking();
    elfparser._elffilepath = file;
    elfparser.testcpppath = 'C:\\Users\\QXZ3F7O\\Documents\\work\\compiler-explorer\\test\\tasking\\example.cpp';
    before(() => {
        parser = new elf_Parse(file);
        parser.start();
        parser._elf_section[2451].sh_end = parser._elf_section[2451].sh_offset + parser._elf_section[2451].sh_size - 1;
        parser.parse_debugLine(2451, '');
    });

    const filters = {
        binary: false,
        directives: false,
        libraryCode: false,
        trim: false,
    };

    it('Link file recursion', () => {
        parser.begin.should.equal(1272149);
    });

    it('button filters.binary', () => {
        const asm = `
        .sdecl '.text.abort.libcw_fpu', CODE AT 0x8004e106
        .sect  '.text.abort.libcw_fpu'
8004e106 91 00 00 f7  abort:           movh.a      a15,#0x7000
8004e10a 99 ff a4 01                   ld.a        a15,[a15]0x1824
8004e10e bc f3                         jz.a        a15,0x8004e114
8004e110 82 64                         mov         d4,#0x6
8004e112 dc 1f                         calli       a15
8004e114 82 14                         mov         d4,#0x1
8004e116 1d ff ad db                   j           _Exit

        .sdecl '.text.atexit.libcw_fpu', CODE AT 0x8004e11a
        .sect  '.text.atexit.libcw_fpu'
8004e11a 91 00 00 f7  atexit:          movh.a      a15,#0x7000
8004e11e d9 ff ee 10                   lea         a15,[a15]0xc6e
8004e122 14 ff                         ld.bu       d15,[a15]
8004e124 3b 00 02 00                   mov         d0,#0x20
8004e128 7e 03                         jne         d15,d0,0x8004e12e
8004e12a 82 12                         mov         d2,#0x1
8004e12c 00 90                         ret
8004e12e 91 00 00 27                   movh.a      a2,#0x7000
8004e132 d9 22 64 07                   lea         a2,[a2]0x7424
8004e136 90 22                         addsc.a     a2,a2,d15,#0x2
8004e138 82 02                         mov         d2,#0x0
8004e13a f4 24                         st.a        [a2],a4
8004e13c c2 1f                         add         d15,#0x1
8004e13e 34 ff                         st.b        [a15],d15
8004e140 00 90                         ret

        .sdecl '.text.calloc.libcw_fpu', CODE AT 0x8004e142
        .sect  '.text.calloc.libcw_fpu'
8004e142 73 45 0a f0  calloc:          mul         d15,d5,d4
8004e146 02 f4                         mov         d4,d15
8004e148 6d 02 97 04                   call        malloc
8004e14c 40 2f                         mov.aa      a15,a2
8004e14e bc f6                         jz.a        a15,0x8004e15a
8004e150 82 04                         mov         d4,#0x0
8004e152 40 f4                         mov.aa      a4,a15
8004e154 02 f5                         mov         d5,d15
8004e156 6d 02 6b 06                   call        memset
8004e15a 40 f2                         mov.aa      a2,a15
8004e15c 00 90                         ret

        .sdecl '.text.example._Z5hellov', CODE AT 0x8004e15e
        .sect  '.text.example._Z5hellov'
8004e15e 85 88 04 00  hello():         ld.w        d8,0x80000004
8004e162 3b 90 00 90                   mov         d9,#0x9
8004e166 3c 24                         jg          0x8004e1ae
8004e168 91 90 00 48                   movh.a      a4,#0x8009
8004e170 6d 02 46 08                   call        printf
8004e174 91 00 00 47                   movh.a      a4,#0x7000
8004e17c c5 85 0c 00                   lea         a5,0x8000000c
8004e184 40 2f                         mov.aa      a15,a2
8004e1aa 3c 01                         jg          0x8004e1ac
8004e1ac c2 19                         add         d9,#0x1
8004e1ae da 64                         mov         d15,#0x64
8004e1b0 3f f9 dc 7f                   jlt         d9,d15,0x8004e168
8004e1b4 02 82                         mov         d2,d8
8004e1b8 00 90                         ret
`;
        filters.binary = true;
        const output = elfparser.process(asm, filters);
        output.asm[0].text.should.equal('8004e15e 85 88 04 00  hello():         ld.w        d8,0x80000004');
    });

    it('button filters.binary && libarycode', () => {
        const asm = `
    .sdecl '.text.abort.libcw_fpu', CODE AT 0x8004e106
    .sect  '.text.abort.libcw_fpu'
8004e106 91 00 00 f7  abort:           movh.a      a15,#0x7000
8004e10a 99 ff a4 01                   ld.a        a15,[a15]0x1824
8004e10e bc f3                         jz.a        a15,0x8004e114
8004e110 82 64                         mov         d4,#0x6
8004e112 dc 1f                         calli       a15
8004e114 82 14                         mov         d4,#0x1
8004e116 1d ff ad db                   j           _Exit

    .sdecl '.text.atexit.libcw_fpu', CODE AT 0x8004e11a
    .sect  '.text.atexit.libcw_fpu'
8004e11a 91 00 00 f7  atexit:          movh.a      a15,#0x7000
8004e11e d9 ff ee 10                   lea         a15,[a15]0xc6e
8004e122 14 ff                         ld.bu       d15,[a15]
8004e124 3b 00 02 00                   mov         d0,#0x20
8004e128 7e 03                         jne         d15,d0,0x8004e12e
8004e12a 82 12                         mov         d2,#0x1
8004e12c 00 90                         ret
8004e12e 91 00 00 27                   movh.a      a2,#0x7000
8004e132 d9 22 64 07                   lea         a2,[a2]0x7424
8004e136 90 22                         addsc.a     a2,a2,d15,#0x2
8004e138 82 02                         mov         d2,#0x0
8004e13a f4 24                         st.a        [a2],a4
8004e13c c2 1f                         add         d15,#0x1
8004e13e 34 ff                         st.b        [a15],d15
8004e140 00 90                         ret

    .sdecl '.text.calloc.libcw_fpu', CODE AT 0x8004e142
    .sect  '.text.calloc.libcw_fpu'
8004e142 73 45 0a f0  calloc:          mul         d15,d5,d4
8004e146 02 f4                         mov         d4,d15
8004e148 6d 02 97 04                   call        malloc
8004e14c 40 2f                         mov.aa      a15,a2
8004e14e bc f6                         jz.a        a15,0x8004e15a
8004e150 82 04                         mov         d4,#0x0
8004e152 40 f4                         mov.aa      a4,a15
8004e154 02 f5                         mov         d5,d15
8004e156 6d 02 6b 06                   call        memset
8004e15a 40 f2                         mov.aa      a2,a15
8004e15c 00 90                         ret

    .sdecl '.text.example._Z5hellov', CODE AT 0x8004e15e
    .sect  '.text.example._Z5hellov'
8004e15e 85 88 04 00  hello():         ld.w        d8,0x80000004
8004e162 3b 90 00 90                   mov         d9,#0x9
8004e166 3c 24                         jg          0x8004e1ae
8004e168 91 90 00 48                   movh.a      a4,#0x8009
8004e170 6d 02 46 08                   call        printf
8004e174 91 00 00 47                   movh.a      a4,#0x7000
8004e17c c5 85 0c 00                   lea         a5,0x8000000c
8004e184 40 2f                         mov.aa      a15,a2
8004e1aa 3c 01                         jg          0x8004e1ac
8004e1ac c2 19                         add         d9,#0x1
8004e1ae da 64                         mov         d15,#0x64
8004e1b0 3f f9 dc 7f                   jlt         d9,d15,0x8004e168
8004e1b4 02 82                         mov         d2,d8
8004e1b8 00 90                         ret
`;
        filters.libraryCode = true;
        filters.binary = true;
        const output = elfparser.process(asm, filters);
        output.asm[2].text.should.equal('8004e106 91 00 00 f7  abort:           movh.a      a15,#0x7000');
    });

    it('button filters.binary && directives', () => {
        const asm = `
    .sdecl '.text.abort.libcw_fpu', CODE AT 0x8004e106
    .sect  '.text.abort.libcw_fpu'
8004e106 91 00 00 f7  abort:           movh.a      a15,#0x7000
8004e10a 99 ff a4 01                   ld.a        a15,[a15]0x1824
8004e10e bc f3                         jz.a        a15,0x8004e114
8004e110 82 64                         mov         d4,#0x6
8004e112 dc 1f                         calli       a15
8004e114 82 14                         mov         d4,#0x1
8004e116 1d ff ad db                   j           _Exit

    .sdecl '.text.atexit.libcw_fpu', CODE AT 0x8004e11a
    .sect  '.text.atexit.libcw_fpu'
8004e11a 91 00 00 f7  atexit:          movh.a      a15,#0x7000
8004e11e d9 ff ee 10                   lea         a15,[a15]0xc6e
8004e122 14 ff                         ld.bu       d15,[a15]
8004e124 3b 00 02 00                   mov         d0,#0x20
8004e128 7e 03                         jne         d15,d0,0x8004e12e
8004e12a 82 12                         mov         d2,#0x1
8004e12c 00 90                         ret
8004e12e 91 00 00 27                   movh.a      a2,#0x7000
8004e132 d9 22 64 07                   lea         a2,[a2]0x7424
8004e136 90 22                         addsc.a     a2,a2,d15,#0x2
8004e138 82 02                         mov         d2,#0x0
8004e13a f4 24                         st.a        [a2],a4
8004e13c c2 1f                         add         d15,#0x1
8004e13e 34 ff                         st.b        [a15],d15
8004e140 00 90                         ret

    .sdecl '.text.calloc.libcw_fpu', CODE AT 0x8004e142
    .sect  '.text.calloc.libcw_fpu'
8004e142 73 45 0a f0  calloc:          mul         d15,d5,d4
8004e146 02 f4                         mov         d4,d15
8004e148 6d 02 97 04                   call        malloc
8004e14c 40 2f                         mov.aa      a15,a2
8004e14e bc f6                         jz.a        a15,0x8004e15a
8004e150 82 04                         mov         d4,#0x0
8004e152 40 f4                         mov.aa      a4,a15
8004e154 02 f5                         mov         d5,d15
8004e156 6d 02 6b 06                   call        memset
8004e15a 40 f2                         mov.aa      a2,a15
8004e15c 00 90                         ret

    .sdecl '.text.example._Z5hellov', CODE AT 0x8004e15e
    .sect  '.text.example._Z5hellov'
8004e15e 85 88 04 00  hello():         ld.w        d8,0x80000004
8004e162 3b 90 00 90                   mov         d9,#0x9
8004e166 3c 24                         jg          0x8004e1ae
8004e168 91 90 00 48                   movh.a      a4,#0x8009
8004e170 6d 02 46 08                   call        printf
8004e174 91 00 00 47                   movh.a      a4,#0x7000
8004e17c c5 85 0c 00                   lea         a5,0x8000000c
8004e184 40 2f                         mov.aa      a15,a2
8004e1aa 3c 01                         jg          0x8004e1ac
8004e1ac c2 19                         add         d9,#0x1
8004e1ae da 64                         mov         d15,#0x64
8004e1b0 3f f9 dc 7f                   jlt         d9,d15,0x8004e168
8004e1b4 02 82                         mov         d2,d8
8004e1b8 00 90                         ret
`;
        filters.libraryCode = false;
        filters.binary = true;
        filters.directives = true;
        const output = elfparser.process(asm, filters);
        output.asm[0].text.should.equal('  ld.w        d8,0x80000004');
    });

    it('button filters.binary && directives && Whitespace', () => {
        const asm = `
    .sdecl '.text.abort.libcw_fpu', CODE AT 0x8004e106
    .sect  '.text.abort.libcw_fpu'
8004e106 91 00 00 f7  abort:           movh.a      a15,#0x7000
8004e10a 99 ff a4 01                   ld.a        a15,[a15]0x1824
8004e10e bc f3                         jz.a        a15,0x8004e114
8004e110 82 64                         mov         d4,#0x6
8004e112 dc 1f                         calli       a15
8004e114 82 14                         mov         d4,#0x1
8004e116 1d ff ad db                   j           _Exit

    .sdecl '.text.atexit.libcw_fpu', CODE AT 0x8004e11a
    .sect  '.text.atexit.libcw_fpu'
8004e11a 91 00 00 f7  atexit:          movh.a      a15,#0x7000
8004e11e d9 ff ee 10                   lea         a15,[a15]0xc6e
8004e122 14 ff                         ld.bu       d15,[a15]
8004e124 3b 00 02 00                   mov         d0,#0x20
8004e128 7e 03                         jne         d15,d0,0x8004e12e
8004e12a 82 12                         mov         d2,#0x1
8004e12c 00 90                         ret
8004e12e 91 00 00 27                   movh.a      a2,#0x7000
8004e132 d9 22 64 07                   lea         a2,[a2]0x7424
8004e136 90 22                         addsc.a     a2,a2,d15,#0x2
8004e138 82 02                         mov         d2,#0x0
8004e13a f4 24                         st.a        [a2],a4
8004e13c c2 1f                         add         d15,#0x1
8004e13e 34 ff                         st.b        [a15],d15
8004e140 00 90                         ret

    .sdecl '.text.calloc.libcw_fpu', CODE AT 0x8004e142
    .sect  '.text.calloc.libcw_fpu'
8004e142 73 45 0a f0  calloc:          mul         d15,d5,d4
8004e146 02 f4                         mov         d4,d15
8004e148 6d 02 97 04                   call        malloc
8004e14c 40 2f                         mov.aa      a15,a2
8004e14e bc f6                         jz.a        a15,0x8004e15a
8004e150 82 04                         mov         d4,#0x0
8004e152 40 f4                         mov.aa      a4,a15
8004e154 02 f5                         mov         d5,d15
8004e156 6d 02 6b 06                   call        memset
8004e15a 40 f2                         mov.aa      a2,a15
8004e15c 00 90                         ret

    .sdecl '.text.example._Z5hellov', CODE AT 0x8004e15e
    .sect  '.text.example._Z5hellov'
8004e15e 85 88 04 00  hello():         ld.w        d8,0x80000004
8004e162 3b 90 00 90                   mov         d9,#0x9
8004e166 3c 24                         jg          0x8004e1ae
8004e168 91 90 00 48                   movh.a      a4,#0x8009
8004e170 6d 02 46 08                   call        printf
8004e174 91 00 00 47                   movh.a      a4,#0x7000
8004e17c c5 85 0c 00                   lea         a5,0x8000000c
8004e184 40 2f                         mov.aa      a15,a2
8004e1aa 3c 01                         jg          0x8004e1ac
8004e1ac c2 19                         add         d9,#0x1
8004e1ae da 64                         mov         d15,#0x64
8004e1b0 3f f9 dc 7f                   jlt         d9,d15,0x8004e168
8004e1b4 02 82                         mov         d2,d8
8004e1b8 00 90                         ret
`;
        filters.libraryCode = false;
        filters.binary = true;
        filters.directives = true;
        filters.trim = true;
        const output = elfparser.process(asm, filters);
        output.asm[0].text.should.equal('  ld.w d8,0x80000004');
    });
});
