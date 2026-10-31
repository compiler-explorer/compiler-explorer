// Copyright (c) 2024, Compiler Explorer Authors
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

import {AsmParser} from '../lib/parsers/asm-parser.js';
import {MlirAsmParser} from '../lib/parsers/asm-parser-mlir.js';
import {PolkaVMAsmParser} from '../lib/parsers/asm-parser-polkavm.js';
import {PTXAsmParser} from '../lib/parsers/asm-parser-ptx.js';
import {ResolcRiscVAsmParser} from '../lib/parsers/asm-parser-resolc-riscv.js';
import {SassAsmParser} from '../lib/parsers/asm-parser-sass.js';
import type {PropertyGetter} from '../lib/properties.interfaces.js';
import type {ParsedAsmResult} from '../types/asmresult/asmresult.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../types/features/filters.interfaces.js';

describe('AsmParser tests', () => {
    const parser = new AsmParser();
    it('should identify generic opcodes', () => {
        expect(parser.hasOpcode('  mov r0, #1')).toBe(true);
        expect(parser.hasOpcode('  ROL A')).toBe(true);
    });
    it('should not identify non-opcodes as opcodes', () => {
        expect(parser.hasOpcode('  ;mov r0, #1')).toBe(false);
        expect(parser.hasOpcode('')).toBe(false);
        expect(parser.hasOpcode('# moose')).toBe(false);
    });
    it('should identify llvm opcodes', () => {
        expect(parser.hasOpcode('  %i1 = phi i32 [ %i2, %.preheader ], [ 0, %bb ]')).toBe(true);
    });
});

describe('AsmParser comment filtering', () => {
    const parser = new AsmParser();
    it('should keep label lines starting with @ when filtering comments', () => {
        const input = '@cube@4:\n    ret';
        const result = parser.processAsm(input, {commentOnly: true});
        const lines = result.asm.map(line => line.text);
        expect(lines[0]).toBe('@cube@4:');
        expect(lines[1]).toBe('    ret');
    });
});

describe('AsmParser binary line records', () => {
    const parser = new AsmParser();
    const gnuObjdump = `0000000000000000 <main>:
main():
/tmp/project/src/example.mach:7
   0:	55                   	push   rbp
/tmp/project/dep/std/src/print.mach:64
   1:	c3                   	ret`;
    const llvmObjdump = `0000000000000000 <main>:
; main():
; /tmp/project/src/example.mach:7
       0: fa010113     	addi	sp, sp, -0x60
; /tmp/project/dep/std/src/print.mach:64
       4: 00008067     	ret`;

    it('should read source lines from gnu objdump', () => {
        const result = parser.processBinaryAsm(gnuObjdump, {});
        expect(result.asm.map(line => line.source?.line)).toEqual([undefined, 7, 64]);
    });

    it('should read source lines from llvm-objdump', () => {
        const result = parser.processBinaryAsm(llvmObjdump, {});
        expect(result.asm.map(line => line.text)).toEqual(['main:', ' addi\tsp, sp, -0x60', ' ret']);
        expect(result.asm.map(line => line.source?.line)).toEqual([undefined, 7, 64]);
    });

    it('should keep the function label when filtering library code in llvm-objdump output', () => {
        const result = parser.processBinaryAsm(llvmObjdump, {libraryCode: true});
        expect(result.asm[0].text).toEqual('main:');
    });
});

describe('AsmParser binary source records after the function label', () => {
    const parser = new AsmParser();
    const cases: [string, string][] = [
        ['POSIX', '/tmp/example.cpp'],
        ['Windows forward slash', 'C:/tmp/example.cpp'],
        ['Windows backslash', 'C:\\tmp\\example.cpp'],
    ];

    for (const [style, file] of cases) {
        it(`should keep a ${style} source line that objdump printed after the label`, () => {
            const asm = [
                '0000000000401000 <foo>:',
                `${file}:3`,
                '  401000:\tc3                   \tret',
                '  401001:\t90                   \tnop',
                `${file}:9`,
                '  401002:\tc3                   \tret',
            ].join('\n');
            const result = parser.processBinaryAsm(asm, {});
            expect(result.asm.map(line => line.source?.line)).toEqual([undefined, 3, 3, 9]);
        });
    }
});

describe('AsmParser binary source corner cases', () => {
    const parser = new AsmParser();
    const label = '0000000000000000 <foo>:';
    const mov = '  10: 48 89 e5  mov %rsp,%rbp';
    const nop = '  11: 90        nop';
    const maskedExample = (line: number) => ({file: null, line, mainsource: true});

    // objdump -l prints a function's first line record after its label.
    function dump(sourceRecord?: string): string {
        const lines = [label];
        if (sourceRecord !== undefined) lines.push(sourceRecord);
        lines.push(mov, nop);
        return lines.join('\n');
    }

    function instructions(asm: string, filters: Partial<ParseFiltersAndOutputOptions> = {}) {
        const result = parser.processBinaryAsm(asm, filters);
        return result.asm.filter(line => line.address !== undefined);
    }

    it('should map a POSIX source record when filenames are masked', () => {
        const kept = instructions(dump('/tmp/example.cpp:3'));
        expect(kept).toHaveLength(2);
        expect(kept.map(line => line.source)).toEqual([maskedExample(3), maskedExample(3)]);
    });

    it('should map a Windows forward-slash source record', () => {
        const kept = instructions(dump('C:/tmp/example.cpp:3'));
        expect(kept).toHaveLength(2);
        expect(kept.map(line => line.source)).toEqual([maskedExample(3), maskedExample(3)]);
    });

    it('should map a Windows backslash source record', () => {
        const record = 'C:\\tmp\\example.cpp:3';
        expect(record).toBe('C:\\tmp\\example.cpp:3');
        expect(record.includes('\\')).toBe(true);
        const kept = instructions(dump(record));
        expect(kept).toHaveLength(2);
        expect(kept.map(line => line.source)).toEqual([maskedExample(3), maskedExample(3)]);
    });

    it('should emit instructions with null source when objdump has no line record', () => {
        const kept = instructions(dump());
        expect(kept).toHaveLength(2);
        expect(kept.map(line => line.source)).toEqual([null, null]);
    });

    it('should keep a sourced instruction when library code is filtered', () => {
        const kept = instructions(dump('/tmp/example.cpp:3'), {libraryCode: true});
        expect(kept).toHaveLength(2);
        expect(kept.map(line => line.source)).toEqual([maskedExample(3), maskedExample(3)]);
    });

    it('should drop an instruction with no source when library code is filtered', () => {
        const kept = instructions(dump(), {libraryCode: true});
        expect(kept).toEqual([]);
    });

    it('should take the line record that follows each function label', () => {
        const asm = [label, '/tmp/a.cpp:3', mov, '0000000000000011 <bar>:', '/tmp/b.cpp:9', nop].join('\n');
        const result = parser.process(asm, {binary: true});
        const kept = result.asm.filter(line => line.address !== undefined);
        expect(kept).toHaveLength(2);
        expect(kept[0].source).toEqual({file: '/tmp/a.cpp', line: 3, mainsource: false});
        expect(kept[1].source).toEqual({file: '/tmp/b.cpp', line: 9, mainsource: false});
    });

    it('should keep the path when filenames are not masked', () => {
        const kept = instructions(dump('/tmp/example.cpp:3'), {dontMaskFilenames: true});
        expect(kept).toHaveLength(2);
        expect(kept.map(line => line.source)).toEqual([
            {file: '/tmp/example.cpp', line: 3, mainsource: true},
            {file: '/tmp/example.cpp', line: 3, mainsource: true},
        ]);
    });

    it('should keep spaces in an unmasked Windows path', () => {
        const kept = instructions(dump('C:/Program Files/example.cpp:12'), {dontMaskFilenames: true});
        expect(kept).toHaveLength(2);
        expect(kept.map(line => line.source)).toEqual([
            {file: 'C:/Program Files/example.cpp', line: 12, mainsource: true},
            {file: 'C:/Program Files/example.cpp', line: 12, mainsource: true},
        ]);
    });

    it('should read the line number and ignore a trailing discriminator', () => {
        const kept = instructions(dump('/tmp/example.cpp:8 (discriminator 2)'));
        expect(kept).toHaveLength(2);
        expect(kept.map(line => line.source)).toEqual([maskedExample(8), maskedExample(8)]);
    });

    it('should read the line number and ignore a trailing column', () => {
        const kept = instructions(dump('/tmp/example.cpp:8:2'));
        expect(kept).toHaveLength(2);
        expect(kept.map(line => line.source)).toEqual([maskedExample(8), maskedExample(8)]);
        expect(kept.every(line => !Object.hasOwn(line.source ?? {}, 'column'))).toBe(true);
    });

    it('should not treat an opcode line as a file:line record', () => {
        const kept = instructions([label, mov].join('\n'));
        expect(kept).toHaveLength(1);
        expect(kept[0].source).toBeNull();
    });

    it('should map a ./-relative source record from a trimmed-path build', () => {
        const kept = instructions(dump('./example.go:9'));
        expect(kept).toHaveLength(2);
        expect(kept.map(line => line.source)).toEqual([maskedExample(9), maskedExample(9)]);
    });

    it('should keep the path of a ./-relative library source record', () => {
        const kept = instructions(dump('; ./runtime/proc.go:482'));
        expect(kept).toHaveLength(2);
        expect(kept.map(line => line.source)).toEqual([
            {file: './runtime/proc.go', line: 482, mainsource: false},
            {file: './runtime/proc.go', line: 482, mainsource: false},
        ]);
    });

    it('should ignore a relative source path', () => {
        const kept = instructions(dump('path/to/example.cpp:3'));
        expect(kept).toHaveLength(2);
        expect(kept.map(line => line.source)).toEqual([null, null]);
    });

    it('should ignore a lowercase drive letter', () => {
        const kept = instructions(dump('c:/tmp/example.cpp:3'));
        expect(kept).toHaveLength(2);
        expect(kept.map(line => line.source)).toEqual([null, null]);
    });

    it('should keep a single-line error document', () => {
        const result = parser.processBinaryAsm('<No output file>', {});
        expect(result.asm).toEqual([{text: '<No output file>', source: null}]);
    });
});

describe('AsmParser binary source lines', () => {
    const parser = new AsmParser();
    const objdump = `0000000000401020 <square(int)>:
square(int)():
/app/example.cpp:4
  401020:	89 f8                	mov    eax,edi
/opt/compiler-explorer/gcc-15.1.0/include/c++/15.1.0/bits/stl_algobase.h:238
  401022:	0f af c7             	imul   eax,edi
/app/example.cpp:5
  401025:	c3                   	ret`;

    it('should only mark lines from the main source file as main source', () => {
        const result = parser.processBinaryAsm(objdump, {});
        expect(result.asm.map(line => line.source)).toEqual([
            null,
            {file: null, line: 4, mainsource: true},
            {
                file: '/opt/compiler-explorer/gcc-15.1.0/include/c++/15.1.0/bits/stl_algobase.h',
                line: 238,
                mainsource: false,
            },
            {file: null, line: 5, mainsource: true},
        ]);
    });

    it('should name the main source file when filenames are not masked', () => {
        const result = parser.processBinaryAsm(objdump, {dontMaskFilenames: true});
        expect(result.asm[1].source).toEqual({file: 'example.cpp', line: 4, mainsource: true});
    });

    it('should keep code inlined from other files when filtering library code', () => {
        const result = parser.processBinaryAsm(objdump, {libraryCode: true});
        expect(result.asm.map(line => line.text)).toEqual([
            'square(int):',
            ' mov    eax,edi',
            ' imul   eax,edi',
            ' ret',
        ]);
    });
});

describe('AsmParser binary source lines in functions without debug info', () => {
    const parser = new AsmParser();
    // Reconstructed from real mingw gcc 11.3 output on winstaging (`-O1 -g`, "Link to binary", an example.c
    // with square() and main()). The API returns only parsed asm, so the label, `func():` and `file:line` lines
    // are rebuilt around the real addresses, opcodes and instructions; the temp directory in the path is a
    // stand-in. _setargv and _matherr are mingw CRT code that has no line records.
    const example = 'C:/Users/ce/AppData/Local/Temp/compiler-explorer-compiler/example.c';
    const objdump = [
        '',
        '00000001400017d0 <square>:',
        'square():',
        `${example}:2`,
        '   1400017d0:\t0f af c9             \timul   ecx,ecx',
        `${example}:3`,
        '   1400017d3:\t89 c8                \tmov    eax,ecx',
        '   1400017d5:\tc3                   \tret',
        '',
        '00000001400017d6 <main>:',
        'main():',
        `${example}:5`,
        '   1400017d6:\t48 83 ec 28          \tsub    rsp,0x28',
        '   1400017da:\te8 c8 00 00 00       \tcall   1400018a7 <__main>',
        '   1400017df:\tb8 09 00 00 00       \tmov    eax,0x9',
        '   1400017e4:\t48 83 c4 28          \tadd    rsp,0x28',
        '   1400017e8:\tc3                   \tret',
        '   1400017e9:\t90                   \tnop',
        '',
        '00000001400018d0 <_setargv>:',
        '   1400018d0:\t55                   \tpush   rbp',
        '   1400018d1:\t48 89 e5             \tmov    rbp,rsp',
        '   1400018d4:\tb8 00 00 00 00       \tmov    eax,0x0',
        '   1400018d9:\t5d                   \tpop    rbp',
        '   1400018da:\tc3                   \tret',
        '',
        '00000001400019f0 <_matherr>:',
        '   1400019f0:\t55                   \tpush   rbp',
        '   1400019f1:\t53                   \tpush   rbx',
        '   1400019f2:\t48 81 ec 88 00 00 00 \tsub    rsp,0x88',
    ].join('\n');

    function linesByFunction(filters: Partial<ParseFiltersAndOutputOptions> = {}) {
        const result: Record<string, (number | null | undefined)[]> = {};
        let current = '';
        for (const line of parser.processBinaryAsm(objdump, filters).asm) {
            if (line.address === undefined) {
                current = line.text.replace(/:$/, '');
                result[current] = [];
            } else {
                result[current].push(line.source?.line);
            }
        }
        return result;
    }

    it('should not carry the last source line into the next function', () => {
        expect(linesByFunction()).toEqual({
            square: [2, 3, 3],
            main: [5, 5, 5, 5, 5, 5],
            _setargv: [undefined, undefined, undefined, undefined, undefined],
            _matherr: [undefined, undefined, undefined],
        });
    });

    it('should filter out functions without debug info as library code', () => {
        expect(linesByFunction({libraryCode: true})).toEqual({
            square: [2, 3, 3],
            main: [5, 5, 5, 5, 5, 5],
        });
    });
});

describe('AsmParser library code filtering in binary asm', () => {
    const parser = new AsmParser();
    const objdump = [
        '0000000000401000 <lib_a>:',
        '  401000:\tc3                   \tret',
        '0000000000401001 <lib_b>:',
        '  401001:\tc3                   \tret',
        '0000000000401002 <lib_c>:',
        '  401002:\tc3                   \tret',
        '0000000000401003 <main>:',
        '/tmp/compiler-explorer-compiler/example.c:3',
        '  401003:\tc3                   \tret',
    ].join('\n');

    it('should drop the label of every consecutive library function', () => {
        const result = parser.processBinaryAsm(objdump, {libraryCode: true});
        expect(result.asm.map(line => line.text)).toEqual(['main:', ' ret']);
    });

    it('should keep every label when library code is shown', () => {
        const result = parser.processBinaryAsm(objdump, {libraryCode: false});
        expect(result.asm.map(line => line.text).filter(text => text.endsWith(':'))).toEqual([
            'lib_a:',
            'lib_b:',
            'lib_c:',
            'main:',
        ]);
    });
});

describe('AsmParser library code filtering of statically linked code in binary asm', () => {
    // Trimmed from clang 23 `-fsanitize=address -g` output for an example.c compiled in /app. The asan runtime is linked
    // statically and keeps the paths it was built from; asan.module_ctor is generated code with no line information.
    const objdump = [
        '000000000002d420 <__asan_report_load1_asm>:',
        '__asan_report_load1_asm():',
        '/root/llvm-project/compiler-rt/lib/asan/asan_rtl_static.cpp:25',
        '   2d420:\te9 6b 6e 0b 00       \tjmp    e4290 <__asan_report_load1>',
        '   2d425:\t0f 1f 00             \tnopl   (%rax)',
        '',
        '000000000002d428 <__asan_report_load2_asm>:',
        '__asan_report_load2_asm():',
        '/root/llvm-project/compiler-rt/lib/asan/asan_rtl_static.cpp:26',
        '   2d428:\te9 23 6f 0b 00       \tjmp    e4350 <__asan_report_load2>',
        '   2d42d:\t0f 1f 00             \tnopl   (%rax)',
        '',
        '000000000011d7c0 <main>:',
        'main():',
        '/app/example.c:3',
        '  11d7c0:\t55                   \tpush   %rbp',
        '/app/example.c:4',
        '  11d7d4:\te8 9f dc fb ff       \tcall   db478 <__interceptor_trampoline_malloc>',
        '/app/example.c:6',
        '  11d7ed:\tc3                   \tret',
        '',
        '000000000011d7f0 <asan.module_ctor>:',
        'asan.module_ctor():',
        '  11d7f0:\t55                   \tpush   %rbp',
        '  11d7f4:\te8 c7 85 fc ff       \tcall   e5dc0 <__asan_init>',
    ].join('\n');
    const main = ['main:', ' push   %rbp', ' call   db478 <__interceptor_trampoline_malloc>', ' ret'];

    function parserWithLineLimit(maxLinesOfAsm: number) {
        const props = (property: string, defaultValue?: unknown) =>
            property === 'maxLinesOfAsm' ? maxLinesOfAsm : defaultValue;
        return new AsmParser(props as PropertyGetter);
    }

    function texts(parser: AsmParser, asm: string, filters: Partial<ParseFiltersAndOutputOptions>) {
        return parser.processBinaryAsm(asm, filters).asm.map(line => line.text);
    }

    it('should filter out a statically linked runtime that has line information', () => {
        const result = new AsmParser().processBinaryAsm(objdump, {libraryCode: true});
        expect(result.asm.map(line => line.text)).toEqual(main);
        expect(result.asm.map(line => line.source?.line)).toEqual([undefined, 3, 4, 6]);
        expect(Object.keys(result.labelDefinitions ?? {})).toEqual(['main']);
    });

    it('should keep the runtime when library code is shown', () => {
        expect(texts(new AsmParser(), objdump, {}).filter(text => text.endsWith(':'))).toEqual([
            '__asan_report_load1_asm:',
            '__asan_report_load2_asm:',
            'main:',
            'asan.module_ctor:',
        ]);
    });

    it('should not count library functions towards the line limit', () => {
        expect(texts(parserWithLineLimit(5), objdump, {libraryCode: true})).toEqual(main);
    });

    it('should still truncate the user code at the line limit', () => {
        expect(texts(parserWithLineLimit(2), objdump, {libraryCode: true})).toEqual([
            'main:',
            ' push   %rbp',
            '[truncated; too many lines]',
        ]);
    });

    it('should truncate the runtime at the line limit when library code is shown', () => {
        expect(texts(parserWithLineLimit(4), objdump, {})).toEqual([
            '__asan_report_load1_asm:',
            ' jmp    e4290 <__asan_report_load1>',
            ' nopl   (%rax)',
            '__asan_report_load2_asm:',
            '[truncated; too many lines]',
        ]);
    });

    it('should truncate a function whose user code only shows up past the line limit', () => {
        const header = '/opt/compiler-explorer/gcc-15.1.0/include/c++/15.1.0/bits/stl_vector.h:100';
        const asm = [
            '0000000000401000 <main>:',
            'main():',
            header,
            '  401000:\t55                   \tpush   %rbp',
            '  401001:\t53                   \tpush   %rbx',
            '0000000000401002 <.Ltmp0>:',
            '  401002:\t90                   \tnop',
            '/app/example.cpp:4',
            '  401003:\tc3                   \tret',
        ].join('\n');
        const result = parserWithLineLimit(3).processBinaryAsm(asm, {libraryCode: true});
        expect(result.asm.map(line => line.text)).toEqual([
            'main:',
            ' push   %rbp',
            ' push   %rbx',
            '[truncated; too many lines]',
        ]);
        expect(Object.keys(result.labelDefinitions ?? {})).toEqual(['main']);
    });

    function functionFrom(file: string) {
        return ['0000000000401000 <helper()>:', 'helper()():', `${file}:3`, '  401000:\tc3                   \tret'];
    }

    it.each([
        ['in /app', '/app/src/helper.cpp'],
        ["in a local build's temporary directory", '/tmp/compiler-explorer-compiler2026ab/src/helper.cpp'],
        ["in a Windows build's temporary directory", 'C:\\Temp\\compiler-explorer-compiler2026ab\\src\\helper.cpp'],
    ])('should keep a function from another file of the program %s', (_, file) => {
        expect(texts(new AsmParser(), functionFrom(file).join('\n'), {libraryCode: true})).toEqual([
            'helper():',
            ' ret',
        ]);
    });

    it('should filter out a function whose code all comes from library headers', () => {
        const asm = functionFrom('/opt/compiler-explorer/gcc-15.1.0/include/c++/15.1.0/bits/vector.tcc').join('\n');
        expect(texts(new AsmParser(), asm, {libraryCode: true})).toEqual([]);
    });
});

describe('AsmParser directive filtering', () => {
    const parser = new AsmParser();
    // GCC wraps inline asm in #APP/#NO_APP and interleaves its own .loc markers there.
    const inlineAsm = `square(int):
	pushq	%rbp
#APP
# 3 "/app/example.cpp" 1
	.rept 5
nop
.endr
# 0 "" 2
	.loc 1 4 5 view .LVU3
#NO_APP
	movl	%edi, %eax
	.loc 1 5 1
	popq	%rbp
	ret`;

    it('should keep inline asm directives when filtering directives', () => {
        const result = parser.processAsm(inlineAsm, {directives: true});
        const lines = result.asm.map(line => line.text.trim());
        expect(lines).toContain('.rept 5');
        expect(lines).toContain('.endr');
    });

    it('should still drop source location markers inside an inline asm block', () => {
        const result = parser.processAsm(inlineAsm, {directives: true});
        const lines = result.asm.map(line => line.text.trim());
        expect(lines.some(line => line.startsWith('.loc'))).toBe(false);
    });

    it('should keep filtering directives once the inline asm block has ended', () => {
        const result = parser.processAsm(
            `#APP
	.rept 5
#NO_APP
	.p2align 4
	ret`,
            {directives: true},
        );
        const lines = result.asm.map(line => line.text.trim());
        expect(lines).toContain('.rept 5');
        expect(lines).not.toContain('.p2align 4');
    });

    // What GCC 13.2 and later emit for a file that only includes <iostream>, cut down.
    it('should filter the debug info after an #APP block that is never closed', () => {
        const result = parser.processAsm(
            `	.text
.Ltext0:
	.file 0 "/app" "/app/example.cpp"
#APP
	.globl _ZSt21ios_base_library_initv
.Letext0:
	.file 1 "<built-in>"
	.section	.debug_info,"",@progbits
.Ldebug_info0:
	.long	0x1c5d
	.value	0x5
	.section	.debug_abbrev,"",@progbits
.Ldebug_abbrev0:
	.uleb128 0x1
	.ident	"GCC: (GNU) 16.2.0"
	.section	.note.GNU-stack,"",@progbits`,
            {directives: true, labels: true, commentOnly: true},
        );
        expect(result.asm.map(line => line.text.trim())).toEqual(['.globl _ZSt21ios_base_library_initv']);
    });

    it('should keep the directives of top-level asm that ends the unit', () => {
        const result = parser.processAsm(
            `	.globl	_Z1fv
_Z1fv:
	ret
	.size	_Z1fv, .-_Z1fv
#APP
	.rept 3
nop
.endr
.Letext0:
	.section	.debug_info,"",@progbits
.Ldebug_info0:
	.long	0x57`,
            {directives: true, labels: true, commentOnly: true},
        );
        expect(result.asm.map(line => line.text.trim())).toEqual(['_Z1fv:', 'ret', '.rept 3', 'nop', '.endr']);
    });

    it('should close an unterminated #APP block at the .ident when there is no debug info', () => {
        const result = parser.processAsm(
            `#APP
	.globl _ZSt21ios_base_library_initv
	.ident	"GCC: (GNU) 16.2.0"
	.section	.note.GNU-stack,"",@progbits`,
            {directives: true, commentOnly: true},
        );
        expect(result.asm.map(line => line.text.trim())).toEqual(['.globl _ZSt21ios_base_library_initv']);
    });
});

describe('AsmParser numeric local labels', () => {
    const parser = new AsmParser();
    const filters = {directives: true, labels: true};

    it('should keep a local label referenced forwards', () => {
        const input = ['square:', '  bne $2,$0,1f', '  break 7', '1:', '  mflo $2', '  jr $31'].join('\n');
        const lines = parser.processAsm(input, filters).asm.map(line => line.text);
        expect(lines).toContain('1:');
    });

    it('should keep a local label referenced backwards', () => {
        const input = ['loop:', '2:', '  add r0, r0, #1', '  b 2b', '  ret'].join('\n');
        const lines = parser.processAsm(input, filters).asm.map(line => line.text);
        expect(lines).toContain('2:');
    });

    it('should still drop a local label nothing refers to', () => {
        const input = ['square:', '  mov r0, #0x1f', '1:', '  ret'].join('\n');
        const lines = parser.processAsm(input, filters).asm.map(line => line.text);
        expect(lines).not.toContain('1:');
    });

    it('should not read a NEON element size as a reference', () => {
        const input = ['square:', '  add v0.16b, v1.16b, v2.16b', '16:', '  ret'].join('\n');
        const lines = parser.processAsm(input, filters).asm.map(line => line.text);
        expect(lines).not.toContain('16:');
    });
});

describe('PTXAsmParser tests', () => {
    const parser = new PTXAsmParser();

    describe('Identifying opcodes', () => {
        it('should identify regular PTX opcodes', () => {
            expect(parser.hasOpcode('  mov.u32 	%r25, %ctaid.x;')).toBe(true);
            expect(parser.hasOpcode('  ld.global.v4.b32 { %r1, %r2, %r3, %r4 }, [ %rd1 + 0 ];')).toBe(true);
            expect(parser.hasOpcode('  mul.wide.s32 	%rd10, %r31, 4;')).toBe(true);
        });
        it('should identify PTX opcodes with predicate', () => {
            expect(parser.hasOpcode('  @%p1 ld.global.v4.b32 { %r1, %r2, %r3, %r4 }, [ %rd1 + 0 ];')).toBe(true);
            expect(parser.hasOpcode('  @!%p1 bra LBB6_2;')).toBe(true);
            expect(parser.hasOpcode('  @%r789 bra LBB6_2;')).toBe(true);
        });
        it('should identify PTX opcodes wrapped in braces', () => {
            expect(parser.hasOpcode('{fma.rn.f16x2 %r9,%r2,%r3,%r270;')).toBe(true);
            expect(parser.hasOpcode('  {fma.rn.f16x2 %r9,%r2,%r3,%r270;')).toBe(true);
            expect(parser.hasOpcode('{ add.s32 %r1, %r2, %r3;')).toBe(true);
        });
    });

    describe('Nested brace indentation', () => {
        it('should indent content inside callseq blocks', () => {
            const input = `{ // callseq 0, 0
.param .b64 param0;
st.param.b64 [param0+0], %rd2;
}`;
            const result = parser.processAsm(input, {});
            const lines = result.asm.map(line => line.text);

            expect(lines[0]).toBe('{ // callseq 0, 0');
            expect(lines[1]).toBe('\t.param .b64 param0;');
            expect(lines[2]).toBe('\tst.param.b64 [param0+0], %rd2;');
            expect(lines[3]).toBe('}');
        });

        it('should properly indent function calls inside nested braces', () => {
            const input = `{ // callseq 0, 0
call.uni (retval0),
vprintf,
(
param0,
param1
);
}`;
            const result = parser.processAsm(input, {});
            const lines = result.asm.map(line => line.text);

            expect(lines[0]).toBe('{ // callseq 0, 0');
            expect(lines[1]).toBe('\tcall.uni (retval0),');
            expect(lines[2]).toBe('\tvprintf,');
            expect(lines[3]).toBe('\t(');
            expect(lines[4]).toBe('\tparam0,');
            expect(lines[5]).toBe('\tparam1');
            expect(lines[6]).toBe('\t);');
            expect(lines[7]).toBe('}');
        });

        it('should have proper indentation logic', () => {
            const input = `.visible .entry _Z6kernelv()
{
mov.u64 %rd1, $str;
{ // callseq 0, 0
call.uni (retval0),
vprintf,
(
param0,
param1
);
}
}`;
            const result = parser.processAsm(input, {});
            const lines = result.asm.map(line => line.text);

            expect(lines[0]).toBe('.visible .entry _Z6kernelv()');
            expect(lines[1]).toBe('{');
            expect(lines[2]).toBe('\tmov.u64 %rd1, $str;');
            expect(lines[3]).toBe('\t{ // callseq 0, 0');
            expect(lines[4]).toBe('\t\tcall.uni (retval0),');
            expect(lines[5]).toBe('\t\tvprintf,');
            expect(lines[6]).toBe('\t\t(');
            expect(lines[7]).toBe('\t\tparam0,');
            expect(lines[8]).toBe('\t\tparam1');
            expect(lines[9]).toBe('\t\t);');
            expect(lines[10]).toBe('\t}');
            expect(lines[11]).toBe('}');
        });
    });

    describe('Label handling', () => {
        it('should never indent labels', () => {
            const input = `{ // callseq 0, 0
$L__BB0_3:
mov.u64 %rd1, $str;
}`;
            const result = parser.processAsm(input, {});
            const lines = result.asm.map(line => line.text);

            expect(lines[0]).toBe('{ // callseq 0, 0');
            expect(lines[1]).toBe('$L__BB0_3:');
            expect(lines[2]).toBe('\tmov.u64 %rd1, $str;');
            expect(lines[3]).toBe('}');
        });
    });

    describe('Braced instructions', () => {
        it('should not drop FMA instructions wrapped in braces on separate lines', () => {
            const input = `{fma.rn.f16x2 %r9,%r2,%r3,%r270;
}
{fma.rn.f16x2 %r13,%r2,%r3,%r9;
}`;
            const result = parser.processAsm(input, {});
            const lines = result.asm.map(line => line.text);

            expect(lines.length).toBe(4);
            expect(lines[0]).toBe('{fma.rn.f16x2 %r9,%r2,%r3,%r270;');
            expect(lines[1]).toBe('}');
            expect(lines[2]).toBe('{fma.rn.f16x2 %r13,%r2,%r3,%r9;');
            expect(lines[3]).toBe('}');
        });

        it('should preserve braced instructions inside functions', () => {
            const input = `.visible .entry kernel()
{
mov.u32 %r1, 0;
{fma.rn.f16x2 %r9,%r2,%r3,%r270;
}
{fma.rn.f16x2 %r13,%r2,%r3,%r9;
}
ret;
}`;
            const result = parser.processAsm(input, {});
            const lines = result.asm.map(line => line.text);

            expect(lines[0]).toBe('.visible .entry kernel()');
            expect(lines[1]).toBe('{');
            expect(lines[2]).toBe('\tmov.u32 %r1, 0;');
            expect(lines[3]).toBe('\t{fma.rn.f16x2 %r9,%r2,%r3,%r270;');
            expect(lines[4]).toBe('\t}');
            expect(lines[5]).toBe('\t{fma.rn.f16x2 %r13,%r2,%r3,%r9;');
            expect(lines[6]).toBe('\t}');
            expect(lines[7]).toBe('\tret;');
            expect(lines[8]).toBe('}');
        });
    });

    describe('Trim filter', () => {
        it('should convert tabs to spaces when trim filter is enabled', () => {
            const input = `\t\t\tcall.uni (retval0),
\t\t\tvprintf,
\t\t\t(
\t\t\tparam0,
\t\t\tparam1
\t\t\t);`;
            const result = parser.processAsm(input, {trim: true});
            const lines = result.asm.map(line => line.text);

            expect(lines[0]).toBe('      call.uni (retval0),');
            expect(lines[1]).toBe('        vprintf,');
            expect(lines[2]).toBe('        (');
            expect(lines[3]).toBe('        param0,');
            expect(lines[4]).toBe('        param1');
            expect(lines[5]).toBe('      );');
        });

        it('should preserve structure but compress whitespace with trim filter', () => {
            const input = `{ // callseq 0, 0
\t\t\tcall.uni (retval0),
\t\t\tvprintf,
}`;
            const result = parser.processAsm(input, {trim: true});
            const lines = result.asm.map(line => line.text);

            expect(lines[0]).toBe('{ // callseq 0, 0');
            expect(lines[1]).toBe('  call.uni (retval0),');
            expect(lines[2]).toBe('  vprintf,');
            expect(lines[3]).toBe('}');
        });

        it('should preserve nested indentation structure with trim filter', () => {
            const input = `.visible .entry _Z6kernelv()
{
mov.u64 %rd1, $str;
{ // callseq 0, 0
call.uni (retval0),
vprintf,
}
}`;
            const result = parser.processAsm(input, {trim: true});
            const lines = result.asm.map(line => line.text);

            expect(lines[0]).toBe('.visible .entry _Z6kernelv()');
            expect(lines[1]).toBe('{');
            expect(lines[2]).toBe('  mov.u64 %rd1, $str;');
            expect(lines[3]).toBe('  { // callseq 0, 0');
            expect(lines[4]).toBe('    call.uni (retval0),');
            expect(lines[5]).toBe('    vprintf,');
            expect(lines[6]).toBe('  }');
            expect(lines[7]).toBe('}');
        });
    });

    describe('Directive filter', () => {
        it('should remove directives when directive filter is enabled', () => {
            const input = `
 	.file	1 "/tmp/compiler-explorer-compilerk2brih/example.py"
    .section	.debug_abbrev
    {
	.b8 1
	}
    .section	.debug_info
    {
    .b32 70
    .b32 .debug_abbrev
    }
    .section	.debug_macinfo	{	}
`;
            const result = parser.processAsm(input, {directives: true});
            const lines = result.asm.map(line => line.text);
            expect(lines.length).toBe(1);
            expect(lines[0]).toBe('');
        });
    });
});

describe('MlirAsmParser tests', () => {
    const parser = new MlirAsmParser();

    describe('Location handling', () => {
        it('should process MLIR with location references', () => {
            const input = `
#loc = loc("<source>":7:0)
module {
  tt.func public @add_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc("<source>":7:0), %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc("<source>":7:0), %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc("<source>":7:0), %arg3: i32 {tt.divisibility = 16 : i32} loc("<source>":7:0)) attributes {noinline = false} {
    %c1024_i32 = arith.constant 1024 : i32 loc(#loc1)
    %0 = tt.get_program_id x : i32 loc(#loc2)
  } loc(#loc)
} loc(#loc)
#loc1 = loc(unknown)
#loc2 = loc("<source>":14:24)`;

            const result = parser.processAsm(input, {});

            // Verify location definitions are removed
            expect(result.asm.find(line => line.text.includes('#loc = loc'))).toBeUndefined();
            expect(result.asm.find(line => line.text.includes('#loc1 = loc'))).toBeUndefined();
            expect(result.asm.find(line => line.text.includes('#loc2 = loc'))).toBeUndefined();

            // Verify location references are removed from displayed text
            expect(result.asm.find(line => line.text.includes('loc(#loc)'))).toBeUndefined();
            expect(result.asm.find(line => line.text.includes('loc(#loc1)'))).toBeUndefined();
            expect(result.asm.find(line => line.text.includes('loc(#loc2)'))).toBeUndefined();

            // Verify inline locations are removed
            expect(result.asm.find(line => line.text.includes('loc("<source>"'))).toBeUndefined();

            // Verify source information is correctly associated
            const programIdLine = result.asm.find(line => line.text.includes('tt.get_program_id'));
            expect(programIdLine).toBeDefined();
            expect(programIdLine?.source).toBeDefined();
            expect(programIdLine?.source?.file).toBe('<source>');
            expect(programIdLine?.source?.line).toBe(14);
            expect(programIdLine?.source?.column).toBe(24);

            // Verify unknown locations are not associated with source information
            const constantLine = result.asm.find(line => line.text.includes('arith.constant'));
            expect(constantLine).toBeDefined();
            expect(constantLine?.source).toBeNull();

            // Verify the structure is preserved
            const moduleStartLine = result.asm.find(line => line.text.includes('module {'));
            expect(moduleStartLine).toBeDefined();

            const funcLine = result.asm.find(line => line.text.includes('tt.func public @add_kernel('));
            expect(funcLine).toBeDefined();
            expect(funcLine?.text?.includes('%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, ')).toBe(true);
            expect(funcLine?.text?.includes('%arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, ')).toBe(true);
            expect(funcLine?.text?.includes('%arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, ')).toBe(true);
            expect(funcLine?.text?.includes('%arg3: i32 {tt.divisibility = 16 : i32})')).toBe(true);

            const constLine = result.asm.find(line => line.text.includes('arith.constant 1024'));
            expect(constLine).toBeDefined();
        });

        it('should resolve nested named MLIR locations', () => {
            const input = `#loc = loc("<source>":16:0)
#loc1 = loc("<source>":17:24)
#loc2 = loc("<source>":18:28)
#loc19 = loc("pid"(#loc1))
#loc20 = loc("block_start"(#loc2))
#loc21 = loc(unknown)
#loc22 = loc("unknown_value"(#loc21))
module {
  tt.func public @kernel(%x_ptr: !tt.ptr<f32> loc("x_ptr"(#loc))) attributes {noinline = false} {
    %pid = tt.get_program_id x : i32 loc(#loc19)
    %block_start = arith.muli %pid, %c1024_i32 : i32 loc(#loc20)
    %unknown = arith.constant 0 : i32 loc(#loc22)
  } loc(#loc)
}`;

            const result = parser.processAsm(input, {});

            expect(result.asm[0].text).toBe('module {');
            expect(result.asm.find(line => line.text.includes('#loc'))).toBeUndefined();
            expect(result.asm.find(line => line.text.includes('loc('))).toBeUndefined();

            const functionLine = result.asm.find(line => line.text.includes('tt.func public @kernel'));
            expect(functionLine?.source?.line).toBe(16);
            expect(functionLine?.source?.column).toBe(0);

            const pidLine = result.asm.find(line => line.text.includes('%pid = tt.get_program_id'));
            expect(pidLine?.source?.line).toBe(17);
            expect(pidLine?.source?.column).toBe(24);

            const blockStartLine = result.asm.find(line => line.text.includes('%block_start = arith.muli'));
            expect(blockStartLine?.source?.line).toBe(18);
            expect(blockStartLine?.source?.column).toBe(28);

            const unknownLine = result.asm.find(line => line.text.includes('%unknown = arith.constant'));
            expect(unknownLine?.source).toBeNull();
        });
    });
});

describe('ResolcRiscVAsmParser tests', () => {
    const parser = new ResolcRiscVAsmParser();

    function expectParsedAsmResult(result: ParsedAsmResult, expected: ParsedAsmResult): void {
        expect(result.labelDefinitions).toEqual(expected.labelDefinitions);
        expect(result.asm.length).toEqual(expected.asm.length);
        for (let i = 0; i < result.asm.length; i++) {
            expect(result.asm[i]).toMatchObject(expected.asm[i]);
        }
    }

    it('should identify RISC-V instruction info and source line numbers', () => {
        const filters: Partial<ParseFiltersAndOutputOptions> = {binaryObject: true};
        const riscv = `
000000000000027a <__entry>:
; __entry():
; path/to/example.sol.Square.yul:1
     27a: 41 11        	addi	sp, sp, -0x10

000000000000028c <.Lpcrel_hi4>:
; .Lpcrel_hi4():
     28c: 97 05 00 00  	auipc	a1, 0x0
; path/to/example.sol.Square.yul:1
     2a8: 1d 71        	addi	sp, sp, -0x60
     2aa: 86 ec        	sd	ra, 0x58(sp)
; path/to/example.sol.Square.yul:7
     2ca: e7 80 00 00  	jalr	ra <.Lpcrel_hi4+0x3a>`;

        const expected: ParsedAsmResult = {
            asm: [
                {
                    text: '__entry:',
                    source: null,
                },
                {
                    text: ' addi	sp, sp, -0x10',
                    address: 0x27a,
                    opcodes: ['41', '11'],
                    source: {
                        line: 1,
                        file: null,
                    },
                },
                {
                    text: '.Lpcrel_hi4:',
                    source: null,
                },
                {
                    text: ' auipc	a1, 0x0',
                    address: 0x28c,
                    opcodes: ['97', '05', '00', '00'],
                    source: {
                        line: 1,
                        file: null,
                    },
                },
                {
                    text: ' addi	sp, sp, -0x60',
                    address: 0x2a8,
                    opcodes: ['1d', '71'],
                    source: {
                        line: 1,
                        file: null,
                    },
                },
                {
                    text: ' sd	ra, 0x58(sp)',
                    address: 0x2aa,
                    opcodes: ['86', 'ec'],
                    source: {
                        line: 1,
                        file: null,
                    },
                },
                {
                    text: ' jalr	ra <.Lpcrel_hi4+0x3a>',
                    address: 0x2ca,
                    opcodes: ['e7', '80', '00', '00'],
                    source: {
                        line: 7,
                        file: null,
                    },
                },
            ],
            labelDefinitions: {
                __entry: 1,
                ['.Lpcrel_hi4']: 3,
            },
        };

        const result = parser.processAsm(riscv, filters);
        expectParsedAsmResult(result, expected);
    });
});

describe('PolkaVMAsmParser tests', () => {
    const parser = new PolkaVMAsmParser();

    function expectParsedAsmResult(result: ParsedAsmResult, expected: ParsedAsmResult): void {
        expect(result.labelDefinitions).toEqual(expected.labelDefinitions);
        expect(result.asm.length).toEqual(expected.asm.length);
        for (let i = 0; i < result.asm.length; i++) {
            expect(result.asm[i]).toMatchObject(expected.asm[i]);
        }
    }

    // Note: We currently have no source mappings from PVM.
    it('should identify PVM instruction info', () => {
        const filters: Partial<ParseFiltersAndOutputOptions> = {
            binaryObject: false,
            commentOnly: false,
        };
        const pvm = `
// Code size = 1078 bytes

<__entry>:
      : @0 (gas: 6)
     0: sp = sp + 0xfffffffffffffff0
     3: u64 [sp + 0x8] = ra
     6: u64 [sp] = s0
     8: s0 = a0 & 0x1
    11: ecalli 2 // 'call_data_size'
    13: fallthrough
      : @1 (gas: 2)
    14: u32 [0x20000] = a0`;

        const expected: ParsedAsmResult = {
            asm: [
                {
                    text: '// Code size = 1078 bytes',
                    source: null,
                },
                {
                    text: '__entry:',
                    source: null,
                },
                {
                    text: '        @0 (gas: 6)',
                    source: null,
                },
                {
                    text: '        sp = sp + 0xfffffffffffffff0',
                    address: 0,
                    source: null,
                },
                {
                    text: '        u64 [sp + 0x8] = ra',
                    address: 3,
                    source: null,
                },
                {
                    text: '        u64 [sp] = s0',
                    address: 6,
                    source: null,
                },
                {
                    text: '        s0 = a0 & 0x1',
                    address: 8,
                    source: null,
                },
                {
                    text: "        ecalli 2 // 'call_data_size'",
                    address: 11,
                    source: null,
                },
                {
                    text: '        fallthrough',
                    address: 13,
                    source: null,
                },
                {
                    text: '        @1 (gas: 2)',
                    source: null,
                },
                {
                    text: '        u32 [0x20000] = a0',
                    address: 14,
                    source: null,
                },
            ],
            labelDefinitions: {__entry: 2},
        };

        const result = parser.process(pvm, filters);
        expectParsedAsmResult(result, expected);
    });
});

describe('Text asm line limit', () => {
    const truncatedLine = '[truncated; too many lines]';
    const x86Asm = (lines: number) => 'square(int):\n' + '        imul    edi, edi\n'.repeat(lines - 1);
    // nvcc puts the file table after the code that refers to it
    const ptxAsm = (lines: number) =>
        '.visible .entry _Z6kernelPf(\n{\n' +
        '\t.loc\t1 3 5\n\tadd.f32 \t%f2, %f1, %f1;\n'.repeat(Math.ceil((lines - 3) / 2)) +
        '\t.file\t1 "/app/example.cu"\n';

    function withLimit<T>(parserClass: new (props: PropertyGetter) => T, maxLinesOfTextAsm: number) {
        const props = (property: string, defaultValue?: unknown) =>
            property === 'maxLinesOfTextAsm' ? maxLinesOfTextAsm : defaultValue;
        return new parserClass(props as PropertyGetter);
    }

    it('should truncate text asm output over the limit', () => {
        const asm = x86Asm(1000);
        const result = withLimit(AsmParser, 100).process(asm, {});
        expect(result.asm).toHaveLength(101);
        expect(result.asm.slice(0, 100)).toEqual(new AsmParser().process(asm, {}).asm.slice(0, 100));
        expect(result.asm[100].text).toEqual(truncatedLine);
    });

    it('should leave text asm output up to the limit unchanged', () => {
        const parser = withLimit(AsmParser, 100);
        for (const asm of [x86Asm(100), x86Asm(100).trimEnd(), x86Asm(99)]) {
            expect(parser.process(asm, {})).toEqual({...parser.processAsm(asm, {}), parsingTime: expect.any(Number)});
        }
    });

    it('should apply the limit to the filtered output, not the input', () => {
        const asm = '\t.section\t.debug_info\n'.repeat(1000) + x86Asm(50) + '\t.section\t.debug_str\n'.repeat(1000);
        const filters = {directives: true, labels: true};
        const result = withLimit(AsmParser, 100).process(asm, filters);
        expect(result).toEqual({...new AsmParser().process(asm, filters), parsingTime: expect.any(Number)});
        expect(result.asm.map(line => line.text)).not.toContain(truncatedLine);
    });

    it('should not apply the text limit to binary asm', () => {
        const parser = withLimit(AsmParser, 10);
        const asm = '0000000000000000 <main>:\n' + '   0:\t90                   \tnop\n'.repeat(49);
        const lines = parser.process(asm, {binary: true}).asm.map(line => line.text);
        expect(lines).toHaveLength(50);
        expect(lines).not.toContain(truncatedLine);
    });

    it('should truncate PTX output over the limit, keeping source info from the file table after the cut', () => {
        const asm = ptxAsm(1000);
        const result = withLimit(PTXAsmParser, 100).process(asm, {});
        expect(result.asm).toHaveLength(101);
        expect(result.asm.slice(0, 100)).toEqual(new PTXAsmParser().process(asm, {}).asm.slice(0, 100));
        expect(result.asm.some(line => line.source?.line === 3)).toBe(true);
        expect(result.asm[100].text).toEqual(truncatedLine);
    });

    it('should truncate text SASS output over the limit', () => {
        const parser = withLimit(SassAsmParser, 100);
        const sass = '\t/*0000*/ \tIMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28] ;\n'.repeat(1000);
        const result = parser.process(sass, {});
        expect(result.asm).toHaveLength(101);
        expect(result.asm[100].text).toEqual(truncatedLine);
    });

    it('should limit text asm output to 200000 lines by default', () => {
        const result = new AsmParser().process(x86Asm(300000), {});
        expect(result.asm).toHaveLength(200001);
        expect(result.asm.at(-1)?.text).toEqual(truncatedLine);
    });
});
