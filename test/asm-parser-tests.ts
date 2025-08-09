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
import {PTXAsmParser} from '../lib/parsers/asm-parser-ptx.js';

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
    });
});
