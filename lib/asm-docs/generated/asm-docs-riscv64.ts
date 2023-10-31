import {AssemblyInstructionInfo} from '../base.js';

export function getAsmOpcode(opcode: string | undefined): AssemblyInstructionInfo | undefined {
    if (!opcode) return;
    switch (opcode.toUpperCase()) {
        case "BEQ":
            return {
                "html": "<div>\n<span class=\"opcode\">rs1, rs2, bimm12</span>\n\n<div>BEQ and BNE take the branch if registers rs1 and rs2 are equal or unequal respectively</div>\n</div>",
                "tooltip": "BEQ and BNE take the branch if registers rs1 and rs2 are equal or unequal respectively",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/rv32.html#conditional-branches"
            };

        case "BLT":
            return {
                "html": "<div>\n<span class=\"opcode\">rs1, rs2, bimm12</span>\n\n<div>BLT and BLTU take the branch if rs1 is less than rs2 , using signed and unsigned comparison respectively</div>\n<div>Note, BGT, BGTU, BLE, and BLEU can be synthesized by reversing the operands to BLT, BLTU, BGE, and BGEU, respectively.</div>\n</div>",
                "tooltip": "BLT and BLTU take the branch if rs1 is less than rs2 , using signed and unsigned comparison respectively\nNote, BGT, BGTU, BLE, and BLEU can be synthesized by reversing the operands to BLT, BLTU, BGE, and BGEU, respectively.",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/rv32.html#conditional-branches"
            };

        case "BGE":
            return {
                "html": "<div>\n<span class=\"opcode\">rs1, rs2, bimm12</span>\n\n<div>BGE and BGEU take the branch if rs1 is greater than or equal to rs2 , using signed and unsigned comparison respectively</div>\n</div>",
                "tooltip": "BGE and BGEU take the branch if rs1 is greater than or equal to rs2 , using signed and unsigned comparison respectively",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/rv32.html#conditional-branches"
            };

        case "BLTU":
            return {
                "html": "<div>\n<span class=\"opcode\">rs1, rs2, bimm12</span>\n\n<div>Signed array bounds may be checked with a single BLTU instruction, since any negative index will compare greater than any nonnegative bound.</div>\n</div>",
                "tooltip": "Signed array bounds may be checked with a single BLTU instruction, since any negative index will compare greater than any nonnegative bound.",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/rv32.html#conditional-branches"
            };

        case "JALR":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, imm12</span>\n\n<div>The indirect jump instruction JALR (jump and link register) uses the I-type encoding</div>\n<div>The JALR instruction was defined to enable a two-instruction sequence to jump anywhere in a 32-bit absolute address range</div>\n<div>Note that the JALR instruction does not treat the 12-bit immediate as multiples of 2 bytes, unlike the conditional branch instructions</div>\n<div>In practice, most uses of JALR will have either a zero immediate or be paired with a LUI or AUIPC, so the slight reduction in range is not significant.</div>\n<div>Clearing the least-significant bit when calculating the JALR target address both simplifies the hardware slightly and allows the low bit of function pointers to be used to store auxiliary information</div>\n<div>When used with a base rs1 = x0 , JALR can be used to implement a single instruction subroutine call to the lowest</div>\n<div>JALR instructions should push/pop a RAS as shown in the Table\u00c3\u0082\u00c2\u00a0</div>\n</div>",
                "tooltip": "The indirect jump instruction JALR (jump and link register) uses the I-type encoding\nThe JALR instruction was defined to enable a two-instruction sequence to jump anywhere in a 32-bit absolute address range\nNote that the JALR instruction does not treat the 12-bit immediate as multiples of 2 bytes, unlike the conditional branch instructions\nIn practice, most uses of JALR will have either a zero immediate or be paired with a LUI or AUIPC, so the slight reduction in range is not significant.\nClearing the least-significant bit when calculating the JALR target address both simplifies the hardware slightly and allows the low bit of function pointers to be used to store auxiliary information\nWhen used with a base rs1 = x0 , JALR can be used to implement a single instruction subroutine call to the lowest\nJALR instructions should push/pop a RAS as shown in the Table\u00c3\u0082\u00c2",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/rv32.html#unconditional-jumps"
            };

        case "JAL":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, jimm20</span>\n\n<div>See the descriptions of the JAL and JALR instructions.</div>\n</div>",
                "tooltip": "See the descriptions of the JAL and JALR instructions.",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/rv32.html#programmers-model-for-base-integer-isa"
            };

        case "LUI":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, imm20</span>\n\n<div>LUI (load upper immediate) is used to build 32-bit constants and uses the U-type format</div>\n<div>LUI places the U-immediate value in the top 20 bits of the destination register rd , filling in the lowest 12 bits with zeros.</div>\n</div>",
                "tooltip": "LUI (load upper immediate) is used to build 32-bit constants and uses the U-type format\nLUI places the U-immediate value in the top 20 bits of the destination register rd , filling in the lowest 12 bits with zeros.",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/rv32.html#integer-register-immediate-instructions"
            };

        case "AUIPC":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, imm20</span>\n\n<div>AUIPC (add upper immediate to pc ) is used to build pc -relative addresses and uses the U-type format</div>\n<div>AUIPC forms a 32-bit offset from the 20-bit U-immediate, filling in the lowest 12 bits with zeros, adds this offset to the address of the AUIPC instruction, then places the result in register rd .</div>\n<div>The AUIPC instruction supports two-instruction sequences to access arbitrary offsets from the PC for both control-flow transfers and data accesses</div>\n<div>The combination of an AUIPC and the 12-bit immediate in a JALR can transfer control to any 32-bit PC-relative address, while an AUIPC plus the 12-bit immediate offset in regular load or store instructions can access any 32-bit PC-relative data address.</div>\n</div>",
                "tooltip": "AUIPC (add upper immediate to pc ) is used to build pc -relative addresses and uses the U-type format\nAUIPC forms a 32-bit offset from the 20-bit U-immediate, filling in the lowest 12 bits with zeros, adds this offset to the address of the AUIPC instruction, then places the result in register rd .\nThe AUIPC instruction supports two-instruction sequences to access arbitrary offsets from the PC for both control-flow transfers and data accesses\nThe combination of an AUIPC and the 12-bit immediate in a JALR can transfer control to any 32-bit PC-relative address, while an AUIPC plus the 12-bit immediate offset in regular load or store instructions can access any 32-bit PC-relative data address.",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/rv32.html#integer-register-immediate-instructions"
            };

        case "ADDI":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, imm12</span>\n\n<div>ADDI adds the sign-extended 12-bit immediate to register rs1</div>\n<div>ADDI rd, rs1, 0 is used to implement the MV rd, rs1 assembler pseudoinstruction.</div>\n</div>",
                "tooltip": "ADDI adds the sign-extended 12-bit immediate to register rs1\nADDI rd, rs1, 0 is used to implement the MV rd, rs1 assembler pseudoinstruction.",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/rv32.html#integer-register-immediate-instructions"
            };

        case "SLLI":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1</span>\n\n<div>The right shift type is encoded in bit 30. SLLI is a logical left shift (zeros are shifted into the lower bits); SRLI is a logical right shift (zeros are shifted into the upper bits); and SRAI is an arithmetic right shift (the original sign bit is copied into the vacated upper bits).</div>\n</div>",
                "tooltip": "The right shift type is encoded in bit 30. SLLI is a logical left shift (zeros are shifted into the lower bits); SRLI is a logical right shift (zeros are shifted into the upper bits); and SRAI is an arithmetic right shift (the original sign bit is copied into the vacated upper bits).",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/rv32.html#integer-register-immediate-instructions"
            };

        case "SLTI":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, imm12</span>\n\n<div>SLTI (set less than immediate) places the value 1 in register rd rs1 is less than the sign-extended immediate when both are treated as signed numbers, else 0 is written to rd</div>\n</div>",
                "tooltip": "SLTI (set less than immediate) places the value 1 in register rd rs1 is less than the sign-extended immediate when both are treated as signed numbers, else 0 is written to rd",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/rv32.html#integer-register-immediate-instructions"
            };

        case "SLTIU":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, imm12</span>\n\n<div>SLTIU is similar but compares the values as unsigned numbers (i.e., the immediate is first sign-extended to XLEN bits then treated as an unsigned number)</div>\n<div>Note, SLTIU rd, rs1, 1 sets rd rs1 equals zero, otherwise sets rd to 0 (assembler pseudoinstruction SEQZ rd, rs ).</div>\n</div>",
                "tooltip": "SLTIU is similar but compares the values as unsigned numbers (i.e., the immediate is first sign-extended to XLEN bits then treated as an unsigned number)\nNote, SLTIU rd, rs1, 1 sets rd rs1 equals zero, otherwise sets rd to 0 (assembler pseudoinstruction SEQZ rd, rs ).",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/rv32.html#integer-register-immediate-instructions"
            };

        case "XORI":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, imm12</span>\n\n<div>Note, XORI rd, rs1, -1 rs1 (assembler pseudoinstruction NOT rd, rs ).</div>\n</div>",
                "tooltip": "Note, XORI rd, rs1, -1 rs1 (assembler pseudoinstruction NOT rd, rs ).",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/rv32.html#integer-register-immediate-instructions"
            };

        case "ANDI":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, imm12</span>\n\n<div>ANDI, ORI, XORI are logical operations that perform bitwise AND, OR, and XOR on register rs1 and the sign-extended 12-bit immediate and place the result in rd</div>\n</div>",
                "tooltip": "ANDI, ORI, XORI are logical operations that perform bitwise AND, OR, and XOR on register rs1 and the sign-extended 12-bit immediate and place the result in rd",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/rv32.html#integer-register-immediate-instructions"
            };

        case "ADD":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, rs2</span>\n\n<div>add t0, t1, t2 slti t3, t2, 0 slt t4, t0, t1 bne t3, t4, overflow In RV64I, checks of 32-bit signed additions can be optimized further by comparing the results of ADD and ADDW on the operands.</div>\n</div>",
                "tooltip": "add t0, t1, t2 slti t3, t2, 0 slt t4, t0, t1 bne t3, t4, overflow In RV64I, checks of 32-bit signed additions can be optimized further by comparing the results of ADD and ADDW on the operands.",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/rv32.html#integer-computational-instructions"
            };

        case "SUB":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, rs2</span>\n\n<div>SUB performs the subtraction of rs2 from rs1</div>\n</div>",
                "tooltip": "SUB performs the subtraction of rs2 from rs1",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/rv32.html#integer-register-register-operations"
            };

        case "SLL":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, rs2</span>\n\n<div>SLL, SRL, and SRA perform logical left, logical right, and arithmetic right shifts on the value in register rs1 by the shift amount held in the lower 5 bits of register rs2 .</div>\n</div>",
                "tooltip": "SLL, SRL, and SRA perform logical left, logical right, and arithmetic right shifts on the value in register rs1 by the shift amount held in the lower 5 bits of register rs2 .",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/rv32.html#integer-register-register-operations"
            };

        case "SLT":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, rs2</span>\n\n<div>SLT and SLTU perform signed and unsigned compares respectively, writing 1 to rd if</div>\n</div>",
                "tooltip": "SLT and SLTU perform signed and unsigned compares respectively, writing 1 to rd if",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/rv32.html#integer-register-register-operations"
            };

        case "SLTU":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, rs2</span>\n\n<div>Note, SLTU rd , x0 , rs2 sets rd to 1 if rs2 is not equal to zero, otherwise sets rd to zero (assembler pseudoinstruction SNEZ rd, rs )</div>\n</div>",
                "tooltip": "Note, SLTU rd , x0 , rs2 sets rd to 1 if rs2 is not equal to zero, otherwise sets rd to zero (assembler pseudoinstruction SNEZ rd, rs )",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/rv32.html#integer-register-register-operations"
            };

        case "AND":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, rs2</span>\n\n<div>AND, OR, and XOR perform bitwise logical operations.</div>\n</div>",
                "tooltip": "AND, OR, and XOR perform bitwise logical operations.",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/rv32.html#integer-register-register-operations"
            };

        case "LB":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, imm12</span>\n\n<div>LB and LBU are defined analogously for 8-bit values</div>\n</div>",
                "tooltip": "LB and LBU are defined analogously for 8-bit values",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/rv32.html#load-and-store-instructions"
            };

        case "LH":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, imm12</span>\n\n<div>LH loads a 16-bit value from memory, then sign-extends to 32-bits before storing in rd</div>\n</div>",
                "tooltip": "LH loads a 16-bit value from memory, then sign-extends to 32-bits before storing in rd",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/rv32.html#load-and-store-instructions"
            };

        case "LW":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, imm12</span>\n\n<div>The LW instruction loads a 32-bit value from memory into rd</div>\n</div>",
                "tooltip": "The LW instruction loads a 32-bit value from memory into rd",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/rv32.html#load-and-store-instructions"
            };

        case "LHU":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, imm12</span>\n\n<div>LHU loads a 16-bit value from memory but then zero extends to 32-bits before storing in rd</div>\n</div>",
                "tooltip": "LHU loads a 16-bit value from memory but then zero extends to 32-bits before storing in rd",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/rv32.html#load-and-store-instructions"
            };

        case "SW":
            return {
                "html": "<div>\n<span class=\"opcode\">imm12hi, rs1, rs2, imm12lo</span>\n\n<div>The SW, SH, and SB instructions store 32-bit, 16-bit, and 8-bit values from the low bits of register rs2 to memory.</div>\n</div>",
                "tooltip": "The SW, SH, and SB instructions store 32-bit, 16-bit, and 8-bit values from the low bits of register rs2 to memory.",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/rv32.html#load-and-store-instructions"
            };

        case "FENCE":
            return {
                "html": "<div>\n<span class=\"opcode\">rs1, rd</span>\n\n<div>The FENCE instruction is used to order device I/O and memory accesses as viewed by other RISC-V harts and external devices or coprocessors</div>\n<div>Informally, no other RISC-V hart or external device can observe any operation in the successor set following a FENCE before any operation in the predecessor set preceding the FENCE</div>\n<div>Instruction-set extensions might also describe new I/O instructions that will also be ordered using the I and O bits in a FENCE.</div>\n<div>The fence mode field fm defines the semantics of the FENCE</div>\n<div>A FENCE with fm =0000 orders all memory operations in its predecessor set before all memory operations in its successor set.</div>\n<div>The optional FENCE.TSO instruction is encoded as a FENCE instruction with fm =1000, predecessor =RW, and successor =RW</div>\n<div>FENCE.TSO orders all load operations in its predecessor set before all memory operations in its successor set, and all store operations in its predecessor set before all store operations in its successor set</div>\n<div>This leaves non-AMO store operations in the FENCE.TSO\u00c3\u00a2\u00c2\u0080\u00c2\u0099s predecessor set unordered with non-AMO loads in its successor set.</div>\n<div>The FENCE.TSO encoding was added as an optional extension to the original base FENCE instruction encoding</div>\n<div>The base definition requires that implementations ignore any set bits and treat the FENCE as global, and so this is a backwards-compatible extension.</div>\n<div>The unused fields in the FENCE instructions\u00c3\u00a2\u00c2\u0080\u00c2\u0094 rs1 and rd \u00c3\u00a2\u00c2\u0080\u00c2\u0094are reserved for finer-grain fences in future extensions</div>\n</div>",
                "tooltip": "The FENCE instruction is used to order device I/O and memory accesses as viewed by other RISC-V harts and external devices or coprocessors\nInformally, no other RISC-V hart or external device can observe any operation in the successor set following a FENCE before any operation in the predecessor set preceding the FENCE\nInstruction-set extensions might also describe new I/O instructions that will also be ordered using the I and O bits in a FENCE.\nThe fence mode field fm defines the semantics of the FENCE\nA FENCE with fm =0000 orders all memory operations in its predecessor set before all memory operations in its successor set.\nThe optional FENCE.TSO instruction is encoded as a FENCE instruction with fm =1000, predecessor =RW, and successor =RW\nFENCE.TSO orders all load operations in its predecessor set before all memory operations in its successor set, and all store operations in its predecessor set before all store operations in its successor set\nThis leaves non-AMO store operations in the FENCE.TSO\u00c3\u00a2\u00c2\u0080\u00c2\u0099s predecessor set unordered with non-AMO loads in its successor set.\nThe FENCE.TSO encoding was added as an optional extension to the original base FENCE instruction encoding\nThe base definition requires that implementations ignore any set bits and treat the FENCE as global, and so this is a backwards-compatible extension.\nThe unused fields in the FENCE instructions\u00c3\u00a2\u00c2\u0080\u00c2\u0094 rs1 and rd \u00c3\u00a2\u00c2\u0080\u00c2\u0094are reserved for finer-grain fences in future extensions",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/rv32.html#sec:fence"
            };

        case "ECALL":
            return {
                "html": "<div>\n<span class=\"opcode\"></span>\n\n<div>RV32I contains 40 unique instructions, though a simple implementation might cover the ECALL/EBREAK instructions with a single SYSTEM hardware instruction that always traps and might be able to implement the FENCE instruction as a NOP, reducing base instruction count to 38 total</div>\n</div>",
                "tooltip": "RV32I contains 40 unique instructions, though a simple implementation might cover the ECALL/EBREAK instructions with a single SYSTEM hardware instruction that always traps and might be able to implement the FENCE instruction as a NOP, reducing base instruction count to 38 total",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/rv32.html#rv32"
            };

        case "EBREAK":
            return {
                "html": "<div>\n<span class=\"opcode\"></span>\n\n<div>The EBREAK instruction is used to return control to a debugging environment.</div>\n<div>EBREAK was primarily designed to be used by a debugger to cause execution to stop and fall back into the debugger</div>\n<div>EBREAK is also used by the standard gcc compiler to mark code paths that should not be executed.</div>\n<div>Another use of EBREAK is to support \u00c3\u00a2\u00c2\u0080\u00c2\u009csemihosting\u00c3\u00a2\u00c2\u0080\u00c2\u009d, where the execution environment includes a debugger that can provide services over an alternate system call interface built around the EBREAK instruction</div>\n<div>Because the RISC-V base ISA does not provide more than one EBREAK instruction, RISC-V semihosting uses a special sequence of instructions to distinguish a semihosting EBREAK from a debugger inserted EBREAK</div>\n</div>",
                "tooltip": "The EBREAK instruction is used to return control to a debugging environment.\nEBREAK was primarily designed to be used by a debugger to cause execution to stop and fall back into the debugger\nEBREAK is also used by the standard gcc compiler to mark code paths that should not be executed.\nAnother use of EBREAK is to support \u00c3\u00a2\u00c2\u0080\u00c2\u009csemihosting\u00c3\u00a2\u00c2\u0080\u00c2\u009d, where the execution environment includes a debugger that can provide services over an alternate system call interface built around the EBREAK instruction\nBecause the RISC-V base ISA does not provide more than one EBREAK instruction, RISC-V semihosting uses a special sequence of instructions to distinguish a semihosting EBREAK from a debugger inserted EBREAK",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/rv32.html#environment-call-and-breakpoints"
            };

        case "ADDIW":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, imm12</span>\n\n<div>ADDIW is an RV64I instruction that adds the sign-extended 12-bit immediate to register rs1 and produces the proper sign-extension of a 32-bit result in rd</div>\n<div>Note, ADDIW rd, rs1, 0 writes the sign-extension of the lower 32 bits of register rs1 into register rd (assembler pseudoinstruction SEXT.W).</div>\n</div>",
                "tooltip": "ADDIW is an RV64I instruction that adds the sign-extended 12-bit immediate to register rs1 and produces the proper sign-extension of a 32-bit result in rd\nNote, ADDIW rd, rs1, 0 writes the sign-extension of the lower 32 bits of register rs1 into register rd (assembler pseudoinstruction SEXT.W).",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/rv64.html#integer-register-immediate-instructions"
            };

        case "SLLIW":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1</span>\n\n<div>SLLIW, SRLIW, and SRAIW are RV64I-only instructions that are analogously defined but operate on 32-bit values and produce signed 32-bit results</div>\n<div>SLLIW, SRLIW, and SRAIW encodings with i m m [5]\u00c3\u00a2\u00c2\u0080\u00c2\u0084\u00c3\u00a2\u00c2\u0089\u00c2\u00a0\u00c3\u00a2\u00c2\u0080\u00c2\u00840</div>\n<div>Previously, SLLIW, SRLIW, and SRAIW with i m m [5]\u00c3\u00a2\u00c2\u0080\u00c2\u0084\u00c3\u00a2\u00c2\u0089\u00c2\u00a0\u00c3\u00a2\u00c2\u0080\u00c2\u00840</div>\n</div>",
                "tooltip": "SLLIW, SRLIW, and SRAIW are RV64I-only instructions that are analogously defined but operate on 32-bit values and produce signed 32-bit results\nSLLIW, SRLIW, and SRAIW encodings with i m m [5]\u00c3\u00a2\u00c2\u0080\u00c2\u0084\u00c3\u00a2\u00c2\u0089\u00c2\u00a0\u00c3\u00a2\u00c2\u0080\u00c2\u00840\nPreviously, SLLIW, SRLIW, and SRAIW with i m m [5]\u00c3\u00a2\u00c2\u0080\u00c2\u0084\u00c3\u00a2\u00c2\u0089\u00c2\u00a0\u00c3\u00a2\u00c2\u0080\u00c2\u00840",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/rv64.html#integer-register-immediate-instructions"
            };

        case "ADDW":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, rs2</span>\n\n<div>ADDW and SUBW are RV64I-only instructions that are defined analogously to ADD and SUB but operate on 32-bit values and produce signed 32-bit results</div>\n</div>",
                "tooltip": "ADDW and SUBW are RV64I-only instructions that are defined analogously to ADD and SUB but operate on 32-bit values and produce signed 32-bit results",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/rv64.html#integer-register-register-operations"
            };

        case "SLLW":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, rs2</span>\n\n<div>SLLW, SRLW, and SRAW are RV64I-only instructions that are analogously defined but operate on 32-bit values and produce signed 32-bit results</div>\n</div>",
                "tooltip": "SLLW, SRLW, and SRAW are RV64I-only instructions that are analogously defined but operate on 32-bit values and produce signed 32-bit results",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/rv64.html#integer-register-register-operations"
            };

        case "LD":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, imm12</span>\n\n<div>The LD instruction loads a 64-bit value from memory into register rd for RV64I.</div>\n</div>",
                "tooltip": "The LD instruction loads a 64-bit value from memory into register rd for RV64I.",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/rv64.html#load-and-store-instructions"
            };

        case "LWU":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, imm12</span>\n\n<div>The LWU instruction, on the other hand, zero-extends the 32-bit value from memory for RV64I</div>\n</div>",
                "tooltip": "The LWU instruction, on the other hand, zero-extends the 32-bit value from memory for RV64I",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/rv64.html#load-and-store-instructions"
            };

        case "SD":
            return {
                "html": "<div>\n<span class=\"opcode\">imm12hi, rs1, rs2, imm12lo</span>\n\n<div>The SD, SW, SH, and SB instructions store 64-bit, 32-bit, 16-bit, and 8-bit values from the low bits of register rs2 to memory respectively.</div>\n</div>",
                "tooltip": "The SD, SW, SH, and SB instructions store 64-bit, 32-bit, 16-bit, and 8-bit values from the low bits of register rs2 to memory respectively.",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/rv64.html#load-and-store-instructions"
            };

        case "FMV.X.Q":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1</span>\n\n<div>The floating-point instruction set is unchanged, although the 128-bit Q floating-point extension can now support FMV.X.Q and FMV.Q.X instructions, together with additional FCVT instructions to and from the T (128-bit) integer format.</div>\n</div>",
                "tooltip": "The floating-point instruction set is unchanged, although the 128-bit Q floating-point extension can now support FMV.X.Q and FMV.Q.X instructions, together with additional FCVT instructions to and from the T (128-bit) integer format.",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/rv128.html#rv128"
            };

        case "AMOADD.W":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, rs2</span>\n\n<div>These AMO instructions atomically load a data value from the address in rs1 , place the value into register rd , apply a binary operator to the loaded value and the original value in rs2 , then store the result back to the address in rs1</div>\n</div>",
                "tooltip": "These AMO instructions atomically load a data value from the address in rs1 , place the value into register rd , apply a binary operator to the loaded value and the original value in rs2 , then store the result back to the address in rs1",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/a.html#sec:amo"
            };

        case "AMOXOR.W":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, rs2</span>\n\n<div>These AMO instructions atomically load a data value from the address in rs1 , place the value into register rd , apply a binary operator to the loaded value and the original value in rs2 , then store the result back to the address in rs1</div>\n</div>",
                "tooltip": "These AMO instructions atomically load a data value from the address in rs1 , place the value into register rd , apply a binary operator to the loaded value and the original value in rs2 , then store the result back to the address in rs1",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/a.html#sec:amo"
            };

        case "AMOOR.W":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, rs2</span>\n\n<div>These AMO instructions atomically load a data value from the address in rs1 , place the value into register rd , apply a binary operator to the loaded value and the original value in rs2 , then store the result back to the address in rs1</div>\n</div>",
                "tooltip": "These AMO instructions atomically load a data value from the address in rs1 , place the value into register rd , apply a binary operator to the loaded value and the original value in rs2 , then store the result back to the address in rs1",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/a.html#sec:amo"
            };

        case "AMOAND.W":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, rs2</span>\n\n<div>These AMO instructions atomically load a data value from the address in rs1 , place the value into register rd , apply a binary operator to the loaded value and the original value in rs2 , then store the result back to the address in rs1</div>\n</div>",
                "tooltip": "These AMO instructions atomically load a data value from the address in rs1 , place the value into register rd , apply a binary operator to the loaded value and the original value in rs2 , then store the result back to the address in rs1",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/a.html#sec:amo"
            };

        case "AMOMIN.W":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, rs2</span>\n\n<div>These AMO instructions atomically load a data value from the address in rs1 , place the value into register rd , apply a binary operator to the loaded value and the original value in rs2 , then store the result back to the address in rs1</div>\n</div>",
                "tooltip": "These AMO instructions atomically load a data value from the address in rs1 , place the value into register rd , apply a binary operator to the loaded value and the original value in rs2 , then store the result back to the address in rs1",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/a.html#sec:amo"
            };

        case "AMOMAX.W":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, rs2</span>\n\n<div>These AMO instructions atomically load a data value from the address in rs1 , place the value into register rd , apply a binary operator to the loaded value and the original value in rs2 , then store the result back to the address in rs1</div>\n</div>",
                "tooltip": "These AMO instructions atomically load a data value from the address in rs1 , place the value into register rd , apply a binary operator to the loaded value and the original value in rs2 , then store the result back to the address in rs1",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/a.html#sec:amo"
            };

        case "AMOMINU.W":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, rs2</span>\n\n<div>These AMO instructions atomically load a data value from the address in rs1 , place the value into register rd , apply a binary operator to the loaded value and the original value in rs2 , then store the result back to the address in rs1</div>\n</div>",
                "tooltip": "These AMO instructions atomically load a data value from the address in rs1 , place the value into register rd , apply a binary operator to the loaded value and the original value in rs2 , then store the result back to the address in rs1",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/a.html#sec:amo"
            };

        case "AMOMAXU.W":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, rs2</span>\n\n<div>These AMO instructions atomically load a data value from the address in rs1 , place the value into register rd , apply a binary operator to the loaded value and the original value in rs2 , then store the result back to the address in rs1</div>\n</div>",
                "tooltip": "These AMO instructions atomically load a data value from the address in rs1 , place the value into register rd , apply a binary operator to the loaded value and the original value in rs2 , then store the result back to the address in rs1",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/a.html#sec:amo"
            };

        case "AMOSWAP.W":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, rs2</span>\n\n<div>These AMO instructions atomically load a data value from the address in rs1 , place the value into register rd , apply a binary operator to the loaded value and the original value in rs2 , then store the result back to the address in rs1</div>\n</div>",
                "tooltip": "These AMO instructions atomically load a data value from the address in rs1 , place the value into register rd , apply a binary operator to the loaded value and the original value in rs2 , then store the result back to the address in rs1",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/a.html#sec:amo"
            };

        case "AMOADD.D":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, rs2</span>\n\n<div>These AMO instructions atomically load a data value from the address in rs1 , place the value into register rd , apply a binary operator to the loaded value and the original value in rs2 , then store the result back to the address in rs1</div>\n</div>",
                "tooltip": "These AMO instructions atomically load a data value from the address in rs1 , place the value into register rd , apply a binary operator to the loaded value and the original value in rs2 , then store the result back to the address in rs1",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/a.html#sec:amo"
            };

        case "AMOXOR.D":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, rs2</span>\n\n<div>These AMO instructions atomically load a data value from the address in rs1 , place the value into register rd , apply a binary operator to the loaded value and the original value in rs2 , then store the result back to the address in rs1</div>\n</div>",
                "tooltip": "These AMO instructions atomically load a data value from the address in rs1 , place the value into register rd , apply a binary operator to the loaded value and the original value in rs2 , then store the result back to the address in rs1",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/a.html#sec:amo"
            };

        case "AMOOR.D":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, rs2</span>\n\n<div>These AMO instructions atomically load a data value from the address in rs1 , place the value into register rd , apply a binary operator to the loaded value and the original value in rs2 , then store the result back to the address in rs1</div>\n</div>",
                "tooltip": "These AMO instructions atomically load a data value from the address in rs1 , place the value into register rd , apply a binary operator to the loaded value and the original value in rs2 , then store the result back to the address in rs1",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/a.html#sec:amo"
            };

        case "AMOAND.D":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, rs2</span>\n\n<div>These AMO instructions atomically load a data value from the address in rs1 , place the value into register rd , apply a binary operator to the loaded value and the original value in rs2 , then store the result back to the address in rs1</div>\n</div>",
                "tooltip": "These AMO instructions atomically load a data value from the address in rs1 , place the value into register rd , apply a binary operator to the loaded value and the original value in rs2 , then store the result back to the address in rs1",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/a.html#sec:amo"
            };

        case "AMOMIN.D":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, rs2</span>\n\n<div>These AMO instructions atomically load a data value from the address in rs1 , place the value into register rd , apply a binary operator to the loaded value and the original value in rs2 , then store the result back to the address in rs1</div>\n</div>",
                "tooltip": "These AMO instructions atomically load a data value from the address in rs1 , place the value into register rd , apply a binary operator to the loaded value and the original value in rs2 , then store the result back to the address in rs1",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/a.html#sec:amo"
            };

        case "AMOMAX.D":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, rs2</span>\n\n<div>These AMO instructions atomically load a data value from the address in rs1 , place the value into register rd , apply a binary operator to the loaded value and the original value in rs2 , then store the result back to the address in rs1</div>\n</div>",
                "tooltip": "These AMO instructions atomically load a data value from the address in rs1 , place the value into register rd , apply a binary operator to the loaded value and the original value in rs2 , then store the result back to the address in rs1",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/a.html#sec:amo"
            };

        case "AMOMINU.D":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, rs2</span>\n\n<div>These AMO instructions atomically load a data value from the address in rs1 , place the value into register rd , apply a binary operator to the loaded value and the original value in rs2 , then store the result back to the address in rs1</div>\n</div>",
                "tooltip": "These AMO instructions atomically load a data value from the address in rs1 , place the value into register rd , apply a binary operator to the loaded value and the original value in rs2 , then store the result back to the address in rs1",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/a.html#sec:amo"
            };

        case "AMOMAXU.D":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, rs2</span>\n\n<div>These AMO instructions atomically load a data value from the address in rs1 , place the value into register rd , apply a binary operator to the loaded value and the original value in rs2 , then store the result back to the address in rs1</div>\n</div>",
                "tooltip": "These AMO instructions atomically load a data value from the address in rs1 , place the value into register rd , apply a binary operator to the loaded value and the original value in rs2 , then store the result back to the address in rs1",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/a.html#sec:amo"
            };

        case "AMOSWAP.D":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, rs2</span>\n\n<div>These AMO instructions atomically load a data value from the address in rs1 , place the value into register rd , apply a binary operator to the loaded value and the original value in rs2 , then store the result back to the address in rs1</div>\n</div>",
                "tooltip": "These AMO instructions atomically load a data value from the address in rs1 , place the value into register rd , apply a binary operator to the loaded value and the original value in rs2 , then store the result back to the address in rs1",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/a.html#sec:amo"
            };

        case "LR.W":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1</span>\n\n<div>LR.W loads a word from the address in rs1 , places the sign-extended value in rd , and registers a reservation set \u00c3\u00a2\u00c2\u0080\u00c2\u0094a set of bytes that subsumes the bytes in the addressed word</div>\n</div>",
                "tooltip": "LR.W loads a word from the address in rs1 , places the sign-extended value in rd , and registers a reservation set \u00c3\u00a2\u00c2\u0080\u00c2\u0094a set of bytes that subsumes the bytes in the addressed word",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/a.html#sec:lrsc"
            };

        case "SC.W":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, rs2</span>\n\n<div>SC.W conditionally writes a word in rs2 to the address in rs1 : the SC.W succeeds only if the reservation is still valid and the reservation set contains the bytes being written</div>\n<div>If the SC.W succeeds, the instruction writes the word in rs2 to memory, and it writes zero to rd</div>\n<div>If the SC.W fails, the instruction does not write to memory, and it writes a nonzero value to rd</div>\n<div>Regardless of success or failure, executing an SC.W instruction invalidates any reservation held by this hart</div>\n</div>",
                "tooltip": "SC.W conditionally writes a word in rs2 to the address in rs1 : the SC.W succeeds only if the reservation is still valid and the reservation set contains the bytes being written\nIf the SC.W succeeds, the instruction writes the word in rs2 to memory, and it writes zero to rd\nIf the SC.W fails, the instruction does not write to memory, and it writes a nonzero value to rd\nRegardless of success or failure, executing an SC.W instruction invalidates any reservation held by this hart",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/a.html#sec:lrsc"
            };

        case "LR.D":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1</span>\n\n<div>LR.D and SC.D act analogously on doublewords and are only available on RV64. For RV64, LR.W and SC.W sign-extend the value placed in rd .</div>\n</div>",
                "tooltip": "LR.D and SC.D act analogously on doublewords and are only available on RV64. For RV64, LR.W and SC.W sign-extend the value placed in rd .",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/a.html#sec:lrsc"
            };

        case "C.ADDI4SPN":
            return {
                "html": "<div>\n<span class=\"opcode\"></span>\n\n<div>In the standard RISC-V calling convention, the stack pointer sp C.ADDI4SPN is a CIW-format instruction that adds a zero -extended non-zero immediate, scaled by 4, to the stack pointer, x2 , and writes the result to rd\u00c3\u00a2\u00c2\u0080\u00c2\u0086'</div>\n<div>C.ADDI4SPN is only valid when nzuimm \u00c3\u00a2\u00c2\u0089\u00c2\u00a0</div>\n</div>",
                "tooltip": "In the standard RISC-V calling convention, the stack pointer sp C.ADDI4SPN is a CIW-format instruction that adds a zero -extended non-zero immediate, scaled by 4, to the stack pointer, x2 , and writes the result to rd\u00c3\u00a2\u00c2\u0080\u00c2\u0086'\nC.ADDI4SPN is only valid when nzuimm \u00c3\u00a2\u00c2\u0089\u00c2",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/c.html#integer-register-immediate-operations"
            };

        case "C.ADDI":
            return {
                "html": "<div>\n<span class=\"opcode\"></span>\n\n<div>C.ADDI adds the non-zero sign-extended 6-bit immediate to the value in register rd then writes the result to rd</div>\n<div>C.ADDI expands into addi rd, rd, nzimm[5:0]</div>\n<div>C.ADDI is only valid when rd \u00c3\u00a2\u00c2\u0089\u00c2\u00a0 x0 and nzimm \u00c3\u00a2\u00c2\u0089\u00c2\u00a0</div>\n</div>",
                "tooltip": "C.ADDI adds the non-zero sign-extended 6-bit immediate to the value in register rd then writes the result to rd\nC.ADDI expands into addi rd, rd, nzimm[5:0]\nC.ADDI is only valid when rd \u00c3\u00a2\u00c2\u0089\u00c2\u00a0 x0 and nzimm \u00c3\u00a2\u00c2\u0089\u00c2",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/c.html#integer-register-immediate-operations"
            };

        case "C.SRLI":
            return {
                "html": "<div>\n<span class=\"opcode\"></span>\n\n<div>C.SRLI is a CB-format instruction that performs a logical right shift of the value in register rd\u00c3\u00a2\u00c2\u0080\u00c2\u0086 \u00c3\u00a2\u00c2\u0080\u00c2\u00b2</div>\n<div>For RV128C, a shift amount of zero is used to encode a shift of 64. Furthermore, the shift amount is sign-extended for RV128C, and so the legal shift amounts are 1\u00c3\u00a2\u00c2\u0080\u00c2\u009331, 64, and 96\u00c3\u00a2\u00c2\u0080\u00c2\u0093127. C.SRLI expands into srli rd\u00c3\u00a2\u00c2\u0080\u00c2\u0086', rd\u00c3\u00a2\u00c2\u0080\u00c2\u0086', shamt[5:0] , except for RV128C with shamt=0 , which expands to srli rd\u00c3\u00a2\u00c2\u0080\u00c2\u0086', rd\u00c3\u00a2\u00c2\u0080\u00c2\u0086', 64 .</div>\n</div>",
                "tooltip": "C.SRLI is a CB-format instruction that performs a logical right shift of the value in register rd\u00c3\u00a2\u00c2\u0080\u00c2\u0086 \u00c3\u00a2\u00c2\u0080\u00c2\u00b2\nFor RV128C, a shift amount of zero is used to encode a shift of 64. Furthermore, the shift amount is sign-extended for RV128C, and so the legal shift amounts are 1\u00c3\u00a2\u00c2\u0080\u00c2\u009331, 64, and 96\u00c3\u00a2\u00c2\u0080\u00c2\u0093127. C.SRLI expands into srli rd\u00c3\u00a2\u00c2\u0080\u00c2\u0086', rd\u00c3\u00a2\u00c2\u0080\u00c2\u0086', shamt[5:0] , except for RV128C with shamt=0 , which expands to srli rd\u00c3\u00a2\u00c2\u0080\u00c2\u0086', rd\u00c3\u00a2\u00c2\u0080\u00c2\u0086', 64 .",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/c.html#integer-register-immediate-operations"
            };

        case "C.SRAI":
            return {
                "html": "<div>\n<span class=\"opcode\"></span>\n\n<div>C.SRAI is defined analogously to C.SRLI, but instead performs an arithmetic right shift</div>\n<div>C.SRAI expands to srai rd\u00c3\u00a2\u00c2\u0080\u00c2\u0086', rd\u00c3\u00a2\u00c2\u0080\u00c2\u0086', shamt[5:0] .</div>\n</div>",
                "tooltip": "C.SRAI is defined analogously to C.SRLI, but instead performs an arithmetic right shift\nC.SRAI expands to srai rd\u00c3\u00a2\u00c2\u0080\u00c2\u0086', rd\u00c3\u00a2\u00c2\u0080\u00c2\u0086', shamt[5:0] .",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/c.html#integer-register-immediate-operations"
            };

        case "C.ANDI":
            return {
                "html": "<div>\n<span class=\"opcode\"></span>\n\n<div>C.ANDI is a CB-format instruction that computes the bitwise AND of the value in register rd\u00c3\u00a2\u00c2\u0080\u00c2\u0086 \u00c3\u00a2\u00c2\u0080\u00c2\u00b2</div>\n<div>. C.ANDI expands to andi rd\u00c3\u00a2\u00c2\u0080\u00c2\u0086', rd\u00c3\u00a2\u00c2\u0080\u00c2\u0086', imm[5:0] .</div>\n</div>",
                "tooltip": "C.ANDI is a CB-format instruction that computes the bitwise AND of the value in register rd\u00c3\u00a2\u00c2\u0080\u00c2\u0086 \u00c3\u00a2\u00c2\u0080\u00c2\u00b2\n. C.ANDI expands to andi rd\u00c3\u00a2\u00c2\u0080\u00c2\u0086', rd\u00c3\u00a2\u00c2\u0080\u00c2\u0086', imm[5:0] .",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/c.html#integer-register-immediate-operations"
            };

        case "C.SLLI":
            return {
                "html": "<div>\n<span class=\"opcode\"></span>\n\n<div>C.SLLI is a CI-format instruction that performs a logical left shift of the value in register rd then writes the result to rd</div>\n<div>For RV128C, a shift amount of zero is used to encode a shift of 64. C.SLLI expands into slli rd, rd, shamt[5:0] , except for RV128C with shamt=0 , which expands to slli rd, rd, 64 .</div>\n</div>",
                "tooltip": "C.SLLI is a CI-format instruction that performs a logical left shift of the value in register rd then writes the result to rd\nFor RV128C, a shift amount of zero is used to encode a shift of 64. C.SLLI expands into slli rd, rd, shamt[5:0] , except for RV128C with shamt=0 , which expands to slli rd, rd, 64 .",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/c.html#integer-register-immediate-operations"
            };

        case "C.FLD":
            return {
                "html": "<div>\n<span class=\"opcode\"></span>\n\n<div>C.FLD is an RV32DC/RV64DC-only instruction that loads a double-precision floating-point value from memory into floating-point register rd\u00c3\u00a2\u00c2\u0080\u00c2\u0086 \u00c3\u00a2\u00c2\u0080\u00c2\u00b2</div>\n</div>",
                "tooltip": "C.FLD is an RV32DC/RV64DC-only instruction that loads a double-precision floating-point value from memory into floating-point register rd\u00c3\u00a2\u00c2\u0080\u00c2\u0086 \u00c3\u00a2\u00c2\u0080\u00c2\u00b2",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/c.html#register-based-loads-and-stores"
            };

        case "C.LW":
            return {
                "html": "<div>\n<span class=\"opcode\"></span>\n\n<div>C.LW loads a 32-bit value from memory into register rd\u00c3\u00a2\u00c2\u0080\u00c2\u0086 \u00c3\u00a2\u00c2\u0080\u00c2\u00b2</div>\n</div>",
                "tooltip": "C.LW loads a 32-bit value from memory into register rd\u00c3\u00a2\u00c2\u0080\u00c2\u0086 \u00c3\u00a2\u00c2\u0080\u00c2\u00b2",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/c.html#register-based-loads-and-stores"
            };

        case "C.FLW":
            return {
                "html": "<div>\n<span class=\"opcode\"></span>\n\n<div>C.FLW is an RV32FC-only instruction that loads a single-precision floating-point value from memory into floating-point register rd\u00c3\u00a2\u00c2\u0080\u00c2\u0086 \u00c3\u00a2\u00c2\u0080\u00c2\u00b2</div>\n</div>",
                "tooltip": "C.FLW is an RV32FC-only instruction that loads a single-precision floating-point value from memory into floating-point register rd\u00c3\u00a2\u00c2\u0080\u00c2\u0086 \u00c3\u00a2\u00c2\u0080\u00c2\u00b2",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/c.html#register-based-loads-and-stores"
            };

        case "C.FSD":
            return {
                "html": "<div>\n<span class=\"opcode\"></span>\n\n<div>C.FSD is an RV32DC/RV64DC-only instruction that stores a double-precision floating-point value in floating-point register rs2\u00c3\u00a2\u00c2\u0080\u00c2\u0086 \u00c3\u00a2\u00c2\u0080\u00c2\u00b2</div>\n</div>",
                "tooltip": "C.FSD is an RV32DC/RV64DC-only instruction that stores a double-precision floating-point value in floating-point register rs2\u00c3\u00a2\u00c2\u0080\u00c2\u0086 \u00c3\u00a2\u00c2\u0080\u00c2\u00b2",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/c.html#register-based-loads-and-stores"
            };

        case "C.SW":
            return {
                "html": "<div>\n<span class=\"opcode\"></span>\n\n<div>C.SW stores a 32-bit value in register rs2\u00c3\u00a2\u00c2\u0080\u00c2\u0086 \u00c3\u00a2\u00c2\u0080\u00c2\u00b2</div>\n</div>",
                "tooltip": "C.SW stores a 32-bit value in register rs2\u00c3\u00a2\u00c2\u0080\u00c2\u0086 \u00c3\u00a2\u00c2\u0080\u00c2\u00b2",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/c.html#register-based-loads-and-stores"
            };

        case "C.FSW":
            return {
                "html": "<div>\n<span class=\"opcode\"></span>\n\n<div>C.FSW is an RV32FC-only instruction that stores a single-precision floating-point value in floating-point register rs2\u00c3\u00a2\u00c2\u0080\u00c2\u0086 \u00c3\u00a2\u00c2\u0080\u00c2\u00b2</div>\n</div>",
                "tooltip": "C.FSW is an RV32FC-only instruction that stores a single-precision floating-point value in floating-point register rs2\u00c3\u00a2\u00c2\u0080\u00c2\u0086 \u00c3\u00a2\u00c2\u0080\u00c2\u00b2",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/c.html#register-based-loads-and-stores"
            };

        case "C.JAL":
            return {
                "html": "<div>\n<span class=\"opcode\"></span>\n\n<div>C.JAL is an RV32C-only instruction that performs the same operation as C.J, but additionally writes the address of the instruction following the jump ( pc +2) to the link register, x1</div>\n<div>C.JAL expands to jal x1, offset[11:1] .</div>\n</div>",
                "tooltip": "C.JAL is an RV32C-only instruction that performs the same operation as C.J, but additionally writes the address of the instruction following the jump ( pc +2) to the link register, x1\nC.JAL expands to jal x1, offset[11:1] .",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/c.html#control-transfer-instructions"
            };

        case "C.J":
            return {
                "html": "<div>\n<span class=\"opcode\"></span>\n\n<div>C.J performs an unconditional control transfer</div>\n<div>C.J can therefore target a</div>\n<div>C.J expands to jal x0, offset[11:1] .</div>\n</div>",
                "tooltip": "C.J performs an unconditional control transfer\nC.J can therefore target a\nC.J expands to jal x0, offset[11:1] .",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/c.html#control-transfer-instructions"
            };

        case "C.BEQZ":
            return {
                "html": "<div>\n<span class=\"opcode\"></span>\n\n<div>C.BEQZ performs conditional control transfers</div>\n<div>C.BEQZ takes the branch if the value in register rs1\u00c3\u00a2\u00c2\u0080\u00c2\u0086 \u00c3\u00a2\u00c2\u0080\u00c2\u00b2</div>\n</div>",
                "tooltip": "C.BEQZ performs conditional control transfers\nC.BEQZ takes the branch if the value in register rs1\u00c3\u00a2\u00c2\u0080\u00c2\u0086 \u00c3\u00a2\u00c2\u0080\u00c2\u00b2",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/c.html#control-transfer-instructions"
            };

        case "C.BNEZ":
            return {
                "html": "<div>\n<span class=\"opcode\"></span>\n\n<div>C.BNEZ is defined analogously, but it takes the branch if rs1\u00c3\u00a2\u00c2\u0080\u00c2\u0086 \u00c3\u00a2\u00c2\u0080\u00c2\u00b2</div>\n</div>",
                "tooltip": "C.BNEZ is defined analogously, but it takes the branch if rs1\u00c3\u00a2\u00c2\u0080\u00c2\u0086 \u00c3\u00a2\u00c2\u0080\u00c2\u00b2",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/c.html#control-transfer-instructions"
            };

        case "C.LI":
            return {
                "html": "<div>\n<span class=\"opcode\"></span>\n\n<div>C.LI loads the sign-extended 6-bit immediate, imm , into register rd</div>\n<div>C.LI expands into addi rd, x0, imm[5:0]</div>\n<div>C.LI is only valid when rd \u00c3\u00a2\u00c2\u0089\u00c2\u00a0 x0 ; the code points with rd = x0 encode HINTs.</div>\n</div>",
                "tooltip": "C.LI loads the sign-extended 6-bit immediate, imm , into register rd\nC.LI expands into addi rd, x0, imm[5:0]\nC.LI is only valid when rd \u00c3\u00a2\u00c2\u0089\u00c2\u00a0 x0 ; the code points with rd = x0 encode HINTs.",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/c.html#integer-constant-generation-instructions"
            };

        case "C.LUI":
            return {
                "html": "<div>\n<span class=\"opcode\">rd=2</span>\n\n<div>C.LUI loads the non-zero 6-bit immediate field into bits 17\u00c3\u00a2\u00c2\u0080\u00c2\u009312 of the destination register, clears the bottom 12 bits, and sign-extends bit 17 into all higher bits of the destination</div>\n<div>C.LUI expands into lui rd, nzimm[17:12]</div>\n<div>C.LUI is only valid when rd \u00c3\u00a2\u00c2\u0080\u00c2\u0084\u00c3\u00a2\u00c2\u0089\u00c2\u00a0\u00c3\u00a2\u00c2\u0080\u00c2\u0084{ x0 , x2 }</div>\n</div>",
                "tooltip": "C.LUI loads the non-zero 6-bit immediate field into bits 17\u00c3\u00a2\u00c2\u0080\u00c2\u009312 of the destination register, clears the bottom 12 bits, and sign-extends bit 17 into all higher bits of the destination\nC.LUI expands into lui rd, nzimm[17:12]\nC.LUI is only valid when rd \u00c3\u00a2\u00c2\u0080\u00c2\u0084\u00c3\u00a2\u00c2\u0089\u00c2\u00a0\u00c3\u00a2\u00c2\u0080\u00c2\u0084{ x0 , x2 }",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/c.html#integer-constant-generation-instructions"
            };

        case "C.SUB":
            return {
                "html": "<div>\n<span class=\"opcode\"></span>\n\n<div>C.SUB subtracts the value in register rs2\u00c3\u00a2\u00c2\u0080\u00c2\u0086 \u00c3\u00a2\u00c2\u0080\u00c2\u00b2</div>\n<div>. C.SUB expands into sub rd\u00c3\u00a2\u00c2\u0080\u00c2\u0086', rd\u00c3\u00a2\u00c2\u0080\u00c2\u0086', rs2\u00c3\u00a2\u00c2\u0080\u00c2\u0086' .</div>\n</div>",
                "tooltip": "C.SUB subtracts the value in register rs2\u00c3\u00a2\u00c2\u0080\u00c2\u0086 \u00c3\u00a2\u00c2\u0080\u00c2\u00b2\n. C.SUB expands into sub rd\u00c3\u00a2\u00c2\u0080\u00c2\u0086', rd\u00c3\u00a2\u00c2\u0080\u00c2\u0086', rs2\u00c3\u00a2\u00c2\u0080\u00c2\u0086' .",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/c.html#integer-register-register-operations"
            };

        case "C.XOR":
            return {
                "html": "<div>\n<span class=\"opcode\"></span>\n\n<div>C.XOR computes the bitwise XOR of the values in registers rd\u00c3\u00a2\u00c2\u0080\u00c2\u0086 \u00c3\u00a2\u00c2\u0080\u00c2\u00b2 rs2\u00c3\u00a2\u00c2\u0080\u00c2\u0086 \u00c3\u00a2\u00c2\u0080\u00c2\u00b2</div>\n<div>. C.XOR expands into xor rd\u00c3\u00a2\u00c2\u0080\u00c2\u0086', rd\u00c3\u00a2\u00c2\u0080\u00c2\u0086', rs2\u00c3\u00a2\u00c2\u0080\u00c2\u0086' .</div>\n</div>",
                "tooltip": "C.XOR computes the bitwise XOR of the values in registers rd\u00c3\u00a2\u00c2\u0080\u00c2\u0086 \u00c3\u00a2\u00c2\u0080\u00c2\u00b2 rs2\u00c3\u00a2\u00c2\u0080\u00c2\u0086 \u00c3\u00a2\u00c2\u0080\u00c2\u00b2\n. C.XOR expands into xor rd\u00c3\u00a2\u00c2\u0080\u00c2\u0086', rd\u00c3\u00a2\u00c2\u0080\u00c2\u0086', rs2\u00c3\u00a2\u00c2\u0080\u00c2\u0086' .",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/c.html#integer-register-register-operations"
            };

        case "C.OR":
            return {
                "html": "<div>\n<span class=\"opcode\"></span>\n\n<div>C.OR computes the bitwise OR of the values in registers rd\u00c3\u00a2\u00c2\u0080\u00c2\u0086 \u00c3\u00a2\u00c2\u0080\u00c2\u00b2 rs2\u00c3\u00a2\u00c2\u0080\u00c2\u0086 \u00c3\u00a2\u00c2\u0080\u00c2\u00b2</div>\n<div>. C.OR expands into or rd\u00c3\u00a2\u00c2\u0080\u00c2\u0086', rd\u00c3\u00a2\u00c2\u0080\u00c2\u0086', rs2\u00c3\u00a2\u00c2\u0080\u00c2\u0086' .</div>\n</div>",
                "tooltip": "C.OR computes the bitwise OR of the values in registers rd\u00c3\u00a2\u00c2\u0080\u00c2\u0086 \u00c3\u00a2\u00c2\u0080\u00c2\u00b2 rs2\u00c3\u00a2\u00c2\u0080\u00c2\u0086 \u00c3\u00a2\u00c2\u0080\u00c2\u00b2\n. C.OR expands into or rd\u00c3\u00a2\u00c2\u0080\u00c2\u0086', rd\u00c3\u00a2\u00c2\u0080\u00c2\u0086', rs2\u00c3\u00a2\u00c2\u0080\u00c2\u0086' .",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/c.html#integer-register-register-operations"
            };

        case "C.AND":
            return {
                "html": "<div>\n<span class=\"opcode\"></span>\n\n<div>C.AND computes the bitwise AND of the values in registers rd\u00c3\u00a2\u00c2\u0080\u00c2\u0086 \u00c3\u00a2\u00c2\u0080\u00c2\u00b2 rs2\u00c3\u00a2\u00c2\u0080\u00c2\u0086 \u00c3\u00a2\u00c2\u0080\u00c2\u00b2</div>\n<div>. C.AND expands into and rd\u00c3\u00a2\u00c2\u0080\u00c2\u0086', rd\u00c3\u00a2\u00c2\u0080\u00c2\u0086', rs2\u00c3\u00a2\u00c2\u0080\u00c2\u0086' .</div>\n</div>",
                "tooltip": "C.AND computes the bitwise AND of the values in registers rd\u00c3\u00a2\u00c2\u0080\u00c2\u0086 \u00c3\u00a2\u00c2\u0080\u00c2\u00b2 rs2\u00c3\u00a2\u00c2\u0080\u00c2\u0086 \u00c3\u00a2\u00c2\u0080\u00c2\u00b2\n. C.AND expands into and rd\u00c3\u00a2\u00c2\u0080\u00c2\u0086', rd\u00c3\u00a2\u00c2\u0080\u00c2\u0086', rs2\u00c3\u00a2\u00c2\u0080\u00c2\u0086' .",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/c.html#integer-register-register-operations"
            };

        case "C.SUBW":
            return {
                "html": "<div>\n<span class=\"opcode\"></span>\n\n<div>C.SUBW is an RV64C/RV128C-only instruction that subtracts the value in register rs2\u00c3\u00a2\u00c2\u0080\u00c2\u0086 \u00c3\u00a2\u00c2\u0080\u00c2\u00b2</div>\n<div>. C.SUBW expands into subw rd\u00c3\u00a2\u00c2\u0080\u00c2\u0086', rd\u00c3\u00a2\u00c2\u0080\u00c2\u0086', rs2\u00c3\u00a2\u00c2\u0080\u00c2\u0086' .</div>\n</div>",
                "tooltip": "C.SUBW is an RV64C/RV128C-only instruction that subtracts the value in register rs2\u00c3\u00a2\u00c2\u0080\u00c2\u0086 \u00c3\u00a2\u00c2\u0080\u00c2\u00b2\n. C.SUBW expands into subw rd\u00c3\u00a2\u00c2\u0080\u00c2\u0086', rd\u00c3\u00a2\u00c2\u0080\u00c2\u0086', rs2\u00c3\u00a2\u00c2\u0080\u00c2\u0086' .",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/c.html#integer-register-register-operations"
            };

        case "C.ADDW":
            return {
                "html": "<div>\n<span class=\"opcode\"></span>\n\n<div>C.ADDW is an RV64C/RV128C-only instruction that adds the values in registers rd\u00c3\u00a2\u00c2\u0080\u00c2\u0086 \u00c3\u00a2\u00c2\u0080\u00c2\u00b2</div>\n<div>. C.ADDW expands into addw rd\u00c3\u00a2\u00c2\u0080\u00c2\u0086', rd\u00c3\u00a2\u00c2\u0080\u00c2\u0086', rs2\u00c3\u00a2\u00c2\u0080\u00c2\u0086' .</div>\n</div>",
                "tooltip": "C.ADDW is an RV64C/RV128C-only instruction that adds the values in registers rd\u00c3\u00a2\u00c2\u0080\u00c2\u0086 \u00c3\u00a2\u00c2\u0080\u00c2\u00b2\n. C.ADDW expands into addw rd\u00c3\u00a2\u00c2\u0080\u00c2\u0086', rd\u00c3\u00a2\u00c2\u0080\u00c2\u0086', rs2\u00c3\u00a2\u00c2\u0080\u00c2\u0086' .",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/c.html#integer-register-register-operations"
            };

        case "C.MV":
            return {
                "html": "<div>\n<span class=\"opcode\">!rs2</span>\n\n<div>C.MV copies the value in register rs2 into register rd</div>\n<div>C.MV expands into add rd, x0, rs2</div>\n<div>C.MV is only valid when rs2 \u00c3\u00a2\u00c2\u0080\u00c2\u0084\u00c3\u00a2\u00c2\u0089\u00c2\u00a0\u00c3\u00a2\u00c2\u0080\u00c2\u0084 x0 ; the code points with rs2 \u00c3\u00a2\u00c2\u0080\u00c2\u0084=\u00c3\u00a2\u00c2\u0080\u00c2\u0084 x0 correspond to the C.JR instruction</div>\n<div>C.MV expands to a different instruction than the canonical MV pseudoinstruction, which instead uses ADDI</div>\n<div>using register-renaming hardware, may find it more convenient to expand C.MV to MV instead of ADD, at slight additional hardware cost.</div>\n</div>",
                "tooltip": "C.MV copies the value in register rs2 into register rd\nC.MV expands into add rd, x0, rs2\nC.MV is only valid when rs2 \u00c3\u00a2\u00c2\u0080\u00c2\u0084\u00c3\u00a2\u00c2\u0089\u00c2\u00a0\u00c3\u00a2\u00c2\u0080\u00c2\u0084 x0 ; the code points with rs2 \u00c3\u00a2\u00c2\u0080\u00c2\u0084=\u00c3\u00a2\u00c2\u0080\u00c2\u0084 x0 correspond to the C.JR instruction\nC.MV expands to a different instruction than the canonical MV pseudoinstruction, which instead uses ADDI\nusing register-renaming hardware, may find it more convenient to expand C.MV to MV instead of ADD, at slight additional hardware cost.",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/c.html#integer-register-register-operations"
            };

        case "C.ADD":
            return {
                "html": "<div>\n<span class=\"opcode\">!rs1, !rs2=c.jalr</span>\n\n<div>C.ADD adds the values in registers rd and rs2 and writes the result to register rd</div>\n<div>C.ADD expands into add rd, rd, rs2</div>\n<div>C.ADD is only valid when rs2 \u00c3\u00a2\u00c2\u0080\u00c2\u0084\u00c3\u00a2\u00c2\u0089\u00c2\u00a0\u00c3\u00a2\u00c2\u0080\u00c2\u0084 x0 ; the code points with rs2 \u00c3\u00a2\u00c2\u0080\u00c2\u0084=\u00c3\u00a2\u00c2\u0080\u00c2\u0084 x0 correspond to the C.JALR and C.EBREAK instructions</div>\n</div>",
                "tooltip": "C.ADD adds the values in registers rd and rs2 and writes the result to register rd\nC.ADD expands into add rd, rd, rs2\nC.ADD is only valid when rs2 \u00c3\u00a2\u00c2\u0080\u00c2\u0084\u00c3\u00a2\u00c2\u0089\u00c2\u00a0\u00c3\u00a2\u00c2\u0080\u00c2\u0084 x0 ; the code points with rs2 \u00c3\u00a2\u00c2\u0080\u00c2\u0084=\u00c3\u00a2\u00c2\u0080\u00c2\u0084 x0 correspond to the C.JALR and C.EBREAK instructions",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/c.html#integer-register-register-operations"
            };

        case "C.FLDSP":
            return {
                "html": "<div>\n<span class=\"opcode\"></span>\n\n<div>C.FLDSP is an RV32DC/RV64DC-only instruction that loads a double-precision floating-point value from memory into floating-point register rd</div>\n</div>",
                "tooltip": "C.FLDSP is an RV32DC/RV64DC-only instruction that loads a double-precision floating-point value from memory into floating-point register rd",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/c.html#stack-pointer-based-loads-and-stores"
            };

        case "C.LWSP":
            return {
                "html": "<div>\n<span class=\"opcode\"></span>\n\n<div>C.LWSP loads a 32-bit value from memory into register rd</div>\n<div>C.LWSP is only valid when rd \u00c3\u00a2\u00c2\u0080\u00c2\u0084\u00c3\u00a2\u00c2\u0089\u00c2\u00a0\u00c3\u00a2\u00c2\u0080\u00c2\u0084 x0 ; the code points with rd \u00c3\u00a2\u00c2\u0080\u00c2\u0084=\u00c3\u00a2\u00c2\u0080\u00c2\u0084 x0 are reserved.</div>\n</div>",
                "tooltip": "C.LWSP loads a 32-bit value from memory into register rd\nC.LWSP is only valid when rd \u00c3\u00a2\u00c2\u0080\u00c2\u0084\u00c3\u00a2\u00c2\u0089\u00c2\u00a0\u00c3\u00a2\u00c2\u0080\u00c2\u0084 x0 ; the code points with rd \u00c3\u00a2\u00c2\u0080\u00c2\u0084=\u00c3\u00a2\u00c2\u0080\u00c2\u0084 x0 are reserved.",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/c.html#stack-pointer-based-loads-and-stores"
            };

        case "C.FLWSP":
            return {
                "html": "<div>\n<span class=\"opcode\"></span>\n\n<div>C.FLWSP is an RV32FC-only instruction that loads a single-precision floating-point value from memory into floating-point register rd</div>\n</div>",
                "tooltip": "C.FLWSP is an RV32FC-only instruction that loads a single-precision floating-point value from memory into floating-point register rd",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/c.html#stack-pointer-based-loads-and-stores"
            };

        case "C.FSDSP":
            return {
                "html": "<div>\n<span class=\"opcode\"></span>\n\n<div>C.FSDSP is an RV32DC/RV64DC-only instruction that stores a double-precision floating-point value in floating-point register rs2 to memory</div>\n</div>",
                "tooltip": "C.FSDSP is an RV32DC/RV64DC-only instruction that stores a double-precision floating-point value in floating-point register rs2 to memory",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/c.html#stack-pointer-based-loads-and-stores"
            };

        case "C.SWSP":
            return {
                "html": "<div>\n<span class=\"opcode\"></span>\n\n<div>C.SWSP stores a 32-bit value in register rs2 to memory</div>\n</div>",
                "tooltip": "C.SWSP stores a 32-bit value in register rs2 to memory",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/c.html#stack-pointer-based-loads-and-stores"
            };

        case "C.FSWSP":
            return {
                "html": "<div>\n<span class=\"opcode\"></span>\n\n<div>C.FSWSP is an RV32FC-only instruction that stores a single-precision floating-point value in floating-point register rs2 to memory</div>\n</div>",
                "tooltip": "C.FSWSP is an RV32FC-only instruction that stores a single-precision floating-point value in floating-point register rs2 to memory",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/c.html#stack-pointer-based-loads-and-stores"
            };

        case "@C.NOP":
            return {
                "html": "<div>\n<span class=\"opcode\"></span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/c.html#compressed"
            };

        case "@C.ADDI16SP":
            return {
                "html": "<div>\n<span class=\"opcode\"></span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/c.html#compressed"
            };

        case "@C.JR":
            return {
                "html": "<div>\n<span class=\"opcode\"></span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/c.html#compressed"
            };

        case "@C.JALR":
            return {
                "html": "<div>\n<span class=\"opcode\"></span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/c.html#compressed"
            };

        case "@C.EBREAK":
            return {
                "html": "<div>\n<span class=\"opcode\"></span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/c.html#compressed"
            };

        case "@C.LD":
            return {
                "html": "<div>\n<span class=\"opcode\"></span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/c.html#compressed"
            };

        case "@C.SD":
            return {
                "html": "<div>\n<span class=\"opcode\"></span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/c.html#compressed"
            };

        case "@C.ADDIW":
            return {
                "html": "<div>\n<span class=\"opcode\"></span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/c.html#compressed"
            };

        case "@C.LDSP":
            return {
                "html": "<div>\n<span class=\"opcode\"></span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/c.html#compressed"
            };

        case "@C.SDSP":
            return {
                "html": "<div>\n<span class=\"opcode\"></span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/c.html#compressed"
            };

        case "@C.LQ":
            return {
                "html": "<div>\n<span class=\"opcode\"></span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/c.html#compressed"
            };

        case "@C.SQ":
            return {
                "html": "<div>\n<span class=\"opcode\"></span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/c.html#compressed"
            };

        case "@C.LQSP":
            return {
                "html": "<div>\n<span class=\"opcode\"></span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/c.html#compressed"
            };

        case "@C.SQSP":
            return {
                "html": "<div>\n<span class=\"opcode\"></span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/c.html#compressed"
            };

        case "FADD.D":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, rs2</span>\n\n<div>FADD.S and FMUL.S perform single-precision floating-point addition and multiplication respectively, between rs1 and rs2</div>\n</div>",
                "tooltip": "FADD.S and FMUL.S perform single-precision floating-point addition and multiplication respectively, between rs1 and rs2",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest//d.html"
            };

        case "FSUB.D":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, rs2</span>\n\n<div>FSUB.S performs the single-precision floating-point subtraction of rs2 from rs1</div>\n</div>",
                "tooltip": "FSUB.S performs the single-precision floating-point subtraction of rs2 from rs1",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest//d.html"
            };

        case "FDIV.D":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, rs2</span>\n\n<div>FDIV.S performs the single-precision floating-point division of rs1 by rs2</div>\n</div>",
                "tooltip": "FDIV.S performs the single-precision floating-point division of rs1 by rs2",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest//d.html"
            };

        case "FMIN.D":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, rs2</span>\n\n<div>Floating-point minimum-number and maximum-number instructions FMIN.S and FMAX.S write, respectively, the smaller or larger of rs1 and rs2 rd</div>\n<div>Note that in version 2.2 of the F extension, the FMIN.S and FMAX.S instructions were amended to implement the proposed IEEE 754-201x minimumNumber and maximumNumber operations, rather than the IEEE 754-2008 minNum and maxNum operations</div>\n</div>",
                "tooltip": "Floating-point minimum-number and maximum-number instructions FMIN.S and FMAX.S write, respectively, the smaller or larger of rs1 and rs2 rd\nNote that in version 2.2 of the F extension, the FMIN.S and FMAX.S instructions were amended to implement the proposed IEEE 754-201x minimumNumber and maximumNumber operations, rather than the IEEE 754-2008 minNum and maxNum operations",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest//d.html"
            };

        case "FSQRT.D":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1</span>\n\n<div>FSQRT.S computes the square root of rs1</div>\n</div>",
                "tooltip": "FSQRT.S computes the square root of rs1",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest//d.html"
            };

        case "FMADD.D":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, rs2, rs3</span>\n\n<div>FMADD.S multiplies the values in rs1 and rs2 , adds the value in rs3 , and writes the final result to rd</div>\n<div>FMADD.S computes (rs1 \u00c3\u0083\u00c2\u0097 rs2)+rs3 .</div>\n</div>",
                "tooltip": "FMADD.S multiplies the values in rs1 and rs2 , adds the value in rs3 , and writes the final result to rd\nFMADD.S computes (rs1 \u00c3\u0083\u00c2\u0097 rs2)+rs3 .",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest//d.html"
            };

        case "FMSUB.D":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, rs2, rs3</span>\n\n<div>FMSUB.S multiplies the values in rs1 and rs2 , subtracts the value in rs3 , and writes the final result to rd</div>\n<div>FMSUB.S computes (rs1 \u00c3\u0083\u00c2\u0097 rs2)-rs3 .</div>\n</div>",
                "tooltip": "FMSUB.S multiplies the values in rs1 and rs2 , subtracts the value in rs3 , and writes the final result to rd\nFMSUB.S computes (rs1 \u00c3\u0083\u00c2\u0097 rs2)-rs3 .",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest//d.html"
            };

        case "FNMSUB.D":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, rs2, rs3</span>\n\n<div>FNMSUB.S multiplies the values in rs1 and rs2 , negates the product, adds the value in rs3 , and writes the final result to rd</div>\n<div>FNMSUB.S computes -(rs1 \u00c3\u0083\u00c2\u0097 rs2)+rs3 .</div>\n</div>",
                "tooltip": "FNMSUB.S multiplies the values in rs1 and rs2 , negates the product, adds the value in rs3 , and writes the final result to rd\nFNMSUB.S computes -(rs1 \u00c3\u0083\u00c2\u0097 rs2)+rs3 .",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest//d.html"
            };

        case "FNMADD.D":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, rs2, rs3</span>\n\n<div>FNMADD.S multiplies the values in rs1 and rs2 , negates the product, subtracts the value in rs3 , and writes the final result to rd</div>\n<div>FNMADD.S computes -(rs1 \u00c3\u0083\u00c2\u0097 rs2)-rs3 .</div>\n</div>",
                "tooltip": "FNMADD.S multiplies the values in rs1 and rs2 , negates the product, subtracts the value in rs3 , and writes the final result to rd\nFNMADD.S computes -(rs1 \u00c3\u0083\u00c2\u0097 rs2)-rs3 .",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest//d.html"
            };

        case "FSGNJ.D":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, rs2</span>\n\n<div>Floating-point to floating-point sign-injection instructions, FSGNJ.D, FSGNJN.D, and FSGNJX.D are defined analogously to the single-precision sign-injection instruction.</div>\n</div>",
                "tooltip": "Floating-point to floating-point sign-injection instructions, FSGNJ.D, FSGNJN.D, and FSGNJX.D are defined analogously to the single-precision sign-injection instruction.",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/d.html#double-precision-floating-point-conversion-and-move-instructions"
            };

        case "FCVT.S.D":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1</span>\n\n<div>The double-precision to single-precision and single-precision to double-precision conversion instructions, FCVT.S.D and FCVT.D.S, are encoded in the OP-FP major opcode space and both the source and destination are floating-point registers</div>\n<div>FCVT.S.D rounds according to the RM field; FCVT.D.S will never round.</div>\n</div>",
                "tooltip": "The double-precision to single-precision and single-precision to double-precision conversion instructions, FCVT.S.D and FCVT.D.S, are encoded in the OP-FP major opcode space and both the source and destination are floating-point registers\nFCVT.S.D rounds according to the RM field; FCVT.D.S will never round.",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/d.html#double-precision-floating-point-conversion-and-move-instructions"
            };

        case "FCVT.W.D":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1</span>\n\n<div>FCVT.W.D or FCVT.L.D converts a double-precision floating-point number in floating-point register rs1 to a signed 32-bit or 64-bit integer, respectively, in integer register rd</div>\n</div>",
                "tooltip": "FCVT.W.D or FCVT.L.D converts a double-precision floating-point number in floating-point register rs1 to a signed 32-bit or 64-bit integer, respectively, in integer register rd",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/d.html#double-precision-floating-point-conversion-and-move-instructions"
            };

        case "FCVT.WU.D":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1</span>\n\n<div>FCVT.WU.D, FCVT.LU.D, FCVT.D.WU, and FCVT.D.LU variants convert to or from unsigned integer values</div>\n</div>",
                "tooltip": "FCVT.WU.D, FCVT.LU.D, FCVT.D.WU, and FCVT.D.LU variants convert to or from unsigned integer values",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/d.html#double-precision-floating-point-conversion-and-move-instructions"
            };

        case "FMV.X.D":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1</span>\n\n<div>FMV.X.D moves the double-precision value in floating-point register rs1 to a representation in IEEE 754-2008 standard encoding in integer register rd</div>\n<div>FMV.X.D and FMV.D.X do not modify the bits being transferred; in particular, the payloads of non-canonical NaNs are preserved.</div>\n</div>",
                "tooltip": "FMV.X.D moves the double-precision value in floating-point register rs1 to a representation in IEEE 754-2008 standard encoding in integer register rd\nFMV.X.D and FMV.D.X do not modify the bits being transferred; in particular, the payloads of non-canonical NaNs are preserved.",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/d.html#double-precision-floating-point-conversion-and-move-instructions"
            };

        case "FCVT.D.W":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1</span>\n\n<div>FCVT.D.W or FCVT.D.L converts a 32-bit or 64-bit signed integer, respectively, in integer register rs1 into a double-precision floating-point number in floating-point register rd</div>\n<div>Note FCVT.D.W[U] always produces an exact result and is unaffected by rounding mode.</div>\n</div>",
                "tooltip": "FCVT.D.W or FCVT.D.L converts a 32-bit or 64-bit signed integer, respectively, in integer register rs1 into a double-precision floating-point number in floating-point register rd\nNote FCVT.D.W[U] always produces an exact result and is unaffected by rounding mode.",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/d.html#double-precision-floating-point-conversion-and-move-instructions"
            };

        case "FCVT.D.L":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1</span>\n\n<div>FCVT.L[U].D and FCVT.D.L[U] are RV64-only instructions</div>\n</div>",
                "tooltip": "FCVT.L[U].D and FCVT.D.L[U] are RV64-only instructions",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/d.html#double-precision-floating-point-conversion-and-move-instructions"
            };

        case "FMV.D.X":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1</span>\n\n<div>FMV.D.X moves the double-precision value encoded in IEEE 754-2008 standard encoding from the integer register rs1 to the floating-point register rd .</div>\n</div>",
                "tooltip": "FMV.D.X moves the double-precision value encoded in IEEE 754-2008 standard encoding from the integer register rs1 to the floating-point register rd .",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/d.html#double-precision-floating-point-conversion-and-move-instructions"
            };

        case "FLT.D":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, rs2</span>\n\n<div>FLT.S and FLE.S perform what the IEEE 754-2008 standard refers to as signaling comparisons: that is, they set the invalid operation exception flag if either input is NaN</div>\n</div>",
                "tooltip": "FLT.S and FLE.S perform what the IEEE 754-2008 standard refers to as signaling comparisons: that is, they set the invalid operation exception flag if either input is NaN",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest//d.html"
            };

        case "FEQ.D":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, rs2</span>\n\n<div>Floating-point compare instructions (FEQ.S, FLT.S, FLE.S) perform the specified comparison between floating-point registers (</div>\n<div>FEQ.S performs a quiet comparison: it only sets the invalid operation exception flag if either input is a signaling NaN</div>\n</div>",
                "tooltip": "Floating-point compare instructions (FEQ.S, FLT.S, FLE.S) perform the specified comparison between floating-point registers (\nFEQ.S performs a quiet comparison: it only sets the invalid operation exception flag if either input is a signaling NaN",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest//d.html"
            };

        case "FCLASS.D":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1</span>\n\n<div>The double-precision floating-point classify instruction, FCLASS.D, is defined analogously to its single-precision counterpart, but operates on double-precision operands.</div>\n</div>",
                "tooltip": "The double-precision floating-point classify instruction, FCLASS.D, is defined analogously to its single-precision counterpart, but operates on double-precision operands.",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/d.html#double-precision-floating-point-classify-instruction"
            };

        case "FLD":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, imm12</span>\n\n<div>The FLD instruction loads a double-precision floating-point value from memory into floating-point register rd</div>\n<div>FLD and FSD are only guaranteed to execute atomically if the effective address is naturally aligned and XLEN</div>\n<div>FLD and FSD do not modify the bits being transferred; in particular, the payloads of non-canonical NaNs are preserved.</div>\n</div>",
                "tooltip": "The FLD instruction loads a double-precision floating-point value from memory into floating-point register rd\nFLD and FSD are only guaranteed to execute atomically if the effective address is naturally aligned and XLEN\nFLD and FSD do not modify the bits being transferred; in particular, the payloads of non-canonical NaNs are preserved.",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/d.html#fld_fsd"
            };

        case "FSD":
            return {
                "html": "<div>\n<span class=\"opcode\">imm12hi, rs1, rs2, imm12lo</span>\n\n<div>FSD stores a double-precision value from the floating-point registers to memory</div>\n</div>",
                "tooltip": "FSD stores a double-precision value from the floating-point registers to memory",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/d.html#fld_fsd"
            };

        case "XOR":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, rs2</span>\n\n<div>For FSGNJ, the result\u00c3\u00a2\u00c2\u0080\u00c2\u0099s sign bit is rs2 \u00c3\u00a2\u00c2\u0080\u00c2\u0099s sign bit; for FSGNJN, the result\u00c3\u00a2\u00c2\u0080\u00c2\u0099s sign bit is the opposite of rs2 \u00c3\u00a2\u00c2\u0080\u00c2\u0099s sign bit; and for FSGNJX, the sign bit is the XOR of the sign bits of rs1 and rs2</div>\n</div>",
                "tooltip": "For FSGNJ, the result\u00c3\u00a2\u00c2\u0080\u00c2\u0099s sign bit is rs2 \u00c3\u00a2\u00c2\u0080\u00c2\u0099s sign bit; for FSGNJN, the result\u00c3\u00a2\u00c2\u0080\u00c2\u0099s sign bit is the opposite of rs2 \u00c3\u00a2\u00c2\u0080\u00c2\u0099s sign bit; and for FSGNJX, the sign bit is the XOR of the sign bits of rs1 and rs2",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/f.html#single-precision-floating-point-conversion-and-move-instructions"
            };

        case "FSGNJ.S":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, rs2</span>\n\n<div>Floating-point to floating-point sign-injection instructions, FSGNJ.S, FSGNJN.S, and FSGNJX.S, produce a result that takes all bits except the sign bit from rs1</div>\n<div>Note, FSGNJ.S rx, ry, ry moves ry to rx (assembler pseudoinstruction FMV.S rx, ry ); FSGNJN.S rx, ry, ry moves the negation of ry to rx (assembler pseudoinstruction FNEG.S rx, ry ); and FSGNJX.S rx, ry, ry moves the absolute value of ry to rx (assembler pseudoinstruction FABS.S rx, ry ).</div>\n</div>",
                "tooltip": "Floating-point to floating-point sign-injection instructions, FSGNJ.S, FSGNJN.S, and FSGNJX.S, produce a result that takes all bits except the sign bit from rs1\nNote, FSGNJ.S rx, ry, ry moves ry to rx (assembler pseudoinstruction FMV.S rx, ry ); FSGNJN.S rx, ry, ry moves the negation of ry to rx (assembler pseudoinstruction FNEG.S rx, ry ); and FSGNJX.S rx, ry, ry moves the absolute value of ry to rx (assembler pseudoinstruction FABS.S rx, ry ).",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/f.html#single-precision-floating-point-conversion-and-move-instructions"
            };

        case "FCVT.W.S":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1</span>\n\n<div>FCVT.W.S or FCVT.L.S converts a floating-point number in floating-point register rs1 to a signed 32-bit or 64-bit integer, respectively, in integer register rd</div>\n<div>FCVT.W.S</div>\n</div>",
                "tooltip": "FCVT.W.S or FCVT.L.S converts a floating-point number in floating-point register rs1 to a signed 32-bit or 64-bit integer, respectively, in integer register rd\nFCVT.W.S",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/f.html#single-precision-floating-point-conversion-and-move-instructions"
            };

        case "FCVT.WU.S":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1</span>\n\n<div>FCVT.WU.S, FCVT.LU.S, FCVT.S.WU, and FCVT.S.LU variants convert to or from unsigned integer values</div>\n<div>FCVT.WU.S</div>\n</div>",
                "tooltip": "FCVT.WU.S, FCVT.LU.S, FCVT.S.WU, and FCVT.S.LU variants convert to or from unsigned integer values\nFCVT.WU.S",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/f.html#single-precision-floating-point-conversion-and-move-instructions"
            };

        case "FCVT.L.S":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1</span>\n\n<div>FCVT.L.S</div>\n</div>",
                "tooltip": "FCVT.L.S",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/f.html#single-precision-floating-point-conversion-and-move-instructions"
            };

        case "FCVT.LU.S":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1</span>\n\n<div>FCVT.LU.S</div>\n</div>",
                "tooltip": "FCVT.LU.S",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/f.html#single-precision-floating-point-conversion-and-move-instructions"
            };

        case "FMV.X.W":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1</span>\n\n<div>FMV.X.W moves the single-precision value in floating-point register rs1 rd</div>\n</div>",
                "tooltip": "FMV.X.W moves the single-precision value in floating-point register rs1 rd",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/f.html#single-precision-floating-point-conversion-and-move-instructions"
            };

        case "FCVT.S.W":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1</span>\n\n<div>FCVT.S.W or FCVT.S.L converts a 32-bit or 64-bit signed integer, respectively, in integer register rs1 into a floating-point number in floating-point register rd</div>\n<div>A floating-point register can be initialized to floating-point positive zero using FCVT.S.W rd , x0 , which will never set any exception flags.</div>\n</div>",
                "tooltip": "FCVT.S.W or FCVT.S.L converts a 32-bit or 64-bit signed integer, respectively, in integer register rs1 into a floating-point number in floating-point register rd\nA floating-point register can be initialized to floating-point positive zero using FCVT.S.W rd , x0 , which will never set any exception flags.",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/f.html#single-precision-floating-point-conversion-and-move-instructions"
            };

        case "FCVT.S.L":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1</span>\n\n<div>FCVT.L[U].S and FCVT.S.L[U] are RV64-only instructions</div>\n</div>",
                "tooltip": "FCVT.L[U].S and FCVT.S.L[U] are RV64-only instructions",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/f.html#single-precision-floating-point-conversion-and-move-instructions"
            };

        case "FMV.W.X":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1</span>\n\n<div>FMV.W.X moves the single-precision value encoded in IEEE 754-2008 standard encoding from the lower 32 bits of integer register rs1 to the floating-point register rd</div>\n<div>The FMV.W.X and FMV.X.W instructions were previously called FMV.S.X and FMV.X.S</div>\n</div>",
                "tooltip": "FMV.W.X moves the single-precision value encoded in IEEE 754-2008 standard encoding from the lower 32 bits of integer register rs1 to the floating-point register rd\nThe FMV.W.X and FMV.X.W instructions were previously called FMV.S.X and FMV.X.S",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/f.html#single-precision-floating-point-conversion-and-move-instructions"
            };

        case "FADD.S":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, rs2</span>\n\n<div>FADD.S and FMUL.S perform single-precision floating-point addition and multiplication respectively, between rs1 and rs2</div>\n</div>",
                "tooltip": "FADD.S and FMUL.S perform single-precision floating-point addition and multiplication respectively, between rs1 and rs2",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/f.html#sec:single-float-compute"
            };

        case "FSUB.S":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, rs2</span>\n\n<div>FSUB.S performs the single-precision floating-point subtraction of rs2 from rs1</div>\n</div>",
                "tooltip": "FSUB.S performs the single-precision floating-point subtraction of rs2 from rs1",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/f.html#sec:single-float-compute"
            };

        case "FDIV.S":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, rs2</span>\n\n<div>FDIV.S performs the single-precision floating-point division of rs1 by rs2</div>\n</div>",
                "tooltip": "FDIV.S performs the single-precision floating-point division of rs1 by rs2",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/f.html#sec:single-float-compute"
            };

        case "FMIN.S":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, rs2</span>\n\n<div>Floating-point minimum-number and maximum-number instructions FMIN.S and FMAX.S write, respectively, the smaller or larger of rs1 and rs2 rd</div>\n<div>Note that in version 2.2 of the F extension, the FMIN.S and FMAX.S instructions were amended to implement the proposed IEEE 754-201x minimumNumber and maximumNumber operations, rather than the IEEE 754-2008 minNum and maxNum operations</div>\n</div>",
                "tooltip": "Floating-point minimum-number and maximum-number instructions FMIN.S and FMAX.S write, respectively, the smaller or larger of rs1 and rs2 rd\nNote that in version 2.2 of the F extension, the FMIN.S and FMAX.S instructions were amended to implement the proposed IEEE 754-201x minimumNumber and maximumNumber operations, rather than the IEEE 754-2008 minNum and maxNum operations",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/f.html#sec:single-float-compute"
            };

        case "FSQRT.S":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1</span>\n\n<div>FSQRT.S computes the square root of rs1</div>\n</div>",
                "tooltip": "FSQRT.S computes the square root of rs1",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/f.html#sec:single-float-compute"
            };

        case "FMADD.S":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, rs2, rs3</span>\n\n<div>FMADD.S multiplies the values in rs1 and rs2 , adds the value in rs3 , and writes the final result to rd</div>\n<div>FMADD.S computes (rs1 \u00c3\u0083\u00c2\u0097 rs2)+rs3 .</div>\n</div>",
                "tooltip": "FMADD.S multiplies the values in rs1 and rs2 , adds the value in rs3 , and writes the final result to rd\nFMADD.S computes (rs1 \u00c3\u0083\u00c2\u0097 rs2)+rs3 .",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/f.html#sec:single-float-compute"
            };

        case "FMSUB.S":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, rs2, rs3</span>\n\n<div>FMSUB.S multiplies the values in rs1 and rs2 , subtracts the value in rs3 , and writes the final result to rd</div>\n<div>FMSUB.S computes (rs1 \u00c3\u0083\u00c2\u0097 rs2)-rs3 .</div>\n</div>",
                "tooltip": "FMSUB.S multiplies the values in rs1 and rs2 , subtracts the value in rs3 , and writes the final result to rd\nFMSUB.S computes (rs1 \u00c3\u0083\u00c2\u0097 rs2)-rs3 .",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/f.html#sec:single-float-compute"
            };

        case "FNMSUB.S":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, rs2, rs3</span>\n\n<div>FNMSUB.S multiplies the values in rs1 and rs2 , negates the product, adds the value in rs3 , and writes the final result to rd</div>\n<div>FNMSUB.S computes -(rs1 \u00c3\u0083\u00c2\u0097 rs2)+rs3 .</div>\n</div>",
                "tooltip": "FNMSUB.S multiplies the values in rs1 and rs2 , negates the product, adds the value in rs3 , and writes the final result to rd\nFNMSUB.S computes -(rs1 \u00c3\u0083\u00c2\u0097 rs2)+rs3 .",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/f.html#sec:single-float-compute"
            };

        case "FNMADD.S":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, rs2, rs3</span>\n\n<div>FNMADD.S multiplies the values in rs1 and rs2 , negates the product, subtracts the value in rs3 , and writes the final result to rd</div>\n<div>FNMADD.S computes -(rs1 \u00c3\u0083\u00c2\u0097 rs2)-rs3 .</div>\n</div>",
                "tooltip": "FNMADD.S multiplies the values in rs1 and rs2 , negates the product, subtracts the value in rs3 , and writes the final result to rd\nFNMADD.S computes -(rs1 \u00c3\u0083\u00c2\u0097 rs2)-rs3 .",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/f.html#sec:single-float-compute"
            };

        case "FLT.S":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, rs2</span>\n\n<div>FLT.S and FLE.S perform what the IEEE 754-2008 standard refers to as signaling comparisons: that is, they set the invalid operation exception flag if either input is NaN</div>\n</div>",
                "tooltip": "FLT.S and FLE.S perform what the IEEE 754-2008 standard refers to as signaling comparisons: that is, they set the invalid operation exception flag if either input is NaN",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/f.html#single-precision-floating-point-compare-instructions"
            };

        case "FEQ.S":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, rs2</span>\n\n<div>Floating-point compare instructions (FEQ.S, FLT.S, FLE.S) perform the specified comparison between floating-point registers (</div>\n<div>FEQ.S performs a quiet comparison: it only sets the invalid operation exception flag if either input is a signaling NaN</div>\n</div>",
                "tooltip": "Floating-point compare instructions (FEQ.S, FLT.S, FLE.S) perform the specified comparison between floating-point registers (\nFEQ.S performs a quiet comparison: it only sets the invalid operation exception flag if either input is a signaling NaN",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/f.html#single-precision-floating-point-compare-instructions"
            };

        case "FCLASS.S":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1</span>\n\n<div>The FCLASS.S instruction examines the value in floating-point register rs1 and writes to integer register rd a 10-bit mask that indicates the class of the floating-point number</div>\n<div>FCLASS.S does not set the floating-point exception flags</div>\n</div>",
                "tooltip": "The FCLASS.S instruction examines the value in floating-point register rs1 and writes to integer register rd a 10-bit mask that indicates the class of the floating-point number\nFCLASS.S does not set the floating-point exception flags",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/f.html#single-precision-floating-point-classify-instruction"
            };

        case "FLW":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, imm12</span>\n\n<div>The FLW instruction loads a single-precision floating-point value from memory into floating-point register rd</div>\n<div>FLW and FSW are only guaranteed to execute atomically if the effective address is naturally aligned.</div>\n<div>FLW and FSW do not modify the bits being transferred; in particular, the payloads of non-canonical NaNs are preserved.</div>\n</div>",
                "tooltip": "The FLW instruction loads a single-precision floating-point value from memory into floating-point register rd\nFLW and FSW are only guaranteed to execute atomically if the effective address is naturally aligned.\nFLW and FSW do not modify the bits being transferred; in particular, the payloads of non-canonical NaNs are preserved.",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/f.html#single-precision-load-and-store-instructions"
            };

        case "FSW":
            return {
                "html": "<div>\n<span class=\"opcode\">imm12hi, rs1, rs2, imm12lo</span>\n\n<div>FSW stores a single-precision value from floating-point register rs2 to memory.</div>\n</div>",
                "tooltip": "FSW stores a single-precision value from floating-point register rs2 to memory.",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/f.html#single-precision-load-and-store-instructions"
            };

        case "MUL":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, rs2</span>\n\n<div>MUL performs an XLEN-bit</div>\n<div>In RV64, MUL can be used to obtain the upper 32 bits of the 64-bit product, but signed arguments must be proper 32-bit signed values, whereas unsigned arguments must have their upper 32 bits clear</div>\n</div>",
                "tooltip": "MUL performs an XLEN-bit\nIn RV64, MUL can be used to obtain the upper 32 bits of the 64-bit product, but signed arguments must be proper 32-bit signed values, whereas unsigned arguments must have their upper 32 bits clear",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/m.html#multiplication-operations"
            };

        case "MULH":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, rs2</span>\n\n<div>MULH, MULHU, and MULHSU perform the same multiplication but return the upper XLEN bits of the full 2</div>\n<div>If both the high and low bits of the same product are required, then the recommended code sequence is: MULH[[S]U] rdh, rs1, rs2 ; MUL rdl, rs1, rs2 (source register specifiers must be in same order and rdh cannot be the same as rs1 or rs2 )</div>\n<div>If the arguments are not known to be sign- or zero-extended, an alternative is to shift both arguments left by 32 bits, then use MULH[[S]U].</div>\n</div>",
                "tooltip": "MULH, MULHU, and MULHSU perform the same multiplication but return the upper XLEN bits of the full 2\nIf both the high and low bits of the same product are required, then the recommended code sequence is: MULH[[S]U] rdh, rs1, rs2 ; MUL rdl, rs1, rs2 (source register specifiers must be in same order and rdh cannot be the same as rs1 or rs2 )\nIf the arguments are not known to be sign- or zero-extended, an alternative is to shift both arguments left by 32 bits, then use MULH[[S]U].",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/m.html#multiplication-operations"
            };

        case "MULHSU":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, rs2</span>\n\n<div>MULHSU is used in multi-word signed multiplication to multiply the most-significant word of the multiplicand (which contains the sign bit) with the less-significant words of the multiplier (which are unsigned).</div>\n</div>",
                "tooltip": "MULHSU is used in multi-word signed multiplication to multiply the most-significant word of the multiplicand (which contains the sign bit) with the less-significant words of the multiplier (which are unsigned).",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/m.html#multiplication-operations"
            };

        case "MULW":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, rs2</span>\n\n<div>MULW is an RV64 instruction that multiplies the lower 32 bits of the source registers, placing the sign-extension of the lower 32 bits of the result into the destination register.</div>\n</div>",
                "tooltip": "MULW is an RV64 instruction that multiplies the lower 32 bits of the source registers, placing the sign-extension of the lower 32 bits of the result into the destination register.",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/m.html#multiplication-operations"
            };

        case "DIV":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, rs2</span>\n\n<div>DIV and DIVU perform an XLEN bits by XLEN bits signed and unsigned integer division of rs1 by rs2 , rounding towards zero</div>\n<div>If both the quotient and remainder are required from the same division, the recommended code sequence is: DIV[U] rdq, rs1, rs2 ; REM[U] rdr, rs1, rs2 ( rdq rs1 or rs2 )</div>\n<div>DIV[W]</div>\n</div>",
                "tooltip": "DIV and DIVU perform an XLEN bits by XLEN bits signed and unsigned integer division of rs1 by rs2 , rounding towards zero\nIf both the quotient and remainder are required from the same division, the recommended code sequence is: DIV[U] rdq, rs1, rs2 ; REM[U] rdr, rs1, rs2 ( rdq rs1 or rs2 )\nDIV[W]",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/m.html#division-operations"
            };

        case "DIVU":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, rs2</span>\n\n<div>DIVU[W]</div>\n</div>",
                "tooltip": "DIVU[W]",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/m.html#division-operations"
            };

        case "REM":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, rs2</span>\n\n<div>REM and REMU provide the remainder of the corresponding division operation</div>\n<div>For REM, the sign of the result equals the sign of the dividend.</div>\n<div>REM[W]</div>\n</div>",
                "tooltip": "REM and REMU provide the remainder of the corresponding division operation\nFor REM, the sign of the result equals the sign of the dividend.\nREM[W]",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/m.html#division-operations"
            };

        case "REMU":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, rs2</span>\n\n<div>REMU[W]</div>\n</div>",
                "tooltip": "REMU[W]",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/m.html#division-operations"
            };

        case "DIVW":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, rs2</span>\n\n<div>DIVW and DIVUW are RV64 instructions that divide the lower 32 bits of rs1 by the lower 32 bits of rs2 , treating them as signed and unsigned integers respectively, placing the 32-bit quotient in rd , sign-extended to 64 bits</div>\n</div>",
                "tooltip": "DIVW and DIVUW are RV64 instructions that divide the lower 32 bits of rs1 by the lower 32 bits of rs2 , treating them as signed and unsigned integers respectively, placing the 32-bit quotient in rd , sign-extended to 64 bits",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/m.html#division-operations"
            };

        case "REMW":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, rs2</span>\n\n<div>REMW and REMUW are RV64 instructions that provide the corresponding signed and unsigned remainder operations respectively</div>\n<div>Both REMW and REMUW always sign-extend the 32-bit result to 64 bits, including on a divide by zero.</div>\n</div>",
                "tooltip": "REMW and REMUW are RV64 instructions that provide the corresponding signed and unsigned remainder operations respectively\nBoth REMW and REMUW always sign-extend the 32-bit result to 64 bits, including on a divide by zero.",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/m.html#division-operations"
            };

        case "URET":
            return {
                "html": "<div>\n<span class=\"opcode\"></span>\n\n<div>A new instruction, URET, is used to return from traps in U-mode</div>\n<div>URET copies UPIE into UIE, then sets UPIE, before copying uepc pc</div>\n</div>",
                "tooltip": "A new instruction, URET, is used to return from traps in U-mode\nURET copies UPIE into UIE, then sets UPIE, before copying uepc pc",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/n.html#user-status-register-ustatus"
            };

        case "FADD.Q":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, rs2</span>\n\n<div>FADD.S and FMUL.S perform single-precision floating-point addition and multiplication respectively, between rs1 and rs2</div>\n</div>",
                "tooltip": "FADD.S and FMUL.S perform single-precision floating-point addition and multiplication respectively, between rs1 and rs2",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest//q.html"
            };

        case "FSUB.Q":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, rs2</span>\n\n<div>FSUB.S performs the single-precision floating-point subtraction of rs2 from rs1</div>\n</div>",
                "tooltip": "FSUB.S performs the single-precision floating-point subtraction of rs2 from rs1",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest//q.html"
            };

        case "FDIV.Q":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, rs2</span>\n\n<div>FDIV.S performs the single-precision floating-point division of rs1 by rs2</div>\n</div>",
                "tooltip": "FDIV.S performs the single-precision floating-point division of rs1 by rs2",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest//q.html"
            };

        case "FMIN.Q":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, rs2</span>\n\n<div>Floating-point minimum-number and maximum-number instructions FMIN.S and FMAX.S write, respectively, the smaller or larger of rs1 and rs2 rd</div>\n<div>Note that in version 2.2 of the F extension, the FMIN.S and FMAX.S instructions were amended to implement the proposed IEEE 754-201x minimumNumber and maximumNumber operations, rather than the IEEE 754-2008 minNum and maxNum operations</div>\n</div>",
                "tooltip": "Floating-point minimum-number and maximum-number instructions FMIN.S and FMAX.S write, respectively, the smaller or larger of rs1 and rs2 rd\nNote that in version 2.2 of the F extension, the FMIN.S and FMAX.S instructions were amended to implement the proposed IEEE 754-201x minimumNumber and maximumNumber operations, rather than the IEEE 754-2008 minNum and maxNum operations",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest//q.html"
            };

        case "FSQRT.Q":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1</span>\n\n<div>FSQRT.S computes the square root of rs1</div>\n</div>",
                "tooltip": "FSQRT.S computes the square root of rs1",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest//q.html"
            };

        case "FMADD.Q":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, rs2, rs3</span>\n\n<div>FMADD.S multiplies the values in rs1 and rs2 , adds the value in rs3 , and writes the final result to rd</div>\n<div>FMADD.S computes (rs1 \u00c3\u0083\u00c2\u0097 rs2)+rs3 .</div>\n</div>",
                "tooltip": "FMADD.S multiplies the values in rs1 and rs2 , adds the value in rs3 , and writes the final result to rd\nFMADD.S computes (rs1 \u00c3\u0083\u00c2\u0097 rs2)+rs3 .",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest//q.html"
            };

        case "FMSUB.Q":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, rs2, rs3</span>\n\n<div>FMSUB.S multiplies the values in rs1 and rs2 , subtracts the value in rs3 , and writes the final result to rd</div>\n<div>FMSUB.S computes (rs1 \u00c3\u0083\u00c2\u0097 rs2)-rs3 .</div>\n</div>",
                "tooltip": "FMSUB.S multiplies the values in rs1 and rs2 , subtracts the value in rs3 , and writes the final result to rd\nFMSUB.S computes (rs1 \u00c3\u0083\u00c2\u0097 rs2)-rs3 .",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest//q.html"
            };

        case "FNMSUB.Q":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, rs2, rs3</span>\n\n<div>FNMSUB.S multiplies the values in rs1 and rs2 , negates the product, adds the value in rs3 , and writes the final result to rd</div>\n<div>FNMSUB.S computes -(rs1 \u00c3\u0083\u00c2\u0097 rs2)+rs3 .</div>\n</div>",
                "tooltip": "FNMSUB.S multiplies the values in rs1 and rs2 , negates the product, adds the value in rs3 , and writes the final result to rd\nFNMSUB.S computes -(rs1 \u00c3\u0083\u00c2\u0097 rs2)+rs3 .",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest//q.html"
            };

        case "FNMADD.Q":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, rs2, rs3</span>\n\n<div>FNMADD.S multiplies the values in rs1 and rs2 , negates the product, subtracts the value in rs3 , and writes the final result to rd</div>\n<div>FNMADD.S computes -(rs1 \u00c3\u0083\u00c2\u0097 rs2)-rs3 .</div>\n</div>",
                "tooltip": "FNMADD.S multiplies the values in rs1 and rs2 , negates the product, subtracts the value in rs3 , and writes the final result to rd\nFNMADD.S computes -(rs1 \u00c3\u0083\u00c2\u0097 rs2)-rs3 .",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest//q.html"
            };

        case "FSGNJ.Q":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, rs2</span>\n\n<div>Floating-point to floating-point sign-injection instructions, FSGNJ.Q, FSGNJN.Q, and FSGNJX.Q are defined analogously to the double-precision sign-injection instruction.</div>\n</div>",
                "tooltip": "Floating-point to floating-point sign-injection instructions, FSGNJ.Q, FSGNJN.Q, and FSGNJX.Q are defined analogously to the double-precision sign-injection instruction.",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/q.html#quad-precision-convert-and-move-instructions"
            };

        case "FCVT.S.Q":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1</span>\n\n<div>FCVT.S.Q or FCVT.Q.S converts a quad-precision floating-point number to a single-precision floating-point number, or vice-versa, respectively</div>\n</div>",
                "tooltip": "FCVT.S.Q or FCVT.Q.S converts a quad-precision floating-point number to a single-precision floating-point number, or vice-versa, respectively",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/q.html#quad-precision-convert-and-move-instructions"
            };

        case "FCVT.D.Q":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1</span>\n\n<div>FCVT.D.Q or FCVT.Q.D converts a quad-precision floating-point number to a double-precision floating-point number, or vice-versa, respectively.</div>\n</div>",
                "tooltip": "FCVT.D.Q or FCVT.Q.D converts a quad-precision floating-point number to a double-precision floating-point number, or vice-versa, respectively.",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/q.html#quad-precision-convert-and-move-instructions"
            };

        case "FCVT.W.Q":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1</span>\n\n<div>FCVT.W.Q or FCVT.L.Q converts a quad-precision floating-point number to a signed 32-bit or 64-bit integer, respectively</div>\n</div>",
                "tooltip": "FCVT.W.Q or FCVT.L.Q converts a quad-precision floating-point number to a signed 32-bit or 64-bit integer, respectively",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/q.html#quad-precision-convert-and-move-instructions"
            };

        case "FCVT.WU.Q":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1</span>\n\n<div>FCVT.WU.Q, FCVT.LU.Q, FCVT.Q.WU, and FCVT.Q.LU variants convert to or from unsigned integer values</div>\n</div>",
                "tooltip": "FCVT.WU.Q, FCVT.LU.Q, FCVT.Q.WU, and FCVT.Q.LU variants convert to or from unsigned integer values",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/q.html#quad-precision-convert-and-move-instructions"
            };

        case "FCVT.Q.W":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1</span>\n\n<div>FCVT.Q.W or FCVT.Q.L converts a 32-bit or 64-bit signed integer, respectively, into a quad-precision floating-point number</div>\n</div>",
                "tooltip": "FCVT.Q.W or FCVT.Q.L converts a 32-bit or 64-bit signed integer, respectively, into a quad-precision floating-point number",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/q.html#quad-precision-convert-and-move-instructions"
            };

        case "FCVT.Q.L":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1</span>\n\n<div>FCVT.L[U].Q and FCVT.Q.L[U] are RV64-only instructions.</div>\n</div>",
                "tooltip": "FCVT.L[U].Q and FCVT.Q.L[U] are RV64-only instructions.",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/q.html#quad-precision-convert-and-move-instructions"
            };

        case "FLT.Q":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, rs2</span>\n\n<div>FLT.S and FLE.S perform what the IEEE 754-2008 standard refers to as signaling comparisons: that is, they set the invalid operation exception flag if either input is NaN</div>\n</div>",
                "tooltip": "FLT.S and FLE.S perform what the IEEE 754-2008 standard refers to as signaling comparisons: that is, they set the invalid operation exception flag if either input is NaN",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest//q.html"
            };

        case "FEQ.Q":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, rs2</span>\n\n<div>Floating-point compare instructions (FEQ.S, FLT.S, FLE.S) perform the specified comparison between floating-point registers (</div>\n<div>FEQ.S performs a quiet comparison: it only sets the invalid operation exception flag if either input is a signaling NaN</div>\n</div>",
                "tooltip": "Floating-point compare instructions (FEQ.S, FLT.S, FLE.S) perform the specified comparison between floating-point registers (\nFEQ.S performs a quiet comparison: it only sets the invalid operation exception flag if either input is a signaling NaN",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest//q.html"
            };

        case "FCLASS.Q":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1</span>\n\n<div>The quad-precision floating-point classify instruction, FCLASS.Q, is defined analogously to its double-precision counterpart, but operates on quad-precision operands.</div>\n</div>",
                "tooltip": "The quad-precision floating-point classify instruction, FCLASS.Q, is defined analogously to its double-precision counterpart, but operates on quad-precision operands.",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/q.html#quad-precision-floating-point-classify-instruction"
            };

        case "FLQ":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, imm12</span>\n\n<div>FLQ and FSQ are only guaranteed to execute atomically if the effective address is naturally aligned and XLEN=128.</div>\n<div>FLQ and FSQ do not modify the bits being transferred; in particular, the payloads of non-canonical NaNs are preserved.</div>\n</div>",
                "tooltip": "FLQ and FSQ are only guaranteed to execute atomically if the effective address is naturally aligned and XLEN=128.\nFLQ and FSQ do not modify the bits being transferred; in particular, the payloads of non-canonical NaNs are preserved.",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/q.html#quad-precision-load-and-store-instructions"
            };

        case "VSETVLI":
            return {
                "html": "<div>\n<span class=\"opcode\">zimm11, rs1, rd</span>\n\n<div>vl The XLEN -bit-wide read-only vl CSR can only be updated by the vsetvli and vsetvl instructions, and the fault-only-first vector load instruction variants.</div>\n</div>",
                "tooltip": "vl The XLEN -bit-wide read-only vl CSR can only be updated by the vsetvli and vsetvl instructions, and the fault-only-first vector load instruction variants.",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_length_register_code_vl_code"
            };

        case "VSETVL":
            return {
                "html": "<div>\n<span class=\"opcode\">rs2, rs1, rd</span>\n\n<div>vtype The read-only XLEN-wide vector type CSR, vtype provides the default type used to interpret the contents of the vector register file, and can only be updated by vsetvl{i} instructions</div>\n<div>Allowing updates only via the vsetvl{i} vtype register state</div>\n</div>",
                "tooltip": "vtype The read-only XLEN-wide vector type CSR, vtype provides the default type used to interpret the contents of the vector register file, and can only be updated by vsetvl{i} instructions\nAllowing updates only via the vsetvl{i} vtype register state",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_type_register_code_vtype_code"
            };

        case "VLB.V":
            return {
                "html": "<div>\n<span class=\"opcode\">rs1, vd</span>\n\n<div>vlb.v  vd, (rs1), vm # 8b signed</div>\n</div>",
                "tooltip": "vlb.v  vd, (rs1), vm # 8b signed",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_unit_stride_instructions"
            };

        case "VLW.V":
            return {
                "html": "<div>\n<span class=\"opcode\">rs1, vd</span>\n\n<div>vlw.v  vd, (rs1), vm # 32b signed</div>\n</div>",
                "tooltip": "vlw.v  vd, (rs1), vm # 32b signed",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_unit_stride_instructions"
            };

        case "VLE.V":
            return {
                "html": "<div>\n<span class=\"opcode\">rs1, vd</span>\n\n<div>vle.v  vd, (rs1), vm # SEW</div>\n</div>",
                "tooltip": "vle.v  vd, (rs1), vm # SEW",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_unit_stride_instructions"
            };

        case "VLBU.V":
            return {
                "html": "<div>\n<span class=\"opcode\">rs1, vd</span>\n\n<div>vlbu.v vd, (rs1), vm # 8b unsigned</div>\n</div>",
                "tooltip": "vlbu.v vd, (rs1), vm # 8b unsigned",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_unit_stride_instructions"
            };

        case "VLHU.V":
            return {
                "html": "<div>\n<span class=\"opcode\">rs1, vd</span>\n\n<div>vlhu.v vd, (rs1), vm # 16b unsigned</div>\n</div>",
                "tooltip": "vlhu.v vd, (rs1), vm # 16b unsigned",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_unit_stride_instructions"
            };

        case "VLWU.V":
            return {
                "html": "<div>\n<span class=\"opcode\">rs1, vd</span>\n\n<div>vlwu.v vd, (rs1), vm # 32b unsigned</div>\n</div>",
                "tooltip": "vlwu.v vd, (rs1), vm # 32b unsigned",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_unit_stride_instructions"
            };

        case "VSB.V":
            return {
                "html": "<div>\n<span class=\"opcode\">rs1, vs3</span>\n\n<div>vsb.v  vs3, (rs1), vm  # 8b store</div>\n</div>",
                "tooltip": "vsb.v  vs3, (rs1), vm  # 8b store",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_unit_stride_instructions"
            };

        case "VSH.V":
            return {
                "html": "<div>\n<span class=\"opcode\">rs1, vs3</span>\n\n<div>vsh.v  vs3, (rs1), vm  # 16b store</div>\n</div>",
                "tooltip": "vsh.v  vs3, (rs1), vm  # 16b store",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_unit_stride_instructions"
            };

        case "VSE.V":
            return {
                "html": "<div>\n<span class=\"opcode\">rs1, vs3</span>\n\n<div>vse.v  vs3, (rs1), vm  # SEW store</div>\n</div>",
                "tooltip": "vse.v  vs3, (rs1), vm  # SEW store",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_unit_stride_instructions"
            };

        case "VLH.V":
            return {
                "html": "<div>\n<span class=\"opcode\">rs1, vd</span>\n\n<div>vlh.v v8, (a1)          # Sign-extend 16b load values to 32b elements</div>\n<div>vlh.v v4, (a1)          # Get 16b vector</div>\n</div>",
                "tooltip": "vlh.v v8, (a1)          # Sign-extend 16b load values to 32b elements\nvlh.v v4, (a1)          # Get 16b vector",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_examples"
            };

        case "VSW.V":
            return {
                "html": "<div>\n<span class=\"opcode\">rs1, vs3</span>\n\n<div>vsw.v v8, (a2)          # Store vector of 32b results</div>\n<div>vsw.v v8, (a2)          # Store vector of 32b</div>\n</div>",
                "tooltip": "vsw.v v8, (a2)          # Store vector of 32b results\nvsw.v v8, (a2)          # Store vector of 32b",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_examples"
            };

        case "VSRL.VX":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vsrl.vi  v8, v8, 3      # Shift elements</div>\n<div>vsrl.vi v8, v8, 3</div>\n</div>",
                "tooltip": "vsrl.vi  v8, v8, 3      # Shift elements\nvsrl.vi v8, v8, 3",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_examples"
            };

        case "VSRL.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vsrl.vi  v8, v8, 3      # Shift elements</div>\n<div>vsrl.vi v8, v8, 3</div>\n</div>",
                "tooltip": "vsrl.vi  v8, v8, 3      # Shift elements\nvsrl.vi v8, v8, 3",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_examples"
            };

        case "VSRL.VI":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, simm5, vd</span>\n\n<div>vsrl.vi  v8, v8, 3      # Shift elements</div>\n<div>vsrl.vi v8, v8, 3</div>\n</div>",
                "tooltip": "vsrl.vi  v8, v8, 3      # Shift elements\nvsrl.vi v8, v8, 3",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_examples"
            };

        case "VMUL.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n<div>vmul.vx  v8, v8, x10    # 32b multiply result</div>\n</div>",
                "tooltip": "vmul.vx  v8, v8, x10    # 32b multiply result",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_examples"
            };

        case "VWMUL.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n<div>vwmul.vx v8, v4, x10    # 32b in &lt;v8--v15&gt;</div>\n</div>",
                "tooltip": "vwmul.vx v8, v4, x10    # 32b in <v8--v15>",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_examples"
            };

        case "VMUL.VX":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vmul.vx  v8, v8, x10    # 32b multiply result</div>\n</div>",
                "tooltip": "vmul.vx  v8, v8, x10    # 32b multiply result",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_examples"
            };

        case "VWMUL.VX":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vwmul.vx v8, v4, x10    # 32b in &lt;v8--v15&gt;</div>\n</div>",
                "tooltip": "vwmul.vx v8, v4, x10    # 32b in <v8--v15>",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_examples"
            };

        case "VLSB.V":
            return {
                "html": "<div>\n<span class=\"opcode\">rs2, rs1, vd</span>\n\n<div>vlsb.v  vd, (rs1), rs2, vm # 8b</div>\n</div>",
                "tooltip": "vlsb.v  vd, (rs1), rs2, vm # 8b",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_strided_instructions"
            };

        case "VLSH.V":
            return {
                "html": "<div>\n<span class=\"opcode\">rs2, rs1, vd</span>\n\n<div>vlsh.v  vd, (rs1), rs2, vm # 16b</div>\n</div>",
                "tooltip": "vlsh.v  vd, (rs1), rs2, vm # 16b",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_strided_instructions"
            };

        case "VLSW.V":
            return {
                "html": "<div>\n<span class=\"opcode\">rs2, rs1, vd</span>\n\n<div>vlsw.v  vd, (rs1), rs2, vm # 32b</div>\n</div>",
                "tooltip": "vlsw.v  vd, (rs1), rs2, vm # 32b",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_strided_instructions"
            };

        case "VLSE.V":
            return {
                "html": "<div>\n<span class=\"opcode\">rs2, rs1, vd</span>\n\n<div>vlse.v  vd, (rs1), rs2, vm  # SEW</div>\n</div>",
                "tooltip": "vlse.v  vd, (rs1), rs2, vm  # SEW",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_strided_instructions"
            };

        case "VLSBU.V":
            return {
                "html": "<div>\n<span class=\"opcode\">rs2, rs1, vd</span>\n\n<div>vlsbu.v vd, (rs1), rs2, vm # unsigned 8b</div>\n</div>",
                "tooltip": "vlsbu.v vd, (rs1), rs2, vm # unsigned 8b",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_strided_instructions"
            };

        case "VLSHU.V":
            return {
                "html": "<div>\n<span class=\"opcode\">rs2, rs1, vd</span>\n\n<div>vlshu.v vd, (rs1), rs2, vm # unsigned 16b</div>\n</div>",
                "tooltip": "vlshu.v vd, (rs1), rs2, vm # unsigned 16b",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_strided_instructions"
            };

        case "VLSWU.V":
            return {
                "html": "<div>\n<span class=\"opcode\">rs2, rs1, vd</span>\n\n<div>vlswu.v vd, (rs1), rs2, vm # unsigned 32b</div>\n</div>",
                "tooltip": "vlswu.v vd, (rs1), rs2, vm # unsigned 32b",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_strided_instructions"
            };

        case "VSSB.V":
            return {
                "html": "<div>\n<span class=\"opcode\">rs2, rs1, vs3</span>\n\n<div>vssb.v vs3, (rs1), rs2, vm  # 8b</div>\n</div>",
                "tooltip": "vssb.v vs3, (rs1), rs2, vm  # 8b",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_strided_instructions"
            };

        case "VSSH.V":
            return {
                "html": "<div>\n<span class=\"opcode\">rs2, rs1, vs3</span>\n\n<div>vssh.v vs3, (rs1), rs2, vm  # 16b</div>\n</div>",
                "tooltip": "vssh.v vs3, (rs1), rs2, vm  # 16b",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_strided_instructions"
            };

        case "VSSW.V":
            return {
                "html": "<div>\n<span class=\"opcode\">rs2, rs1, vs3</span>\n\n<div>vssw.v vs3, (rs1), rs2, vm  # 32b</div>\n</div>",
                "tooltip": "vssw.v vs3, (rs1), rs2, vm  # 32b",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_strided_instructions"
            };

        case "VSSE.V":
            return {
                "html": "<div>\n<span class=\"opcode\">rs2, rs1, vs3</span>\n\n<div>vsse.v vs3, (rs1), rs2, vm  # SEW</div>\n</div>",
                "tooltip": "vsse.v vs3, (rs1), rs2, vm  # SEW",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_strided_instructions"
            };

        case "VLXB.V":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vlxb.v  vd, (rs1), vs2, vm  # 8b</div>\n</div>",
                "tooltip": "vlxb.v  vd, (rs1), vs2, vm  # 8b",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_indexed_instructions"
            };

        case "VLXH.V":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vlxh.v  vd, (rs1), vs2, vm  # 16b</div>\n</div>",
                "tooltip": "vlxh.v  vd, (rs1), vs2, vm  # 16b",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_indexed_instructions"
            };

        case "VLXW.V":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vlxw.v  vd, (rs1), vs2, vm  # 32b</div>\n</div>",
                "tooltip": "vlxw.v  vd, (rs1), vs2, vm  # 32b",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_indexed_instructions"
            };

        case "VLXE.V":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vlxe.v  vd, (rs1), vs2, vm  # SEW</div>\n</div>",
                "tooltip": "vlxe.v  vd, (rs1), vs2, vm  # SEW",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_indexed_instructions"
            };

        case "VLXBU.V":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vlxbu.v vd, (rs1), vs2, vm  # 8b unsigned</div>\n</div>",
                "tooltip": "vlxbu.v vd, (rs1), vs2, vm  # 8b unsigned",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_indexed_instructions"
            };

        case "VLXHU.V":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vlxhu.v vd, (rs1), vs2, vm  # 16b unsigned</div>\n</div>",
                "tooltip": "vlxhu.v vd, (rs1), vs2, vm  # 16b unsigned",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_indexed_instructions"
            };

        case "VLXWU.V":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vlxwu.v vd, (rs1), vs2, vm  # 32b unsigned</div>\n</div>",
                "tooltip": "vlxwu.v vd, (rs1), vs2, vm  # 32b unsigned",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_indexed_instructions"
            };

        case "VSXB.V":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vs3</span>\n\n<div>vsxb.v vs3, (rs1), vs2, vm  # 8b</div>\n</div>",
                "tooltip": "vsxb.v vs3, (rs1), vs2, vm  # 8b",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_indexed_instructions"
            };

        case "VSXH.V":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vs3</span>\n\n<div>vsxh.v vs3, (rs1), vs2, vm  # 16b</div>\n</div>",
                "tooltip": "vsxh.v vs3, (rs1), vs2, vm  # 16b",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_indexed_instructions"
            };

        case "VSXW.V":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vs3</span>\n\n<div>vsxw.v vs3, (rs1), vs2, vm  # 32b</div>\n</div>",
                "tooltip": "vsxw.v vs3, (rs1), vs2, vm  # 32b",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_indexed_instructions"
            };

        case "VSXE.V":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vs3</span>\n\n<div>vsxe.v vs3, (rs1), vs2, vm  # SEW</div>\n</div>",
                "tooltip": "vsxe.v vs3, (rs1), vs2, vm  # SEW",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_indexed_instructions"
            };

        case "VSUXB.V":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vs3</span>\n\n<div>vsuxb.v vs3, (rs1), vs2, vm  # 8b</div>\n</div>",
                "tooltip": "vsuxb.v vs3, (rs1), vs2, vm  # 8b",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_indexed_instructions"
            };

        case "VSUXH.V":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vs3</span>\n\n<div>vsuxh.v vs3, (rs1), vs2, vm  # 16b</div>\n</div>",
                "tooltip": "vsuxh.v vs3, (rs1), vs2, vm  # 16b",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_indexed_instructions"
            };

        case "VSUXW.V":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vs3</span>\n\n<div>vsuxw.v vs3, (rs1), vs2, vm  # 32b</div>\n</div>",
                "tooltip": "vsuxw.v vs3, (rs1), vs2, vm  # 32b",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_indexed_instructions"
            };

        case "VSUXE.V":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vs3</span>\n\n<div>vsuxe.v vs3, (rs1), vs2, vm  # SEW</div>\n</div>",
                "tooltip": "vsuxe.v vs3, (rs1), vs2, vm  # SEW",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_indexed_instructions"
            };

        case "VLBFF.V":
            return {
                "html": "<div>\n<span class=\"opcode\">rs1, vd</span>\n\n<div>vlbff.v  vd, (rs1), vm # 8b</div>\n<div>vlbff.v v1, (a3)      # Load bytes</div>\n</div>",
                "tooltip": "vlbff.v  vd, (rs1), vm # 8b\nvlbff.v v1, (a3)      # Load bytes",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_unit_stride_fault_only_first_loads"
            };

        case "VLHFF.V":
            return {
                "html": "<div>\n<span class=\"opcode\">rs1, vd</span>\n\n<div>vlhff.v  vd, (rs1), vm # 16b</div>\n</div>",
                "tooltip": "vlhff.v  vd, (rs1), vm # 16b",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_unit_stride_fault_only_first_loads"
            };

        case "VLWFF.V":
            return {
                "html": "<div>\n<span class=\"opcode\">rs1, vd</span>\n\n<div>vlwff.v  vd, (rs1), vm # 32b</div>\n</div>",
                "tooltip": "vlwff.v  vd, (rs1), vm # 32b",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_unit_stride_fault_only_first_loads"
            };

        case "VLEFF.V":
            return {
                "html": "<div>\n<span class=\"opcode\">rs1, vd</span>\n\n<div>vleff.v  vd, (rs1), vm # SEW</div>\n</div>",
                "tooltip": "vleff.v  vd, (rs1), vm # SEW",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_unit_stride_fault_only_first_loads"
            };

        case "VLBUFF.V":
            return {
                "html": "<div>\n<span class=\"opcode\">rs1, vd</span>\n\n<div>vlbuff.v vd, (rs1), vm # unsigned 8b</div>\n</div>",
                "tooltip": "vlbuff.v vd, (rs1), vm # unsigned 8b",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_unit_stride_fault_only_first_loads"
            };

        case "VLHUFF.V":
            return {
                "html": "<div>\n<span class=\"opcode\">rs1, vd</span>\n\n<div>vlhuff.v vd, (rs1), vm # unsigned 16b</div>\n</div>",
                "tooltip": "vlhuff.v vd, (rs1), vm # unsigned 16b",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_unit_stride_fault_only_first_loads"
            };

        case "VLWUFF.V":
            return {
                "html": "<div>\n<span class=\"opcode\">rs1, vd</span>\n\n<div>vlwuff.v vd, (rs1), vm # unsigned 32b</div>\n</div>",
                "tooltip": "vlwuff.v vd, (rs1), vm # unsigned 32b",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_unit_stride_fault_only_first_loads"
            };

        case "VFADD.VF":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vfadd.vv vd, vs2, vs1, vm   # Vector-vector</div>\n<div>vfadd.vf vd, vs2, rs1, vm   # vector-scalar</div>\n</div>",
                "tooltip": "vfadd.vv vd, vs2, vs1, vm   # Vector-vector\nvfadd.vf vd, vs2, rs1, vm   # vector-scalar",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_floating_point_add_subtract_instructions"
            };

        case "VFSUB.VF":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vfsub.vv vd, vs2, vs1, vm   # Vector-vector</div>\n<div>vfsub.vf vd, vs2, rs1, vm   # Vector-scalar vd[i] = vs2[i] - f[rs1]</div>\n</div>",
                "tooltip": "vfsub.vv vd, vs2, vs1, vm   # Vector-vector\nvfsub.vf vd, vs2, rs1, vm   # Vector-scalar vd[i] = vs2[i] - f[rs1]",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_floating_point_add_subtract_instructions"
            };

        case "VFADD.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n<div>vfadd.vv vd, vs2, vs1, vm   # Vector-vector</div>\n<div>vfadd.vf vd, vs2, rs1, vm   # vector-scalar</div>\n</div>",
                "tooltip": "vfadd.vv vd, vs2, vs1, vm   # Vector-vector\nvfadd.vf vd, vs2, rs1, vm   # vector-scalar",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_floating_point_add_subtract_instructions"
            };

        case "VFSUB.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n<div>vfsub.vv vd, vs2, vs1, vm   # Vector-vector</div>\n<div>vfsub.vf vd, vs2, rs1, vm   # Vector-scalar vd[i] = vs2[i] - f[rs1]</div>\n</div>",
                "tooltip": "vfsub.vv vd, vs2, vs1, vm   # Vector-vector\nvfsub.vf vd, vs2, rs1, vm   # Vector-scalar vd[i] = vs2[i] - f[rs1]",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_floating_point_add_subtract_instructions"
            };

        case "VFMIN.VF":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>The vector floating-point vfmin and vfmax instructions have the same behavior as the corresponding scalar floating-point instructions in version 2.2 of the RISC-V F/D/Q extension.</div>\n<div>vfmin.vv vd, vs2, vs1, vm   # Vector-vector</div>\n<div>vfmin.vf vd, vs2, rs1, vm   # vector-scalar</div>\n</div>",
                "tooltip": "The vector floating-point vfmin and vfmax instructions have the same behavior as the corresponding scalar floating-point instructions in version 2.2 of the RISC-V F/D/Q extension.\nvfmin.vv vd, vs2, vs1, vm   # Vector-vector\nvfmin.vf vd, vs2, rs1, vm   # vector-scalar",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_floating_point_min_max_instructions"
            };

        case "VFMAX.VF":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vfmax.vv vd, vs2, vs1, vm   # Vector-vector</div>\n<div>vfmax.vf vd, vs2, rs1, vm   # vector-scalar</div>\n</div>",
                "tooltip": "vfmax.vv vd, vs2, vs1, vm   # Vector-vector\nvfmax.vf vd, vs2, rs1, vm   # vector-scalar",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_floating_point_min_max_instructions"
            };

        case "VFMIN.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n<div>The vector floating-point vfmin and vfmax instructions have the same behavior as the corresponding scalar floating-point instructions in version 2.2 of the RISC-V F/D/Q extension.</div>\n<div>vfmin.vv vd, vs2, vs1, vm   # Vector-vector</div>\n<div>vfmin.vf vd, vs2, rs1, vm   # vector-scalar</div>\n</div>",
                "tooltip": "The vector floating-point vfmin and vfmax instructions have the same behavior as the corresponding scalar floating-point instructions in version 2.2 of the RISC-V F/D/Q extension.\nvfmin.vv vd, vs2, vs1, vm   # Vector-vector\nvfmin.vf vd, vs2, rs1, vm   # vector-scalar",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_floating_point_min_max_instructions"
            };

        case "VFMAX.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n<div>vfmax.vv vd, vs2, vs1, vm   # Vector-vector</div>\n<div>vfmax.vf vd, vs2, rs1, vm   # vector-scalar</div>\n</div>",
                "tooltip": "vfmax.vv vd, vs2, vs1, vm   # Vector-vector\nvfmax.vf vd, vs2, rs1, vm   # vector-scalar",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_floating_point_min_max_instructions"
            };

        case "VFSGNJ.VF":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vfsgnj.vv vd, vs2, vs1, vm   # Vector-vector</div>\n<div>vfsgnj.vf vd, vs2, rs1, vm   # vector-scalar</div>\n</div>",
                "tooltip": "vfsgnj.vv vd, vs2, vs1, vm   # Vector-vector\nvfsgnj.vf vd, vs2, rs1, vm   # vector-scalar",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_floating_point_sign_injection_instructions"
            };

        case "VFSGNJN.VF":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vfsgnjn.vv vd, vs2, vs1, vm   # Vector-vector</div>\n<div>vfsgnjn.vf vd, vs2, rs1, vm   # vector-scalar</div>\n</div>",
                "tooltip": "vfsgnjn.vv vd, vs2, vs1, vm   # Vector-vector\nvfsgnjn.vf vd, vs2, rs1, vm   # vector-scalar",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_floating_point_sign_injection_instructions"
            };

        case "VFSGNJX.VF":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vfsgnjx.vv vd, vs2, vs1, vm   # Vector-vector</div>\n<div>vfsgnjx.vf vd, vs2, rs1, vm   # vector-scalar</div>\n</div>",
                "tooltip": "vfsgnjx.vv vd, vs2, vs1, vm   # Vector-vector\nvfsgnjx.vf vd, vs2, rs1, vm   # vector-scalar",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_floating_point_sign_injection_instructions"
            };

        case "VFSGNJ.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n<div>vfsgnj.vv vd, vs2, vs1, vm   # Vector-vector</div>\n<div>vfsgnj.vf vd, vs2, rs1, vm   # vector-scalar</div>\n</div>",
                "tooltip": "vfsgnj.vv vd, vs2, vs1, vm   # Vector-vector\nvfsgnj.vf vd, vs2, rs1, vm   # vector-scalar",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_floating_point_sign_injection_instructions"
            };

        case "VFSGNJN.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n<div>vfsgnjn.vv vd, vs2, vs1, vm   # Vector-vector</div>\n<div>vfsgnjn.vf vd, vs2, rs1, vm   # vector-scalar</div>\n</div>",
                "tooltip": "vfsgnjn.vv vd, vs2, vs1, vm   # Vector-vector\nvfsgnjn.vf vd, vs2, rs1, vm   # vector-scalar",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_floating_point_sign_injection_instructions"
            };

        case "VFSGNJX.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n<div>vfsgnjx.vv vd, vs2, vs1, vm   # Vector-vector</div>\n<div>vfsgnjx.vf vd, vs2, rs1, vm   # vector-scalar</div>\n</div>",
                "tooltip": "vfsgnjx.vv vd, vs2, vs1, vm   # Vector-vector\nvfsgnjx.vf vd, vs2, rs1, vm   # vector-scalar",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_floating_point_sign_injection_instructions"
            };

        case "VFMV.S.F":
            return {
                "html": "<div>\n<span class=\"opcode\">rs1, vd</span>\n\n<div>Note vfmv.v.f instruction shares the encoding with the vfmerge.vfm vm=1 and vs2=v0</div>\n<div>Note vfmv.v.f substitutes a canonical NaN for f[rs1] if the latter is not properly NaN-boxed</div>\n<div>vfmv.v.f vd, rs1  # vd[i] = f[rs1]</div>\n</div>",
                "tooltip": "Note vfmv.v.f instruction shares the encoding with the vfmerge.vfm vm=1 and vs2=v0\nNote vfmv.v.f substitutes a canonical NaN for f[rs1] if the latter is not properly NaN-boxed\nvfmv.v.f vd, rs1  # vd[i] = f[rs1]",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_floating_point_move_instruction"
            };

        case "VFMV.V.F":
            return {
                "html": "<div>\n<span class=\"opcode\">rs1, vd</span>\n\n<div>Note vfmv.v.f instruction shares the encoding with the vfmerge.vfm vm=1 and vs2=v0</div>\n<div>Note vfmv.v.f substitutes a canonical NaN for f[rs1] if the latter is not properly NaN-boxed</div>\n<div>vfmv.v.f vd, rs1  # vd[i] = f[rs1]</div>\n</div>",
                "tooltip": "Note vfmv.v.f instruction shares the encoding with the vfmerge.vfm vm=1 and vs2=v0\nNote vfmv.v.f substitutes a canonical NaN for f[rs1] if the latter is not properly NaN-boxed\nvfmv.v.f vd, rs1  # vd[i] = f[rs1]",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_floating_point_move_instruction"
            };

        case "VFMV.F.S":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rd</span>\n\n<div>Note vfmv.v.f instruction shares the encoding with the vfmerge.vfm vm=1 and vs2=v0</div>\n<div>Note vfmv.v.f substitutes a canonical NaN for f[rs1] if the latter is not properly NaN-boxed</div>\n<div>vfmv.v.f vd, rs1  # vd[i] = f[rs1]</div>\n</div>",
                "tooltip": "Note vfmv.v.f instruction shares the encoding with the vfmerge.vfm vm=1 and vs2=v0\nNote vfmv.v.f substitutes a canonical NaN for f[rs1] if the latter is not properly NaN-boxed\nvfmv.v.f vd, rs1  # vd[i] = f[rs1]",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_floating_point_move_instruction"
            };

        case "VFMERGE.VFM":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>The vfmerge.vfm instruction is always masked ( vm=0 )</div>\n<div>Note vfmerge.vfm substitutes a canonical NaN for f[rs1] if the latter is not properly NaN-boxed</div>\n<div>vfmerge.vfm vd, vs2, rs1, v0  # vd[i] = v0[i].LSB ? f[rs1] : vs2[i]</div>\n</div>",
                "tooltip": "The vfmerge.vfm instruction is always masked ( vm=0 )\nNote vfmerge.vfm substitutes a canonical NaN for f[rs1] if the latter is not properly NaN-boxed\nvfmerge.vfm vd, vs2, rs1, v0  # vd[i] = v0[i].LSB ? f[rs1] : vs2[i]",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_floating_point_merge_instruction"
            };

        case "VFEQ.VF":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_introduction"
            };

        case "VFLE.VF":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_introduction"
            };

        case "VFORD.VF":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_introduction"
            };

        case "VFLT.VF":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_introduction"
            };

        case "VFNE.VF":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_introduction"
            };

        case "VFGT.VF":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_introduction"
            };

        case "VFGE.VF":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_introduction"
            };

        case "VFEQ.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_introduction"
            };

        case "VFLE.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_introduction"
            };

        case "VFORD.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_introduction"
            };

        case "VFLT.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_introduction"
            };

        case "VFNE.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_introduction"
            };

        case "VFUNARY0.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_introduction"
            };

        case "VFUNARY1.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_introduction"
            };

        case "VSEQ.VX":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_introduction"
            };

        case "VSNE.VX":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_introduction"
            };

        case "VSLTU.VX":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_introduction"
            };

        case "VSLT.VX":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_introduction"
            };

        case "VSLEU.VX":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_introduction"
            };

        case "VSLE.VX":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_introduction"
            };

        case "VSGTU.VX":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_introduction"
            };

        case "VSGT.VX":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_introduction"
            };

        case "VWSMACCU.VX":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_introduction"
            };

        case "VWSMACC.VX":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_introduction"
            };

        case "VWSMACCSU.VX":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_introduction"
            };

        case "VWSMACCUS.VX":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_introduction"
            };

        case "VSEQ.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_introduction"
            };

        case "VSNE.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_introduction"
            };

        case "VSLTU.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_introduction"
            };

        case "VSLT.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_introduction"
            };

        case "VSLEU.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_introduction"
            };

        case "VSLE.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_introduction"
            };

        case "VWSMACCU.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_introduction"
            };

        case "VWSMACC.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_introduction"
            };

        case "VWSMACCSU.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_introduction"
            };

        case "VSEQ.VI":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, simm5, vd</span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_introduction"
            };

        case "VSNE.VI":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, simm5, vd</span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_introduction"
            };

        case "VSLEU.VI":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, simm5, vd</span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_introduction"
            };

        case "VSLE.VI":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, simm5, vd</span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_introduction"
            };

        case "VSGTU.VI":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, simm5, vd</span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_introduction"
            };

        case "VSGT.VI":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, simm5, vd</span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_introduction"
            };

        case "VEXT.X.V":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_introduction"
            };

        case "VMPOPC.M":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, rd</span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_introduction"
            };

        case "VMFIRST.M":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, rd</span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_introduction"
            };

        case "VFDIV.VF":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vfdiv.vv vd, vs2, vs1, vm   # Vector-vector</div>\n<div>vfdiv.vf vd, vs2, rs1, vm   # vector-scalar</div>\n</div>",
                "tooltip": "vfdiv.vv vd, vs2, vs1, vm   # Vector-vector\nvfdiv.vf vd, vs2, rs1, vm   # vector-scalar",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_floating_point_multiply_divide_instructions"
            };

        case "VFRDIV.VF":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vfrdiv.vf vd, vs2, rs1, vm  # scalar-vector, vd[i] = f[rs1]/vs2[i]</div>\n</div>",
                "tooltip": "vfrdiv.vf vd, vs2, rs1, vm  # scalar-vector, vd[i] = f[rs1]/vs2[i]",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_floating_point_multiply_divide_instructions"
            };

        case "VFMUL.VF":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vfmul.vv vd, vs2, vs1, vm   # Vector-vector</div>\n<div>vfmul.vf vd, vs2, rs1, vm   # vector-scalar</div>\n</div>",
                "tooltip": "vfmul.vv vd, vs2, vs1, vm   # Vector-vector\nvfmul.vf vd, vs2, rs1, vm   # vector-scalar",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_floating_point_multiply_divide_instructions"
            };

        case "VFDIV.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n<div>vfdiv.vv vd, vs2, vs1, vm   # Vector-vector</div>\n<div>vfdiv.vf vd, vs2, rs1, vm   # vector-scalar</div>\n</div>",
                "tooltip": "vfdiv.vv vd, vs2, vs1, vm   # Vector-vector\nvfdiv.vf vd, vs2, rs1, vm   # vector-scalar",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_floating_point_multiply_divide_instructions"
            };

        case "VFMUL.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n<div>vfmul.vv vd, vs2, vs1, vm   # Vector-vector</div>\n<div>vfmul.vf vd, vs2, rs1, vm   # vector-scalar</div>\n</div>",
                "tooltip": "vfmul.vv vd, vs2, vs1, vm   # Vector-vector\nvfmul.vf vd, vs2, rs1, vm   # vector-scalar",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_floating_point_multiply_divide_instructions"
            };

        case "VFMADD.VF":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vfmadd.vv vd, vs1, vs2, vm    # vd[i] = +(vs1[i] * vd[i]) + vs2[i]</div>\n<div>vfmadd.vf vd, rs1, vs2, vm    # vd[i] = +(f[rs1] * vd[i]) + vs2[i]</div>\n</div>",
                "tooltip": "vfmadd.vv vd, vs1, vs2, vm    # vd[i] = +(vs1[i] * vd[i]) + vs2[i]\nvfmadd.vf vd, rs1, vs2, vm    # vd[i] = +(f[rs1] * vd[i]) + vs2[i]",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_floating_point_fused_multiply_add_instructions"
            };

        case "VFNMADD.VF":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vfnmadd.vv vd, vs1, vs2, vm   # vd[i] = -(vs1[i] * vd[i]) - vs2[i]</div>\n<div>vfnmadd.vf vd, rs1, vs2, vm   # vd[i] = -(f[rs1] * vd[i]) - vs2[i]</div>\n</div>",
                "tooltip": "vfnmadd.vv vd, vs1, vs2, vm   # vd[i] = -(vs1[i] * vd[i]) - vs2[i]\nvfnmadd.vf vd, rs1, vs2, vm   # vd[i] = -(f[rs1] * vd[i]) - vs2[i]",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_floating_point_fused_multiply_add_instructions"
            };

        case "VFMSUB.VF":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vfmsub.vv vd, vs1, vs2, vm    # vd[i] = +(vs1[i] * vd[i]) - vs2[i]</div>\n<div>vfmsub.vf vd, rs1, vs2, vm    # vd[i] = +(f[rs1] * vd[i]) - vs2[i]</div>\n</div>",
                "tooltip": "vfmsub.vv vd, vs1, vs2, vm    # vd[i] = +(vs1[i] * vd[i]) - vs2[i]\nvfmsub.vf vd, rs1, vs2, vm    # vd[i] = +(f[rs1] * vd[i]) - vs2[i]",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_floating_point_fused_multiply_add_instructions"
            };

        case "VFNMSUB.VF":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vfnmsub.vv vd, vs1, vs2, vm   # vd[i] = -(vs1[i] * vd[i]) + vs2[i]</div>\n<div>vfnmsub.vf vd, rs1, vs2, vm   # vd[i] = -(f[rs1] * vd[i]) + vs2[i]</div>\n</div>",
                "tooltip": "vfnmsub.vv vd, vs1, vs2, vm   # vd[i] = -(vs1[i] * vd[i]) + vs2[i]\nvfnmsub.vf vd, rs1, vs2, vm   # vd[i] = -(f[rs1] * vd[i]) + vs2[i]",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_floating_point_fused_multiply_add_instructions"
            };

        case "VFMACC.VF":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vfmacc.vv vd, vs1, vs2, vm    # vd[i] = +(vs1[i] * vs2[i]) + vd[i]</div>\n<div>vfmacc.vf vd, rs1, vs2, vm    # vd[i] = +(f[rs1] * vs2[i]) + vd[i]</div>\n</div>",
                "tooltip": "vfmacc.vv vd, vs1, vs2, vm    # vd[i] = +(vs1[i] * vs2[i]) + vd[i]\nvfmacc.vf vd, rs1, vs2, vm    # vd[i] = +(f[rs1] * vs2[i]) + vd[i]",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_floating_point_fused_multiply_add_instructions"
            };

        case "VFNMACC.VF":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vfnmacc.vv vd, vs1, vs2, vm   # vd[i] = -(vs1[i] * vs2[i]) - vd[i]</div>\n<div>vfnmacc.vf vd, rs1, vs2, vm   # vd[i] = -(f[rs1] * vs2[i]) - vd[i]</div>\n</div>",
                "tooltip": "vfnmacc.vv vd, vs1, vs2, vm   # vd[i] = -(vs1[i] * vs2[i]) - vd[i]\nvfnmacc.vf vd, rs1, vs2, vm   # vd[i] = -(f[rs1] * vs2[i]) - vd[i]",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_floating_point_fused_multiply_add_instructions"
            };

        case "VFMSAC.VF":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vfmsac.vv vd, vs1, vs2, vm    # vd[i] = +(vs1[i] * vs2[i]) - vd[i]</div>\n<div>vfmsac.vf vd, rs1, vs2, vm    # vd[i] = +(f[rs1] * vs2[i]) - vd[i]</div>\n</div>",
                "tooltip": "vfmsac.vv vd, vs1, vs2, vm    # vd[i] = +(vs1[i] * vs2[i]) - vd[i]\nvfmsac.vf vd, rs1, vs2, vm    # vd[i] = +(f[rs1] * vs2[i]) - vd[i]",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_floating_point_fused_multiply_add_instructions"
            };

        case "VFNMSAC.VF":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vfnmsac.vv vd, vs1, vs2, vm   # vd[i] = -(vs1[i] * vs2[i]) + vd[i]</div>\n<div>vfnmsac.vf vd, rs1, vs2, vm   # vd[i] = -(f[rs1] * vs2[i]) + vd[i]</div>\n</div>",
                "tooltip": "vfnmsac.vv vd, vs1, vs2, vm   # vd[i] = -(vs1[i] * vs2[i]) + vd[i]\nvfnmsac.vf vd, rs1, vs2, vm   # vd[i] = -(f[rs1] * vs2[i]) + vd[i]",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_floating_point_fused_multiply_add_instructions"
            };

        case "VFMADD.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n<div>vfmadd.vv vd, vs1, vs2, vm    # vd[i] = +(vs1[i] * vd[i]) + vs2[i]</div>\n<div>vfmadd.vf vd, rs1, vs2, vm    # vd[i] = +(f[rs1] * vd[i]) + vs2[i]</div>\n</div>",
                "tooltip": "vfmadd.vv vd, vs1, vs2, vm    # vd[i] = +(vs1[i] * vd[i]) + vs2[i]\nvfmadd.vf vd, rs1, vs2, vm    # vd[i] = +(f[rs1] * vd[i]) + vs2[i]",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_floating_point_fused_multiply_add_instructions"
            };

        case "VFNMADD.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n<div>vfnmadd.vv vd, vs1, vs2, vm   # vd[i] = -(vs1[i] * vd[i]) - vs2[i]</div>\n<div>vfnmadd.vf vd, rs1, vs2, vm   # vd[i] = -(f[rs1] * vd[i]) - vs2[i]</div>\n</div>",
                "tooltip": "vfnmadd.vv vd, vs1, vs2, vm   # vd[i] = -(vs1[i] * vd[i]) - vs2[i]\nvfnmadd.vf vd, rs1, vs2, vm   # vd[i] = -(f[rs1] * vd[i]) - vs2[i]",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_floating_point_fused_multiply_add_instructions"
            };

        case "VFMSUB.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n<div>vfmsub.vv vd, vs1, vs2, vm    # vd[i] = +(vs1[i] * vd[i]) - vs2[i]</div>\n<div>vfmsub.vf vd, rs1, vs2, vm    # vd[i] = +(f[rs1] * vd[i]) - vs2[i]</div>\n</div>",
                "tooltip": "vfmsub.vv vd, vs1, vs2, vm    # vd[i] = +(vs1[i] * vd[i]) - vs2[i]\nvfmsub.vf vd, rs1, vs2, vm    # vd[i] = +(f[rs1] * vd[i]) - vs2[i]",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_floating_point_fused_multiply_add_instructions"
            };

        case "VFNMSUB.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n<div>vfnmsub.vv vd, vs1, vs2, vm   # vd[i] = -(vs1[i] * vd[i]) + vs2[i]</div>\n<div>vfnmsub.vf vd, rs1, vs2, vm   # vd[i] = -(f[rs1] * vd[i]) + vs2[i]</div>\n</div>",
                "tooltip": "vfnmsub.vv vd, vs1, vs2, vm   # vd[i] = -(vs1[i] * vd[i]) + vs2[i]\nvfnmsub.vf vd, rs1, vs2, vm   # vd[i] = -(f[rs1] * vd[i]) + vs2[i]",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_floating_point_fused_multiply_add_instructions"
            };

        case "VFMACC.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n<div>vfmacc.vv vd, vs1, vs2, vm    # vd[i] = +(vs1[i] * vs2[i]) + vd[i]</div>\n<div>vfmacc.vf vd, rs1, vs2, vm    # vd[i] = +(f[rs1] * vs2[i]) + vd[i]</div>\n</div>",
                "tooltip": "vfmacc.vv vd, vs1, vs2, vm    # vd[i] = +(vs1[i] * vs2[i]) + vd[i]\nvfmacc.vf vd, rs1, vs2, vm    # vd[i] = +(f[rs1] * vs2[i]) + vd[i]",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_floating_point_fused_multiply_add_instructions"
            };

        case "VFNMACC.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n<div>vfnmacc.vv vd, vs1, vs2, vm   # vd[i] = -(vs1[i] * vs2[i]) - vd[i]</div>\n<div>vfnmacc.vf vd, rs1, vs2, vm   # vd[i] = -(f[rs1] * vs2[i]) - vd[i]</div>\n</div>",
                "tooltip": "vfnmacc.vv vd, vs1, vs2, vm   # vd[i] = -(vs1[i] * vs2[i]) - vd[i]\nvfnmacc.vf vd, rs1, vs2, vm   # vd[i] = -(f[rs1] * vs2[i]) - vd[i]",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_floating_point_fused_multiply_add_instructions"
            };

        case "VFMSAC.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n<div>vfmsac.vv vd, vs1, vs2, vm    # vd[i] = +(vs1[i] * vs2[i]) - vd[i]</div>\n<div>vfmsac.vf vd, rs1, vs2, vm    # vd[i] = +(f[rs1] * vs2[i]) - vd[i]</div>\n</div>",
                "tooltip": "vfmsac.vv vd, vs1, vs2, vm    # vd[i] = +(vs1[i] * vs2[i]) - vd[i]\nvfmsac.vf vd, rs1, vs2, vm    # vd[i] = +(f[rs1] * vs2[i]) - vd[i]",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_floating_point_fused_multiply_add_instructions"
            };

        case "VFNMSAC.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n<div>vfnmsac.vv vd, vs1, vs2, vm   # vd[i] = -(vs1[i] * vs2[i]) + vd[i]</div>\n<div>vfnmsac.vf vd, rs1, vs2, vm   # vd[i] = -(f[rs1] * vs2[i]) + vd[i]</div>\n</div>",
                "tooltip": "vfnmsac.vv vd, vs1, vs2, vm   # vd[i] = -(vs1[i] * vs2[i]) + vd[i]\nvfnmsac.vf vd, rs1, vs2, vm   # vd[i] = -(f[rs1] * vs2[i]) + vd[i]",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_floating_point_fused_multiply_add_instructions"
            };

        case "VFWADD.VF":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vfwadd.vv vd, vs2, vs1, vm  # vector-vector</div>\n<div>vfwadd.vf vd, vs2, rs1, vm  # vector-scalar</div>\n<div>vfwadd.wv  vd, vs2, vs1, vm  # vector-vector</div>\n<div>vfwadd.wf  vd, vs2, rs1, vm  # vector-scalar</div>\n</div>",
                "tooltip": "vfwadd.vv vd, vs2, vs1, vm  # vector-vector\nvfwadd.vf vd, vs2, rs1, vm  # vector-scalar\nvfwadd.wv  vd, vs2, vs1, vm  # vector-vector\nvfwadd.wf  vd, vs2, rs1, vm  # vector-scalar",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_widening_floating_point_add_subtract_instructions"
            };

        case "VFWSUB.VF":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vfwsub.vv vd, vs2, vs1, vm  # vector-vector</div>\n<div>vfwsub.vf vd, vs2, rs1, vm  # vector-scalar</div>\n<div>vfwsub.wv  vd, vs2, vs1, vm  # vector-vector</div>\n<div>vfwsub.wf  vd, vs2, rs1, vm  # vector-scalar</div>\n</div>",
                "tooltip": "vfwsub.vv vd, vs2, vs1, vm  # vector-vector\nvfwsub.vf vd, vs2, rs1, vm  # vector-scalar\nvfwsub.wv  vd, vs2, vs1, vm  # vector-vector\nvfwsub.wf  vd, vs2, rs1, vm  # vector-scalar",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_widening_floating_point_add_subtract_instructions"
            };

        case "VFWADD.WF":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vfwadd.vv vd, vs2, vs1, vm  # vector-vector</div>\n<div>vfwadd.vf vd, vs2, rs1, vm  # vector-scalar</div>\n<div>vfwadd.wv  vd, vs2, vs1, vm  # vector-vector</div>\n<div>vfwadd.wf  vd, vs2, rs1, vm  # vector-scalar</div>\n</div>",
                "tooltip": "vfwadd.vv vd, vs2, vs1, vm  # vector-vector\nvfwadd.vf vd, vs2, rs1, vm  # vector-scalar\nvfwadd.wv  vd, vs2, vs1, vm  # vector-vector\nvfwadd.wf  vd, vs2, rs1, vm  # vector-scalar",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_widening_floating_point_add_subtract_instructions"
            };

        case "VFWSUB.WF":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vfwsub.vv vd, vs2, vs1, vm  # vector-vector</div>\n<div>vfwsub.vf vd, vs2, rs1, vm  # vector-scalar</div>\n<div>vfwsub.wv  vd, vs2, vs1, vm  # vector-vector</div>\n<div>vfwsub.wf  vd, vs2, rs1, vm  # vector-scalar</div>\n</div>",
                "tooltip": "vfwsub.vv vd, vs2, vs1, vm  # vector-vector\nvfwsub.vf vd, vs2, rs1, vm  # vector-scalar\nvfwsub.wv  vd, vs2, vs1, vm  # vector-vector\nvfwsub.wf  vd, vs2, rs1, vm  # vector-scalar",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_widening_floating_point_add_subtract_instructions"
            };

        case "VFWADD.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n<div>vfwadd.vv vd, vs2, vs1, vm  # vector-vector</div>\n<div>vfwadd.vf vd, vs2, rs1, vm  # vector-scalar</div>\n<div>vfwadd.wv  vd, vs2, vs1, vm  # vector-vector</div>\n<div>vfwadd.wf  vd, vs2, rs1, vm  # vector-scalar</div>\n</div>",
                "tooltip": "vfwadd.vv vd, vs2, vs1, vm  # vector-vector\nvfwadd.vf vd, vs2, rs1, vm  # vector-scalar\nvfwadd.wv  vd, vs2, vs1, vm  # vector-vector\nvfwadd.wf  vd, vs2, rs1, vm  # vector-scalar",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_widening_floating_point_add_subtract_instructions"
            };

        case "VFWSUB.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n<div>vfwsub.vv vd, vs2, vs1, vm  # vector-vector</div>\n<div>vfwsub.vf vd, vs2, rs1, vm  # vector-scalar</div>\n<div>vfwsub.wv  vd, vs2, vs1, vm  # vector-vector</div>\n<div>vfwsub.wf  vd, vs2, rs1, vm  # vector-scalar</div>\n</div>",
                "tooltip": "vfwsub.vv vd, vs2, vs1, vm  # vector-vector\nvfwsub.vf vd, vs2, rs1, vm  # vector-scalar\nvfwsub.wv  vd, vs2, vs1, vm  # vector-vector\nvfwsub.wf  vd, vs2, rs1, vm  # vector-scalar",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_widening_floating_point_add_subtract_instructions"
            };

        case "VFWADD.WV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n<div>vfwadd.vv vd, vs2, vs1, vm  # vector-vector</div>\n<div>vfwadd.vf vd, vs2, rs1, vm  # vector-scalar</div>\n<div>vfwadd.wv  vd, vs2, vs1, vm  # vector-vector</div>\n<div>vfwadd.wf  vd, vs2, rs1, vm  # vector-scalar</div>\n</div>",
                "tooltip": "vfwadd.vv vd, vs2, vs1, vm  # vector-vector\nvfwadd.vf vd, vs2, rs1, vm  # vector-scalar\nvfwadd.wv  vd, vs2, vs1, vm  # vector-vector\nvfwadd.wf  vd, vs2, rs1, vm  # vector-scalar",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_widening_floating_point_add_subtract_instructions"
            };

        case "VFWSUB.WV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n<div>vfwsub.vv vd, vs2, vs1, vm  # vector-vector</div>\n<div>vfwsub.vf vd, vs2, rs1, vm  # vector-scalar</div>\n<div>vfwsub.wv  vd, vs2, vs1, vm  # vector-vector</div>\n<div>vfwsub.wf  vd, vs2, rs1, vm  # vector-scalar</div>\n</div>",
                "tooltip": "vfwsub.vv vd, vs2, vs1, vm  # vector-vector\nvfwsub.vf vd, vs2, rs1, vm  # vector-scalar\nvfwsub.wv  vd, vs2, vs1, vm  # vector-vector\nvfwsub.wf  vd, vs2, rs1, vm  # vector-scalar",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_widening_floating_point_add_subtract_instructions"
            };

        case "VFWMUL.VF":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vfwmul.vv    vd, vs2, vs1, vm # vector-vector</div>\n<div>vfwmul.vf    vd, vs2, rs1, vm # vector-scalar</div>\n</div>",
                "tooltip": "vfwmul.vv    vd, vs2, vs1, vm # vector-vector\nvfwmul.vf    vd, vs2, rs1, vm # vector-scalar",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_widening_floating_point_multiply"
            };

        case "VFWMUL.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n<div>vfwmul.vv    vd, vs2, vs1, vm # vector-vector</div>\n<div>vfwmul.vf    vd, vs2, rs1, vm # vector-scalar</div>\n</div>",
                "tooltip": "vfwmul.vv    vd, vs2, vs1, vm # vector-vector\nvfwmul.vf    vd, vs2, rs1, vm # vector-scalar",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_widening_floating_point_multiply"
            };

        case "VFWMACC.VF":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vfwmacc.vv vd, vs1, vs2, vm    # vd[i] = +(vs1[i] * vs2[i]) + vd[i]</div>\n<div>vfwmacc.vf vd, rs1, vs2, vm    # vd[i] = +(f[rs1] * vs2[i]) + vd[i]</div>\n</div>",
                "tooltip": "vfwmacc.vv vd, vs1, vs2, vm    # vd[i] = +(vs1[i] * vs2[i]) + vd[i]\nvfwmacc.vf vd, rs1, vs2, vm    # vd[i] = +(f[rs1] * vs2[i]) + vd[i]",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_widening_floating_point_fused_multiply_add_instructions"
            };

        case "VFWNMACC.VF":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vfwnmacc.vv vd, vs1, vs2, vm   # vd[i] = -(vs1[i] * vs2[i]) - vd[i]</div>\n<div>vfwnmacc.vf vd, rs1, vs2, vm   # vd[i] = -(f[rs1] * vs2[i]) - vd[i]</div>\n</div>",
                "tooltip": "vfwnmacc.vv vd, vs1, vs2, vm   # vd[i] = -(vs1[i] * vs2[i]) - vd[i]\nvfwnmacc.vf vd, rs1, vs2, vm   # vd[i] = -(f[rs1] * vs2[i]) - vd[i]",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_widening_floating_point_fused_multiply_add_instructions"
            };

        case "VFWMSAC.VF":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vfwmsac.vv vd, vs1, vs2, vm    # vd[i] = +(vs1[i] * vs2[i]) - vd[i]</div>\n<div>vfwmsac.vf vd, rs1, vs2, vm    # vd[i] = +(f[rs1] * vs2[i]) - vd[i]</div>\n</div>",
                "tooltip": "vfwmsac.vv vd, vs1, vs2, vm    # vd[i] = +(vs1[i] * vs2[i]) - vd[i]\nvfwmsac.vf vd, rs1, vs2, vm    # vd[i] = +(f[rs1] * vs2[i]) - vd[i]",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_widening_floating_point_fused_multiply_add_instructions"
            };

        case "VFWNMSAC.VF":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vfwnmsac.vv vd, vs1, vs2, vm   # vd[i] = -(vs1[i] * vs2[i]) + vd[i]</div>\n<div>vfwnmsac.vf vd, rs1, vs2, vm   # vd[i] = -(f[rs1] * vs2[i]) + vd[i]</div>\n</div>",
                "tooltip": "vfwnmsac.vv vd, vs1, vs2, vm   # vd[i] = -(vs1[i] * vs2[i]) + vd[i]\nvfwnmsac.vf vd, rs1, vs2, vm   # vd[i] = -(f[rs1] * vs2[i]) + vd[i]",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_widening_floating_point_fused_multiply_add_instructions"
            };

        case "VFWMACC.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n<div>vfwmacc.vv vd, vs1, vs2, vm    # vd[i] = +(vs1[i] * vs2[i]) + vd[i]</div>\n<div>vfwmacc.vf vd, rs1, vs2, vm    # vd[i] = +(f[rs1] * vs2[i]) + vd[i]</div>\n</div>",
                "tooltip": "vfwmacc.vv vd, vs1, vs2, vm    # vd[i] = +(vs1[i] * vs2[i]) + vd[i]\nvfwmacc.vf vd, rs1, vs2, vm    # vd[i] = +(f[rs1] * vs2[i]) + vd[i]",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_widening_floating_point_fused_multiply_add_instructions"
            };

        case "VFWNMACC.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n<div>vfwnmacc.vv vd, vs1, vs2, vm   # vd[i] = -(vs1[i] * vs2[i]) - vd[i]</div>\n<div>vfwnmacc.vf vd, rs1, vs2, vm   # vd[i] = -(f[rs1] * vs2[i]) - vd[i]</div>\n</div>",
                "tooltip": "vfwnmacc.vv vd, vs1, vs2, vm   # vd[i] = -(vs1[i] * vs2[i]) - vd[i]\nvfwnmacc.vf vd, rs1, vs2, vm   # vd[i] = -(f[rs1] * vs2[i]) - vd[i]",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_widening_floating_point_fused_multiply_add_instructions"
            };

        case "VFWMSAC.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n<div>vfwmsac.vv vd, vs1, vs2, vm    # vd[i] = +(vs1[i] * vs2[i]) - vd[i]</div>\n<div>vfwmsac.vf vd, rs1, vs2, vm    # vd[i] = +(f[rs1] * vs2[i]) - vd[i]</div>\n</div>",
                "tooltip": "vfwmsac.vv vd, vs1, vs2, vm    # vd[i] = +(vs1[i] * vs2[i]) - vd[i]\nvfwmsac.vf vd, rs1, vs2, vm    # vd[i] = +(f[rs1] * vs2[i]) - vd[i]",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_widening_floating_point_fused_multiply_add_instructions"
            };

        case "VFWNMSAC.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n<div>vfwnmsac.vv vd, vs1, vs2, vm   # vd[i] = -(vs1[i] * vs2[i]) + vd[i]</div>\n<div>vfwnmsac.vf vd, rs1, vs2, vm   # vd[i] = -(f[rs1] * vs2[i]) + vd[i]</div>\n</div>",
                "tooltip": "vfwnmsac.vv vd, vs1, vs2, vm   # vd[i] = -(vs1[i] * vs2[i]) + vd[i]\nvfwnmsac.vf vd, rs1, vs2, vm   # vd[i] = -(f[rs1] * vs2[i]) + vd[i]",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_widening_floating_point_fused_multiply_add_instructions"
            };

        case "VFREDSUM.VS":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n<div>vfredsum.vs  vd, vs2, vs1, vm # Unordered sum</div>\n</div>",
                "tooltip": "vfredsum.vs  vd, vs2, vs1, vm # Unordered sum",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_floating_point_reduction_instructions"
            };

        case "VFREDOSUM.VS":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n<div>vfredosum.vs vd, vs2, vs1, vm # Ordered sum</div>\n</div>",
                "tooltip": "vfredosum.vs vd, vs2, vs1, vm # Ordered sum",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_floating_point_reduction_instructions"
            };

        case "VFREDMIN.VS":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n<div>vfredmin.vs  vd, vs2, vs1, vm # Minimum value</div>\n</div>",
                "tooltip": "vfredmin.vs  vd, vs2, vs1, vm # Minimum value",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_floating_point_reduction_instructions"
            };

        case "VFREDMAX.VS":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n<div>vfredmax.vs  vd, vs2, vs1, vm # Maximum value</div>\n</div>",
                "tooltip": "vfredmax.vs  vd, vs2, vs1, vm # Maximum value",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_floating_point_reduction_instructions"
            };

        case "VFWREDSUM.VS":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n<div>vfwredsum.vs vd, vs2, vs1, vm  # Unordered sum</div>\n</div>",
                "tooltip": "vfwredsum.vs vd, vs2, vs1, vm  # Unordered sum",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_widening_floating_point_reduction_instructions"
            };

        case "VFWREDOSUM.VS":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n<div>vfwredosum.vs vd, vs2, vs1, vm # Ordered sum</div>\n</div>",
                "tooltip": "vfwredosum.vs vd, vs2, vs1, vm # Ordered sum",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_widening_floating_point_reduction_instructions"
            };

        case "VFDOT.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n<div>The floating-point dot-product reduction vfdot.vv performs an element-wise multiplication between the source sub-elements then accumulates the results into the destination vector element</div>\n<div>vfdot.vv vd, vs2, vs1, vm   # Vector-vector</div>\n<div>vfdot.vv  vd, vs2, vs1, vm # vd[i][31:0] += vs2[i][31:16] * vs1[i][31:16]</div>\n<div>vfdot.vv v1, v2, v3   # v1[i][31:0] +=  v2[i][31:16]*v3[i][31:16] + v2[i][16:0]*v3[i][16:0]</div>\n<div>vfdot.vv v1, v2, v3</div>\n</div>",
                "tooltip": "The floating-point dot-product reduction vfdot.vv performs an element-wise multiplication between the source sub-elements then accumulates the results into the destination vector element\nvfdot.vv vd, vs2, vs1, vm   # Vector-vector\nvfdot.vv  vd, vs2, vs1, vm # vd[i][31:0] += vs2[i][31:16] * vs1[i][31:16]\nvfdot.vv v1, v2, v3   # v1[i][31:0] +=  v2[i][31:16]*v3[i][31:16] + v2[i][16:0]*v3[i][16:0]\nvfdot.vv v1, v2, v3",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_floating_point_dot_product_instruction"
            };

        case "VADD.VX":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vadd.vv vd, vs2, vs1, vm   # Vector-vector</div>\n<div>vadd.vx vd, vs2, rs1, vm   # vector-scalar</div>\n<div>vadd.vi vd, vs2, imm, vm   # vector-immediate</div>\n</div>",
                "tooltip": "vadd.vv vd, vs2, vs1, vm   # Vector-vector\nvadd.vx vd, vs2, rs1, vm   # vector-scalar\nvadd.vi vd, vs2, imm, vm   # vector-immediate",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_integer_add_and_subtract"
            };

        case "VSUB.VX":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vsub.vv vd, vs2, vs1, vm   # Vector-vector</div>\n<div>vsub.vx vd, vs2, rs1, vm   # vector-scalar</div>\n</div>",
                "tooltip": "vsub.vv vd, vs2, vs1, vm   # Vector-vector\nvsub.vx vd, vs2, rs1, vm   # vector-scalar",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_integer_add_and_subtract"
            };

        case "VRSUB.VX":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vrsub.vx vd, vs2, rs1, vm   # vd[i] = rs1 - vs2[i]</div>\n<div>vrsub.vi vd, vs2, imm, vm   # vd[i] = imm - vs2[i]</div>\n</div>",
                "tooltip": "vrsub.vx vd, vs2, rs1, vm   # vd[i] = rs1 - vs2[i]\nvrsub.vi vd, vs2, imm, vm   # vd[i] = imm - vs2[i]",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_integer_add_and_subtract"
            };

        case "VADD.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vadd.vv vd, vs2, vs1, vm   # Vector-vector</div>\n<div>vadd.vx vd, vs2, rs1, vm   # vector-scalar</div>\n<div>vadd.vi vd, vs2, imm, vm   # vector-immediate</div>\n</div>",
                "tooltip": "vadd.vv vd, vs2, vs1, vm   # Vector-vector\nvadd.vx vd, vs2, rs1, vm   # vector-scalar\nvadd.vi vd, vs2, imm, vm   # vector-immediate",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_integer_add_and_subtract"
            };

        case "VSUB.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vsub.vv vd, vs2, vs1, vm   # Vector-vector</div>\n<div>vsub.vx vd, vs2, rs1, vm   # vector-scalar</div>\n</div>",
                "tooltip": "vsub.vv vd, vs2, vs1, vm   # Vector-vector\nvsub.vx vd, vs2, rs1, vm   # vector-scalar",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_integer_add_and_subtract"
            };

        case "VADD.VI":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, simm5, vd</span>\n\n<div>vadd.vv vd, vs2, vs1, vm   # Vector-vector</div>\n<div>vadd.vx vd, vs2, rs1, vm   # vector-scalar</div>\n<div>vadd.vi vd, vs2, imm, vm   # vector-immediate</div>\n</div>",
                "tooltip": "vadd.vv vd, vs2, vs1, vm   # Vector-vector\nvadd.vx vd, vs2, rs1, vm   # vector-scalar\nvadd.vi vd, vs2, imm, vm   # vector-immediate",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_integer_add_and_subtract"
            };

        case "VRSUB.VI":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, simm5, vd</span>\n\n<div>vrsub.vx vd, vs2, rs1, vm   # vd[i] = rs1 - vs2[i]</div>\n<div>vrsub.vi vd, vs2, imm, vm   # vd[i] = imm - vs2[i]</div>\n</div>",
                "tooltip": "vrsub.vx vd, vs2, rs1, vm   # vd[i] = rs1 - vs2[i]\nvrsub.vi vd, vs2, imm, vm   # vd[i] = imm - vs2[i]",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_integer_add_and_subtract"
            };

        case "VMINU.VX":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vminu.vv vd, vs2, vs1, vm   # Vector-vector</div>\n<div>vminu.vx vd, vs2, rs1, vm   # vector-scalar</div>\n</div>",
                "tooltip": "vminu.vv vd, vs2, vs1, vm   # Vector-vector\nvminu.vx vd, vs2, rs1, vm   # vector-scalar",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_integer_min_max_instructions"
            };

        case "VMIN.VX":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vmin.vv vd, vs2, vs1, vm   # Vector-vector</div>\n<div>vmin.vx vd, vs2, rs1, vm   # vector-scalar</div>\n</div>",
                "tooltip": "vmin.vv vd, vs2, vs1, vm   # Vector-vector\nvmin.vx vd, vs2, rs1, vm   # vector-scalar",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_integer_min_max_instructions"
            };

        case "VMAXU.VX":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vmaxu.vv vd, vs2, vs1, vm   # Vector-vector</div>\n<div>vmaxu.vx vd, vs2, rs1, vm   # vector-scalar</div>\n</div>",
                "tooltip": "vmaxu.vv vd, vs2, vs1, vm   # Vector-vector\nvmaxu.vx vd, vs2, rs1, vm   # vector-scalar",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_integer_min_max_instructions"
            };

        case "VMAX.VX":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vmax.vv vd, vs2, vs1, vm   # Vector-vector</div>\n<div>vmax.vx vd, vs2, rs1, vm   # vector-scalar</div>\n</div>",
                "tooltip": "vmax.vv vd, vs2, vs1, vm   # Vector-vector\nvmax.vx vd, vs2, rs1, vm   # vector-scalar",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_integer_min_max_instructions"
            };

        case "VMINU.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vminu.vv vd, vs2, vs1, vm   # Vector-vector</div>\n<div>vminu.vx vd, vs2, rs1, vm   # vector-scalar</div>\n</div>",
                "tooltip": "vminu.vv vd, vs2, vs1, vm   # Vector-vector\nvminu.vx vd, vs2, rs1, vm   # vector-scalar",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_integer_min_max_instructions"
            };

        case "VMIN.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vmin.vv vd, vs2, vs1, vm   # Vector-vector</div>\n<div>vmin.vx vd, vs2, rs1, vm   # vector-scalar</div>\n</div>",
                "tooltip": "vmin.vv vd, vs2, vs1, vm   # Vector-vector\nvmin.vx vd, vs2, rs1, vm   # vector-scalar",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_integer_min_max_instructions"
            };

        case "VMAXU.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vmaxu.vv vd, vs2, vs1, vm   # Vector-vector</div>\n<div>vmaxu.vx vd, vs2, rs1, vm   # vector-scalar</div>\n</div>",
                "tooltip": "vmaxu.vv vd, vs2, vs1, vm   # Vector-vector\nvmaxu.vx vd, vs2, rs1, vm   # vector-scalar",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_integer_min_max_instructions"
            };

        case "VMAX.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vmax.vv vd, vs2, vs1, vm   # Vector-vector</div>\n<div>vmax.vx vd, vs2, rs1, vm   # vector-scalar</div>\n</div>",
                "tooltip": "vmax.vv vd, vs2, vs1, vm   # Vector-vector\nvmax.vx vd, vs2, rs1, vm   # vector-scalar",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_integer_min_max_instructions"
            };

        case "VAND.VX":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vand.vv vd, vs2, vs1, vm   # Vector-vector</div>\n<div>vand.vx vd, vs2, rs1, vm   # vector-scalar</div>\n<div>vand.vi vd, vs2, imm, vm   # vector-immediate</div>\n</div>",
                "tooltip": "vand.vv vd, vs2, vs1, vm   # Vector-vector\nvand.vx vd, vs2, rs1, vm   # vector-scalar\nvand.vi vd, vs2, imm, vm   # vector-immediate",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_bitwise_logical_instructions"
            };

        case "VOR.VX":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vor.vv vd, vs2, vs1, vm    # Vector-vector</div>\n<div>vor.vx vd, vs2, rs1, vm    # vector-scalar</div>\n<div>vor.vi vd, vs2, imm, vm    # vector-immediate</div>\n</div>",
                "tooltip": "vor.vv vd, vs2, vs1, vm    # Vector-vector\nvor.vx vd, vs2, rs1, vm    # vector-scalar\nvor.vi vd, vs2, imm, vm    # vector-immediate",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_bitwise_logical_instructions"
            };

        case "VXOR.VX":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>Note vxor vnot.v</div>\n<div>vxor.vv vd, vs2, vs1, vm    # Vector-vector</div>\n<div>vxor.vx vd, vs2, rs1, vm    # vector-scalar</div>\n<div>vxor.vi vd, vs2, imm, vm    # vector-immediate</div>\n</div>",
                "tooltip": "Note vxor vnot.v\nvxor.vv vd, vs2, vs1, vm    # Vector-vector\nvxor.vx vd, vs2, rs1, vm    # vector-scalar\nvxor.vi vd, vs2, imm, vm    # vector-immediate",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_bitwise_logical_instructions"
            };

        case "VAND.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vand.vv vd, vs2, vs1, vm   # Vector-vector</div>\n<div>vand.vx vd, vs2, rs1, vm   # vector-scalar</div>\n<div>vand.vi vd, vs2, imm, vm   # vector-immediate</div>\n</div>",
                "tooltip": "vand.vv vd, vs2, vs1, vm   # Vector-vector\nvand.vx vd, vs2, rs1, vm   # vector-scalar\nvand.vi vd, vs2, imm, vm   # vector-immediate",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_bitwise_logical_instructions"
            };

        case "VOR.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vor.vv vd, vs2, vs1, vm    # Vector-vector</div>\n<div>vor.vx vd, vs2, rs1, vm    # vector-scalar</div>\n<div>vor.vi vd, vs2, imm, vm    # vector-immediate</div>\n</div>",
                "tooltip": "vor.vv vd, vs2, vs1, vm    # Vector-vector\nvor.vx vd, vs2, rs1, vm    # vector-scalar\nvor.vi vd, vs2, imm, vm    # vector-immediate",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_bitwise_logical_instructions"
            };

        case "VXOR.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>Note vxor vnot.v</div>\n<div>vxor.vv vd, vs2, vs1, vm    # Vector-vector</div>\n<div>vxor.vx vd, vs2, rs1, vm    # vector-scalar</div>\n<div>vxor.vi vd, vs2, imm, vm    # vector-immediate</div>\n</div>",
                "tooltip": "Note vxor vnot.v\nvxor.vv vd, vs2, vs1, vm    # Vector-vector\nvxor.vx vd, vs2, rs1, vm    # vector-scalar\nvxor.vi vd, vs2, imm, vm    # vector-immediate",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_bitwise_logical_instructions"
            };

        case "VAND.VI":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, simm5, vd</span>\n\n<div>vand.vv vd, vs2, vs1, vm   # Vector-vector</div>\n<div>vand.vx vd, vs2, rs1, vm   # vector-scalar</div>\n<div>vand.vi vd, vs2, imm, vm   # vector-immediate</div>\n</div>",
                "tooltip": "vand.vv vd, vs2, vs1, vm   # Vector-vector\nvand.vx vd, vs2, rs1, vm   # vector-scalar\nvand.vi vd, vs2, imm, vm   # vector-immediate",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_bitwise_logical_instructions"
            };

        case "VOR.VI":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, simm5, vd</span>\n\n<div>vor.vv vd, vs2, vs1, vm    # Vector-vector</div>\n<div>vor.vx vd, vs2, rs1, vm    # vector-scalar</div>\n<div>vor.vi vd, vs2, imm, vm    # vector-immediate</div>\n</div>",
                "tooltip": "vor.vv vd, vs2, vs1, vm    # Vector-vector\nvor.vx vd, vs2, rs1, vm    # vector-scalar\nvor.vi vd, vs2, imm, vm    # vector-immediate",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_bitwise_logical_instructions"
            };

        case "VXOR.VI":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, simm5, vd</span>\n\n<div>Note vxor vnot.v</div>\n<div>vxor.vv vd, vs2, vs1, vm    # Vector-vector</div>\n<div>vxor.vx vd, vs2, rs1, vm    # vector-scalar</div>\n<div>vxor.vi vd, vs2, imm, vm    # vector-immediate</div>\n</div>",
                "tooltip": "Note vxor vnot.v\nvxor.vv vd, vs2, vs1, vm    # Vector-vector\nvxor.vx vd, vs2, rs1, vm    # vector-scalar\nvxor.vi vd, vs2, imm, vm    # vector-immediate",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_bitwise_logical_instructions"
            };

        case "VRGATHER.VX":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>For any vrgather instruction, the destination vector register group cannot overlap with the source vector register groups, including the mask register if the operation is masked, otherwise an illegal instruction exception is raised.</div>\n<div>Note vrgather.vv can only reference vector elements 0-255.</div>\n<div>vrgather.vv vd, vs2, vs1, vm # vd[i] = (vs1[i] &gt;= VLMAX) ? 0 : vs2[vs1[i]];</div>\n<div>vrgather.vx vd, vs2, rs1, vm # vd[i] = (x[rs1] &gt;= VLMAX) ? 0 : vs2[x[rs1]]</div>\n<div>vrgather.vi vd, vs2, uimm, vm # vd[i] = (uimm &gt;= VLMAX) ? 0 : vs2[uimm]</div>\n</div>",
                "tooltip": "For any vrgather instruction, the destination vector register group cannot overlap with the source vector register groups, including the mask register if the operation is masked, otherwise an illegal instruction exception is raised.\nNote vrgather.vv can only reference vector elements 0-255.\nvrgather.vv vd, vs2, vs1, vm # vd[i] = (vs1[i] >= VLMAX) ? 0 : vs2[vs1[i]];\nvrgather.vx vd, vs2, rs1, vm # vd[i] = (x[rs1] >= VLMAX) ? 0 : vs2[x[rs1]]\nvrgather.vi vd, vs2, uimm, vm # vd[i] = (uimm >= VLMAX) ? 0 : vs2[uimm]",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_register_gather_instruction"
            };

        case "VRGATHER.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>For any vrgather instruction, the destination vector register group cannot overlap with the source vector register groups, including the mask register if the operation is masked, otherwise an illegal instruction exception is raised.</div>\n<div>Note vrgather.vv can only reference vector elements 0-255.</div>\n<div>vrgather.vv vd, vs2, vs1, vm # vd[i] = (vs1[i] &gt;= VLMAX) ? 0 : vs2[vs1[i]];</div>\n<div>vrgather.vx vd, vs2, rs1, vm # vd[i] = (x[rs1] &gt;= VLMAX) ? 0 : vs2[x[rs1]]</div>\n<div>vrgather.vi vd, vs2, uimm, vm # vd[i] = (uimm &gt;= VLMAX) ? 0 : vs2[uimm]</div>\n</div>",
                "tooltip": "For any vrgather instruction, the destination vector register group cannot overlap with the source vector register groups, including the mask register if the operation is masked, otherwise an illegal instruction exception is raised.\nNote vrgather.vv can only reference vector elements 0-255.\nvrgather.vv vd, vs2, vs1, vm # vd[i] = (vs1[i] >= VLMAX) ? 0 : vs2[vs1[i]];\nvrgather.vx vd, vs2, rs1, vm # vd[i] = (x[rs1] >= VLMAX) ? 0 : vs2[x[rs1]]\nvrgather.vi vd, vs2, uimm, vm # vd[i] = (uimm >= VLMAX) ? 0 : vs2[uimm]",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_register_gather_instruction"
            };

        case "VRGATHER.VI":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, simm5, vd</span>\n\n<div>For any vrgather instruction, the destination vector register group cannot overlap with the source vector register groups, including the mask register if the operation is masked, otherwise an illegal instruction exception is raised.</div>\n<div>Note vrgather.vv can only reference vector elements 0-255.</div>\n<div>vrgather.vv vd, vs2, vs1, vm # vd[i] = (vs1[i] &gt;= VLMAX) ? 0 : vs2[vs1[i]];</div>\n<div>vrgather.vx vd, vs2, rs1, vm # vd[i] = (x[rs1] &gt;= VLMAX) ? 0 : vs2[x[rs1]]</div>\n<div>vrgather.vi vd, vs2, uimm, vm # vd[i] = (uimm &gt;= VLMAX) ? 0 : vs2[uimm]</div>\n</div>",
                "tooltip": "For any vrgather instruction, the destination vector register group cannot overlap with the source vector register groups, including the mask register if the operation is masked, otherwise an illegal instruction exception is raised.\nNote vrgather.vv can only reference vector elements 0-255.\nvrgather.vv vd, vs2, vs1, vm # vd[i] = (vs1[i] >= VLMAX) ? 0 : vs2[vs1[i]];\nvrgather.vx vd, vs2, rs1, vm # vd[i] = (x[rs1] >= VLMAX) ? 0 : vs2[x[rs1]]\nvrgather.vi vd, vs2, uimm, vm # vd[i] = (uimm >= VLMAX) ? 0 : vs2[uimm]",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_register_gather_instruction"
            };

        case "VSLIDEUP.VX":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>Note vslideup and vslidedown</div>\n<div>For all of the vslideup , vslidedown , vslide1up , and vslide1down instructions, if vstart \u00e2\u0089\u00a5 vl , the instruction performs no operation and leaves the destination vector register unchanged.</div>\n</div>",
                "tooltip": "Note vslideup and vslidedown\nFor all of the vslideup , vslidedown , vslide1up , and vslide1down instructions, if vstart \u00e2\u0089\u00a5 vl , the instruction performs no operation and leaves the destination vector register unchanged.",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_slide_instructions"
            };

        case "VSLIDEUP.VI":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, simm5, vd</span>\n\n<div>Note vslideup and vslidedown</div>\n<div>For all of the vslideup , vslidedown , vslide1up , and vslide1down instructions, if vstart \u00e2\u0089\u00a5 vl , the instruction performs no operation and leaves the destination vector register unchanged.</div>\n</div>",
                "tooltip": "Note vslideup and vslidedown\nFor all of the vslideup , vslidedown , vslide1up , and vslide1down instructions, if vstart \u00e2\u0089\u00a5 vl , the instruction performs no operation and leaves the destination vector register unchanged.",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_slide_instructions"
            };

        case "VSLIDEDOWN.VX":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>For vslidedown , the value in vl specifies the number of destination elements that are written.</div>\n<div>vslidedown.vx vd, vs2, rs1, vm       # vd[i] = vs2[i+rs1]</div>\n<div>vslidedown.vi vd, vs2, uimm[4:0], vm # vd[i] = vs2[i+uimm]</div>\n<div>vslidedown behavior for source elements for element i in slide</div>\n<div>vslidedown behavior for destination element i in slide</div>\n</div>",
                "tooltip": "For vslidedown , the value in vl specifies the number of destination elements that are written.\nvslidedown.vx vd, vs2, rs1, vm       # vd[i] = vs2[i+rs1]\nvslidedown.vi vd, vs2, uimm[4:0], vm # vd[i] = vs2[i+uimm]\nvslidedown behavior for source elements for element i in slide\nvslidedown behavior for destination element i in slide",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_slidedown_instructions"
            };

        case "VSLIDEDOWN.VI":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, simm5, vd</span>\n\n<div>For vslidedown , the value in vl specifies the number of destination elements that are written.</div>\n<div>vslidedown.vx vd, vs2, rs1, vm       # vd[i] = vs2[i+rs1]</div>\n<div>vslidedown.vi vd, vs2, uimm[4:0], vm # vd[i] = vs2[i+uimm]</div>\n<div>vslidedown behavior for source elements for element i in slide</div>\n<div>vslidedown behavior for destination element i in slide</div>\n</div>",
                "tooltip": "For vslidedown , the value in vl specifies the number of destination elements that are written.\nvslidedown.vx vd, vs2, rs1, vm       # vd[i] = vs2[i+rs1]\nvslidedown.vi vd, vs2, uimm[4:0], vm # vd[i] = vs2[i+uimm]\nvslidedown behavior for source elements for element i in slide\nvslidedown behavior for destination element i in slide",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_slidedown_instructions"
            };

        case "VADC.VXM":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>. Due to encoding constraints, the carry input must come from the implicit v0 vadc and vsbc add or subtract the source operands and the carry-in or borrow-in, and write the result to vector register vd</div>\n<div>For vadc and vsbc , an illegal instruction exception is raised if the destination vector register is v0 and LMUL &gt; 1.</div>\n<div>vadc.vvm   vd, vs2, vs1, v0  # Vector-vector</div>\n<div>vadc.vxm   vd, vs2, rs1, v0  # Vector-scalar</div>\n<div>vadc.vim   vd, vs2, imm, v0  # Vector-immediate</div>\n<div>vadc.vvm v4, v4, v8, v0   # Calc new sum</div>\n</div>",
                "tooltip": ". Due to encoding constraints, the carry input must come from the implicit v0 vadc and vsbc add or subtract the source operands and the carry-in or borrow-in, and write the result to vector register vd\nFor vadc and vsbc , an illegal instruction exception is raised if the destination vector register is v0 and LMUL > 1.\nvadc.vvm   vd, vs2, vs1, v0  # Vector-vector\nvadc.vxm   vd, vs2, rs1, v0  # Vector-scalar\nvadc.vim   vd, vs2, imm, v0  # Vector-immediate\nvadc.vvm v4, v4, v8, v0   # Calc new sum",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_integer_add_with_carry_subtract_with_borrow_instructions"
            };

        case "VMADC.VXM":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vmadc and vmsbc add or subtract the source operands, optionally add the carry-in or subtract the borrow-in if masked ( vm=0 ), and write the result back to mask register vd</div>\n<div>For vmadc and vmsbc , an illegal instruction exception is raised if the destination vector register overlaps a source vector register group and LMUL &gt; 1.</div>\n<div>vmadc.vvm   vd, vs2, vs1, v0  # Vector-vector</div>\n<div>vmadc.vxm   vd, vs2, rs1, v0  # Vector-scalar</div>\n<div>vmadc.vim   vd, vs2, imm, v0  # Vector-immediate</div>\n<div>vmadc.vv    vd, vs2, vs1      # Vector-vector, no carry-in</div>\n<div>vmadc.vx    vd, vs2, rs1      # Vector-scalar, no carry-in</div>\n<div>vmadc.vi    vd, vs2, imm      # Vector-immediate, no carry-in</div>\n<div>vmadc.vvm v1, v4, v8, v0  # Get carry into temp register v1</div>\n</div>",
                "tooltip": "vmadc and vmsbc add or subtract the source operands, optionally add the carry-in or subtract the borrow-in if masked ( vm=0 ), and write the result back to mask register vd\nFor vmadc and vmsbc , an illegal instruction exception is raised if the destination vector register overlaps a source vector register group and LMUL > 1.\nvmadc.vvm   vd, vs2, vs1, v0  # Vector-vector\nvmadc.vxm   vd, vs2, rs1, v0  # Vector-scalar\nvmadc.vim   vd, vs2, imm, v0  # Vector-immediate\nvmadc.vv    vd, vs2, vs1      # Vector-vector, no carry-in\nvmadc.vx    vd, vs2, rs1      # Vector-scalar, no carry-in\nvmadc.vi    vd, vs2, imm      # Vector-immediate, no carry-in\nvmadc.vvm v1, v4, v8, v0  # Get carry into temp register v1",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_integer_add_with_carry_subtract_with_borrow_instructions"
            };

        case "VSBC.VXM":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>The subtract with borrow instruction vsbc performs the equivalent function to support long word arithmetic for subtraction</div>\n<div>vsbc.vvm   vd, vs2, vs1, v0  # Vector-vector</div>\n<div>vsbc.vxm   vd, vs2, rs1, v0  # Vector-scalar</div>\n</div>",
                "tooltip": "The subtract with borrow instruction vsbc performs the equivalent function to support long word arithmetic for subtraction\nvsbc.vvm   vd, vs2, vs1, v0  # Vector-vector\nvsbc.vxm   vd, vs2, rs1, v0  # Vector-scalar",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_integer_add_with_carry_subtract_with_borrow_instructions"
            };

        case "VMSBC.VXM":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>For vmsbc , the borrow is defined to be 1 iff the difference, prior to truncation, is negative.</div>\n<div>vmsbc.vvm   vd, vs2, vs1, v0  # Vector-vector</div>\n<div>vmsbc.vxm   vd, vs2, rs1, v0  # Vector-scalar</div>\n<div>vmsbc.vv    vd, vs2, vs1      # Vector-vector, no borrow-in</div>\n<div>vmsbc.vx    vd, vs2, rs1      # Vector-scalar, no borrow-in</div>\n</div>",
                "tooltip": "For vmsbc , the borrow is defined to be 1 iff the difference, prior to truncation, is negative.\nvmsbc.vvm   vd, vs2, vs1, v0  # Vector-vector\nvmsbc.vxm   vd, vs2, rs1, v0  # Vector-scalar\nvmsbc.vv    vd, vs2, vs1      # Vector-vector, no borrow-in\nvmsbc.vx    vd, vs2, rs1      # Vector-scalar, no borrow-in",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_integer_add_with_carry_subtract_with_borrow_instructions"
            };

        case "VADC.VVM":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>. Due to encoding constraints, the carry input must come from the implicit v0 vadc and vsbc add or subtract the source operands and the carry-in or borrow-in, and write the result to vector register vd</div>\n<div>For vadc and vsbc , an illegal instruction exception is raised if the destination vector register is v0 and LMUL &gt; 1.</div>\n<div>vadc.vvm   vd, vs2, vs1, v0  # Vector-vector</div>\n<div>vadc.vxm   vd, vs2, rs1, v0  # Vector-scalar</div>\n<div>vadc.vim   vd, vs2, imm, v0  # Vector-immediate</div>\n<div>vadc.vvm v4, v4, v8, v0   # Calc new sum</div>\n</div>",
                "tooltip": ". Due to encoding constraints, the carry input must come from the implicit v0 vadc and vsbc add or subtract the source operands and the carry-in or borrow-in, and write the result to vector register vd\nFor vadc and vsbc , an illegal instruction exception is raised if the destination vector register is v0 and LMUL > 1.\nvadc.vvm   vd, vs2, vs1, v0  # Vector-vector\nvadc.vxm   vd, vs2, rs1, v0  # Vector-scalar\nvadc.vim   vd, vs2, imm, v0  # Vector-immediate\nvadc.vvm v4, v4, v8, v0   # Calc new sum",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_integer_add_with_carry_subtract_with_borrow_instructions"
            };

        case "VMADC.VVM":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vmadc and vmsbc add or subtract the source operands, optionally add the carry-in or subtract the borrow-in if masked ( vm=0 ), and write the result back to mask register vd</div>\n<div>For vmadc and vmsbc , an illegal instruction exception is raised if the destination vector register overlaps a source vector register group and LMUL &gt; 1.</div>\n<div>vmadc.vvm   vd, vs2, vs1, v0  # Vector-vector</div>\n<div>vmadc.vxm   vd, vs2, rs1, v0  # Vector-scalar</div>\n<div>vmadc.vim   vd, vs2, imm, v0  # Vector-immediate</div>\n<div>vmadc.vv    vd, vs2, vs1      # Vector-vector, no carry-in</div>\n<div>vmadc.vx    vd, vs2, rs1      # Vector-scalar, no carry-in</div>\n<div>vmadc.vi    vd, vs2, imm      # Vector-immediate, no carry-in</div>\n<div>vmadc.vvm v1, v4, v8, v0  # Get carry into temp register v1</div>\n</div>",
                "tooltip": "vmadc and vmsbc add or subtract the source operands, optionally add the carry-in or subtract the borrow-in if masked ( vm=0 ), and write the result back to mask register vd\nFor vmadc and vmsbc , an illegal instruction exception is raised if the destination vector register overlaps a source vector register group and LMUL > 1.\nvmadc.vvm   vd, vs2, vs1, v0  # Vector-vector\nvmadc.vxm   vd, vs2, rs1, v0  # Vector-scalar\nvmadc.vim   vd, vs2, imm, v0  # Vector-immediate\nvmadc.vv    vd, vs2, vs1      # Vector-vector, no carry-in\nvmadc.vx    vd, vs2, rs1      # Vector-scalar, no carry-in\nvmadc.vi    vd, vs2, imm      # Vector-immediate, no carry-in\nvmadc.vvm v1, v4, v8, v0  # Get carry into temp register v1",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_integer_add_with_carry_subtract_with_borrow_instructions"
            };

        case "VSBC.VVM":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>The subtract with borrow instruction vsbc performs the equivalent function to support long word arithmetic for subtraction</div>\n<div>vsbc.vvm   vd, vs2, vs1, v0  # Vector-vector</div>\n<div>vsbc.vxm   vd, vs2, rs1, v0  # Vector-scalar</div>\n</div>",
                "tooltip": "The subtract with borrow instruction vsbc performs the equivalent function to support long word arithmetic for subtraction\nvsbc.vvm   vd, vs2, vs1, v0  # Vector-vector\nvsbc.vxm   vd, vs2, rs1, v0  # Vector-scalar",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_integer_add_with_carry_subtract_with_borrow_instructions"
            };

        case "VMSBC.VVM":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>For vmsbc , the borrow is defined to be 1 iff the difference, prior to truncation, is negative.</div>\n<div>vmsbc.vvm   vd, vs2, vs1, v0  # Vector-vector</div>\n<div>vmsbc.vxm   vd, vs2, rs1, v0  # Vector-scalar</div>\n<div>vmsbc.vv    vd, vs2, vs1      # Vector-vector, no borrow-in</div>\n<div>vmsbc.vx    vd, vs2, rs1      # Vector-scalar, no borrow-in</div>\n</div>",
                "tooltip": "For vmsbc , the borrow is defined to be 1 iff the difference, prior to truncation, is negative.\nvmsbc.vvm   vd, vs2, vs1, v0  # Vector-vector\nvmsbc.vxm   vd, vs2, rs1, v0  # Vector-scalar\nvmsbc.vv    vd, vs2, vs1      # Vector-vector, no borrow-in\nvmsbc.vx    vd, vs2, rs1      # Vector-scalar, no borrow-in",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_integer_add_with_carry_subtract_with_borrow_instructions"
            };

        case "VADC.VIM":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, simm5, vd</span>\n\n<div>. Due to encoding constraints, the carry input must come from the implicit v0 vadc and vsbc add or subtract the source operands and the carry-in or borrow-in, and write the result to vector register vd</div>\n<div>For vadc and vsbc , an illegal instruction exception is raised if the destination vector register is v0 and LMUL &gt; 1.</div>\n<div>vadc.vvm   vd, vs2, vs1, v0  # Vector-vector</div>\n<div>vadc.vxm   vd, vs2, rs1, v0  # Vector-scalar</div>\n<div>vadc.vim   vd, vs2, imm, v0  # Vector-immediate</div>\n<div>vadc.vvm v4, v4, v8, v0   # Calc new sum</div>\n</div>",
                "tooltip": ". Due to encoding constraints, the carry input must come from the implicit v0 vadc and vsbc add or subtract the source operands and the carry-in or borrow-in, and write the result to vector register vd\nFor vadc and vsbc , an illegal instruction exception is raised if the destination vector register is v0 and LMUL > 1.\nvadc.vvm   vd, vs2, vs1, v0  # Vector-vector\nvadc.vxm   vd, vs2, rs1, v0  # Vector-scalar\nvadc.vim   vd, vs2, imm, v0  # Vector-immediate\nvadc.vvm v4, v4, v8, v0   # Calc new sum",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_integer_add_with_carry_subtract_with_borrow_instructions"
            };

        case "VMADC.VIM":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, simm5, vd</span>\n\n<div>vmadc and vmsbc add or subtract the source operands, optionally add the carry-in or subtract the borrow-in if masked ( vm=0 ), and write the result back to mask register vd</div>\n<div>For vmadc and vmsbc , an illegal instruction exception is raised if the destination vector register overlaps a source vector register group and LMUL &gt; 1.</div>\n<div>vmadc.vvm   vd, vs2, vs1, v0  # Vector-vector</div>\n<div>vmadc.vxm   vd, vs2, rs1, v0  # Vector-scalar</div>\n<div>vmadc.vim   vd, vs2, imm, v0  # Vector-immediate</div>\n<div>vmadc.vv    vd, vs2, vs1      # Vector-vector, no carry-in</div>\n<div>vmadc.vx    vd, vs2, rs1      # Vector-scalar, no carry-in</div>\n<div>vmadc.vi    vd, vs2, imm      # Vector-immediate, no carry-in</div>\n<div>vmadc.vvm v1, v4, v8, v0  # Get carry into temp register v1</div>\n</div>",
                "tooltip": "vmadc and vmsbc add or subtract the source operands, optionally add the carry-in or subtract the borrow-in if masked ( vm=0 ), and write the result back to mask register vd\nFor vmadc and vmsbc , an illegal instruction exception is raised if the destination vector register overlaps a source vector register group and LMUL > 1.\nvmadc.vvm   vd, vs2, vs1, v0  # Vector-vector\nvmadc.vxm   vd, vs2, rs1, v0  # Vector-scalar\nvmadc.vim   vd, vs2, imm, v0  # Vector-immediate\nvmadc.vv    vd, vs2, vs1      # Vector-vector, no carry-in\nvmadc.vx    vd, vs2, rs1      # Vector-scalar, no carry-in\nvmadc.vi    vd, vs2, imm      # Vector-immediate, no carry-in\nvmadc.vvm v1, v4, v8, v0  # Get carry into temp register v1",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_integer_add_with_carry_subtract_with_borrow_instructions"
            };

        case "VMERGE.VXM":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>The vmerge instructions are always masked ( vm=0 )</div>\n<div>vmerge.vvm vd, vs2, vs1, v0  # vd[i] = v0[i].LSB ? vs1[i] : vs2[i]</div>\n<div>vmerge.vxm vd, vs2, rs1, v0  # vd[i] = v0[i].LSB ? x[rs1] : vs2[i]</div>\n<div>vmerge.vim vd, vs2, imm, v0  # vd[i] = v0[i].LSB ? imm    : vs2[i]</div>\n</div>",
                "tooltip": "The vmerge instructions are always masked ( vm=0 )\nvmerge.vvm vd, vs2, vs1, v0  # vd[i] = v0[i].LSB ? vs1[i] : vs2[i]\nvmerge.vxm vd, vs2, rs1, v0  # vd[i] = v0[i].LSB ? x[rs1] : vs2[i]\nvmerge.vim vd, vs2, imm, v0  # vd[i] = v0[i].LSB ? imm    : vs2[i]",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_integer_merge_instructions"
            };

        case "VMERGE.VVM":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>The vmerge instructions are always masked ( vm=0 )</div>\n<div>vmerge.vvm vd, vs2, vs1, v0  # vd[i] = v0[i].LSB ? vs1[i] : vs2[i]</div>\n<div>vmerge.vxm vd, vs2, rs1, v0  # vd[i] = v0[i].LSB ? x[rs1] : vs2[i]</div>\n<div>vmerge.vim vd, vs2, imm, v0  # vd[i] = v0[i].LSB ? imm    : vs2[i]</div>\n</div>",
                "tooltip": "The vmerge instructions are always masked ( vm=0 )\nvmerge.vvm vd, vs2, vs1, v0  # vd[i] = v0[i].LSB ? vs1[i] : vs2[i]\nvmerge.vxm vd, vs2, rs1, v0  # vd[i] = v0[i].LSB ? x[rs1] : vs2[i]\nvmerge.vim vd, vs2, imm, v0  # vd[i] = v0[i].LSB ? imm    : vs2[i]",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_integer_merge_instructions"
            };

        case "VMERGE.VIM":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, simm5, vd</span>\n\n<div>The vmerge instructions are always masked ( vm=0 )</div>\n<div>vmerge.vvm vd, vs2, vs1, v0  # vd[i] = v0[i].LSB ? vs1[i] : vs2[i]</div>\n<div>vmerge.vxm vd, vs2, rs1, v0  # vd[i] = v0[i].LSB ? x[rs1] : vs2[i]</div>\n<div>vmerge.vim vd, vs2, imm, v0  # vd[i] = v0[i].LSB ? imm    : vs2[i]</div>\n</div>",
                "tooltip": "The vmerge instructions are always masked ( vm=0 )\nvmerge.vvm vd, vs2, vs1, v0  # vd[i] = v0[i].LSB ? vs1[i] : vs2[i]\nvmerge.vxm vd, vs2, rs1, v0  # vd[i] = v0[i].LSB ? x[rs1] : vs2[i]\nvmerge.vim vd, vs2, imm, v0  # vd[i] = v0[i].LSB ? imm    : vs2[i]",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_integer_merge_instructions"
            };

        case "VMV.V.X":
            return {
                "html": "<div>\n<span class=\"opcode\">rs1, vd</span>\n\n<div>This instruction copies the vs1 , rs1 , or immediate operand to the first vl Note vmv.v.i vd, 0; vmerge.vim vd, vd, 1, v0</div>\n<div>vmv.v.v vd, vs1 # vd[i] = vs1[i]</div>\n<div>vmv.v.x vd, rs1 # vd[i] = rs1</div>\n<div>vmv.v.i vd, imm # vd[i] = imm</div>\n</div>",
                "tooltip": "This instruction copies the vs1 , rs1 , or immediate operand to the first vl Note vmv.v.i vd, 0; vmerge.vim vd, vd, 1, v0\nvmv.v.v vd, vs1 # vd[i] = vs1[i]\nvmv.v.x vd, rs1 # vd[i] = rs1\nvmv.v.i vd, imm # vd[i] = imm",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_integer_move_instructions"
            };

        case "VMV.V.V":
            return {
                "html": "<div>\n<span class=\"opcode\">rs1, vd</span>\n\n<div>This instruction copies the vs1 , rs1 , or immediate operand to the first vl Note vmv.v.i vd, 0; vmerge.vim vd, vd, 1, v0</div>\n<div>vmv.v.v vd, vs1 # vd[i] = vs1[i]</div>\n<div>vmv.v.x vd, rs1 # vd[i] = rs1</div>\n<div>vmv.v.i vd, imm # vd[i] = imm</div>\n</div>",
                "tooltip": "This instruction copies the vs1 , rs1 , or immediate operand to the first vl Note vmv.v.i vd, 0; vmerge.vim vd, vd, 1, v0\nvmv.v.v vd, vs1 # vd[i] = vs1[i]\nvmv.v.x vd, rs1 # vd[i] = rs1\nvmv.v.i vd, imm # vd[i] = imm",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_integer_move_instructions"
            };

        case "VMV.V.I":
            return {
                "html": "<div>\n<span class=\"opcode\">simm5, vd</span>\n\n<div>This instruction copies the vs1 , rs1 , or immediate operand to the first vl Note vmv.v.i vd, 0; vmerge.vim vd, vd, 1, v0</div>\n<div>vmv.v.v vd, vs1 # vd[i] = vs1[i]</div>\n<div>vmv.v.x vd, rs1 # vd[i] = rs1</div>\n<div>vmv.v.i vd, imm # vd[i] = imm</div>\n</div>",
                "tooltip": "This instruction copies the vs1 , rs1 , or immediate operand to the first vl Note vmv.v.i vd, 0; vmerge.vim vd, vd, 1, v0\nvmv.v.v vd, vs1 # vd[i] = vs1[i]\nvmv.v.x vd, rs1 # vd[i] = rs1\nvmv.v.i vd, imm # vd[i] = imm",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_integer_move_instructions"
            };

        case "VMV.S.X":
            return {
                "html": "<div>\n<span class=\"opcode\">rs1, vd</span>\n\n<div>This instruction copies the vs1 , rs1 , or immediate operand to the first vl Note vmv.v.i vd, 0; vmerge.vim vd, vd, 1, v0</div>\n<div>vmv.v.v vd, vs1 # vd[i] = vs1[i]</div>\n<div>vmv.v.x vd, rs1 # vd[i] = rs1</div>\n<div>vmv.v.i vd, imm # vd[i] = imm</div>\n</div>",
                "tooltip": "This instruction copies the vs1 , rs1 , or immediate operand to the first vl Note vmv.v.i vd, 0; vmerge.vim vd, vd, 1, v0\nvmv.v.v vd, vs1 # vd[i] = vs1[i]\nvmv.v.x vd, rs1 # vd[i] = rs1\nvmv.v.i vd, imm # vd[i] = imm",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_integer_move_instructions"
            };

        case "VSADDU.VX":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vsaddu.vv vd, vs2, vs1, vm   # Vector-vector</div>\n<div>vsaddu.vx vd, vs2, rs1, vm   # vector-scalar</div>\n<div>vsaddu.vi vd, vs2, imm, vm   # vector-immediate</div>\n</div>",
                "tooltip": "vsaddu.vv vd, vs2, vs1, vm   # Vector-vector\nvsaddu.vx vd, vs2, rs1, vm   # vector-scalar\nvsaddu.vi vd, vs2, imm, vm   # vector-immediate",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_saturating_add_and_subtract"
            };

        case "VSADD.VX":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vsadd.vv vd, vs2, vs1, vm   # Vector-vector</div>\n<div>vsadd.vx vd, vs2, rs1, vm   # vector-scalar</div>\n<div>vsadd.vi vd, vs2, imm, vm   # vector-immediate</div>\n</div>",
                "tooltip": "vsadd.vv vd, vs2, vs1, vm   # Vector-vector\nvsadd.vx vd, vs2, rs1, vm   # vector-scalar\nvsadd.vi vd, vs2, imm, vm   # vector-immediate",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_saturating_add_and_subtract"
            };

        case "VSSUBU.VX":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vssubu.vv vd, vs2, vs1, vm   # Vector-vector</div>\n<div>vssubu.vx vd, vs2, rs1, vm   # vector-scalar</div>\n</div>",
                "tooltip": "vssubu.vv vd, vs2, vs1, vm   # Vector-vector\nvssubu.vx vd, vs2, rs1, vm   # vector-scalar",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_saturating_add_and_subtract"
            };

        case "VSSUB.VX":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vssub.vv vd, vs2, vs1, vm   # Vector-vector</div>\n<div>vssub.vx vd, vs2, rs1, vm   # vector-scalar</div>\n</div>",
                "tooltip": "vssub.vv vd, vs2, vs1, vm   # Vector-vector\nvssub.vx vd, vs2, rs1, vm   # vector-scalar",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_saturating_add_and_subtract"
            };

        case "VSADDU.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vsaddu.vv vd, vs2, vs1, vm   # Vector-vector</div>\n<div>vsaddu.vx vd, vs2, rs1, vm   # vector-scalar</div>\n<div>vsaddu.vi vd, vs2, imm, vm   # vector-immediate</div>\n</div>",
                "tooltip": "vsaddu.vv vd, vs2, vs1, vm   # Vector-vector\nvsaddu.vx vd, vs2, rs1, vm   # vector-scalar\nvsaddu.vi vd, vs2, imm, vm   # vector-immediate",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_saturating_add_and_subtract"
            };

        case "VSADD.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vsadd.vv vd, vs2, vs1, vm   # Vector-vector</div>\n<div>vsadd.vx vd, vs2, rs1, vm   # vector-scalar</div>\n<div>vsadd.vi vd, vs2, imm, vm   # vector-immediate</div>\n</div>",
                "tooltip": "vsadd.vv vd, vs2, vs1, vm   # Vector-vector\nvsadd.vx vd, vs2, rs1, vm   # vector-scalar\nvsadd.vi vd, vs2, imm, vm   # vector-immediate",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_saturating_add_and_subtract"
            };

        case "VSSUBU.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vssubu.vv vd, vs2, vs1, vm   # Vector-vector</div>\n<div>vssubu.vx vd, vs2, rs1, vm   # vector-scalar</div>\n</div>",
                "tooltip": "vssubu.vv vd, vs2, vs1, vm   # Vector-vector\nvssubu.vx vd, vs2, rs1, vm   # vector-scalar",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_saturating_add_and_subtract"
            };

        case "VSSUB.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vssub.vv vd, vs2, vs1, vm   # Vector-vector</div>\n<div>vssub.vx vd, vs2, rs1, vm   # vector-scalar</div>\n</div>",
                "tooltip": "vssub.vv vd, vs2, vs1, vm   # Vector-vector\nvssub.vx vd, vs2, rs1, vm   # vector-scalar",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_saturating_add_and_subtract"
            };

        case "VSADDU.VI":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, simm5, vd</span>\n\n<div>vsaddu.vv vd, vs2, vs1, vm   # Vector-vector</div>\n<div>vsaddu.vx vd, vs2, rs1, vm   # vector-scalar</div>\n<div>vsaddu.vi vd, vs2, imm, vm   # vector-immediate</div>\n</div>",
                "tooltip": "vsaddu.vv vd, vs2, vs1, vm   # Vector-vector\nvsaddu.vx vd, vs2, rs1, vm   # vector-scalar\nvsaddu.vi vd, vs2, imm, vm   # vector-immediate",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_saturating_add_and_subtract"
            };

        case "VSADD.VI":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, simm5, vd</span>\n\n<div>vsadd.vv vd, vs2, vs1, vm   # Vector-vector</div>\n<div>vsadd.vx vd, vs2, rs1, vm   # vector-scalar</div>\n<div>vsadd.vi vd, vs2, imm, vm   # vector-immediate</div>\n</div>",
                "tooltip": "vsadd.vv vd, vs2, vs1, vm   # Vector-vector\nvsadd.vx vd, vs2, rs1, vm   # vector-scalar\nvsadd.vi vd, vs2, imm, vm   # vector-immediate",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_saturating_add_and_subtract"
            };

        case "VAADD.VX":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>For vaaddu , vaadd , and vasub , there can be no overflow in the result</div>\n<div>vaadd.vv vd, vs2, vs1, vm   # roundoff_signed(vs2[i] + vs1[i], 1)</div>\n<div>vaadd.vx vd, vs2, rs1, vm   # roundoff_signed(vs2[i] + x[rs1], 1)</div>\n</div>",
                "tooltip": "For vaaddu , vaadd , and vasub , there can be no overflow in the result\nvaadd.vv vd, vs2, vs1, vm   # roundoff_signed(vs2[i] + vs1[i], 1)\nvaadd.vx vd, vs2, rs1, vm   # roundoff_signed(vs2[i] + x[rs1], 1)",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_averaging_add_and_subtract"
            };

        case "VASUB.VX":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vasub.vv vd, vs2, vs1, vm   # roundoff_signed(vs2[i] - vs1[i], 1)</div>\n<div>vasub.vx vd, vs2, rs1, vm   # roundoff_signed(vs2[i] - x[rs1], 1)</div>\n</div>",
                "tooltip": "vasub.vv vd, vs2, vs1, vm   # roundoff_signed(vs2[i] - vs1[i], 1)\nvasub.vx vd, vs2, rs1, vm   # roundoff_signed(vs2[i] - x[rs1], 1)",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_averaging_add_and_subtract"
            };

        case "VAADD.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>For vaaddu , vaadd , and vasub , there can be no overflow in the result</div>\n<div>vaadd.vv vd, vs2, vs1, vm   # roundoff_signed(vs2[i] + vs1[i], 1)</div>\n<div>vaadd.vx vd, vs2, rs1, vm   # roundoff_signed(vs2[i] + x[rs1], 1)</div>\n</div>",
                "tooltip": "For vaaddu , vaadd , and vasub , there can be no overflow in the result\nvaadd.vv vd, vs2, vs1, vm   # roundoff_signed(vs2[i] + vs1[i], 1)\nvaadd.vx vd, vs2, rs1, vm   # roundoff_signed(vs2[i] + x[rs1], 1)",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_averaging_add_and_subtract"
            };

        case "VASUB.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vasub.vv vd, vs2, vs1, vm   # roundoff_signed(vs2[i] - vs1[i], 1)</div>\n<div>vasub.vx vd, vs2, rs1, vm   # roundoff_signed(vs2[i] - x[rs1], 1)</div>\n</div>",
                "tooltip": "vasub.vv vd, vs2, vs1, vm   # roundoff_signed(vs2[i] - vs1[i], 1)\nvasub.vx vd, vs2, rs1, vm   # roundoff_signed(vs2[i] - x[rs1], 1)",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_averaging_add_and_subtract"
            };

        case "VAADD.VI":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, simm5, vd</span>\n\n<div>For vaaddu , vaadd , and vasub , there can be no overflow in the result</div>\n<div>vaadd.vv vd, vs2, vs1, vm   # roundoff_signed(vs2[i] + vs1[i], 1)</div>\n<div>vaadd.vx vd, vs2, rs1, vm   # roundoff_signed(vs2[i] + x[rs1], 1)</div>\n</div>",
                "tooltip": "For vaaddu , vaadd , and vasub , there can be no overflow in the result\nvaadd.vv vd, vs2, vs1, vm   # roundoff_signed(vs2[i] + vs1[i], 1)\nvaadd.vx vd, vs2, rs1, vm   # roundoff_signed(vs2[i] + x[rs1], 1)",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_averaging_add_and_subtract"
            };

        case "VSLL.VX":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vsll.vv vd, vs2, vs1, vm   # Vector-vector</div>\n<div>vsll.vx vd, vs2, rs1, vm   # vector-scalar</div>\n<div>vsll.vi vd, vs2, uimm, vm   # vector-immediate</div>\n</div>",
                "tooltip": "vsll.vv vd, vs2, vs1, vm   # Vector-vector\nvsll.vx vd, vs2, rs1, vm   # vector-scalar\nvsll.vi vd, vs2, uimm, vm   # vector-immediate",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_bit_shift_instructions"
            };

        case "VSRA.VX":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vsra.vv vd, vs2, vs1, vm   # Vector-vector</div>\n<div>vsra.vx vd, vs2, rs1, vm   # vector-scalar</div>\n<div>vsra.vi vd, vs2, uimm, vm   # vector-immediate</div>\n</div>",
                "tooltip": "vsra.vv vd, vs2, vs1, vm   # Vector-vector\nvsra.vx vd, vs2, rs1, vm   # vector-scalar\nvsra.vi vd, vs2, uimm, vm   # vector-immediate",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_bit_shift_instructions"
            };

        case "VSLL.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vsll.vv vd, vs2, vs1, vm   # Vector-vector</div>\n<div>vsll.vx vd, vs2, rs1, vm   # vector-scalar</div>\n<div>vsll.vi vd, vs2, uimm, vm   # vector-immediate</div>\n</div>",
                "tooltip": "vsll.vv vd, vs2, vs1, vm   # Vector-vector\nvsll.vx vd, vs2, rs1, vm   # vector-scalar\nvsll.vi vd, vs2, uimm, vm   # vector-immediate",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_bit_shift_instructions"
            };

        case "VSRA.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vsra.vv vd, vs2, vs1, vm   # Vector-vector</div>\n<div>vsra.vx vd, vs2, rs1, vm   # vector-scalar</div>\n<div>vsra.vi vd, vs2, uimm, vm   # vector-immediate</div>\n</div>",
                "tooltip": "vsra.vv vd, vs2, vs1, vm   # Vector-vector\nvsra.vx vd, vs2, rs1, vm   # vector-scalar\nvsra.vi vd, vs2, uimm, vm   # vector-immediate",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_bit_shift_instructions"
            };

        case "VSLL.VI":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, simm5, vd</span>\n\n<div>vsll.vv vd, vs2, vs1, vm   # Vector-vector</div>\n<div>vsll.vx vd, vs2, rs1, vm   # vector-scalar</div>\n<div>vsll.vi vd, vs2, uimm, vm   # vector-immediate</div>\n</div>",
                "tooltip": "vsll.vv vd, vs2, vs1, vm   # Vector-vector\nvsll.vx vd, vs2, rs1, vm   # vector-scalar\nvsll.vi vd, vs2, uimm, vm   # vector-immediate",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_bit_shift_instructions"
            };

        case "VSRA.VI":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, simm5, vd</span>\n\n<div>vsra.vv vd, vs2, vs1, vm   # Vector-vector</div>\n<div>vsra.vx vd, vs2, rs1, vm   # vector-scalar</div>\n<div>vsra.vi vd, vs2, uimm, vm   # vector-immediate</div>\n</div>",
                "tooltip": "vsra.vv vd, vs2, vs1, vm   # Vector-vector\nvsra.vx vd, vs2, rs1, vm   # vector-scalar\nvsra.vi vd, vs2, uimm, vm   # vector-immediate",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_bit_shift_instructions"
            };

        case "VSMUL.VX":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vsmul.vv vd, vs2, vs1, vm  # vd[i] = clip(roundoff_signed(vs2[i]*vs1[i], SEW-1))</div>\n<div>vsmul.vx vd, vs2, rs1, vm  # vd[i] = clip(roundoff_signed(vs2[i]*x[rs1], SEW-1))</div>\n</div>",
                "tooltip": "vsmul.vv vd, vs2, vs1, vm  # vd[i] = clip(roundoff_signed(vs2[i]*vs1[i], SEW-1))\nvsmul.vx vd, vs2, rs1, vm  # vd[i] = clip(roundoff_signed(vs2[i]*x[rs1], SEW-1))",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_fractional_multiply_with_rounding_and_saturation"
            };

        case "VSMUL.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vsmul.vv vd, vs2, vs1, vm  # vd[i] = clip(roundoff_signed(vs2[i]*vs1[i], SEW-1))</div>\n<div>vsmul.vx vd, vs2, rs1, vm  # vd[i] = clip(roundoff_signed(vs2[i]*x[rs1], SEW-1))</div>\n</div>",
                "tooltip": "vsmul.vv vd, vs2, vs1, vm  # vd[i] = clip(roundoff_signed(vs2[i]*vs1[i], SEW-1))\nvsmul.vx vd, vs2, rs1, vm  # vd[i] = clip(roundoff_signed(vs2[i]*x[rs1], SEW-1))",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_fractional_multiply_with_rounding_and_saturation"
            };

        case "VSSRL.VX":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>The scaling right shifts have both zero-extending ( vssrl ) and sign-extending ( vssra ) forms</div>\n<div>vssrl.vv vd, vs2, vs1, vm   # vd[i] = roundoff_unsigned(vs2[i], vs1[i])</div>\n<div>vssrl.vx vd, vs2, rs1, vm   # vd[i] = roundoff_unsigned(vs2[i], x[rs1])</div>\n<div>vssrl.vi vd, vs2, uimm, vm   # vd[i] = roundoff_unsigned(vs2[i], uimm)</div>\n</div>",
                "tooltip": "The scaling right shifts have both zero-extending ( vssrl ) and sign-extending ( vssra ) forms\nvssrl.vv vd, vs2, vs1, vm   # vd[i] = roundoff_unsigned(vs2[i], vs1[i])\nvssrl.vx vd, vs2, rs1, vm   # vd[i] = roundoff_unsigned(vs2[i], x[rs1])\nvssrl.vi vd, vs2, uimm, vm   # vd[i] = roundoff_unsigned(vs2[i], uimm)",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_scaling_shift_instructions"
            };

        case "VSSRA.VX":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vssra.vv vd, vs2, vs1, vm   # vd[i] = roundoff_signed(vs2[i],vs1[i])</div>\n<div>vssra.vx vd, vs2, rs1, vm   # vd[i] = roundoff_signed(vs2[i], x[rs1])</div>\n<div>vssra.vi vd, vs2, uimm, vm   # vd[i] = roundoff_signed(vs2[i], uimm)</div>\n</div>",
                "tooltip": "vssra.vv vd, vs2, vs1, vm   # vd[i] = roundoff_signed(vs2[i],vs1[i])\nvssra.vx vd, vs2, rs1, vm   # vd[i] = roundoff_signed(vs2[i], x[rs1])\nvssra.vi vd, vs2, uimm, vm   # vd[i] = roundoff_signed(vs2[i], uimm)",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_scaling_shift_instructions"
            };

        case "VSSRL.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>The scaling right shifts have both zero-extending ( vssrl ) and sign-extending ( vssra ) forms</div>\n<div>vssrl.vv vd, vs2, vs1, vm   # vd[i] = roundoff_unsigned(vs2[i], vs1[i])</div>\n<div>vssrl.vx vd, vs2, rs1, vm   # vd[i] = roundoff_unsigned(vs2[i], x[rs1])</div>\n<div>vssrl.vi vd, vs2, uimm, vm   # vd[i] = roundoff_unsigned(vs2[i], uimm)</div>\n</div>",
                "tooltip": "The scaling right shifts have both zero-extending ( vssrl ) and sign-extending ( vssra ) forms\nvssrl.vv vd, vs2, vs1, vm   # vd[i] = roundoff_unsigned(vs2[i], vs1[i])\nvssrl.vx vd, vs2, rs1, vm   # vd[i] = roundoff_unsigned(vs2[i], x[rs1])\nvssrl.vi vd, vs2, uimm, vm   # vd[i] = roundoff_unsigned(vs2[i], uimm)",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_scaling_shift_instructions"
            };

        case "VSSRA.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vssra.vv vd, vs2, vs1, vm   # vd[i] = roundoff_signed(vs2[i],vs1[i])</div>\n<div>vssra.vx vd, vs2, rs1, vm   # vd[i] = roundoff_signed(vs2[i], x[rs1])</div>\n<div>vssra.vi vd, vs2, uimm, vm   # vd[i] = roundoff_signed(vs2[i], uimm)</div>\n</div>",
                "tooltip": "vssra.vv vd, vs2, vs1, vm   # vd[i] = roundoff_signed(vs2[i],vs1[i])\nvssra.vx vd, vs2, rs1, vm   # vd[i] = roundoff_signed(vs2[i], x[rs1])\nvssra.vi vd, vs2, uimm, vm   # vd[i] = roundoff_signed(vs2[i], uimm)",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_scaling_shift_instructions"
            };

        case "VSSRL.VI":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, simm5, vd</span>\n\n<div>The scaling right shifts have both zero-extending ( vssrl ) and sign-extending ( vssra ) forms</div>\n<div>vssrl.vv vd, vs2, vs1, vm   # vd[i] = roundoff_unsigned(vs2[i], vs1[i])</div>\n<div>vssrl.vx vd, vs2, rs1, vm   # vd[i] = roundoff_unsigned(vs2[i], x[rs1])</div>\n<div>vssrl.vi vd, vs2, uimm, vm   # vd[i] = roundoff_unsigned(vs2[i], uimm)</div>\n</div>",
                "tooltip": "The scaling right shifts have both zero-extending ( vssrl ) and sign-extending ( vssra ) forms\nvssrl.vv vd, vs2, vs1, vm   # vd[i] = roundoff_unsigned(vs2[i], vs1[i])\nvssrl.vx vd, vs2, rs1, vm   # vd[i] = roundoff_unsigned(vs2[i], x[rs1])\nvssrl.vi vd, vs2, uimm, vm   # vd[i] = roundoff_unsigned(vs2[i], uimm)",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_scaling_shift_instructions"
            };

        case "VSSRA.VI":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, simm5, vd</span>\n\n<div>vssra.vv vd, vs2, vs1, vm   # vd[i] = roundoff_signed(vs2[i],vs1[i])</div>\n<div>vssra.vx vd, vs2, rs1, vm   # vd[i] = roundoff_signed(vs2[i], x[rs1])</div>\n<div>vssra.vi vd, vs2, uimm, vm   # vd[i] = roundoff_signed(vs2[i], uimm)</div>\n</div>",
                "tooltip": "vssra.vv vd, vs2, vs1, vm   # vd[i] = roundoff_signed(vs2[i],vs1[i])\nvssra.vx vd, vs2, rs1, vm   # vd[i] = roundoff_signed(vs2[i], x[rs1])\nvssra.vi vd, vs2, uimm, vm   # vd[i] = roundoff_signed(vs2[i], uimm)",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_scaling_shift_instructions"
            };

        case "VNSRL.VX":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vnsrl.wv vd, vs2, vs1, vm   # vector-vector</div>\n<div>vnsrl.wx vd, vs2, rs1, vm   # vector-scalar</div>\n<div>vnsrl.wi vd, vs2, uimm, vm   # vector-immediate</div>\n</div>",
                "tooltip": "vnsrl.wv vd, vs2, vs1, vm   # vector-vector\nvnsrl.wx vd, vs2, rs1, vm   # vector-scalar\nvnsrl.wi vd, vs2, uimm, vm   # vector-immediate",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_narrowing_integer_right_shift_instructions"
            };

        case "VNSRL.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vnsrl.wv vd, vs2, vs1, vm   # vector-vector</div>\n<div>vnsrl.wx vd, vs2, rs1, vm   # vector-scalar</div>\n<div>vnsrl.wi vd, vs2, uimm, vm   # vector-immediate</div>\n</div>",
                "tooltip": "vnsrl.wv vd, vs2, vs1, vm   # vector-vector\nvnsrl.wx vd, vs2, rs1, vm   # vector-scalar\nvnsrl.wi vd, vs2, uimm, vm   # vector-immediate",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_narrowing_integer_right_shift_instructions"
            };

        case "VNSRL.VI":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, simm5, vd</span>\n\n<div>vnsrl.wv vd, vs2, vs1, vm   # vector-vector</div>\n<div>vnsrl.wx vd, vs2, rs1, vm   # vector-scalar</div>\n<div>vnsrl.wi vd, vs2, uimm, vm   # vector-immediate</div>\n</div>",
                "tooltip": "vnsrl.wv vd, vs2, vs1, vm   # vector-vector\nvnsrl.wx vd, vs2, rs1, vm   # vector-scalar\nvnsrl.wi vd, vs2, uimm, vm   # vector-immediate",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_narrowing_integer_right_shift_instructions"
            };

        case "VNSRA.VX":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>The double-width source vector register group is signified by a w in the source operand suffix (e.g., vnsra.wv )</div>\n</div>",
                "tooltip": "The double-width source vector register group is signified by a w in the source operand suffix (e.g., vnsra.wv )",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#sec-narrowing"
            };

        case "VNSRA.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>The double-width source vector register group is signified by a w in the source operand suffix (e.g., vnsra.wv )</div>\n</div>",
                "tooltip": "The double-width source vector register group is signified by a w in the source operand suffix (e.g., vnsra.wv )",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#sec-narrowing"
            };

        case "VNSRA.VI":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, simm5, vd</span>\n\n<div>The double-width source vector register group is signified by a w in the source operand suffix (e.g., vnsra.wv )</div>\n</div>",
                "tooltip": "The double-width source vector register group is signified by a w in the source operand suffix (e.g., vnsra.wv )",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#sec-narrowing"
            };

        case "VNCLIPU.VX":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>For vnclipu / vnclip , the rounding mode is specified in the vxrm For vnclipu , the shifted rounded source value is treated as an unsigned integer and saturates if the result would overflow the destination viewed as an unsigned integer.</div>\n<div>vnclipu.wv vd, vs2, vs1, vm   # vd[i] = clip(roundoff_unsigned(vs2[i], vs1[i]))</div>\n<div>vnclipu.wx vd, vs2, rs1, vm   # vd[i] = clip(roundoff_unsigned(vs2[i], x[rs1]))</div>\n<div>vnclipu.wi vd, vs2, uimm, vm  # vd[i] = clip(roundoff_unsigned(vs2[i], uimm5))</div>\n</div>",
                "tooltip": "For vnclipu / vnclip , the rounding mode is specified in the vxrm For vnclipu , the shifted rounded source value is treated as an unsigned integer and saturates if the result would overflow the destination viewed as an unsigned integer.\nvnclipu.wv vd, vs2, vs1, vm   # vd[i] = clip(roundoff_unsigned(vs2[i], vs1[i]))\nvnclipu.wx vd, vs2, rs1, vm   # vd[i] = clip(roundoff_unsigned(vs2[i], x[rs1]))\nvnclipu.wi vd, vs2, uimm, vm  # vd[i] = clip(roundoff_unsigned(vs2[i], uimm5))",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_narrowing_fixed_point_clip_instructions"
            };

        case "VNCLIP.VX":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>The vnclip instructions are used to pack a fixed-point value into a narrower destination</div>\n<div>For vnclip , the shifted rounded source value is treated as a signed integer and saturates if the result would overflow the destination viewed as a signed integer.</div>\n<div>vnclip.wv vd, vs2, vs1, vm   # vd[i] = clip(roundoff_signed(vs2[i], vs1[i]))</div>\n<div>vnclip.wx vd, vs2, rs1, vm   # vd[i] = clip(roundoff_signed(vs2[i], x[rs1]))</div>\n<div>vnclip.wi vd, vs2, uimm, vm  # vd[i] = clip(roundoff_signed(vs2[i], uimm5))</div>\n</div>",
                "tooltip": "The vnclip instructions are used to pack a fixed-point value into a narrower destination\nFor vnclip , the shifted rounded source value is treated as a signed integer and saturates if the result would overflow the destination viewed as a signed integer.\nvnclip.wv vd, vs2, vs1, vm   # vd[i] = clip(roundoff_signed(vs2[i], vs1[i]))\nvnclip.wx vd, vs2, rs1, vm   # vd[i] = clip(roundoff_signed(vs2[i], x[rs1]))\nvnclip.wi vd, vs2, uimm, vm  # vd[i] = clip(roundoff_signed(vs2[i], uimm5))",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_narrowing_fixed_point_clip_instructions"
            };

        case "VNCLIPU.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>For vnclipu / vnclip , the rounding mode is specified in the vxrm For vnclipu , the shifted rounded source value is treated as an unsigned integer and saturates if the result would overflow the destination viewed as an unsigned integer.</div>\n<div>vnclipu.wv vd, vs2, vs1, vm   # vd[i] = clip(roundoff_unsigned(vs2[i], vs1[i]))</div>\n<div>vnclipu.wx vd, vs2, rs1, vm   # vd[i] = clip(roundoff_unsigned(vs2[i], x[rs1]))</div>\n<div>vnclipu.wi vd, vs2, uimm, vm  # vd[i] = clip(roundoff_unsigned(vs2[i], uimm5))</div>\n</div>",
                "tooltip": "For vnclipu / vnclip , the rounding mode is specified in the vxrm For vnclipu , the shifted rounded source value is treated as an unsigned integer and saturates if the result would overflow the destination viewed as an unsigned integer.\nvnclipu.wv vd, vs2, vs1, vm   # vd[i] = clip(roundoff_unsigned(vs2[i], vs1[i]))\nvnclipu.wx vd, vs2, rs1, vm   # vd[i] = clip(roundoff_unsigned(vs2[i], x[rs1]))\nvnclipu.wi vd, vs2, uimm, vm  # vd[i] = clip(roundoff_unsigned(vs2[i], uimm5))",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_narrowing_fixed_point_clip_instructions"
            };

        case "VNCLIP.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>The vnclip instructions are used to pack a fixed-point value into a narrower destination</div>\n<div>For vnclip , the shifted rounded source value is treated as a signed integer and saturates if the result would overflow the destination viewed as a signed integer.</div>\n<div>vnclip.wv vd, vs2, vs1, vm   # vd[i] = clip(roundoff_signed(vs2[i], vs1[i]))</div>\n<div>vnclip.wx vd, vs2, rs1, vm   # vd[i] = clip(roundoff_signed(vs2[i], x[rs1]))</div>\n<div>vnclip.wi vd, vs2, uimm, vm  # vd[i] = clip(roundoff_signed(vs2[i], uimm5))</div>\n</div>",
                "tooltip": "The vnclip instructions are used to pack a fixed-point value into a narrower destination\nFor vnclip , the shifted rounded source value is treated as a signed integer and saturates if the result would overflow the destination viewed as a signed integer.\nvnclip.wv vd, vs2, vs1, vm   # vd[i] = clip(roundoff_signed(vs2[i], vs1[i]))\nvnclip.wx vd, vs2, rs1, vm   # vd[i] = clip(roundoff_signed(vs2[i], x[rs1]))\nvnclip.wi vd, vs2, uimm, vm  # vd[i] = clip(roundoff_signed(vs2[i], uimm5))",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_narrowing_fixed_point_clip_instructions"
            };

        case "VNCLIPU.VI":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, simm5, vd</span>\n\n<div>For vnclipu / vnclip , the rounding mode is specified in the vxrm For vnclipu , the shifted rounded source value is treated as an unsigned integer and saturates if the result would overflow the destination viewed as an unsigned integer.</div>\n<div>vnclipu.wv vd, vs2, vs1, vm   # vd[i] = clip(roundoff_unsigned(vs2[i], vs1[i]))</div>\n<div>vnclipu.wx vd, vs2, rs1, vm   # vd[i] = clip(roundoff_unsigned(vs2[i], x[rs1]))</div>\n<div>vnclipu.wi vd, vs2, uimm, vm  # vd[i] = clip(roundoff_unsigned(vs2[i], uimm5))</div>\n</div>",
                "tooltip": "For vnclipu / vnclip , the rounding mode is specified in the vxrm For vnclipu , the shifted rounded source value is treated as an unsigned integer and saturates if the result would overflow the destination viewed as an unsigned integer.\nvnclipu.wv vd, vs2, vs1, vm   # vd[i] = clip(roundoff_unsigned(vs2[i], vs1[i]))\nvnclipu.wx vd, vs2, rs1, vm   # vd[i] = clip(roundoff_unsigned(vs2[i], x[rs1]))\nvnclipu.wi vd, vs2, uimm, vm  # vd[i] = clip(roundoff_unsigned(vs2[i], uimm5))",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_narrowing_fixed_point_clip_instructions"
            };

        case "VNCLIP.VI":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, simm5, vd</span>\n\n<div>The vnclip instructions are used to pack a fixed-point value into a narrower destination</div>\n<div>For vnclip , the shifted rounded source value is treated as a signed integer and saturates if the result would overflow the destination viewed as a signed integer.</div>\n<div>vnclip.wv vd, vs2, vs1, vm   # vd[i] = clip(roundoff_signed(vs2[i], vs1[i]))</div>\n<div>vnclip.wx vd, vs2, rs1, vm   # vd[i] = clip(roundoff_signed(vs2[i], x[rs1]))</div>\n<div>vnclip.wi vd, vs2, uimm, vm  # vd[i] = clip(roundoff_signed(vs2[i], uimm5))</div>\n</div>",
                "tooltip": "The vnclip instructions are used to pack a fixed-point value into a narrower destination\nFor vnclip , the shifted rounded source value is treated as a signed integer and saturates if the result would overflow the destination viewed as a signed integer.\nvnclip.wv vd, vs2, vs1, vm   # vd[i] = clip(roundoff_signed(vs2[i], vs1[i]))\nvnclip.wx vd, vs2, rs1, vm   # vd[i] = clip(roundoff_signed(vs2[i], x[rs1]))\nvnclip.wi vd, vs2, uimm, vm  # vd[i] = clip(roundoff_signed(vs2[i], uimm5))",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_narrowing_fixed_point_clip_instructions"
            };

        case "VWREDSUMU.VS":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>The unsigned vwredsumu.vs instruction zero-extends the SEW-wide vector elements before summing them, then adds the 2*SEW-width scalar element, and stores the result in a 2*SEW-width scalar element.</div>\n<div>vwredsumu.vs vd, vs2, vs1, vm   # 2*SEW = 2*SEW + sum(zero-extend(SEW))</div>\n</div>",
                "tooltip": "The unsigned vwredsumu.vs instruction zero-extends the SEW-wide vector elements before summing them, then adds the 2*SEW-width scalar element, and stores the result in a 2*SEW-width scalar element.\nvwredsumu.vs vd, vs2, vs1, vm   # 2*SEW = 2*SEW + sum(zero-extend(SEW))",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_widening_integer_reduction_instructions"
            };

        case "VWREDSUM.VS":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>The vwredsum.vs instruction sign-extends the SEW-wide vector elements before summing them.</div>\n<div>vwredsum.vs  vd, vs2, vs1, vm   # 2*SEW = 2*SEW + sum(sign-extend(SEW))</div>\n</div>",
                "tooltip": "The vwredsum.vs instruction sign-extends the SEW-wide vector elements before summing them.\nvwredsum.vs  vd, vs2, vs1, vm   # 2*SEW = 2*SEW + sum(sign-extend(SEW))",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_widening_integer_reduction_instructions"
            };

        case "VDOTU.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vdotu.vv vd, vs2, vs1, vm  # Vector-vector</div>\n</div>",
                "tooltip": "vdotu.vv vd, vs2, vs1, vm  # Vector-vector",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_integer_dot_product_instruction"
            };

        case "VDOT.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>The integer dot-product reduction vdot.vv performs an element-wise multiplication between the source sub-elements then accumulates the results into the destination vector element</div>\n<div>vdot.vv vd, vs2, vs1, vm   # Vector-vector</div>\n<div>vdot.vv  vd, vs2, vs1, vm   # vd[i][31:0] += vs2[i][31:0] * vs1[i][31:0]</div>\n<div>vdot.vv vd, vs2, vs1, vm # vd[i][31:0] += vs2[i][31:16] * vs1[i][31:16]</div>\n<div>vdot.vv vd, vs2, vs1, vm # vd[i][31:0] += vs2[i][31:24] * vs1[i][31:24]</div>\n</div>",
                "tooltip": "The integer dot-product reduction vdot.vv performs an element-wise multiplication between the source sub-elements then accumulates the results into the destination vector element\nvdot.vv vd, vs2, vs1, vm   # Vector-vector\nvdot.vv  vd, vs2, vs1, vm   # vd[i][31:0] += vs2[i][31:0] * vs1[i][31:0]\nvdot.vv vd, vs2, vs1, vm # vd[i][31:0] += vs2[i][31:16] * vs1[i][31:16]\nvdot.vv vd, vs2, vs1, vm # vd[i][31:0] += vs2[i][31:24] * vs1[i][31:24]",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_integer_dot_product_instruction"
            };

        case "VREDSUM.VS":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n<div>vredsum.vs  vd, vs2, vs1, vm   # vd[0] =  sum( vs1[0] , vs2[*] )</div>\n</div>",
                "tooltip": "vredsum.vs  vd, vs2, vs1, vm   # vd[0] =  sum( vs1[0] , vs2[*] )",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_integer_reduction_instructions"
            };

        case "VREDAND.VS":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n<div>vredand.vs  vd, vs2, vs1, vm   # vd[0] =  and( vs1[0] , vs2[*] )</div>\n</div>",
                "tooltip": "vredand.vs  vd, vs2, vs1, vm   # vd[0] =  and( vs1[0] , vs2[*] )",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_integer_reduction_instructions"
            };

        case "VREDOR.VS":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n<div>vredor.vs   vd, vs2, vs1, vm   # vd[0] =   or( vs1[0] , vs2[*] )</div>\n</div>",
                "tooltip": "vredor.vs   vd, vs2, vs1, vm   # vd[0] =   or( vs1[0] , vs2[*] )",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_integer_reduction_instructions"
            };

        case "VREDXOR.VS":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n<div>vredxor.vs  vd, vs2, vs1, vm   # vd[0] =  xor( vs1[0] , vs2[*] )</div>\n</div>",
                "tooltip": "vredxor.vs  vd, vs2, vs1, vm   # vd[0] =  xor( vs1[0] , vs2[*] )",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_integer_reduction_instructions"
            };

        case "VREDMINU.VS":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n<div>vredminu.vs vd, vs2, vs1, vm   # vd[0] = minu( vs1[0] , vs2[*] )</div>\n</div>",
                "tooltip": "vredminu.vs vd, vs2, vs1, vm   # vd[0] = minu( vs1[0] , vs2[*] )",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_integer_reduction_instructions"
            };

        case "VREDMIN.VS":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n<div>vredmin.vs  vd, vs2, vs1, vm   # vd[0] =  min( vs1[0] , vs2[*] )</div>\n</div>",
                "tooltip": "vredmin.vs  vd, vs2, vs1, vm   # vd[0] =  min( vs1[0] , vs2[*] )",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_integer_reduction_instructions"
            };

        case "VREDMAXU.VS":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n<div>vredmaxu.vs vd, vs2, vs1, vm   # vd[0] = maxu( vs1[0] , vs2[*] )</div>\n</div>",
                "tooltip": "vredmaxu.vs vd, vs2, vs1, vm   # vd[0] = maxu( vs1[0] , vs2[*] )",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_integer_reduction_instructions"
            };

        case "VREDMAX.VS":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n<div>vredmax.vs  vd, vs2, vs1, vm   # vd[0] =  max( vs1[0] , vs2[*] )</div>\n</div>",
                "tooltip": "vredmax.vs  vd, vs2, vs1, vm   # vd[0] =  max( vs1[0] , vs2[*] )",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_integer_reduction_instructions"
            };

        case "VCOMPRESS.VM":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n<div>vcompress is encoded as an unmasked instruction ( vm=1 )</div>\n<div>A trap on a vcompress instruction is always reported with a vstart of 0. Executing a vcompress instruction with a non-zero vstart raises an illegal instruction exception.</div>\n<div>Note vcompress is one of the more difficult instructions to restart with a non-zero vstart , so assumption is implementations will choose not do that but will instead restart from element 0. This does mean elements in destination register after vstart will already have been updated</div>\n<div>vcompress.vm vd, vs2, vs1  # Compress into vd elements of vs2 where vs1 is enabled</div>\n<div>Example use of vcompress instruction</div>\n<div>vcompress.vm v2, v1, v0</div>\n</div>",
                "tooltip": "vcompress is encoded as an unmasked instruction ( vm=1 )\nA trap on a vcompress instruction is always reported with a vstart of 0. Executing a vcompress instruction with a non-zero vstart raises an illegal instruction exception.\nNote vcompress is one of the more difficult instructions to restart with a non-zero vstart , so assumption is implementations will choose not do that but will instead restart from element 0. This does mean elements in destination register after vstart will already have been updated\nvcompress.vm vd, vs2, vs1  # Compress into vd elements of vs2 where vs1 is enabled\nExample use of vcompress instruction\nvcompress.vm v2, v1, v0",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_compress_instruction"
            };

        case "VMANDNOT.MM":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n<div>expansion: vmslt{u}.vx vt, va, x;  vmandnot.mm vd, vd, vt</div>\n</div>",
                "tooltip": "expansion: vmslt{u}.vx vt, va, x;  vmandnot.mm vd, vd, vt",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_integer_comparison_instructions"
            };

        case "VMXOR.MM":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n<div>expansion: vmslt{u}.vx vd, va, x, v0.t; vmxor.mm vd, vd, v0</div>\n</div>",
                "tooltip": "expansion: vmslt{u}.vx vd, va, x, v0.t; vmxor.mm vd, vd, v0",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_integer_comparison_instructions"
            };

        case "VMNAND.MM":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n<div>expansion: vmslt{u}.vx vd, va, x; vmnand.mm vd, vd, vd</div>\n</div>",
                "tooltip": "expansion: vmslt{u}.vx vd, va, x; vmnand.mm vd, vd, vd",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_integer_comparison_instructions"
            };

        case "VMAND.MM":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n<div>Note vmfeq vmand instruction, but this more efficient sequence incorrectly fails to raise the invalid exception when an element of va contains a quiet NaN and the corresponding element in vb contains a signaling NaN</div>\n<div>vmand.mm v0, v0, v1        # Only set where A and B are ordered,</div>\n</div>",
                "tooltip": "Note vmfeq vmand instruction, but this more efficient sequence incorrectly fails to raise the invalid exception when an element of va contains a quiet NaN and the corresponding element in vb contains a signaling NaN\nvmand.mm v0, v0, v1        # Only set where A and B are ordered,",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_floating_point_compare_instructions"
            };

        case "VMOR.MM":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n<div>vmor.mm  vd, vs2, vs1     # vd[i] =   vs2[i].LSB ||  vs1[i].LSB</div>\n</div>",
                "tooltip": "vmor.mm  vd, vs2, vs1     # vd[i] =   vs2[i].LSB ||  vs1[i].LSB",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#sec-mask-register-logical"
            };

        case "VMORNOT.MM":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n<div>vmornot.mm vd, src2, src1</div>\n<div>vmornot.mm vd, src1, src2</div>\n<div>vmornot.mm  vd, vs2, vs1  # vd[i] =   vs2[i].LSB || !vs1[i].LSB</div>\n</div>",
                "tooltip": "vmornot.mm vd, src2, src1\nvmornot.mm vd, src1, src2\nvmornot.mm  vd, vs2, vs1  # vd[i] =   vs2[i].LSB || !vs1[i].LSB",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#sec-mask-register-logical"
            };

        case "VMNOR.MM":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n<div>vmnor.mm vd, src1, src2</div>\n<div>vmnor.mm  vd, vs2, vs1    # vd[i] = !(vs2[i[.LSB ||  vs1[i].LSB)</div>\n</div>",
                "tooltip": "vmnor.mm vd, src1, src2\nvmnor.mm  vd, vs2, vs1    # vd[i] = !(vs2[i[.LSB ||  vs1[i].LSB)",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#sec-mask-register-logical"
            };

        case "VMXNOR.MM":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n<div>vmxnor.mm vd, src1, src2</div>\n<div>vmxnor.mm vd, vd, vd</div>\n<div>vmxnor.mm vd, vs2, vs1    # vd[i] = !(vs2[i].LSB ^^  vs1[i].LSB)</div>\n<div>vmset.m vd     =&gt; vmxnor.mm vd, vd, vd  # Set mask register</div>\n</div>",
                "tooltip": "vmxnor.mm vd, src1, src2\nvmxnor.mm vd, vd, vd\nvmxnor.mm vd, vs2, vs1    # vd[i] = !(vs2[i].LSB ^^  vs1[i].LSB)\nvmset.m vd     => vmxnor.mm vd, vd, vd  # Set mask register",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#sec-mask-register-logical"
            };

        case "VMSBF.M":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vd</span>\n\n<div>vmsbf.m</div>\n</div>",
                "tooltip": "vmsbf.m",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#__code_vfirst_code_find_first_set_mask_bit"
            };

        case "VMSOF.M":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vd</span>\n\n<div>vmsof.m</div>\n</div>",
                "tooltip": "vmsof.m",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#__code_vmsif_m_code_set_including_first_mask_bit"
            };

        case "VMSIF.M":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vd</span>\n\n<div>vmsif.m</div>\n</div>",
                "tooltip": "vmsif.m",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#__code_vmsbf_m_code_set_before_first_mask_bit"
            };

        case "VIOTA.M":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vd</span>\n\n<div>The viota.m instruction reads a source vector mask register and writes to each element of the destination vector register group the sum of all the least-significant bits of elements in the mask register whose index is less than the element, e.g., a parallel prefix sum of the mask values.</div>\n<div>Traps on viota.m are always reported with a vstart of 0, and execution is always restarted from the beginning when resuming after a trap handler</div>\n<div>The viota.m instruction can be combined with memory scatter instructions (indexed stores) to perform vector compress functions.</div>\n<div>viota.m vd, vs2, vm</div>\n<div>viota.m v4, v2 # Unmasked</div>\n<div>viota.m v4, v2, v0.t # Masked</div>\n<div>viota.m v16, v0               # Get destination offsets of active elements</div>\n</div>",
                "tooltip": "The viota.m instruction reads a source vector mask register and writes to each element of the destination vector register group the sum of all the least-significant bits of elements in the mask register whose index is less than the element, e.g., a parallel prefix sum of the mask values.\nTraps on viota.m are always reported with a vstart of 0, and execution is always restarted from the beginning when resuming after a trap handler\nThe viota.m instruction can be combined with memory scatter instructions (indexed stores) to perform vector compress functions.\nviota.m vd, vs2, vm\nviota.m v4, v2 # Unmasked\nviota.m v4, v2, v0.t # Masked\nviota.m v16, v0               # Get destination offsets of active elements",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_iota_instruction"
            };

        case "VID.V":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vd</span>\n\n<div>The vid.v instruction writes each element\u00e2\u0080\u0099s index to the destination vector register group, from 0 to vl -1.</div>\n<div>Note vid.v instruction using the same datapath as viota.m but with an implicit set mask source</div>\n<div>vid.v vd, vm  # Write element ID to destination.</div>\n</div>",
                "tooltip": "The vid.v instruction writes each element\u00e2\u0080\u0099s index to the destination vector register group, from 0 to vl -1.\nNote vid.v instruction using the same datapath as viota.m but with an implicit set mask source\nvid.v vd, vm  # Write element ID to destination.",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_element_index_instruction"
            };

        case "VDIVU.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n<div>vdivu.vv vd, vs2, vs1, vm   # Vector-vector</div>\n<div>vdivu.vx vd, vs2, rs1, vm   # vector-scalar</div>\n</div>",
                "tooltip": "vdivu.vv vd, vs2, vs1, vm   # Vector-vector\nvdivu.vx vd, vs2, rs1, vm   # vector-scalar",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_integer_divide_instructions"
            };

        case "VDIV.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n<div>vdiv.vv vd, vs2, vs1, vm   # Vector-vector</div>\n<div>vdiv.vx vd, vs2, rs1, vm   # vector-scalar</div>\n</div>",
                "tooltip": "vdiv.vv vd, vs2, vs1, vm   # Vector-vector\nvdiv.vx vd, vs2, rs1, vm   # vector-scalar",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_integer_divide_instructions"
            };

        case "VREMU.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n<div>vremu.vv vd, vs2, vs1, vm   # Vector-vector</div>\n<div>vremu.vx vd, vs2, rs1, vm   # vector-scalar</div>\n</div>",
                "tooltip": "vremu.vv vd, vs2, vs1, vm   # Vector-vector\nvremu.vx vd, vs2, rs1, vm   # vector-scalar",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_integer_divide_instructions"
            };

        case "VREM.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n<div>vrem.vv vd, vs2, vs1, vm   # Vector-vector</div>\n<div>vrem.vx vd, vs2, rs1, vm   # vector-scalar</div>\n</div>",
                "tooltip": "vrem.vv vd, vs2, vs1, vm   # Vector-vector\nvrem.vx vd, vs2, rs1, vm   # vector-scalar",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_integer_divide_instructions"
            };

        case "VDIVU.VX":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vdivu.vv vd, vs2, vs1, vm   # Vector-vector</div>\n<div>vdivu.vx vd, vs2, rs1, vm   # vector-scalar</div>\n</div>",
                "tooltip": "vdivu.vv vd, vs2, vs1, vm   # Vector-vector\nvdivu.vx vd, vs2, rs1, vm   # vector-scalar",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_integer_divide_instructions"
            };

        case "VDIV.VX":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vdiv.vv vd, vs2, vs1, vm   # Vector-vector</div>\n<div>vdiv.vx vd, vs2, rs1, vm   # vector-scalar</div>\n</div>",
                "tooltip": "vdiv.vv vd, vs2, vs1, vm   # Vector-vector\nvdiv.vx vd, vs2, rs1, vm   # vector-scalar",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_integer_divide_instructions"
            };

        case "VREMU.VX":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vremu.vv vd, vs2, vs1, vm   # Vector-vector</div>\n<div>vremu.vx vd, vs2, rs1, vm   # vector-scalar</div>\n</div>",
                "tooltip": "vremu.vv vd, vs2, vs1, vm   # Vector-vector\nvremu.vx vd, vs2, rs1, vm   # vector-scalar",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_integer_divide_instructions"
            };

        case "VREM.VX":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vrem.vv vd, vs2, vs1, vm   # Vector-vector</div>\n<div>vrem.vx vd, vs2, rs1, vm   # vector-scalar</div>\n</div>",
                "tooltip": "vrem.vv vd, vs2, vs1, vm   # Vector-vector\nvrem.vx vd, vs2, rs1, vm   # vector-scalar",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_integer_divide_instructions"
            };

        case "VMULHU.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n<div>vmulhu.vv vd, vs2, vs1, vm   # Vector-vector</div>\n<div>vmulhu.vx vd, vs2, rs1, vm   # vector-scalar</div>\n</div>",
                "tooltip": "vmulhu.vv vd, vs2, vs1, vm   # Vector-vector\nvmulhu.vx vd, vs2, rs1, vm   # vector-scalar",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_integer_multiply_instructions"
            };

        case "VMULHSU.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n<div>vmulhsu.vv vd, vs2, vs1, vm   # Vector-vector</div>\n<div>vmulhsu.vx vd, vs2, rs1, vm   # vector-scalar</div>\n</div>",
                "tooltip": "vmulhsu.vv vd, vs2, vs1, vm   # Vector-vector\nvmulhsu.vx vd, vs2, rs1, vm   # vector-scalar",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_integer_multiply_instructions"
            };

        case "VMULH.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n<div>Note vmulh* opcodes perform simple fractional multiplies, but with no option to scale, round, and/or saturate the result</div>\n<div>Can consider changing definition of vmulh , vmulhu , vmulhsu to use vxrm rounding mode when discarding low half of product</div>\n<div>vmulh.vv vd, vs2, vs1, vm   # Vector-vector</div>\n<div>vmulh.vx vd, vs2, rs1, vm   # vector-scalar</div>\n</div>",
                "tooltip": "Note vmulh* opcodes perform simple fractional multiplies, but with no option to scale, round, and/or saturate the result\nCan consider changing definition of vmulh , vmulhu , vmulhsu to use vxrm rounding mode when discarding low half of product\nvmulh.vv vd, vs2, vs1, vm   # Vector-vector\nvmulh.vx vd, vs2, rs1, vm   # vector-scalar",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_integer_multiply_instructions"
            };

        case "VMULHU.VX":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vmulhu.vv vd, vs2, vs1, vm   # Vector-vector</div>\n<div>vmulhu.vx vd, vs2, rs1, vm   # vector-scalar</div>\n</div>",
                "tooltip": "vmulhu.vv vd, vs2, vs1, vm   # Vector-vector\nvmulhu.vx vd, vs2, rs1, vm   # vector-scalar",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_integer_multiply_instructions"
            };

        case "VMULHSU.VX":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vmulhsu.vv vd, vs2, vs1, vm   # Vector-vector</div>\n<div>vmulhsu.vx vd, vs2, rs1, vm   # vector-scalar</div>\n</div>",
                "tooltip": "vmulhsu.vv vd, vs2, vs1, vm   # Vector-vector\nvmulhsu.vx vd, vs2, rs1, vm   # vector-scalar",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_integer_multiply_instructions"
            };

        case "VMULH.VX":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>Note vmulh* opcodes perform simple fractional multiplies, but with no option to scale, round, and/or saturate the result</div>\n<div>Can consider changing definition of vmulh , vmulhu , vmulhsu to use vxrm rounding mode when discarding low half of product</div>\n<div>vmulh.vv vd, vs2, vs1, vm   # Vector-vector</div>\n<div>vmulh.vx vd, vs2, rs1, vm   # vector-scalar</div>\n</div>",
                "tooltip": "Note vmulh* opcodes perform simple fractional multiplies, but with no option to scale, round, and/or saturate the result\nCan consider changing definition of vmulh , vmulhu , vmulhsu to use vxrm rounding mode when discarding low half of product\nvmulh.vv vd, vs2, vs1, vm   # Vector-vector\nvmulh.vx vd, vs2, rs1, vm   # vector-scalar",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_integer_multiply_instructions"
            };

        case "VMADD.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n<div>vmadd.vv vd, vs1, vs2, vm    # vd[i] = (vs1[i] * vd[i]) + vs2[i]</div>\n<div>vmadd.vx vd, rs1, vs2, vm    # vd[i] = (x[rs1] * vd[i]) + vs2[i]</div>\n</div>",
                "tooltip": "vmadd.vv vd, vs1, vs2, vm    # vd[i] = (vs1[i] * vd[i]) + vs2[i]\nvmadd.vx vd, rs1, vs2, vm    # vd[i] = (x[rs1] * vd[i]) + vs2[i]",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_integer_multiply_add_instructions"
            };

        case "VNMSUB.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n<div>Similarly for the \"vnmsub\" opcode</div>\n<div>vnmsub.vv vd, vs1, vs2, vm    # vd[i] = -(vs1[i] * vd[i]) + vs2[i]</div>\n<div>vnmsub.vx vd, rs1, vs2, vm    # vd[i] = -(x[rs1] * vd[i]) + vs2[i]</div>\n</div>",
                "tooltip": "Similarly for the \"vnmsub\" opcode\nvnmsub.vv vd, vs1, vs2, vm    # vd[i] = -(vs1[i] * vd[i]) + vs2[i]\nvnmsub.vx vd, rs1, vs2, vm    # vd[i] = -(x[rs1] * vd[i]) + vs2[i]",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_integer_multiply_add_instructions"
            };

        case "VMACC.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n<div>The integer multiply-add instructions are destructive and are provided in two forms, one that overwrites the addend or minuend ( vmacc , vnmsac ) and one that overwrites the first multiplicand ( vmadd , vnmsub ).</div>\n<div>vmacc.vv vd, vs1, vs2, vm    # vd[i] = +(vs1[i] * vs2[i]) + vd[i]</div>\n<div>vmacc.vx vd, rs1, vs2, vm    # vd[i] = +(x[rs1] * vs2[i]) + vd[i]</div>\n</div>",
                "tooltip": "The integer multiply-add instructions are destructive and are provided in two forms, one that overwrites the addend or minuend ( vmacc , vnmsac ) and one that overwrites the first multiplicand ( vmadd , vnmsub ).\nvmacc.vv vd, vs1, vs2, vm    # vd[i] = +(vs1[i] * vs2[i]) + vd[i]\nvmacc.vx vd, rs1, vs2, vm    # vd[i] = +(x[rs1] * vs2[i]) + vd[i]",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_integer_multiply_add_instructions"
            };

        case "VNMSAC.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n<div>vnmsac.vv vd, vs1, vs2, vm    # vd[i] = -(vs1[i] * vs2[i]) + vd[i]</div>\n<div>vnmsac.vx vd, rs1, vs2, vm    # vd[i] = -(x[rs1] * vs2[i]) + vd[i]</div>\n</div>",
                "tooltip": "vnmsac.vv vd, vs1, vs2, vm    # vd[i] = -(vs1[i] * vs2[i]) + vd[i]\nvnmsac.vx vd, rs1, vs2, vm    # vd[i] = -(x[rs1] * vs2[i]) + vd[i]",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_integer_multiply_add_instructions"
            };

        case "VMADD.VX":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vmadd.vv vd, vs1, vs2, vm    # vd[i] = (vs1[i] * vd[i]) + vs2[i]</div>\n<div>vmadd.vx vd, rs1, vs2, vm    # vd[i] = (x[rs1] * vd[i]) + vs2[i]</div>\n</div>",
                "tooltip": "vmadd.vv vd, vs1, vs2, vm    # vd[i] = (vs1[i] * vd[i]) + vs2[i]\nvmadd.vx vd, rs1, vs2, vm    # vd[i] = (x[rs1] * vd[i]) + vs2[i]",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_integer_multiply_add_instructions"
            };

        case "VNMSUB.VX":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>Similarly for the \"vnmsub\" opcode</div>\n<div>vnmsub.vv vd, vs1, vs2, vm    # vd[i] = -(vs1[i] * vd[i]) + vs2[i]</div>\n<div>vnmsub.vx vd, rs1, vs2, vm    # vd[i] = -(x[rs1] * vd[i]) + vs2[i]</div>\n</div>",
                "tooltip": "Similarly for the \"vnmsub\" opcode\nvnmsub.vv vd, vs1, vs2, vm    # vd[i] = -(vs1[i] * vd[i]) + vs2[i]\nvnmsub.vx vd, rs1, vs2, vm    # vd[i] = -(x[rs1] * vd[i]) + vs2[i]",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_integer_multiply_add_instructions"
            };

        case "VMACC.VX":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>The integer multiply-add instructions are destructive and are provided in two forms, one that overwrites the addend or minuend ( vmacc , vnmsac ) and one that overwrites the first multiplicand ( vmadd , vnmsub ).</div>\n<div>vmacc.vv vd, vs1, vs2, vm    # vd[i] = +(vs1[i] * vs2[i]) + vd[i]</div>\n<div>vmacc.vx vd, rs1, vs2, vm    # vd[i] = +(x[rs1] * vs2[i]) + vd[i]</div>\n</div>",
                "tooltip": "The integer multiply-add instructions are destructive and are provided in two forms, one that overwrites the addend or minuend ( vmacc , vnmsac ) and one that overwrites the first multiplicand ( vmadd , vnmsub ).\nvmacc.vv vd, vs1, vs2, vm    # vd[i] = +(vs1[i] * vs2[i]) + vd[i]\nvmacc.vx vd, rs1, vs2, vm    # vd[i] = +(x[rs1] * vs2[i]) + vd[i]",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_integer_multiply_add_instructions"
            };

        case "VNMSAC.VX":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vnmsac.vv vd, vs1, vs2, vm    # vd[i] = -(vs1[i] * vs2[i]) + vd[i]</div>\n<div>vnmsac.vx vd, rs1, vs2, vm    # vd[i] = -(x[rs1] * vs2[i]) + vd[i]</div>\n</div>",
                "tooltip": "vnmsac.vv vd, vs1, vs2, vm    # vd[i] = -(vs1[i] * vs2[i]) + vd[i]\nvnmsac.vx vd, rs1, vs2, vm    # vd[i] = -(x[rs1] * vs2[i]) + vd[i]",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_single_width_integer_multiply_add_instructions"
            };

        case "VWADDU.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n<div>vwaddu.vv  vd, vs2, vs1, vm  # vector-vector</div>\n<div>vwaddu.vx  vd, vs2, rs1, vm  # vector-scalar</div>\n<div>vwaddu.wv  vd, vs2, vs1, vm  # vector-vector</div>\n<div>vwaddu.wx  vd, vs2, rs1, vm  # vector-scalar</div>\n</div>",
                "tooltip": "vwaddu.vv  vd, vs2, vs1, vm  # vector-vector\nvwaddu.vx  vd, vs2, rs1, vm  # vector-scalar\nvwaddu.wv  vd, vs2, vs1, vm  # vector-vector\nvwaddu.wx  vd, vs2, rs1, vm  # vector-scalar",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_widening_integer_add_subtract"
            };

        case "VWADD.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n<div>Can define assembly pseudoinstructions vwcvt.x.x.v vd,vs,vm = vwadd.vx vd,vs,x0,vm and vwcvtu.x.x.v vd,vs,vm = vwaddu.vx vd,vs,x0,vm</div>\n<div>vwadd.vv  vd, vs2, vs1, vm  # vector-vector</div>\n<div>vwadd.vx  vd, vs2, rs1, vm  # vector-scalar</div>\n<div>vwadd.wv  vd, vs2, vs1, vm  # vector-vector</div>\n<div>vwadd.wx  vd, vs2, rs1, vm  # vector-scalar</div>\n</div>",
                "tooltip": "Can define assembly pseudoinstructions vwcvt.x.x.v vd,vs,vm = vwadd.vx vd,vs,x0,vm and vwcvtu.x.x.v vd,vs,vm = vwaddu.vx vd,vs,x0,vm\nvwadd.vv  vd, vs2, vs1, vm  # vector-vector\nvwadd.vx  vd, vs2, rs1, vm  # vector-scalar\nvwadd.wv  vd, vs2, vs1, vm  # vector-vector\nvwadd.wx  vd, vs2, rs1, vm  # vector-scalar",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_widening_integer_add_subtract"
            };

        case "VWSUBU.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n<div>vwsubu.vv  vd, vs2, vs1, vm  # vector-vector</div>\n<div>vwsubu.vx  vd, vs2, rs1, vm  # vector-scalar</div>\n<div>vwsubu.wv  vd, vs2, vs1, vm  # vector-vector</div>\n<div>vwsubu.wx  vd, vs2, rs1, vm  # vector-scalar</div>\n</div>",
                "tooltip": "vwsubu.vv  vd, vs2, vs1, vm  # vector-vector\nvwsubu.vx  vd, vs2, rs1, vm  # vector-scalar\nvwsubu.wv  vd, vs2, vs1, vm  # vector-vector\nvwsubu.wx  vd, vs2, rs1, vm  # vector-scalar",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_widening_integer_add_subtract"
            };

        case "VWSUB.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n<div>vwsub.vv  vd, vs2, vs1, vm  # vector-vector</div>\n<div>vwsub.vx  vd, vs2, rs1, vm  # vector-scalar</div>\n<div>vwsub.wv  vd, vs2, vs1, vm  # vector-vector</div>\n<div>vwsub.wx  vd, vs2, rs1, vm  # vector-scalar</div>\n</div>",
                "tooltip": "vwsub.vv  vd, vs2, vs1, vm  # vector-vector\nvwsub.vx  vd, vs2, rs1, vm  # vector-scalar\nvwsub.wv  vd, vs2, vs1, vm  # vector-vector\nvwsub.wx  vd, vs2, rs1, vm  # vector-scalar",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_widening_integer_add_subtract"
            };

        case "VWADDU.WV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n<div>vwaddu.vv  vd, vs2, vs1, vm  # vector-vector</div>\n<div>vwaddu.vx  vd, vs2, rs1, vm  # vector-scalar</div>\n<div>vwaddu.wv  vd, vs2, vs1, vm  # vector-vector</div>\n<div>vwaddu.wx  vd, vs2, rs1, vm  # vector-scalar</div>\n</div>",
                "tooltip": "vwaddu.vv  vd, vs2, vs1, vm  # vector-vector\nvwaddu.vx  vd, vs2, rs1, vm  # vector-scalar\nvwaddu.wv  vd, vs2, vs1, vm  # vector-vector\nvwaddu.wx  vd, vs2, rs1, vm  # vector-scalar",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_widening_integer_add_subtract"
            };

        case "VWADD.WV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n<div>Can define assembly pseudoinstructions vwcvt.x.x.v vd,vs,vm = vwadd.vx vd,vs,x0,vm and vwcvtu.x.x.v vd,vs,vm = vwaddu.vx vd,vs,x0,vm</div>\n<div>vwadd.vv  vd, vs2, vs1, vm  # vector-vector</div>\n<div>vwadd.vx  vd, vs2, rs1, vm  # vector-scalar</div>\n<div>vwadd.wv  vd, vs2, vs1, vm  # vector-vector</div>\n<div>vwadd.wx  vd, vs2, rs1, vm  # vector-scalar</div>\n</div>",
                "tooltip": "Can define assembly pseudoinstructions vwcvt.x.x.v vd,vs,vm = vwadd.vx vd,vs,x0,vm and vwcvtu.x.x.v vd,vs,vm = vwaddu.vx vd,vs,x0,vm\nvwadd.vv  vd, vs2, vs1, vm  # vector-vector\nvwadd.vx  vd, vs2, rs1, vm  # vector-scalar\nvwadd.wv  vd, vs2, vs1, vm  # vector-vector\nvwadd.wx  vd, vs2, rs1, vm  # vector-scalar",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_widening_integer_add_subtract"
            };

        case "VWSUBU.WV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n<div>vwsubu.vv  vd, vs2, vs1, vm  # vector-vector</div>\n<div>vwsubu.vx  vd, vs2, rs1, vm  # vector-scalar</div>\n<div>vwsubu.wv  vd, vs2, vs1, vm  # vector-vector</div>\n<div>vwsubu.wx  vd, vs2, rs1, vm  # vector-scalar</div>\n</div>",
                "tooltip": "vwsubu.vv  vd, vs2, vs1, vm  # vector-vector\nvwsubu.vx  vd, vs2, rs1, vm  # vector-scalar\nvwsubu.wv  vd, vs2, vs1, vm  # vector-vector\nvwsubu.wx  vd, vs2, rs1, vm  # vector-scalar",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_widening_integer_add_subtract"
            };

        case "VWSUB.WV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n<div>vwsub.vv  vd, vs2, vs1, vm  # vector-vector</div>\n<div>vwsub.vx  vd, vs2, rs1, vm  # vector-scalar</div>\n<div>vwsub.wv  vd, vs2, vs1, vm  # vector-vector</div>\n<div>vwsub.wx  vd, vs2, rs1, vm  # vector-scalar</div>\n</div>",
                "tooltip": "vwsub.vv  vd, vs2, vs1, vm  # vector-vector\nvwsub.vx  vd, vs2, rs1, vm  # vector-scalar\nvwsub.wv  vd, vs2, vs1, vm  # vector-vector\nvwsub.wx  vd, vs2, rs1, vm  # vector-scalar",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_widening_integer_add_subtract"
            };

        case "VWADDU.VX":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vwaddu.vv  vd, vs2, vs1, vm  # vector-vector</div>\n<div>vwaddu.vx  vd, vs2, rs1, vm  # vector-scalar</div>\n<div>vwaddu.wv  vd, vs2, vs1, vm  # vector-vector</div>\n<div>vwaddu.wx  vd, vs2, rs1, vm  # vector-scalar</div>\n</div>",
                "tooltip": "vwaddu.vv  vd, vs2, vs1, vm  # vector-vector\nvwaddu.vx  vd, vs2, rs1, vm  # vector-scalar\nvwaddu.wv  vd, vs2, vs1, vm  # vector-vector\nvwaddu.wx  vd, vs2, rs1, vm  # vector-scalar",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_widening_integer_add_subtract"
            };

        case "VWADD.VX":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>Can define assembly pseudoinstructions vwcvt.x.x.v vd,vs,vm = vwadd.vx vd,vs,x0,vm and vwcvtu.x.x.v vd,vs,vm = vwaddu.vx vd,vs,x0,vm</div>\n<div>vwadd.vv  vd, vs2, vs1, vm  # vector-vector</div>\n<div>vwadd.vx  vd, vs2, rs1, vm  # vector-scalar</div>\n<div>vwadd.wv  vd, vs2, vs1, vm  # vector-vector</div>\n<div>vwadd.wx  vd, vs2, rs1, vm  # vector-scalar</div>\n</div>",
                "tooltip": "Can define assembly pseudoinstructions vwcvt.x.x.v vd,vs,vm = vwadd.vx vd,vs,x0,vm and vwcvtu.x.x.v vd,vs,vm = vwaddu.vx vd,vs,x0,vm\nvwadd.vv  vd, vs2, vs1, vm  # vector-vector\nvwadd.vx  vd, vs2, rs1, vm  # vector-scalar\nvwadd.wv  vd, vs2, vs1, vm  # vector-vector\nvwadd.wx  vd, vs2, rs1, vm  # vector-scalar",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_widening_integer_add_subtract"
            };

        case "VWSUBU.VX":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vwsubu.vv  vd, vs2, vs1, vm  # vector-vector</div>\n<div>vwsubu.vx  vd, vs2, rs1, vm  # vector-scalar</div>\n<div>vwsubu.wv  vd, vs2, vs1, vm  # vector-vector</div>\n<div>vwsubu.wx  vd, vs2, rs1, vm  # vector-scalar</div>\n</div>",
                "tooltip": "vwsubu.vv  vd, vs2, vs1, vm  # vector-vector\nvwsubu.vx  vd, vs2, rs1, vm  # vector-scalar\nvwsubu.wv  vd, vs2, vs1, vm  # vector-vector\nvwsubu.wx  vd, vs2, rs1, vm  # vector-scalar",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_widening_integer_add_subtract"
            };

        case "VWSUB.VX":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vwsub.vv  vd, vs2, vs1, vm  # vector-vector</div>\n<div>vwsub.vx  vd, vs2, rs1, vm  # vector-scalar</div>\n<div>vwsub.wv  vd, vs2, vs1, vm  # vector-vector</div>\n<div>vwsub.wx  vd, vs2, rs1, vm  # vector-scalar</div>\n</div>",
                "tooltip": "vwsub.vv  vd, vs2, vs1, vm  # vector-vector\nvwsub.vx  vd, vs2, rs1, vm  # vector-scalar\nvwsub.wv  vd, vs2, vs1, vm  # vector-vector\nvwsub.wx  vd, vs2, rs1, vm  # vector-scalar",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_widening_integer_add_subtract"
            };

        case "VWADDU.WX":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vwaddu.vv  vd, vs2, vs1, vm  # vector-vector</div>\n<div>vwaddu.vx  vd, vs2, rs1, vm  # vector-scalar</div>\n<div>vwaddu.wv  vd, vs2, vs1, vm  # vector-vector</div>\n<div>vwaddu.wx  vd, vs2, rs1, vm  # vector-scalar</div>\n</div>",
                "tooltip": "vwaddu.vv  vd, vs2, vs1, vm  # vector-vector\nvwaddu.vx  vd, vs2, rs1, vm  # vector-scalar\nvwaddu.wv  vd, vs2, vs1, vm  # vector-vector\nvwaddu.wx  vd, vs2, rs1, vm  # vector-scalar",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_widening_integer_add_subtract"
            };

        case "VWADD.WX":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>Can define assembly pseudoinstructions vwcvt.x.x.v vd,vs,vm = vwadd.vx vd,vs,x0,vm and vwcvtu.x.x.v vd,vs,vm = vwaddu.vx vd,vs,x0,vm</div>\n<div>vwadd.vv  vd, vs2, vs1, vm  # vector-vector</div>\n<div>vwadd.vx  vd, vs2, rs1, vm  # vector-scalar</div>\n<div>vwadd.wv  vd, vs2, vs1, vm  # vector-vector</div>\n<div>vwadd.wx  vd, vs2, rs1, vm  # vector-scalar</div>\n</div>",
                "tooltip": "Can define assembly pseudoinstructions vwcvt.x.x.v vd,vs,vm = vwadd.vx vd,vs,x0,vm and vwcvtu.x.x.v vd,vs,vm = vwaddu.vx vd,vs,x0,vm\nvwadd.vv  vd, vs2, vs1, vm  # vector-vector\nvwadd.vx  vd, vs2, rs1, vm  # vector-scalar\nvwadd.wv  vd, vs2, vs1, vm  # vector-vector\nvwadd.wx  vd, vs2, rs1, vm  # vector-scalar",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_widening_integer_add_subtract"
            };

        case "VWSUBU.WX":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vwsubu.vv  vd, vs2, vs1, vm  # vector-vector</div>\n<div>vwsubu.vx  vd, vs2, rs1, vm  # vector-scalar</div>\n<div>vwsubu.wv  vd, vs2, vs1, vm  # vector-vector</div>\n<div>vwsubu.wx  vd, vs2, rs1, vm  # vector-scalar</div>\n</div>",
                "tooltip": "vwsubu.vv  vd, vs2, vs1, vm  # vector-vector\nvwsubu.vx  vd, vs2, rs1, vm  # vector-scalar\nvwsubu.wv  vd, vs2, vs1, vm  # vector-vector\nvwsubu.wx  vd, vs2, rs1, vm  # vector-scalar",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_widening_integer_add_subtract"
            };

        case "VWSUB.WX":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vwsub.vv  vd, vs2, vs1, vm  # vector-vector</div>\n<div>vwsub.vx  vd, vs2, rs1, vm  # vector-scalar</div>\n<div>vwsub.wv  vd, vs2, vs1, vm  # vector-vector</div>\n<div>vwsub.wx  vd, vs2, rs1, vm  # vector-scalar</div>\n</div>",
                "tooltip": "vwsub.vv  vd, vs2, vs1, vm  # vector-vector\nvwsub.vx  vd, vs2, rs1, vm  # vector-scalar\nvwsub.wv  vd, vs2, vs1, vm  # vector-vector\nvwsub.wx  vd, vs2, rs1, vm  # vector-scalar",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_widening_integer_add_subtract"
            };

        case "VWMULU.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n<div>vwmulu.vv vd, vs2, vs1, vm # vector-vector</div>\n<div>vwmulu.vx vd, vs2, rs1, vm # vector-scalar</div>\n</div>",
                "tooltip": "vwmulu.vv vd, vs2, vs1, vm # vector-vector\nvwmulu.vx vd, vs2, rs1, vm # vector-scalar",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_widening_integer_multiply_instructions"
            };

        case "VWMULSU.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n<div>vwmulsu.vv vd, vs2, vs1, vm # vector-vector</div>\n<div>vwmulsu.vx vd, vs2, rs1, vm # vector-scalar</div>\n</div>",
                "tooltip": "vwmulsu.vv vd, vs2, vs1, vm # vector-vector\nvwmulsu.vx vd, vs2, rs1, vm # vector-scalar",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_widening_integer_multiply_instructions"
            };

        case "VWMULU.VX":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vwmulu.vv vd, vs2, vs1, vm # vector-vector</div>\n<div>vwmulu.vx vd, vs2, rs1, vm # vector-scalar</div>\n</div>",
                "tooltip": "vwmulu.vv vd, vs2, vs1, vm # vector-vector\nvwmulu.vx vd, vs2, rs1, vm # vector-scalar",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_widening_integer_multiply_instructions"
            };

        case "VWMULSU.VX":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vwmulsu.vv vd, vs2, vs1, vm # vector-vector</div>\n<div>vwmulsu.vx vd, vs2, rs1, vm # vector-scalar</div>\n</div>",
                "tooltip": "vwmulsu.vv vd, vs2, vs1, vm # vector-vector\nvwmulsu.vx vd, vs2, rs1, vm # vector-scalar",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_widening_integer_multiply_instructions"
            };

        case "VWMACCU.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n<div>vwmaccu.vv vd, vs1, vs2, vm    # vd[i] = +(vs1[i] * vs2[i]) + vd[i]</div>\n<div>vwmaccu.vx vd, rs1, vs2, vm    # vd[i] = +(x[rs1] * vs2[i]) + vd[i]</div>\n</div>",
                "tooltip": "vwmaccu.vv vd, vs1, vs2, vm    # vd[i] = +(vs1[i] * vs2[i]) + vd[i]\nvwmaccu.vx vd, rs1, vs2, vm    # vd[i] = +(x[rs1] * vs2[i]) + vd[i]",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_widening_integer_multiply_add_instructions"
            };

        case "VWMACC.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n<div>vwmacc.vv vd, vs1, vs2, vm    # vd[i] = +(vs1[i] * vs2[i]) + vd[i]</div>\n<div>vwmacc.vx vd, rs1, vs2, vm    # vd[i] = +(x[rs1] * vs2[i]) + vd[i]</div>\n</div>",
                "tooltip": "vwmacc.vv vd, vs1, vs2, vm    # vd[i] = +(vs1[i] * vs2[i]) + vd[i]\nvwmacc.vx vd, rs1, vs2, vm    # vd[i] = +(x[rs1] * vs2[i]) + vd[i]",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_widening_integer_multiply_add_instructions"
            };

        case "VWMACCSU.VV":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, vs1, vd</span>\n\n<div>vwmaccsu.vv vd, vs1, vs2, vm    # vd[i] = +(signed(vs1[i]) * unsigned(vs2[i])) + vd[i]</div>\n<div>vwmaccsu.vx vd, rs1, vs2, vm    # vd[i] = +(signed(x[rs1]) * unsigned(vs2[i])) + vd[i]</div>\n</div>",
                "tooltip": "vwmaccsu.vv vd, vs1, vs2, vm    # vd[i] = +(signed(vs1[i]) * unsigned(vs2[i])) + vd[i]\nvwmaccsu.vx vd, rs1, vs2, vm    # vd[i] = +(signed(x[rs1]) * unsigned(vs2[i])) + vd[i]",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_widening_integer_multiply_add_instructions"
            };

        case "VWMACCU.VX":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vwmaccu.vv vd, vs1, vs2, vm    # vd[i] = +(vs1[i] * vs2[i]) + vd[i]</div>\n<div>vwmaccu.vx vd, rs1, vs2, vm    # vd[i] = +(x[rs1] * vs2[i]) + vd[i]</div>\n</div>",
                "tooltip": "vwmaccu.vv vd, vs1, vs2, vm    # vd[i] = +(vs1[i] * vs2[i]) + vd[i]\nvwmaccu.vx vd, rs1, vs2, vm    # vd[i] = +(x[rs1] * vs2[i]) + vd[i]",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_widening_integer_multiply_add_instructions"
            };

        case "VWMACC.VX":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vwmacc.vv vd, vs1, vs2, vm    # vd[i] = +(vs1[i] * vs2[i]) + vd[i]</div>\n<div>vwmacc.vx vd, rs1, vs2, vm    # vd[i] = +(x[rs1] * vs2[i]) + vd[i]</div>\n</div>",
                "tooltip": "vwmacc.vv vd, vs1, vs2, vm    # vd[i] = +(vs1[i] * vs2[i]) + vd[i]\nvwmacc.vx vd, rs1, vs2, vm    # vd[i] = +(x[rs1] * vs2[i]) + vd[i]",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_widening_integer_multiply_add_instructions"
            };

        case "VWMACCSU.VX":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vwmaccsu.vv vd, vs1, vs2, vm    # vd[i] = +(signed(vs1[i]) * unsigned(vs2[i])) + vd[i]</div>\n<div>vwmaccsu.vx vd, rs1, vs2, vm    # vd[i] = +(signed(x[rs1]) * unsigned(vs2[i])) + vd[i]</div>\n</div>",
                "tooltip": "vwmaccsu.vv vd, vs1, vs2, vm    # vd[i] = +(signed(vs1[i]) * unsigned(vs2[i])) + vd[i]\nvwmaccsu.vx vd, rs1, vs2, vm    # vd[i] = +(signed(x[rs1]) * unsigned(vs2[i])) + vd[i]",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_widening_integer_multiply_add_instructions"
            };

        case "VWMACCUS.VX":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>vwmaccus.vx vd, rs1, vs2, vm    # vd[i] = +(unsigned(x[rs1]) * signed(vs2[i])) + vd[i]</div>\n</div>",
                "tooltip": "vwmaccus.vx vd, rs1, vs2, vm    # vd[i] = +(unsigned(x[rs1]) * signed(vs2[i])) + vd[i]",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_widening_integer_multiply_add_instructions"
            };

        case "VSLIDE1UP.VX":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>The vslide1up instruction places the x register argument at location 0 of the destination vector register group, provided that element 0 is active, otherwise the destination element is unchanged</div>\n<div>The vslide1up instruction requires that the destination vector register group does not overlap the source vector register group or the mask register</div>\n<div>vslide1up.vx vd, vs2, rs1, vm        # vd[0]=x[rs1], vd[i+1] = vs2[i]</div>\n<div>vslide1up behavior</div>\n</div>",
                "tooltip": "The vslide1up instruction places the x register argument at location 0 of the destination vector register group, provided that element 0 is active, otherwise the destination element is unchanged\nThe vslide1up instruction requires that the destination vector register group does not overlap the source vector register group or the mask register\nvslide1up.vx vd, vs2, rs1, vm        # vd[0]=x[rs1], vd[i+1] = vs2[i]\nvslide1up behavior",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_slide1up"
            };

        case "VSLIDE1DOWN.VX":
            return {
                "html": "<div>\n<span class=\"opcode\">vs2, rs1, vd</span>\n\n<div>The vslide1down instruction copies the first vl -1 active elements values from index i +1 in the source vector register group to index i in the destination vector register group.</div>\n<div>The vslide1down instruction places the x register argument at location vl -1 in the destination vector register, provided that element vl-1 is active, otherwise the destination element is unchanged</div>\n<div>Note vslide1down instruction can be used to load values into a vector register without using memory and without disturbing other vector registers</div>\n<div>This provides a path for debuggers to modify the contents of a vector register, albeit slowly, with multiple repeated vslide1down invocations</div>\n<div>vslide1down.vx vd, vs2, rs1, vm      # vd[i] = vs2[i+1], vd[vl-1]=x[rs1]</div>\n<div>vslide1down behavior</div>\n</div>",
                "tooltip": "The vslide1down instruction copies the first vl -1 active elements values from index i +1 in the source vector register group to index i in the destination vector register group.\nThe vslide1down instruction places the x register argument at location vl -1 in the destination vector register, provided that element vl-1 is active, otherwise the destination element is unchanged\nNote vslide1down instruction can be used to load values into a vector register without using memory and without disturbing other vector registers\nThis provides a path for debuggers to modify the contents of a vector register, albeit slowly, with multiple repeated vslide1down invocations\nvslide1down.vx vd, vs2, rs1, vm      # vd[i] = vs2[i+1], vd[vl-1]=x[rs1]\nvslide1down behavior",
                "url": "https://five-embeddev.com/riscv-v-spec/draft/v-spec.html#_vector_slide1down_instruction"
            };

        case "@CUSTOM0":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, imm12</span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/"
            };

        case "@CUSTOM0.RS1":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, imm12</span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/"
            };

        case "@CUSTOM0.RS1.RS2":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, imm12</span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/"
            };

        case "@CUSTOM0.RD":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, imm12</span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/"
            };

        case "@CUSTOM0.RD.RS1":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, imm12</span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/"
            };

        case "@CUSTOM0.RD.RS1.RS2":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, imm12</span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/"
            };

        case "@CUSTOM1":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, imm12</span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/"
            };

        case "@CUSTOM1.RS1":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, imm12</span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/"
            };

        case "@CUSTOM1.RS1.RS2":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, imm12</span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/"
            };

        case "@CUSTOM1.RD":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, imm12</span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/"
            };

        case "@CUSTOM1.RD.RS1":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, imm12</span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/"
            };

        case "@CUSTOM1.RD.RS1.RS2":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, imm12</span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/"
            };

        case "@CUSTOM2":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, imm12</span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/"
            };

        case "@CUSTOM2.RS1":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, imm12</span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/"
            };

        case "@CUSTOM2.RS1.RS2":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, imm12</span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/"
            };

        case "@CUSTOM2.RD":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, imm12</span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/"
            };

        case "@CUSTOM2.RD.RS1":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, imm12</span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/"
            };

        case "@CUSTOM2.RD.RS1.RS2":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, imm12</span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/"
            };

        case "@CUSTOM3":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, imm12</span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/"
            };

        case "@CUSTOM3.RS1":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, imm12</span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/"
            };

        case "@CUSTOM3.RS1.RS2":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, imm12</span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/"
            };

        case "@CUSTOM3.RD":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, imm12</span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/"
            };

        case "@CUSTOM3.RD.RS1":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, imm12</span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/"
            };

        case "@CUSTOM3.RD.RS1.RS2":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, imm12</span>\n\n</div>",
                "tooltip": "",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/"
            };

        case "CSRRW":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, imm12</span>\n\n<div>The CSRRW (Atomic Read/Write CSR) instruction atomically swaps values in the CSRs and integer registers</div>\n<div>CSRRW reads the old value of the CSR, zero-extends the value to XLEN bits, then writes it to integer register rd</div>\n<div>A CSRRW with rs1 = x0 will attempt to write zero to the destination CSR.</div>\n<div>The assembler pseudoinstruction to write a CSR, CSRW csr, rs1 , is encoded as CSRRW x0, csr, rs1 , while CSRWI csr, uimm , is encoded as CSRRWI x0, csr, uimm .</div>\n</div>",
                "tooltip": "The CSRRW (Atomic Read/Write CSR) instruction atomically swaps values in the CSRs and integer registers\nCSRRW reads the old value of the CSR, zero-extends the value to XLEN bits, then writes it to integer register rd\nA CSRRW with rs1 = x0 will attempt to write zero to the destination CSR.\nThe assembler pseudoinstruction to write a CSR, CSRW csr, rs1 , is encoded as CSRRW x0, csr, rs1 , while CSRWI csr, uimm , is encoded as CSRRWI x0, csr, uimm .",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/csr.html#csr-instructions"
            };

        case "CSRRS":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, imm12</span>\n\n<div>The CSRRS (Atomic Read and Set Bits in CSR) instruction reads the value of the CSR, zero-extends the value to XLEN bits, and writes it to integer register rd</div>\n<div>For both CSRRS and CSRRC, if rs1 = x0 , then the instruction will not write to the CSR at all, and so shall not cause any of the side effects that might otherwise occur on a CSR write, such as raising illegal instruction exceptions on accesses to read-only CSRs</div>\n<div>Both CSRRS and CSRRC always read the addressed CSR and cause any read side effects regardless of rs1 and rd fields</div>\n<div>The CSRRS and CSRRC instructions have same behavior so are shown as CSRR</div>\n<div>The assembler pseudoinstruction to read a CSR, CSRR rd, csr , is encoded as CSRRS rd, csr, x0</div>\n</div>",
                "tooltip": "The CSRRS (Atomic Read and Set Bits in CSR) instruction reads the value of the CSR, zero-extends the value to XLEN bits, and writes it to integer register rd\nFor both CSRRS and CSRRC, if rs1 = x0 , then the instruction will not write to the CSR at all, and so shall not cause any of the side effects that might otherwise occur on a CSR write, such as raising illegal instruction exceptions on accesses to read-only CSRs\nBoth CSRRS and CSRRC always read the addressed CSR and cause any read side effects regardless of rs1 and rd fields\nThe CSRRS and CSRRC instructions have same behavior so are shown as CSRR\nThe assembler pseudoinstruction to read a CSR, CSRR rd, csr , is encoded as CSRRS rd, csr, x0",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/csr.html#csr-instructions"
            };

        case "CSRRC":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, imm12</span>\n\n<div>The CSRRC (Atomic Read and Clear Bits in CSR) instruction reads the value of the CSR, zero-extends the value to XLEN bits, and writes it to integer register rd</div>\n</div>",
                "tooltip": "The CSRRC (Atomic Read and Clear Bits in CSR) instruction reads the value of the CSR, zero-extends the value to XLEN bits, and writes it to integer register rd",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/csr.html#csr-instructions"
            };

        case "CSRRWI":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, imm12</span>\n\n<div>The CSRRWI, CSRRSI, and CSRRCI variants are similar to CSRRW, CSRRS, and CSRRC respectively, except they update the CSR using an XLEN-bit value obtained by zero-extending a 5-bit unsigned immediate (uimm[4:0]) field encoded in the rs1 field instead of a value from an integer register</div>\n<div>For CSRRWI, if rd = x0 , then the instruction shall not read the CSR and shall not cause any of the side effects that might occur on a CSR read</div>\n</div>",
                "tooltip": "The CSRRWI, CSRRSI, and CSRRCI variants are similar to CSRRW, CSRRS, and CSRRC respectively, except they update the CSR using an XLEN-bit value obtained by zero-extending a 5-bit unsigned immediate (uimm[4:0]) field encoded in the rs1 field instead of a value from an integer register\nFor CSRRWI, if rd = x0 , then the instruction shall not read the CSR and shall not cause any of the side effects that might occur on a CSR read",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/csr.html#csr-instructions"
            };

        case "CSRRSI":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, imm12</span>\n\n<div>For CSRRSI and CSRRCI, if the uimm[4:0] field is zero, then these instructions will not write to the CSR, and shall not cause any of the side effects that might otherwise occur on a CSR write</div>\n<div>Both CSRRSI and CSRRCI will always read the CSR and cause any read side effects regardless of rd and rs1 fields.</div>\n</div>",
                "tooltip": "For CSRRSI and CSRRCI, if the uimm[4:0] field is zero, then these instructions will not write to the CSR, and shall not cause any of the side effects that might otherwise occur on a CSR write\nBoth CSRRSI and CSRRCI will always read the CSR and cause any read side effects regardless of rd and rs1 fields.",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/csr.html#csr-instructions"
            };

        case "SRET":
            return {
                "html": "<div>\n<span class=\"opcode\"></span>\n\n<div>When an SRET instruction (see Section\u00c3\u0082\u00c2\u00a0</div>\n<div>When a trap is taken into supervisor mode, SPIE is set to SIE, and SIE is set to 0. When an SRET instruction is executed, SIE is set to SPIE, then SPIE is set to 1.</div>\n</div>",
                "tooltip": "When an SRET instruction (see Section\u00c3\u0082\u00c2\u00a0\nWhen a trap is taken into supervisor mode, SPIE is set to SIE, and SIE is set to 0. When an SRET instruction is executed, SIE is set to SPIE, then SPIE is set to 1.",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/supervisor.html#sstatus"
            };

        case "SFENCE.VMA":
            return {
                "html": "<div>\n<span class=\"opcode\">rs1, rs2</span>\n\n<div>If the new address space\u00c3\u00a2\u00c2\u0080\u00c2\u0099s page tables have been modified, or if an ASID is reused, it may be necessary to execute an SFENCE.VMA instruction (see Section\u00c3\u0082\u00c2\u00a0</div>\n</div>",
                "tooltip": "If the new address space\u00c3\u00a2\u00c2\u0080\u00c2\u0099s page tables have been modified, or if an ASID is reused, it may be necessary to execute an SFENCE.VMA instruction (see Section\u00c3\u0082\u00c2",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/supervisor.html#sec:satp"
            };

        case "OR":
            return {
                "html": "<div>\n<span class=\"opcode\">rd, rs1, rs2</span>\n\n<div>VS-level external interrupts are made pending based on the logical-OR of:</div>\n<div>When hip is read with a CSR instruction, the value of the VSEIP bit returned in the rd destination register is the logical-OR of all the sources listed above</div>\n</div>",
                "tooltip": "VS-level external interrupts are made pending based on the logical-OR of:\nWhen hip is read with a CSR instruction, the value of the VSEIP bit returned in the rd destination register is the logical-OR of all the sources listed above",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/hypervisor.html#sec:hinterruptregs"
            };

        case "SB":
            return {
                "html": "<div>\n<span class=\"opcode\">imm12hi, rs1, rs2, imm12lo</span>\n\n<div>For a standard store instruction that is not a compressed instruction and is one of SB, SH, SW, SD, FSW, FSD, or FSQ, the transformed instruction has the format shown in Figure\u00c3\u0082\u00c2\u00a0</div>\n<div>Transformed noncompressed store instruction (SB, SH, SW, SD, FSW, FSD, or FSQ)</div>\n</div>",
                "tooltip": "For a standard store instruction that is not a compressed instruction and is one of SB, SH, SW, SD, FSW, FSD, or FSQ, the transformed instruction has the format shown in Figure\u00c3\u0082\u00c2\u00a0\nTransformed noncompressed store instruction (SB, SH, SW, SD, FSW, FSD, or FSQ)",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/hypervisor.html#sec:tinst-vals"
            };

        case "MRET":
            return {
                "html": "<div>\n<span class=\"opcode\"></span>\n\n<div>An MRET or SRET instruction that changes the operating mode to U-mode, VS-mode, or VU-mode also sets SPRV=0.</div>\n</div>",
                "tooltip": "An MRET or SRET instruction that changes the operating mode to U-mode, VS-mode, or VU-mode also sets SPRV=0.",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/hypervisor.html#hypervisor-status-register-hstatus"
            };

        case "WFI":
            return {
                "html": "<div>\n<span class=\"opcode\"></span>\n\n<div>Executing instruction WFI when V=1 causes an illegal instruction exception, unless it completes within an implementation-specific, bounded time limit.</div>\n<div>The behavior required of WFI in VS-mode and VU-mode is the same as required of it in U-mode when S-mode exists.</div>\n</div>",
                "tooltip": "Executing instruction WFI when V=1 causes an illegal instruction exception, unless it completes within an implementation-specific, bounded time limit.\nThe behavior required of WFI in VS-mode and VU-mode is the same as required of it in U-mode when S-mode exists.",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/hypervisor.html#wfi-in-virtual-operating-modes"
            };

        case "HFENCE.GVMA":
            return {
                "html": "<div>\n<span class=\"opcode\">rs1, rs2</span>\n\n<div>If the new virtual machine\u00c3\u00a2\u00c2\u0080\u00c2\u0099s guest physical page tables have been modified, it may be necessary to execute an HFENCE.GVMA instruction (see Section\u00c3\u0082\u00c2\u00a0</div>\n</div>",
                "tooltip": "If the new virtual machine\u00c3\u00a2\u00c2\u0080\u00c2\u0099s guest physical page tables have been modified, it may be necessary to execute an HFENCE.GVMA instruction (see Section\u00c3\u0082\u00c2",
                "url": "https://five-embeddev.com/riscv-isa-manual/latest/hypervisor.html#sec:hgatp"
            };


    }
}
