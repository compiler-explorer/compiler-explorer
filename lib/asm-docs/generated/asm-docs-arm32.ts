import {AssemblyInstructionInfo} from '../base.js';

export function getAsmOpcode(opcode: string | undefined): AssemblyInstructionInfo | undefined {
    if (!opcode) return;
    switch (opcode) {
        case "ADC":
        case "ADCS":
            return {
                "tooltip": "Add with Carry (immediate) adds an immediate value and the Carry flag value to a register value, and writes the result to the destination register.",
                "html": "<p>Add with Carry (immediate) adds an immediate value and the Carry flag value to a register value, and writes the result to the destination register.</p><p>If the destination register is not the PC, the ADCS variant of the instruction updates the condition flags based on the result.</p><p>The field descriptions for <syntax>&lt;Rd&gt;</syntax> identify the encodings where the PC is permitted as the destination register. Arm deprecates any use of these encodings. However, when the destination register is the PC:</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "ADC":
        case "ADCS":
            return {
                "tooltip": "Add with Carry (register) adds a register value, the Carry flag value, and an optionally-shifted register value, and writes the result to the destination register.",
                "html": "<p>Add with Carry (register) adds a register value, the Carry flag value, and an optionally-shifted register value, and writes the result to the destination register.</p><p>If the destination register is not the PC, the ADCS variant of the instruction updates the condition flags based on the result.</p><p>The field descriptions for <syntax>&lt;Rd&gt;</syntax> identify the encodings where the PC is permitted as the destination register. Arm deprecates any use of these encodings. However, when the destination register is the PC:</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "ADC":
        case "ADCS":
            return {
                "tooltip": "Add with Carry (register-shifted register) adds a register value, the Carry flag value, and a register-shifted register value.  It writes the result to the destination register, and can optionally update the condition flags based on the result.",
                "html": "<p>Add with Carry (register-shifted register) adds a register value, the Carry flag value, and a register-shifted register value.  It writes the result to the destination register, and can optionally update the condition flags based on the result.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "ADD":
        case "ADDS":
            return {
                "tooltip": "Add (immediate) adds an immediate value to a register value, and writes the result to the destination register.",
                "html": "<p>Add (immediate) adds an immediate value to a register value, and writes the result to the destination register.</p><p>If the destination register is not the PC, the ADDS variant of the instruction updates the condition flags based on the result.</p><p>The field descriptions for <syntax>&lt;Rd&gt;</syntax> identify the encodings where the PC is permitted as the destination register. If the destination register is the PC:</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "ADD":
        case "ADDS":
            return {
                "tooltip": "Add (register) adds a register value and an optionally-shifted register value, and writes the result to the destination register.",
                "html": "<p>Add (register) adds a register value and an optionally-shifted register value, and writes the result to the destination register.</p><p>If the destination register is not the PC, the ADDS variant of the instruction updates the condition flags based on the result.</p><p>The field descriptions for <syntax>&lt;Rd&gt;</syntax> identify the encodings where the PC is permitted as the destination register. If the destination register is the PC:</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "ADD":
        case "ADDS":
            return {
                "tooltip": "Add (register-shifted register) adds a register value and a register-shifted register value. It writes the result to the destination register, and can optionally update the condition flags based on the result.",
                "html": "<p>Add (register-shifted register) adds a register value and a register-shifted register value. It writes the result to the destination register, and can optionally update the condition flags based on the result.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "ADD":
        case "ADDS":
            return {
                "tooltip": "Add to SP (immediate) adds an immediate value to the SP value, and writes the result to the destination register.",
                "html": "<p>Add to SP (immediate) adds an immediate value to the SP value, and writes the result to the destination register.</p><p>If the destination register is not the PC, the ADDS variant of the instruction updates the condition flags based on the result.</p><p>The field descriptions for <syntax>&lt;Rd&gt;</syntax> identify the encodings where the PC is permitted as the destination register. However, when the destination register is the PC:</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "ADD":
        case "ADDS":
            return {
                "tooltip": "Add to SP (register) adds an optionally-shifted register value to the SP value, and writes the result to the destination register.",
                "html": "<p>Add to SP (register) adds an optionally-shifted register value to the SP value, and writes the result to the destination register.</p><p>If the destination register is not the PC, the ADDS variant of the instruction updates the condition flags based on the result.</p><p>The field descriptions for <syntax>&lt;Rd&gt;</syntax> identify the encodings where the PC is permitted as the destination register. Arm deprecates any use of these encodings. However, when the destination register is the PC:</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "ADR":
            return {
                "tooltip": "Form PC-relative address adds an immediate value to the PC value to form a PC-relative address, and writes the result to the destination register.",
                "html": "<p>Form PC-relative address adds an immediate value to the PC value to form a PC-relative address, and writes the result to the destination register.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "AESD":
            return {
                "tooltip": "AES single round decryption.",
                "html": "<p>AES single round decryption.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "AESE":
            return {
                "tooltip": "AES single round encryption.",
                "html": "<p>AES single round encryption.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "AESIMC":
            return {
                "tooltip": "AES inverse mix columns.",
                "html": "<p>AES inverse mix columns.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "AESMC":
            return {
                "tooltip": "AES mix columns.",
                "html": "<p>AES mix columns.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "AND":
        case "ANDS":
            return {
                "tooltip": "Bitwise AND (immediate) performs a bitwise AND of a register value and an immediate value, and writes the result to the destination register.",
                "html": "<p>Bitwise AND (immediate) performs a bitwise AND of a register value and an immediate value, and writes the result to the destination register.</p><p>If the destination register is not the PC, the ANDS variant of the instruction updates the condition flags based on the result.</p><p>The field descriptions for <syntax>&lt;Rd&gt;</syntax> identify the encodings where the PC is permitted as the destination register. Arm deprecates any use of these encodings. However, when the destination register is the PC:</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "AND":
        case "ANDS":
            return {
                "tooltip": "Bitwise AND (register) performs a bitwise AND of a register value and an optionally-shifted register value, and writes the result to the destination register.",
                "html": "<p>Bitwise AND (register) performs a bitwise AND of a register value and an optionally-shifted register value, and writes the result to the destination register.</p><p>If the destination register is not the PC, the ANDS variant of the instruction updates the condition flags based on the result.</p><p>The field descriptions for <syntax>&lt;Rd&gt;</syntax> identify the encodings where the PC is permitted as the destination register. Arm deprecates any use of these encodings. However, when the destination register is the PC:</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "AND":
        case "ANDS":
            return {
                "tooltip": "Bitwise AND (register-shifted register) performs a bitwise AND of a register value and a register-shifted register value. It writes the result to the destination register, and can optionally update the condition flags based on the result.",
                "html": "<p>Bitwise AND (register-shifted register) performs a bitwise AND of a register value and a register-shifted register value. It writes the result to the destination register, and can optionally update the condition flags based on the result.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "ASR":
            return {
                "tooltip": "Arithmetic Shift Right (immediate) shifts a register value right by an immediate number of bits, shifting in copies of its sign bit, and writes the result to the destination register.",
                "html": "<p>Arithmetic Shift Right (immediate) shifts a register value right by an immediate number of bits, shifting in copies of its sign bit, and writes the result to the destination register.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "ASRS":
            return {
                "tooltip": "Arithmetic Shift Right, setting flags (immediate) shifts a register value right by an immediate number of bits, shifting in copies of its sign bit, and writes the result to the destination register.",
                "html": "<p>Arithmetic Shift Right, setting flags (immediate) shifts a register value right by an immediate number of bits, shifting in copies of its sign bit, and writes the result to the destination register.</p><p>If the destination register is not the PC, this instruction updates the condition flags based on the result.</p><p>The field descriptions for <syntax>&lt;Rd&gt;</syntax> identify the encodings where the PC is permitted as the destination register. Arm deprecates any use of these encodings. However, when the destination register is the PC:</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "B":
            return {
                "tooltip": "Branch causes a branch to a target address.",
                "html": "<p>Branch causes a branch to a target address.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "BFC":
            return {
                "tooltip": "Bit Field Clear clears any number of adjacent bits at any position in a register, without affecting the other bits in the register.",
                "html": "<p>Bit Field Clear clears any number of adjacent bits at any position in a register, without affecting the other bits in the register.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "BFI":
            return {
                "tooltip": "Bit Field Insert copies any number of low order bits from a register into the same number of adjacent bits at any position in the destination register.",
                "html": "<p>Bit Field Insert copies any number of low order bits from a register into the same number of adjacent bits at any position in the destination register.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "BIC":
        case "BICS":
            return {
                "tooltip": "Bitwise Bit Clear (immediate) performs a bitwise AND of a register value and the complement of an immediate value, and writes the result to the destination register.",
                "html": "<p>Bitwise Bit Clear (immediate) performs a bitwise AND of a register value and the complement of an immediate value, and writes the result to the destination register.</p><p>If the destination register is not the PC, the BICS variant of the instruction updates the condition flags based on the result.</p><p>The field descriptions for <syntax>&lt;Rd&gt;</syntax> identify the encodings where the PC is permitted as the destination register. Arm deprecates any use of these encodings. However, when the destination register is the PC:</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "BIC":
        case "BICS":
            return {
                "tooltip": "Bitwise Bit Clear (register) performs a bitwise AND of a register value and the complement of an optionally-shifted register value, and writes the result to the destination register.",
                "html": "<p>Bitwise Bit Clear (register) performs a bitwise AND of a register value and the complement of an optionally-shifted register value, and writes the result to the destination register.</p><p>If the destination register is not the PC, the BICS variant of the instruction updates the condition flags based on the result.</p><p>The field descriptions for <syntax>&lt;Rd&gt;</syntax> identify the encodings where the PC is permitted as the destination register. Arm deprecates any use of these encodings. However, when the destination register is the PC:</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "BIC":
        case "BICS":
            return {
                "tooltip": "Bitwise Bit Clear (register-shifted register) performs a bitwise AND of a register value and the complement of a register-shifted register value. It writes the result to the destination register, and can optionally update the condition flags based on the result.",
                "html": "<p>Bitwise Bit Clear (register-shifted register) performs a bitwise AND of a register value and the complement of a register-shifted register value. It writes the result to the destination register, and can optionally update the condition flags based on the result.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "BKPT":
            return {
                "tooltip": "Breakpoint causes a Breakpoint Instruction exception.",
                "html": "<p>Breakpoint causes a Breakpoint Instruction exception.</p><p>Breakpoint is always unconditional, even when inside an IT block.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "BL":
        case "BLX":
            return {
                "tooltip": "Branch with Link calls a subroutine at a PC-relative address, and setting LR to the return address.",
                "html": "<p>Branch with Link calls a subroutine at a PC-relative address, and setting LR to the return address.</p><p>Branch with Link and Exchange Instruction Sets (immediate) calls a subroutine at a PC-relative address, setting LR to the return address, and changes the instruction set from A32 to T32, or from T32 to A32.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "BLX":
            return {
                "tooltip": "Branch with Link and Exchange (register) calls a subroutine at an address specified in the register, and if necessary changes to the instruction set indicated by bit[0] of the register value. If the value in bit[0] is 0, the instruction set after the branch will be A32. If the value in bit[0] is 1, the instruction set after the branch will be T32.",
                "html": "<p>Branch with Link and Exchange (register) calls a subroutine at an address specified in the register, and if necessary changes to the instruction set indicated by bit[0] of the register value. If the value in bit[0] is 0, the instruction set after the branch will be A32. If the value in bit[0] is 1, the instruction set after the branch will be T32.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "BX":
            return {
                "tooltip": "Branch and Exchange causes a branch to an address and instruction set specified by a register.",
                "html": "<p>Branch and Exchange causes a branch to an address and instruction set specified by a register.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "BXJ":
            return {
                "tooltip": "Branch and Exchange, previously Branch and Exchange Jazelle.",
                "html": "<p>Branch and Exchange, previously Branch and Exchange Jazelle.</p><p><instruction>BXJ</instruction> behaves as a <instruction>BX</instruction> instruction, see <xref linkend=\"A32T32-base.instructions.BX\">BX</xref>. This means it causes a branch to an address and instruction set specified by a register.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "CBNZ":
        case "CBZ":
            return {
                "tooltip": "Compare and Branch on Nonzero and Compare and Branch on Zero compare the value in a register with zero, and conditionally branch forward a constant value. They do not affect the condition flags.",
                "html": "<p>Compare and Branch on Nonzero and Compare and Branch on Zero compare the value in a register with zero, and conditionally branch forward a constant value. They do not affect the condition flags.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "CLRBHB":
            return {
                "tooltip": "Clear Branch History clears the branch history for the current context to the extent that branch history information created before the CLRBHB instruction cannot be used by code before the CLRBHB instruction to exploitatively control the execution of any indirect branches in code in the current context that appear in program order after the instruction.",
                "html": "<p>Clear Branch History clears the branch history for the current context to the extent that branch history information created before the <instruction>CLRBHB</instruction> instruction cannot be used by code before the <instruction>CLRBHB</instruction> instruction to exploitatively control the execution of any indirect branches in code in the current context that appear in program order after the instruction.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "CLREX":
            return {
                "tooltip": "Clear-Exclusive clears the local monitor of the executing PE.",
                "html": "<p>Clear-Exclusive clears the local monitor of the executing PE.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "CLZ":
            return {
                "tooltip": "Count Leading Zeros returns the number of binary zero bits before the first binary one bit in a value.",
                "html": "<p>Count Leading Zeros returns the number of binary zero bits before the first binary one bit in a value.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "CMN":
            return {
                "tooltip": "Compare Negative (immediate) adds a register value and an immediate value. It updates the condition flags based on the result, and discards the result.",
                "html": "<p>Compare Negative (immediate) adds a register value and an immediate value. It updates the condition flags based on the result, and discards the result.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "CMN":
            return {
                "tooltip": "Compare Negative (register) adds a register value and an optionally-shifted register value. It updates the condition flags based on the result, and discards the result.",
                "html": "<p>Compare Negative (register) adds a register value and an optionally-shifted register value. It updates the condition flags based on the result, and discards the result.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "CMN":
            return {
                "tooltip": "Compare Negative (register-shifted register) adds a register value and a register-shifted register value. It updates the condition flags based on the result, and discards the result.",
                "html": "<p>Compare Negative (register-shifted register) adds a register value and a register-shifted register value. It updates the condition flags based on the result, and discards the result.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "CMP":
            return {
                "tooltip": "Compare (immediate) subtracts an immediate value from a register value. It updates the condition flags based on the result, and discards the result.",
                "html": "<p>Compare (immediate) subtracts an immediate value from a register value. It updates the condition flags based on the result, and discards the result.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "CMP":
            return {
                "tooltip": "Compare (register) subtracts an optionally-shifted register value from a register value. It updates the condition flags based on the result, and discards the result.",
                "html": "<p>Compare (register) subtracts an optionally-shifted register value from a register value. It updates the condition flags based on the result, and discards the result.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "CMP":
            return {
                "tooltip": "Compare (register-shifted register) subtracts a register-shifted register value from a register value. It updates the condition flags based on the result, and discards the result.",
                "html": "<p>Compare (register-shifted register) subtracts a register-shifted register value from a register value. It updates the condition flags based on the result, and discards the result.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "CPS":
        case "CPSID":
        case "CPSIE":
            return {
                "tooltip": "Change PE State changes one or more of the PSTATE.{A, I, F} interrupt mask bits and, optionally, the PSTATE.M mode field, without changing any other PSTATE bits.",
                "html": "<p>Change PE State changes one or more of the <xref linkend=\"BEIDIGBH\">PSTATE</xref>.{A, I, F} interrupt mask bits and, optionally, the <xref linkend=\"BEIDIGBH\">PSTATE</xref>.M mode field, without changing any other <xref linkend=\"BEIDIGBH\">PSTATE</xref> bits.</p><p><instruction>CPS</instruction> is treated as <instruction>NOP</instruction> if executed in User mode unless it is defined as being <arm-defined-word>constrained unpredictable</arm-defined-word> elsewhere in this section.</p><p>The PE checks whether the value being written to PSTATE.M is legal. See <xref linkend=\"CHDDFIGE\">Illegal changes to PSTATE.M</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "CRC32":
            return {
                "tooltip": "CRC32 performs a cyclic redundancy check (CRC) calculation on a value held in a general-purpose register. It takes an input CRC value in the first source operand, performs a CRC on the input value in the second source operand, and returns the output CRC value. The second source operand can be 8, 16, or 32 bits. To align with common usage, the bit order of the values is reversed as part of the operation, and the polynomial 0x04C11DB7 is used for the CRC calculation.",
                "html": "<p><instruction>CRC32</instruction> performs a cyclic redundancy check (CRC) calculation on a value held in a general-purpose register. It takes an input CRC value in the first source operand, performs a CRC on the input value in the second source operand, and returns the output CRC value. The second source operand can be 8, 16, or 32 bits. To align with common usage, the bit order of the values is reversed as part of the operation, and the polynomial <hexnumber>0x04C11DB7</hexnumber> is used for the CRC calculation.</p><p>In an Armv8.0 implementation, this is an <arm-defined-word>optional</arm-defined-word> instruction. From Armv8.1, it is mandatory for all implementations to implement this instruction.</p><p><xref linkend=\"AArch32.id_isar5\">ID_ISAR5</xref>.CRC32 indicates whether this instruction is supported in the T32 and A32 instruction sets.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "CRC32C":
            return {
                "tooltip": "CRC32C performs a cyclic redundancy check (CRC) calculation on a value held in a general-purpose register. It takes an input CRC value in the first source operand, performs a CRC on the input value in the second source operand, and returns the output CRC value. The second source operand can be 8, 16, or 32 bits. To align with common usage, the bit order of the values is reversed as part of the operation, and the polynomial 0x1EDC6F41 is used for the CRC calculation.",
                "html": "<p><instruction>CRC32C</instruction> performs a cyclic redundancy check (CRC) calculation on a value held in a general-purpose register. It takes an input CRC value in the first source operand, performs a CRC on the input value in the second source operand, and returns the output CRC value. The second source operand can be 8, 16, or 32 bits. To align with common usage, the bit order of the values is reversed as part of the operation, and the polynomial <hexnumber>0x1EDC6F41</hexnumber> is used for the CRC calculation.</p><p>In an Armv8.0 implementation, this is an <arm-defined-word>optional</arm-defined-word> instruction. From Armv8.1, it is mandatory for all implementations to implement this instruction.</p><p><xref linkend=\"AArch32.id_isar5\">ID_ISAR5</xref>.CRC32 indicates whether this instruction is supported in the T32 and A32 instruction sets.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "CSDB":
            return {
                "tooltip": "Consumption of Speculative Data Barrier is a memory barrier that controls speculative execution and data value prediction.",
                "html": "<p>Consumption of Speculative Data Barrier is a memory barrier that controls speculative execution and data value prediction.</p><p>No instruction other than branch instructions and instructions that write to the PC appearing in program order after the CSDB can be speculatively executed using the results of any:</p><p>For purposes of the definition of CSDB, PSTATE.{N,Z,C,V} is not considered a data value. This definition permits:</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "DBG":
            return {
                "tooltip": "DBG executes as a NOP. Arm deprecates any use of the DBG instruction.",
                "html": "<p><instruction>DBG</instruction> executes as a <instruction>NOP</instruction>. Arm deprecates any use of the <instruction>DBG</instruction> instruction.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "DCPS1":
            return {
                "tooltip": "Debug Change PE State to EL1 allows the debugger to move the PE into EL1 from EL0 or to a specific mode at the current Exception level.",
                "html": "<p>Debug Change PE State to EL1 allows the debugger to move the PE into EL1 from EL0 or to a specific mode at the current Exception level.</p><p><instruction>DCPS1</instruction> is <arm-defined-word>undefined</arm-defined-word> if any of:</p><p>When the PE executes <instruction>DCPS1</instruction> at EL0, EL1 or EL3:</p><p>When the PE executes <instruction>DCPS1</instruction> at EL2 the PE does not change mode, and ELR_hyp, HSR, SPSR_hyp, DLR and DSPSR become <arm-defined-word>UNKNOWN</arm-defined-word>.</p><p>For more information on the operation of the DCPS&lt;n&gt; instructions, see <xref linkend=\"dcps\">DCPS</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "DCPS2":
            return {
                "tooltip": "Debug Change PE State to EL2 allows the debugger to move the PE into EL2 from a lower Exception level.",
                "html": "<p>Debug Change PE State to EL2 allows the debugger to move the PE into EL2 from a lower Exception level.</p><p><instruction>DCPS2</instruction> is <arm-defined-word>undefined</arm-defined-word> if any of:</p><p>When the PE executes <instruction>DCPS2</instruction>:</p><p>For more information on the operation of the DCPS&lt;n&gt; instructions, see <xref linkend=\"dcps\">DCPS</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "DCPS3":
            return {
                "tooltip": "Debug Change PE State to EL3 allows the debugger to move the PE into EL3 from a lower Exception level or to a specific mode at the current Exception level.",
                "html": "<p>Debug Change PE State to EL3 allows the debugger to move the PE into EL3 from a lower Exception level or to a specific mode at the current Exception level.</p><p><instruction>DCPS3</instruction> is <arm-defined-word>undefined</arm-defined-word> if any of:</p><p>When the PE executes <instruction>DCPS3</instruction>:</p><p>For more information on the operation of the DCPS&lt;n&gt; instructions, see <xref linkend=\"dcps\">DCPS</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "DMB":
            return {
                "tooltip": "Data Memory Barrier is a memory barrier that ensures the ordering of observations of memory accesses, see Data Memory Barrier (DMB).",
                "html": "<p>Data Memory Barrier is a memory barrier that ensures the ordering of observations of memory accesses, see <xref linkend=\"AA32CHDCIFHJ\">Data Memory Barrier (DMB)</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "DSB":
            return {
                "tooltip": "Data Synchronization Barrier is a memory barrier that ensures the completion of memory accesses, see Data Synchronization Barrier (DSB).",
                "html": "<p>Data Synchronization Barrier is a memory barrier that ensures the completion of memory accesses, see <xref linkend=\"AA32CHDIJCDJ\">Data Synchronization Barrier (DSB)</xref>.</p><p>An AArch32 DSB instruction does not require the completion of any AArch64 TLB maintenance instructions, regardless of the nXS qualifier, appearing in program order before the AArch32 DSB.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "EOR":
        case "EORS":
            return {
                "tooltip": "Bitwise Exclusive-OR (immediate) performs a bitwise exclusive-OR of a register value and an immediate value, and writes the result to the destination register.",
                "html": "<p>Bitwise Exclusive-OR (immediate) performs a bitwise exclusive-OR of a register value and an immediate value, and writes the result to the destination register.</p><p>If the destination register is not the PC, the EORS variant of the instruction updates the condition flags based on the result.</p><p>The field descriptions for <syntax>&lt;Rd&gt;</syntax> identify the encodings where the PC is permitted as the destination register. Arm deprecates any use of these encodings. However, when the destination register is the PC:</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "EOR":
        case "EORS":
            return {
                "tooltip": "Bitwise Exclusive-OR (register) performs a bitwise exclusive-OR of a register value and an optionally-shifted register value, and writes the result to the destination register.",
                "html": "<p>Bitwise Exclusive-OR (register) performs a bitwise exclusive-OR of a register value and an optionally-shifted register value, and writes the result to the destination register.</p><p>If the destination register is not the PC, the EORS variant of the instruction updates the condition flags based on the result.</p><p>The field descriptions for <syntax>&lt;Rd&gt;</syntax> identify the encodings where the PC is permitted as the destination register. Arm deprecates any use of these encodings. However, when the destination register is the PC:</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "EOR":
        case "EORS":
            return {
                "tooltip": "Bitwise Exclusive-OR (register-shifted register) performs a bitwise exclusive-OR of a register value and a register-shifted register value. It writes the result to the destination register, and can optionally update the condition flags based on the result.",
                "html": "<p>Bitwise Exclusive-OR (register-shifted register) performs a bitwise exclusive-OR of a register value and a register-shifted register value. It writes the result to the destination register, and can optionally update the condition flags based on the result.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "ERET":
            return {
                "tooltip": "Exception Return.",
                "html": "<p>Exception Return.</p><p>The PE branches to the address held in the register holding the preferred return address, and restores <xref linkend=\"BEIDIGBH\">PSTATE</xref> from SPSR_&lt;current_mode&gt;.</p><p>The register holding the preferred return address is:</p><p>The PE checks SPSR_&lt;current_mode&gt; for an illegal return event. See <xref linkend=\"CHDDDJDB\">Illegal return events from AArch32 state</xref>.</p><p>Exception Return is <arm-defined-word>constrained unpredictable</arm-defined-word> in User mode and System mode.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "ESB":
            return {
                "tooltip": "Error Synchronization Barrier is an error synchronization event that might also update DISR and VDISR. This instruction can be used at all Exception levels and in Debug state.",
                "html": "<p>Error Synchronization Barrier is an error synchronization event that might also update DISR and VDISR. This instruction can be used at all Exception levels and in Debug state.</p><p>In Debug state, this instruction behaves as if SError interrupts are masked at all Exception levels. See Error Synchronization Barrier in the ARM(R) Reliability, Availability, and Serviceability (RAS) Specification, Armv8, for Armv8-A architecture profile.</p><p>If the RAS Extension is not implemented, this instruction executes as a <instruction>NOP</instruction>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "FLDM*X":
        case "FLDMDBX":
        case "FLDMIAX":
            return {
                "tooltip": "FLDMDBX is the Decrement Before variant of this instruction, and FLDMIAX is the Increment After variant. FLDM*X loads multiple SIMD&FP registers from consecutive locations in the Advanced SIMD and floating-point register file using an address from a general-purpose register.",
                "html": "<p>FLDMDBX is the Decrement Before variant of this instruction, and FLDMIAX is the Increment After variant. FLDM*X loads multiple SIMD&amp;FP registers from consecutive locations in the Advanced SIMD and floating-point register file using an address from a general-purpose register.</p><p>Arm deprecates use of FLDMDBX and FLDMIAX, except for disassembly purposes, and reassembly of disassembled code.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information, see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "FSTMDBX":
        case "FSTMIAX":
            return {
                "tooltip": "FSTMX stores multiple SIMD&FP registers from the Advanced SIMD and floating-point register file to consecutive locations in using an address from a general-purpose register.",
                "html": "<p>FSTMX stores multiple SIMD&amp;FP registers from the Advanced SIMD and floating-point register file to consecutive locations in using an address from a general-purpose register.</p><p>Arm deprecates use of FSTMDBX and FSTMIAX, except for disassembly purposes, and reassembly of disassembled code.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, <xref linkend=\"AArch32.hcptr\">HCPTR</xref>, and <xref linkend=\"AArch32.fpexc\">FPEXC</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information, see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "HLT":
            return {
                "tooltip": "Halting breakpoint causes a software breakpoint to occur.",
                "html": "<p>Halting breakpoint causes a software breakpoint to occur.</p><p>Halting breakpoint is always unconditional, even inside an IT block.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "HVC":
            return {
                "tooltip": "Hypervisor Call causes a Hypervisor Call exception. For more information, see Hypervisor Call (HVC) exception. Software executing at EL1 can use this instruction to call the hypervisor to request a service.",
                "html": "<p>Hypervisor Call causes a Hypervisor Call exception. For more information, see <xref linkend=\"BEIBEBHJ\">Hypervisor Call (HVC) exception</xref>. Software executing at EL1 can use this instruction to call the hypervisor to request a service.</p><p>The <instruction>HVC</instruction> instruction is <arm-defined-word>undefined</arm-defined-word>:</p><p>The <instruction>HVC</instruction> instruction is <arm-defined-word>constrained unpredictable</arm-defined-word> in Hyp mode when EL3 is implemented and using AArch32, and <xref linkend=\"AArch32.scr\">SCR</xref>.HCE is set to 0.</p><p>On executing an <instruction>HVC</instruction> instruction, the <xref linkend=\"AArch32.hsr\">HSR, Hyp Syndrome Register</xref> reports the exception as a Hypervisor Call exception, using the EC value <hexnumber>0x12</hexnumber>, and captures the value of the immediate argument, see <xref linkend=\"BEIDBEAG\">Use of the HSR</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "ISB":
            return {
                "tooltip": "Instruction Synchronization Barrier flushes the pipeline in the PE and is a context synchronization event. For more information, see Instruction Synchronization Barrier (ISB).",
                "html": "<p>Instruction Synchronization Barrier flushes the pipeline in the PE and is a context synchronization event. For more information, see <xref linkend=\"AA32CHDHFJFC\">Instruction Synchronization Barrier (ISB)</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "IT":
            return {
                "tooltip": "If-Then makes up to four following instructions (the IT block) conditional. The conditions for the instructions in the IT block are the same as, or the inverse of, the condition the IT instruction specifies for the first instruction in the block.",
                "html": "<p>If-Then makes up to four following instructions (the IT block) conditional. The conditions for the instructions in the IT block are the same as, or the inverse of, the condition the <instruction>IT</instruction> instruction specifies for the first instruction in the block.</p><p>The <instruction>IT</instruction> instruction itself does not affect the condition flags, but the execution of the instructions in the IT block can change the condition flags.</p><p>16-bit instructions in the IT block, other than <instruction>CMP</instruction>, <instruction>CMN</instruction> and <instruction>TST</instruction>, do not set the condition flags. An <instruction>IT</instruction> instruction with the <value>AL</value> condition can change the behavior without conditional execution.</p><p>The architecture permits exception return to an instruction in the IT block only if the restoration of the <xref linkend=\"CIHJBHJA\">CPSR</xref> restores <xref linkend=\"BEIDIGBH\">PSTATE</xref>.IT to a state consistent with the conditions specified by the <instruction>IT</instruction> instruction.  Any other exception return to an instruction in an IT block is <arm-defined-word>unpredictable</arm-defined-word>. Any branch to a target instruction in an IT block is not permitted, and if such a branch is made it is <arm-defined-word>unpredictable</arm-defined-word> what condition is used when executing that target instruction and any subsequent instruction in the IT block.</p><p>Many uses of the IT instruction are deprecated for performance reasons, and an implementation might include ITD controls that can disable those uses of IT, making them <arm-defined-word>undefined</arm-defined-word>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "LDA":
            return {
                "tooltip": "Load-Acquire Word loads a word from memory and writes it to a register. The instruction also has memory ordering semantics as described in Load-Acquire, Store-Release",
                "html": "<p>Load-Acquire Word loads a word from memory and writes it to a register. The instruction also has memory ordering semantics as described in <xref linkend=\"AA32CHDBDIDF\">Load-Acquire, Store-Release</xref></p><p>For more information about support for shared memory see <xref linkend=\"CEGDAEAG\">Synchronization and semaphores</xref>. For information about memory accesses see <xref linkend=\"Chddjfjf\">Memory accesses</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "LDAB":
            return {
                "tooltip": "Load-Acquire Byte loads a byte from memory, zero-extends it to form a 32-bit word and writes it to a register. The instruction also has memory ordering semantics as described in Load-Acquire, Store-Release.",
                "html": "<p>Load-Acquire Byte loads a byte from memory, zero-extends it to form a 32-bit word and writes it to a register. The instruction also has memory ordering semantics as described in <xref linkend=\"AA32CHDBDIDF\">Load-Acquire, Store-Release</xref>.</p><p>For more information about support for shared memory see <xref linkend=\"CEGDAEAG\">Synchronization and semaphores</xref>. For information about memory accesses see <xref linkend=\"Chddjfjf\">Memory accesses</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "LDAEX":
            return {
                "tooltip": "Load-Acquire Exclusive Word loads a word from memory, writes it to a register and",
                "html": "<p>Load-Acquire Exclusive Word loads a word from memory, writes it to a register and:</p><p>The instruction also has memory ordering semantics as described in <xref linkend=\"AA32CHDBDIDF\">Load-Acquire, Store-Release</xref>.</p><p>For more information about support for shared memory see <xref linkend=\"CEGDAEAG\">Synchronization and semaphores</xref>. For information about memory accesses see <xref linkend=\"Chddjfjf\">Memory accesses</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "LDAEXB":
            return {
                "tooltip": "Load-Acquire Exclusive Byte loads a byte from memory, zero-extends it to form a 32-bit word, writes it to a register and",
                "html": "<p>Load-Acquire Exclusive Byte loads a byte from memory, zero-extends it to form a 32-bit word, writes it to a register and:</p><p>The instruction also has memory ordering semantics as described in <xref linkend=\"AA32CHDBDIDF\">Load-Acquire, Store-Release</xref>.</p><p>For more information about support for shared memory see <xref linkend=\"CEGDAEAG\">Synchronization and semaphores</xref>. For information about memory accesses see <xref linkend=\"Chddjfjf\">Memory accesses</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "LDAEXD":
            return {
                "tooltip": "Load-Acquire Exclusive Doubleword loads a doubleword from memory, writes it to two registers and",
                "html": "<p>Load-Acquire Exclusive Doubleword loads a doubleword from memory, writes it to two registers and:</p><p>The instruction also acts as a barrier instruction with the ordering requirements described in <xref linkend=\"AA32CHDBDIDF\">Load-Acquire, Store-Release</xref>.</p><p>For more information about support for shared memory see <xref linkend=\"CEGDAEAG\">Synchronization and semaphores</xref>. For information about memory accesses see <xref linkend=\"Chddjfjf\">Memory accesses</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "LDAEXH":
            return {
                "tooltip": "Load-Acquire Exclusive Halfword loads a halfword from memory, zero-extends it to form a 32-bit word, writes it to a register and",
                "html": "<p>Load-Acquire Exclusive Halfword loads a halfword from memory, zero-extends it to form a 32-bit word, writes it to a register and:</p><p>The instruction also has memory ordering semantics as described in <xref linkend=\"AA32CHDBDIDF\">Load-Acquire, Store-Release</xref>.</p><p>For more information about support for shared memory see <xref linkend=\"CEGDAEAG\">Synchronization and semaphores</xref>. For information about memory accesses see <xref linkend=\"Chddjfjf\">Memory accesses</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "LDAH":
            return {
                "tooltip": "Load-Acquire Halfword loads a halfword from memory, zero-extends it to form a 32-bit word and writes it to a register. The instruction also has memory ordering semantics as described in Load-Acquire, Store-Release.",
                "html": "<p>Load-Acquire Halfword loads a halfword from memory, zero-extends it to form a 32-bit word and writes it to a register. The instruction also has memory ordering semantics as described in <xref linkend=\"AA32CHDBDIDF\">Load-Acquire, Store-Release</xref>.</p><p>For more information about support for shared memory see <xref linkend=\"CEGDAEAG\">Synchronization and semaphores</xref>. For information about memory accesses see <xref linkend=\"Chddjfjf\">Memory accesses</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "LDC":
            return {
                "tooltip": "Load data to System register (immediate) calculates an address from a base register value and an immediate offset, loads a word from memory, and writes it to the DBGDTRTXint System register. It can use offset, post-indexed, pre-indexed, or unindexed addressing. For information about memory accesses see Memory accesses.",
                "html": "<p>Load data to System register (immediate) calculates an address from a base register value and an immediate offset, loads a word from memory, and writes it to the <xref linkend=\"AArch32.dbgdtrtxint\">DBGDTRTXint</xref> System register. It can use offset, post-indexed, pre-indexed, or unindexed addressing. For information about memory accesses see <xref linkend=\"Chddjfjf\">Memory accesses</xref>.</p><p>In an implementation that includes EL2, the permitted <instruction>LDC</instruction> access to <xref linkend=\"AArch32.dbgdtrtxint\">DBGDTRTXint</xref> can be trapped to Hyp mode, meaning that an attempt to execute an <instruction>LDC</instruction> instruction in a Non-secure mode other than Hyp mode, that would be permitted in the absence of the Hyp trap controls, generates a Hyp Trap exception.  For more information, see <xref linkend=\"BEICAABI\">Trapping general Non-secure System register accesses to debug registers</xref>.</p><p>For simplicity, the <instruction>LDC</instruction> pseudocode does not show this possible trap to Hyp mode.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "LDC":
            return {
                "tooltip": "Load data to System register (literal) calculates an address from the PC value and an immediate offset, loads a word from memory, and writes it to the DBGDTRTXint System register. For information about memory accesses see Memory accesses.",
                "html": "<p>Load data to System register (literal) calculates an address from the PC value and an immediate offset, loads a word from memory, and writes it to the <xref linkend=\"AArch32.dbgdtrtxint\">DBGDTRTXint</xref> System register. For information about memory accesses see <xref linkend=\"Chddjfjf\">Memory accesses</xref>.</p><p>In an implementation that includes EL2, the permitted <instruction>LDC</instruction> access to <xref linkend=\"AArch32.dbgdtrtxint\">DBGDTRTXint</xref> can be trapped to Hyp mode, meaning that an attempt to execute an <instruction>LDC</instruction> instruction in a Non-secure mode other than Hyp mode, that would be permitted in the absence of the Hyp trap controls, generates a Hyp Trap exception.  For more information, see <xref linkend=\"BEICAABI\">Trapping general Non-secure System register accesses to debug registers</xref>.</p><p>For simplicity, the <instruction>LDC</instruction> pseudocode does not show this possible trap to Hyp mode.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "LDM":
        case "LDMFD":
        case "LDMIA":
            return {
                "tooltip": "Load Multiple (Increment After, Full Descending) loads multiple registers from consecutive memory locations using an address from a base register. The consecutive memory locations start at this address, and the address just above the highest of those locations can optionally be written back to the base register.",
                "html": "<p>Load Multiple (Increment After, Full Descending) loads multiple registers from consecutive memory locations using an address from a base register. The consecutive memory locations start at this address, and the address just above the highest of those locations can optionally be written back to the base register.</p><p>The lowest-numbered register is loaded from the lowest memory address, through to the highest-numbered register from the highest memory address. See also <xref linkend=\"CHDDBEDG\">Encoding of lists of general-purpose registers and the PC</xref>.</p><p>Armv8.2 permits the deprecation of some Load Multiple ordering behaviors in AArch32 state, for more information see <xref linkend=\"v8.2.LSMAOC\">FEAT_LSMAOC</xref>. The registers loaded can include the PC, causing a branch to a loaded address. This is an interworking branch, see <xref linkend=\"BEICJFEH\">Pseudocode description of operations on the AArch32 general-purpose registers and the PC</xref>. Related system instructions are <xref linkend=\"A32T32-base.instructions.LDM_u\">LDM (User registers)</xref> and <xref linkend=\"A32T32-base.instructions.LDM_e\">LDM (exception return)</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "LDM":
            return {
                "tooltip": "Load Multiple (exception return) loads multiple registers from consecutive memory locations using an address from a base register. The SPSR of the current mode is copied to the CPSR. An address adjusted by the size of the data loaded can optionally be written back to the base register.",
                "html": "<p>Load Multiple (exception return) loads multiple registers from consecutive memory locations using an address from a base register. The <xref linkend=\"CHDDAABB\">SPSR</xref> of the current mode is copied to the <xref linkend=\"CIHJBHJA\">CPSR</xref>. An address adjusted by the size of the data loaded can optionally be written back to the base register.</p><p>The registers loaded include the PC. The word loaded for the PC is treated as an address and a branch occurs to that address.</p><p>The PE checks the encoding that is copied to the <xref linkend=\"CIHJBHJA\">CPSR</xref> for an illegal return event. See <xref linkend=\"CHDDDJDB\">Illegal return events from AArch32 state</xref>.</p><p>Load Multiple (exception return) is:</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "LDM":
            return {
                "tooltip": "In an EL1 mode other than System mode, Load Multiple (User registers) loads multiple User mode registers from consecutive memory locations using an address from a base register. The registers loaded cannot include the PC. The PE reads the base register value normally, using the current mode to determine the correct Banked version of the register. This instruction cannot writeback to the base register.",
                "html": "<p>In an EL1 mode other than System mode, Load Multiple (User registers) loads multiple User mode registers from consecutive memory locations using an address from a base register. The registers loaded cannot include the PC. The PE reads the base register value normally, using the current mode to determine the correct Banked version of the register. This instruction cannot writeback to the base register.</p><p>Load Multiple (User registers) is <arm-defined-word>undefined</arm-defined-word> in Hyp mode, and <arm-defined-word>unpredictable</arm-defined-word> in User and System modes.</p><p>Armv8.2 permits the deprecation of some Load Multiple ordering behaviors in AArch32 state, for more information see <xref linkend=\"v8.2.LSMAOC\">FEAT_LSMAOC</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "LDMDA":
        case "LDMFA":
            return {
                "tooltip": "Load Multiple Decrement After (Full Ascending) loads multiple registers from consecutive memory locations using an address from a base register. The consecutive memory locations end at this address, and the address just below the lowest of those locations can optionally be written back to the base register.",
                "html": "<p>Load Multiple Decrement After (Full Ascending) loads multiple registers from consecutive memory locations using an address from a base register. The consecutive memory locations end at this address, and the address just below the lowest of those locations can optionally be written back to the base register.</p><p>The lowest-numbered register is loaded from the lowest memory address, through to the highest-numbered register from the highest memory address. See also <xref linkend=\"CHDDBEDG\">Encoding of lists of general-purpose registers and the PC</xref>.</p><p>Armv8.2 permits the deprecation of some Load Multiple ordering behaviors in AArch32 state, for more information see <xref linkend=\"v8.2.LSMAOC\">FEAT_LSMAOC</xref>. The registers loaded can include the PC, causing a branch to a loaded address. This is an interworking branch, see <xref linkend=\"BEICJFEH\">Pseudocode description of operations on the AArch32 general-purpose registers and the PC</xref>. Related system instructions are <xref linkend=\"A32T32-base.instructions.LDM_u\">LDM (User registers)</xref> and <xref linkend=\"A32T32-base.instructions.LDM_e\">LDM (exception return)</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "LDMDB":
        case "LDMEA":
            return {
                "tooltip": "Load Multiple Decrement Before (Empty Ascending) loads multiple registers from consecutive memory locations using an address from a base register. The consecutive memory locations end just below this address, and the address of the lowest of those locations can optionally be written back to the base register.",
                "html": "<p>Load Multiple Decrement Before (Empty Ascending) loads multiple registers from consecutive memory locations using an address from a base register. The consecutive memory locations end just below this address, and the address of the lowest of those locations can optionally be written back to the base register.</p><p>The lowest-numbered register is loaded from the lowest memory address, through to the highest-numbered register from the highest memory address. See also <xref linkend=\"CHDDBEDG\">Encoding of lists of general-purpose registers and the PC</xref>.</p><p>Armv8.2 permits the deprecation of some Load Multiple ordering behaviors in AArch32 state, for more information see <xref linkend=\"v8.2.LSMAOC\">FEAT_LSMAOC</xref>. The registers loaded can include the PC, causing a branch to a loaded address. This is an interworking branch, see <xref linkend=\"BEICJFEH\">Pseudocode description of operations on the AArch32 general-purpose registers and the PC</xref>. Related system instructions are <xref linkend=\"A32T32-base.instructions.LDM_u\">LDM (User registers)</xref> and <xref linkend=\"A32T32-base.instructions.LDM_e\">LDM (exception return)</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "LDMED":
        case "LDMIB":
            return {
                "tooltip": "Load Multiple Increment Before (Empty Descending) loads multiple registers from consecutive memory locations using an address from a base register. The consecutive memory locations start just above this address, and the address of the last of those locations can optionally be written back to the base register.",
                "html": "<p>Load Multiple Increment Before (Empty Descending) loads multiple registers from consecutive memory locations using an address from a base register. The consecutive memory locations start just above this address, and the address of the last of those locations can optionally be written back to the base register.</p><p>The lowest-numbered register is loaded from the lowest memory address, through to the highest-numbered register from the highest memory address. See also <xref linkend=\"CHDDBEDG\">Encoding of lists of general-purpose registers and the PC</xref>.</p><p>Armv8.2 permits the deprecation of some Load Multiple ordering behaviors in AArch32 state, for more information see <xref linkend=\"v8.2.LSMAOC\">FEAT_LSMAOC</xref>. The registers loaded can include the PC, causing a branch to a loaded address. This is an interworking branch, see <xref linkend=\"BEICJFEH\">Pseudocode description of operations on the AArch32 general-purpose registers and the PC</xref>. Related system instructions are <xref linkend=\"A32T32-base.instructions.LDM_u\">LDM (User registers)</xref> and <xref linkend=\"A32T32-base.instructions.LDM_e\">LDM (exception return)</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "LDR":
            return {
                "tooltip": "Load Register (immediate) calculates an address from a base register value and an immediate offset, loads a word from memory, and writes it to a register. It can use offset, post-indexed, or pre-indexed addressing. For information about memory accesses see Memory accesses.",
                "html": "<p>Load Register (immediate) calculates an address from a base register value and an immediate offset, loads a word from memory, and writes it to a register. It can use offset, post-indexed, or pre-indexed addressing. For information about memory accesses see <xref linkend=\"Chddjfjf\">Memory accesses</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "LDR":
            return {
                "tooltip": "Load Register (literal) calculates an address from the PC value and an immediate offset, loads a word from memory, and writes it to a register. For information about memory accesses see Memory accesses.",
                "html": "<p>Load Register (literal) calculates an address from the PC value and an immediate offset, loads a word from memory, and writes it to a register. For information about memory accesses see <xref linkend=\"Chddjfjf\">Memory accesses</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "LDR":
            return {
                "tooltip": "Load Register (register) calculates an address from a base register value and an offset register value, loads a word from memory, and writes it to a register. The offset register value can optionally be shifted. For information about memory accesses, see Memory accesses.",
                "html": "<p>Load Register (register) calculates an address from a base register value and an offset register value, loads a word from memory, and writes it to a register. The offset register value can optionally be shifted. For information about memory accesses, see <xref linkend=\"Chddjfjf\">Memory accesses</xref>.</p><p>The T32 form of <instruction>LDR</instruction> (register) does not support register writeback.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "LDRB":
            return {
                "tooltip": "Load Register Byte (immediate) calculates an address from a base register value and an immediate offset, loads a byte from memory, zero-extends it to form a 32-bit word, and writes it to a register. It can use offset, post-indexed, or pre-indexed addressing.  For information about memory accesses see Memory accesses.",
                "html": "<p>Load Register Byte (immediate) calculates an address from a base register value and an immediate offset, loads a byte from memory, zero-extends it to form a 32-bit word, and writes it to a register. It can use offset, post-indexed, or pre-indexed addressing.  For information about memory accesses see <xref linkend=\"Chddjfjf\">Memory accesses</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "LDRB":
            return {
                "tooltip": "Load Register Byte (literal) calculates an address from the PC value and an immediate offset, loads a byte from memory, zero-extends it to form a 32-bit word, and writes it to a register. For information about memory accesses see Memory accesses.",
                "html": "<p>Load Register Byte (literal) calculates an address from the PC value and an immediate offset, loads a byte from memory, zero-extends it to form a 32-bit word, and writes it to a register. For information about memory accesses see <xref linkend=\"Chddjfjf\">Memory accesses</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "LDRB":
            return {
                "tooltip": "Load Register Byte (register) calculates an address from a base register value and an offset register value, loads a byte from memory, zero-extends it to form a 32-bit word, and writes it to a register. The offset register value can optionally be shifted.  For information about memory accesses see Memory accesses.",
                "html": "<p>Load Register Byte (register) calculates an address from a base register value and an offset register value, loads a byte from memory, zero-extends it to form a 32-bit word, and writes it to a register. The offset register value can optionally be shifted.  For information about memory accesses see <xref linkend=\"Chddjfjf\">Memory accesses</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "LDRBT":
            return {
                "tooltip": "Load Register Byte Unprivileged loads a byte from memory, zero-extends it to form a 32-bit word, and writes it to a register. For information about memory accesses see Memory accesses.",
                "html": "<p>Load Register Byte Unprivileged loads a byte from memory, zero-extends it to form a 32-bit word, and writes it to a register. For information about memory accesses see <xref linkend=\"Chddjfjf\">Memory accesses</xref>.</p><p>The memory access is restricted as if the PE were running in User mode. This makes no difference if the PE is actually running in User mode.</p><p><instruction>LDRBT</instruction> is <arm-defined-word>unpredictable</arm-defined-word> in Hyp mode.</p><p>The T32 instruction uses an offset addressing mode, that calculates the address used for the memory access from a base register value and an immediate offset, and leaves the base register unchanged.</p><p>The A32 instruction uses a post-indexed addressing mode, that uses a base register value as the address for the memory access, and calculates a new address from a base register value and an offset and writes it back to the base register. The offset can be an immediate value or an optionally-shifted register value.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "LDRD":
            return {
                "tooltip": "Load Register Dual (immediate) calculates an address from a base register value and an immediate offset, loads two words from memory, and writes them to two registers. It can use offset, post-indexed, or pre-indexed addressing. For information about memory accesses see Memory accesses.",
                "html": "<p>Load Register Dual (immediate) calculates an address from a base register value and an immediate offset, loads two words from memory, and writes them to two registers. It can use offset, post-indexed, or pre-indexed addressing. For information about memory accesses see <xref linkend=\"Chddjfjf\">Memory accesses</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "LDRD":
            return {
                "tooltip": "Load Register Dual (literal) calculates an address from the PC value and an immediate offset, loads two words from memory, and writes them to two registers. For information about memory accesses see Memory accesses.",
                "html": "<p>Load Register Dual (literal) calculates an address from the PC value and an immediate offset, loads two words from memory, and writes them to two registers. For information about memory accesses see <xref linkend=\"Chddjfjf\">Memory accesses</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "LDRD":
            return {
                "tooltip": "Load Register Dual (register) calculates an address from a base register value and a register offset, loads two words from memory, and writes them to two registers. It can use offset, post-indexed, or pre-indexed addressing. For information about memory accesses see Memory accesses.",
                "html": "<p>Load Register Dual (register) calculates an address from a base register value and a register offset, loads two words from memory, and writes them to two registers. It can use offset, post-indexed, or pre-indexed addressing. For information about memory accesses see <xref linkend=\"Chddjfjf\">Memory accesses</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "LDREX":
            return {
                "tooltip": "Load Register Exclusive calculates an address from a base register value and an immediate offset, loads a word from memory, writes it to a register and",
                "html": "<p>Load Register Exclusive calculates an address from a base register value and an immediate offset, loads a word from memory, writes it to a register and:</p><p>For more information about support for shared memory see <xref linkend=\"CEGDAEAG\">Synchronization and semaphores</xref>. For information about memory accesses see <xref linkend=\"Chddjfjf\">Memory accesses</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "LDREXB":
            return {
                "tooltip": "Load Register Exclusive Byte derives an address from a base register value, loads a byte from memory, zero-extends it to form a 32-bit word, writes it to a register and",
                "html": "<p>Load Register Exclusive Byte derives an address from a base register value, loads a byte from memory, zero-extends it to form a 32-bit word, writes it to a register and:</p><p>For more information about support for shared memory see <xref linkend=\"CEGDAEAG\">Synchronization and semaphores</xref>. For information about memory accesses see <xref linkend=\"Chddjfjf\">Memory accesses</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "LDREXD":
            return {
                "tooltip": "Load Register Exclusive Doubleword derives an address from a base register value, loads a 64-bit doubleword from memory, writes it to two registers and",
                "html": "<p>Load Register Exclusive Doubleword derives an address from a base register value, loads a 64-bit doubleword from memory, writes it to two registers and:</p><p>For more information about support for shared memory see <xref linkend=\"CEGDAEAG\">Synchronization and semaphores</xref>. For information about memory accesses see <xref linkend=\"Chddjfjf\">Memory accesses</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "LDREXH":
            return {
                "tooltip": "Load Register Exclusive Halfword derives an address from a base register value, loads a halfword from memory, zero-extends it to form a 32-bit word, writes it to a register and",
                "html": "<p>Load Register Exclusive Halfword derives an address from a base register value, loads a halfword from memory, zero-extends it to form a 32-bit word, writes it to a register and:</p><p>For more information about support for shared memory see <xref linkend=\"CEGDAEAG\">Synchronization and semaphores</xref>. For information about memory accesses see <xref linkend=\"Chddjfjf\">Memory accesses</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "LDRH":
            return {
                "tooltip": "Load Register Halfword (immediate) calculates an address from a base register value and an immediate offset, loads a halfword from memory, zero-extends it to form a 32-bit word, and writes it to a register. It can use offset, post-indexed, or pre-indexed addressing.  For information about memory accesses see Memory accesses.",
                "html": "<p>Load Register Halfword (immediate) calculates an address from a base register value and an immediate offset, loads a halfword from memory, zero-extends it to form a 32-bit word, and writes it to a register. It can use offset, post-indexed, or pre-indexed addressing.  For information about memory accesses see <xref linkend=\"Chddjfjf\">Memory accesses</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "LDRH":
            return {
                "tooltip": "Load Register Halfword (literal) calculates an address from the PC value and an immediate offset, loads a halfword from memory, zero-extends it to form a 32-bit word, and writes it to a register. For information about memory accesses see Memory accesses.",
                "html": "<p>Load Register Halfword (literal) calculates an address from the PC value and an immediate offset, loads a halfword from memory, zero-extends it to form a 32-bit word, and writes it to a register. For information about memory accesses see <xref linkend=\"Chddjfjf\">Memory accesses</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "LDRH":
            return {
                "tooltip": "Load Register Halfword (register) calculates an address from a base register value and an offset register value, loads a halfword from memory, zero-extends it to form a 32-bit word, and writes it to a register. The offset register value can be shifted left by 0, 1, 2, or 3 bits. For information about memory accesses see Memory accesses.",
                "html": "<p>Load Register Halfword (register) calculates an address from a base register value and an offset register value, loads a halfword from memory, zero-extends it to form a 32-bit word, and writes it to a register. The offset register value can be shifted left by 0, 1, 2, or 3 bits. For information about memory accesses see <xref linkend=\"Chddjfjf\">Memory accesses</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "LDRHT":
            return {
                "tooltip": "Load Register Halfword Unprivileged loads a halfword from memory, zero-extends it to form a 32-bit word, and writes it to a register. For information about memory accesses see Memory accesses.",
                "html": "<p>Load Register Halfword Unprivileged loads a halfword from memory, zero-extends it to form a 32-bit word, and writes it to a register. For information about memory accesses see <xref linkend=\"Chddjfjf\">Memory accesses</xref>.</p><p>The memory access is restricted as if the PE were running in User mode. This makes no difference if the PE is actually running in User mode.</p><p><instruction>LDRHT</instruction> is <arm-defined-word>unpredictable</arm-defined-word> in Hyp mode.</p><p>The T32 instruction uses an offset addressing mode, that calculates the address used for the memory access from a base register value and an immediate offset, and leaves the base register unchanged.</p><p>The A32 instruction uses a post-indexed addressing mode, that uses a base register value as the address for the memory access, and calculates a new address from a base register value and an offset and writes it back to the base register. The offset can be an immediate value or a register value.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "LDRSB":
            return {
                "tooltip": "Load Register Signed Byte (immediate) calculates an address from a base register value and an immediate offset, loads a byte from memory, sign-extends it to form a 32-bit word, and writes it to a register. It can use offset, post-indexed, or pre-indexed addressing.  For information about memory accesses see Memory accesses.",
                "html": "<p>Load Register Signed Byte (immediate) calculates an address from a base register value and an immediate offset, loads a byte from memory, sign-extends it to form a 32-bit word, and writes it to a register. It can use offset, post-indexed, or pre-indexed addressing.  For information about memory accesses see <xref linkend=\"Chddjfjf\">Memory accesses</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "LDRSB":
            return {
                "tooltip": "Load Register Signed Byte (literal) calculates an address from the PC value and an immediate offset, loads a byte from memory, sign-extends it to form a 32-bit word, and writes it to a register. For information about memory accesses see Memory accesses.",
                "html": "<p>Load Register Signed Byte (literal) calculates an address from the PC value and an immediate offset, loads a byte from memory, sign-extends it to form a 32-bit word, and writes it to a register. For information about memory accesses see <xref linkend=\"Chddjfjf\">Memory accesses</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "LDRSB":
            return {
                "tooltip": "Load Register Signed Byte (register) calculates an address from a base register value and an offset register value, loads a byte from memory, sign-extends it to form a 32-bit word, and writes it to a register. The offset register value can be shifted left by 0, 1, 2, or 3 bits. For information about memory accesses see Memory accesses.",
                "html": "<p>Load Register Signed Byte (register) calculates an address from a base register value and an offset register value, loads a byte from memory, sign-extends it to form a 32-bit word, and writes it to a register. The offset register value can be shifted left by 0, 1, 2, or 3 bits. For information about memory accesses see <xref linkend=\"Chddjfjf\">Memory accesses</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "LDRSBT":
            return {
                "tooltip": "Load Register Signed Byte Unprivileged loads a byte from memory, sign-extends it to form a 32-bit word, and writes it to a register. For information about memory accesses see Memory accesses.",
                "html": "<p>Load Register Signed Byte Unprivileged loads a byte from memory, sign-extends it to form a 32-bit word, and writes it to a register. For information about memory accesses see <xref linkend=\"Chddjfjf\">Memory accesses</xref>.</p><p>The memory access is restricted as if the PE were running in User mode. This makes no difference if the PE is actually running in User mode.</p><p><instruction>LDRSBT</instruction> is <arm-defined-word>unpredictable</arm-defined-word> in Hyp mode.</p><p>The T32 instruction uses an offset addressing mode, that calculates the address used for the memory access from a base register value and an immediate offset, and leaves the base register unchanged.</p><p>The A32 instruction uses a post-indexed addressing mode, that uses a base register value as the address for the memory access, and calculates a new address from a base register value and an offset and writes it back to the base register. The offset can be an immediate value or a register value.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "LDRSH":
            return {
                "tooltip": "Load Register Signed Halfword (immediate) calculates an address from a base register value and an immediate offset, loads a halfword from memory, sign-extends it to form a 32-bit word, and writes it to a register. It can use offset, post-indexed, or pre-indexed addressing.  For information about memory accesses see Memory accesses.",
                "html": "<p>Load Register Signed Halfword (immediate) calculates an address from a base register value and an immediate offset, loads a halfword from memory, sign-extends it to form a 32-bit word, and writes it to a register. It can use offset, post-indexed, or pre-indexed addressing.  For information about memory accesses see <xref linkend=\"Chddjfjf\">Memory accesses</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "LDRSH":
            return {
                "tooltip": "Load Register Signed Halfword (literal) calculates an address from the PC value and an immediate offset, loads a halfword from memory, sign-extends it to form a 32-bit word, and writes it to a register. For information about memory accesses see Memory accesses.",
                "html": "<p>Load Register Signed Halfword (literal) calculates an address from the PC value and an immediate offset, loads a halfword from memory, sign-extends it to form a 32-bit word, and writes it to a register. For information about memory accesses see <xref linkend=\"Chddjfjf\">Memory accesses</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "LDRSH":
            return {
                "tooltip": "Load Register Signed Halfword (register) calculates an address from a base register value and an offset register value, loads a halfword from memory, sign-extends it to form a 32-bit word, and writes it to a register. The offset register value can be shifted left by 0, 1, 2, or 3 bits. For information about memory accesses see Memory accesses.",
                "html": "<p>Load Register Signed Halfword (register) calculates an address from a base register value and an offset register value, loads a halfword from memory, sign-extends it to form a 32-bit word, and writes it to a register. The offset register value can be shifted left by 0, 1, 2, or 3 bits. For information about memory accesses see <xref linkend=\"Chddjfjf\">Memory accesses</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "LDRSHT":
            return {
                "tooltip": "Load Register Signed Halfword Unprivileged loads a halfword from memory, sign-extends it to form a 32-bit word, and writes it to a register. For information about memory accesses see Memory accesses.",
                "html": "<p>Load Register Signed Halfword Unprivileged loads a halfword from memory, sign-extends it to form a 32-bit word, and writes it to a register. For information about memory accesses see <xref linkend=\"Chddjfjf\">Memory accesses</xref>.</p><p>The memory access is restricted as if the PE were running in User mode. This makes no difference if the PE is actually running in User mode.</p><p><instruction>LDRSHT</instruction> is <arm-defined-word>unpredictable</arm-defined-word> in Hyp mode.</p><p>The T32 instruction uses an offset addressing mode, that calculates the address used for the memory access from a base register value and an immediate offset, and leaves the base register unchanged.</p><p>The A32 instruction uses a post-indexed addressing mode, that uses a base register value as the address for the memory access, and calculates a new address from a base register value and an offset and writes it back to the base register. The offset can be an immediate value or a register value.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "LDRT":
            return {
                "tooltip": "Load Register Unprivileged loads a word from memory, and writes it to a register. For information about memory accesses see Memory accesses.",
                "html": "<p>Load Register Unprivileged loads a word from memory, and writes it to a register. For information about memory accesses see <xref linkend=\"Chddjfjf\">Memory accesses</xref>.</p><p>The memory access is restricted as if the PE were running in User mode. This makes no difference if the PE is actually running in User mode.</p><p><instruction>LDRT</instruction> is <arm-defined-word>unpredictable</arm-defined-word> in Hyp mode.</p><p>The T32 instruction uses an offset addressing mode, that calculates the address used for the memory access from a base register value and an immediate offset, and leaves the base register unchanged.</p><p>The A32 instruction uses a post-indexed addressing mode, that uses a base register value as the address for the memory access, and calculates a new address from a base register value and an offset and writes it back to the base register. The offset can be an immediate value or an optionally-shifted register value.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "LSL":
            return {
                "tooltip": "Logical Shift Left (immediate) shifts a register value left by an immediate number of bits, shifting in zeros, and writes the result to the destination register.",
                "html": "<p>Logical Shift Left (immediate) shifts a register value left by an immediate number of bits, shifting in zeros, and writes the result to the destination register.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "LSLS":
            return {
                "tooltip": "Logical Shift Left, setting flags (immediate) shifts a register value left by an immediate number of bits, shifting in zeros, and writes the result to the destination register.",
                "html": "<p>Logical Shift Left, setting flags (immediate) shifts a register value left by an immediate number of bits, shifting in zeros, and writes the result to the destination register.</p><p>If the destination register is not the PC, this instruction updates the condition flags based on the result.</p><p>The field descriptions for <syntax>&lt;Rd&gt;</syntax> identify the encodings where the PC is permitted as the destination register. Arm deprecates any use of these encodings. However, when the destination register is the PC:</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "LSR":
            return {
                "tooltip": "Logical Shift Right (immediate) shifts a register value right by an immediate number of bits, shifting in zeros, and writes the result to the destination register.",
                "html": "<p>Logical Shift Right (immediate) shifts a register value right by an immediate number of bits, shifting in zeros, and writes the result to the destination register.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "LSRS":
            return {
                "tooltip": "Logical Shift Right, setting flags (immediate) shifts a register value right by an immediate number of bits, shifting in zeros, and writes the result to the destination register.",
                "html": "<p>Logical Shift Right, setting flags (immediate) shifts a register value right by an immediate number of bits, shifting in zeros, and writes the result to the destination register.</p><p>If the destination register is not the PC, this instruction updates the condition flags based on the result.</p><p>The field descriptions for <syntax>&lt;Rd&gt;</syntax> identify the encodings where the PC is permitted as the destination register. Arm deprecates any use of these encodings. However, when the destination register is the PC:</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "MCR":
            return {
                "tooltip": "Move to System register from general-purpose register or execute a System instruction. This instruction copies the value of a general-purpose register to a System register, or executes a System instruction.",
                "html": "<p>Move to System register from general-purpose register or execute a System instruction. This instruction copies the value of a general-purpose register to a System register, or executes a System instruction.</p><p>The System register and System instruction descriptions identify valid encodings for this instruction. Other encodings are <arm-defined-word>undefined</arm-defined-word>. For more information see <xref linkend=\"CFIDGFBF\">About the AArch32 System register interface</xref> and <xref linkend=\"BABDFCJB\">General behavior of System registers</xref>.</p><p>In an implementation that includes EL2, <instruction>MCR</instruction> accesses to System registers can be trapped to Hyp mode, meaning that an attempt to execute an <instruction>MCR</instruction> instruction in a Non-secure mode other than Hyp mode, that would be permitted in the absence of the Hyp trap controls, generates a Hyp Trap exception.  For more information, see <xref linkend=\"BEIDFAFD\">EL2 configurable instruction enables, disables, and traps</xref>.</p><p>Because of the range of possible traps to Hyp mode, the <instruction>MCR</instruction> pseudocode does not show these possible traps.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "MCRR":
            return {
                "tooltip": "Move to System register from two general-purpose registers. This instruction copies the values of two general-purpose registers to a System register.",
                "html": "<p>Move to System register from two general-purpose registers. This instruction copies the values of two general-purpose registers to a System register.</p><p>The System register descriptions identify valid encodings for this instruction. Other encodings are <arm-defined-word>undefined</arm-defined-word>. For more information see <xref linkend=\"CFIDGFBF\">About the AArch32 System register interface</xref> and <xref linkend=\"BABDFCJB\">General behavior of System registers</xref>.</p><p>In an implementation that includes EL2, <instruction>MCRR</instruction> accesses to System registers can be trapped to Hyp mode, meaning that an attempt to execute an <instruction>MCRR</instruction> instruction in a Non-secure mode other than Hyp mode, that would be permitted in the absence of the Hyp trap controls, generates a Hyp Trap exception.  For more information, see <xref linkend=\"BEIDFAFD\">EL2 configurable instruction enables, disables, and traps</xref>.</p><p>Because of the range of possible traps to Hyp mode, the <instruction>MCRR</instruction> pseudocode does not show these possible traps.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "MLA":
        case "MLAS":
            return {
                "tooltip": "Multiply Accumulate multiplies two register values, and adds a third register value. The least significant 32 bits of the result are written to the destination register. These 32 bits do not depend on whether the source register values are considered to be signed values or unsigned values.",
                "html": "<p>Multiply Accumulate multiplies two register values, and adds a third register value. The least significant 32 bits of the result are written to the destination register. These 32 bits do not depend on whether the source register values are considered to be signed values or unsigned values.</p><p>In an A32 instruction, the condition flags can optionally be updated based on the result. Use of this option adversely affects performance on many implementations.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "MLS":
            return {
                "tooltip": "Multiply and Subtract multiplies two register values, and subtracts the product from a third register value. The least significant 32 bits of the result are written to the destination register. These 32 bits do not depend on whether the source register values are considered to be signed values or unsigned values.",
                "html": "<p>Multiply and Subtract multiplies two register values, and subtracts the product from a third register value. The least significant 32 bits of the result are written to the destination register. These 32 bits do not depend on whether the source register values are considered to be signed values or unsigned values.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "MOV":
        case "MOVS":
            return {
                "tooltip": "Move (immediate) writes an immediate value to the destination register.",
                "html": "<p>Move (immediate) writes an immediate value to the destination register.</p><p>If the destination register is not the PC, the MOVS variant of the instruction updates the condition flags based on the result.</p><p>The field descriptions for <syntax>&lt;Rd&gt;</syntax> identify the encodings where the PC is permitted as the destination register. Arm deprecates any use of these encodings. However, when the destination register is the PC:</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "MOV":
        case "MOVS":
            return {
                "tooltip": "Move (register) copies a value from a register to the destination register.",
                "html": "<p>Move (register) copies a value from a register to the destination register.</p><p>If the destination register is not the PC, the MOVS variant of the instruction updates the condition flags based on the result.</p><p>The field descriptions for <syntax>&lt;Rd&gt;</syntax> identify the encodings where the PC is permitted as the destination register. If the destination register is the PC:</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "MOV":
        case "MOVS":
            return {
                "tooltip": "Move (register-shifted register) copies a register-shifted register value to the destination register. It can optionally update the condition flags based on the value.",
                "html": "<p>Move (register-shifted register) copies a register-shifted register value to the destination register. It can optionally update the condition flags based on the value.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "MOVT":
            return {
                "tooltip": "Move Top writes an immediate value to the top halfword of the destination register. It does not affect the contents of the bottom halfword.",
                "html": "<p>Move Top writes an immediate value to the top halfword of the destination register. It does not affect the contents of the bottom halfword.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "MRC":
            return {
                "tooltip": "Move to general-purpose register from System register. This instruction copies the value of a System register to a general-purpose register.",
                "html": "<p>Move to general-purpose register from System register. This instruction copies the value of a System register to a general-purpose register.</p><p>The System register descriptions identify valid encodings for this instruction. Other encodings are <arm-defined-word>undefined</arm-defined-word>. For more information see <xref linkend=\"CFIDGFBF\">About the AArch32 System register interface</xref> and <xref linkend=\"BABDFCJB\">General behavior of System registers</xref>.</p><p>In an implementation that includes EL2, <instruction>MRC</instruction> accesses to system control registers can be trapped to Hyp mode, meaning that an attempt to execute an <instruction>MRC</instruction> instruction in a Non-secure mode other than Hyp mode, that would be permitted in the absence of the Hyp trap controls, generates a Hyp Trap exception.  For more information, see <xref linkend=\"BEIDFAFD\">EL2 configurable instruction enables, disables, and traps</xref>.</p><p>Because of the range of possible traps to Hyp mode, the <instruction>MRC</instruction> pseudocode does not show these possible traps.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "MRRC":
            return {
                "tooltip": "Move to two general-purpose registers from System register. This instruction copies the value of a System register to two general-purpose registers.",
                "html": "<p>Move to two general-purpose registers from System register. This instruction copies the value of a System register to two general-purpose registers.</p><p>The System register descriptions identify valid encodings for this instruction. Other encodings are <arm-defined-word>undefined</arm-defined-word>. For more information see <xref linkend=\"CFIDGFBF\">About the AArch32 System register interface</xref> and <xref linkend=\"BABDFCJB\">General behavior of System registers</xref>.</p><p>In an implementation that includes EL2, <instruction>MRRC</instruction> accesses to System registers can be trapped to Hyp mode, meaning that an attempt to execute an <instruction>MRRC</instruction> instruction in a Non-secure mode other than Hyp mode, that would be permitted in the absence of the Hyp trap controls, generates a Hyp Trap exception.  For more information, see <xref linkend=\"BEIDFAFD\">EL2 configurable instruction enables, disables, and traps</xref>.</p><p>Because of the range of possible traps to Hyp mode, the <instruction>MRRC</instruction> pseudocode does not show these possible traps.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "MRS":
            return {
                "tooltip": "Move Special register to general-purpose register moves the value of the APSR, CPSR, or SPSR_<current_mode> into a general-purpose register.",
                "html": "<p>Move Special register to general-purpose register moves the value of the <xref linkend=\"CJAGBHBH\">APSR</xref>, <xref linkend=\"CIHJBHJA\">CPSR</xref>, or <xref linkend=\"CHDDAABB\">SPSR</xref>_&lt;current_mode&gt; into a general-purpose register.</p><p>Arm recommends the <value>APSR</value> form when only the N, Z, C, V, Q, and GE[3:0] bits are being written. For more information, see <xref linkend=\"CJAGBHBH\">APSR</xref>.</p><p>An <instruction>MRS</instruction> that accesses the <xref linkend=\"CHDDAABB\">SPSRs</xref> is <arm-defined-word>unpredictable</arm-defined-word> if executed in User mode or System mode.</p><p>An <instruction>MRS</instruction> that is executed in User mode and accesses the <xref linkend=\"CIHJBHJA\">CPSR</xref> returns an <arm-defined-word>unknown</arm-defined-word> value for the <xref linkend=\"CIHJBHJA\">CPSR</xref>.{E, A, I, F, M} fields.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "MRS":
            return {
                "tooltip": "Move to Register from Banked or Special register moves the value from the Banked general-purpose register or Saved Program Status Registers (SPSRs) of the specified mode, or the value of ELR_hyp, to a general-purpose register.",
                "html": "<p>Move to Register from Banked or Special register moves the value from the Banked general-purpose register or <xref linkend=\"CHDDAABB\">Saved Program Status Registers (SPSRs)</xref> of the specified mode, or the value of <xref linkend=\"BEIJHFCF\">ELR_hyp</xref>, to a general-purpose register.</p><p><instruction>MRS</instruction> (Banked register) is <arm-defined-word>unpredictable</arm-defined-word> if executed in User mode.</p><p>When EL3 is using AArch64, if an MRS (Banked register) instruction that is executed in a Secure EL1 mode would access SPSR_mon, SP_mon, or LR_mon, it is trapped to EL3.</p><p>The effect of using an <instruction>MRS</instruction> (Banked register) instruction with a register argument that is not valid for the current mode is <arm-defined-word>unpredictable</arm-defined-word>. For more information see <xref linkend=\"CHDFDJDA\">Usage restrictions on the Banked register transfer instructions</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "MSR":
            return {
                "tooltip": "Move to Banked or Special register from general-purpose register moves the value of a general-purpose register to the Banked general-purpose register or Saved Program Status Registers (SPSRs) of the specified mode, or to ELR_hyp.",
                "html": "<p>Move to Banked or Special register from general-purpose register moves the value of a general-purpose register to the Banked general-purpose register or <xref linkend=\"CHDDAABB\">Saved Program Status Registers (SPSRs)</xref> of the specified mode, or to <xref linkend=\"BEIJHFCF\">ELR_hyp</xref>.</p><p><instruction>MSR</instruction> (Banked register) is <arm-defined-word>unpredictable</arm-defined-word> if executed in User mode.</p><p>When EL3 is using AArch64, if an MSR (Banked register) instruction that is executed in a Secure EL1 mode would access SPSR_mon, SP_mon, or LR_mon, it is trapped to EL3.</p><p>The effect of using an <instruction>MSR</instruction> (Banked register) instruction with a register argument that is not valid for the current mode is <arm-defined-word>unpredictable</arm-defined-word>. For more information see <xref linkend=\"CHDFDJDA\">Usage restrictions on the Banked register transfer instructions</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "MSR":
            return {
                "tooltip": "Move immediate value to Special register moves selected bits of an immediate value to the corresponding bits in the APSR, CPSR, or SPSR_<current_mode>.",
                "html": "<p>Move immediate value to Special register moves selected bits of an immediate value to the corresponding bits in the <xref linkend=\"CJAGBHBH\">APSR</xref>, <xref linkend=\"CIHJBHJA\">CPSR</xref>, or <xref linkend=\"CHDDAABB\">SPSR</xref>_&lt;current_mode&gt;.</p><p>Because of the Do-Not-Modify nature of its reserved bits, the immediate form of <instruction>MSR</instruction> is normally only useful at the Application level for writing to <value>APSR_nzcvq</value> (<value>CPSR_f</value>).</p><p>If an <instruction>MSR</instruction> (immediate) moves selected bits of an immediate value to the <xref linkend=\"CIHJBHJA\">CPSR</xref>, the PE checks whether the value being written to <xref linkend=\"BEIDIGBH\">PSTATE</xref>.M is legal. See <xref linkend=\"CHDDFIGE\">Illegal changes to PSTATE.M</xref>.</p><p>An <instruction>MSR</instruction> (immediate) executed in User mode:</p><p>An <instruction>MSR</instruction> (immediate) executed in System mode is <arm-defined-word>constrained unpredictable</arm-defined-word> if it attempts to update the <xref linkend=\"CHDDAABB\">SPSR</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "MSR":
            return {
                "tooltip": "Move general-purpose register to Special register moves selected bits of a general-purpose register to the APSR, CPSR or SPSR_<current_mode>.",
                "html": "<p>Move general-purpose register to Special register moves selected bits of a general-purpose register to the <xref linkend=\"CJAGBHBH\">APSR</xref>, <xref linkend=\"CIHJBHJA\">CPSR</xref> or <xref linkend=\"CHDDAABB\">SPSR</xref>_&lt;current_mode&gt;.</p><p>Because of the Do-Not-Modify nature of its reserved bits, a read-modify-write sequence is normally required when the <instruction>MSR</instruction> instruction is being used at Application level and its destination is not <value>APSR_nzcvq</value> (<value>CPSR_f</value>).</p><p>If an <instruction>MSR</instruction> (register) moves selected bits of an immediate value to the <xref linkend=\"CIHJBHJA\">CPSR</xref>, the PE checks whether the value being written to <xref linkend=\"BEIDIGBH\">PSTATE</xref>.M is legal. See <xref linkend=\"CHDDFIGE\">Illegal changes to PSTATE.M</xref>.</p><p>An <instruction>MSR</instruction> (register) executed in User mode:</p><p>An <instruction>MSR</instruction> (register) executed in System mode is <arm-defined-word>unpredictable</arm-defined-word> if it attempts to update the <xref linkend=\"CHDDAABB\">SPSR</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "MUL":
        case "MULS":
            return {
                "tooltip": "Multiply multiplies two register values. The least significant 32 bits of the result are written to the destination register. These 32 bits do not depend on whether the source register values are considered to be signed values or unsigned values.",
                "html": "<p>Multiply multiplies two register values. The least significant 32 bits of the result are written to the destination register. These 32 bits do not depend on whether the source register values are considered to be signed values or unsigned values.</p><p>Optionally, it can update the condition flags based on the result. In the T32 instruction set, this option is limited to only a few forms of the instruction. Use of this option adversely affects performance on many implementations.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "MVN":
        case "MVNS":
            return {
                "tooltip": "Bitwise NOT (immediate) writes the bitwise inverse of an immediate value to the destination register.",
                "html": "<p>Bitwise NOT (immediate) writes the bitwise inverse of an immediate value to the destination register.</p><p>If the destination register is not the PC, the MVNS variant of the instruction updates the condition flags based on the result.</p><p>The field descriptions for <syntax>&lt;Rd&gt;</syntax> identify the encodings where the PC is permitted as the destination register. ARM deprecates any use of these encodings. However, when the destination register is the PC:</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "MVN":
        case "MVNS":
            return {
                "tooltip": "Bitwise NOT (register) writes the bitwise inverse of a register value to the destination register.",
                "html": "<p>Bitwise NOT (register) writes the bitwise inverse of a register value to the destination register.</p><p>If the destination register is not the PC, the MVNS variant of the instruction updates the condition flags based on the result.</p><p>The field descriptions for <syntax>&lt;Rd&gt;</syntax> identify the encodings where the PC is permitted as the destination register. ARM deprecates any use of these encodings. However, when the destination register is the PC:</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "MVN":
        case "MVNS":
            return {
                "tooltip": "Bitwise NOT (register-shifted register) writes the bitwise inverse of a register-shifted register value to the destination register. It can optionally update the condition flags based on the result.",
                "html": "<p>Bitwise NOT (register-shifted register) writes the bitwise inverse of a register-shifted register value to the destination register. It can optionally update the condition flags based on the result.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "NOP":
            return {
                "tooltip": "No Operation does nothing. This instruction can be used for instruction alignment purposes.",
                "html": "<p>No Operation does nothing. This instruction can be used for instruction alignment purposes.</p><p>The timing effects of including a <instruction>NOP</instruction> instruction in a program are not guaranteed. It can increase execution time, leave it unchanged, or even reduce it. Therefore, <instruction>NOP</instruction> instructions are not suitable for timing loops.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "ORN":
        case "ORNS":
            return {
                "tooltip": "Bitwise OR NOT (immediate) performs a bitwise (inclusive) OR of a register value and the complement of an immediate value, and writes the result to the destination register. It can optionally update the condition flags based on the result.",
                "html": "<p>Bitwise OR NOT (immediate) performs a bitwise (inclusive) OR of a register value and the complement of an immediate value, and writes the result to the destination register. It can optionally update the condition flags based on the result.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "ORN":
        case "ORNS":
            return {
                "tooltip": "Bitwise OR NOT (register) performs a bitwise (inclusive) OR of a register value and the complement of an optionally-shifted register value, and writes the result to the destination register. It can optionally update the condition flags based on the result.",
                "html": "<p>Bitwise OR NOT (register) performs a bitwise (inclusive) OR of a register value and the complement of an optionally-shifted register value, and writes the result to the destination register. It can optionally update the condition flags based on the result.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "ORR":
        case "ORRS":
            return {
                "tooltip": "Bitwise OR (immediate) performs a bitwise (inclusive) OR of a register value and an immediate value, and writes the result to the destination register.",
                "html": "<p>Bitwise OR (immediate) performs a bitwise (inclusive) OR of a register value and an immediate value, and writes the result to the destination register.</p><p>If the destination register is not the PC, the ORRS variant of the instruction updates the condition flags based on the result.</p><p>The field descriptions for <syntax>&lt;Rd&gt;</syntax> identify the encodings where the PC is permitted as the destination register. ARM deprecates any use of these encodings. However, when the destination register is the PC:</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "ORR":
        case "ORRS":
            return {
                "tooltip": "Bitwise OR (register) performs a bitwise (inclusive) OR of a register value and an optionally-shifted register value, and writes the result to the destination register.",
                "html": "<p>Bitwise OR (register) performs a bitwise (inclusive) OR of a register value and an optionally-shifted register value, and writes the result to the destination register.</p><p>If the destination register is not the PC, the ORRS variant of the instruction updates the condition flags based on the result.</p><p>The field descriptions for <syntax>&lt;Rd&gt;</syntax> identify the encodings where the PC is permitted as the destination register. ARM deprecates any use of these encodings. However, when the destination register is the PC:</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "ORR":
        case "ORRS":
            return {
                "tooltip": "Bitwise OR (register-shifted register) performs a bitwise (inclusive) OR of a register value and a register-shifted register value, and writes the result to the destination register. It can optionally update the condition flags based on the result.",
                "html": "<p>Bitwise OR (register-shifted register) performs a bitwise (inclusive) OR of a register value and a register-shifted register value, and writes the result to the destination register. It can optionally update the condition flags based on the result.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "PKHBT":
        case "PKHTB":
            return {
                "tooltip": "Pack Halfword combines one halfword of its first operand with the other halfword of its shifted second operand.",
                "html": "<p>Pack Halfword combines one halfword of its first operand with the other halfword of its shifted second operand.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "PLD":
        case "PLDW":
            return {
                "tooltip": "Preload Data (immediate) signals the memory system that data memory accesses from a specified address are likely in the near future. The memory system can respond by taking actions that are expected to speed up the memory accesses when they do occur, such as preloading the cache line containing the specified address into the data cache.",
                "html": "<p>Preload Data (immediate) signals the memory system that data memory accesses from a specified address are likely in the near future. The memory system can respond by taking actions that are expected to speed up the memory accesses when they do occur, such as preloading the cache line containing the specified address into the data cache.</p><p>The <instruction>PLD</instruction> instruction signals that the likely memory access is a read, and the <instruction>PLDW</instruction> instruction signals that it is a write.</p><p>The effect of a <instruction>PLD</instruction> or <instruction>PLDW</instruction> instruction is <arm-defined-word>implementation defined</arm-defined-word>.  For more information, see <xref linkend=\"CEGJJFCA\">Preloading caches</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "PLD":
            return {
                "tooltip": "Preload Data (literal) signals the memory system that data memory accesses from a specified address are likely in the near future. The memory system can respond by taking actions that are expected to speed up the memory accesses when they do occur, such as preloading the cache line containing the specified address into the data cache.",
                "html": "<p>Preload Data (literal) signals the memory system that data memory accesses from a specified address are likely in the near future. The memory system can respond by taking actions that are expected to speed up the memory accesses when they do occur, such as preloading the cache line containing the specified address into the data cache.</p><p>The effect of a <instruction>PLD</instruction> instruction is <arm-defined-word>implementation defined</arm-defined-word>.  For more information, see <xref linkend=\"CEGJJFCA\">Preloading caches</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "PLD":
        case "PLDW":
            return {
                "tooltip": "Preload Data (register) signals the memory system that data memory accesses from a specified address are likely in the near future. The memory system can respond by taking actions that are expected to speed up the memory accesses when they do occur, such as preloading the cache line containing the specified address into the data cache.",
                "html": "<p>Preload Data (register) signals the memory system that data memory accesses from a specified address are likely in the near future. The memory system can respond by taking actions that are expected to speed up the memory accesses when they do occur, such as preloading the cache line containing the specified address into the data cache.</p><p>The <instruction>PLD</instruction> instruction signals that the likely memory access is a read, and the <instruction>PLDW</instruction> instruction signals that it is a write.</p><p>The effect of a <instruction>PLD</instruction> or <instruction>PLDW</instruction> instruction is <arm-defined-word>implementation defined</arm-defined-word>.  For more information, see <xref linkend=\"CEGJJFCA\">Preloading caches</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "PLI":
            return {
                "tooltip": "Preload Instruction signals the memory system that instruction memory accesses from a specified address are likely in the near future. The memory system can respond by taking actions that are expected to speed up the memory accesses when they do occur, such as pre-loading the cache line containing the specified address into the instruction cache.",
                "html": "<p>Preload Instruction signals the memory system that instruction memory accesses from a specified address are likely in the near future. The memory system can respond by taking actions that are expected to speed up the memory accesses when they do occur, such as pre-loading the cache line containing the specified address into the instruction cache.</p><p>The effect of a <instruction>PLI</instruction> instruction is <arm-defined-word>implementation defined</arm-defined-word>. For more information, see <xref linkend=\"CEGJJFCA\">Preloading caches</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "PLI":
            return {
                "tooltip": "Preload Instruction signals the memory system that instruction memory accesses from a specified address are likely in the near future. The memory system can respond by taking actions that are expected to speed up the memory accesses when they do occur, such as pre-loading the cache line containing the specified address into the instruction cache.",
                "html": "<p>Preload Instruction signals the memory system that instruction memory accesses from a specified address are likely in the near future. The memory system can respond by taking actions that are expected to speed up the memory accesses when they do occur, such as pre-loading the cache line containing the specified address into the instruction cache.</p><p>The effect of a <instruction>PLI</instruction> instruction is <arm-defined-word>implementation defined</arm-defined-word>. For more information, see <xref linkend=\"CEGJJFCA\">Preloading caches</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "POP":
            return {
                "tooltip": "Pop Multiple Registers from Stack loads multiple general-purpose registers from the stack, loading from consecutive memory locations starting at the address in SP, and updates SP to point just above the loaded data.",
                "html": "<p>Pop Multiple Registers from Stack loads multiple general-purpose registers from the stack, loading from consecutive memory locations starting at the address in SP, and updates SP to point just above the loaded data.</p><p>The lowest-numbered register is loaded from the lowest memory address, through to the highest-numbered register from the highest memory address. See also <xref linkend=\"CHDDBEDG\">Encoding of lists of general-purpose registers and the PC</xref>.</p><p>The registers loaded can include the PC, causing a branch to a loaded address. This is an interworking branch, see <xref linkend=\"BEICJFEH\">Pseudocode description of operations on the AArch32 general-purpose registers and the PC</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "PSSBB":
            return {
                "tooltip": "Physical Speculative Store Bypass Barrier is a memory barrier that prevents speculative loads from bypassing earlier stores to the same physical address under certain conditions. For more information and details of the semantics, see Physical Speculative Store Bypass Barrier (PSSBB).",
                "html": "<p>Physical Speculative Store Bypass Barrier is a memory barrier that prevents speculative loads from bypassing earlier stores to the same physical address under certain conditions. For more information and details of the semantics, see <xref linkend=\"CJAECGBC\">Physical Speculative Store Bypass Barrier (PSSBB)</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "PUSH":
            return {
                "tooltip": "Push Multiple Registers to Stack stores multiple general-purpose registers to the stack, storing to consecutive memory locations ending just below the address in SP, and updates SP to point to the start of the stored data.",
                "html": "<p>Push Multiple Registers to Stack stores multiple general-purpose registers to the stack, storing to consecutive memory locations ending just below the address in SP, and updates SP to point to the start of the stored data.</p><p>The lowest-numbered register is stored to the lowest memory address, through to the highest-numbered register to the highest memory address. See also <xref linkend=\"CHDDBEDG\">Encoding of lists of general-purpose registers and the PC</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "QADD":
            return {
                "tooltip": "Saturating Add adds two register values, saturates the result to the 32-bit signed integer range -231 to (231 - 1), and writes the result to the destination register. If saturation occurs, it sets PSTATE.Q to 1.",
                "html": "<p>Saturating Add adds two register values, saturates the result to the 32-bit signed integer range -2<sup>31</sup> to (2<sup>31</sup> - 1), and writes the result to the destination register. If saturation occurs, it sets <xref linkend=\"BEIDIGBH\">PSTATE</xref>.Q to 1.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "QADD16":
            return {
                "tooltip": "Saturating Add 16 performs two 16-bit integer additions, saturates the results to the 16-bit signed integer range -215 <= x <= 215 - 1, and writes the results to the destination register.",
                "html": "<p>Saturating Add 16 performs two 16-bit integer additions, saturates the results to the 16-bit signed integer range -2<sup>15</sup> &lt;= x &lt;= 2<sup>15</sup> - 1, and writes the results to the destination register.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "QADD8":
            return {
                "tooltip": "Saturating Add 8 performs four 8-bit integer additions, saturates the results to the 8-bit signed integer range -27 <= x <= 27 - 1, and writes the results to the destination register.",
                "html": "<p>Saturating Add 8 performs four 8-bit integer additions, saturates the results to the 8-bit signed integer range -2<sup>7</sup> &lt;= x &lt;= 2<sup>7</sup> - 1, and writes the results to the destination register.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "QASX":
            return {
                "tooltip": "Saturating Add and Subtract with Exchange exchanges the two halfwords of the second operand, performs one 16-bit integer addition and one 16-bit subtraction, saturates the results to the 16-bit signed integer range -215 <= x <= 215 - 1, and writes the results to the destination register.",
                "html": "<p>Saturating Add and Subtract with Exchange exchanges the two halfwords of the second operand, performs one 16-bit integer addition and one 16-bit subtraction, saturates the results to the 16-bit signed integer range -2<sup>15</sup> &lt;= x &lt;= 2<sup>15</sup> - 1, and writes the results to the destination register.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "QDADD":
            return {
                "tooltip": "Saturating Double and Add adds a doubled register value to another register value, and writes the result to the destination register. Both the doubling and the addition have their results saturated to the 32-bit signed integer range -231 <= x <= 231 - 1. If saturation occurs in either operation, it sets PSTATE.Q to 1.",
                "html": "<p>Saturating Double and Add adds a doubled register value to another register value, and writes the result to the destination register. Both the doubling and the addition have their results saturated to the 32-bit signed integer range -2<sup>31</sup> &lt;= x &lt;= 2<sup>31</sup> - 1. If saturation occurs in either operation, it sets <xref linkend=\"BEIDIGBH\">PSTATE</xref>.Q to 1.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "QDSUB":
            return {
                "tooltip": "Saturating Double and Subtract subtracts a doubled register value from another register value, and writes the result to the destination register. Both the doubling and the subtraction have their results saturated to the 32-bit signed integer range -231 <= x <= 231 - 1. If saturation occurs in either operation, it sets PSTATE.Q to 1.",
                "html": "<p>Saturating Double and Subtract subtracts a doubled register value from another register value, and writes the result to the destination register. Both the doubling and the subtraction have their results saturated to the 32-bit signed integer range -2<sup>31</sup> &lt;= x &lt;= 2<sup>31</sup> - 1. If saturation occurs in either operation, it sets <xref linkend=\"BEIDIGBH\">PSTATE</xref>.Q to 1.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "QSAX":
            return {
                "tooltip": "Saturating Subtract and Add with Exchange exchanges the two halfwords of the second operand, performs one 16-bit integer subtraction and one 16-bit addition, saturates the results to the 16-bit signed integer range -215 <= x <= 215 - 1, and writes the results to the destination register.",
                "html": "<p>Saturating Subtract and Add with Exchange exchanges the two halfwords of the second operand, performs one 16-bit integer subtraction and one 16-bit addition, saturates the results to the 16-bit signed integer range -2<sup>15</sup> &lt;= x &lt;= 2<sup>15</sup> - 1, and writes the results to the destination register.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "QSUB":
            return {
                "tooltip": "Saturating Subtract subtracts one register value from another register value, saturates the result to the 32-bit signed integer range -231 <= x <= 231 - 1, and writes the result to the destination register. If saturation occurs, it sets PSTATE.Q to 1.",
                "html": "<p>Saturating Subtract subtracts one register value from another register value, saturates the result to the 32-bit signed integer range -2<sup>31</sup> &lt;= x &lt;= 2<sup>31</sup> - 1, and writes the result to the destination register. If saturation occurs, it sets <xref linkend=\"BEIDIGBH\">PSTATE</xref>.Q to 1.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "QSUB16":
            return {
                "tooltip": "Saturating Subtract 16 performs two 16-bit integer subtractions, saturates the results to the 16-bit signed integer range -215 <= x <= 215 - 1, and writes the results to the destination register.",
                "html": "<p>Saturating Subtract 16 performs two 16-bit integer subtractions, saturates the results to the 16-bit signed integer range -2<sup>15</sup> &lt;= x &lt;= 2<sup>15</sup> - 1, and writes the results to the destination register.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "QSUB8":
            return {
                "tooltip": "Saturating Subtract 8 performs four 8-bit integer subtractions, saturates the results to the 8-bit signed integer range -27 <= x <= 27 - 1, and writes the results to the destination register.",
                "html": "<p>Saturating Subtract 8 performs four 8-bit integer subtractions, saturates the results to the 8-bit signed integer range -2<sup>7</sup> &lt;= x &lt;= 2<sup>7</sup> - 1, and writes the results to the destination register.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "RBIT":
            return {
                "tooltip": "Reverse Bits reverses the bit order in a 32-bit register.",
                "html": "<p>Reverse Bits reverses the bit order in a 32-bit register.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "REV":
            return {
                "tooltip": "Byte-Reverse Word reverses the byte order in a 32-bit register.",
                "html": "<p>Byte-Reverse Word reverses the byte order in a 32-bit register.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "REV16":
            return {
                "tooltip": "Byte-Reverse Packed Halfword reverses the byte order in each16-bit halfword of a 32-bit register.",
                "html": "<p>Byte-Reverse Packed Halfword reverses the byte order in each16-bit halfword of a 32-bit register.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "REVSH":
            return {
                "tooltip": "Byte-Reverse Signed Halfword reverses the byte order in the lower 16-bit halfword of a 32-bit register, and sign-extends the result to 32 bits.",
                "html": "<p>Byte-Reverse Signed Halfword reverses the byte order in the lower 16-bit halfword of a 32-bit register, and sign-extends the result to 32 bits.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "RFE":
        case "RFEDA":
        case "RFEDB":
        case "RFEIA":
        case "RFEIB":
            return {
                "tooltip": "Return From Exception loads two consecutive memory locations using an address in a base register",
                "html": "<p>Return From Exception loads two consecutive memory locations using an address in a base register:</p><p>An address adjusted by the size of the data loaded can optionally be written back to the base register.</p><p>The PE checks the value of the word loaded from the higher address for an illegal return event. See <xref linkend=\"CHDDDJDB\">Illegal return events from AArch32 state</xref>.</p><p><instruction>RFE</instruction> is <arm-defined-word>undefined</arm-defined-word> in Hyp mode and <arm-defined-word>constrained unpredictable</arm-defined-word> in User mode.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "ROR":
            return {
                "tooltip": "Rotate Right (immediate) provides the value of the contents of a register rotated by a constant value. The bits that are rotated off the right end are inserted into the vacated bit positions on the left.",
                "html": "<p>Rotate Right (immediate) provides the value of the contents of a register rotated by a constant value. The bits that are rotated off the right end are inserted into the vacated bit positions on the left.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "RORS":
            return {
                "tooltip": "Rotate Right, setting flags (immediate) provides the value of the contents of a register rotated by a constant value. The bits that are rotated off the right end are inserted into the vacated bit positions on the left.",
                "html": "<p>Rotate Right, setting flags (immediate) provides the value of the contents of a register rotated by a constant value. The bits that are rotated off the right end are inserted into the vacated bit positions on the left.</p><p>If the destination register is not the PC, this instruction updates the condition flags based on the result.</p><p>The field descriptions for <syntax>&lt;Rd&gt;</syntax> identify the encodings where the PC is permitted as the destination register. Arm deprecates any use of these encodings. However, when the destination register is the PC:</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "RRX":
            return {
                "tooltip": "Rotate Right with Extend provides the value of the contents of a register shifted right by one place, with the Carry flag shifted into bit[31].",
                "html": "<p>Rotate Right with Extend provides the value of the contents of a register shifted right by one place, with the Carry flag shifted into bit[31].</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "RRXS":
            return {
                "tooltip": "Rotate Right with Extend, setting flags provides the value of the contents of a register shifted right by one place, with the Carry flag shifted into bit[31].",
                "html": "<p>Rotate Right with Extend, setting flags provides the value of the contents of a register shifted right by one place, with the Carry flag shifted into bit[31].</p><p>If the destination register is not the PC, this instruction updates the condition flags based on the result, and bit[0] is shifted into the Carry flag.</p><p>The field descriptions for <syntax>&lt;Rd&gt;</syntax> identify the encodings where the PC is permitted as the destination register. Arm deprecates any use of these encodings. However, when the destination register is the PC:</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "RSB":
        case "RSBS":
            return {
                "tooltip": "Reverse Subtract (immediate) subtracts a register value from an immediate value, and writes the result to the destination register.",
                "html": "<p>Reverse Subtract (immediate) subtracts a register value from an immediate value, and writes the result to the destination register.</p><p>If the destination register is not the PC, the RSBS variant of the instruction updates the condition flags based on the result.</p><p>The field descriptions for <syntax>&lt;Rd&gt;</syntax> identify the encodings where the PC is permitted as the destination register. ARM deprecates any use of these encodings. However, when the destination register is the PC:</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "RSB":
        case "RSBS":
            return {
                "tooltip": "Reverse Subtract (register) subtracts a register value from an optionally-shifted register value, and writes the result to the destination register.",
                "html": "<p>Reverse Subtract (register) subtracts a register value from an optionally-shifted register value, and writes the result to the destination register.</p><p>If the destination register is not the PC, the RSBS variant of the instruction updates the condition flags based on the result.</p><p>The field descriptions for <syntax>&lt;Rd&gt;</syntax> identify the encodings where the PC is permitted as the destination register. ARM deprecates any use of these encodings. However, when the destination register is the PC:</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "RSB":
        case "RSBS":
            return {
                "tooltip": "Reverse Subtract (register-shifted register) subtracts a register value from a register-shifted register value, and writes the result to the destination register. It can optionally update the condition flags based on the result.",
                "html": "<p>Reverse Subtract (register-shifted register) subtracts a register value from a register-shifted register value, and writes the result to the destination register. It can optionally update the condition flags based on the result.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "RSC":
        case "RSCS":
            return {
                "tooltip": "Reverse Subtract with Carry (immediate) subtracts a register value and the value of NOT (Carry flag) from an immediate value, and writes the result to the destination register.",
                "html": "<p>Reverse Subtract with Carry (immediate) subtracts a register value and the value of NOT (Carry flag) from an immediate value, and writes the result to the destination register.</p><p>If the destination register is not the PC, the RSCS variant of the instruction updates the condition flags based on the result.</p><p>The field descriptions for <syntax>&lt;Rd&gt;</syntax> identify the encodings where the PC is permitted as the destination register. ARM deprecates any use of these encodings. However, when the destination register is the PC:</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "RSC":
        case "RSCS":
            return {
                "tooltip": "Reverse Subtract with Carry (register) subtracts a register value and the value of NOT (Carry flag) from an optionally-shifted register value, and writes the result to the destination register.",
                "html": "<p>Reverse Subtract with Carry (register) subtracts a register value and the value of NOT (Carry flag) from an optionally-shifted register value, and writes the result to the destination register.</p><p>If the destination register is not the PC, the RSCS variant of the instruction updates the condition flags based on the result.</p><p>The field descriptions for <syntax>&lt;Rd&gt;</syntax> identify the encodings where the PC is permitted as the destination register. ARM deprecates any use of these encodings. However, when the destination register is the PC:</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "RSC":
        case "RSCS":
            return {
                "tooltip": "Reverse Subtract (register-shifted register) subtracts a register value and the value of NOT (Carry flag) from a register-shifted register value, and writes the result to the destination register. It can optionally update the condition flags based on the result.",
                "html": "<p>Reverse Subtract (register-shifted register) subtracts a register value and the value of NOT (Carry flag) from a register-shifted register value, and writes the result to the destination register. It can optionally update the condition flags based on the result.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "SADD16":
            return {
                "tooltip": "Signed Add 16 performs two 16-bit signed integer additions, and writes the results to the destination register. It sets PSTATE.GE according to the results of the additions.",
                "html": "<p>Signed Add 16 performs two 16-bit signed integer additions, and writes the results to the destination register. It sets <xref linkend=\"BEIDIGBH\">PSTATE</xref>.GE according to the results of the additions.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "SADD8":
            return {
                "tooltip": "Signed Add 8 performs four 8-bit signed integer additions, and writes the results to the destination register. It sets PSTATE.GE according to the results of the additions.",
                "html": "<p>Signed Add 8 performs four 8-bit signed integer additions, and writes the results to the destination register. It sets <xref linkend=\"BEIDIGBH\">PSTATE</xref>.GE according to the results of the additions.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "SASX":
            return {
                "tooltip": "Signed Add and Subtract with Exchange exchanges the two halfwords of the second operand, performs one 16-bit integer addition and one 16-bit subtraction, and writes the results to the destination register. It sets PSTATE.GE according to the results.",
                "html": "<p>Signed Add and Subtract with Exchange exchanges the two halfwords of the second operand, performs one 16-bit integer addition and one 16-bit subtraction, and writes the results to the destination register. It sets <xref linkend=\"BEIDIGBH\">PSTATE</xref>.GE according to the results.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "SB":
            return {
                "tooltip": "Speculation Barrier is a barrier that controls speculation.",
                "html": "<p>Speculation Barrier is a barrier that controls speculation.</p><p>The semantics of the Speculation Barrier are that the execution, until the barrier completes, of any instruction that appears later in the program order than the barrier:</p><p>In particular, any instruction that appears later in the program order than the barrier cannot cause a speculative allocation into any caching structure where the allocation of that entry could be indicative of any data value present in memory or in the registers.</p><p>The SB instruction:</p><p>When the prediction of the instruction stream is not informed by data taken from the register outputs of the speculative execution of instructions appearing in program order after an uncompleted SB instruction, the SB instruction has no effect on the use of prediction resources to predict the instruction stream that is being fetched.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "SBC":
        case "SBCS":
            return {
                "tooltip": "Subtract with Carry (immediate) subtracts an immediate value and the value of NOT (Carry flag) from a register value, and writes the result to the destination register.",
                "html": "<p>Subtract with Carry (immediate) subtracts an immediate value and the value of NOT (Carry flag) from a register value, and writes the result to the destination register.</p><p>If the destination register is not the PC, the SBCS variant of the instruction updates the condition flags based on the result.</p><p>The field descriptions for <syntax>&lt;Rd&gt;</syntax> identify the encodings where the PC is permitted as the destination register. ARM deprecates any use of these encodings. However, when the destination register is the PC:</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "SBC":
        case "SBCS":
            return {
                "tooltip": "Subtract with Carry (register) subtracts an optionally-shifted register value and the value of NOT (Carry flag) from a register value, and writes the result to the destination register.",
                "html": "<p>Subtract with Carry (register) subtracts an optionally-shifted register value and the value of NOT (Carry flag) from a register value, and writes the result to the destination register.</p><p>If the destination register is not the PC, the SBCS variant of the instruction updates the condition flags based on the result.</p><p>The field descriptions for <syntax>&lt;Rd&gt;</syntax> identify the encodings where the PC is permitted as the destination register. ARM deprecates any use of these encodings. However, when the destination register is the PC:</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "SBC":
        case "SBCS":
            return {
                "tooltip": "Subtract with Carry (register-shifted register) subtracts a register-shifted register value and the value of NOT (Carry flag) from a register value, and writes the result to the destination register. It can optionally update the condition flags based on the result.",
                "html": "<p>Subtract with Carry (register-shifted register) subtracts a register-shifted register value and the value of NOT (Carry flag) from a register value, and writes the result to the destination register. It can optionally update the condition flags based on the result.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "SBFX":
            return {
                "tooltip": "Signed Bit Field Extract extracts any number of adjacent bits at any position from a register, sign-extends them to 32 bits, and writes the result to the destination register.",
                "html": "<p>Signed Bit Field Extract extracts any number of adjacent bits at any position from a register, sign-extends them to 32 bits, and writes the result to the destination register.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "SDIV":
            return {
                "tooltip": "Signed Divide divides a 32-bit signed integer register value by a 32-bit signed integer register value, and writes the result to the destination register. The condition flags are not affected.",
                "html": "<p>Signed Divide divides a 32-bit signed integer register value by a 32-bit signed integer register value, and writes the result to the destination register. The condition flags are not affected.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "SEL":
            return {
                "tooltip": "Select Bytes selects each byte of its result from either its first operand or its second operand, according to the values of the PSTATE.GE flags.",
                "html": "<p>Select Bytes selects each byte of its result from either its first operand or its second operand, according to the values of the <xref linkend=\"BEIDIGBH\">PSTATE</xref>.GE flags.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "SETEND":
            return {
                "tooltip": "Set Endianness writes a new value to PSTATE.E.",
                "html": "<p>Set Endianness writes a new value to <xref linkend=\"BEIDIGBH\">PSTATE</xref>.E.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "SETPAN":
            return {
                "tooltip": "Set Privileged Access Never writes a new value to PSTATE.PAN.",
                "html": "<p>Set Privileged Access Never writes a new value to <xref linkend=\"BEIDIGBH\">PSTATE</xref>.PAN.</p><p>This instruction is available only in privileged mode and it is a NOP when executed in User mode.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "SEV":
            return {
                "tooltip": "Send Event is a hint instruction. It causes an event to be signaled to all PEs in the multiprocessor system. For more information, see Wait For Event and Send Event.",
                "html": "<p>Send Event is a hint instruction. It causes an event to be signaled to all PEs in the multiprocessor system. For more information, see <xref linkend=\"CFIJIIHE\">Wait For Event and Send Event</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "SEVL":
            return {
                "tooltip": "Send Event Local is a hint instruction that causes an event to be signaled locally without requiring the event to be signaled to other PEs in the multiprocessor system. It can prime a wait-loop which starts with a WFE instruction.",
                "html": "<p>Send Event Local is a hint instruction that causes an event to be signaled locally without requiring the event to be signaled to other PEs in the multiprocessor system. It can prime a wait-loop which starts with a <instruction>WFE</instruction> instruction.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "SHA1C":
            return {
                "tooltip": "SHA1 hash update (choose).",
                "html": "<p>SHA1 hash update (choose).</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "SHA1H":
            return {
                "tooltip": "SHA1 fixed rotate.",
                "html": "<p>SHA1 fixed rotate.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "SHA1M":
            return {
                "tooltip": "SHA1 hash update (majority).",
                "html": "<p>SHA1 hash update (majority).</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "SHA1P":
            return {
                "tooltip": "SHA1 hash update (parity).",
                "html": "<p>SHA1 hash update (parity).</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "SHA1SU0":
            return {
                "tooltip": "SHA1 schedule update 0.",
                "html": "<p>SHA1 schedule update 0.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "SHA1SU1":
            return {
                "tooltip": "SHA1 schedule update 1.",
                "html": "<p>SHA1 schedule update 1.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "SHA256H":
            return {
                "tooltip": "SHA256 hash update part 1.",
                "html": "<p>SHA256 hash update part 1.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "SHA256H2":
            return {
                "tooltip": "SHA256 hash update part 2.",
                "html": "<p>SHA256 hash update part 2.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "SHA256SU0":
            return {
                "tooltip": "SHA256 schedule update 0.",
                "html": "<p>SHA256 schedule update 0.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "SHA256SU1":
            return {
                "tooltip": "SHA256 schedule update 1.",
                "html": "<p>SHA256 schedule update 1.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "SHADD16":
            return {
                "tooltip": "Signed Halving Add 16 performs two signed 16-bit integer additions, halves the results, and writes the results to the destination register.",
                "html": "<p>Signed Halving Add 16 performs two signed 16-bit integer additions, halves the results, and writes the results to the destination register.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "SHADD8":
            return {
                "tooltip": "Signed Halving Add 8 performs four signed 8-bit integer additions, halves the results, and writes the results to the destination register.",
                "html": "<p>Signed Halving Add 8 performs four signed 8-bit integer additions, halves the results, and writes the results to the destination register.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "SHASX":
            return {
                "tooltip": "Signed Halving Add and Subtract with Exchange exchanges the two halfwords of the second operand, performs one signed 16-bit integer addition and one signed 16-bit subtraction, halves the results, and writes the results to the destination register.",
                "html": "<p>Signed Halving Add and Subtract with Exchange exchanges the two halfwords of the second operand, performs one signed 16-bit integer addition and one signed 16-bit subtraction, halves the results, and writes the results to the destination register.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "SHSAX":
            return {
                "tooltip": "Signed Halving Subtract and Add with Exchange exchanges the two halfwords of the second operand, performs one signed 16-bit integer subtraction and one signed 16-bit addition, halves the results, and writes the results to the destination register.",
                "html": "<p>Signed Halving Subtract and Add with Exchange exchanges the two halfwords of the second operand, performs one signed 16-bit integer subtraction and one signed 16-bit addition, halves the results, and writes the results to the destination register.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "SHSUB16":
            return {
                "tooltip": "Signed Halving Subtract 16 performs two signed 16-bit integer subtractions, halves the results, and writes the results to the destination register.",
                "html": "<p>Signed Halving Subtract 16 performs two signed 16-bit integer subtractions, halves the results, and writes the results to the destination register.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "SHSUB8":
            return {
                "tooltip": "Signed Halving Subtract 8 performs four signed 8-bit integer subtractions, halves the results, and writes the results to the destination register.",
                "html": "<p>Signed Halving Subtract 8 performs four signed 8-bit integer subtractions, halves the results, and writes the results to the destination register.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "SMC":
            return {
                "tooltip": "Secure Monitor Call causes a Secure Monitor Call exception.  For more information see Secure Monitor Call (SMC) exception.",
                "html": "<p>Secure Monitor Call causes a Secure Monitor Call exception.  For more information see <xref linkend=\"CIHEHJBH\">Secure Monitor Call (SMC) exception</xref>.</p><p><instruction>SMC</instruction> is available only for software executing at EL1 or higher. It is <arm-defined-word>undefined</arm-defined-word> in User mode.</p><p>If the values of <xref linkend=\"CEGFDIFJ\">HCR</xref>.TSC and <xref linkend=\"CEGCEECB\">SCR</xref>.SCD are both 0, execution of an <instruction>SMC</instruction> instruction at EL1 or higher generates a Secure Monitor Call exception that is taken to EL3. When EL3 is using AArch32 this exception is taken to Monitor mode. When EL3 is using AArch64, it is the <xref linkend=\"AArch64.scr_el3\">SCR_EL3</xref>.SMD bit, rather than the <xref linkend=\"CEGCEECB\">SCR</xref>.SCD bit, that can change the effect of executing an SMC instruction.</p><p>If the value of <xref linkend=\"CEGFDIFJ\">HCR</xref>.TSC is 1, execution of an <instruction>SMC</instruction> instruction in a Non-secure EL1 mode generates an exception that is taken to EL2, regardless of the value of <xref linkend=\"CEGCEECB\">SCR</xref>.SCD. When EL2 is using AArch32, this is a Hyp Trap exception that is taken to Hyp mode. For more information see <xref linkend=\"CHDBHAEI\">Traps to Hyp mode of Non-secure EL1 execution of SMC instructions</xref>.</p><p>If the value of <xref linkend=\"CEGFDIFJ\">HCR</xref>.TSC is 0 and the value of <xref linkend=\"CEGCEECB\">SCR</xref>.SCD is 1, the SMC instruction is:</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "SMLABB":
        case "SMLABT":
        case "SMLATB":
        case "SMLATT":
            return {
                "tooltip": "Signed Multiply Accumulate (halfwords) performs a signed multiply accumulate operation. The multiply acts on two signed 16-bit quantities, taken from either the bottom or the top half of their respective source registers. The other halves of these source registers are ignored.  The 32-bit product is added to a 32-bit accumulate value and the result is written to the destination register.",
                "html": "<p>Signed Multiply Accumulate (halfwords) performs a signed multiply accumulate operation. The multiply acts on two signed 16-bit quantities, taken from either the bottom or the top half of their respective source registers. The other halves of these source registers are ignored.  The 32-bit product is added to a 32-bit accumulate value and the result is written to the destination register.</p><p>If overflow occurs during the addition of the accumulate value, the instruction sets <xref linkend=\"BEIDIGBH\">PSTATE</xref>.Q to 1. It is not possible for overflow to occur during the multiplication.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "SMLAD":
        case "SMLADX":
            return {
                "tooltip": "Signed Multiply Accumulate Dual performs two signed 16 x 16-bit multiplications. It adds the products to a 32-bit accumulate operand.",
                "html": "<p>Signed Multiply Accumulate Dual performs two signed 16 x 16-bit multiplications. It adds the products to a 32-bit accumulate operand.</p><p>Optionally, the instruction can exchange the halfwords of the second operand before performing the arithmetic. This produces top x bottom and bottom x top multiplication.</p><p>This instruction sets <xref linkend=\"BEIDIGBH\">PSTATE</xref>.Q to 1 if the accumulate operation overflows. Overflow cannot occur during the multiplications.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "SMLAL":
        case "SMLALS":
            return {
                "tooltip": "Signed Multiply Accumulate Long multiplies two signed 32-bit values to produce a 64-bit value, and accumulates this with a 64-bit value.",
                "html": "<p>Signed Multiply Accumulate Long multiplies two signed 32-bit values to produce a 64-bit value, and accumulates this with a 64-bit value.</p><p>In A32 instructions, the condition flags can optionally be updated based on the result. Use of this option adversely affects performance on many implementations.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "SMLALBB":
        case "SMLALBT":
        case "SMLALTB":
        case "SMLALTT":
            return {
                "tooltip": "Signed Multiply Accumulate Long (halfwords) multiplies two signed 16-bit values to produce a 32-bit value, and accumulates this with a 64-bit value. The multiply acts on two signed 16-bit quantities, taken from either the bottom or the top half of their respective source registers. The other halves of these source registers are ignored. The 32-bit product is sign-extended and accumulated with a 64-bit accumulate value.",
                "html": "<p>Signed Multiply Accumulate Long (halfwords) multiplies two signed 16-bit values to produce a 32-bit value, and accumulates this with a 64-bit value. The multiply acts on two signed 16-bit quantities, taken from either the bottom or the top half of their respective source registers. The other halves of these source registers are ignored. The 32-bit product is sign-extended and accumulated with a 64-bit accumulate value.</p><p>Overflow is possible during this instruction, but only as a result of the 64-bit addition. This overflow is not detected if it occurs. Instead, the result wraps around modulo 2<sup>64</sup>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "SMLALD":
        case "SMLALDX":
            return {
                "tooltip": "Signed Multiply Accumulate Long Dual performs two signed 16 x 16-bit multiplications. It adds the products to a 64-bit accumulate operand.",
                "html": "<p>Signed Multiply Accumulate Long Dual performs two signed 16 x 16-bit multiplications. It adds the products to a 64-bit accumulate operand.</p><p>Optionally, the instruction can exchange the halfwords of the second operand before performing the arithmetic. This produces top x bottom and bottom x top multiplication.</p><p>Overflow is possible during this instruction, but only as a result of the 64-bit addition. This overflow is not detected if it occurs. Instead, the result wraps around modulo 2<sup>64</sup>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "SMLAWB":
        case "SMLAWT":
            return {
                "tooltip": "Signed Multiply Accumulate (word by halfword) performs a signed multiply accumulate operation. The multiply acts on a signed 32-bit quantity and a signed 16-bit quantity. The signed 16-bit quantity is taken from either the bottom or the top half of its source register. The other half of the second source register is ignored. The top 32 bits of the 48-bit product are added to a 32-bit accumulate value and the result is written to the destination register. The bottom 16 bits of the 48-bit product are ignored.",
                "html": "<p>Signed Multiply Accumulate (word by halfword) performs a signed multiply accumulate operation. The multiply acts on a signed 32-bit quantity and a signed 16-bit quantity. The signed 16-bit quantity is taken from either the bottom or the top half of its source register. The other half of the second source register is ignored. The top 32 bits of the 48-bit product are added to a 32-bit accumulate value and the result is written to the destination register. The bottom 16 bits of the 48-bit product are ignored.</p><p>If overflow occurs during the addition of the accumulate value, the instruction sets <xref linkend=\"BEIDIGBH\">PSTATE</xref>.Q to 1. No overflow can occur during the multiplication.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "SMLSD":
        case "SMLSDX":
            return {
                "tooltip": "Signed Multiply Subtract Dual performs two signed 16 x 16-bit multiplications. It adds the difference of the products to a 32-bit accumulate operand.",
                "html": "<p>Signed Multiply Subtract Dual performs two signed 16 x 16-bit multiplications. It adds the difference of the products to a 32-bit accumulate operand.</p><p>Optionally, the instruction can exchange the halfwords of the second operand before performing the arithmetic. This produces top x bottom and bottom x top multiplication.</p><p>This instruction sets <xref linkend=\"BEIDIGBH\">PSTATE</xref>.Q to 1 if the accumulate operation overflows. Overflow cannot occur during the multiplications or subtraction.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "SMLSLD":
        case "SMLSLDX":
            return {
                "tooltip": "Signed Multiply Subtract Long Dual performs two signed 16 x 16-bit multiplications. It adds the difference of the products to a 64-bit accumulate operand.",
                "html": "<p>Signed Multiply Subtract Long Dual performs two signed 16 x 16-bit multiplications. It adds the difference of the products to a 64-bit accumulate operand.</p><p>Optionally, the instruction can exchange the halfwords of the second operand before performing the arithmetic. This produces top x bottom and bottom x top multiplication.</p><p>Overflow is possible during this instruction, but only as a result of the 64-bit addition. This overflow is not detected if it occurs. Instead, the result wraps around modulo 2<sup>64</sup>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "SMMLA":
        case "SMMLAR":
            return {
                "tooltip": "Signed Most Significant Word Multiply Accumulate multiplies two signed 32-bit values, extracts the most significant 32 bits of the result, and adds an accumulate value.",
                "html": "<p>Signed Most Significant Word Multiply Accumulate multiplies two signed 32-bit values, extracts the most significant 32 bits of the result, and adds an accumulate value.</p><p>Optionally, the instruction can specify that the result is rounded instead of being truncated. In this case, the constant <hexnumber>0x80000000</hexnumber> is added to the product before the high word is extracted.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "SMMLS":
        case "SMMLSR":
            return {
                "tooltip": "Signed Most Significant Word Multiply Subtract multiplies two signed 32-bit values, subtracts the result from a 32-bit accumulate value that is shifted left by 32 bits, and extracts the most significant 32 bits of the result of that subtraction.",
                "html": "<p>Signed Most Significant Word Multiply Subtract multiplies two signed 32-bit values, subtracts the result from a 32-bit accumulate value that is shifted left by 32 bits, and extracts the most significant 32 bits of the result of that subtraction.</p><p>Optionally, the instruction can specify that the result of the instruction is rounded instead of being truncated. In this case, the constant <hexnumber>0x80000000</hexnumber> is added to the result of the subtraction before the high word is extracted.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "SMMUL":
        case "SMMULR":
            return {
                "tooltip": "Signed Most Significant Word Multiply multiplies two signed 32-bit values, extracts the most significant 32 bits of the result, and writes those bits to the destination register.",
                "html": "<p>Signed Most Significant Word Multiply multiplies two signed 32-bit values, extracts the most significant 32 bits of the result, and writes those bits to the destination register.</p><p>Optionally, the instruction can specify that the result is rounded instead of being truncated. In this case, the constant <hexnumber>0x80000000</hexnumber> is added to the product before the high word is extracted.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "SMUAD":
        case "SMUADX":
            return {
                "tooltip": "Signed Dual Multiply Add performs two signed 16 x 16-bit multiplications. It adds the products together, and writes the result to the destination register.",
                "html": "<p>Signed Dual Multiply Add performs two signed 16 x 16-bit multiplications. It adds the products together, and writes the result to the destination register.</p><p>Optionally, the instruction can exchange the halfwords of the second operand before performing the arithmetic. This produces top x bottom and bottom x top multiplication.</p><p>This instruction sets <xref linkend=\"BEIDIGBH\">PSTATE</xref>.Q to 1 if the addition overflows. The multiplications cannot overflow.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "SMULBB":
        case "SMULBT":
        case "SMULTB":
        case "SMULTT":
            return {
                "tooltip": "Signed Multiply (halfwords) multiplies two signed 16-bit quantities, taken from either the bottom or the top half of their respective source registers. The other halves of these source registers are ignored. The 32-bit product is written to the destination register. No overflow is possible during this instruction.",
                "html": "<p>Signed Multiply (halfwords) multiplies two signed 16-bit quantities, taken from either the bottom or the top half of their respective source registers. The other halves of these source registers are ignored. The 32-bit product is written to the destination register. No overflow is possible during this instruction.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "SMULL":
        case "SMULLS":
            return {
                "tooltip": "Signed Multiply Long multiplies two 32-bit signed values to produce a 64-bit result.",
                "html": "<p>Signed Multiply Long multiplies two 32-bit signed values to produce a 64-bit result.</p><p>In A32 instructions, the condition flags can optionally be updated based on the result. Use of this option adversely affects performance on many implementations.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "SMULWB":
        case "SMULWT":
            return {
                "tooltip": "Signed Multiply (word by halfword) multiplies a signed 32-bit quantity and a signed 16-bit quantity. The signed 16-bit quantity is taken from either the bottom or the top half of its source register. The other half of the second source register is ignored. The top 32 bits of the 48-bit product are written to the destination register. The bottom 16 bits of the 48-bit product are ignored. No overflow is possible during this instruction.",
                "html": "<p>Signed Multiply (word by halfword) multiplies a signed 32-bit quantity and a signed 16-bit quantity. The signed 16-bit quantity is taken from either the bottom or the top half of its source register. The other half of the second source register is ignored. The top 32 bits of the 48-bit product are written to the destination register. The bottom 16 bits of the 48-bit product are ignored. No overflow is possible during this instruction.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "SMUSD":
        case "SMUSDX":
            return {
                "tooltip": "Signed Multiply Subtract Dual performs two signed 16 x 16-bit multiplications. It subtracts one of the products from the other, and writes the result to the destination register.",
                "html": "<p>Signed Multiply Subtract Dual performs two signed 16 x 16-bit multiplications. It subtracts one of the products from the other, and writes the result to the destination register.</p><p>Optionally, the instruction can exchange the halfwords of the second operand before performing the arithmetic. This produces top x bottom and bottom x top multiplication.</p><p>Overflow cannot occur.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "SRS":
        case "SRSDA":
        case "SRSDB":
        case "SRSIA":
        case "SRSIB":
            return {
                "tooltip": "Store Return State stores the LR_<current_mode> and SPSR_<current_mode> to the stack of a specified mode. For information about memory accesses see Memory accesses.",
                "html": "<p>Store Return State stores the LR_&lt;current_mode&gt; and <xref linkend=\"CHDDAABB\">SPSR</xref>_&lt;current_mode&gt; to the stack of a specified mode. For information about memory accesses see <xref linkend=\"Chddjfjf\">Memory accesses</xref>.</p><p><instruction>SRS</instruction> is <arm-defined-word>undefined</arm-defined-word> in Hyp mode.</p><p><instruction>SRS</instruction> is <arm-defined-word>constrained unpredictable</arm-defined-word> if it is executed in User or System mode, or if the specified mode is any of the following:</p><p>If EL3 is using AArch64 and an <instruction>SRS</instruction> instruction that is executed in a Secure EL1 mode specifies Monitor mode, it is trapped to EL3.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "SSAT":
            return {
                "tooltip": "Signed Saturate saturates an optionally-shifted signed value to a selectable signed range.",
                "html": "<p>Signed Saturate saturates an optionally-shifted signed value to a selectable signed range.</p><p>This instruction sets <xref linkend=\"BEIDIGBH\">PSTATE</xref>.Q to 1 if the operation saturates.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "SSAT16":
            return {
                "tooltip": "Signed Saturate 16 saturates two signed 16-bit values to a selected signed range.",
                "html": "<p>Signed Saturate 16 saturates two signed 16-bit values to a selected signed range.</p><p>This instruction sets <xref linkend=\"BEIDIGBH\">PSTATE</xref>.Q to 1 if the operation saturates.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "SSAX":
            return {
                "tooltip": "Signed Subtract and Add with Exchange exchanges the two halfwords of the second operand, performs one 16-bit integer subtraction and one 16-bit addition, and writes the results to the destination register. It sets PSTATE.GE according to the results.",
                "html": "<p>Signed Subtract and Add with Exchange exchanges the two halfwords of the second operand, performs one 16-bit integer subtraction and one 16-bit addition, and writes the results to the destination register. It sets <xref linkend=\"BEIDIGBH\">PSTATE</xref>.GE according to the results.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "SSBB":
            return {
                "tooltip": "Speculative Store Bypass Barrier is a memory barrier that prevents speculative loads from bypassing earlier stores to the same virtual address under certain conditions. For more information and details of the semantics, see Speculative Store Bypass Barrier (SSBB).",
                "html": "<p>Speculative Store Bypass Barrier is a memory barrier that prevents speculative loads from bypassing earlier stores to the same virtual address under certain conditions. For more information and details of the semantics, see <xref linkend=\"CJAGECIG\">Speculative Store Bypass Barrier (SSBB)</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "SSUB16":
            return {
                "tooltip": "Signed Subtract 16 performs two 16-bit signed integer subtractions, and writes the results to the destination register. It sets PSTATE.GE according to the results of the subtractions.",
                "html": "<p>Signed Subtract 16 performs two 16-bit signed integer subtractions, and writes the results to the destination register. It sets <xref linkend=\"BEIDIGBH\">PSTATE</xref>.GE according to the results of the subtractions.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "SSUB8":
            return {
                "tooltip": "Signed Subtract 8 performs four 8-bit signed integer subtractions, and writes the results to the destination register. It sets PSTATE.GE according to the results of the subtractions.",
                "html": "<p>Signed Subtract 8 performs four 8-bit signed integer subtractions, and writes the results to the destination register. It sets <xref linkend=\"BEIDIGBH\">PSTATE</xref>.GE according to the results of the subtractions.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "STC":
            return {
                "tooltip": "Store data to System register calculates an address from a base register value and an immediate offset, and stores a word from the DBGDTRRXint System register to memory. It can use offset, post-indexed, pre-indexed, or unindexed addressing. For information about memory accesses, see Memory accesses.",
                "html": "<p>Store data to System register calculates an address from a base register value and an immediate offset, and stores a word from the <xref linkend=\"AArch32.dbgdtrrxint\">DBGDTRRXint</xref> System register to memory. It can use offset, post-indexed, pre-indexed, or unindexed addressing. For information about memory accesses, see <xref linkend=\"Chddjfjf\">Memory accesses</xref>.</p><p>In an implementation that includes EL2, the permitted <instruction>STC</instruction> access to <xref linkend=\"AArch32.dbgdtrrxint\">DBGDTRRXint</xref> can be trapped to Hyp mode, meaning that an attempt to execute an <instruction>STC</instruction> instruction in a Non-secure mode other than Hyp mode, that would be permitted in the absence of the Hyp trap controls, generates a Hyp Trap exception. For more information, see <xref linkend=\"BEICAABI\">Trapping general Non-secure System register accesses to debug registers</xref>.</p><p>For simplicity, the <instruction>STC</instruction> pseudocode does not show this possible trap to Hyp mode.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "STL":
            return {
                "tooltip": "Store-Release Word stores a word from a register to memory.  The instruction also has memory ordering semantics as described in Load-Acquire, Store-Release.",
                "html": "<p>Store-Release Word stores a word from a register to memory.  The instruction also has memory ordering semantics as described in <xref linkend=\"AA32CHDBDIDF\">Load-Acquire, Store-Release</xref>.</p><p>For more information about support for shared memory see <xref linkend=\"CEGDAEAG\">Synchronization and semaphores</xref>. For information about memory accesses see <xref linkend=\"Chddjfjf\">Memory accesses</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "STLB":
            return {
                "tooltip": "Store-Release Byte stores a byte from a register to memory.  The instruction also has memory ordering semantics as described in Load-Acquire, Store-Release.",
                "html": "<p>Store-Release Byte stores a byte from a register to memory.  The instruction also has memory ordering semantics as described in <xref linkend=\"AA32CHDBDIDF\">Load-Acquire, Store-Release</xref>.</p><p>For more information about support for shared memory see <xref linkend=\"CEGDAEAG\">Synchronization and semaphores</xref>. For information about memory accesses see <xref linkend=\"Chddjfjf\">Memory accesses</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "STLEX":
            return {
                "tooltip": "Store-Release Exclusive Word stores a word from a register to memory if the executing PE has exclusive access to the memory at that address, and returns a status value of 0 if the store was successful, or of 1 if no store was performed.",
                "html": "<p>Store-Release Exclusive Word stores a word from a register to memory if the executing PE has exclusive access to the memory at that address, and returns a status value of 0 if the store was successful, or of 1 if no store was performed.</p><p>The instruction also has memory ordering semantics as described in <xref linkend=\"AA32CHDBDIDF\">Load-Acquire, Store-Release</xref>.</p><p>For more information about support for shared memory see <xref linkend=\"CEGDAEAG\">Synchronization and semaphores</xref>. For information about memory accesses see <xref linkend=\"Chddjfjf\">Memory accesses</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "STLEXB":
            return {
                "tooltip": "Store-Release Exclusive Byte stores a byte from a register to memory if the executing PE has exclusive access to the memory at that address, and returns a status value of 0 if the store was successful, or of 1 if no store was performed.",
                "html": "<p>Store-Release Exclusive Byte stores a byte from a register to memory if the executing PE has exclusive access to the memory at that address, and returns a status value of 0 if the store was successful, or of 1 if no store was performed.</p><p>The instruction also has memory ordering semantics as described in <xref linkend=\"AA32CHDBDIDF\">Load-Acquire, Store-Release</xref>.</p><p>For more information about support for shared memory see <xref linkend=\"CEGDAEAG\">Synchronization and semaphores</xref>. For information about memory accesses see <xref linkend=\"Chddjfjf\">Memory accesses</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "STLEXD":
            return {
                "tooltip": "Store-Release Exclusive Doubleword stores a doubleword from two registers to memory if the executing PE has exclusive access to the memory at that address, and returns a status value of 0 if the store was successful, or of 1 if no store was performed.",
                "html": "<p>Store-Release Exclusive Doubleword stores a doubleword from two registers to memory if the executing PE has exclusive access to the memory at that address, and returns a status value of 0 if the store was successful, or of 1 if no store was performed.</p><p>The instruction also has memory ordering semantics as described in <xref linkend=\"AA32CHDBDIDF\">Load-Acquire, Store-Release</xref>.</p><p>For more information about support for shared memory see <xref linkend=\"CEGDAEAG\">Synchronization and semaphores</xref>. For information about memory accesses see <xref linkend=\"Chddjfjf\">Memory accesses</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "STLEXH":
            return {
                "tooltip": "Store-Release Exclusive Halfword stores a halfword from a register to memory if the executing PE has exclusive access to the memory at that address, and returns a status value of 0 if the store was successful, or of 1 if no store was performed.",
                "html": "<p>Store-Release Exclusive Halfword stores a halfword from a register to memory if the executing PE has exclusive access to the memory at that address, and returns a status value of 0 if the store was successful, or of 1 if no store was performed.</p><p>The instruction also has memory ordering semantics as described in <xref linkend=\"AA32CHDBDIDF\">Load-Acquire, Store-Release</xref>.</p><p>For more information about support for shared memory see <xref linkend=\"CEGDAEAG\">Synchronization and semaphores</xref>. For information about memory accesses see <xref linkend=\"Chddjfjf\">Memory accesses</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "STLH":
            return {
                "tooltip": "Store-Release Halfword stores a halfword from a register to memory. The instruction also has memory ordering semantics as described in Load-Acquire, Store-Release.",
                "html": "<p>Store-Release Halfword stores a halfword from a register to memory. The instruction also has memory ordering semantics as described in <xref linkend=\"AA32CHDBDIDF\">Load-Acquire, Store-Release</xref>.</p><p>For more information about support for shared memory see <xref linkend=\"CEGDAEAG\">Synchronization and semaphores</xref>. For information about memory accesses see <xref linkend=\"Chddjfjf\">Memory accesses</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "STM":
        case "STMEA":
        case "STMIA":
            return {
                "tooltip": "Store Multiple (Increment After, Empty Ascending) stores multiple registers to consecutive memory locations using an address from a base register. The consecutive memory locations start at this address, and the address just above the last of those locations can optionally be written back to the base register.",
                "html": "<p>Store Multiple (Increment After, Empty Ascending) stores multiple registers to consecutive memory locations using an address from a base register. The consecutive memory locations start at this address, and the address just above the last of those locations can optionally be written back to the base register.</p><p>The lowest-numbered register is loaded from the lowest memory address, through to the highest-numbered register from the highest memory address. See also <xref linkend=\"CHDDBEDG\">Encoding of lists of general-purpose registers and the PC</xref>.</p><p>Armv8.2 permits the deprecation of some Store Multiple ordering behaviors in AArch32 state, for more information see <xref linkend=\"v8.2.LSMAOC\">FEAT_LSMAOC</xref>. For details of related system instructions see <xref linkend=\"A32T32-base.instructions.STM_u\">STM (User registers)</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "STM":
            return {
                "tooltip": "In an EL1 mode other than System mode, Store Multiple (User registers) stores multiple User mode registers to consecutive memory locations using an address from a base register. The PE reads the base register value normally, using the current mode to determine the correct Banked version of the register. This instruction cannot writeback to the base register.",
                "html": "<p>In an EL1 mode other than System mode, Store Multiple (User registers) stores multiple User mode registers to consecutive memory locations using an address from a base register. The PE reads the base register value normally, using the current mode to determine the correct Banked version of the register. This instruction cannot writeback to the base register.</p><p>Store Multiple (User registers) is <arm-defined-word>undefined</arm-defined-word> in Hyp mode, and <arm-defined-word>constrained unpredictable</arm-defined-word> in User or System modes.</p><p>Armv8.2 permits the deprecation of some Store Multiple ordering behaviors in AArch32 state, for more information see <xref linkend=\"v8.2.LSMAOC\">FEAT_LSMAOC</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "STMDA":
        case "STMED":
            return {
                "tooltip": "Store Multiple Decrement After (Empty Descending) stores multiple registers to consecutive memory locations using an address from a base register. The consecutive memory locations end at this address, and the address just below the lowest of those locations can optionally be written back to the base register.",
                "html": "<p>Store Multiple Decrement After (Empty Descending) stores multiple registers to consecutive memory locations using an address from a base register. The consecutive memory locations end at this address, and the address just below the lowest of those locations can optionally be written back to the base register.</p><p>The lowest-numbered register is loaded from the lowest memory address, through to the highest-numbered register from the highest memory address. See also <xref linkend=\"CHDDBEDG\">Encoding of lists of general-purpose registers and the PC</xref>.</p><p>Armv8.2 permits the deprecation of some Store Multiple ordering behaviors in AArch32 state, for more information see <xref linkend=\"v8.2.LSMAOC\">FEAT_LSMAOC</xref>. For details of related system instructions see <xref linkend=\"A32T32-base.instructions.STM_u\">STM (User registers)</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "STMDB":
        case "STMFD":
            return {
                "tooltip": "Store Multiple Decrement Before (Full Descending) stores multiple registers to consecutive memory locations using an address from a base register. The consecutive memory locations end just below this address, and the address of the first of those locations can optionally be written back to the base register.",
                "html": "<p>Store Multiple Decrement Before (Full Descending) stores multiple registers to consecutive memory locations using an address from a base register. The consecutive memory locations end just below this address, and the address of the first of those locations can optionally be written back to the base register.</p><p>The lowest-numbered register is loaded from the lowest memory address, through to the highest-numbered register from the highest memory address. See also <xref linkend=\"CHDDBEDG\">Encoding of lists of general-purpose registers and the PC</xref>.</p><p>Armv8.2 permits the deprecation of some Store Multiple ordering behaviors in AArch32 state, for more information see <xref linkend=\"v8.2.LSMAOC\">FEAT_LSMAOC</xref>. For details of related system instructions see <xref linkend=\"A32T32-base.instructions.STM_u\">STM (User registers)</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "STMFA":
        case "STMIB":
            return {
                "tooltip": "Store Multiple Increment Before (Full Ascending) stores multiple registers to consecutive memory locations using an address from a base register. The consecutive memory locations start just above this address, and the address of the last of those locations can optionally be written back to the base register.",
                "html": "<p>Store Multiple Increment Before (Full Ascending) stores multiple registers to consecutive memory locations using an address from a base register. The consecutive memory locations start just above this address, and the address of the last of those locations can optionally be written back to the base register.</p><p>The lowest-numbered register is loaded from the lowest memory address, through to the highest-numbered register from the highest memory address. See also <xref linkend=\"CHDDBEDG\">Encoding of lists of general-purpose registers and the PC</xref>.</p><p>Armv8.2 permits the deprecation of some Store Multiple ordering behaviors in AArch32 state, for more information see <xref linkend=\"v8.2.LSMAOC\">FEAT_LSMAOC</xref>. For details of related system instructions see <xref linkend=\"A32T32-base.instructions.STM_u\">STM (User registers)</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "STR":
            return {
                "tooltip": "Store Register (immediate) calculates an address from a base register value and an immediate offset, and stores a word from a register to memory. It can use offset, post-indexed, or pre-indexed addressing. For information about memory accesses see Memory accesses.",
                "html": "<p>Store Register (immediate) calculates an address from a base register value and an immediate offset, and stores a word from a register to memory. It can use offset, post-indexed, or pre-indexed addressing. For information about memory accesses see <xref linkend=\"Chddjfjf\">Memory accesses</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "STR":
            return {
                "tooltip": "Store Register (register) calculates an address from a base register value and an offset register value, stores a word from a register to memory. The offset register value can optionally be shifted. For information about memory accesses see Memory accesses.",
                "html": "<p>Store Register (register) calculates an address from a base register value and an offset register value, stores a word from a register to memory. The offset register value can optionally be shifted. For information about memory accesses see <xref linkend=\"Chddjfjf\">Memory accesses</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "STRB":
            return {
                "tooltip": "Store Register Byte (immediate) calculates an address from a base register value and an immediate offset, and stores a byte from a register to memory. It can use offset, post-indexed, or pre-indexed addressing. For information about memory accesses see Memory accesses.",
                "html": "<p>Store Register Byte (immediate) calculates an address from a base register value and an immediate offset, and stores a byte from a register to memory. It can use offset, post-indexed, or pre-indexed addressing. For information about memory accesses see <xref linkend=\"Chddjfjf\">Memory accesses</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "STRB":
            return {
                "tooltip": "Store Register Byte (register) calculates an address from a base register value and an offset register value, and stores a byte from a register to memory. The offset register value can optionally be shifted. For information about memory accesses see Memory accesses.",
                "html": "<p>Store Register Byte (register) calculates an address from a base register value and an offset register value, and stores a byte from a register to memory. The offset register value can optionally be shifted. For information about memory accesses see <xref linkend=\"Chddjfjf\">Memory accesses</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "STRBT":
            return {
                "tooltip": "Store Register Byte Unprivileged stores a byte from a register to memory. For information about memory accesses see Memory accesses.",
                "html": "<p>Store Register Byte Unprivileged stores a byte from a register to memory. For information about memory accesses see <xref linkend=\"Chddjfjf\">Memory accesses</xref>.</p><p>The memory access is restricted as if the PE were running in User mode. This makes no difference if the PE is actually running in User mode.</p><p><instruction>STRBT</instruction> is <arm-defined-word>unpredictable</arm-defined-word> in Hyp mode.</p><p>The T32 instruction uses an offset addressing mode, that calculates the address used for the memory access from a base register value and an immediate offset, and leaves the base register unchanged.</p><p>The A32 instruction uses a post-indexed addressing mode, that uses a base register value as the address for the memory access, and calculates a new address from a base register value and an offset and writes it back to the base register. The offset can be an immediate value or an optionally-shifted register value.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "STRD":
            return {
                "tooltip": "Store Register Dual (immediate) calculates an address from a base register value and an immediate offset, and stores two words from two registers to memory. It can use offset, post-indexed, or pre-indexed addressing. For information about memory accesses see Memory accesses.",
                "html": "<p>Store Register Dual (immediate) calculates an address from a base register value and an immediate offset, and stores two words from two registers to memory. It can use offset, post-indexed, or pre-indexed addressing. For information about memory accesses see <xref linkend=\"Chddjfjf\">Memory accesses</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "STRD":
            return {
                "tooltip": "Store Register Dual (register) calculates an address from a base register value and a register offset, and stores two words from two registers to memory. It can use offset, post-indexed, or pre-indexed addressing. For information about memory accesses see Memory accesses.",
                "html": "<p>Store Register Dual (register) calculates an address from a base register value and a register offset, and stores two words from two registers to memory. It can use offset, post-indexed, or pre-indexed addressing. For information about memory accesses see <xref linkend=\"Chddjfjf\">Memory accesses</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "STREX":
            return {
                "tooltip": "Store Register Exclusive calculates an address from a base register value and an immediate offset, stores a word from a register to the calculated address if the PE has exclusive access to the memory at that address, and returns a status value of 0 if the store was successful, or of 1 if no store was performed.",
                "html": "<p>Store Register Exclusive calculates an address from a base register value and an immediate offset, stores a word from a register to the calculated address if the PE has exclusive access to the memory at that address, and returns a status value of 0 if the store was successful, or of 1 if no store was performed.</p><p>For more information about support for shared memory see <xref linkend=\"CEGDAEAG\">Synchronization and semaphores</xref>. For information about memory accesses see <xref linkend=\"Chddjfjf\">Memory accesses</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "STREXB":
            return {
                "tooltip": "Store Register Exclusive Byte derives an address from a base register value, stores a byte from a register to the derived address if the executing PE has exclusive access to the memory at that address, and returns a status value of 0 if the store was successful, or of 1 if no store was performed.",
                "html": "<p>Store Register Exclusive Byte derives an address from a base register value, stores a byte from a register to the derived address if the executing PE has exclusive access to the memory at that address, and returns a status value of 0 if the store was successful, or of 1 if no store was performed.</p><p>For more information about support for shared memory see <xref linkend=\"CEGDAEAG\">Synchronization and semaphores</xref>. For information about memory accesses see <xref linkend=\"Chddjfjf\">Memory accesses</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "STREXD":
            return {
                "tooltip": "Store Register Exclusive Doubleword derives an address from a base register value, stores a 64-bit doubleword from two registers to the derived address if the executing PE has exclusive access to the memory at that address, and returns a status value of 0 if the store was successful, or of 1 if no store was performed.",
                "html": "<p>Store Register Exclusive Doubleword derives an address from a base register value, stores a 64-bit doubleword from two registers to the derived address if the executing PE has exclusive access to the memory at that address, and returns a status value of 0 if the store was successful, or of 1 if no store was performed.</p><p>For more information about support for shared memory see <xref linkend=\"CEGDAEAG\">Synchronization and semaphores</xref>. For information about memory accesses see <xref linkend=\"Chddjfjf\">Memory accesses</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "STREXH":
            return {
                "tooltip": "Store Register Exclusive Halfword derives an address from a base register value, stores a halfword from a register to the derived address if the executing PE has exclusive access to the memory at that address, and returns a status value of 0 if the store was successful, or of 1 if no store was performed.",
                "html": "<p>Store Register Exclusive Halfword derives an address from a base register value, stores a halfword from a register to the derived address if the executing PE has exclusive access to the memory at that address, and returns a status value of 0 if the store was successful, or of 1 if no store was performed.</p><p>For more information about support for shared memory see <xref linkend=\"CEGDAEAG\">Synchronization and semaphores</xref>. For information about memory accesses see <xref linkend=\"Chddjfjf\">Memory accesses</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "STRH":
            return {
                "tooltip": "Store Register Halfword (immediate) calculates an address from a base register value and an immediate offset, and stores a halfword from a register to memory. It can use offset, post-indexed, or pre-indexed addressing. For information about memory accesses see Memory accesses.",
                "html": "<p>Store Register Halfword (immediate) calculates an address from a base register value and an immediate offset, and stores a halfword from a register to memory. It can use offset, post-indexed, or pre-indexed addressing. For information about memory accesses see <xref linkend=\"Chddjfjf\">Memory accesses</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "STRH":
            return {
                "tooltip": "Store Register Halfword (register) calculates an address from a base register value and an offset register value, and stores a halfword from a register to memory. The offset register value can be shifted left by 0, 1, 2, or 3 bits. For information about memory accesses see Memory accesses.",
                "html": "<p>Store Register Halfword (register) calculates an address from a base register value and an offset register value, and stores a halfword from a register to memory. The offset register value can be shifted left by 0, 1, 2, or 3 bits. For information about memory accesses see <xref linkend=\"Chddjfjf\">Memory accesses</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "STRHT":
            return {
                "tooltip": "Store Register Halfword Unprivileged stores a halfword from a register to memory. For information about memory accesses see Memory accesses.",
                "html": "<p>Store Register Halfword Unprivileged stores a halfword from a register to memory. For information about memory accesses see <xref linkend=\"Chddjfjf\">Memory accesses</xref>.</p><p>The memory access is restricted as if the PE were running in User mode. This makes no difference if the PE is actually running in User mode.</p><p><instruction>STRHT</instruction> is <arm-defined-word>unpredictable</arm-defined-word> in Hyp mode.</p><p>The T32 instruction uses an offset addressing mode, that calculates the address used for the memory access from a base register value and an immediate offset, and leaves the base register unchanged.</p><p>The A32 instruction uses a post-indexed addressing mode, that uses a base register value as the address for the memory access, and calculates a new address from a base register value and an offset and writes it back to the base register. The offset can be an immediate value or a register value.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "STRT":
            return {
                "tooltip": "Store Register Unprivileged stores a word from a register to memory. For information about memory accesses see Memory accesses.",
                "html": "<p>Store Register Unprivileged stores a word from a register to memory. For information about memory accesses see <xref linkend=\"Chddjfjf\">Memory accesses</xref>.</p><p>The memory access is restricted as if the PE were running in User mode. This makes no difference if the PE is actually running in User mode.</p><p><instruction>STRT</instruction> is <arm-defined-word>unpredictable</arm-defined-word> in Hyp mode.</p><p>The T32 instruction uses an offset addressing mode, that calculates the address used for the memory access from a base register value and an immediate offset, and leaves the base register unchanged.</p><p>The A32 instruction uses a post-indexed addressing mode, that uses a base register value as the address for the memory access, and calculates a new address from a base register value and an offset and writes it back to the base register. The offset can be an immediate value or an optionally-shifted register value.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "SUB":
        case "SUBS":
            return {
                "tooltip": "Subtract (immediate) subtracts an immediate value from a register value, and writes the result to the destination register.",
                "html": "<p>Subtract (immediate) subtracts an immediate value from a register value, and writes the result to the destination register.</p><p>If the destination register is not the PC, the SUBS variant of the instruction updates the condition flags based on the result.</p><p>The field descriptions for <syntax>&lt;Rd&gt;</syntax> identify the encodings where the PC is permitted as the destination register. If the destination register is the PC:</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "SUB":
        case "SUBS":
            return {
                "tooltip": "Subtract (register) subtracts an optionally-shifted register value from a register value, and writes the result to the destination register.",
                "html": "<p>Subtract (register) subtracts an optionally-shifted register value from a register value, and writes the result to the destination register.</p><p>If the destination register is not the PC, the SUBS variant of the instruction updates the condition flags based on the result.</p><p>The field descriptions for <syntax>&lt;Rd&gt;</syntax> identify the encodings where the PC is permitted as the destination register. However, when the destination register is the PC:</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "SUB":
        case "SUBS":
            return {
                "tooltip": "Subtract (register-shifted register) subtracts a register-shifted register value from a register value, and writes the result to the destination register. It can optionally update the condition flags based on the result.",
                "html": "<p>Subtract (register-shifted register) subtracts a register-shifted register value from a register value, and writes the result to the destination register. It can optionally update the condition flags based on the result.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "SUB":
        case "SUBS":
            return {
                "tooltip": "Subtract from SP (immediate) subtracts an immediate value from the SP value, and writes the result to the destination register.",
                "html": "<p>Subtract from SP (immediate) subtracts an immediate value from the SP value, and writes the result to the destination register.</p><p>If the destination register is not the PC, the SUBS variant of the instruction updates the condition flags based on the result.</p><p>The field descriptions for <syntax>&lt;Rd&gt;</syntax> identify the encodings where the PC is permitted as the destination register. If the destination register is the PC:</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "SUB":
        case "SUBS":
            return {
                "tooltip": "Subtract from SP (register) subtracts an optionally-shifted register value from the SP value, and writes the result to the destination register.",
                "html": "<p>Subtract from SP (register) subtracts an optionally-shifted register value from the SP value, and writes the result to the destination register.</p><p>If the destination register is not the PC, the SUBS variant of the instruction updates the condition flags based on the result.</p><p>The field descriptions for <syntax>&lt;Rd&gt;</syntax> identify the encodings where the PC is permitted as the destination register. If the destination register is the PC:</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "SVC":
            return {
                "tooltip": "Supervisor Call causes a Supervisor Call exception. For more information, see Supervisor Call (SVC) exception.",
                "html": "<p>Supervisor Call causes a Supervisor Call exception. For more information, see <xref linkend=\"CIHFBAIJ\">Supervisor Call (SVC) exception</xref>.</p><p><instruction>SVC</instruction> was previously called <instruction>SWI</instruction>, Software Interrupt, and this name is still found in some documentation.</p><p>Software can use this instruction as a call to an operating system to provide a service.</p><p>In the following cases, the Supervisor Call exception generated by the <instruction>SVC</instruction> instruction is taken to Hyp mode:</p><p>In these cases, the <xref linkend=\"AArch32.hsr\">HSR, Hyp Syndrome Register</xref> identifies that the exception entry was caused by a Supervisor Call exception, EC value <hexnumber>0x11</hexnumber>, see <xref linkend=\"BEIDBEAG\">Use of the HSR</xref>. The immediate field in the <xref linkend=\"AArch32.hsr\">HSR</xref>:</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "SXTAB":
            return {
                "tooltip": "Signed Extend and Add Byte extracts an 8-bit value from a register, sign-extends it to 32 bits, adds the result to the value in another register, and writes the final result to the destination register. The instruction can specify a rotation by 0, 8, 16, or 24 bits before extracting the 8-bit value.",
                "html": "<p>Signed Extend and Add Byte extracts an 8-bit value from a register, sign-extends it to 32 bits, adds the result to the value in another register, and writes the final result to the destination register. The instruction can specify a rotation by 0, 8, 16, or 24 bits before extracting the 8-bit value.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "SXTAB16":
            return {
                "tooltip": "Signed Extend and Add Byte 16 extracts two 8-bit values from a register, sign-extends them to 16 bits each, adds the results to two 16-bit values from another register, and writes the final results to the destination register. The instruction can specify a rotation by 0, 8, 16, or 24 bits before extracting the 8-bit values.",
                "html": "<p>Signed Extend and Add Byte 16 extracts two 8-bit values from a register, sign-extends them to 16 bits each, adds the results to two 16-bit values from another register, and writes the final results to the destination register. The instruction can specify a rotation by 0, 8, 16, or 24 bits before extracting the 8-bit values.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "SXTAH":
            return {
                "tooltip": "Signed Extend and Add Halfword extracts a 16-bit value from a register, sign-extends it to 32 bits, adds the result to a value from another register, and writes the final result to the destination register. The instruction can specify a rotation by 0, 8, 16, or 24 bits before extracting the 16-bit value.",
                "html": "<p>Signed Extend and Add Halfword extracts a 16-bit value from a register, sign-extends it to 32 bits, adds the result to a value from another register, and writes the final result to the destination register. The instruction can specify a rotation by 0, 8, 16, or 24 bits before extracting the 16-bit value.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "SXTB":
            return {
                "tooltip": "Signed Extend Byte extracts an 8-bit value from a register, sign-extends it to 32 bits, and writes the result to the destination register. The instruction can specify a rotation by 0, 8, 16, or 24 bits before extracting the 8-bit value.",
                "html": "<p>Signed Extend Byte extracts an 8-bit value from a register, sign-extends it to 32 bits, and writes the result to the destination register. The instruction can specify a rotation by 0, 8, 16, or 24 bits before extracting the 8-bit value.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "SXTB16":
            return {
                "tooltip": "Signed Extend Byte 16 extracts two 8-bit values from a register, sign-extends them to 16 bits each, and writes the results to the destination register. The instruction can specify a rotation by 0, 8, 16, or 24 bits before extracting the 8-bit values.",
                "html": "<p>Signed Extend Byte 16 extracts two 8-bit values from a register, sign-extends them to 16 bits each, and writes the results to the destination register. The instruction can specify a rotation by 0, 8, 16, or 24 bits before extracting the 8-bit values.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "SXTH":
            return {
                "tooltip": "Signed Extend Halfword extracts a 16-bit value from a register, sign-extends it to 32 bits, and writes the result to the destination register. The instruction can specify a rotation by 0, 8, 16, or 24 bits before extracting the 16-bit value.",
                "html": "<p>Signed Extend Halfword extracts a 16-bit value from a register, sign-extends it to 32 bits, and writes the result to the destination register. The instruction can specify a rotation by 0, 8, 16, or 24 bits before extracting the 16-bit value.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "TBB":
        case "TBH":
            return {
                "tooltip": "Table Branch Byte or Halfword causes a PC-relative forward branch using a table of single byte or halfword offsets. A base register provides a pointer to the table, and a second register supplies an index into the table. The branch length is twice the value returned from the table.",
                "html": "<p>Table Branch Byte or Halfword causes a PC-relative forward branch using a table of single byte or halfword offsets. A base register provides a pointer to the table, and a second register supplies an index into the table. The branch length is twice the value returned from the table.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "TEQ":
            return {
                "tooltip": "Test Equivalence (immediate) performs a bitwise exclusive OR operation on a register value and an immediate value. It updates the condition flags based on the result, and discards the result.",
                "html": "<p>Test Equivalence (immediate) performs a bitwise exclusive OR operation on a register value and an immediate value. It updates the condition flags based on the result, and discards the result.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "TEQ":
            return {
                "tooltip": "Test Equivalence (register) performs a bitwise exclusive-OR operation on a register value and an optionally-shifted register value. It updates the condition flags based on the result, and discards the result.",
                "html": "<p>Test Equivalence (register) performs a bitwise exclusive-OR operation on a register value and an optionally-shifted register value. It updates the condition flags based on the result, and discards the result.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "TEQ":
            return {
                "tooltip": "Test Equivalence (register-shifted register) performs a bitwise exclusive-OR operation on a register value and a register-shifted register value. It updates the condition flags based on the result, and discards the result.",
                "html": "<p>Test Equivalence (register-shifted register) performs a bitwise exclusive-OR operation on a register value and a register-shifted register value. It updates the condition flags based on the result, and discards the result.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "TSBCSYNC":
            return {
                "tooltip": "Trace Synchronization Barrier. This instruction is a barrier that synchronizes the trace operations of instructions, see Trace Synchronization Buffer (TSB CSYNC).",
                "html": "<p>Trace Synchronization Barrier. This instruction is a barrier that synchronizes the trace operations of instructions, see <xref linkend=\"CJAHFCID\">Trace Synchronization Buffer (TSB CSYNC)</xref>.</p><p>If <xref linkend=\"v8.4.Trace\">FEAT_TRF</xref> is not implemented, this instruction executes as a <instruction>NOP</instruction>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "TST":
            return {
                "tooltip": "Test (immediate) performs a bitwise AND operation on a register value and an immediate value. It updates the condition flags based on the result, and discards the result.",
                "html": "<p>Test (immediate) performs a bitwise AND operation on a register value and an immediate value. It updates the condition flags based on the result, and discards the result.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "TST":
            return {
                "tooltip": "Test (register) performs a bitwise AND operation on a register value and an optionally-shifted register value. It updates the condition flags based on the result, and discards the result.",
                "html": "<p>Test (register) performs a bitwise AND operation on a register value and an optionally-shifted register value. It updates the condition flags based on the result, and discards the result.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "TST":
            return {
                "tooltip": "Test (register-shifted register) performs a bitwise AND operation on a register value and a register-shifted register value. It updates the condition flags based on the result, and discards the result.",
                "html": "<p>Test (register-shifted register) performs a bitwise AND operation on a register value and a register-shifted register value. It updates the condition flags based on the result, and discards the result.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "UADD16":
            return {
                "tooltip": "Unsigned Add 16 performs two 16-bit unsigned integer additions, and writes the results to the destination register. It sets PSTATE.GE according to the results of the additions.",
                "html": "<p>Unsigned Add 16 performs two 16-bit unsigned integer additions, and writes the results to the destination register. It sets <xref linkend=\"BEIDIGBH\">PSTATE</xref>.GE according to the results of the additions.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "UADD8":
            return {
                "tooltip": "Unsigned Add 8 performs four unsigned 8-bit integer additions, and writes the results to the destination register. It sets PSTATE.GE according to the results of the additions.",
                "html": "<p>Unsigned Add 8 performs four unsigned 8-bit integer additions, and writes the results to the destination register. It sets <xref linkend=\"BEIDIGBH\">PSTATE</xref>.GE according to the results of the additions.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "UASX":
            return {
                "tooltip": "Unsigned Add and Subtract with Exchange exchanges the two halfwords of the second operand, performs one unsigned 16-bit integer addition and one unsigned 16-bit subtraction, and writes the results to the destination register. It sets PSTATE.GE according to the results.",
                "html": "<p>Unsigned Add and Subtract with Exchange exchanges the two halfwords of the second operand, performs one unsigned 16-bit integer addition and one unsigned 16-bit subtraction, and writes the results to the destination register. It sets <xref linkend=\"BEIDIGBH\">PSTATE</xref>.GE according to the results.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "UBFX":
            return {
                "tooltip": "Unsigned Bit Field Extract extracts any number of adjacent bits at any position from a register, zero-extends them to 32 bits, and writes the result to the destination register.",
                "html": "<p>Unsigned Bit Field Extract extracts any number of adjacent bits at any position from a register, zero-extends them to 32 bits, and writes the result to the destination register.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "UDF":
            return {
                "tooltip": "Permanently Undefined generates an Undefined Instruction exception.",
                "html": "<p>Permanently Undefined generates an Undefined Instruction exception.</p><p>The encodings for <instruction>UDF</instruction> used in this section are defined as permanently <arm-defined-word>undefined</arm-defined-word>. However:</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "UDIV":
            return {
                "tooltip": "Unsigned Divide divides a 32-bit unsigned integer register value by a 32-bit unsigned integer register value, and writes the result to the destination register. The condition flags are not affected.",
                "html": "<p>Unsigned Divide divides a 32-bit unsigned integer register value by a 32-bit unsigned integer register value, and writes the result to the destination register. The condition flags are not affected.</p><p>See <xref linkend=\"CFIBGHCC\">Divide instructions</xref> for more information about this instruction.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "UHADD16":
            return {
                "tooltip": "Unsigned Halving Add 16 performs two unsigned 16-bit integer additions, halves the results, and writes the results to the destination register.",
                "html": "<p>Unsigned Halving Add 16 performs two unsigned 16-bit integer additions, halves the results, and writes the results to the destination register.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "UHADD8":
            return {
                "tooltip": "Unsigned Halving Add 8 performs four unsigned 8-bit integer additions, halves the results, and writes the results to the destination register.",
                "html": "<p>Unsigned Halving Add 8 performs four unsigned 8-bit integer additions, halves the results, and writes the results to the destination register.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "UHASX":
            return {
                "tooltip": "Unsigned Halving Add and Subtract with Exchange exchanges the two halfwords of the second operand, performs one unsigned 16-bit integer addition and one unsigned 16-bit subtraction, halves the results, and writes the results to the destination register.",
                "html": "<p>Unsigned Halving Add and Subtract with Exchange exchanges the two halfwords of the second operand, performs one unsigned 16-bit integer addition and one unsigned 16-bit subtraction, halves the results, and writes the results to the destination register.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "UHSAX":
            return {
                "tooltip": "Unsigned Halving Subtract and Add with Exchange exchanges the two halfwords of the second operand, performs one unsigned 16-bit integer subtraction and one unsigned 16-bit addition, halves the results, and writes the results to the destination register.",
                "html": "<p>Unsigned Halving Subtract and Add with Exchange exchanges the two halfwords of the second operand, performs one unsigned 16-bit integer subtraction and one unsigned 16-bit addition, halves the results, and writes the results to the destination register.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "UHSUB16":
            return {
                "tooltip": "Unsigned Halving Subtract 16 performs two unsigned 16-bit integer subtractions, halves the results, and writes the results to the destination register.",
                "html": "<p>Unsigned Halving Subtract 16 performs two unsigned 16-bit integer subtractions, halves the results, and writes the results to the destination register.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "UHSUB8":
            return {
                "tooltip": "Unsigned Halving Subtract 8 performs four unsigned 8-bit integer subtractions, halves the results, and writes the results to the destination register.",
                "html": "<p>Unsigned Halving Subtract 8 performs four unsigned 8-bit integer subtractions, halves the results, and writes the results to the destination register.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "UMAAL":
            return {
                "tooltip": "Unsigned Multiply Accumulate Accumulate Long multiplies two unsigned 32-bit values to produce a 64-bit value, adds two unsigned 32-bit values, and writes the 64-bit result to two registers.",
                "html": "<p>Unsigned Multiply Accumulate Accumulate Long multiplies two unsigned 32-bit values to produce a 64-bit value, adds two unsigned 32-bit values, and writes the 64-bit result to two registers.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "UMLAL":
        case "UMLALS":
            return {
                "tooltip": "Unsigned Multiply Accumulate Long multiplies two unsigned 32-bit values to produce a 64-bit value, and accumulates this with a 64-bit value.",
                "html": "<p>Unsigned Multiply Accumulate Long multiplies two unsigned 32-bit values to produce a 64-bit value, and accumulates this with a 64-bit value.</p><p>In A32 instructions, the condition flags can optionally be updated based on the result. Use of this option adversely affects performance on many implementations.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "UMULL":
        case "UMULLS":
            return {
                "tooltip": "Unsigned Multiply Long multiplies two 32-bit unsigned values to produce a 64-bit result.",
                "html": "<p>Unsigned Multiply Long multiplies two 32-bit unsigned values to produce a 64-bit result.</p><p>In A32 instructions, the condition flags can optionally be updated based on the result. Use of this option adversely affects performance on many implementations.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "UQADD16":
            return {
                "tooltip": "Unsigned Saturating Add 16 performs two unsigned 16-bit integer additions, saturates the results to the 16-bit unsigned integer range 0 <= x <= 216 - 1, and writes the results to the destination register.",
                "html": "<p>Unsigned Saturating Add 16 performs two unsigned 16-bit integer additions, saturates the results to the 16-bit unsigned integer range 0 &lt;= x &lt;= 2<sup>16</sup> - 1, and writes the results to the destination register.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "UQADD8":
            return {
                "tooltip": "Unsigned Saturating Add 8 performs four unsigned 8-bit integer additions, saturates the results to the 8-bit unsigned integer range 0 <= x <= 28 - 1, and writes the results to the destination register.",
                "html": "<p>Unsigned Saturating Add 8 performs four unsigned 8-bit integer additions, saturates the results to the 8-bit unsigned integer range 0 &lt;= x &lt;= 2<sup>8</sup> - 1, and writes the results to the destination register.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "UQASX":
            return {
                "tooltip": "Unsigned Saturating Add and Subtract with Exchange exchanges the two halfwords of the second operand, performs one unsigned 16-bit integer addition and one unsigned 16-bit subtraction, saturates the results to the 16-bit unsigned integer range 0 <= x <= 216 - 1, and writes the results to the destination register.",
                "html": "<p>Unsigned Saturating Add and Subtract with Exchange exchanges the two halfwords of the second operand, performs one unsigned 16-bit integer addition and one unsigned 16-bit subtraction, saturates the results to the 16-bit unsigned integer range 0 &lt;= x &lt;= 2<sup>16</sup> - 1, and writes the results to the destination register.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "UQSAX":
            return {
                "tooltip": "Unsigned Saturating Subtract and Add with Exchange exchanges the two halfwords of the second operand, performs one unsigned 16-bit integer subtraction and one unsigned 16-bit addition, saturates the results to the 16-bit unsigned integer range 0 <= x <= 216 - 1, and writes the results to the destination register.",
                "html": "<p>Unsigned Saturating Subtract and Add with Exchange exchanges the two halfwords of the second operand, performs one unsigned 16-bit integer subtraction and one unsigned 16-bit addition, saturates the results to the 16-bit unsigned integer range 0 &lt;= x &lt;= 2<sup>16</sup> - 1, and writes the results to the destination register.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "UQSUB16":
            return {
                "tooltip": "Unsigned Saturating Subtract 16 performs two unsigned 16-bit integer subtractions, saturates the results to the 16-bit unsigned integer range 0 <= x <= 216 - 1, and writes the results to the destination register.",
                "html": "<p>Unsigned Saturating Subtract 16 performs two unsigned 16-bit integer subtractions, saturates the results to the 16-bit unsigned integer range 0 &lt;= x &lt;= 2<sup>16</sup> - 1, and writes the results to the destination register.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "UQSUB8":
            return {
                "tooltip": "Unsigned Saturating Subtract 8 performs four unsigned 8-bit integer subtractions, saturates the results to the 8-bit unsigned integer range 0 <= x <= 28 - 1, and writes the results to the destination register.",
                "html": "<p>Unsigned Saturating Subtract 8 performs four unsigned 8-bit integer subtractions, saturates the results to the 8-bit unsigned integer range 0 &lt;= x &lt;= 2<sup>8</sup> - 1, and writes the results to the destination register.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "USAD8":
            return {
                "tooltip": "Unsigned Sum of Absolute Differences performs four unsigned 8-bit subtractions, and adds the absolute values of the differences together.",
                "html": "<p>Unsigned Sum of Absolute Differences performs four unsigned 8-bit subtractions, and adds the absolute values of the differences together.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "USADA8":
            return {
                "tooltip": "Unsigned Sum of Absolute Differences and Accumulate performs four unsigned 8-bit subtractions, and adds the absolute values of the differences to a 32-bit accumulate operand.",
                "html": "<p>Unsigned Sum of Absolute Differences and Accumulate performs four unsigned 8-bit subtractions, and adds the absolute values of the differences to a 32-bit accumulate operand.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "USAT":
            return {
                "tooltip": "Unsigned Saturate saturates an optionally-shifted signed value to a selected unsigned range.",
                "html": "<p>Unsigned Saturate saturates an optionally-shifted signed value to a selected unsigned range.</p><p>This instruction sets <xref linkend=\"BEIDIGBH\">PSTATE</xref>.Q to 1 if the operation saturates.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "USAT16":
            return {
                "tooltip": "Unsigned Saturate 16 saturates two signed 16-bit values to a selected unsigned range.",
                "html": "<p>Unsigned Saturate 16 saturates two signed 16-bit values to a selected unsigned range.</p><p>This instruction sets <xref linkend=\"BEIDIGBH\">PSTATE</xref>.Q to 1 if the operation saturates.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "USAX":
            return {
                "tooltip": "Unsigned Subtract and Add with Exchange exchanges the two halfwords of the second operand, performs one unsigned 16-bit integer subtraction and one unsigned 16-bit addition, and writes the results to the destination register. It sets PSTATE.GE according to the results.",
                "html": "<p>Unsigned Subtract and Add with Exchange exchanges the two halfwords of the second operand, performs one unsigned 16-bit integer subtraction and one unsigned 16-bit addition, and writes the results to the destination register. It sets <xref linkend=\"BEIDIGBH\">PSTATE</xref>.GE according to the results.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "USUB16":
            return {
                "tooltip": "Unsigned Subtract 16 performs two 16-bit unsigned integer subtractions, and writes the results to the destination register. It sets PSTATE.GE according to the results of the subtractions.",
                "html": "<p>Unsigned Subtract 16 performs two 16-bit unsigned integer subtractions, and writes the results to the destination register. It sets <xref linkend=\"BEIDIGBH\">PSTATE</xref>.GE according to the results of the subtractions.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "USUB8":
            return {
                "tooltip": "Unsigned Subtract 8 performs four 8-bit unsigned integer subtractions, and writes the results to the destination register. It sets PSTATE.GE according to the results of the subtractions.",
                "html": "<p>Unsigned Subtract 8 performs four 8-bit unsigned integer subtractions, and writes the results to the destination register. It sets <xref linkend=\"BEIDIGBH\">PSTATE</xref>.GE according to the results of the subtractions.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "UXTAB":
            return {
                "tooltip": "Unsigned Extend and Add Byte extracts an 8-bit value from a register, zero-extends it to 32 bits, adds the result to the value in another register, and writes the final result to the destination register. The instruction can specify a rotation by 0, 8, 16, or 24 bits before extracting the 8-bit value.",
                "html": "<p>Unsigned Extend and Add Byte extracts an 8-bit value from a register, zero-extends it to 32 bits, adds the result to the value in another register, and writes the final result to the destination register. The instruction can specify a rotation by 0, 8, 16, or 24 bits before extracting the 8-bit value.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "UXTAB16":
            return {
                "tooltip": "Unsigned Extend and Add Byte 16 extracts two 8-bit values from a register, zero-extends them to 16 bits each, adds the results to two 16-bit values from another register, and writes the final results to the destination register. The instruction can specify a rotation by 0, 8, 16, or 24 bits before extracting the 8-bit values.",
                "html": "<p>Unsigned Extend and Add Byte 16 extracts two 8-bit values from a register, zero-extends them to 16 bits each, adds the results to two 16-bit values from another register, and writes the final results to the destination register. The instruction can specify a rotation by 0, 8, 16, or 24 bits before extracting the 8-bit values.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "UXTAH":
            return {
                "tooltip": "Unsigned Extend and Add Halfword extracts a 16-bit value from a register, zero-extends it to 32 bits, adds the result to a value from another register, and writes the final result to the destination register. The instruction can specify a rotation by 0, 8, 16, or 24 bits before extracting the 16-bit value.",
                "html": "<p>Unsigned Extend and Add Halfword extracts a 16-bit value from a register, zero-extends it to 32 bits, adds the result to a value from another register, and writes the final result to the destination register. The instruction can specify a rotation by 0, 8, 16, or 24 bits before extracting the 16-bit value.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "UXTB":
            return {
                "tooltip": "Unsigned Extend Byte extracts an 8-bit value from a register, zero-extends it to 32 bits, and writes the result to the destination register. The instruction can specify a rotation by 0, 8, 16, or 24 bits before extracting the 8-bit value.",
                "html": "<p>Unsigned Extend Byte extracts an 8-bit value from a register, zero-extends it to 32 bits, and writes the result to the destination register. The instruction can specify a rotation by 0, 8, 16, or 24 bits before extracting the 8-bit value.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "UXTB16":
            return {
                "tooltip": "Unsigned Extend Byte 16 extracts two 8-bit values from a register, zero-extends them to 16 bits each, and writes the results to the destination register. The instruction can specify a rotation by 0, 8, 16, or 24 bits before extracting the 8-bit values.",
                "html": "<p>Unsigned Extend Byte 16 extracts two 8-bit values from a register, zero-extends them to 16 bits each, and writes the results to the destination register. The instruction can specify a rotation by 0, 8, 16, or 24 bits before extracting the 8-bit values.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "UXTH":
            return {
                "tooltip": "Unsigned Extend Halfword extracts a 16-bit value from a register, zero-extends it to 32 bits, and writes the result to the destination register. The instruction can specify a rotation by 0, 8, 16, or 24 bits before extracting the 16-bit value.",
                "html": "<p>Unsigned Extend Halfword extracts a 16-bit value from a register, zero-extends it to 32 bits, and writes the result to the destination register. The instruction can specify a rotation by 0, 8, 16, or 24 bits before extracting the 16-bit value.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VABA":
            return {
                "tooltip": "Vector Absolute Difference and Accumulate subtracts the elements of one vector from the corresponding elements of another vector, and accumulates the absolute values of the results into the elements of the destination vector.",
                "html": "<p>Vector Absolute Difference and Accumulate subtracts the elements of one vector from the corresponding elements of another vector, and accumulates the absolute values of the results into the elements of the destination vector.</p><p>Operand and result elements are all integers of the same length.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VABAL":
            return {
                "tooltip": "Vector Absolute Difference and Accumulate Long subtracts the elements of one vector from the corresponding elements of another vector, and accumulates the absolute values of the results into the elements of the destination vector.",
                "html": "<p>Vector Absolute Difference and Accumulate Long subtracts the elements of one vector from the corresponding elements of another vector, and accumulates the absolute values of the results into the elements of the destination vector.</p><p>Operand elements are all integers of the same length, and the result elements are double the length of the operands.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VABD":
            return {
                "tooltip": "Vector Absolute Difference (floating-point) subtracts the elements of one vector from the corresponding elements of another vector, and places the absolute values of the results in the elements of the destination vector.",
                "html": "<p>Vector Absolute Difference (floating-point) subtracts the elements of one vector from the corresponding elements of another vector, and places the absolute values of the results in the elements of the destination vector.</p><p>Operand and result elements are floating-point numbers of the same size.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VABD":
            return {
                "tooltip": "Vector Absolute Difference (integer) subtracts the elements of one vector from the corresponding elements of another vector, and places the absolute values of the results in the elements of the destination vector.",
                "html": "<p>Vector Absolute Difference (integer) subtracts the elements of one vector from the corresponding elements of another vector, and places the absolute values of the results in the elements of the destination vector.</p><p>Operand and result elements are all integers of the same length.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VABDL":
            return {
                "tooltip": "Vector Absolute Difference Long (integer) subtracts the elements of one vector from the corresponding elements of another vector, and places the absolute values of the results in the elements of the destination vector.",
                "html": "<p>Vector Absolute Difference Long (integer) subtracts the elements of one vector from the corresponding elements of another vector, and places the absolute values of the results in the elements of the destination vector.</p><p>Operand elements are all integers of the same length, and the result elements are double the length of the operands.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VABS":
            return {
                "tooltip": "Vector Absolute takes the absolute value of each element in a vector, and places the results in a second vector. The floating-point version only clears the sign bit.",
                "html": "<p>Vector Absolute takes the absolute value of each element in a vector, and places the results in a second vector. The floating-point version only clears the sign bit.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, <xref linkend=\"AArch32.hcptr\">HCPTR</xref>, and <xref linkend=\"AArch32.fpexc\">FPEXC</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VACGE":
            return {
                "tooltip": "Vector Absolute Compare Greater Than or Equal takes the absolute value of each element in a vector, and compares it with the absolute value of the corresponding element of a second vector. If the first is greater than or equal to the second, the corresponding element in the destination vector is set to all ones. Otherwise, it is set to all zeros.",
                "html": "<p>Vector Absolute Compare Greater Than or Equal takes the absolute value of each element in a vector, and compares it with the absolute value of the corresponding element of a second vector. If the first is greater than or equal to the second, the corresponding element in the destination vector is set to all ones. Otherwise, it is set to all zeros.</p><p>The operands and result can be quadword or doubleword vectors.  They must all be the same size.</p><p>The operand vector elements are floating-point numbers. The result vector elements are the same size as the operand vector elements.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VACGT":
            return {
                "tooltip": "Vector Absolute Compare Greater Than takes the absolute value of each element in a vector, and compares it with the absolute value of the corresponding element of a second vector. If the first is greater than the second, the corresponding element in the destination vector is set to all ones. Otherwise, it is set to all zeros.",
                "html": "<p>Vector Absolute Compare Greater Than takes the absolute value of each element in a vector, and compares it with the absolute value of the corresponding element of a second vector. If the first is greater than the second, the corresponding element in the destination vector is set to all ones. Otherwise, it is set to all zeros.</p><p>The operands and result can be quadword or doubleword vectors.  They must all be the same size.</p><p>The operand vector elements are floating-point numbers. The result vector elements are the same size as the operand vector elements.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VACLE":
            return {
                "tooltip": "Vector Absolute Compare Less Than or Equal takes the absolute value of each element in a vector, and compares it with the absolute value of the corresponding element of a second vector. If the first is less than or equal to the second, the corresponding element in the destination vector is set to all ones. Otherwise, it is set to all zeros.",
                "html": "<p>Vector Absolute Compare Less Than or Equal takes the absolute value of each element in a vector, and compares it with the absolute value of the corresponding element of a second vector. If the first is less than or equal to the second, the corresponding element in the destination vector is set to all ones. Otherwise, it is set to all zeros.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VACLT":
            return {
                "tooltip": "Vector Absolute Compare Less Than takes the absolute value of each element in a vector, and compares it with the absolute value of the corresponding element of a second vector. If the first is less than the second, the corresponding element in the destination vector is set to all ones. Otherwise, it is set to all zeros.",
                "html": "<p>Vector Absolute Compare Less Than takes the absolute value of each element in a vector, and compares it with the absolute value of the corresponding element of a second vector. If the first is less than the second, the corresponding element in the destination vector is set to all ones. Otherwise, it is set to all zeros.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VADD":
            return {
                "tooltip": "Vector Add (floating-point) adds corresponding elements in two vectors, and places the results in the destination vector.",
                "html": "<p>Vector Add (floating-point) adds corresponding elements in two vectors, and places the results in the destination vector.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, <xref linkend=\"AArch32.hcptr\">HCPTR</xref>, and <xref linkend=\"AArch32.fpexc\">FPEXC</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VADD":
            return {
                "tooltip": "Vector Add (integer) adds corresponding elements in two vectors, and places the results in the destination vector.",
                "html": "<p>Vector Add (integer) adds corresponding elements in two vectors, and places the results in the destination vector.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VADDHN":
            return {
                "tooltip": "Vector Add and Narrow, returning High Half adds corresponding elements in two quadword vectors, and places the most significant half of each result in a doubleword vector. The results are truncated. For rounded results, see VRADDHN.",
                "html": "<p>Vector Add and Narrow, returning High Half adds corresponding elements in two quadword vectors, and places the most significant half of each result in a doubleword vector. The results are truncated. For rounded results, see <xref linkend=\"A32T32-fpsimd.instructions.VRADDHN\">VRADDHN</xref>.</p><p>The operand elements can be 16-bit, 32-bit, or 64-bit integers. There is no distinction between signed and unsigned integers.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VADDL":
            return {
                "tooltip": "Vector Add Long adds corresponding elements in two doubleword vectors, and places the results in a quadword vector. Before adding, it sign-extends or zero-extends the elements of both operands.",
                "html": "<p>Vector Add Long adds corresponding elements in two doubleword vectors, and places the results in a quadword vector. Before adding, it sign-extends or zero-extends the elements of both operands.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VADDW":
            return {
                "tooltip": "Vector Add Wide adds corresponding elements in one quadword and one doubleword vector, and places the results in a quadword vector. Before adding, it sign-extends or zero-extends the elements of the doubleword operand.",
                "html": "<p>Vector Add Wide adds corresponding elements in one quadword and one doubleword vector, and places the results in a quadword vector. Before adding, it sign-extends or zero-extends the elements of the doubleword operand.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VAND":
            return {
                "tooltip": "Vector Bitwise AND (register) performs a bitwise AND operation between two registers, and places the result in the destination register.",
                "html": "<p>Vector Bitwise AND (register) performs a bitwise AND operation between two registers, and places the result in the destination register.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VBIC":
            return {
                "tooltip": "Vector Bitwise Bit Clear (immediate) performs a bitwise AND between a register value and the complement of an immediate value, and returns the result into the destination vector.",
                "html": "<p>Vector Bitwise Bit Clear (immediate) performs a bitwise AND between a register value and the complement of an immediate value, and returns the result into the destination vector.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VBIC":
            return {
                "tooltip": "Vector Bitwise Bit Clear (register) performs a bitwise AND between a register value and the complement of a register value, and places the result in the destination register.",
                "html": "<p>Vector Bitwise Bit Clear (register) performs a bitwise AND between a register value and the complement of a register value, and places the result in the destination register.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VBIF":
            return {
                "tooltip": "Vector Bitwise Insert if False inserts each bit from the first source register into the destination register if the corresponding bit of the second source register is 0, otherwise leaves the bit in the destination register unchanged.",
                "html": "<p>Vector Bitwise Insert if False inserts each bit from the first source register into the destination register if the corresponding bit of the second source register is 0, otherwise leaves the bit in the destination register unchanged.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VBIT":
            return {
                "tooltip": "Vector Bitwise Insert if True inserts each bit from the first source register into the destination register if the corresponding bit of the second source register is 1, otherwise leaves the bit in the destination register unchanged.",
                "html": "<p>Vector Bitwise Insert if True inserts each bit from the first source register into the destination register if the corresponding bit of the second source register is 1, otherwise leaves the bit in the destination register unchanged.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VBSL":
            return {
                "tooltip": "Vector Bitwise Select sets each bit in the destination to the corresponding bit from the first source operand when the original destination bit was 1, otherwise from the second source operand.",
                "html": "<p>Vector Bitwise Select sets each bit in the destination to the corresponding bit from the first source operand when the original destination bit was 1, otherwise from the second source operand.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VCADD":
            return {
                "tooltip": "Vector Complex Add.",
                "html": "<p>Vector Complex Add.</p><p>This instruction operates on complex numbers that are represented in SIMD&amp;FP registers as pairs of elements, with the more significant element holding the imaginary part of the number and the less significant element holding the real part of the number. Each element holds a floating-point value. It performs the following computation on the corresponding complex number element pairs from the two source registers:</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VCEQ":
            return {
                "tooltip": "Vector Compare Equal to Zero takes each element in a vector, and compares it with zero.  If it is equal to zero, the corresponding element in the destination vector is set to all ones. Otherwise, it is set to all zeros.",
                "html": "<p>Vector Compare Equal to Zero takes each element in a vector, and compares it with zero.  If it is equal to zero, the corresponding element in the destination vector is set to all ones. Otherwise, it is set to all zeros.</p><p>The operand vector elements are the same type, and are integers or floating-point numbers. The result vector elements are fields the same size as the operand vector elements.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VCEQ":
            return {
                "tooltip": "Vector Compare Equal takes each element in a vector, and compares it with the corresponding element of a second vector. If they are equal, the corresponding element in the destination vector is set to all ones. Otherwise, it is set to all zeros.",
                "html": "<p>Vector Compare Equal takes each element in a vector, and compares it with the corresponding element of a second vector. If they are equal, the corresponding element in the destination vector is set to all ones. Otherwise, it is set to all zeros.</p><p>The operand vector elements are the same type, and are integers or floating-point numbers. The result vector elements are fields the same size as the operand vector elements.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VCGE":
            return {
                "tooltip": "Vector Compare Greater Than or Equal to Zero takes each element in a vector, and compares it with zero. If it is greater than or equal to zero, the corresponding element in the destination vector is set to all ones. Otherwise, it is set to all zeros.",
                "html": "<p>Vector Compare Greater Than or Equal to Zero takes each element in a vector, and compares it with zero. If it is greater than or equal to zero, the corresponding element in the destination vector is set to all ones. Otherwise, it is set to all zeros.</p><p>The operand vector elements are the same type, and are signed integers or floating-point numbers. The result vector elements are fields the same size as the operand vector elements.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VCGE":
            return {
                "tooltip": "Vector Compare Greater Than or Equal takes each element in a vector, and compares it with the corresponding element of a second vector. If the first is greater than or equal to the second, the corresponding element in the destination vector is set to all ones. Otherwise, it is set to all zeros.",
                "html": "<p>Vector Compare Greater Than or Equal takes each element in a vector, and compares it with the corresponding element of a second vector. If the first is greater than or equal to the second, the corresponding element in the destination vector is set to all ones. Otherwise, it is set to all zeros.</p><p>The operand vector elements are the same type, and are signed integers, unsigned integers, or floating-point numbers. The result vector elements are fields the same size as the operand vector elements.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VCGT":
            return {
                "tooltip": "Vector Compare Greater Than Zero takes each element in a vector, and compares it with zero.  If it is greater than zero, the corresponding element in the destination vector is set to all ones. Otherwise, it is set to all zeros.",
                "html": "<p>Vector Compare Greater Than Zero takes each element in a vector, and compares it with zero.  If it is greater than zero, the corresponding element in the destination vector is set to all ones. Otherwise, it is set to all zeros.</p><p>The operand vector elements are the same type, and are signed integers or floating-point numbers. The result vector elements are fields the same size as the operand vector elements.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VCGT":
            return {
                "tooltip": "Vector Compare Greater Than takes each element in a vector, and compares it with the corresponding element of a second vector. If the first is greater than the second, the corresponding element in the destination vector is set to all ones. Otherwise, it is set to all zeros.",
                "html": "<p>Vector Compare Greater Than takes each element in a vector, and compares it with the corresponding element of a second vector. If the first is greater than the second, the corresponding element in the destination vector is set to all ones. Otherwise, it is set to all zeros.</p><p>The operand vector elements are the same type, and are signed integers, unsigned integers, or floating-point numbers. The result vector elements are fields the same size as the operand vector elements.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VCLE":
            return {
                "tooltip": "Vector Compare Less Than or Equal to Zero takes each element in a vector, and compares it with zero. If it is less than or equal to zero, the corresponding element in the destination vector is set to all ones. Otherwise, it is set to all zeros.",
                "html": "<p>Vector Compare Less Than or Equal to Zero takes each element in a vector, and compares it with zero. If it is less than or equal to zero, the corresponding element in the destination vector is set to all ones. Otherwise, it is set to all zeros.</p><p>The operand vector elements are the same type, and are signed integers or floating-point numbers. The result vector elements are fields the same size as the operand vector elements.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VCLE":
            return {
                "tooltip": "Vector Compare Less Than or Equal takes each element in a vector, and compares it with the corresponding element of a second vector. If the first is less than or equal to the second, the corresponding element in the destination vector is set to all ones. Otherwise, it is set to all zeros.",
                "html": "<p>Vector Compare Less Than or Equal takes each element in a vector, and compares it with the corresponding element of a second vector. If the first is less than or equal to the second, the corresponding element in the destination vector is set to all ones. Otherwise, it is set to all zeros.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VCLS":
            return {
                "tooltip": "Vector Count Leading Sign Bits counts the number of consecutive bits following the topmost bit, that are the same as the topmost bit, in each element in a vector, and places the results in a second vector. The count does not include the topmost bit itself.",
                "html": "<p>Vector Count Leading Sign Bits counts the number of consecutive bits following the topmost bit, that are the same as the topmost bit, in each element in a vector, and places the results in a second vector. The count does not include the topmost bit itself.</p><p>The operand vector elements can be any one of 8-bit, 16-bit, or 32-bit signed integers.</p><p>The result vector elements are the same data type as the operand vector elements.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VCLT":
            return {
                "tooltip": "Vector Compare Less Than Zero takes each element in a vector, and compares it with zero.  If it is less than zero, the corresponding element in the destination vector is set to all ones. Otherwise, it is set to all zeros.",
                "html": "<p>Vector Compare Less Than Zero takes each element in a vector, and compares it with zero.  If it is less than zero, the corresponding element in the destination vector is set to all ones. Otherwise, it is set to all zeros.</p><p>The operand vector elements are the same type, and are signed integers or floating-point numbers. The result vector elements are fields the same size as the operand vector elements.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VCLT":
            return {
                "tooltip": "Vector Compare Less Than takes each element in a vector, and compares it with the corresponding element of a second vector. If the first is less than the second, the corresponding element in the destination vector is set to all ones. Otherwise, it is set to all zeros.",
                "html": "<p>Vector Compare Less Than takes each element in a vector, and compares it with the corresponding element of a second vector. If the first is less than the second, the corresponding element in the destination vector is set to all ones. Otherwise, it is set to all zeros.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VCLZ":
            return {
                "tooltip": "Vector Count Leading Zeros counts the number of consecutive zeros, starting from the most significant bit, in each element in a vector, and places the results in a second vector.",
                "html": "<p>Vector Count Leading Zeros counts the number of consecutive zeros, starting from the most significant bit, in each element in a vector, and places the results in a second vector.</p><p>The operand vector elements can be any one of 8-bit, 16-bit, or 32-bit integers. There is no distinction between signed and unsigned integers.</p><p>The result vector elements are the same data type as the operand vector elements.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VCMLA":
            return {
                "tooltip": "Vector Complex Multiply Accumulate.",
                "html": "<p>Vector Complex Multiply Accumulate.</p><p>This instruction operates on complex numbers that are represented in SIMD&amp;FP registers as pairs of elements, with the more significant element holding the imaginary part of the number and the less significant element holding the real part of the number. Each element holds a floating-point value. It performs the following computation on the corresponding complex number element pairs from the two source registers and the destination register:</p><p>The multiplication and addition operations are performed as a fused multiply-add, without any intermediate rounding.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VCMLA":
            return {
                "tooltip": "Vector Complex Multiply Accumulate (by element).",
                "html": "<p>Vector Complex Multiply Accumulate (by element).</p><p>This instruction operates on complex numbers that are represented in SIMD&amp;FP registers as pairs of elements, with the more significant element holding the imaginary part of the number and the less significant element holding the real part of the number. Each element holds a floating-point value. It performs the following computation on complex numbers from the first source register and the destination register with the specified complex number from the second source register:</p><p>The multiplication and addition operations are performed as a fused multiply-add, without any intermediate rounding.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VCMP":
            return {
                "tooltip": "Vector Compare compares two floating-point registers, or one floating-point register and zero. It writes the result to the FPSCR flags. These are normally transferred to the PSTATE.{N, Z, C, V} Condition flags by a subsequent VMRS instruction.",
                "html": "<p>Vector Compare compares two floating-point registers, or one floating-point register and zero. It writes the result to the <xref linkend=\"AArch32.fpscr\">FPSCR</xref> flags. These are normally transferred to the <xref linkend=\"BEIDIGBH\">PSTATE</xref>.{N, Z, C, V} Condition flags by a subsequent <instruction>VMRS</instruction> instruction.</p><p>This instruction raises an Invalid Operation floating-point exception if either or both of the operands is a signaling NaN.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VCMPE":
            return {
                "tooltip": "Vector Compare, raising Invalid Operation on NaN compares two floating-point registers, or one floating-point register and zero. It writes the result to the FPSCR flags. These are normally transferred to the PSTATE.{N, Z, C, V} Condition flags by a subsequent VMRS instruction.",
                "html": "<p>Vector Compare, raising Invalid Operation on NaN compares two floating-point registers, or one floating-point register and zero. It writes the result to the <xref linkend=\"AArch32.fpscr\">FPSCR</xref> flags. These are normally transferred to the <xref linkend=\"BEIDIGBH\">PSTATE</xref>.{N, Z, C, V} Condition flags by a subsequent <instruction>VMRS</instruction> instruction.</p><p>This instruction raises an Invalid Operation floating-point exception if either or both of the operands is any type of NaN.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VCNT":
            return {
                "tooltip": "Vector Count Set Bits counts the number of bits that are one in each element in a vector, and places the results in a second vector.",
                "html": "<p>Vector Count Set Bits counts the number of bits that are one in each element in a vector, and places the results in a second vector.</p><p>The operand vector elements must be 8-bit fields.</p><p>The result vector elements are 8-bit integers.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VCVT":
            return {
                "tooltip": "Vector Convert from single-precision to BFloat16 converts each 32-bit element in a vector from single-precision floating-point to BFloat16 format, and writes the result into a second vector. The result vector elements are half the width of the source vector elements.",
                "html": "<p>Vector Convert from single-precision to BFloat16 converts each 32-bit element in a vector from single-precision floating-point to BFloat16 format, and writes the result into a second vector. The result vector elements are half the width of the source vector elements.</p><p>Unlike the BFloat16 multiplication instructions, this instruction uses the Round to Nearest rounding mode, and can generate a floating-point exception that causes cumulative exception bits in the <xref linkend=\"AArch32.fpscr\">FPSCR</xref> to be set.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VCVT":
            return {
                "tooltip": "Convert between double-precision and single-precision does one of the following",
                "html": "<p>Convert between double-precision and single-precision does one of the following:</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, <xref linkend=\"AArch32.hcptr\">HCPTR</xref>, and <xref linkend=\"AArch32.fpexc\">FPEXC</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VCVT":
            return {
                "tooltip": "Vector Convert between half-precision and single-precision converts each element in a vector from single-precision to half-precision floating-point, or from half-precision to single-precision, and places the results in a second vector.",
                "html": "<p>Vector Convert between half-precision and single-precision converts each element in a vector from single-precision to half-precision floating-point, or from half-precision to single-precision, and places the results in a second vector.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VCVT":
            return {
                "tooltip": "Vector Convert between floating-point and integer converts each element in a vector from floating-point to integer, or from integer to floating-point, and places the results in a second vector.",
                "html": "<p>Vector Convert between floating-point and integer converts each element in a vector from floating-point to integer, or from integer to floating-point, and places the results in a second vector.</p><p>The vector elements are the same type, and are floating-point numbers or integers. Signed and unsigned integers are distinct.</p><p>The floating-point to integer operation uses the Round towards Zero rounding mode. The integer to floating-point operation uses the Round to Nearest rounding mode.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VCVT":
            return {
                "tooltip": "Convert floating-point to integer with Round towards Zero converts a value in a register from floating-point to a 32-bit integer, using the Round towards Zero rounding mode, and places the result in a second register.",
                "html": "<p>Convert floating-point to integer with Round towards Zero converts a value in a register from floating-point to a 32-bit integer, using the Round towards Zero rounding mode, and places the result in a second register.</p><p><xref linkend=\"A32T32-fpsimd.instructions.VCVT_xv\">VCVT (between floating-point and fixed-point, floating-point)</xref> describes conversions between floating-point and 16-bit integers.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, <xref linkend=\"AArch32.hcptr\">HCPTR</xref>, and <xref linkend=\"AArch32.fpexc\">FPEXC</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VCVT":
            return {
                "tooltip": "Convert integer to floating-point converts a 32-bit integer to floating-point using the rounding mode specified by the FPSCR, and places the result in a second register.",
                "html": "<p>Convert integer to floating-point converts a 32-bit integer to floating-point using the rounding mode specified by the <xref linkend=\"AArch32.fpscr\">FPSCR</xref>, and places the result in a second register.</p><p><xref linkend=\"A32T32-fpsimd.instructions.VCVT_xv\">VCVT (between floating-point and fixed-point, floating-point)</xref> describes conversions between floating-point and 16-bit integers.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, <xref linkend=\"AArch32.hcptr\">HCPTR</xref>, and <xref linkend=\"AArch32.fpexc\">FPEXC</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VCVT":
            return {
                "tooltip": "Vector Convert between floating-point and fixed-point converts each element in a vector from floating-point to fixed-point, or from fixed-point to floating-point, and places the results in a second vector.",
                "html": "<p>Vector Convert between floating-point and fixed-point converts each element in a vector from floating-point to fixed-point, or from fixed-point to floating-point, and places the results in a second vector.</p><p>The vector elements are the same type, and are floating-point numbers or integers. Signed and unsigned integers are distinct.</p><p>The floating-point to fixed-point operation uses the Round towards Zero rounding mode. The fixed-point to floating-point operation uses the Round to Nearest rounding mode.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VCVT":
            return {
                "tooltip": "Convert between floating-point and fixed-point converts a value in a register from floating-point to fixed-point, or from fixed-point to floating-point. Software can specify the fixed-point value as either signed or unsigned.",
                "html": "<p>Convert between floating-point and fixed-point converts a value in a register from floating-point to fixed-point, or from fixed-point to floating-point. Software can specify the fixed-point value as either signed or unsigned.</p><p>The fixed-point value can be 16-bit or 32-bit. Conversions from fixed-point values take their operand from the low-order bits of the source register and ignore any remaining bits. Signed conversions to fixed-point values sign-extend the result value to the destination register width. Unsigned conversions to fixed-point values zero-extend the result value to the destination register width.</p><p>The floating-point to fixed-point operation uses the Round towards Zero rounding mode. The fixed-point to floating-point operation uses the Round to Nearest rounding mode.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, <xref linkend=\"AArch32.hcptr\">HCPTR</xref>, and <xref linkend=\"AArch32.fpexc\">FPEXC</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VCVTA":
            return {
                "tooltip": "Vector Convert floating-point to integer with Round to Nearest with Ties to Away converts each element in a vector from floating-point to integer using the Round to Nearest with Ties to Away rounding mode, and places the results in a second vector.",
                "html": "<p>Vector Convert floating-point to integer with Round to Nearest with Ties to Away converts each element in a vector from floating-point to integer using the Round to Nearest with Ties to Away rounding mode, and places the results in a second vector.</p><p>The operand vector elements are floating-point numbers.</p><p>The result vector elements are integers, and the same size as the operand vector elements. Signed and unsigned integers are distinct.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, <xref linkend=\"AArch32.hcptr\">HCPTR</xref>, and <xref linkend=\"AArch32.fpexc\">FPEXC</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VCVTA":
            return {
                "tooltip": "Convert floating-point to integer with Round to Nearest with Ties to Away converts a value in a register from floating-point to a 32-bit integer using the Round to Nearest with Ties to Away rounding mode, and places the result in a second register.",
                "html": "<p>Convert floating-point to integer with Round to Nearest with Ties to Away converts a value in a register from floating-point to a 32-bit integer using the Round to Nearest with Ties to Away rounding mode, and places the result in a second register.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, <xref linkend=\"AArch32.hcptr\">HCPTR</xref>, and <xref linkend=\"AArch32.fpexc\">FPEXC</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VCVTB":
            return {
                "tooltip": "Convert to or from a half-precision value in the bottom half of a single-precision register does one of the following",
                "html": "<p>Convert to or from a half-precision value in the bottom half of a single-precision register does one of the following:</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, <xref linkend=\"AArch32.hcptr\">HCPTR</xref>, and <xref linkend=\"AArch32.fpexc\">FPEXC</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VCVTB":
            return {
                "tooltip": "Converts the single-precision value in a single-precision register to BFloat16 format and writes the result into the bottom half of a single precision register, preserving the top 16 bits of the destination register.",
                "html": "<p>Converts the single-precision value in a single-precision register to BFloat16 format and writes the result into the bottom half of a single precision register, preserving the top 16 bits of the destination register.</p><p>Unlike the BFloat16 multiplication instructions, this instruction honors all the control bits in the <xref linkend=\"AArch32.fpscr\">FPSCR</xref> that apply to single-precision arithmetic, including the rounding mode. This instruction can generate a floating-point exception which causes a cumulative exception bit in the <xref linkend=\"AArch32.fpscr\">FPSCR</xref> to be set, or a synchronous exception to be taken, depending on the enable bits in the <xref linkend=\"AArch32.fpscr\">FPSCR</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VCVTM":
            return {
                "tooltip": "Vector Convert floating-point to integer with Round towards -Infinity converts each element in a vector from floating-point to integer using the Round towards -Infinity rounding mode, and places the results in a second vector.",
                "html": "<p>Vector Convert floating-point to integer with Round towards -Infinity converts each element in a vector from floating-point to integer using the Round towards -Infinity rounding mode, and places the results in a second vector.</p><p>The operand vector elements are floating-point numbers.</p><p>The result vector elements are integers, and the same size as the operand vector elements. Signed and unsigned integers are distinct.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, <xref linkend=\"AArch32.hcptr\">HCPTR</xref>, and <xref linkend=\"AArch32.fpexc\">FPEXC</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VCVTM":
            return {
                "tooltip": "Convert floating-point to integer with Round towards -Infinity converts a value in a register from floating-point to a 32-bit integer using the Round towards -Infinity rounding mode, and places the result in a second register.",
                "html": "<p>Convert floating-point to integer with Round towards -Infinity converts a value in a register from floating-point to a 32-bit integer using the Round towards -Infinity rounding mode, and places the result in a second register.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, <xref linkend=\"AArch32.hcptr\">HCPTR</xref>, and <xref linkend=\"AArch32.fpexc\">FPEXC</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VCVTN":
            return {
                "tooltip": "Vector Convert floating-point to integer with Round to Nearest converts each element in a vector from floating-point to integer using the Round to Nearest rounding mode, and places the results in a second vector.",
                "html": "<p>Vector Convert floating-point to integer with Round to Nearest converts each element in a vector from floating-point to integer using the Round to Nearest rounding mode, and places the results in a second vector.</p><p>The operand vector elements are floating-point numbers.</p><p>The result vector elements are integers, and the same size as the operand vector elements. Signed and unsigned integers are distinct.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, <xref linkend=\"AArch32.hcptr\">HCPTR</xref>, and <xref linkend=\"AArch32.fpexc\">FPEXC</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VCVTN":
            return {
                "tooltip": "Convert floating-point to integer with Round to Nearest converts a value in a register from floating-point to a 32-bit integer using the Round to Nearest rounding mode, and places the result in a second register.",
                "html": "<p>Convert floating-point to integer with Round to Nearest converts a value in a register from floating-point to a 32-bit integer using the Round to Nearest rounding mode, and places the result in a second register.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, <xref linkend=\"AArch32.hcptr\">HCPTR</xref>, and <xref linkend=\"AArch32.fpexc\">FPEXC</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VCVTP":
            return {
                "tooltip": "Vector Convert floating-point to integer with Round towards +Infinity converts each element in a vector from floating-point to integer using the Round towards +Infinity rounding mode, and places the results in a second vector.",
                "html": "<p>Vector Convert floating-point to integer with Round towards +Infinity converts each element in a vector from floating-point to integer using the Round towards +Infinity rounding mode, and places the results in a second vector.</p><p>The operand vector elements are floating-point numbers.</p><p>The result vector elements are integers, and the same size as the operand vector elements. Signed and unsigned integers are distinct.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, <xref linkend=\"AArch32.hcptr\">HCPTR</xref>, and <xref linkend=\"AArch32.fpexc\">FPEXC</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VCVTP":
            return {
                "tooltip": "Convert floating-point to integer with Round towards +Infinity converts a value in a register from floating-point to a 32-bit integer using the Round towards +Infinity rounding mode, and places the result in a second register.",
                "html": "<p>Convert floating-point to integer with Round towards +Infinity converts a value in a register from floating-point to a 32-bit integer using the Round towards +Infinity rounding mode, and places the result in a second register.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, <xref linkend=\"AArch32.hcptr\">HCPTR</xref>, and <xref linkend=\"AArch32.fpexc\">FPEXC</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VCVTR":
            return {
                "tooltip": "Convert floating-point to integer converts a value in a register from floating-point to a 32-bit integer, using the rounding mode specified by the FPSCR and places the result in a second register.",
                "html": "<p>Convert floating-point to integer converts a value in a register from floating-point to a 32-bit integer, using the rounding mode specified by the <xref linkend=\"AArch32.fpscr\">FPSCR</xref> and places the result in a second register.</p><p><xref linkend=\"A32T32-fpsimd.instructions.VCVT_xv\">VCVT (between floating-point and fixed-point, floating-point)</xref> describes conversions between floating-point and 16-bit integers.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, <xref linkend=\"AArch32.hcptr\">HCPTR</xref>, and <xref linkend=\"AArch32.fpexc\">FPEXC</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VCVTT":
            return {
                "tooltip": "Convert to or from a half-precision value in the top half of a single-precision register does one of the following",
                "html": "<p>Convert to or from a half-precision value in the top half of a single-precision register does one of the following:</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, <xref linkend=\"AArch32.hcptr\">HCPTR</xref>, and <xref linkend=\"AArch32.fpexc\">FPEXC</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VCVTT":
            return {
                "tooltip": "Converts the single-precision value in a single-precision register to BFloat16 format and writes the result in the top half of a single-precision register, preserving the bottom 16 bits of the register.",
                "html": "<p>Converts the single-precision value in a single-precision register to BFloat16 format and writes the result in the top half of a single-precision register, preserving the bottom 16 bits of the register.</p><p>Unlike the BFloat16 multiplication instructions, this instruction honors all the control bits in the <xref linkend=\"AArch32.fpscr\">FPSCR</xref> that apply to single-precision arithmetic, including the rounding mode. This instruction can generate a floating-point exception which causes a cumulative exception bit in the <xref linkend=\"AArch32.fpscr\">FPSCR</xref> to be set, or a synchronous exception to be taken, depending on the enable bits in the <xref linkend=\"AArch32.fpscr\">FPSCR</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VDIV":
            return {
                "tooltip": "Divide divides one floating-point value by another floating-point value and writes the result to a third floating-point register.",
                "html": "<p>Divide divides one floating-point value by another floating-point value and writes the result to a third floating-point register.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, <xref linkend=\"AArch32.hcptr\">HCPTR</xref>, and <xref linkend=\"AArch32.fpexc\">FPEXC</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VDOT":
            return {
                "tooltip": "BFloat16 floating-point (BF16) dot product (vector). This instruction delimits the source vectors into pairs of 16-bit BF16 elements. Within each pair, the elements in the first source vector are multiplied by the corresponding elements in the second source vector. The resulting single-precision products are then summed and added destructively to the single-precision element in the destination vector which aligns with the pair of BF16 values in the first source vector. The instruction does not update the FPSCR exception status.",
                "html": "<p>BFloat16 floating-point (BF16) dot product (vector). This instruction delimits the source vectors into pairs of 16-bit BF16 elements. Within each pair, the elements in the first source vector are multiplied by the corresponding elements in the second source vector. The resulting single-precision products are then summed and added destructively to the single-precision element in the destination vector which aligns with the pair of BF16 values in the first source vector. The instruction does not update the <xref linkend=\"AArch32.fpscr\">FPSCR</xref> exception status.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VDOT":
            return {
                "tooltip": "BFloat16 floating-point indexed dot product (vector, by element). This instruction delimits the source vectors into pairs of 16-bit BF16 elements. Each pair of elements in the first source vector is multiplied by the indexed pair of elements in the second source vector. The resulting single-precision products are then summed and added destructively to the single-precision element in the destination vector which aligns with the pair of BFloat16 values in the first source vector. The instruction does not update the FPSCR exception status.",
                "html": "<p>BFloat16 floating-point indexed dot product (vector, by element). This instruction delimits the source vectors into pairs of 16-bit BF16 elements. Each pair of elements in the first source vector is multiplied by the indexed pair of elements in the second source vector. The resulting single-precision products are then summed and added destructively to the single-precision element in the destination vector which aligns with the pair of BFloat16 values in the first source vector. The instruction does not update the <xref linkend=\"AArch32.fpscr\">FPSCR</xref> exception status.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VDUP":
            return {
                "tooltip": "Duplicate general-purpose register to vector duplicates an element from a general-purpose register into every element of the destination vector.",
                "html": "<p>Duplicate general-purpose register to vector duplicates an element from a general-purpose register into every element of the destination vector.</p><p>The destination vector elements can be 8-bit, 16-bit, or 32-bit fields. The source element is the least significant 8, 16, or 32 bits of the general-purpose register. There is no distinction between data types.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VDUP":
            return {
                "tooltip": "Duplicate vector element to vector duplicates a single element of a vector into every element of the destination vector.",
                "html": "<p>Duplicate vector element to vector duplicates a single element of a vector into every element of the destination vector.</p><p>The scalar, and the destination vector elements, can be any one of 8-bit, 16-bit, or 32-bit fields. There is no distinction between data types.</p><p>For more information about scalars see <xref linkend=\"Cjaibjhd\">Advanced SIMD scalars</xref>.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VEOR":
            return {
                "tooltip": "Vector Bitwise Exclusive-OR performs a bitwise exclusive-OR operation between two registers, and places the result in the destination register. The operand and result registers can be quadword or doubleword. They must all be the same size.",
                "html": "<p>Vector Bitwise Exclusive-OR performs a bitwise exclusive-OR operation between two registers, and places the result in the destination register. The operand and result registers can be quadword or doubleword. They must all be the same size.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VEXT":
            return {
                "tooltip": "Vector Extract extracts elements from the bottom end of the second operand vector and the top end of the first, concatenates them and places the result in the destination vector.",
                "html": "<p>Vector Extract extracts elements from the bottom end of the second operand vector and the top end of the first, concatenates them and places the result in the destination vector.</p><p>The elements of the vectors are treated as being 8-bit fields. There is no distinction between data types.</p><p></p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VFMA":
            return {
                "tooltip": "Vector Fused Multiply Accumulate multiplies corresponding elements of two vectors, and accumulates the results into the elements of the destination vector. The instruction does not round the result of the multiply before the accumulation.",
                "html": "<p>Vector Fused Multiply Accumulate multiplies corresponding elements of two vectors, and accumulates the results into the elements of the destination vector. The instruction does not round the result of the multiply before the accumulation.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, <xref linkend=\"AArch32.hcptr\">HCPTR</xref>, and <xref linkend=\"AArch32.fpexc\">FPEXC</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VFMAB":
        case "VFMAT":
            return {
                "tooltip": "The Bfloat16 floating-point widening multiply-add long instruction widens the even-numbered (bottom) or odd-numbered (top) 16-bit elements in the first and second source vectors from Bfloat16 to single-precision format. The instruction then multiplies and adds these values to the overlapping single-precision elements of the destination vector.",
                "html": "<p>The Bfloat16 floating-point widening multiply-add long instruction widens the even-numbered (bottom) or odd-numbered (top) 16-bit elements in the first and second source vectors from Bfloat16 to single-precision format. The instruction then multiplies and adds these values to the overlapping single-precision elements of the destination vector.</p><p>Unlike other BFloat16 multiplication instructions, this performs a fused multiply-add, without intermediate rounding that uses the Round to Nearest rounding mode and can generate a floating-point exception that causes cumulative exception bits in the <xref linkend=\"AArch32.fpscr\">FPSCR</xref> to be set.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VFMAB":
        case "VFMAT":
            return {
                "tooltip": "The BFloat16 floating-point widening multiply-add long instruction widens the even-numbered (bottom) or odd-numbered (top) 16-bit elements in the first source vector, and an indexed element in the second source vector from Bfloat16 to single-precision format. The instruction then multiplies and adds these values to the overlapping single-precision elements of the destination vector.",
                "html": "<p>The BFloat16 floating-point widening multiply-add long instruction widens the even-numbered (bottom) or odd-numbered (top) 16-bit elements in the first source vector, and an indexed element in the second source vector from Bfloat16 to single-precision format. The instruction then multiplies and adds these values to the overlapping single-precision elements of the destination vector.</p><p>Unlike other BFloat16 multiplication instructions, this performs a fused multiply-add, without intermediate rounding that uses the Round to Nearest rounding mode and can generate a floating-point exception that causes cumulative exception bits in the <xref linkend=\"AArch32.fpscr\">FPSCR</xref> to be set.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VFMAL":
            return {
                "tooltip": "Vector Floating-point Multiply-Add Long to accumulator (vector). This instruction multiplies corresponding values in the vectors in the two source SIMD&FP registers, and accumulates the product to the corresponding vector element of the destination SIMD&FP register. The instruction does not round the result of the multiply before the accumulation.",
                "html": "<p>Vector Floating-point Multiply-Add Long to accumulator (vector). This instruction multiplies corresponding values in the vectors in the two source SIMD&amp;FP registers, and accumulates the product to the corresponding vector element of the destination SIMD&amp;FP register. The instruction does not round the result of the multiply before the accumulation.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, <xref linkend=\"AArch32.hcptr\">HCPTR</xref>, and <xref linkend=\"AArch32.fpexc\">FPEXC</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p><p>In Armv8.2 and Armv8.3, this is an <arm-defined-word>optional</arm-defined-word> instruction. From Armv8.4 it is mandatory for all implementations to support it.</p><p><xref linkend=\"AArch32.id_isar6\">ID_ISAR6</xref>.FHM indicates whether this instruction is supported.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VFMAL":
            return {
                "tooltip": "Vector Floating-point Multiply-Add Long to accumulator (by scalar). This instruction multiplies the vector elements in the first source SIMD&FP register by the specified value in the second source SIMD&FP register, and accumulates the product to the corresponding vector element of the destination SIMD&FP register. The instruction does not round the result of the multiply before the accumulation.",
                "html": "<p>Vector Floating-point Multiply-Add Long to accumulator (by scalar). This instruction multiplies the vector elements in the first source SIMD&amp;FP register by the specified value in the second source SIMD&amp;FP register, and accumulates the product to the corresponding vector element of the destination SIMD&amp;FP register. The instruction does not round the result of the multiply before the accumulation.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, <xref linkend=\"AArch32.hcptr\">HCPTR</xref>, and <xref linkend=\"AArch32.fpexc\">FPEXC</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p><p>In Armv8.2 and Armv8.3, this is an <arm-defined-word>optional</arm-defined-word> instruction. From Armv8.4 it is mandatory for all implementations to support it.</p><p><xref linkend=\"AArch32.id_isar6\">ID_ISAR6</xref>.FHM indicates whether this instruction is supported.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VFMS":
            return {
                "tooltip": "Vector Fused Multiply Subtract negates the elements of one vector and multiplies them with the corresponding elements of another vector, adds the products to the corresponding elements of the destination vector, and places the results in the destination vector. The instruction does not round the result of the multiply before the addition.",
                "html": "<p>Vector Fused Multiply Subtract negates the elements of one vector and multiplies them with the corresponding elements of another vector, adds the products to the corresponding elements of the destination vector, and places the results in the destination vector. The instruction does not round the result of the multiply before the addition.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, <xref linkend=\"AArch32.hcptr\">HCPTR</xref>, and <xref linkend=\"AArch32.fpexc\">FPEXC</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VFMSL":
            return {
                "tooltip": "Vector Floating-point Multiply-Subtract Long from accumulator (vector). This instruction negates the values in the vector of one SIMD&FP register, multiplies these with the corresponding values in another vector, and accumulates the product to the corresponding vector element of the destination SIMD&FP register. The instruction does not round the result of the multiply before the accumulation.",
                "html": "<p>Vector Floating-point Multiply-Subtract Long from accumulator (vector). This instruction negates the values in the vector of one SIMD&amp;FP register, multiplies these with the corresponding values in another vector, and accumulates the product to the corresponding vector element of the destination SIMD&amp;FP register. The instruction does not round the result of the multiply before the accumulation.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, <xref linkend=\"AArch32.hcptr\">HCPTR</xref>, and <xref linkend=\"AArch32.fpexc\">FPEXC</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p><p>In Armv8.2 and Armv8.3, this is an <arm-defined-word>optional</arm-defined-word> instruction. From Armv8.4 it is mandatory for all implementations to support it.</p><p><xref linkend=\"AArch32.id_isar6\">ID_ISAR6</xref>.FHM indicates whether this instruction is supported.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VFMSL":
            return {
                "tooltip": "Vector Floating-point Multiply-Subtract Long from accumulator (by scalar). This instruction multiplies the negated vector elements in the first source SIMD&FP register by the specified value in the second source SIMD&FP register, and accumulates the product to the corresponding vector element of the destination SIMD&FP register. The instruction does not round the result of the multiply before the accumulation.",
                "html": "<p>Vector Floating-point Multiply-Subtract Long from accumulator (by scalar). This instruction multiplies the negated vector elements in the first source SIMD&amp;FP register by the specified value in the second source SIMD&amp;FP register, and accumulates the product to the corresponding vector element of the destination SIMD&amp;FP register. The instruction does not round the result of the multiply before the accumulation.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, <xref linkend=\"AArch32.hcptr\">HCPTR</xref>, and <xref linkend=\"AArch32.fpexc\">FPEXC</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p><p>In Armv8.2 and Armv8.3, this is an <arm-defined-word>optional</arm-defined-word> instruction. From Armv8.4 it is mandatory for all implementations to support it.</p><p><xref linkend=\"AArch32.id_isar6\">ID_ISAR6</xref>.FHM indicates whether this instruction is supported.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VFNMA":
            return {
                "tooltip": "Vector Fused Negate Multiply Accumulate negates one floating-point register value and multiplies it by another floating-point register value, adds the negation of the floating-point value in the destination register to the product, and writes the result back to the destination register. The instruction does not round the result of the multiply before the addition.",
                "html": "<p>Vector Fused Negate Multiply Accumulate negates one floating-point register value and multiplies it by another floating-point register value, adds the negation of the floating-point value in the destination register to the product, and writes the result back to the destination register. The instruction does not round the result of the multiply before the addition.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, <xref linkend=\"AArch32.hcptr\">HCPTR</xref>, and <xref linkend=\"AArch32.fpexc\">FPEXC</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VFNMS":
            return {
                "tooltip": "Vector Fused Negate Multiply Subtract multiplies together two floating-point register values, adds the negation of the floating-point value in the destination register to the product, and writes the result back to the destination register. The instruction does not round the result of the multiply before the addition.",
                "html": "<p>Vector Fused Negate Multiply Subtract multiplies together two floating-point register values, adds the negation of the floating-point value in the destination register to the product, and writes the result back to the destination register. The instruction does not round the result of the multiply before the addition.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, <xref linkend=\"AArch32.hcptr\">HCPTR</xref>, and <xref linkend=\"AArch32.fpexc\">FPEXC</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VHADD":
            return {
                "tooltip": "Vector Halving Add adds corresponding elements in two vectors of integers, shifts each result right one bit, and places the final results in the destination vector. The results of the halving operations are truncated. For rounded results, see VRHADD).",
                "html": "<p>Vector Halving Add adds corresponding elements in two vectors of integers, shifts each result right one bit, and places the final results in the destination vector. The results of the halving operations are truncated. For rounded results, see <xref linkend=\"A32T32-fpsimd.instructions.VRHADD\">VRHADD</xref>).</p><p>The operand and result elements are all the same type, and can be any one of:</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VHSUB":
            return {
                "tooltip": "Vector Halving Subtract subtracts the elements of the second operand from the corresponding elements of the first operand, shifts each result right one bit, and places the final results in the destination vector. The results of the halving operations are truncated. There is no rounding version.",
                "html": "<p>Vector Halving Subtract subtracts the elements of the second operand from the corresponding elements of the first operand, shifts each result right one bit, and places the final results in the destination vector. The results of the halving operations are truncated. There is no rounding version.</p><p>The operand and result elements are all the same type, and can be any one of:</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VINS":
            return {
                "tooltip": "Vector move Insertion. This instruction copies the lower 16 bits of the 32-bit source SIMD&FP register into the upper 16 bits of the 32-bit destination SIMD&FP register, while preserving the values in the remaining bits.",
                "html": "<p>Vector move Insertion. This instruction copies the lower 16 bits of the 32-bit source SIMD&amp;FP register into the upper 16 bits of the 32-bit destination SIMD&amp;FP register, while preserving the values in the remaining bits.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, <xref linkend=\"AArch32.hcptr\">HCPTR</xref>, and <xref linkend=\"AArch32.fpexc\">FPEXC</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VJCVT":
            return {
                "tooltip": "Javascript Convert to signed fixed-point, rounding toward Zero. This instruction converts the double-precision floating-point value in the SIMD&FP source register to a 32-bit signed integer using the Round towards Zero rounding mode, and writes the result to the SIMD&FP destination register. If the result is too large to be accommodated as a signed 32-bit integer, then the result is the integer modulo 232, as held in a 32-bit signed integer.",
                "html": "<p>Javascript Convert to signed fixed-point, rounding toward Zero. This instruction converts the double-precision floating-point value in the SIMD&amp;FP source register to a 32-bit signed integer using the Round towards Zero rounding mode, and writes the result to the SIMD&amp;FP destination register. If the result is too large to be accommodated as a signed 32-bit integer, then the result is the integer modulo 2<sup>32</sup>, as held in a 32-bit signed integer.</p><p>This instruction can generate a floating-point exception. Depending on the settings in <xref linkend=\"AArch32.fpscr\">FPSCR</xref>, the exception results in either a flag being set or a synchronous exception being generated. For more information, see <xref linkend=\"CFIHBHHD\">Floating-point exceptions and exception traps</xref>.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, <xref linkend=\"AArch32.hcptr\">HCPTR</xref>, and <xref linkend=\"AArch32.fpexc\">FPEXC</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VLD1":
            return {
                "tooltip": "Load single 1-element structure to one lane of one register loads one element from memory into one element of a register. Elements of the register that are not loaded are unchanged. For details of the addressing mode, see Advanced SIMD addressing mode.",
                "html": "<p>Load single 1-element structure to one lane of one register loads one element from memory into one element of a register. Elements of the register that are not loaded are unchanged. For details of the addressing mode, see <xref linkend=\"Cjaefebe\">Advanced SIMD addressing mode</xref>.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information, see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VLD1":
            return {
                "tooltip": "Load single 1-element structure and replicate to all lanes of one register loads one element from memory into every element of one or two vectors. For details of the addressing mode, see Advanced SIMD addressing mode.",
                "html": "<p>Load single 1-element structure and replicate to all lanes of one register loads one element from memory into every element of one or two vectors. For details of the addressing mode, see <xref linkend=\"Cjaefebe\">Advanced SIMD addressing mode</xref>.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information, see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VLD1":
            return {
                "tooltip": "Load multiple single 1-element structures to one, two, three, or four registers loads elements from memory into one, two, three, or four registers, without de-interleaving. Every element of each register is loaded. For details of the addressing mode, see Advanced SIMD addressing mode.",
                "html": "<p>Load multiple single 1-element structures to one, two, three, or four registers loads elements from memory into one, two, three, or four registers, without de-interleaving. Every element of each register is loaded. For details of the addressing mode, see <xref linkend=\"Cjaefebe\">Advanced SIMD addressing mode</xref>.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information, see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VLD2":
            return {
                "tooltip": "Load single 2-element structure to one lane of two registers loads one 2-element structure from memory into corresponding elements of two registers. Elements of the registers that are not loaded are unchanged. For details of the addressing mode, see Advanced SIMD addressing mode.",
                "html": "<p>Load single 2-element structure to one lane of two registers loads one 2-element structure from memory into corresponding elements of two registers. Elements of the registers that are not loaded are unchanged. For details of the addressing mode, see <xref linkend=\"Cjaefebe\">Advanced SIMD addressing mode</xref>.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information, see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VLD2":
            return {
                "tooltip": "Load single 2-element structure and replicate to all lanes of two registers loads one 2-element structure from memory into all lanes of two registers. For details of the addressing mode, see Advanced SIMD addressing mode.",
                "html": "<p>Load single 2-element structure and replicate to all lanes of two registers loads one 2-element structure from memory into all lanes of two registers. For details of the addressing mode, see <xref linkend=\"Cjaefebe\">Advanced SIMD addressing mode</xref>.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information, see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VLD2":
            return {
                "tooltip": "Load multiple 2-element structures to two or four registers loads multiple 2-element structures from memory into two or four registers, with de-interleaving. For more information, see Element and structure load/store instructions. Every element of each register is loaded. For details of the addressing mode, see Advanced SIMD addressing mode.",
                "html": "<p>Load multiple 2-element structures to two or four registers loads multiple 2-element structures from memory into two or four registers, with de-interleaving. For more information, see <xref linkend=\"BABHJAGF\">Element and structure load/store instructions</xref>. Every element of each register is loaded. For details of the addressing mode, see <xref linkend=\"Cjaefebe\">Advanced SIMD addressing mode</xref>.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information, see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VLD3":
            return {
                "tooltip": "Load single 3-element structure to one lane of three registers loads one 3-element structure from memory into corresponding elements of three registers. Elements of the registers that are not loaded are unchanged. For details of the addressing mode, see Advanced SIMD addressing mode.",
                "html": "<p>Load single 3-element structure to one lane of three registers loads one 3-element structure from memory into corresponding elements of three registers. Elements of the registers that are not loaded are unchanged. For details of the addressing mode, see <xref linkend=\"Cjaefebe\">Advanced SIMD addressing mode</xref>.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information, see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VLD3":
            return {
                "tooltip": "Load single 3-element structure and replicate to all lanes of three registers loads one 3-element structure from memory into all lanes of three registers. For details of the addressing mode, see Advanced SIMD addressing mode.",
                "html": "<p>Load single 3-element structure and replicate to all lanes of three registers loads one 3-element structure from memory into all lanes of three registers. For details of the addressing mode, see <xref linkend=\"Cjaefebe\">Advanced SIMD addressing mode</xref>.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information, see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VLD3":
            return {
                "tooltip": "Load multiple 3-element structures to three registers loads multiple 3-element structures from memory into three registers, with de-interleaving. For more information, see Element and structure load/store instructions. Every element of each register is loaded. For details of the addressing mode, see Advanced SIMD addressing mode.",
                "html": "<p>Load multiple 3-element structures to three registers loads multiple 3-element structures from memory into three registers, with de-interleaving. For more information, see <xref linkend=\"BABHJAGF\">Element and structure load/store instructions</xref>. Every element of each register is loaded. For details of the addressing mode, see <xref linkend=\"Cjaefebe\">Advanced SIMD addressing mode</xref>.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information, see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VLD4":
            return {
                "tooltip": "Load single 4-element structure to one lane of four registers loads one 4-element structure from memory into corresponding elements of four registers. Elements of the registers that are not loaded are unchanged. For details of the addressing mode, see Advanced SIMD addressing mode.",
                "html": "<p>Load single 4-element structure to one lane of four registers loads one 4-element structure from memory into corresponding elements of four registers. Elements of the registers that are not loaded are unchanged. For details of the addressing mode, see <xref linkend=\"Cjaefebe\">Advanced SIMD addressing mode</xref>.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information, see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VLD4":
            return {
                "tooltip": "Load single 4-element structure and replicate to all lanes of four registers loads one 4-element structure from memory into all lanes of four registers. For details of the addressing mode, see Advanced SIMD addressing mode.",
                "html": "<p>Load single 4-element structure and replicate to all lanes of four registers loads one 4-element structure from memory into all lanes of four registers. For details of the addressing mode, see <xref linkend=\"Cjaefebe\">Advanced SIMD addressing mode</xref>.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information, see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VLD4":
            return {
                "tooltip": "Load multiple 4-element structures to four registers loads multiple 4-element structures from memory into four registers, with de-interleaving. For more information, see Element and structure load/store instructions. Every element of each register is loaded. For details of the addressing mode, see Advanced SIMD addressing mode.",
                "html": "<p>Load multiple 4-element structures to four registers loads multiple 4-element structures from memory into four registers, with de-interleaving. For more information, see <xref linkend=\"BABHJAGF\">Element and structure load/store instructions</xref>. Every element of each register is loaded. For details of the addressing mode, see <xref linkend=\"Cjaefebe\">Advanced SIMD addressing mode</xref>.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information, see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VLDM":
        case "VLDMDB":
        case "VLDMIA":
            return {
                "tooltip": "Load Multiple SIMD&FP registers loads multiple registers from consecutive locations in the Advanced SIMD and floating-point register file using an address from a general-purpose register.",
                "html": "<p>Load Multiple SIMD&amp;FP registers loads multiple registers from consecutive locations in the Advanced SIMD and floating-point register file using an address from a general-purpose register.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information, see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VLDR":
            return {
                "tooltip": "Load SIMD&FP register (immediate) loads a single register from the Advanced SIMD and floating-point register file, using an address from a general-purpose register, with an optional offset.",
                "html": "<p>Load SIMD&amp;FP register (immediate) loads a single register from the Advanced SIMD and floating-point register file, using an address from a general-purpose register, with an optional offset.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, <xref linkend=\"AArch32.hcptr\">HCPTR</xref>, and <xref linkend=\"AArch32.fpexc\">FPEXC</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information, see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VLDR":
            return {
                "tooltip": "Load SIMD&FP register (literal) loads a single register from the Advanced SIMD and floating-point register file, using an address from the PC value and an immediate offset.",
                "html": "<p>Load SIMD&amp;FP register (literal) loads a single register from the Advanced SIMD and floating-point register file, using an address from the PC value and an immediate offset.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, <xref linkend=\"AArch32.hcptr\">HCPTR</xref>, and <xref linkend=\"AArch32.fpexc\">FPEXC</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information, see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VMAX":
            return {
                "tooltip": "Vector Maximum compares corresponding elements in two vectors, and copies the larger of each pair into the corresponding element in the destination vector.",
                "html": "<p>Vector Maximum compares corresponding elements in two vectors, and copies the larger of each pair into the corresponding element in the destination vector.</p><p>The operand vector elements are floating-point numbers.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VMAX":
            return {
                "tooltip": "Vector Maximum compares corresponding elements in two vectors, and copies the larger of each pair into the corresponding element in the destination vector.",
                "html": "<p>Vector Maximum compares corresponding elements in two vectors, and copies the larger of each pair into the corresponding element in the destination vector.</p><p>The operand vector elements can be any one of:</p><p>The result vector elements are the same size as the operand vector elements.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VMAXNM":
            return {
                "tooltip": "This instruction determines the floating-point maximum number.",
                "html": "<p>This instruction determines the floating-point maximum number.</p><p>It handles NaNs in consistence with the IEEE754-2008 specification. It returns the numerical operand when one operand is numerical and the other is a quiet NaN, but otherwise the result is identical to floating-point <instruction>VMAX</instruction>.</p><p>This instruction is not conditional.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VMIN":
            return {
                "tooltip": "Vector Minimum compares corresponding elements in two vectors, and copies the smaller of each pair into the corresponding element in the destination vector.",
                "html": "<p>Vector Minimum compares corresponding elements in two vectors, and copies the smaller of each pair into the corresponding element in the destination vector.</p><p>The operand vector elements are floating-point numbers.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VMIN":
            return {
                "tooltip": "Vector Minimum compares corresponding elements in two vectors, and copies the smaller of each pair into the corresponding element in the destination vector.",
                "html": "<p>Vector Minimum compares corresponding elements in two vectors, and copies the smaller of each pair into the corresponding element in the destination vector.</p><p>The operand vector elements can be any one of:</p><p>The result vector elements are the same size as the operand vector elements.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VMINNM":
            return {
                "tooltip": "This instruction determines the floating point minimum number.",
                "html": "<p>This instruction determines the floating point minimum number.</p><p>It handles NaNs in consistence with the IEEE754-2008 specification. It returns the numerical operand when one operand is numerical and the other is a quiet NaN, but otherwise the result is identical to floating-point <instruction>VMIN</instruction>.</p><p>This instruction is not conditional.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VMLA":
            return {
                "tooltip": "Vector Multiply Accumulate multiplies corresponding elements in two vectors, and accumulates the results into the elements of the destination vector.",
                "html": "<p>Vector Multiply Accumulate multiplies corresponding elements in two vectors, and accumulates the results into the elements of the destination vector.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, <xref linkend=\"AArch32.hcptr\">HCPTR</xref>, and <xref linkend=\"AArch32.fpexc\">FPEXC</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VMLA":
            return {
                "tooltip": "Vector Multiply Accumulate multiplies corresponding elements in two vectors, and adds the products to the corresponding elements of the destination vector.",
                "html": "<p>Vector Multiply Accumulate multiplies corresponding elements in two vectors, and adds the products to the corresponding elements of the destination vector.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VMLA":
            return {
                "tooltip": "Vector Multiply Accumulate multiplies elements of a vector by a scalar, and adds the products to corresponding elements of the destination vector.",
                "html": "<p>Vector Multiply Accumulate multiplies elements of a vector by a scalar, and adds the products to corresponding elements of the destination vector.</p><p>For more information about scalars see <xref linkend=\"Cjaibjhd\">Advanced SIMD scalars</xref>.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VMLAL":
            return {
                "tooltip": "Vector Multiply Accumulate Long multiplies corresponding elements in two vectors, and add the products to the corresponding element of the destination vector. The destination vector element is twice as long as the elements that are multiplied.",
                "html": "<p>Vector Multiply Accumulate Long multiplies corresponding elements in two vectors, and add the products to the corresponding element of the destination vector. The destination vector element is twice as long as the elements that are multiplied.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VMLAL":
            return {
                "tooltip": "Vector Multiply Accumulate Long multiplies elements of a vector by a scalar, and adds the products to corresponding elements of the destination vector. The destination vector elements are twice as long as the elements that are multiplied.",
                "html": "<p>Vector Multiply Accumulate Long multiplies elements of a vector by a scalar, and adds the products to corresponding elements of the destination vector. The destination vector elements are twice as long as the elements that are multiplied.</p><p>For more information about scalars see <xref linkend=\"Cjaibjhd\">Advanced SIMD scalars</xref>.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VMLS":
            return {
                "tooltip": "Vector Multiply Subtract multiplies corresponding elements in two vectors, subtracts the products from corresponding elements of the destination vector, and places the results in the destination vector.",
                "html": "<p>Vector Multiply Subtract multiplies corresponding elements in two vectors, subtracts the products from corresponding elements of the destination vector, and places the results in the destination vector.</p><p>Arm recommends that software does not use the <instruction>VMLS</instruction> instruction in the Round towards Plus Infinity and Round towards Minus Infinity rounding modes, because the rounding of the product and of the sum can change the result of the instruction in opposite directions, defeating the purpose of these rounding modes.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, <xref linkend=\"AArch32.hcptr\">HCPTR</xref>, and <xref linkend=\"AArch32.fpexc\">FPEXC</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VMLS":
            return {
                "tooltip": "Vector Multiply Subtract multiplies corresponding elements in two vectors, and subtracts the products from the corresponding elements of the destination vector.",
                "html": "<p>Vector Multiply Subtract multiplies corresponding elements in two vectors, and subtracts the products from the corresponding elements of the destination vector.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VMLS":
            return {
                "tooltip": "Vector Multiply Subtract multiplies elements of a vector by a scalar, and either subtracts the products from corresponding elements of the destination vector.",
                "html": "<p>Vector Multiply Subtract multiplies elements of a vector by a scalar, and either subtracts the products from corresponding elements of the destination vector.</p><p>For more information about scalars see <xref linkend=\"Cjaibjhd\">Advanced SIMD scalars</xref>.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VMLSL":
            return {
                "tooltip": "Vector Multiply Subtract Long multiplies corresponding elements in two vectors, and subtract the products from the corresponding elements of the destination vector. The destination vector element is twice as long as the elements that are multiplied.",
                "html": "<p>Vector Multiply Subtract Long multiplies corresponding elements in two vectors, and subtract the products from the corresponding elements of the destination vector. The destination vector element is twice as long as the elements that are multiplied.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VMLSL":
            return {
                "tooltip": "Vector Multiply Subtract Long multiplies elements of a vector by a scalar, and subtracts the products from corresponding elements of the destination vector. The destination vector elements are twice as long as the elements that are multiplied.",
                "html": "<p>Vector Multiply Subtract Long multiplies elements of a vector by a scalar, and subtracts the products from corresponding elements of the destination vector. The destination vector elements are twice as long as the elements that are multiplied.</p><p>For more information about scalars see <xref linkend=\"Cjaibjhd\">Advanced SIMD scalars</xref>.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VMMLA":
            return {
                "tooltip": "BFloat16 floating-point matrix multiply-accumulate. This instruction multiplies the 2x4 matrix of BF16 values in the first 128-bit source vector by the 4x2 BF16 matrix in the second 128-bit source vector. The resulting 2x2 single-precision matrix product is then added destructively to the 2x2 single-precision matrix in the 128-bit destination vector. This is equivalent to performing a 4-way dot product per destination element. The instruction does not update the FPSCR exception status.",
                "html": "<p>BFloat16 floating-point matrix multiply-accumulate. This instruction multiplies the 2x4 matrix of BF16 values in the first 128-bit source vector by the 4x2 BF16 matrix in the second 128-bit source vector. The resulting 2x2 single-precision matrix product is then added destructively to the 2x2 single-precision matrix in the 128-bit destination vector. This is equivalent to performing a 4-way dot product per destination element. The instruction does not update the <xref linkend=\"AArch32.fpscr\">FPSCR</xref> exception status.</p><p>Arm expects that the VMMLA instruction will deliver a peak BF16 multiply throughput that is at least as high as can be achieved using two VDOT instructions, with a goal that it should have significantly higher throughput.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VMOV":
            return {
                "tooltip": "Copy two general-purpose registers to or from a SIMD&FP register copies two words from two general-purpose registers into a doubleword register in the Advanced SIMD and floating-point register file, or from a doubleword register in the Advanced SIMD and floating-point register file to two general-purpose registers.",
                "html": "<p>Copy two general-purpose registers to or from a SIMD&amp;FP register copies two words from two general-purpose registers into a doubleword register in the Advanced SIMD and floating-point register file, or from a doubleword register in the Advanced SIMD and floating-point register file to two general-purpose registers.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, <xref linkend=\"AArch32.hcptr\">HCPTR</xref>, and <xref linkend=\"AArch32.fpexc\">FPEXC</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VMOV":
            return {
                "tooltip": "Copy 16 bits of a general-purpose register to or from a 32-bit SIMD&FP register.  This instruction transfers the value held in the bottom 16 bits of a 32-bit SIMD&FP register to the bottom 16 bits of a general-purpose register, or the value held in the bottom 16 bits of a general-purpose register to the bottom 16 bits of a 32-bit SIMD&FP register.",
                "html": "<p>Copy 16 bits of a general-purpose register to or from a 32-bit SIMD&amp;FP register.  This instruction transfers the value held in the bottom 16 bits of a 32-bit SIMD&amp;FP register to the bottom 16 bits of a general-purpose register, or the value held in the bottom 16 bits of a general-purpose register to the bottom 16 bits of a 32-bit SIMD&amp;FP register.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, <xref linkend=\"AArch32.hcptr\">HCPTR</xref>, and <xref linkend=\"AArch32.fpexc\">FPEXC</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VMOV":
            return {
                "tooltip": "Copy immediate value to a SIMD&FP register places an immediate constant into every element of the destination register.",
                "html": "<p>Copy immediate value to a SIMD&amp;FP register places an immediate constant into every element of the destination register.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, <xref linkend=\"AArch32.hcptr\">HCPTR</xref>, and <xref linkend=\"AArch32.fpexc\">FPEXC</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode.  For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VMOV":
            return {
                "tooltip": "Copy between FP registers copies the contents of one FP register to another.",
                "html": "<p>Copy between FP registers copies the contents of one FP register to another.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, <xref linkend=\"AArch32.hcptr\">HCPTR</xref>, and <xref linkend=\"AArch32.fpexc\">FPEXC</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VMOV":
            return {
                "tooltip": "Copy a general-purpose register to a vector element copies a byte, halfword, or word from a general-purpose register into an Advanced SIMD scalar.",
                "html": "<p>Copy a general-purpose register to a vector element copies a byte, halfword, or word from a general-purpose register into an Advanced SIMD scalar.</p><p>On a Floating-point-only system, this instruction transfers one word to the upper or lower half of a double-precision floating-point register from a general-purpose register. This is an identical operation to the Advanced SIMD single word transfer.</p><p>For more information about scalars see <xref linkend=\"Cjaibjhd\">Advanced SIMD scalars</xref>.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, <xref linkend=\"AArch32.hcptr\">HCPTR</xref>, and <xref linkend=\"AArch32.fpexc\">FPEXC</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VMOV":
            return {
                "tooltip": "Copy a general-purpose register to or from a 32-bit SIMD&FP register. This instruction transfers the value held in a 32-bit SIMD&FP register to a general-purpose register, or the value held in a general-purpose register to a 32-bit SIMD&FP register.",
                "html": "<p>Copy a general-purpose register to or from a 32-bit SIMD&amp;FP register. This instruction transfers the value held in a 32-bit SIMD&amp;FP register to a general-purpose register, or the value held in a general-purpose register to a 32-bit SIMD&amp;FP register.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, <xref linkend=\"AArch32.hcptr\">HCPTR</xref>, and <xref linkend=\"AArch32.fpexc\">FPEXC</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VMOV":
            return {
                "tooltip": "Copy a vector element to a general-purpose register with sign or zero extension copies a byte, halfword, or word from an Advanced SIMD scalar to a general-purpose register. Bytes and halfwords can be either zero-extended or sign-extended.",
                "html": "<p>Copy a vector element to a general-purpose register with sign or zero extension copies a byte, halfword, or word from an Advanced SIMD scalar to a general-purpose register. Bytes and halfwords can be either zero-extended or sign-extended.</p><p>On a Floating-point-only system, this instruction transfers one word from the upper or lower half of a double-precision floating-point register to a general-purpose register. This is an identical operation to the Advanced SIMD single word transfer.</p><p>For more information about scalars see <xref linkend=\"Cjaibjhd\">Advanced SIMD scalars</xref>.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, <xref linkend=\"AArch32.hcptr\">HCPTR</xref>, and <xref linkend=\"AArch32.fpexc\">FPEXC</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VMOV":
            return {
                "tooltip": "Copy two general-purpose registers to a pair of 32-bit SIMD&FP registers transfers the contents of two consecutively numbered single-precision Floating-point registers to two general-purpose registers, or the contents of two general-purpose registers to a pair of single-precision Floating-point registers. The general-purpose registers do not have to be contiguous.",
                "html": "<p>Copy two general-purpose registers to a pair of 32-bit SIMD&amp;FP registers transfers the contents of two consecutively numbered single-precision Floating-point registers to two general-purpose registers, or the contents of two general-purpose registers to a pair of single-precision Floating-point registers. The general-purpose registers do not have to be contiguous.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, <xref linkend=\"AArch32.hcptr\">HCPTR</xref>, and <xref linkend=\"AArch32.fpexc\">FPEXC</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VMOVL":
            return {
                "tooltip": "Vector Move Long takes each element in a doubleword vector, sign or zero-extends them to twice their original length, and places the results in a quadword vector.",
                "html": "<p>Vector Move Long takes each element in a doubleword vector, sign or zero-extends them to twice their original length, and places the results in a quadword vector.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VMOVN":
            return {
                "tooltip": "Vector Move and Narrow copies the least significant half of each element of a quadword vector into the corresponding elements of a doubleword vector.",
                "html": "<p>Vector Move and Narrow copies the least significant half of each element of a quadword vector into the corresponding elements of a doubleword vector.</p><p>The operand vector elements can be any one of 16-bit, 32-bit, or 64-bit integers. There is no distinction between signed and unsigned integers.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VMOVX":
            return {
                "tooltip": "Vector Move extraction. This instruction copies the upper 16 bits of the 32-bit source SIMD&FP register into the lower 16 bits of the 32-bit destination SIMD&FP register, while clearing the remaining bits to zero.",
                "html": "<p>Vector Move extraction. This instruction copies the upper 16 bits of the 32-bit source SIMD&amp;FP register into the lower 16 bits of the 32-bit destination SIMD&amp;FP register, while clearing the remaining bits to zero.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, <xref linkend=\"AArch32.hcptr\">HCPTR</xref>, and <xref linkend=\"AArch32.fpexc\">FPEXC</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VMRS":
            return {
                "tooltip": "Move SIMD&FP Special register to general-purpose register moves the value of an Advanced SIMD and floating-point System register to a general-purpose register. When the specified System register is the FPSCR, a form of the instruction transfers the FPSCR.{N, Z, C, V} condition flags to the APSR.{N, Z, C, V} condition flags.",
                "html": "<p>Move SIMD&amp;FP Special register to general-purpose register moves the value of an Advanced SIMD and floating-point System register to a general-purpose register. When the specified System register is the <xref linkend=\"AArch32.fpscr\">FPSCR</xref>, a form of the instruction transfers the <xref linkend=\"AArch32.fpscr\">FPSCR</xref>.{N, Z, C, V} condition flags to the <xref linkend=\"AArch32.apsr\">APSR</xref>.{N, Z, C, V} condition flags.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, <xref linkend=\"AArch32.hcptr\">HCPTR</xref>, and <xref linkend=\"AArch32.fpexc\">FPEXC</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p><p>When these settings permit the execution of Advanced SIMD and floating-point instructions, if the specified floating-point System register is not the <xref linkend=\"AArch32.fpscr\">FPSCR</xref>, the instruction is <arm-defined-word>undefined</arm-defined-word> if executed in User mode.</p><p>In an implementation that includes EL2, when <xref linkend=\"AArch32.hcr\">HCR</xref>.TID0 is set to 1, any <instruction>VMRS</instruction> access to <xref linkend=\"AArch32.fpsid\">FPSID</xref> from a Non-secure EL1 mode that would be permitted if <xref linkend=\"AArch32.hcr\">HCR</xref>.TID0 was set to 0 generates a Hyp Trap exception. For more information, see <xref linkend=\"CHDFGEDI\">ID group 0, Primary device identification registers</xref>.</p><p>For simplicity, the <instruction>VMRS</instruction> pseudocode does not show the possible trap to Hyp mode.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VMSR":
            return {
                "tooltip": "Move general-purpose register to SIMD&FP Special register moves the value of a general-purpose register to a floating-point System register.",
                "html": "<p>Move general-purpose register to SIMD&amp;FP Special register moves the value of a general-purpose register to a floating-point System register.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, <xref linkend=\"AArch32.hcptr\">HCPTR</xref>, and <xref linkend=\"AArch32.fpexc\">FPEXC</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p><p>When these settings permit the execution of Advanced SIMD and floating-point instructions:</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VMUL":
            return {
                "tooltip": "Vector Multiply multiplies corresponding elements in two vectors, and places the results in the destination vector.",
                "html": "<p>Vector Multiply multiplies corresponding elements in two vectors, and places the results in the destination vector.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, <xref linkend=\"AArch32.hcptr\">HCPTR</xref>, and <xref linkend=\"AArch32.fpexc\">FPEXC</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VMUL":
            return {
                "tooltip": "Vector Multiply multiplies corresponding elements in two vectors.",
                "html": "<p>Vector Multiply multiplies corresponding elements in two vectors.</p><p>For information about multiplying polynomials, see <xref linkend=\"BABDGBIC\">Polynomial arithmetic over {0, 1}</xref>.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information, see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VMUL":
            return {
                "tooltip": "Vector Multiply multiplies each element in a vector by a scalar, and places the results in a second vector.",
                "html": "<p>Vector Multiply multiplies each element in a vector by a scalar, and places the results in a second vector.</p><p>For more information about scalars see <xref linkend=\"Cjaibjhd\">Advanced SIMD scalars</xref>.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VMULL":
            return {
                "tooltip": "Vector Multiply Long multiplies corresponding elements in two vectors. The destination vector elements are twice as long as the elements that are multiplied.",
                "html": "<p>Vector Multiply Long multiplies corresponding elements in two vectors. The destination vector elements are twice as long as the elements that are multiplied.</p><p>For information about multiplying polynomials see <xref linkend=\"BABDGBIC\">Polynomial arithmetic over {0, 1}</xref>.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VMULL":
            return {
                "tooltip": "Vector Multiply Long multiplies each element in a vector by a scalar, and places the results in a second vector. The destination vector elements are twice as long as the elements that are multiplied.",
                "html": "<p>Vector Multiply Long multiplies each element in a vector by a scalar, and places the results in a second vector. The destination vector elements are twice as long as the elements that are multiplied.</p><p>For more information about scalars see <xref linkend=\"Cjaibjhd\">Advanced SIMD scalars</xref>.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VMVN":
            return {
                "tooltip": "Vector Bitwise NOT (immediate) places the bitwise inverse of an immediate integer constant into every element of the destination register.",
                "html": "<p>Vector Bitwise NOT (immediate) places the bitwise inverse of an immediate integer constant into every element of the destination register.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VMVN":
            return {
                "tooltip": "Vector Bitwise NOT (register) takes a value from a register, inverts the value of each bit, and places the result in the destination register. The registers can be either doubleword or quadword.",
                "html": "<p>Vector Bitwise NOT (register) takes a value from a register, inverts the value of each bit, and places the result in the destination register. The registers can be either doubleword or quadword.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VNEG":
            return {
                "tooltip": "Vector Negate negates each element in a vector, and places the results in a second vector. The floating-point version only inverts the sign bit.",
                "html": "<p>Vector Negate negates each element in a vector, and places the results in a second vector. The floating-point version only inverts the sign bit.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, <xref linkend=\"AArch32.hcptr\">HCPTR</xref>, and <xref linkend=\"AArch32.fpexc\">FPEXC</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VNMLA":
            return {
                "tooltip": "Vector Negate Multiply Accumulate multiplies together two floating-point register values, adds the negation of the floating-point value in the destination register to the negation of the product, and writes the result back to the destination register.",
                "html": "<p>Vector Negate Multiply Accumulate multiplies together two floating-point register values, adds the negation of the floating-point value in the destination register to the negation of the product, and writes the result back to the destination register.</p><p>Arm recommends that software does not use the <instruction>VNMLA</instruction> instruction in the Round towards Plus Infinity and Round towards Minus Infinity rounding modes, because the rounding of the product and of the sum can change the result of the instruction in opposite directions, defeating the purpose of these rounding modes.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, <xref linkend=\"AArch32.hcptr\">HCPTR</xref>, and <xref linkend=\"AArch32.fpexc\">FPEXC</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VNMLS":
            return {
                "tooltip": "Vector Negate Multiply Subtract multiplies together two floating-point register values, adds the negation of the floating-point value in the destination register to the product, and writes the result back to the destination register.",
                "html": "<p>Vector Negate Multiply Subtract multiplies together two floating-point register values, adds the negation of the floating-point value in the destination register to the product, and writes the result back to the destination register.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, <xref linkend=\"AArch32.hcptr\">HCPTR</xref>, and <xref linkend=\"AArch32.fpexc\">FPEXC</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VNMUL":
            return {
                "tooltip": "Vector Negate Multiply multiplies together two floating-point register values, and writes the negation of the result to the destination register.",
                "html": "<p>Vector Negate Multiply multiplies together two floating-point register values, and writes the negation of the result to the destination register.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, <xref linkend=\"AArch32.hcptr\">HCPTR</xref>, and <xref linkend=\"AArch32.fpexc\">FPEXC</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VORN":
            return {
                "tooltip": "Vector bitwise OR NOT (register) performs a bitwise OR NOT operation between two registers, and places the result in the destination register.  The operand and result registers can be quadword or doubleword.  They must all be the same size.",
                "html": "<p>Vector bitwise OR NOT (register) performs a bitwise OR NOT operation between two registers, and places the result in the destination register.  The operand and result registers can be quadword or doubleword.  They must all be the same size.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VORR":
            return {
                "tooltip": "Vector Bitwise OR (immediate) performs a bitwise OR between a register value and an immediate value, and returns the result into the destination vector.",
                "html": "<p>Vector Bitwise OR (immediate) performs a bitwise OR between a register value and an immediate value, and returns the result into the destination vector.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VORR":
            return {
                "tooltip": "Vector bitwise OR (register) performs a bitwise OR operation between two registers, and places the result in the destination register. The operand and result registers can be quadword or doubleword. They must all be the same size.",
                "html": "<p>Vector bitwise OR (register) performs a bitwise OR operation between two registers, and places the result in the destination register. The operand and result registers can be quadword or doubleword. They must all be the same size.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VPADAL":
            return {
                "tooltip": "Vector Pairwise Add and Accumulate Long adds adjacent pairs of elements of a vector, and accumulates the results into the elements of the destination vector.",
                "html": "<p>Vector Pairwise Add and Accumulate Long adds adjacent pairs of elements of a vector, and accumulates the results into the elements of the destination vector.</p><p>The vectors can be doubleword or quadword. The operand elements can be 8-bit, 16-bit, or 32-bit integers. The result elements are twice the length of the operand elements.</p><p></p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VPADD":
            return {
                "tooltip": "Vector Pairwise Add (floating-point) adds adjacent pairs of elements of two vectors, and places the results in the destination vector.",
                "html": "<p>Vector Pairwise Add (floating-point) adds adjacent pairs of elements of two vectors, and places the results in the destination vector.</p><p>The operands and result are doubleword vectors.</p><p>The operand and result elements are floating-point numbers.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VPADD":
            return {
                "tooltip": "Vector Pairwise Add (integer) adds adjacent pairs of elements of two vectors, and places the results in the destination vector.",
                "html": "<p>Vector Pairwise Add (integer) adds adjacent pairs of elements of two vectors, and places the results in the destination vector.</p><p>The operands and result are doubleword vectors.</p><p>The operand and result elements must all be the same type, and can be 8-bit, 16-bit, or 32-bit integers. There is no distinction between signed and unsigned integers.</p><p></p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VPADDL":
            return {
                "tooltip": "Vector Pairwise Add Long adds adjacent pairs of elements of two vectors, and places the results in the destination vector.",
                "html": "<p>Vector Pairwise Add Long adds adjacent pairs of elements of two vectors, and places the results in the destination vector.</p><p>The vectors can be doubleword or quadword. The operand elements can be 8-bit, 16-bit, or 32-bit integers. The result elements are twice the length of the operand elements.</p><p></p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VPMAX":
            return {
                "tooltip": "Vector Pairwise Maximum compares adjacent pairs of elements in two doubleword vectors, and copies the larger of each pair into the corresponding element in the destination doubleword vector.",
                "html": "<p>Vector Pairwise Maximum compares adjacent pairs of elements in two doubleword vectors, and copies the larger of each pair into the corresponding element in the destination doubleword vector.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VPMAX":
            return {
                "tooltip": "Vector Pairwise Maximum compares adjacent pairs of elements in two doubleword vectors, and copies the larger of each pair into the corresponding element in the destination doubleword vector.",
                "html": "<p>Vector Pairwise Maximum compares adjacent pairs of elements in two doubleword vectors, and copies the larger of each pair into the corresponding element in the destination doubleword vector.</p><p></p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VPMIN":
            return {
                "tooltip": "Vector Pairwise Minimum compares adjacent pairs of elements in two doubleword vectors, and copies the smaller of each pair into the corresponding element in the destination doubleword vector.",
                "html": "<p>Vector Pairwise Minimum compares adjacent pairs of elements in two doubleword vectors, and copies the smaller of each pair into the corresponding element in the destination doubleword vector.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VPMIN":
            return {
                "tooltip": "Vector Pairwise Minimum compares adjacent pairs of elements in two doubleword vectors, and copies the smaller of each pair into the corresponding element in the destination doubleword vector.",
                "html": "<p>Vector Pairwise Minimum compares adjacent pairs of elements in two doubleword vectors, and copies the smaller of each pair into the corresponding element in the destination doubleword vector.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VPOP":
            return {
                "tooltip": "Pop SIMD&FP registers from stack loads multiple consecutive Advanced SIMD and floating-point register file registers from the stack.",
                "html": "<p>Pop SIMD&amp;FP registers from stack loads multiple consecutive Advanced SIMD and floating-point register file registers from the stack.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VPUSH":
            return {
                "tooltip": "Push SIMD&FP registers to stack stores multiple consecutive registers from the Advanced SIMD and floating-point register file to the stack.",
                "html": "<p>Push SIMD&amp;FP registers to stack stores multiple consecutive registers from the Advanced SIMD and floating-point register file to the stack.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VQABS":
            return {
                "tooltip": "Vector Saturating Absolute takes the absolute value of each element in a vector, and places the results in the destination vector.",
                "html": "<p>Vector Saturating Absolute takes the absolute value of each element in a vector, and places the results in the destination vector.</p><p>If any of the results overflow, they are saturated. The cumulative saturation bit, <xref linkend=\"AArch32.fpscr\">FPSCR</xref>.QC, is set if saturation occurs. For details see <xref linkend=\"BEIHABGJ\">Pseudocode details of saturation</xref>.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VQADD":
            return {
                "tooltip": "Vector Saturating Add adds the values of corresponding elements of two vectors, and places the results in the destination vector.",
                "html": "<p>Vector Saturating Add adds the values of corresponding elements of two vectors, and places the results in the destination vector.</p><p>If any of the results overflow, they are saturated. The cumulative saturation bit, <xref linkend=\"AArch32.fpscr\">FPSCR</xref>.QC, is set if saturation occurs. For details see <xref linkend=\"BEIHABGJ\">Pseudocode details of saturation</xref>.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VQDMLAL":
            return {
                "tooltip": "Vector Saturating Doubling Multiply Accumulate Long multiplies corresponding elements in two doubleword vectors, doubles the products, and accumulates the results into the elements of a quadword vector.",
                "html": "<p>Vector Saturating Doubling Multiply Accumulate Long multiplies corresponding elements in two doubleword vectors, doubles the products, and accumulates the results into the elements of a quadword vector.</p><p>The second operand can be a scalar instead of a vector. For more information about scalars see <xref linkend=\"Cjaibjhd\">Advanced SIMD scalars</xref>.</p><p>If any of the results overflow, they are saturated. The cumulative saturation bit, <xref linkend=\"AArch32.fpscr\">FPSCR</xref>.QC, is set if saturation occurs. For details see <xref linkend=\"BEIHABGJ\">Pseudocode details of saturation</xref>.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VQDMLSL":
            return {
                "tooltip": "Vector Saturating Doubling Multiply Subtract Long multiplies corresponding elements in two doubleword vectors, subtracts double the products from corresponding elements of a quadword vector, and places the results in the same quadword vector.",
                "html": "<p>Vector Saturating Doubling Multiply Subtract Long multiplies corresponding elements in two doubleword vectors, subtracts double the products from corresponding elements of a quadword vector, and places the results in the same quadword vector.</p><p>The second operand can be a scalar instead of a vector. For more information about scalars see <xref linkend=\"Cjaibjhd\">Advanced SIMD scalars</xref>.</p><p>If any of the results overflow, they are saturated. The cumulative saturation bit, <xref linkend=\"AArch32.fpscr\">FPSCR</xref>.QC, is set if saturation occurs. For details see <xref linkend=\"BEIHABGJ\">Pseudocode details of saturation</xref>.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VQDMULH":
            return {
                "tooltip": "Vector Saturating Doubling Multiply Returning High Half multiplies corresponding elements in two vectors, doubles the results, and places the most significant half of the final results in the destination vector. The results are truncated, for rounded results see VQRDMULH.",
                "html": "<p>Vector Saturating Doubling Multiply Returning High Half multiplies corresponding elements in two vectors, doubles the results, and places the most significant half of the final results in the destination vector. The results are truncated, for rounded results see <xref linkend=\"A32T32-fpsimd.instructions.VQRDMULH\">VQRDMULH</xref>.</p><p>The second operand can be a scalar instead of a vector. For more information about scalars see <xref linkend=\"Cjaibjhd\">Advanced SIMD scalars</xref>.</p><p>If any of the results overflow, they are saturated. The cumulative saturation bit, <xref linkend=\"AArch32.fpscr\">FPSCR</xref>.QC, is set if saturation occurs. For details see <xref linkend=\"BEIHABGJ\">Pseudocode details of saturation</xref>.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VQDMULL":
            return {
                "tooltip": "Vector Saturating Doubling Multiply Long multiplies corresponding elements in two doubleword vectors, doubles the products, and places the results in a quadword vector.",
                "html": "<p>Vector Saturating Doubling Multiply Long multiplies corresponding elements in two doubleword vectors, doubles the products, and places the results in a quadword vector.</p><p>The second operand can be a scalar instead of a vector. For more information about scalars see <xref linkend=\"Cjaibjhd\">Advanced SIMD scalars</xref>.</p><p>If any of the results overflow, they are saturated. The cumulative saturation bit, <xref linkend=\"AArch32.fpscr\">FPSCR</xref>.QC, is set if saturation occurs. For details see <xref linkend=\"BEIHABGJ\">Pseudocode details of saturation</xref>.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VQMOVN":
        case "VQMOVUN":
            return {
                "tooltip": "Vector Saturating Move and Narrow copies each element of the operand vector to the corresponding element of the destination vector.",
                "html": "<p>Vector Saturating Move and Narrow copies each element of the operand vector to the corresponding element of the destination vector.</p><p>The operand is a quadword vector. The elements can be any one of:</p><p>The result is a doubleword vector. The elements are half the length of the operand vector elements. If the operand is unsigned, the results are unsigned. If the operand is signed, the results can be signed or unsigned.</p><p>If any of the results overflow, they are saturated. The cumulative saturation bit, <xref linkend=\"AArch32.fpscr\">FPSCR</xref>.QC, is set if saturation occurs. For details see <xref linkend=\"BEIHABGJ\">Pseudocode details of saturation</xref>.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VQNEG":
            return {
                "tooltip": "Vector Saturating Negate negates each element in a vector, and places the results in the destination vector.",
                "html": "<p>Vector Saturating Negate negates each element in a vector, and places the results in the destination vector.</p><p>If any of the results overflow, they are saturated. The cumulative saturation bit, <xref linkend=\"AArch32.fpscr\">FPSCR</xref>.QC, is set if saturation occurs. For details see <xref linkend=\"BEIHABGJ\">Pseudocode details of saturation</xref>.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VQRDMLAH":
            return {
                "tooltip": "Vector Saturating Rounding Doubling Multiply Accumulate Returning High Half. This instruction multiplies the vector elements of the first source SIMD&FP register with either the corresponding vector elements of the second source SIMD&FP register or the value of a vector element of the second source SIMD&FP register, without saturating the multiply results, doubles the results, and accumulates the most significant half of the final results with the vector elements of the destination SIMD&FP register. The results are rounded.",
                "html": "<p>Vector Saturating Rounding Doubling Multiply Accumulate Returning High Half. This instruction multiplies the vector elements of the first source SIMD&amp;FP register with either the corresponding vector elements of the second source SIMD&amp;FP register or the value of a vector element of the second source SIMD&amp;FP register, without saturating the multiply results, doubles the results, and accumulates the most significant half of the final results with the vector elements of the destination SIMD&amp;FP register. The results are rounded.</p><p>If any of the results overflow, they are saturated. The cumulative saturation bit, <xref linkend=\"AArch32.fpscr\">FPSCR</xref>.QC, is set if saturation occurs. For details see <xref linkend=\"BEIHABGJ\">Pseudocode details of saturation</xref>.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VQRDMLSH":
            return {
                "tooltip": "Vector Saturating Rounding Doubling Multiply Subtract Returning High Half. This instruction multiplies the vector elements of the first source SIMD&FP register with either the corresponding vector elements of the second source SIMD&FP register or the value of a vector element of the second source SIMD&FP register, without saturating the multiply results, doubles the results, and subtracts the most significant half of the final results from the vector elements of the destination SIMD&FP register. The results are rounded.",
                "html": "<p>Vector Saturating Rounding Doubling Multiply Subtract Returning High Half. This instruction multiplies the vector elements of the first source SIMD&amp;FP register with either the corresponding vector elements of the second source SIMD&amp;FP register or the value of a vector element of the second source SIMD&amp;FP register, without saturating the multiply results, doubles the results, and subtracts the most significant half of the final results from the vector elements of the destination SIMD&amp;FP register. The results are rounded.</p><p>If any of the results overflow, they are saturated. The cumulative saturation bit, <xref linkend=\"AArch32.fpscr\">FPSCR</xref>.QC, is set if saturation occurs. For details see <xref linkend=\"BEIHABGJ\">Pseudocode details of saturation</xref>.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VQRDMULH":
            return {
                "tooltip": "Vector Saturating Rounding Doubling Multiply Returning High Half multiplies corresponding elements in two vectors, doubles the results, and places the most significant half of the final results in the destination vector. The results are rounded. For truncated results see VQDMULH.",
                "html": "<p>Vector Saturating Rounding Doubling Multiply Returning High Half multiplies corresponding elements in two vectors, doubles the results, and places the most significant half of the final results in the destination vector. The results are rounded. For truncated results see <xref linkend=\"A32T32-fpsimd.instructions.VQDMULH\">VQDMULH</xref>.</p><p>The second operand can be a scalar instead of a vector. For more information about scalars see <xref linkend=\"Cjaibjhd\">Advanced SIMD scalars</xref>.</p><p>If any of the results overflow, they are saturated. The cumulative saturation bit, <xref linkend=\"AArch32.fpscr\">FPSCR</xref>.QC, is set if saturation occurs. For details see <xref linkend=\"BEIHABGJ\">Pseudocode details of saturation</xref>.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VQRSHL":
            return {
                "tooltip": "Vector Saturating Rounding Shift Left takes each element in a vector, shifts them by a value from the least significant byte of the corresponding element of a second vector, and places the results in the destination vector. If the shift value is positive, the operation is a left shift. Otherwise, it is a right shift.",
                "html": "<p>Vector Saturating Rounding Shift Left takes each element in a vector, shifts them by a value from the least significant byte of the corresponding element of a second vector, and places the results in the destination vector. If the shift value is positive, the operation is a left shift. Otherwise, it is a right shift.</p><p>For truncated results see <xref linkend=\"A32T32-fpsimd.instructions.VQSHL_r\">VQSHL (register)</xref>.</p><p>The first operand and result elements are the same data type, and can be any one of:</p><p>The second operand is a signed integer of the same size.</p><p>If any of the results overflow, they are saturated. The cumulative saturation bit, <xref linkend=\"AArch32.fpscr\">FPSCR</xref>.QC, is set if saturation occurs. For details see <xref linkend=\"BEIHABGJ\">Pseudocode details of saturation</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VQRSHRN":
        case "VQRSHRUN":
            return {
                "tooltip": "Vector Saturating Rounding Shift Right, Narrow takes each element in a quadword vector of integers, right shifts them by an immediate value, and places the rounded results in a doubleword vector.",
                "html": "<p>Vector Saturating Rounding Shift Right, Narrow takes each element in a quadword vector of integers, right shifts them by an immediate value, and places the rounded results in a doubleword vector.</p><p>For truncated results, see <xref linkend=\"A32T32-fpsimd.instructions.VQSHL_r\">VQSHRN and VQSHRUN</xref>.</p><p>The operand elements must all be the same size, and can be any one of:</p><p>The result elements are half the width of the operand elements. If the operand elements are signed, the results can be either signed or unsigned. If the operand elements are unsigned, the result elements must also be unsigned.</p><p>If any of the results overflow, they are saturated. The cumulative saturation bit, <xref linkend=\"AArch32.fpscr\">FPSCR</xref>.QC, is set if saturation occurs. For details see <xref linkend=\"BEIHABGJ\">Pseudocode details of saturation</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VQSHL":
        case "VQSHLU":
            return {
                "tooltip": "Vector Saturating Shift Left (immediate) takes each element in a vector of integers, left shifts them by an immediate value, and places the results in a second vector.",
                "html": "<p>Vector Saturating Shift Left (immediate) takes each element in a vector of integers, left shifts them by an immediate value, and places the results in a second vector.</p><p>The operand elements must all be the same size, and can be any one of:</p><p>The result elements are the same size as the operand elements. If the operand elements are signed, the results can be either signed or unsigned. If the operand elements are unsigned, the result elements must also be unsigned.</p><p>If any of the results overflow, they are saturated. The cumulative saturation bit, <xref linkend=\"AArch32.fpscr\">FPSCR</xref>.QC, is set if saturation occurs. For details see <xref linkend=\"BEIHABGJ\">Pseudocode details of saturation</xref>.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VQSHL":
            return {
                "tooltip": "Vector Saturating Shift Left (register) takes each element in a vector, shifts them by a value from the least significant byte of the corresponding element of a second vector, and places the results in the destination vector. If the shift value is positive, the operation is a left shift. Otherwise, it is a right shift.",
                "html": "<p>Vector Saturating Shift Left (register) takes each element in a vector, shifts them by a value from the least significant byte of the corresponding element of a second vector, and places the results in the destination vector. If the shift value is positive, the operation is a left shift. Otherwise, it is a right shift.</p><p>The results are truncated. For rounded results, see <xref linkend=\"A32T32-fpsimd.instructions.VQRSHL\">VQRSHL</xref>.</p><p>The first operand and result elements are the same data type, and can be any one of:</p><p>The second operand is a signed integer of the same size.</p><p>If any of the results overflow, they are saturated. The cumulative saturation bit, <xref linkend=\"AArch32.fpscr\">FPSCR</xref>.QC, is set if saturation occurs. For details see <xref linkend=\"BEIHABGJ\">Pseudocode details of saturation</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VQSHRN":
        case "VQSHRUN":
            return {
                "tooltip": "Vector Saturating Shift Right, Narrow takes each element in a quadword vector of integers, right shifts them by an immediate value, and places the truncated results in a doubleword vector.",
                "html": "<p>Vector Saturating Shift Right, Narrow takes each element in a quadword vector of integers, right shifts them by an immediate value, and places the truncated results in a doubleword vector.</p><p>For rounded results, see <xref linkend=\"A32T32-fpsimd.instructions.VQRSHRN\">VQRSHRN and VQRSHRUN</xref>.</p><p>The operand elements must all be the same size, and can be any one of:</p><p>The result elements are half the width of the operand elements. If the operand elements are signed, the results can be either signed or unsigned. If the operand elements are unsigned, the result elements must also be unsigned.</p><p>If any of the results overflow, they are saturated. The cumulative saturation bit, <xref linkend=\"AArch32.fpscr\">FPSCR</xref>.QC, is set if saturation occurs. For details see <xref linkend=\"BEIHABGJ\">Pseudocode details of saturation</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VQSUB":
            return {
                "tooltip": "Vector Saturating Subtract subtracts the elements of the second operand vector from the corresponding elements of the first operand vector, and places the results in the destination vector. Signed and unsigned operations are distinct.",
                "html": "<p>Vector Saturating Subtract subtracts the elements of the second operand vector from the corresponding elements of the first operand vector, and places the results in the destination vector. Signed and unsigned operations are distinct.</p><p>The operand and result elements must all be the same type, and can be any one of:</p><p>If any of the results overflow, they are saturated. The cumulative saturation bit, <xref linkend=\"AArch32.fpscr\">FPSCR</xref>.QC, is set if saturation occurs. For details see <xref linkend=\"BEIHABGJ\">Pseudocode details of saturation</xref>.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VRADDHN":
            return {
                "tooltip": "Vector Rounding Add and Narrow, returning High Half adds corresponding elements in two quadword vectors, and places the most significant half of each result in a doubleword vector. The results are rounded.  For truncated results, see VADDHN.",
                "html": "<p>Vector Rounding Add and Narrow, returning High Half adds corresponding elements in two quadword vectors, and places the most significant half of each result in a doubleword vector. The results are rounded.  For truncated results, see <xref linkend=\"A32T32-fpsimd.instructions.VADDHN\">VADDHN</xref>.</p><p>The operand elements can be 16-bit, 32-bit, or 64-bit integers. There is no distinction between signed and unsigned integers.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VRECPE":
            return {
                "tooltip": "Vector Reciprocal Estimate finds an approximate reciprocal of each element in the operand vector, and places the results in the destination vector.",
                "html": "<p>Vector Reciprocal Estimate finds an approximate reciprocal of each element in the operand vector, and places the results in the destination vector.</p><p>The operand and result elements are the same type, and can be floating-point numbers or unsigned integers.</p><p>For details of the operation performed by this instruction see <xref linkend=\"CFIGIFDB\">Floating-point reciprocal square root estimate and step</xref>.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VRECPS":
            return {
                "tooltip": "Vector Reciprocal Step multiplies the elements of one vector by the corresponding elements of another vector, subtracts each of the products from 2.0, and places the results into the elements of the destination vector.",
                "html": "<p>Vector Reciprocal Step multiplies the elements of one vector by the corresponding elements of another vector, subtracts each of the products from 2.0, and places the results into the elements of the destination vector.</p><p>The operand and result elements are floating-point numbers.</p><p>For details of the operation performed by this instruction see <xref linkend=\"CFIJFDFG\">Floating-point reciprocal estimate and step</xref>.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VREV16":
            return {
                "tooltip": "Vector Reverse in halfwords reverses the order of 8-bit elements in each halfword of the vector, and places the result in the corresponding destination vector.",
                "html": "<p>Vector Reverse in halfwords reverses the order of 8-bit elements in each halfword of the vector, and places the result in the corresponding destination vector.</p><p>There is no distinction between data types, other than size.</p><p></p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VREV32":
            return {
                "tooltip": "Vector Reverse in words reverses the order of 8-bit or 16-bit elements in each word of the vector, and places the result in the corresponding destination vector.",
                "html": "<p>Vector Reverse in words reverses the order of 8-bit or 16-bit elements in each word of the vector, and places the result in the corresponding destination vector.</p><p>There is no distinction between data types, other than size.</p><p></p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VREV64":
            return {
                "tooltip": "Vector Reverse in doublewords reverses the order of 8-bit, 16-bit, or 32-bit elements in each doubleword of the vector, and places the result in the corresponding destination vector.",
                "html": "<p>Vector Reverse in doublewords reverses the order of 8-bit, 16-bit, or 32-bit elements in each doubleword of the vector, and places the result in the corresponding destination vector.</p><p>There is no distinction between data types, other than size.</p><p></p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VRHADD":
            return {
                "tooltip": "Vector Rounding Halving Add adds corresponding elements in two vectors of integers, shifts each result right one bit, and places the final results in the destination vector.",
                "html": "<p>Vector Rounding Halving Add adds corresponding elements in two vectors of integers, shifts each result right one bit, and places the final results in the destination vector.</p><p>The operand and result elements are all the same type, and can be any one of:</p><p>The results of the halving operations are rounded. For truncated results, see <xref linkend=\"A32T32-fpsimd.instructions.VHADD\">VHADD</xref>.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VRINTA":
            return {
                "tooltip": "Vector Round floating-point to integer towards Nearest with Ties to Away rounds a vector of floating-point values to integral floating-point values of the same size using the Round to Nearest with Ties to Away rounding mode. A zero input gives a zero result with the same sign, an infinite input gives an infinite result with the same sign, and a NaN is propagated as for normal arithmetic.",
                "html": "<p>Vector Round floating-point to integer towards Nearest with Ties to Away rounds a vector of floating-point values to integral floating-point values of the same size using the Round to Nearest with Ties to Away rounding mode. A zero input gives a zero result with the same sign, an infinite input gives an infinite result with the same sign, and a NaN is propagated as for normal arithmetic.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VRINTA":
            return {
                "tooltip": "Round floating-point to integer to Nearest with Ties to Away rounds a floating-point value to an integral floating-point value of the same size using the Round to Nearest with Ties to Away rounding mode. A zero input gives a zero result with the same sign, an infinite input gives an infinite result with the same sign, and a NaN is propagated as for normal arithmetic.",
                "html": "<p>Round floating-point to integer to Nearest with Ties to Away rounds a floating-point value to an integral floating-point value of the same size using the Round to Nearest with Ties to Away rounding mode. A zero input gives a zero result with the same sign, an infinite input gives an infinite result with the same sign, and a NaN is propagated as for normal arithmetic.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VRINTM":
            return {
                "tooltip": "Vector Round floating-point to integer towards -Infinity rounds a vector of floating-point values to integral floating-point values of the same size, using the Round towards -Infinity rounding mode. A zero input gives a zero result with the same sign, an infinite input gives an infinite result with the same sign, and a NaN is propagated as for normal arithmetic.",
                "html": "<p>Vector Round floating-point to integer towards -Infinity rounds a vector of floating-point values to integral floating-point values of the same size, using the Round towards -Infinity rounding mode. A zero input gives a zero result with the same sign, an infinite input gives an infinite result with the same sign, and a NaN is propagated as for normal arithmetic.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VRINTM":
            return {
                "tooltip": "Round floating-point to integer towards -Infinity rounds a floating-point value to an integral floating-point value of the same size using the Round towards -Infinity rounding mode. A zero input gives a zero result with the same sign, an infinite input gives an infinite result with the same sign, and a NaN is propagated as for normal arithmetic.",
                "html": "<p>Round floating-point to integer towards -Infinity rounds a floating-point value to an integral floating-point value of the same size using the Round towards -Infinity rounding mode. A zero input gives a zero result with the same sign, an infinite input gives an infinite result with the same sign, and a NaN is propagated as for normal arithmetic.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VRINTN":
            return {
                "tooltip": "Vector Round floating-point to integer to Nearest rounds a vector of floating-point values to integral floating-point values of the same size using the Round to Nearest rounding mode. A zero input gives a zero result with the same sign, an infinite input gives an infinite result with the same sign, and a NaN is propagated as for normal arithmetic.",
                "html": "<p>Vector Round floating-point to integer to Nearest rounds a vector of floating-point values to integral floating-point values of the same size using the Round to Nearest rounding mode. A zero input gives a zero result with the same sign, an infinite input gives an infinite result with the same sign, and a NaN is propagated as for normal arithmetic.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VRINTN":
            return {
                "tooltip": "Round floating-point to integer to Nearest rounds a floating-point value to an integral floating-point value of the same size using the Round to Nearest rounding mode. A zero input gives a zero result with the same sign, an infinite input gives an infinite result with the same sign, and a NaN is propagated as for normal arithmetic.",
                "html": "<p>Round floating-point to integer to Nearest rounds a floating-point value to an integral floating-point value of the same size using the Round to Nearest rounding mode. A zero input gives a zero result with the same sign, an infinite input gives an infinite result with the same sign, and a NaN is propagated as for normal arithmetic.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VRINTP":
            return {
                "tooltip": "Vector Round floating-point to integer towards +Infinity rounds a vector of floating-point values to integral floating-point values of the same size using the Round towards +Infinity rounding mode. A zero input gives a zero result with the same sign, an infinite input gives an infinite result with the same sign, and a NaN is propagated as for normal arithmetic.",
                "html": "<p>Vector Round floating-point to integer towards +Infinity rounds a vector of floating-point values to integral floating-point values of the same size using the Round towards +Infinity rounding mode. A zero input gives a zero result with the same sign, an infinite input gives an infinite result with the same sign, and a NaN is propagated as for normal arithmetic.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VRINTP":
            return {
                "tooltip": "Round floating-point to integer towards +Infinity rounds a floating-point value to an integral floating-point value of the same size using the Round towards +Infinity rounding mode. A zero input gives a zero result with the same sign, an infinite input gives an infinite result with the same sign, and a NaN is propagated as for normal arithmetic.",
                "html": "<p>Round floating-point to integer towards +Infinity rounds a floating-point value to an integral floating-point value of the same size using the Round towards +Infinity rounding mode. A zero input gives a zero result with the same sign, an infinite input gives an infinite result with the same sign, and a NaN is propagated as for normal arithmetic.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VRINTR":
            return {
                "tooltip": "Round floating-point to integer rounds a floating-point value to an integral floating-point value of the same size using the rounding mode specified in the FPSCR. A zero input gives a zero result with the same sign, an infinite input gives an infinite result with the same sign, and a NaN is propagated as for normal arithmetic.",
                "html": "<p>Round floating-point to integer rounds a floating-point value to an integral floating-point value of the same size using the rounding mode specified in the FPSCR. A zero input gives a zero result with the same sign, an infinite input gives an infinite result with the same sign, and a NaN is propagated as for normal arithmetic.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VRINTX":
            return {
                "tooltip": "Vector round floating-point to integer inexact rounds a vector of floating-point values to integral floating-point values of the same size, using the Round to Nearest rounding mode, and raises the Inexact exception when the result value is not numerically equal to the input value. A zero input gives a zero result with the same sign, an infinite input gives an infinite result with the same sign, and a NaN is propagated as for normal arithmetic.",
                "html": "<p>Vector round floating-point to integer inexact rounds a vector of floating-point values to integral floating-point values of the same size, using the Round to Nearest rounding mode, and raises the Inexact exception when the result value is not numerically equal to the input value. A zero input gives a zero result with the same sign, an infinite input gives an infinite result with the same sign, and a NaN is propagated as for normal arithmetic.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VRINTX":
            return {
                "tooltip": "Round floating-point to integer inexact rounds a floating-point value to an integral floating-point value of the same size, using the rounding mode specified in the FPSCR, and raises an Inexact exception when the result value is not numerically equal to the input value. A zero input gives a zero result with the same sign, an infinite input gives an infinite result with the same sign, and a NaN is propagated as for normal arithmetic.",
                "html": "<p>Round floating-point to integer inexact rounds a floating-point value to an integral floating-point value of the same size, using the rounding mode specified in the FPSCR, and raises an Inexact exception when the result value is not numerically equal to the input value. A zero input gives a zero result with the same sign, an infinite input gives an infinite result with the same sign, and a NaN is propagated as for normal arithmetic.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VRINTZ":
            return {
                "tooltip": "Vector round floating-point to integer towards Zero rounds a vector of floating-point values to integral floating-point values of the same size, using the Round towards Zero rounding mode. A zero input gives a zero result with the same sign, an infinite input gives an infinite result with the same sign, and a NaN is propagated as for normal arithmetic.",
                "html": "<p>Vector round floating-point to integer towards Zero rounds a vector of floating-point values to integral floating-point values of the same size, using the Round towards Zero rounding mode. A zero input gives a zero result with the same sign, an infinite input gives an infinite result with the same sign, and a NaN is propagated as for normal arithmetic.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VRINTZ":
            return {
                "tooltip": "Round floating-point to integer towards Zero rounds a floating-point value to an integral floating-point value of the same size, using the Round towards Zero rounding mode. A zero input gives a zero result with the same sign, an infinite input gives an infinite result with the same sign, and a NaN is propagated as for normal arithmetic.",
                "html": "<p>Round floating-point to integer towards Zero rounds a floating-point value to an integral floating-point value of the same size, using the Round towards Zero rounding mode. A zero input gives a zero result with the same sign, an infinite input gives an infinite result with the same sign, and a NaN is propagated as for normal arithmetic.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VRSHL":
            return {
                "tooltip": "Vector Rounding Shift Left takes each element in a vector, shifts them by a value from the least significant byte of the corresponding element of a second vector, and places the results in the destination vector. If the shift value is positive, the operation is a left shift. If the shift value is negative, it is a rounding right shift. For a truncating shift, see VSHL.",
                "html": "<p>Vector Rounding Shift Left takes each element in a vector, shifts them by a value from the least significant byte of the corresponding element of a second vector, and places the results in the destination vector. If the shift value is positive, the operation is a left shift. If the shift value is negative, it is a rounding right shift. For a truncating shift, see <instruction>VSHL</instruction>.</p><p>The first operand and result elements are the same data type, and can be any one of:</p><p>The second operand is always a signed integer of the same size.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VRSHR":
            return {
                "tooltip": "Vector Rounding Shift Right takes each element in a vector, right shifts them by an immediate value, and places the rounded results in the destination vector. For truncated results, see VSHR.",
                "html": "<p>Vector Rounding Shift Right takes each element in a vector, right shifts them by an immediate value, and places the rounded results in the destination vector. For truncated results, see <xref linkend=\"A32T32-fpsimd.instructions.VSHR\">VSHR</xref>.</p><p>The operand and result elements must be the same size, and can be any one of:</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VRSHRN":
            return {
                "tooltip": "Vector Rounding Shift Right and Narrow takes each element in a vector, right shifts them by an immediate value, and places the rounded results in the destination vector. For truncated results, see VSHRN.",
                "html": "<p>Vector Rounding Shift Right and Narrow takes each element in a vector, right shifts them by an immediate value, and places the rounded results in the destination vector. For truncated results, see <xref linkend=\"A32T32-fpsimd.instructions.VSHRN\">VSHRN</xref>.</p><p>The operand elements can be 16-bit, 32-bit, or 64-bit integers. There is no distinction between signed and unsigned integers. The destination elements are half the size of the source elements.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VRSQRTE":
            return {
                "tooltip": "Vector Reciprocal Square Root Estimate finds an approximate reciprocal square root of each element in a vector, and places the results in a second vector.",
                "html": "<p>Vector Reciprocal Square Root Estimate finds an approximate reciprocal square root of each element in a vector, and places the results in a second vector.</p><p>The operand and result elements are the same type, and can be floating-point numbers or unsigned integers.</p><p>For details of the operation performed by this instruction see <xref linkend=\"CFIJFDFG\">Floating-point reciprocal estimate and step</xref>.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VRSQRTS":
            return {
                "tooltip": "Vector Reciprocal Square Root Step multiplies the elements of one vector by the corresponding elements of another vector, subtracts each of the products from 3.0, divides these results by 2.0, and places the results into the elements of the destination vector.",
                "html": "<p>Vector Reciprocal Square Root Step multiplies the elements of one vector by the corresponding elements of another vector, subtracts each of the products from 3.0, divides these results by 2.0, and places the results into the elements of the destination vector.</p><p>The operand and result elements are floating-point numbers.</p><p>For details of the operation performed by this instruction see <xref linkend=\"CFIJFDFG\">Floating-point reciprocal estimate and step</xref>.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VRSRA":
            return {
                "tooltip": "Vector Rounding Shift Right and Accumulate takes each element in a vector, right shifts them by an immediate value, and accumulates the rounded results into the destination vector. For truncated results, see VSRA.",
                "html": "<p>Vector Rounding Shift Right and Accumulate takes each element in a vector, right shifts them by an immediate value, and accumulates the rounded results into the destination vector. For truncated results, see <xref linkend=\"A32T32-fpsimd.instructions.VSRA\">VSRA</xref>.</p><p>The operand and result elements must all be the same type, and can be any one of:</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VRSUBHN":
            return {
                "tooltip": "Vector Rounding Subtract and Narrow, returning High Half subtracts the elements of one quadword vector from the corresponding elements of another quadword vector, takes the most significant half of each result, and places the final results in a doubleword vector. The results are rounded. For truncated results, see VSUBHN.",
                "html": "<p>Vector Rounding Subtract and Narrow, returning High Half subtracts the elements of one quadword vector from the corresponding elements of another quadword vector, takes the most significant half of each result, and places the final results in a doubleword vector. The results are rounded. For truncated results, see <xref linkend=\"A32T32-fpsimd.instructions.VSUBHN\">VSUBHN</xref>.</p><p>The operand elements can be 16-bit, 32-bit, or 64-bit integers. There is no distinction between signed and unsigned integers.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VSDOT":
            return {
                "tooltip": "Dot Product vector form with signed integers. This instruction performs the dot product of the four 8-bit elements in each 32-bit element of the first source register with the four 8-bit elements of the corresponding 32-bit element in the second source register, accumulating the result into the corresponding 32-bit element of the destination register.",
                "html": "<p>Dot Product vector form with signed integers. This instruction performs the dot product of the four 8-bit elements in each 32-bit element of the first source register with the four 8-bit elements of the corresponding 32-bit element in the second source register, accumulating the result into the corresponding 32-bit element of the destination register.</p><p>In Armv8.2 and Armv8.3, this is an <arm-defined-word>optional</arm-defined-word> instruction. From Armv8.4 it is mandatory for all implementations to support it.</p><p><xref linkend=\"AArch32.id_isar6\">ID_ISAR6</xref>.DP indicates whether this instruction is supported.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VSDOT":
            return {
                "tooltip": "Dot Product index form with signed integers. This instruction performs the dot product of the four 8-bit elements in each 32-bit element of the first source register with the four 8-bit elements of an indexed 32-bit element in the second source register, accumulating the result into the corresponding 32-bit element of the destination register.",
                "html": "<p>Dot Product index form with signed integers. This instruction performs the dot product of the four 8-bit elements in each 32-bit element of the first source register with the four 8-bit elements of an indexed 32-bit element in the second source register, accumulating the result into the corresponding 32-bit element of the destination register.</p><p>In Armv8.2 and Armv8.3, this is an <arm-defined-word>optional</arm-defined-word> instruction. From Armv8.4 it is mandatory for all implementations to support it.</p><p><xref linkend=\"AArch32.id_isar6\">ID_ISAR6</xref>.DP indicates whether this instruction is supported.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VSELEQ":
        case "VSELGE":
        case "VSELGT":
        case "VSELVS":
            return {
                "tooltip": "Floating-point conditional select allows the destination register to take the value in either one or the other source register according to the condition codes in the APSR.",
                "html": "<p>Floating-point conditional select allows the destination register to take the value in either one or the other source register according to the condition codes in the <xref linkend=\"CJAGBHBH\">APSR</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VSHL":
            return {
                "tooltip": "Vector Shift Left (immediate) takes each element in a vector of integers, left shifts them by an immediate value, and places the results in the destination vector.",
                "html": "<p>Vector Shift Left (immediate) takes each element in a vector of integers, left shifts them by an immediate value, and places the results in the destination vector.</p><p>Bits shifted out of the left of each element are lost.</p><p>The elements must all be the same size, and can be 8-bit, 16-bit, 32-bit, or 64-bit integers. There is no distinction between signed and unsigned integers.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VSHL":
            return {
                "tooltip": "Vector Shift Left (register) takes each element in a vector, shifts them by a value from the least significant byte of the corresponding element of a second vector, and places the results in the destination vector. If the shift value is positive, the operation is a left shift. If the shift value is negative, it is a truncating right shift.",
                "html": "<p>Vector Shift Left (register) takes each element in a vector, shifts them by a value from the least significant byte of the corresponding element of a second vector, and places the results in the destination vector. If the shift value is positive, the operation is a left shift. If the shift value is negative, it is a truncating right shift.</p><p>For a rounding shift, see <xref linkend=\"A32T32-fpsimd.instructions.VRSHL\">VRSHL</xref>.</p><p>The first operand and result elements are the same data type, and can be any one of:</p><p>The second operand is always a signed integer of the same size.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VSHLL":
            return {
                "tooltip": "Vector Shift Left Long takes each element in a doubleword vector, left shifts them by an immediate value, and places the results in a quadword vector.",
                "html": "<p>Vector Shift Left Long takes each element in a doubleword vector, left shifts them by an immediate value, and places the results in a quadword vector.</p><p>The operand elements can be:</p><p>The result elements are twice the length of the operand elements.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VSHR":
            return {
                "tooltip": "Vector Shift Right takes each element in a vector, right shifts them by an immediate value, and places the truncated results in the destination vector. For rounded results, see VRSHR.",
                "html": "<p>Vector Shift Right takes each element in a vector, right shifts them by an immediate value, and places the truncated results in the destination vector. For rounded results, see <xref linkend=\"A32T32-fpsimd.instructions.VRSHR\">VRSHR</xref>.</p><p>The operand and result elements must be the same size, and can be any one of:</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VSHRN":
            return {
                "tooltip": "Vector Shift Right Narrow takes each element in a vector, right shifts them by an immediate value, and places the truncated results in the destination vector. For rounded results, see VRSHRN.",
                "html": "<p>Vector Shift Right Narrow takes each element in a vector, right shifts them by an immediate value, and places the truncated results in the destination vector. For rounded results, see <xref linkend=\"A32T32-fpsimd.instructions.VRSHRN\">VRSHRN</xref>.</p><p>The operand elements can be 16-bit, 32-bit, or 64-bit integers. There is no distinction between signed and unsigned integers. The destination elements are half the size of the source elements.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VSLI":
            return {
                "tooltip": "Vector Shift Left and Insert takes each element in the operand vector, left shifts them by an immediate value, and inserts the results in the destination vector. Bits shifted out of the left of each element are lost.",
                "html": "<p>Vector Shift Left and Insert takes each element in the operand vector, left shifts them by an immediate value, and inserts the results in the destination vector. Bits shifted out of the left of each element are lost.</p><p>The elements must all be the same size, and can be 8-bit, 16-bit, 32-bit, or 64-bit. There is no distinction between data types.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VSMMLA":
            return {
                "tooltip": "The widening integer matrix multiply-accumulate instruction multiplies the 2x8 matrix of signed 8-bit integer values held in the first source vector by the 8x2 matrix of signed 8-bit integer values in the second source vector. The resulting 2x2 32-bit integer matrix product is destructively added to the 32-bit integer matrix accumulator held in the destination vector. This is equivalent to performing an 8-way dot product per destination element.",
                "html": "<p>The widening integer matrix multiply-accumulate instruction multiplies the 2x8 matrix of signed 8-bit integer values held in the first source vector by the 8x2 matrix of signed 8-bit integer values in the second source vector. The resulting 2x2 32-bit integer matrix product is destructively added to the 32-bit integer matrix accumulator held in the destination vector. This is equivalent to performing an 8-way dot product per destination element.</p><p>From Armv8.2, this is an <arm-defined-word>optional</arm-defined-word> instruction. <xref linkend=\"AArch32.id_isar6\">ID_ISAR6</xref>.I8MM indicates whether this instruction is supported in the T32 and A32 instruction sets.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VSQRT":
            return {
                "tooltip": "Square Root calculates the square root of the value in a floating-point register and writes the result to another floating-point register.",
                "html": "<p>Square Root calculates the square root of the value in a floating-point register and writes the result to another floating-point register.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, <xref linkend=\"AArch32.hcptr\">HCPTR</xref>, and <xref linkend=\"AArch32.fpexc\">FPEXC</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VSRA":
            return {
                "tooltip": "Vector Shift Right and Accumulate takes each element in a vector, right shifts them by an immediate value, and accumulates the truncated results into the destination vector. For rounded results, see VRSRA.",
                "html": "<p>Vector Shift Right and Accumulate takes each element in a vector, right shifts them by an immediate value, and accumulates the truncated results into the destination vector. For rounded results, see <xref linkend=\"A32T32-fpsimd.instructions.VRSRA\">VRSRA</xref>.</p><p>The operand and result elements must all be the same type, and can be any one of:</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VSRI":
            return {
                "tooltip": "Vector Shift Right and Insert takes each element in the operand vector, right shifts them by an immediate value, and inserts the results in the destination vector. Bits shifted out of the right of each element are lost.",
                "html": "<p>Vector Shift Right and Insert takes each element in the operand vector, right shifts them by an immediate value, and inserts the results in the destination vector. Bits shifted out of the right of each element are lost.</p><p>The elements must all be the same size, and can be 8-bit, 16-bit, 32-bit, or 64-bit. There is no distinction between data types.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VST1":
            return {
                "tooltip": "Store single element from one lane of one register stores one element to memory from one element of a register. For details of the addressing mode, see Advanced SIMD addressing mode.",
                "html": "<p>Store single element from one lane of one register stores one element to memory from one element of a register. For details of the addressing mode, see <xref linkend=\"Cjaefebe\">Advanced SIMD addressing mode</xref>.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information, see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VST1":
            return {
                "tooltip": "Store multiple single elements from one, two, three, or four registers stores elements to memory from one, two, three, or four registers, without interleaving.  Every element of each register is stored. For details of the addressing mode, see Advanced SIMD addressing mode.",
                "html": "<p>Store multiple single elements from one, two, three, or four registers stores elements to memory from one, two, three, or four registers, without interleaving.  Every element of each register is stored. For details of the addressing mode, see <xref linkend=\"Cjaefebe\">Advanced SIMD addressing mode</xref>.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information, see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VST2":
            return {
                "tooltip": "Store single 2-element structure from one lane of two registers stores one 2-element structure to memory from corresponding elements of two registers. For details of the addressing mode, see Advanced SIMD addressing mode.",
                "html": "<p>Store single 2-element structure from one lane of two registers stores one 2-element structure to memory from corresponding elements of two registers. For details of the addressing mode, see <xref linkend=\"Cjaefebe\">Advanced SIMD addressing mode</xref>.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information, see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VST2":
            return {
                "tooltip": "Store multiple 2-element structures from two or four registers stores multiple 2-element structures from two or four registers to memory, with interleaving. For more information, see Element and structure load/store instructions. Every element of each register is saved. For details of the addressing mode, see Advanced SIMD addressing mode.",
                "html": "<p>Store multiple 2-element structures from two or four registers stores multiple 2-element structures from two or four registers to memory, with interleaving. For more information, see <xref linkend=\"BABHJAGF\">Element and structure load/store instructions</xref>. Every element of each register is saved. For details of the addressing mode, see <xref linkend=\"Cjaefebe\">Advanced SIMD addressing mode</xref>.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information, see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VST3":
            return {
                "tooltip": "Store single 3-element structure from one lane of three registers stores one 3-element structure to memory from corresponding elements of three registers. For details of the addressing mode, see Advanced SIMD addressing mode.",
                "html": "<p>Store single 3-element structure from one lane of three registers stores one 3-element structure to memory from corresponding elements of three registers. For details of the addressing mode, see <xref linkend=\"Cjaefebe\">Advanced SIMD addressing mode</xref>.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information, see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VST3":
            return {
                "tooltip": "Store multiple 3-element structures from three registers stores multiple 3-element structures to memory from three registers, with interleaving. For more information, see Element and structure load/store instructions. Every element of each register is saved. For details of the addressing mode, see Advanced SIMD addressing mode.",
                "html": "<p>Store multiple 3-element structures from three registers stores multiple 3-element structures to memory from three registers, with interleaving. For more information, see <xref linkend=\"BABHJAGF\">Element and structure load/store instructions</xref>. Every element of each register is saved. For details of the addressing mode, see <xref linkend=\"Cjaefebe\">Advanced SIMD addressing mode</xref>.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information, see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VST4":
            return {
                "tooltip": "Store single 4-element structure from one lane of four registers stores one 4-element structure to memory from corresponding elements of four registers. For details of the addressing mode, see Advanced SIMD addressing mode.",
                "html": "<p>Store single 4-element structure from one lane of four registers stores one 4-element structure to memory from corresponding elements of four registers. For details of the addressing mode, see <xref linkend=\"Cjaefebe\">Advanced SIMD addressing mode</xref>.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information, see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VST4":
            return {
                "tooltip": "Store multiple 4-element structures from four registers stores multiple 4-element structures to memory from four registers, with interleaving. For more information, see Element and structure load/store instructions. Every element of each register is saved. For details of the addressing mode, see Advanced SIMD addressing mode.",
                "html": "<p>Store multiple 4-element structures from four registers stores multiple 4-element structures to memory from four registers, with interleaving. For more information, see <xref linkend=\"BABHJAGF\">Element and structure load/store instructions</xref>. Every element of each register is saved. For details of the addressing mode, see <xref linkend=\"Cjaefebe\">Advanced SIMD addressing mode</xref>.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information, see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VSTM":
        case "VSTMDB":
        case "VSTMIA":
            return {
                "tooltip": "Store multiple SIMD&FP registers stores multiple registers from the Advanced SIMD and floating-point register file to consecutive memory locations using an address from a general-purpose register.",
                "html": "<p>Store multiple SIMD&amp;FP registers stores multiple registers from the Advanced SIMD and floating-point register file to consecutive memory locations using an address from a general-purpose register.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, <xref linkend=\"AArch32.hcptr\">HCPTR</xref>, and <xref linkend=\"AArch32.fpexc\">FPEXC</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information, see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VSTR":
            return {
                "tooltip": "Store SIMD&FP register stores a single register from the Advanced SIMD and floating-point register file to memory, using an address from a general-purpose register, with an optional offset.",
                "html": "<p>Store SIMD&amp;FP register stores a single register from the Advanced SIMD and floating-point register file to memory, using an address from a general-purpose register, with an optional offset.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, <xref linkend=\"AArch32.hcptr\">HCPTR</xref>, and <xref linkend=\"AArch32.fpexc\">FPEXC</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information, see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VSUB":
            return {
                "tooltip": "Vector Subtract (floating-point) subtracts the elements of one vector from the corresponding elements of another vector, and places the results in the destination vector.",
                "html": "<p>Vector Subtract (floating-point) subtracts the elements of one vector from the corresponding elements of another vector, and places the results in the destination vector.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, <xref linkend=\"AArch32.hcptr\">HCPTR</xref>, and <xref linkend=\"AArch32.fpexc\">FPEXC</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VSUB":
            return {
                "tooltip": "Vector Subtract (integer) subtracts the elements of one vector from the corresponding elements of another vector, and places the results in the destination vector.",
                "html": "<p>Vector Subtract (integer) subtracts the elements of one vector from the corresponding elements of another vector, and places the results in the destination vector.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VSUBHN":
            return {
                "tooltip": "Vector Subtract and Narrow, returning High Half subtracts the elements of one quadword vector from the corresponding elements of another quadword vector, takes the most significant half of each result, and places the final results in a doubleword vector. The results are truncated. For rounded results, see VRSUBHN.",
                "html": "<p>Vector Subtract and Narrow, returning High Half subtracts the elements of one quadword vector from the corresponding elements of another quadword vector, takes the most significant half of each result, and places the final results in a doubleword vector. The results are truncated. For rounded results, see <xref linkend=\"A32T32-fpsimd.instructions.VRSUBHN\">VRSUBHN</xref>.</p><p>There is no distinction between signed and unsigned integers.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VSUBL":
            return {
                "tooltip": "Vector Subtract Long subtracts the elements of one doubleword vector from the corresponding elements of another doubleword vector, and places the results in a quadword vector. Before subtracting, it sign-extends or zero-extends the elements of both operands.",
                "html": "<p>Vector Subtract Long subtracts the elements of one doubleword vector from the corresponding elements of another doubleword vector, and places the results in a quadword vector. Before subtracting, it sign-extends or zero-extends the elements of both operands.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VSUBW":
            return {
                "tooltip": "Vector Subtract Wide subtracts the elements of a doubleword vector from the corresponding elements of a quadword vector, and places the results in another quadword vector. Before subtracting, it sign-extends or zero-extends the elements of the doubleword operand.",
                "html": "<p>Vector Subtract Wide subtracts the elements of a doubleword vector from the corresponding elements of a quadword vector, and places the results in another quadword vector. Before subtracting, it sign-extends or zero-extends the elements of the doubleword operand.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VSUDOT":
            return {
                "tooltip": "Dot Product index form with signed and unsigned integers. This instruction performs the dot product of the four signed 8-bit integer values in each 32-bit element of the first source register with the four unsigned 8-bit integer values in an indexed 32-bit element of the second source register, accumulating the result into the corresponding 32-bit element of the destination register.",
                "html": "<p>Dot Product index form with signed and unsigned integers. This instruction performs the dot product of the four signed 8-bit integer values in each 32-bit element of the first source register with the four unsigned 8-bit integer values in an indexed 32-bit element of the second source register, accumulating the result into the corresponding 32-bit element of the destination register.</p><p>From Armv8.2, this is an <arm-defined-word>optional</arm-defined-word> instruction. <xref linkend=\"AArch32.id_isar6\">ID_ISAR6</xref>.I8MM indicates whether this instruction is supported in the T32 and A32 instruction sets.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VSWP":
            return {
                "tooltip": "Vector Swap exchanges the contents of two vectors. The vectors can be either doubleword or quadword. There is no distinction between data types.",
                "html": "<p>Vector Swap exchanges the contents of two vectors. The vectors can be either doubleword or quadword. There is no distinction between data types.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VTBL":
        case "VTBX":
            return {
                "tooltip": "Vector Table Lookup uses byte indexes in a control vector to look up byte values in a table and generate a new vector. Indexes out of range return 0.",
                "html": "<p>Vector Table Lookup uses byte indexes in a control vector to look up byte values in a table and generate a new vector. Indexes out of range return 0.</p><p>Vector Table Extension works in the same way, except that indexes out of range leave the destination element unchanged.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VTRN":
            return {
                "tooltip": "Vector Transpose treats the elements of its operand vectors as elements of 2 x 2 matrices, and transposes the matrices.",
                "html": "<p>Vector Transpose treats the elements of its operand vectors as elements of 2 x 2 matrices, and transposes the matrices.</p><p>The elements of the vectors can be 8-bit, 16-bit, or 32-bit. There is no distinction between data types.</p><p></p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VTST":
            return {
                "tooltip": "Vector Test Bits takes each element in a vector, and bitwise ANDs it with the corresponding element of a second vector. If the result is not zero, the corresponding element in the destination vector is set to all ones. Otherwise, it is set to all zeros.",
                "html": "<p>Vector Test Bits takes each element in a vector, and bitwise ANDs it with the corresponding element of a second vector. If the result is not zero, the corresponding element in the destination vector is set to all ones. Otherwise, it is set to all zeros.</p><p>The operand vector elements can be any one of:</p><p>The result vector elements are fields the same size as the operand vector elements.</p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VUDOT":
            return {
                "tooltip": "Dot Product vector form with unsigned integers. This instruction performs the dot product of the four 8-bit elements in each 32-bit element of the first source register with the four 8-bit elements of the corresponding 32-bit element in the second source register, accumulating the result into the corresponding 32-bit element of the destination register.",
                "html": "<p>Dot Product vector form with unsigned integers. This instruction performs the dot product of the four 8-bit elements in each 32-bit element of the first source register with the four 8-bit elements of the corresponding 32-bit element in the second source register, accumulating the result into the corresponding 32-bit element of the destination register.</p><p>In Armv8.2 and Armv8.3, this is an <arm-defined-word>optional</arm-defined-word> instruction. From Armv8.4 it is mandatory for all implementations to support it.</p><p><xref linkend=\"AArch32.id_isar6\">ID_ISAR6</xref>.DP indicates whether this instruction is supported.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VUDOT":
            return {
                "tooltip": "Dot Product index form with unsigned integers. This instruction performs the dot product of the four 8-bit elements in each 32-bit element of the first source register with the four 8-bit elements of an indexed 32-bit element in the second source register, accumulating the result into the corresponding 32-bit element of the destination register.",
                "html": "<p>Dot Product index form with unsigned integers. This instruction performs the dot product of the four 8-bit elements in each 32-bit element of the first source register with the four 8-bit elements of an indexed 32-bit element in the second source register, accumulating the result into the corresponding 32-bit element of the destination register.</p><p>In Armv8.2 and Armv8.3, this is an <arm-defined-word>optional</arm-defined-word> instruction. From Armv8.4 it is mandatory for all implementations to support it.</p><p><xref linkend=\"AArch32.id_isar6\">ID_ISAR6</xref>.DP indicates whether this instruction is supported.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VUMMLA":
            return {
                "tooltip": "The widening integer matrix multiply-accumulate instruction multiplies the 2x8 matrix of unsigned 8-bit integer values held in the first source vector by the 8x2 matrix of unsigned 8-bit integer values in the second source vector. The resulting 2x2 32-bit integer matrix product is destructively added to the 32-bit integer matrix accumulator held in the destination vector. This is equivalent to performing an 8-way dot product per destination element.",
                "html": "<p>The widening integer matrix multiply-accumulate instruction multiplies the 2x8 matrix of unsigned 8-bit integer values held in the first source vector by the 8x2 matrix of unsigned 8-bit integer values in the second source vector. The resulting 2x2 32-bit integer matrix product is destructively added to the 32-bit integer matrix accumulator held in the destination vector. This is equivalent to performing an 8-way dot product per destination element.</p><p>From Armv8.2, this is an <arm-defined-word>optional</arm-defined-word> instruction. <xref linkend=\"AArch32.id_isar6\">ID_ISAR6</xref>.I8MM indicates whether this instruction is supported in the T32 and A32 instruction sets.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VUSDOT":
            return {
                "tooltip": "Dot Product vector form with mixed-sign integers. This instruction performs the dot product of the four unsigned 8-bit integer values in each 32-bit element of the first source register with the four signed 8-bit integer values in the corresponding 32-bit element of the second source register, accumulating the result into the corresponding 32-bit element of the destination register.",
                "html": "<p>Dot Product vector form with mixed-sign integers. This instruction performs the dot product of the four unsigned 8-bit integer values in each 32-bit element of the first source register with the four signed 8-bit integer values in the corresponding 32-bit element of the second source register, accumulating the result into the corresponding 32-bit element of the destination register.</p><p>From Armv8.2, this is an <arm-defined-word>optional</arm-defined-word> instruction. <xref linkend=\"AArch32.id_isar6\">ID_ISAR6</xref>.I8MM indicates whether this instruction is supported in the T32 and A32 instruction sets.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VUSDOT":
            return {
                "tooltip": "Dot Product index form with unsigned and signed integers. This instruction performs the dot product of the four unsigned 8-bit integer values in each 32-bit element of the first source register with the four signed 8-bit integer values in an indexed 32-bit element of the second source register, accumulating the result into the corresponding 32-bit element of the destination register.",
                "html": "<p>Dot Product index form with unsigned and signed integers. This instruction performs the dot product of the four unsigned 8-bit integer values in each 32-bit element of the first source register with the four signed 8-bit integer values in an indexed 32-bit element of the second source register, accumulating the result into the corresponding 32-bit element of the destination register.</p><p>From Armv8.2, this is an <arm-defined-word>optional</arm-defined-word> instruction. <xref linkend=\"AArch32.id_isar6\">ID_ISAR6</xref>.I8MM indicates whether this instruction is supported in the T32 and A32 instruction sets.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VUSMMLA":
            return {
                "tooltip": "The widening integer matrix multiply-accumulate instruction multiplies the 2x8 matrix of unsigned 8-bit integer values held in the first source vector by the 8x2 matrix of signed 8-bit integer values in the second source vector. The resulting 2x2 32-bit integer matrix product is destructively added to the 32-bit integer matrix accumulator held in the destination vector. This is equivalent to performing an 8-way dot product per destination element.",
                "html": "<p>The widening integer matrix multiply-accumulate instruction multiplies the 2x8 matrix of unsigned 8-bit integer values held in the first source vector by the 8x2 matrix of signed 8-bit integer values in the second source vector. The resulting 2x2 32-bit integer matrix product is destructively added to the 32-bit integer matrix accumulator held in the destination vector. This is equivalent to performing an 8-way dot product per destination element.</p><p>From Armv8.2, this is an <arm-defined-word>optional</arm-defined-word> instruction. <xref linkend=\"AArch32.id_isar6\">ID_ISAR6</xref>.I8MM indicates whether this instruction is supported in the T32 and A32 instruction sets.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VUZP":
            return {
                "tooltip": "Vector Unzip de-interleaves the elements of two vectors.",
                "html": "<p>Vector Unzip de-interleaves the elements of two vectors.</p><p>The elements of the vectors can be 8-bit, 16-bit, or 32-bit. There is no distinction between data types.</p><p></p><p></p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "VZIP":
            return {
                "tooltip": "Vector Zip interleaves the elements of two vectors.",
                "html": "<p>Vector Zip interleaves the elements of two vectors.</p><p>The elements of the vectors can be 8-bit, 16-bit, or 32-bit. There is no distinction between data types.</p><p></p><p></p><p>Depending on settings in the <xref linkend=\"AArch32.cpacr\">CPACR</xref>, <xref linkend=\"AArch32.nsacr\">NSACR</xref>, and <xref linkend=\"AArch32.hcptr\">HCPTR</xref> registers, and the Security state and PE mode in which the instruction is executed, an attempt to execute the instruction might be <arm-defined-word>undefined</arm-defined-word>, or trapped to Hyp mode. For more information see <xref linkend=\"CIHIDDFF\">Enabling Advanced SIMD and floating-point support</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "WFE":
            return {
                "tooltip": "Wait For Event is a hint instruction that indicates that the PE can enter a low-power state and remain there until a wakeup event occurs. Wakeup events include the event signaled as a result of executing the SEV instruction on any PE in the multiprocessor system. For more information, see Wait For Event and Send Event.",
                "html": "<p>Wait For Event is a hint instruction that indicates that the PE can enter a low-power state and remain there until a wakeup event occurs. Wakeup events include the event signaled as a result of executing the <instruction>SEV</instruction> instruction on any PE in the multiprocessor system. For more information, see <xref linkend=\"CFIJIIHE\">Wait For Event and Send Event</xref>.</p><p>As described in <xref linkend=\"CFIJIIHE\">Wait For Event and Send Event</xref>, the execution of a <instruction>WFE</instruction> instruction that would otherwise cause entry to a low-power state can be trapped to a higher Exception level, see:</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "WFI":
            return {
                "tooltip": "Wait For Interrupt is a hint instruction that indicates that the PE can enter a low-power state and remain there until a wakeup event occurs. For more information, see Wait For Interrupt.",
                "html": "<p>Wait For Interrupt is a hint instruction that indicates that the PE can enter a low-power state and remain there until a wakeup event occurs. For more information, see <xref linkend=\"CFIBBGJG\">Wait For Interrupt</xref>.</p><p>As described in <xref linkend=\"CFIBBGJG\">Wait For Interrupt</xref>, the execution of a <instruction>WFI</instruction> instruction that would otherwise cause entry to a low-power state can be trapped to a higher Exception level, see:</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };

        case "YIELD":
            return {
                "tooltip": "YIELD is a hint instruction. Software with a multithreading capability can use a YIELD instruction to indicate to the PE that it is performing a task, for example a spin-lock, that could be swapped out to improve overall system performance. The PE can use this hint to suspend and resume multiple software threads if it supports the capability.",
                "html": "<p><instruction>YIELD</instruction> is a hint instruction. Software with a multithreading capability can use a <instruction>YIELD</instruction> instruction to indicate to the PE that it is performing a task, for example a spin-lock, that could be swapped out to improve overall system performance. The PE can use this hint to suspend and resume multiple software threads if it supports the capability.</p><p>For more information about the recommended use of this instruction see <xref linkend=\"CFIGJJJG\">The Yield instruction</xref>.</p>",
                "url": "https://developer.arm.com/documentation/ddi0597/latest/Base-Instructions/"
            };


    }
}
