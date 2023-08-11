import {AssemblyInstructionInfo} from '../base.js';

export function getAsmOpcode(opcode: string | undefined): AssemblyInstructionInfo | undefined {
    if (!opcode) return;
    switch (opcode.toUpperCase()) {
        case "ADC":
            return {
                "html": "<p>Adds two registers and the contents of the C flag and places the result in the destination register Rd.</p>",
                "tooltip": "Add with Carry",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=24",
            };

        case "ADD":
            return {
                "html": "<p>Adds two registers without the C flag and places the result in the destination register Rd.</p>",
                "tooltip": "Add without Carry",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=25",
            };

        case "ADIW":
            return {
                "html": "<p>Adds an immediate value (0-63) to a register pair and places the result in the register pair. This instruction operates on the upper four register pairs and is well suited for operations on the Pointer Registers.</p><p>This instruction is not available on all devices. Refer to Appendix A.</p>",
                "tooltip": "Add Immediate to Word",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=26",
            };

        case "AND":
            return {
                "html": "<p>Performs the logical AND between the contents of register Rd and register Rr, and places the result in the destination register Rd.</p>",
                "tooltip": "Logical AND",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=27",
            };

        case "ANDI":
            return {
                "html": "<p>Performs the logical AND between the contents of register Rd and a constant, and places the result in the destination register Rd.</p>",
                "tooltip": "Logical AND with Immediate",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=28",
            };

        case "ASR":
            return {
                "html": "<p>Shifts all bits in Rd one place to the right. Bit 7 is held constant. Bit 0 is loaded into the C flag of the SREG. This operation effectively divides a signed value by two without changing its sign. The Carry flag can be used to round the result.</p>",
                "tooltip": "Arithmetic Shift Right",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=29",
            };

        case "BCLR":
            return {
                "html": "<p>Clears a single flag in SREG.</p>",
                "tooltip": "Bit Clear in SREG",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=30",
            };

        case "BLD":
            return {
                "html": "<p>Copies the T bit in the SREG (Status Register) to bit b in register Rd.</p>",
                "tooltip": "Bit Load from the T Bit in SREG to a Bit in Register",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=31",
            };

        case "BRBC":
            return {
                "html": "<p>Conditional relative branch. Tests a single bit in SREG and branches relatively to the PC if the bit is cleared. This instruction branches relatively to the PC in either direction (PC - 63 ≤ destination ≤ PC + 64). Parameter k is the offset from the PC and is represented in two’s complement form.</p>",
                "tooltip": "Branch if Bit in SREG is Cleared",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=32",
            };

        case "BRBS":
            return {
                "html": "<p>Conditional relative branch. Tests a single bit in SREG and branches relatively to the PC if the bit is set. This instruction branches relatively to the PC in either direction (PC - 63 ≤ destination ≤ PC + 64). Parameter k is the offset from the PC and is represented in two’s complement form.</p>",
                "tooltip": "Branch if Bit in SREG is Set",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=33",
            };

        case "BRCC":
            return {
                "html": "<p>Conditional relative branch. Tests the Carry (C) flag and branches relatively to the PC if C is cleared. This instruction branches relatively to the PC in either direction (PC - 63 ≤ destination ≤ PC + 64). Parameter k is the offset from the PC and is represented in two’s complement form. (Equivalent to instruction BRBC 0,k.)</p>",
                "tooltip": "Branch if Carry Cleared",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=34",
            };

        case "BRCS":
            return {
                "html": "<p>Conditional relative branch. Tests the Carry (C) flag and branches relatively to the PC if C is set. This instruction branches relatively to the PC in either direction (PC - 63 ≤ destination ≤ PC + 64). Parameter k is the offset from the PC and is represented in two’s complement form. (Equivalent to instruction BRBS 0,k.)</p>",
                "tooltip": "Branch if Carry Set",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=35",
            };

        case "BREAK":
            return {
                "html": "<p>The BREAK instruction is used by the On-chip Debug system and not used by the application software. When the BREAK instruction is executed, the AVR CPU is set in the Stopped state. This gives the On-chip Debugger access to internal resources.</p><p>If the device is locked, or the on-chip debug system is not enabled, the CPU will treat the BREAK instruction as a NOP and will not enter the Stopped state.</p><p>This instruction is not available on all devices. Refer to Appendix A.</p>",
                "tooltip": "Break",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=36",
            };

        case "BREQ":
            return {
                "html": "<p>Conditional relative branch. Tests the Zero (Z) flag and branches relatively to the PC if Z is set. If the instruction is executed immediately after any of the instructions CP, CPI, SUB, or SUBI, the branch will occur only if the unsigned or signed binary number represented in Rd was equal to the unsigned or signed binary number represented in Rr. This instruction branches relatively to the PC in either direction (PC - 63 ≤ destination ≤ PC + 64). Parameter k is the offset from the PC and is represented in two’s complement form. (Equivalent to instruction BRBS 1,k.)</p>",
                "tooltip": "Branch if Equal",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=36",
            };

        case "BRGE":
            return {
                "html": "<p>Conditional relative branch. Tests the Sign (S) flag and branches relatively to the PC if S is cleared. If the instruction is executed immediately after any of the instructions CP, CPI, SUB, or SUBI, the branch will occur only if the signed binary number represented in Rd was greater than or equal to the signed binary number represented in Rr. This instruction branches relatively to the PC in either direction (PC - 63 ≤ destination ≤ PC + 64). Parameter k is the offset from the PC and is represented in two’s complement form. (Equivalent to instruction BRBC 4,k.)</p>",
                "tooltip": "Branch if Greater or Equal (Signed)",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=37",
            };

        case "BRHC":
            return {
                "html": "<p>Conditional relative branch. Tests the Half Carry (H) flag and branches relatively to the PC if H is cleared. This instruction branches relatively to the PC in either direction (PC - 63 ≤ destination ≤ PC + 64). Parameter k is the offset from the PC and is represented in two’s complement form. (Equivalent to instruction BRBC 5,k.)</p>",
                "tooltip": "Branch if Half Carry Flag is Cleared",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=38",
            };

        case "BRHS":
            return {
                "html": "<p>Conditional relative branch. Tests the Half Carry (H) flag and branches relatively to the PC if H is set. This instruction branches relatively to the PC in either direction (PC - 63 ≤ destination ≤ PC + 64). Parameter k is the offset from the PC and is represented in two’s complement form. (Equivalent to instruction BRBS 5,k.)</p>",
                "tooltip": "Branch if Half Carry Flag is Set",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=39",
            };

        case "BRID":
            return {
                "html": "<p>Conditional relative branch. Tests the Global Interrupt Enable (I) bit and branches relatively to the PC if I is cleared. This instruction branches relatively to the PC in either direction (PC - 63 ≤ destination ≤ PC + 64). Parameter k is the offset from the PC and is represented in two’s complement form. (Equivalent to instruction BRBC 7,k.)</p>",
                "tooltip": "Branch if Global Interrupt is Disabled",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=40",
            };

        case "BRIE":
            return {
                "html": "<p>Conditional relative branch. Tests the Global Interrupt Enable (I) bit and branches relatively to the PC if I is set. This instruction branches relatively to the PC in either direction (PC - 63 ≤ destination ≤ PC + 64). Parameter k is the offset from the PC and is represented in two’s complement form. (Equivalent to instruction BRBS 7,k.)</p>",
                "tooltip": "Branch if Global Interrupt is Enabled",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=41",
            };

        case "BRLO":
            return {
                "html": "<p>Conditional relative branch. Tests the Carry (C) flag and branches relatively to the PC if C is set. If the instruction is executed immediately after any of the instructions CP, CPI, SUB, or SUBI, the branch will occur only if the unsigned binary number represented in Rd was smaller than the unsigned binary number represented in Rr. This instruction branches relatively to the PC in either direction (PC - 63 ≤ destination ≤ PC + 64). Parameter k is the offset from the PC and is represented in two’s complement form. (Equivalent to instruction BRBS 0,k.)</p>",
                "tooltip": "Branch if Lower (Unsigned)",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=42",
            };

        case "BRLT":
            return {
                "html": "<p>Conditional relative branch. Tests the Sign (S) flag and branches relatively to the PC if S is set. If the instruction is executed immediately after any of the instructions CP, CPI, SUB, or SUBI, the branch will occur only if the signed binary number represented in Rd was less than the signed binary number represented in Rr. This instruction branches relatively to the PC in either direction (PC - 63 ≤ destination ≤ PC + 64). Parameter k is the offset from the PC and is represented in two’s complement form. (Equivalent to instruction BRBS 4,k.)</p>",
                "tooltip": "Branch if Less Than (Signed)",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=43",
            };

        case "BRMI":
            return {
                "html": "<p>Conditional relative branch. Tests the Negative (N) flag and branches relatively to the PC if N is set. This instruction branches relatively to the PC in either direction (PC - 63 ≤ destination ≤ PC + 64). Parameter k is the offset from the PC and is represented in two’s complement form. (Equivalent to instruction BRBS 2,k.)</p>",
                "tooltip": "Branch if Minus",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=44",
            };

        case "BRNE":
            return {
                "html": "<p>Conditional relative branch. Tests the Zero (Z) flag and branches relatively to the PC if Z is cleared. If the instruction is executed immediately after any of the instructions CP, CPI, SUB, or SUBI, the branch will occur only if the unsigned or signed binary number represented in Rd was not equal to the unsigned or signed binary number represented in Rr. This instruction branches relatively to the PC in either direction (PC - 63 ≤ destination ≤ PC + 64). Parameter k is the offset from the PC and is represented in two’s complement form. (Equivalent to instruction BRBC 1,k.)</p>",
                "tooltip": "Branch if Not Equal",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=45",
            };

        case "BRPL":
            return {
                "html": "<p>Conditional relative branch. Tests the Negative (N) flag and branches relatively to the PC if N is cleared. This instruction branches relatively to the PC in either direction (PC - 63 ≤ destination ≤ PC + 64). Parameter k is the offset from the PC and is represented in two’s complement form. (Equivalent to instruction BRBC 2,k.)</p>",
                "tooltip": "Branch if Plus",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=46",
            };

        case "BRSH":
            return {
                "html": "<p>Conditional relative branch. Tests the Carry (C) flag and branches relatively to the PC if C is cleared. If the instruction is executed immediately after execution of any of the instructions CP, CPI, SUB, or SUBI, the branch will occur only if the unsigned binary number represented in Rd was greater than or equal to the unsigned binary number represented in Rr. This instruction branches relatively to the PC in either direction (PC - 63 ≤ destination ≤ PC + 64). Parameter k is the offset from the PC and is represented in two’s complement form. (Equivalent to instruction BRBC 0,k.)</p>",
                "tooltip": "Branch if Same or Higher (Unsigned)",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=47",
            };

        case "BRTC":
            return {
                "html": "<p>Conditional relative branch. Tests the T bit and branches relatively to the PC if T is cleared. This instruction branches relatively to the PC in either direction (PC - 63 ≤ destination ≤ PC + 64). Parameter k is the offset from the PC and is represented in two’s complement form. (Equivalent to instruction BRBC 6,k.)</p>",
                "tooltip": "Branch if the T Bit is Cleared",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=48",
            };

        case "BRTS":
            return {
                "html": "<p>Conditional relative branch. Tests the T bit and branches relatively to the PC if T is set. This instruction branches relatively to the PC in either direction (PC - 63 ≤ destination ≤ PC + 64). Parameter k is the offset from the PC and is represented in two’s complement form. (Equivalent to instruction BRBS 6,k.)</p>",
                "tooltip": "Branch if the T Bit is Set",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=49",
            };

        case "BRVC":
            return {
                "html": "<p>Conditional relative branch. Tests the Overflow (V) flag and branches relatively to the PC if V is cleared. This instruction branches relatively to the PC in either direction (PC - 63 ≤ destination ≤ PC + 64). Parameter k is the offset from the PC and is represented in two’s complement form. (Equivalent to instruction BRBC 3,k.)</p>",
                "tooltip": "Branch if Overflow Cleared",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=50",
            };

        case "BRVS":
            return {
                "html": "<p>Conditional relative branch. Tests the Overflow (V) flag and branches relatively to the PC if V is set. This instruction branches relatively to the PC in either direction (PC - 63 ≤ destination ≤ PC + 64). Parameter k is the offset from the PC and is represented in two’s complement form. (Equivalent to instruction BRBS 3,k.)</p>",
                "tooltip": "Branch if Overflow Set",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=51",
            };

        case "BSET":
            return {
                "html": "<p>Sets a single flag or bit in SREG.</p>",
                "tooltip": "Bit Set in SREG",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=52",
            };

        case "BST":
            return {
                "html": "<p>Stores bit b from Rd to the T bit in SREG (Status Register).</p>",
                "tooltip": "Bit Store from Bit in Register to T Bit in SREG",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=53",
            };

        case "CALL":
            return {
                "html": "<p>Calls to a subroutine within the entire program memory. The return address (to the instruction after the CALL) will be stored on the Stack. (See also RCALL.) The Stack Pointer uses a post-decrement scheme during CALL.</p><p>This instruction is not available on all devices. Refer to Appendix A.</p>",
                "tooltip": "Long Call to a Subroutine",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=54",
            };

        case "CBI":
            return {
                "html": "<p>Clears a specified bit in an I/O Register. This instruction operates on the lower 32 I/O Registers – addresses 0-31.</p>",
                "tooltip": "Clear Bit in I/O Register",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=55",
            };

        case "CBR":
            return {
                "html": "<p>Clears the specified bits in register Rd. Performs the logical AND between the contents of register Rd and the complement of the constant mask K. The result will be placed in register Rd. (Equivalent to ANDI Rd,(0xFF - K).)</p>",
                "tooltip": "Clear Bits in Register",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=56",
            };

        case "CLC":
            return {
                "html": "<p>Clears the Carry (C) flag in SREG (Status Register). (Equivalent to instruction BCLR 0.)</p>",
                "tooltip": "Clear Carry Flag",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=57",
            };

        case "CLH":
            return {
                "html": "<p>Clears the Half Carry (H) flag in SREG (Status Register). (Equivalent to instruction BCLR 5.)</p>",
                "tooltip": "Clear Half Carry Flag",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=57",
            };

        case "CLI":
            return {
                "html": "<p>Clears the Global Interrupt Enable (I) bit in SREG (Status Register). The interrupts will be immediately disabled. No interrupt will be executed after the CLI instruction, even if it occurs simultaneously with the CLI instruction. (Equivalent to instruction BCLR 7.)</p>",
                "tooltip": "Clear Global Interrupt Enable Bit",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=58",
            };

        case "CLN":
            return {
                "html": "<p>Clears the Negative (N) flag in SREG (Status Register). (Equivalent to instruction BCLR 2.)</p>",
                "tooltip": "Clear Negative Flag",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=59",
            };

        case "CLR":
            return {
                "html": "<p>Clears a register. This instruction performs an Exclusive OR between a register and itself. This will clear all bits in the register. (Equivalent to instruction EOR Rd,Rd.)</p>",
                "tooltip": "Clear Register",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=60",
            };

        case "CLS":
            return {
                "html": "<p>Clears the Sign (S) flag in SREG (Status Register). (Equivalent to instruction BCLR 4.)</p>",
                "tooltip": "Clear Sign Flag",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=61",
            };

        case "CLT":
            return {
                "html": "<p>Clears the T bit in SREG (Status Register). (Equivalent to instruction BCLR 6.)</p>",
                "tooltip": "Clear T Bit",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=62",
            };

        case "CLV":
            return {
                "html": "<p>Clears the Overflow (V) flag in SREG (Status Register). (Equivalent to instruction BCLR 3.)</p>",
                "tooltip": "Clear Overflow Flag",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=62",
            };

        case "CLZ":
            return {
                "html": "<p>Clears the Zero (Z) flag in SREG (Status Register). (Equivalent to instruction BCLR 1.)</p>",
                "tooltip": "Clear Zero Flag",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=63",
            };

        case "COM":
            return {
                "html": "<p>This instruction performs a One’s Complement of register Rd.</p>",
                "tooltip": "One’s Complement",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=64",
            };

        case "CP":
            return {
                "html": "<p>This instruction performs a compare between two registers Rd and Rr. None of the registers are changed. All conditional branches can be used after this instruction.</p>",
                "tooltip": "Compare",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=65",
            };

        case "CPC":
            return {
                "html": "<p>This instruction performs a compare between two registers Rd and Rr and also takes into account the previous carry. None of the registers are changed. All conditional branches can be used after this instruction.</p>",
                "tooltip": "Compare with Carry",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=66",
            };

        case "CPI":
            return {
                "html": "<p>This instruction performs a compare between register Rd and a constant. The register is not changed. All conditional branches can be used after this instruction.</p>",
                "tooltip": "Compare with Immediate",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=67",
            };

        case "CPSE":
            return {
                "html": "<p>This instruction performs a compare between two registers Rd and Rr and skips the next instruction if Rd == Rr.</p>",
                "tooltip": "Compare Skip if Equal",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=68",
            };

        case "DEC":
            return {
                "html": "<p>Subtracts one -1- from the contents of register Rd and places the result in the destination register Rd.</p><p>The C flag in SREG is not affected by the operation, thus allowing the DEC instruction to be used on a loop counter in multiple-precision computations.</p><p>When operating on unsigned values, only BREQ and BRNE branches can be expected to perform consistently. When operating on two’s complement values, all signed branches are available.</p>",
                "tooltip": "Decrement",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=69",
            };

        case "DES":
            return {
                "html": "<p>The module is an instruction set extension to the AVR CPU, performing DES iterations. The 64-bit data block (plaintext or ciphertext) is placed in the CPU Register File, registers R0-R7, where the LSB of data is placed in the LSB of R0 and the MSB of data is placed in the MSB of R7. The full 64-bit key (including parity bits) is placed in registers R8-R15, organized in the Register File with the LSB of the key in the LSB of R8 and the MSB of the key in the MSB of R15. Executing one DES instruction performs one round in the DES algorithm. Sixteen rounds must be executed in increasing order to form the correct DES ciphertext or plaintext. Intermediate results are stored in the Register File (R0-R15) after each DES instruction. The instruction's operand (K) determines which round is executed, and the Half Carry (H) flag determines whether encryption or decryption is performed.</p><p>The DES algorithm is described in “Specifications for the Data Encryption Standard” (Federal Information Processing Standards Publication 46). Intermediate results in this implementation differ from the standard because the initial permutation and the inverse initial permutation are performed in each iteration. This does not affect the result in the final ciphertext or plaintext but reduces the execution time.</p>",
                "tooltip": "Data Encryption Standard",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=71",
            };

        case "EICALL":
            return {
                "html": "<p>Indirect call of a subroutine pointed to by the Z (16-bit) Pointer Register in the Register File and the EIND Register in the I/O space. This instruction allows for indirect calls to the entire 4M (words) program memory space. See also ICALL. The Stack Pointer uses a post-decrement scheme during EICALL.</p><p>This instruction is not available on all devices. Refer to Appendix A.</p>",
                "tooltip": "Extended Indirect Call to Subroutine",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=72",
            };

        case "EIJMP":
            return {
                "html": "<p>Indirect jump to the address pointed to by the Z (16-bit) Pointer Register in the Register File and the EIND Register in the I/O space. This instruction allows for indirect jumps to the entire 4M (words) program memory space. See also IJMP.</p><p>This instruction is not available on all devices. Refer to Appendix A.</p>",
                "tooltip": "Extended Indirect Jump",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=73",
            };

        case "ELPM":
            return {
                "html": "<p>Loads one byte pointed to by the Z-register and the RAMPZ Register in the I/O space, and places this byte in the destination register Rd. This instruction features a 100% space-effective constant initialization or constant data fetch. The program memory is organized in 16-bit words while the Z-pointer is a byte address. Thus, the least significant bit of the Z-pointer selects either low byte (ZLSB == 0) or high byte (ZLSB == 1). This instruction can address theentire program memory space. The Z-Pointer Register can either be left unchanged by the operation, or it can be incremented. The incrementation applies to the entire 24-bit concatenation of the RAMPZ and Z-Pointer Registers.</p><p>Devices with self-programming capability can use the ELPM instruction to read the Fuse and Lock bit value. Refer to the device documentation for a detailed description.</p><p>This instruction is not available on all devices. Refer to Appendix A.</p><p>The result of these combinations is undefined:</p><p>ELPM r30, Z+</p><p>ELPM r31, Z+</p>",
                "tooltip": "Extended Load Program Memory",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=73",
            };

        case "EOR":
            return {
                "html": "<p>Performs the logical EOR between the contents of register Rd and register Rr and places the result in the destination register Rd.</p>",
                "tooltip": "Exclusive OR",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=75",
            };

        case "FMUL":
            return {
                "html": "<p>This instruction performs 8-bit × 8-bit → 16-bit unsigned multiplication and shifts the result one bit left.</p><p>Let (N.Q) denote a fractional number with N binary digits left of the radix point, and Q binary digits right of the radix point. A multiplication between two numbers in the formats (N1.Q1) and (N2.Q2) results in the format ((N1+N2). (Q1+Q2)). For signal processing applications, the format (1.7) is widely used for the inputs, resulting in a (2.14) format for the product. A left shift is required for the high byte of the product to be in the same format as the inputs. The FMUL instruction incorporates the shift operation in the same number of cycles as MUL.</p><p>The (1.7) format is most commonly used with signed numbers, while FMUL performs an unsigned multiplication. This instruction is, therefore, most useful for calculating one of the partial products when performing a signed multiplication with 16-bit inputs in the (1.15) format, yielding a result in the (1.31) format.</p><p>Note:  The result of the FMUL operation may suffer from a 2’s complement overflow if interpreted as a number in the (1.15) format. The MSB of the multiplication before shifting must be taken into account and is found in the carry bit. See the following example.</p><p>The multiplicand Rd and the multiplier Rr are two registers containing unsigned fractional numbers where the implicit radix point lies between bit 6 and bit 7. The 16-bit unsigned fractional product with the implicit radix point between bit 14 and bit 15 is placed in R1 (high byte) and R0 (low byte).</p><p>This instruction is not available on all devices. Refer to Appendix A.</p>",
                "tooltip": "Fractional Multiply Unsigned",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=76",
            };

        case "FMULS":
            return {
                "html": "<p>This instruction performs 8-bit × 8-bit → 16-bit signed multiplication and shifts the result one bit left.</p><p>Let (N.Q) denote a fractional number with N binary digits left of the radix point, and Q binary digits right of the radix point. A multiplication between two numbers in the formats (N1.Q1) and (N2.Q2) results in the format ((N1+N2). (Q1+Q2)). For signal processing applications, the format (1.7) is widely used for the inputs, resulting in a (2.14)format for the product. A left shift is required for the high byte of the product to be in the same format as the inputs. The FMULS instruction incorporates the shift operation in the same number of cycles as MULS.</p><p>The multiplicand Rd and the multiplier Rr are two registers containing signed fractional numbers where the implicit radix point lies between bit 6 and bit 7. The 16-bit signed fractional product with the implicit radix point between bit 14 and bit 15 is placed in R1 (high byte) and R0 (low byte).</p><p>Note:  That when multiplying 0x80 (-1) with 0x80 (-1), the result of the shift operation is 0x8000 (-1). The shift operation thus gives a two’s complement overflow. This must be checked and handled by software.</p><p>This instruction is not available on all devices. Refer to Appendix A.</p>",
                "tooltip": "Fractional Multiply Signed",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=77",
            };

        case "FMULSU":
            return {
                "html": "<p>This instruction performs 8-bit × 8-bit → 16-bit signed multiplication and shifts the result one bit left.</p><p>Let (N.Q) denote a fractional number with N binary digits left of the radix point, and Q binary digits right of the radix point. A multiplication between two numbers in the formats (N1.Q1) and (N2.Q2) results in the format ((N1+N2). (Q1+Q2)). For signal processing applications, the format (1.7) is widely used for the inputs, resulting in a (2.14) format for the product. A left shift is required for the high byte of the product to be in the same format as the inputs. The FMULSU instruction incorporates the shift operation in the same number of cycles as MULSU.</p><p>The (1.7) format is most commonly used with signed numbers, while FMULSU performs a multiplication with one unsigned and one signed input. This instruction is, therefore, most useful for calculating two of the partial products when performing a signed multiplication with 16-bit inputs in the (1.15) format, yielding a result in the (1.31) format.</p><p>Note:  The result of the FMULSU operation may suffer from a 2's complement overflow if interpreted as a number in the (1.15) format. The MSB of the multiplication before shifting must be taken into account and is found in the carry bit. See the following example.</p><p>The multiplicand Rd and the multiplier Rr are two registers containing fractional numbers where the implicit radix point lies between bit 6 and bit 7. The multiplicand Rd is a signed fractional number, and the multiplier Rr is an unsigned fractional number. The 16-bit signed fractional product with the implicit radix point between bit 14 and bit 15 is placed in R1 (high byte) and R0 (low byte).</p><p>This instruction is not available on all devices. Refer to Appendix A.</p>",
                "tooltip": "Fractional Multiply Signed with Unsigned",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=79",
            };

        case "ICALL":
            return {
                "html": "<p>Indirect call of a subroutine pointed to by the Z (16-bit) Pointer Register in the Register File. The Z-Pointer Register is 16 bits wide and allows a call to a subroutine within the lowest 64K words (128 KB) section in the program memory space. The Stack Pointer uses a post-decrement scheme during ICALL.</p><p>This instruction is not available on all devices. Refer to Appendix A.</p>",
                "tooltip": "Indirect Call to Subroutine",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=80",
            };

        case "IJMP":
            return {
                "html": "<p>Indirect jump to the address pointed to by the Z (16-bit) Pointer Register in the Register File. The Z-Pointer Register is 16 bits wide and allows jump within the lowest 64K words (128 KB) section of program memory.</p><p>This instruction is not available on all devices. Refer to Appendix A.</p>",
                "tooltip": "Indirect Jump",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=81",
            };

        case "IN":
            return {
                "html": "<p>Loads data from the I/O space into register Rd in the Register File.</p>",
                "tooltip": "Load an I/O Location to Register",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=82",
            };

        case "INC":
            return {
                "html": "<p>Adds one -1- to the contents of register Rd and places the result in the destination register Rd.</p><p>The C flag in SREG is not affected by the operation, thus allowing the INC instruction to be used on a loop counter in multiple-precision computations.</p><p>When operating on unsigned numbers, only BREQ and BRNE branches can be expected to perform consistently. When operating on two’s complement values, all signed branches are available.</p>",
                "tooltip": "Increment",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=83",
            };

        case "JMP":
            return {
                "html": "<p>Jump to an address within the entire 4M (words) program memory. See also RJMP.</p><p>This instruction is not available on all devices. Refer to Appendix A.</p>",
                "tooltip": "Jump",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=84",
            };

        case "LAC":
            return {
                "html": "<p>Load one byte indirect from data space to register and stores and clear the bits in data space specified by the register. The instruction can only be used towards internal SRAM.</p><p>The data location is pointed to by the Z (16-bit) Pointer Register in the Register File. Memory access is limited to the current data segment of 64 KB. To access another data segment in devices with more than 64 KB data space, the RAMPZ in the register in the I/O area has to be changed.</p><p>The Z-Pointer Register is left unchanged by the operation. This instruction is especially suited for clearing status bits stored in SRAM.</p>",
                "tooltip": "Load and Clear",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=85",
            };

        case "LAS":
            return {
                "html": "<p>Load one byte indirect from data space to register and set bits in the data space specified by the register. The instruction can only be used towards internal SRAM.</p><p>The data location is pointed to by the Z (16-bit) Pointer Register in the Register File. Memory access is limited to the current data segment of 64 KB. To access another data segment in devices with more than 64 KB data space, the RAMPZ in the register in the I/O area has to be changed.</p><p>The Z-Pointer Register is left unchanged by the operation. This instruction is especially suited for setting status bits stored in SRAM.</p>",
                "tooltip": "Load and Set",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=86",
            };

        case "LAT":
            return {
                "html": "<p>Load one byte indirect from data space to register and toggles bits in the data space specified by the register. The instruction can only be used towards SRAM.</p><p>The data location is pointed to by the Z (16-bit) Pointer Register in the Register File. Memory access is limited to the current data segment of 64 KB. To access another data segment in devices with more than 64 KB data space, the RAMPZ in the register in the I/O area has to be changed.</p><p>The Z-Pointer Register is left unchanged by the operation. This instruction is especially suited for changing status bits stored in SRAM.</p>",
                "tooltip": "Load and Toggle",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=86",
            };

        case "LD":
        case "LDD":
            return {
                "html": "<p>Loads one byte indirect from the data space to a register. The data space usually consists of the Register File, I/O memory, and SRAM, refer to the device data sheet for a detailed definition of the data space.</p><p>The data location is pointed to by the X (16-bit) Pointer Register in the Register File. Memory access is limited to the current data segment of 64 KB. To access another data segment in devices with more than 64 KB data space, the RAMPX in the register in the I/O area has to be changed.</p><p>The X-Pointer Register can either be left unchanged by the operation, or it can be post-incremented or pre- decremented. These features are especially suited for accessing arrays, tables, and Stack Pointer usage of the X-Pointer Register. Note that only the low byte of the X-pointer is updated in devices with no more than 256 bytes of data space. For such devices, the high byte of the pointer is not used by this instruction and can be used for other purposes. The RAMPX Register in the I/O area is updated in parts with more than 64 KB data space or more than 64 KB program memory, and the increment/decrement is added to the entire 24-bit address on such devices.</p><p>Not all variants of this instruction are available on all devices.</p><p>In the Reduced Core AVRrc, the LD instruction can be used to achieve the same operation as LPM since the program memory is mapped to the data memory space.</p><p>The result of these combinations is undefined:</p><p>LD r26, X+</p><p>LD r27, X+</p><p>LD r26, -X</p><p>LD r27, -X</p><p>Using the X-pointer:</p>",
                "tooltip": "Load Indirect from Data Space to Register using X",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=87",
            };

        case "LDI":
            return {
                "html": "<p>Loads an 8-bit constant directly to register 16 to 31.</p>",
                "tooltip": "Load Immediate",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=92",
            };

        case "LDS":
        case "AVRrc":
            return {
                "html": "<p>Loads one byte from the data space to a register. The data space usually consists of the Register File, I/O memory, and SRAM, refer to the device data sheet for a detailed definition of the data space.</p><p>A 16-bit address must be supplied. Memory access is limited to the current data segment of 64 KB. The LDS instruction uses the RAMPD Register to access memory above 64 KB. To access another data segment in devices with more than 64 KB data space, the RAMPD in the register in the I/O area has to be changed.</p><p>This instruction is not available on all devices. Refer to Appendix A.</p>",
                "tooltip": "Load Direct from Data Space",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=93",
            };

        case "LPM":
            return {
                "html": "<p>Loads one byte pointed to by the Z-register into the destination register Rd. This instruction features a 100% space-effective constant initialization or constant data fetch. The program memory is organized in 16-bit words while the Z-pointer is a byte address. Thus, the least significant bit of the Z-pointer selects either low byte (ZLSb == 0) or high byte (ZLSb == 1). This instruction can address the first 64 KB (32K words) of program memory. The Z-Pointer Register can either be left unchanged by the operation, or it can be incremented. The incrementation does not apply to the RAMPZ Register.</p><p>Devices with self-programming capability can use the LPM instruction to read the Fuse and Lock bit values. Refer to the device documentation for a detailed description.</p><p>The LPM instruction is not available on all devices. Refer to Appendix A.</p><p>The result of these combinations is undefined:</p><p>LPM r30, Z+</p><p>LPM r31, Z+</p>",
                "tooltip": "Load Program Memory",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=95",
            };

        case "LSL":
            return {
                "html": "<p>Shifts all bits in Rd one place to the left. Bit 0 is cleared. Bit 7 is loaded into the C flag of the SREG. This operation effectively multiplies signed and unsigned values by two.</p>",
                "tooltip": "Logical Shift Left",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=96",
            };

        case "LSR":
            return {
                "html": "<p>Shifts all bits in Rd one place to the right. Bit 7 is cleared. Bit 0 is loaded into the C flag of the SREG. This operation effectively divides an unsigned value by two. The C flag can be used to round the result.</p>",
                "tooltip": "Logical Shift Right",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=97",
            };

        case "MOV":
            return {
                "html": "<p>This instruction makes a copy of one register into another. The source register Rr is left unchanged, while the destination register Rd is loaded with a copy of Rr.</p>",
                "tooltip": "Copy Register",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=98",
            };

        case "MOVW":
            return {
                "html": "<p>This instruction makes a copy of one register pair into another register pair. The source register pair Rr+1:Rr is left unchanged, while the destination register pair Rd+1:Rd is loaded with a copy of Rr + 1:Rr.</p><p>This instruction is not available on all devices. Refer to Appendix A.</p>",
                "tooltip": "Copy Register Word",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=99",
            };

        case "MUL":
            return {
                "html": "<p>This instruction performs 8-bit × 8-bit → 16-bit unsigned multiplication.</p><p>The multiplicand Rd and the multiplier Rr are two registers containing unsigned numbers. The 16-bit unsigned product is placed in R1 (high byte) and R0 (low byte). Note that if the multiplicand or the multiplier is selected from R0 or R1, the result will overwrite those after multiplication.</p><p>This instruction is not available on all devices. Refer to Appendix A.</p>",
                "tooltip": "Multiply Unsigned",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=100",
            };

        case "MULS":
            return {
                "html": "<p>This instruction performs 8-bit × 8-bit → 16-bit signed multiplication.</p><p>The multiplicand Rd and the multiplier Rr are two registers containing signed numbers. The 16-bit signed product is placed in R1 (high byte) and R0 (low byte).</p><p>This instruction is not available on all devices. Refer to Appendix A.</p>",
                "tooltip": "Multiply Signed",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=101",
            };

        case "MULSU":
            return {
                "html": "<p>This instruction performs 8-bit × 8-bit → 16-bit multiplication of a signed and an unsigned number.</p><p>The multiplicand Rd and the multiplier Rr are two registers. The multiplicand Rd is a signed number, and the multiplier Rr is unsigned. The 16-bit signed product is placed in R1 (high byte) and R0 (low byte).</p><p>This instruction is not available on all devices. Refer to Appendix A.</p>",
                "tooltip": "Multiply Signed with Unsigned",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=102",
            };

        case "NEG":
            return {
                "html": "<p>Replaces the contents of register Rd with its two’s complement; the value 0x80 is left unchanged.</p>",
                "tooltip": "Two’s Complement",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=103",
            };

        case "NOP":
            return {
                "html": "<p>This instruction performs a single cycle No Operation.</p>",
                "tooltip": "No Operation",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=104",
            };

        case "OR":
            return {
                "html": "<p>Performs the logical OR between the contents of register Rd and register Rr, and places the result in the destination register Rd.</p>",
                "tooltip": "Logical OR",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=105",
            };

        case "ORI":
            return {
                "html": "<p>Performs the logical OR between the contents of register Rd and a constant, and places the result in the destination register Rd.</p>",
                "tooltip": "Logical OR with Immediate",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=106",
            };

        case "OUT":
            return {
                "html": "<p>Stores data from register Rr in the Register File to I/O space.</p>",
                "tooltip": "Store Register to I/O Location",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=107",
            };

        case "POP":
            return {
                "html": "<p>This instruction loads register Rd with a byte from the STACK. The Stack Pointer is pre-incremented by 1 before the POP.</p><p>This instruction is not available on all devices. Refer to Appendix A.</p>",
                "tooltip": "Pop Register from Stack",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=108",
            };

        case "PUSH":
            return {
                "html": "<p>This instruction stores the contents of register Rr on the STACK. The Stack Pointer is post-decremented by 1 after the PUSH.</p><p>This instruction is not available on all devices. Refer to Appendix A.</p>",
                "tooltip": "Push Register on Stack",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=109",
            };

        case "RCALL":
            return {
                "html": "<p>Relative call to an address within PC - 2K + 1 and PC + 2K (words). The return address (the instruction after the RCALL) is stored onto the Stack. See also CALL. For AVR microcontrollers with program memory not exceeding 4K words (8 KB), this instruction can address the entire memory from every address location. The Stack Pointer uses a post-decrement scheme during RCALL.</p>",
                "tooltip": "Relative Call to Subroutine",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=110",
            };

        case "RET":
            return {
                "html": "<p>Returns from the subroutine. The return address is loaded from the STACK. The Stack Pointer uses a pre-increment scheme during RET.</p>",
                "tooltip": "Return from Subroutine",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=111",
            };

        case "RETI":
            return {
                "html": "<p>Returns from the interrupt. The return address is loaded from the STACK, and the Global Interrupt Enable bit is set.</p><p>Note that the Status Register is not automatically stored when entering an interrupt routine, and it is not restored when returning from an interrupt routine. This must be handled by the application program. The Stack Pointer uses a pre-increment scheme during RETI.</p>",
                "tooltip": "Return from Interrupt",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=112",
            };

        case "RJMP":
            return {
                "html": "<p>Relative jump to an address within PC - 2K +1 and PC + 2K (words). For AVR microcontrollers with pogram memory not exceeding 4K words (8 KB), this instruction can address the entire memory from every address location. See also JMP.</p>",
                "tooltip": "Relative Jump",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=113",
            };

        case "ROL":
            return {
                "html": "<p>Shifts all bits in Rd one place to the left. The C flag is shifted into bit 0 of Rd. Bit 7 is shifted into the C flag. This operation, combined with LSL, effectively multiplies multi-byte signed and unsigned values by two.</p>",
                "tooltip": "Rotate Left trough Carry",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=114",
            };

        case "ROR":
            return {
                "html": "<p>Shifts all bits in Rd one place to the right. The C flag is shifted into bit 7 of Rd. Bit 0 is shifted into the C flag. This operation, combined with ASR, effectively divides multi-byte signed values by two. Combined with LSR, it effectively divides multi-byte unsigned values by two. The Carry flag can be used to round the result.</p>",
                "tooltip": "Rotate Right through Carry",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=115",
            };

        case "SBC":
            return {
                "html": "<p>Subtracts two registers and subtracts with the C flag, and places the result in the destination register Rd.</p>",
                "tooltip": "Subtract with Carry",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=116",
            };

        case "SBCI":
            return {
                "html": "<p>Subtracts a constant from a register and subtracts with the C flag, and places the result in the destination register Rd.</p>",
                "tooltip": "Subtract Immediate with Carry SBI – Set Bit in I/O Register",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=117",
            };

        case "SBI":
            return {
                "html": "<p>Sets a specified bit in an I/O Register. This instruction operates on the lower 32 I/O Registers – addresses 0-31.</p>",
                "tooltip": "Set Bit in I/O Register",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=118",
            };

        case "SBIC":
            return {
                "html": "<p>This instruction tests a single bit in an I/O Register and skips the next instruction if the bit is cleared. This instruction operates on the lower 32 I/O Registers – addresses 0-31.</p>",
                "tooltip": "Skip if Bit in I/O Register is Cleared",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=119",
            };

        case "SBIS":
            return {
                "html": "<p>This instruction tests a single bit in an I/O Register and skips the next instruction if the bit is set. This instruction operates on the lower 32 I/O Registers – addresses 0-31.</p>",
                "tooltip": "Skip if Bit in I/O Register is Set",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=120",
            };

        case "SBIW":
            return {
                "html": "<p>Subtracts an immediate value (0-63) from a register pair and places the result in the register pair. This instruction operates on the upper four register pairs and is well suited for operations on the Pointer Registers.</p><p>This instruction is not available on all devices. Refer to Appendix A.</p>",
                "tooltip": "Subtract Immediate from Word",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=121",
            };

        case "SBR":
            return {
                "html": "<p>Sets specified bits in register Rd. Performs the logical ORI between the contents of register Rd and a constant mask K, and places the result in the destination register Rd. (Equivalent to ORI Rd,K.)</p>",
                "tooltip": "Set Bits in Register",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=122",
            };

        case "SBRC":
            return {
                "html": "<p>This instruction tests a single bit in a register and skips the next instruction if the bit is cleared.</p>",
                "tooltip": "Skip if Bit in Register is Cleared",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=123",
            };

        case "SBRS":
            return {
                "html": "<p>This instruction tests a single bit in a register and skips the next instruction if the bit is set.</p>",
                "tooltip": "Skip if Bit in Register is Set",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=124",
            };

        case "SEC":
            return {
                "html": "<p>Sets the Carry (C) flag in SREG (Status Register). (Equivalent to instruction BSET 0.)</p>",
                "tooltip": "Set Carry Flag",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=125",
            };

        case "SEH":
            return {
                "html": "<p>Sets the Half Carry (H) flag in SREG (Status Register). (Equivalent to instruction BSET 5.)</p>",
                "tooltip": "Set Half Carry Flag",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=126",
            };

        case "SEI":
            return {
                "html": "<p>Sets the Global Interrupt Enable (I) bit in SREG (Status Register). The instruction following SEI will be executed before any pending interrupts.</p>",
                "tooltip": "Set Global Interrupt Enable Bit",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=127",
            };

        case "SEN":
            return {
                "html": "<p>Sets the Negative (N) flag in SREG (Status Register). (Equivalent to instruction BSET 2.)</p>",
                "tooltip": "Set Negative Flag",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=128",
            };

        case "SER":
            return {
                "html": "<p>Loads 0xFF directly to register Rd. (Equivalent to instruction LDI Rd,0xFF).</p>",
                "tooltip": "Set all Bits in Register",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=128",
            };

        case "SES":
            return {
                "html": "<p>Sets the Sign (S) flag in SREG (Status Register). (Equivalent to instruction BSET 4.)</p>",
                "tooltip": "Set Sign Flag",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=129",
            };

        case "SET":
            return {
                "html": "<p>Sets the T bit in SREG (Status Register). (Equivalent to instruction BSET 6.)</p>",
                "tooltip": "Set T Bit",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=130",
            };

        case "SEV":
            return {
                "html": "<p>Sets the Overflow (V) flag in SREG (Status Register). (Equivalent to instruction BSET 3.)</p>",
                "tooltip": "Set Overflow Flag",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=131",
            };

        case "SEZ":
            return {
                "html": "<p>Sets the Zero (Z) flag in SREG (Status Register). (Equivalent to instruction BSET 1.)</p>",
                "tooltip": "Set Zero Flag",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=132",
            };

        case "SPM":
        case "AVRe":
            return {
                "html": "<p>SPM can be used to erase a page in the program memory, to write a page in the program memory (that is already erased), and to set Boot Loader Lock bits. In some devices, the Program memory can be written one word at a time. In other devices, an entire page can be programmed simultaneously after first filling a temporary page buffer. In all cases, the program memory must be erased one page at a time. When erasing the program memory, the RAMPZ and Z-register are used as page address. When writing the program memory, the RAMPZ and Z-register are used as page or word address, and the R1:R0 register pair is used as data(1). The Flash is word-accessed for code space write operations, so the least significant bit of the RAMPZ register concatenated with the Z register should be set to ‘0’. When setting the Boot Loader Lock bits, the R1:R0 register pair is used as data. Refer to the device documentation for the detailed description of SPM usage. This instruction can address the entire program memory.</p><p>The SPM instruction is not available on all devices. Refer to Appendix A.</p><p>Note:  1. R1 determines the instruction high byte, and R0 determines the instruction low byte.</p>",
                "tooltip": "Store Program Memory",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=133",
            };

        case "ST":
        case "STD":
            return {
                "html": "<p>Stores one byte indirect from a register to data space. The data space usually consists of the Register File, I/O memory, and SRAM, refer to the device data sheet for a detailed definition of the data space.</p><p>The data location is pointed to by the X (16-bit) Pointer Register in the Register File. Memory access is limited to the current data segment of 64 KB. To access another data segment in devices with more than 64 KB data space, the RAMPX in the register in the I/O area has to be changed.</p><p>The X-Pointer Register can either be left unchanged by the operation, or it can be post-incremented or pre- decremented. These features are especially suited for accessing arrays, tables, and Stack Pointer usage of the X-Pointer Register. Note that only the low byte of the X-pointer is updated in devices with no more than 256 bytes of data space. For such devices, the high byte of the pointer is not used by this instruction and can be used for otherpurposes. The RAMPX Register in the I/O area is updated in parts with more than 64 KB data space or more than 64 KB program memory, and the increment/ decrement is added to the entire 24-bit address on such devices.</p><p>Not all variants of this instruction are available on all devices.</p><p>The result of these combinations is undefined:</p><p>ST X+, r26</p><p>ST X+, r27</p><p>ST -X, r26</p><p>ST -X, r27</p><p>Using the X-pointer:</p>",
                "tooltip": "Store Indirect From Register to Data Space using Index X",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=136",
            };

        case "STS":
        case "AVRrc":
            return {
                "html": "<p>Stores one byte from a Register to the data space. The data space usually consists of the Register File, I/O memory, and SRAM, refer to the device data sheet for a detailed definition of the data space.</p><p>A 16-bit address must be supplied. Memory access is limited to the current data segment of 64 KB. The STS instruction uses the RAMPD Register to access memory above 64 KB. To access another data segment in devices with more than 64 KB data space, the RAMPD in the register in the I/O area has to be changed.</p><p>This instruction is not available on all devices. Refer to Appendix A.</p>",
                "tooltip": "Store Direct to Data Space",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=141",
            };

        case "SUB":
            return {
                "html": "<p>Subtracts two registers and places the result in the destination register Rd.</p>",
                "tooltip": "Subtract Without Carry",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=143",
            };

        case "SUBI":
            return {
                "html": "<p>Subtracts a register and a constant, and places the result in the destination register Rd. This instruction is working on Register R16 to R31 and is very well suited for operations on the X, Y, and Z-pointers.</p>",
                "tooltip": "Subtract Immediate",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=144",
            };

        case "SWAP":
            return {
                "html": "<p>Swaps high and low nibbles in a register.</p>",
                "tooltip": "Swap Nibbles",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=145",
            };

        case "TST":
            return {
                "html": "<p>Tests if a register is zero or negative. Performs a logical AND between a register and itself. The register will remain unchanged. (Equivalent to instruction AND Rd,Rd.)</p>",
                "tooltip": "Test for Zero or Minus",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=146",
            };

        case "WDR":
            return {
                "html": "<p>This instruction resets the Watchdog Timer. This instruction must be executed within a limited time given by the WD prescaler. See the Watchdog Timer hardware specification.</p>",
                "tooltip": "Watchdog Reset",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=147",
            };

        case "XCH":
            return {
                "html": "<p>Exchanges one byte indirect between the register and data space.</p><p>The data location is pointed to by the Z (16-bit) Pointer Register in the Register File. Memory access is limited to the current data segment of 64 KB. To access another data segment in devices with more than 64 KB data space, the RAMPZ in the register in the I/O area has to be changed.</p><p>The Z-Pointer Register is left unchanged by the operation. This instruction is especially suited for writing/reading status bits stored in SRAM.</p>",
                "tooltip": "Exchange",
                "url": "https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf#page=148",
            };

    }
}