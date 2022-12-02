import {AssemblyInstructionInfo} from '../base';

export function getAsmOpcode(opcode: string | undefined): AssemblyInstructionInfo | undefined {
    if (!opcode) return;
    switch (opcode.toUpperCase()) {
        case "ADC":
            return {
                "html": "<p>This instruction adds the value of memory and carry from the previous operation to the value of the accumulator and stores the result in the accumulator.</p><p>This instruction affects the accumulator; sets the carry flag when the sum of a binary add exceeds 255 or when the sum of a decimal add exceeds 99, otherwise carry is reset. The overflow flag is set when the sign or bit 7 is changed due to the result exceeding +127 or -128, otherwise overflow is reset. The negative flag is set if the accumulator result contains bit 7 on, otherwise the negative flag is reset. The zero flag is set if the accumulator result is 0, otherwise the zero flag is reset.</p>",
                "tooltip": "Add Memory to Accumulator with Carry",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#ADC",
            };

        case "AND":
            return {
                "html": "<p>The AND instruction transfer the accumulator and memory to the adder which performs a bit-by-bit AND operation and stores the result back in the accumulator.</p><p>This instruction affects the accumulator; sets the zero flag if the result in the accumulator is 0, otherwise resets the zero flag; sets the negative flag if the result in the accumulator has bit 7 on, otherwise resets the negative flag.</p>",
                "tooltip": "\"AND\" Memory with Accumulator",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#AND",
            };

        case "ASL":
            return {
                "html": "<p>The shift left instruction shifts either the accumulator or the address memory location 1 bit to the left, with the bit 0 always being set to 0 and the the input bit 7 being stored in the carry flag. ASL either shifts the accumulator left 1 bit or is a read/modify/write instruction that affects only memory.</p><p>The instruction does not affect the overflow bit, sets N equal to the result bit 7 (bit 6 in the input), sets Z flag if the result is equal to 0, otherwise resets Z and stores the input bit 7 in the carry flag.</p>",
                "tooltip": "Arithmetic Shift Left",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#ASL",
            };

        case "BCC":
            return {
                "html": "<p>This instruction tests the state of the carry bit and takes a conditional branch if the carry bit is reset.</p><p>It affects no flags or registers other than the program counter and then only if the C flag is not on.</p>",
                "tooltip": "Branch on Carry Clear",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#BCC",
            };

        case "BCS":
            return {
                "html": "<p>This instruction takes the conditional branch if the carry flag is on.</p><p>BCS does not affect any of the flags or registers except for the program counter and only then if the carry flag is on.</p>",
                "tooltip": "Branch on Carry Set",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#BCS",
            };

        case "BEQ":
            return {
                "html": "<p>This instruction could also be called \"Branch on Equal.\"</p><p>It takes a conditional branch whenever the Z flag is on or the previ ous result is equal to 0.</p><p>BEQ does not affect any of the flags or registers other than the program counter and only then when the Z flag is set.</p>",
                "tooltip": "Branch on Result Zero",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#BEQ",
            };

        case "BIT":
            return {
                "html": "<p>This instruction performs an AND between a memory location and the accumulator but does not store the result of the AND into the accumulator.</p><p>The bit instruction affects the N flag with N being set to the value of bit 7 of the memory being tested, the V flag with V being set equal to bit 6 of the memory being tested and Z being set by the result of the AND operation between the accumulator and the memory if the result is Zero, Z is reset otherwise. It does not affect the accumulator.</p>",
                "tooltip": "Test Bits in Memory with Accumulator",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#BIT",
            };

        case "BMI":
            return {
                "html": "<p>This instruction takes the conditional branch if the N bit is set.</p><p>BMI does not affect any of the flags or any other part of the machine other than the program counter and then only if the N bit is on.</p>",
                "tooltip": "Branch on Result Minus",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#BMI",
            };

        case "BNE":
            return {
                "html": "<p>This instruction could also be called \"Branch on Not Equal.\" It tests the Z flag and takes the conditional branch if the Z flag is not on, indicating that the previous result was not zero.</p><p>BNE does not affect any of the flags or registers other than the program counter and only then if the Z flag is reset.</p>",
                "tooltip": "Branch on Result Not Zero",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#BNE",
            };

        case "BPL":
            return {
                "html": "<p>This instruction is the complementary branch to branch on result minus. It is a conditional branch which takes the branch when the N bit is reset (0). BPL is used to test if the previous result bit 7 was off (0) and branch on result minus is used to determine if the previous result was minus or bit 7 was on (1).</p><p>The instruction affects no flags or other registers other than the P counter and only affects the P counter when the N bit is reset.</p>",
                "tooltip": "Branch on Result Plus",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#BPL",
            };

        case "BRK":
            return {
                "html": "<p>The break command causes the microprocessor to go through an inter rupt sequence under program control. This means that the program counter of the second byte after the BRK. is automatically stored on the stack along with the processor status at the beginning of the break instruction. The microprocessor then transfers control to the interrupt vector.</p><p>Other than changing the program counter, the break instruction changes no values in either the registers or the flags.</p>",
                "tooltip": "Break Command",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#BRK",
            };

        case "BVC":
            return {
                "html": "<p>This instruction tests the status of the V flag and takes the conditional branch if the flag is not set.</p><p>BVC does not affect any of the flags and registers other than the program counter and only when the overflow flag is reset.</p>",
                "tooltip": "Branch on Overflow Clear",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#BVC",
            };

        case "BVS":
            return {
                "html": "<p>This instruction tests the V flag and takes the conditional branch if V is on.</p><p>BVS does not affect any flags or registers other than the program, counter and only when the overflow flag is set.</p>",
                "tooltip": "Branch on Overflow Set",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#BVS",
            };

        case "CLC":
            return {
                "html": "<p>This instruction initializes the carry flag to a 0. This op eration should normally precede an ADC loop. It is also useful when used with a R0L instruction to clear a bit in memory.</p><p>This instruction affects no registers in the microprocessor and no flags other than the carry flag which is reset.</p>",
                "tooltip": "Clear Carry Flag",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#CLC",
            };

        case "CLD":
            return {
                "html": "<p>This instruction sets the decimal mode flag to a 0. This all subsequent ADC and SBC instructions to operate as simple operations.</p><p>CLD affects no registers in the microprocessor and no flags other than the decimal mode flag which is set to a 0.</p>",
                "tooltip": "Clear Decimal Mode",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#CLD",
            };

        case "CLI":
            return {
                "html": "<p>This instruction initializes the interrupt disable to a 0. This allows the microprocessor to receive interrupts.</p><p>It affects no registers in the microprocessor and no flags other than the interrupt disable which is cleared.</p>",
                "tooltip": "Clear Interrupt Disable",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#CLI",
            };

        case "CLV":
            return {
                "html": "<p>This instruction clears the overflow flag to a 0. This com mand is used in conjunction with the set overflow pin which can change the state of the overflow flag with an external signal.</p><p>CLV affects no registers in the microprocessor and no flags other than the overflow flag which is set to a 0.</p>",
                "tooltip": "Clear Overflow Flag",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#CLV",
            };

        case "CMP":
            return {
                "html": "<p>This instruction subtracts the contents of memory from the contents of the accumulator.</p><p>The use of the CMP affects the following flags: Z flag is set on an equal comparison, reset otherwise; the N flag is set or reset by the result bit 7, the carry flag is set when the value in memory is less than or equal to the accumulator, reset when it is greater than the accumulator. The accumulator is not affected.</p>",
                "tooltip": "Compare Memory and Accumulator",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#CMP",
            };

        case "CPX":
            return {
                "html": "<p>This instruction subtracts the value of the addressed memory location from the content of index register X using the adder but does not store the result; therefore, its only use is to set the N, Z and C flags to allow for comparison between the index register X and the value in memory.</p><p>The CPX instruction does not affect any register in the machine; it also does not affect the overflow flag. It causes the carry to be set on if the absolute value of the index register X is equal to or greater than the data from memory. If the value of the memory is greater than the content of the index register X, carry is reset. If the results of the subtraction contain a bit 7, then the N flag is set, if not, it is reset. If the value in memory is equal to the value in index register X, the Z flag is set, otherwise it is reset.</p>",
                "tooltip": "Compare Index Register X To Memory",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#CPX",
            };

        case "CPY":
            return {
                "html": "<p>This instruction performs a two's complement subtraction between the index register Y and the specified memory location. The results of the subtraction are not stored anywhere. The instruction is strictly used to set the flags.</p><p>CPY affects no registers in the microprocessor and also does not affect the overflow flag. If the value in the index register Y is equal to or greater than the value in the memory, the carry flag will be set, otherwise it will be cleared. If the results of the subtract- tion contain bit 7 on the N bit will be set, otherwise it will be cleared. If the value in the index register Y and the value in the memory are equal, the zero flag will be set, otherwise it will be cleared.</p>",
                "tooltip": "Compare Index Register Y To Memory",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#CPY",
            };

        case "DEC":
            return {
                "html": "<p>This instruction subtracts 1, in two's complement, from the contents of the addressed memory location.</p><p>The decrement instruction does not affect any internal register in the microprocessor. It does not affect the carry or overflow flags. If bit 7 is on as a result of the decrement, then the N flag is set, otherwise it is reset. If the result of the decrement is 0, the Z flag is set, otherwise it is reset.</p>",
                "tooltip": "Decrement Memory By One",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#DEC",
            };

        case "DEX":
            return {
                "html": "<p>This instruction subtracts one from the current value of the index register X and stores the result in the index register X.</p><p>DEX does not affect the carry or overflow flag, it sets the N flag if it has bit 7 on as a result of the decrement, otherwise it resets the N flag; sets the Z flag if X is a 0 as a result of the decrement, otherwise it resets the Z flag.</p>",
                "tooltip": "Decrement Index Register X By One",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#DEX",
            };

        case "DEY":
            return {
                "html": "<p>This instruction subtracts one from the current value in the in dex register Y and stores the result into the index register Y. The result does not affect or consider carry so that the value in the index register Y is decremented to 0 and then through 0 to FF.</p><p>Decrement Y does not affect the carry or overflow flags; if the Y register contains bit 7 on as a result of the decrement the N flag is set, otherwise the N flag is reset. If the Y register is 0 as a result of the decrement, the Z flag is set otherwise the Z flag is reset. This instruction only affects the index register Y.</p>",
                "tooltip": "Decrement Index Register Y By One",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#DEY",
            };

        case "EOR":
            return {
                "html": "<p>The EOR instruction transfers the memory and the accumulator to the adder which performs a binary \"EXCLUSIVE OR\" on a bit-by-bit basis and stores the result in the accumulator.</p><p>This instruction affects the accumulator; sets the zero flag if the result in the accumulator is 0, otherwise resets the zero flag sets the negative flag if the result in the accumulator has bit 7 on, otherwise resets the negative flag.</p>",
                "tooltip": "\"Exclusive OR\" Memory with Accumulator",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#EOR",
            };

        case "INC":
            return {
                "html": "<p>This instruction adds 1 to the contents of the addressed memory location.</p><p>The increment memory instruction does not affect any internal registers and does not affect the carry or overflow flags. If bit 7 is on as the result of the increment,N is set, otherwise it is reset; if the increment causes the result to become 0, the Z flag is set on, otherwise it is reset.</p>",
                "tooltip": "Increment Memory By One",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#INC",
            };

        case "INX":
            return {
                "html": "<p>Increment X adds 1 to the current value of the X register. This is an 8-bit increment which does not affect the carry operation, therefore, if the value of X before the increment was FF, the resulting value is 00.</p><p>INX does not affect the carry or overflow flags; it sets the N flag if the result of the increment has a one in bit 7, otherwise resets N; sets the Z flag if the result of the increment is 0, otherwise it resets the Z flag.</p><p>INX does not affect any other register other than the X register.</p>",
                "tooltip": "Increment Index Register X By One",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#INX",
            };

        case "INY":
            return {
                "html": "<p>Increment Y increments or adds one to the current value in the Y register, storing the result in the Y register. As in the case of INX the primary application is to step thru a set of values using the Y register.</p><p>The INY does not affect the carry or overflow flags, sets the N flag if the result of the increment has a one in bit 7, otherwise resets N, sets Z if as a result of the increment the Y register is zero otherwise resets the Z flag.</p>",
                "tooltip": "Increment Index Register Y By One",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#INY",
            };

        case "JMP":
            return {
                "html": "<p>In this instruction, the data from the memory location located in the program sequence after the OP CODE is loaded into the low order byte of the program counter (PCL) and the data from the next memory location after that is loaded into the high order byte of the program counter (PCH).</p><p>This instruction establishes a new valne for the program counter.</p><p>It affects only the program counter in the microprocessor and affects no flags in the status register.</p>",
                "tooltip": "JMP Indirect",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#JMP",
            };

        case "JSR":
            return {
                "html": "<p>This instruction transfers control of the program counter to a subroutine location but leaves a return pointer on the stack to allow the user to return to perform the next instruction in the main program after the subroutine is complete. To accomplish this, JSR instruction stores the program counter address which points to the last byte of the jump instruc tion onto the stack using the stack pointer. The stack byte contains the program count high first, followed by program count low. The JSR then transfers the addresses following the jump instruction to the program counter low and the program counter high, thereby directing the program to begin at that new address.</p><p>The JSR instruction affects no flags, causes the stack pointer to be decremented by 2 and substitutes new values into the program counter low and the program counter high.</p>",
                "tooltip": "Jump To Subroutine",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#JSR",
            };

        case "LDA":
            return {
                "html": "<p>When instruction LDA is executed by the microprocessor, data is transferred from memory to the accumulator and stored in the accumulator.</p><p>LDA affects the contents of the accumulator, does not affect the carry or overflow flags; sets the zero flag if the accumulator is zero as a result of the LDA, otherwise resets the zero flag; sets the negative flag if bit 7 of the accumulator is a 1, other wise resets the negative flag.</p>",
                "tooltip": "Load Accumulator with Memory",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#LDA",
            };

        case "LDX":
            return {
                "html": "<p>Load the index register X from memory.</p><p>LDX does not affect the C or V flags; sets Z if the value loaded was zero, otherwise resets it; sets N if the value loaded in bit 7 is a 1; otherwise N is reset, and affects only the X register.</p>",
                "tooltip": "Load Index Register X From Memory",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#LDX",
            };

        case "LDY":
            return {
                "html": "<p>Load the index register Y from memory.</p><p>LDY does not affect the C or V flags, sets the N flag if the value loaded in bit 7 is a 1, otherwise resets N, sets Z flag if the loaded value is zero otherwise resets Z and only affects the Y register.</p>",
                "tooltip": "Load Index Register Y From Memory",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#LDY",
            };

        case "LSR":
            return {
                "html": "<p>This instruction shifts either the accumulator or a specified memory location 1 bit to the right, with the higher bit of the result always being set to 0, and the low bit which is shifted out of the field being stored in the carry flag.</p><p>The shift right instruction either affects the accumulator by shifting it right 1 or is a read/modify/write instruction which changes a specified memory location but does not affect any internal registers. The shift right does not affect the overflow flag. The N flag is always reset. The Z flag is set if the result of the shift is 0 and reset otherwise. The carry is set equal to bit 0 of the input.</p>",
                "tooltip": "Logical Shift Right",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#LSR",
            };

        case "NOP":
            return {
                "html": "<p>No Operation</p>",
                "tooltip": "No Operation",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#NOP",
            };

        case "ORA":
            return {
                "html": "<p>The ORA instruction transfers the memory and the accumulator to the adder which performs a binary \"OR\" on a bit-by-bit basis and stores the result in the accumulator.</p><p>This instruction affects the accumulator; sets the zero flag if the result in the accumulator is 0, otherwise resets the zero flag; sets the negative flag if the result in the accumulator has bit 7 on, otherwise resets the negative flag.</p>",
                "tooltip": "\"OR\" Memory with Accumulator",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#ORA",
            };

        case "PHA":
            return {
                "html": "<p>This instruction transfers the current value of the accumulator to the next location on the stack, automatically decrementing the stack to point to the next empty location.</p><p>The Push A instruction only affects the stack pointer register which is decremented by 1 as a result of the operation. It affects no flags.</p>",
                "tooltip": "Push Accumulator On Stack",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#PHA",
            };

        case "PHP":
            return {
                "html": "<p>This instruction transfers the contents of the processor status reg ister unchanged to the stack, as governed by the stack pointer.</p><p>The PHP instruction affects no registers or flags in the microprocessor.</p>",
                "tooltip": "Push Processor Status On Stack",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#PHP",
            };

        case "PLA":
            return {
                "html": "<p>This instruction adds 1 to the current value of the stack pointer and uses it to address the stack and loads the contents of the stack into the A register.</p><p>The PLA instruction does not affect the carry or overflow flags. It sets N if the bit 7 is on in accumulator A as a result of instructions, otherwise it is reset. If accumulator A is zero as a result of the PLA, then the Z flag is set, otherwise it is reset. The PLA instruction changes content of the accumulator A to the contents of the memory location at stack register plus 1 and also increments the stack register.</p>",
                "tooltip": "Pull Accumulator From Stack",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#PLA",
            };

        case "PLP":
            return {
                "html": "<p>This instruction transfers the next value on the stack to the Proces sor Status register, thereby changing all of the flags and setting the mode switches to the values from the stack.</p><p>The PLP instruction affects no registers in the processor other than the status register. This instruction could affect all flags in the status register.</p>",
                "tooltip": "Pull Processor Status From Stack",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#PLP",
            };

        case "ROL":
            return {
                "html": "<p>The rotate left instruction shifts either the accumulator or addressed memory left 1 bit, with the input carry being stored in bit 0 and with the input bit 7 being stored in the carry flags.</p><p>The ROL instruction either shifts the accumulator left 1 bit and stores the carry in accumulator bit 0 or does not affect the internal registers at all. The ROL instruction sets carry equal to the input bit 7, sets N equal to the input bit 6 , sets the Z flag if the result of the ro tate is 0, otherwise it resets Z and does not affect the overflow flag at all.</p>",
                "tooltip": "Rotate Left",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#ROL",
            };

        case "ROR":
            return {
                "html": "<p>The rotate right instruction shifts either the accumulator or addressed memory right 1 bit with bit 0 shifted into the carry and carry shifted into bit 7.</p><p>The ROR instruction either shifts the accumulator right 1 bit and stores the carry in accumulator bit 7 or does not affect the internal regis ters at all. The ROR instruction sets carry equal to input bit 0, sets N equal to the input carry and sets the Z flag if the result of the rotate is 0; otherwise it resets Z and does not affect the overflow flag at all.</p><p>(Available on Microprocessors after June, 1976)</p>",
                "tooltip": "Rotate Right",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#ROR",
            };

        case "RTI":
            return {
                "html": "<p>This instruction transfers from the stack into the microprocessor the processor status and the program counter location for the instruction which was interrupted. By virtue of the interrupt having stored this data before executing the instruction and thei fact that the RTI reinitializes the microprocessor to the same state as when it was interrupted, the combination of interrupt plus RTI allows truly reentrant coding.</p><p>The RTI instruction reinitializes all flags to the position to the point they were at the time the interrupt was taken and sets the program counter back to its pre-interrupt state. It affects no other registers in the microprocessor.</p>",
                "tooltip": "Return From Interrupt",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#RTI",
            };

        case "RTS":
            return {
                "html": "<p>This instruction loads the program count low and program count high from the stack into the program counter and increments the program counter so that it points to the instruction following the JSR. The stack pointer is adjusted by incrementing it twice.</p><p>The RTS instruction does not affect any flags and affects only PCL and PCH.</p>",
                "tooltip": "Return From Subroutme",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#RTS",
            };

        case "SBC":
            return {
                "html": "<p>This instruction subtracts the value of memory and borrow from the value of the accumulator, using two's complement arithmetic, and stores the result in the accumulator. Borrow is defined as the carry flag complemented; therefore, a resultant carry flag indicates that a borrow has not occurred.</p><p>This instruction affects the accumulator. The carry flag is set if the result is greater than or equal to 0. The carry flag is reset when the result is less than 0, indicating a borrow. The overflow flag is set when the result exceeds +127 or -127, otherwise it is reset. The negative flag is set if the result in the accumulator has bit 7 on, otherwise it is reset. The Z flag is set if the result in the accumulator is 0, otherwise it is reset.</p>",
                "tooltip": "Subtract Memory from Accumulator with Borrow",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#SBC",
            };

        case "SEC":
            return {
                "html": "<p>This instruction initializes the carry flag to a 1. This op eration should normally precede a SBC loop. It is also useful when used with a ROL instruction to initialize a bit in memory to a 1.</p><p>This instruction affects no registers in the microprocessor and no flags other than the carry flag which is set.</p>",
                "tooltip": "Set Carry Flag",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#SEC",
            };

        case "SED":
            return {
                "html": "<p>This instruction sets the decimal mode flag D to a 1. This makes all subsequent ADC and SBC instructions operate as a decimal arithmetic operation.</p><p>SED affects no registers in the microprocessor and no flags other than the decimal mode which is set to a 1.</p>",
                "tooltip": "Set Decimal Mode",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#SED",
            };

        case "SEI":
            return {
                "html": "<p>This instruction initializes the interrupt disable to a 1. It is used to mask interrupt requests during system reset operations and during interrupt commands.</p><p>It affects no registers in the microprocessor and no flags other than the interrupt disable which is set.</p>",
                "tooltip": "Set Interrupt Disable",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#SEI",
            };

        case "STA":
            return {
                "html": "<p>This instruction transfers the contents of the accumulator to memory.</p><p>This instruction affects none of the flags in the processor status register and does not affect the accumulator.</p>",
                "tooltip": "Store Accumulator in Memory",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#STA",
            };

        case "STX":
            return {
                "html": "<p>Transfers value of X register to addressed memory location.</p><p>No flags or registers in the microprocessor are affected by the store operation.</p>",
                "tooltip": "Store Index Register X In Memory",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#STX",
            };

        case "STY":
            return {
                "html": "<p>Transfer the value of the Y register to the addressed memory location.</p><p>STY does not affect any flags or registers in the microprocessor.</p>",
                "tooltip": "Store Index Register Y In Memory",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#STY",
            };

        case "TAX":
            return {
                "html": "<p>This instruction takes the value from accumulator A and trans fers or loads it into the index register X without disturbing the content of the accumulator A.</p><p>TAX only affects the index register X, does not affect the carry or overflow flags. The N flag is set if the resultant value in the index register X has bit 7 on, otherwise N is reset. The Z bit is set if the content of the register X is 0 as aresult of theopera tion, otherwise it is reset.</p>",
                "tooltip": "Transfer Accumulator To Index X",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#TAX",
            };

        case "TAY":
            return {
                "html": "<p>This instruction moves the value of the accumulator into index register Y without affecting the accumulator.</p><p>TAY instruction only affects the Y register and does not affect either the carry or overflow flags. If the index register Y has bit 7 on, then N is set, otherwise it is reset. If the content of the index register Y equals 0 as a result of the operation, Z is set on, otherwise it is reset.</p>",
                "tooltip": "Transfer Accumula Tor To Index Y",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#TAY",
            };

        case "TSX":
            return {
                "html": "<p>This instruction transfers the value in the stack pointer to the index register X.</p><p>TSX does not affect the carry or overflow flags. It sets N if bit 7 is on in index X as a result of the instruction, otherwise it is reset. If index X is zero as a result of the TSX, the Z flag is set, other wise it is reset. TSX changes the value of index X, making it equal to the content of the stack pointer.</p>",
                "tooltip": "Transfer Stack Pointer To Index X",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#TSX",
            };

        case "TXA":
            return {
                "html": "<p>This instruction moves the value that is in the index register X to the accumulator A without disturbing the content of the index register X.</p><p>TXA does not affect any register other than the accumulator and does not affect the carry or overflow flag. If the result in A has bit 7 on, then the N flag is set, otherwise it is reset. If the resultant value in the accumulator is 0, then the Z flag is set, other wise it is reset.</p>",
                "tooltip": "Transfer Index X To Accumulator",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#TXA",
            };

        case "TXS":
            return {
                "html": "<p>This instruction transfers the value in the index register X to the stack pointer.</p><p>TXS changes only the stack pointer, making it equal to the content of the index register X. It does not affect any of the flags.</p>",
                "tooltip": "Transfer Index X To Stack Pointer",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#TXS",
            };

        case "TYA":
            return {
                "html": "<p>This instruction moves the value that is in the index register Y to accumulator A without disturbing the content of the register Y.</p><p>TYA does not affect any other register other than the accumula tor and does not affect the carry or overflow flag. If the result in the accumulator A has bit 7 on, the N flag is set, otherwise it is reset. If the resultant value in the accumulator A is 0, then the Z flag is set, otherwise it is reset.</p>",
                "tooltip": "Transfer Index Y To Accumulator",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#TYA",
            };

        case "SAX":
            return {
                "html": "<p>The undocumented SAX instruction performs a bit-by-bit AND operation of the accumulator and the X register and transfers the result to the addressed memory location.</p><p>No flags or registers in the microprocessor are affected by the store operation.</p><p>The undocumented SAX instruction performs a bit-by-bit AND operation of the value of the accumulator and the value of the index register X and stores the result in memory.</p><p>No flags or registers in the microprocessor are affected by the store operation.</p>",
                "tooltip": "Store Accumulator \"AND\" Index Register X in Memory",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#SAX",
            };

        case "SHA":
            return {
                "html": "<p>The undocumented SHA instruction performs a bit-by-bit AND operation of the following three operands: The first two are the accumulator and the index register X.</p><p>The third operand depends on the addressing mode. In the zero page indirect Y-indexed case, the third operand is the data in memory at the given zero page address (ignoring the the addressing mode's Y offset) plus 1. In the Y-indexed absolute case, it is the upper 8 bits of the given address (ignoring the the addressing mode's Y offset), plus 1.</p><p>It then transfers the result to the addressed memory location.</p><p>No flags or registers in the microprocessor are affected by the store operation.</p>",
                "tooltip": "Store Accumulator \"AND\" Index Register X \"AND\" Value",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#SHA",
            };

        case "ASR":
            return {
                "html": "<p>The undocumented ASR instruction performs a bit-by-bit AND operation of the accumulator and memory, then shifts the accumulator 1 bit to the right, with the higher bit of the result always being set to 0, and the low bit which is shifted out of the field being stored in the carry flag.</p><p>This instruction affects the accumulator. It does not affect the overflow flag. The N flag is always reset. The Z flag is set if the result of the shift is 0 and reset otherwise. The carry is set equal to bit 0 of the result of the \"AND\" operation.</p>",
                "tooltip": "\"AND\" then Logical Shift Right",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#ASR",
            };

        case "ANC":
            return {
                "html": "<p>The undocumented ANC instruction performs a bit-by-bit AND operation of the accumulator and memory and stores the result back in the accumulator.</p><p>This instruction affects the accumulator; sets the zero flag if the result in the accumulator is 0, otherwise resets the zero flag; sets the negative flag and the carry flag if the result in the accumulator has bit 7 on, otherwise resets the negative flag and the carry flag.</p>",
                "tooltip": "\"AND\" Memory with Accumulator then Move Negative Flag to Carry Flag",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#ANC",
            };

        case "ARR":
            return {
                "html": "<p>The undocumented ARR instruction performs a bit-by-bit \"AND\" operation of the accumulator and memory, then shifts the result right 1 bit with bit 0 shifted into the carry and carry shifted into bit 7. It then stores the result back in the accumulator.</p><p>If bit 7 of the result is on, then the N flag is set, otherwise it is reset. The instruction sets the Z flag if the result is 0; otherwise it resets Z.</p><p>The V and C flags depends on the Decimal Mode Flag:</p><p>In decimal mode, the V flag is set if bit 6 is different than the original data's bit 6, otherwise the V flag is reset. The C flag is set if (operand & 0xF0) + (operand & 0x10) is greater than 0x50, otherwise the C flag is reset.</p><p>In binary mode, the V flag is set if bit 6 of the result is different than bit 5 of the result, otherwise the V flag is reset. The C flag is set if the result in the accumulator has bit 6 on, otherwise it is reset.</p>",
                "tooltip": "\"AND\" Accumulator then Rotate Right",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#ARR",
            };

        case "SBX":
            return {
                "html": "<p>This undocumented instruction performs a bit-by-bit \"AND\" of the value of the accumulator and the index register X and subtracts the value of memory from this result, using two's complement arithmetic, and stores the result in the index register X.</p><p>This instruction affects the index register X. The carry flag is set if the result is greater than or equal to 0. The carry flag is reset when the result is less than 0, indicating a borrow. The negative flag is set if the result in index register X has bit 7 on, otherwise it is reset. The Z flag is set if the result in index register X is 0, otherwise it is reset. The overflow flag not affected at all.</p>",
                "tooltip": "Subtract Memory from Accumulator \"AND\" Index Register X",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#SBX",
            };

        case "DCP":
            return {
                "html": "<p>This undocumented instruction subtracts 1, in two's complement, from the contents of the addressed memory location. It then subtracts the contents of memory from the contents of the accumulator.</p><p>The DCP instruction does not affect any internal register in the microprocessor. It does not affect the overflow flag. Z flag is set on an equal comparison, reset otherwise; the N flag is set or reset by the result bit 7, the carry flag is set when the result in memory is less than or equal to the accumulator, reset when it is greater than the accumulator.</p>",
                "tooltip": "Decrement Memory By One then Compare with Accumulator",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#DCP",
            };

        case "ISC":
            return {
                "html": "<p>This undocumented instruction adds 1 to the contents of the addressed memory location. It then subtracts the value of the result in memory and borrow from the value of the accumulator, using two's complement arithmetic, and stores the result in the accumulator.</p><p>This instruction affects the accumulator. The carry flag is set if the result is greater than or equal to 0. The carry flag is reset when the result is less than 0, indicating a borrow. The overflow flag is set when the result exceeds +127 or -127, otherwise it is reset. The negative flag is set if the result in the accumulator has bit 7 on, otherwise it is reset. The Z flag is set if the result in the accumulator is 0, otherwise it is reset.</p>",
                "tooltip": "Increment Memory By One then SBC then Subtract Memory from Accumulator with Borrow",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#ISC",
            };

        case "JAM":
            return {
                "html": "<p>This undocumented instruction stops execution. The microprocessor will not fetch further instructions, and will neither handle IRQs nor NMIs. It will handle a RESET though.</p>",
                "tooltip": "Halt the CPU",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#JAM",
            };

        case "LAS":
            return {
                "html": "<p>This undocumented instruction performs a bit-by-bit \"AND\" operation of the stack pointer and memory and stores the result back in the accumulator, the index register X and the stack pointer.</p><p>The LAS instruction does not affect the carry or overflow flags. It sets N if the bit 7 of the result is on, otherwise it is reset. If the result is zero, then the Z flag is set, otherwise it is reset.</p>",
                "tooltip": "\"AND\" Memory with Stack Pointer",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#LAS",
            };

        case "LAX":
            return {
                "html": "<p>The undocumented LAX instruction loads the accumulator and the index register X from memory.</p><p>LAX does not affect the C or V flags; sets Z if the value loaded was zero, otherwise resets it; sets N if the value loaded in bit 7 is a 1; otherwise N is reset, and affects only the X register.</p>",
                "tooltip": "Load Accumulator and Index Register X From Memory",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#LAX",
            };

        case "RLA":
            return {
                "html": "<p>The undocumented RLA instruction shifts the addressed memory left 1 bit, with the input carry being stored in bit 0 and with the input bit 7 being stored in the carry flags. It then performs a bit-by-bit AND operation of the result and the value of the accumulator and stores the result back in the accumulator.</p><p>This instruction affects the accumulator; sets the zero flag if the result in the accumulator is 0, otherwise resets the zero flag; sets the negative flag if the result in the accumulator has bit 7 on, otherwise resets the negative flag.</p>",
                "tooltip": "Rotate Left then \"AND\" with Accumulator",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#RLA",
            };

        case "RRA":
            return {
                "html": "<p>The undocumented RRA instruction shifts the addressed memory right 1 bit with bit 0 shifted into the carry and carry shifted into bit 7. It then adds the result and generated carry to the value of the accumulator and stores the result in the accumulator.</p><p>This instruction affects the accumulator; sets the carry flag when the sum of a binary add exceeds 255 or when the sum of a decimal add exceeds 99, otherwise carry is reset. The overflow flag is set when the sign or bit 7 is changed due to the result exceeding +127 or -128, otherwise overflow is reset. The negative flag is set if the accumulator result contains bit 7 on, otherwise the negative flag is reset. The zero flag is set if the accumulator result is 0, otherwise the zero flag is reset.</p>",
                "tooltip": "Rotate Right and Add Memory to Accumulator",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#RRA",
            };

        case "SHX":
            return {
                "html": "<p>The undocumented SHX instruction performs a bit-by-bit AND operation of the index register X and the upper 8 bits of the given address (ignoring the the addressing mode's Y offset), plus 1. It then transfers the result to the addressed memory location.</p><p>No flags or registers in the microprocessor are affected by the store operation.</p>",
                "tooltip": "Store Index Register X \"AND\" Value",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#SHX",
            };

        case "SHY":
            return {
                "html": "<p>The undocumented SHY instruction performs a bit-by-bit AND operation of the index register Y and the upper 8 bits of the given address (ignoring the the addressing mode's X offset), plus 1. It then transfers the result to the addressed memory location.</p><p>No flags or registers in the microprocessor are affected by the store operation.</p>",
                "tooltip": "Store Index Register Y \"AND\" Value",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#SHY",
            };

        case "SLO":
            return {
                "html": "<p>The undocumented SLO instruction shifts the address memory location 1 bit to the left, with the bit 0 always being set to 0 and the bit 7 output always being contained in the carry flag. It then performs a bit-by-bit \"OR\" operation on the result and the accumulator and stores the result in the accumulator.</p><p>The negative flag is set if the accumulator result contains bit 7 on, otherwise the negative flag is reset. It sets Z flag if the result is equal to 0, otherwise resets Z and stores the input bit 7 in the carry flag.</p>",
                "tooltip": "Arithmetic Shift Left then \"OR\" Memory with Accumulator",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#SLO",
            };

        case "SRE":
            return {
                "html": "<p>The undocumented SRE instruction shifts the specified memory location 1 bit to the right, with the higher bit of the result always being set to 0, and the low bit which is shifted out of the field being stored in the carry flag. It then performs a bit-by-bit \"EXCLUSIVE OR\" of the result and the value of the accumulator and stores the result in the accumulator.</p><p>This instruction affects the accumulator. It does not affect the overflow flag. The negative flag is set if the accumulator result contains bit 7 on, otherwise the negative flag is reset. The Z flag is set if the result is 0 and reset otherwise. The carry is set equal to input bit 0.</p>",
                "tooltip": "Logical Shift Right then \"Exclusive OR\" Memory with Accumulator",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#SRE",
            };

        case "SHS":
            return {
                "html": "<p>The undocumented SHS instruction performs a bit-by-bit AND operation of the value of the accumulator and the value of the index register X and stores the result in the stack pointer. It then performs a bit-by-bit AND operation of the resulting stack pointer and the upper 8 bits of the given address (ignoring the addressing mode's Y offset), plus 1, and transfers the result to the addressed memory location.</p><p>No flags or registers in the microprocessor are affected by the store operation.</p>",
                "tooltip": "Transfer Accumulator \"AND\" Index Register X to Stack Pointer then Store Stack Pointer \"AND\" Hi-Byte In Memory",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#SHS",
            };

        case "XAA":
            return {
                "html": "<p>The operation of the undocumented XAA instruction depends on the individual microprocessor. On most machines, it performs a bit-by-bit AND operation of the following three operands: The first two are the index register X and memory.</p><p>The third operand is the result of a bit-by-bit AND operation of the accumulator and a magic component. This magic component depends on the individual microprocessor and is usually one of $00, $EE, $EF, $FE and $FF, and may be influenced by the RDY pin, leftover contents of the data bus, the temperature of the microprocessor, the supplied voltage, and other factors.</p><p>On some machines, additional bits of the result may be set or reset depending on non-deterministic factors.</p><p>It then transfers the result to the accumulator.</p><p>XAA does not affect the C or V flags; sets Z if the value loaded was zero, otherwise resets it; sets N if the result in bit 7 is a 1; otherwise N is reset.</p>",
                "tooltip": "Non-deterministic Operation of Accumulator, Index Register X, Memory and Bus Contents",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#XAA",
            };

        case "BRA":
            return {
                "html": "<p>This instruction takes an unconditional branch.</p><p>BRA does not affect any of the flags or any other part of the machine other than the program counter.</p>",
                "tooltip": "Branch Always",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c02&tab=2#BRA",
            };

        case "PHX":
            return {
                "html": "<p>This instruction transfers the current value of the index register X to the next location on the stack, automatically decrementing the stack to point to the next empty location.</p><p>The Push X instruction only affects the stack pointer register which is decremented by 1 as a result of the operation. It affects no flags.</p>",
                "tooltip": "Push Index Register X On Stack",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c02&tab=2#PHX",
            };

        case "PHY":
            return {
                "html": "<p>This instruction transfers the current value of the index register Y to the next location on the stack, automatically decrementing the stack to point to the next empty location.</p><p>The Push Y instruction only affects the stack pointer register which is decremented by 1 as a result of the operation. It affects no flags.</p>",
                "tooltip": "Push Index Register Y On Stack",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c02&tab=2#PHY",
            };

        case "PLX":
            return {
                "html": "<p>This instruction adds 1 to the current value of the stack pointer and uses it to address the stack and loads the contents of the stack into the X register.</p><p>The PLX instruction does not affect the carry or overflow flags. It sets N if the bit 7 is on in index register X as a result of instructions, otherwise it is reset. If index register X is zero as a result of the PLA, then the Z flag is set, otherwise it is reset. The PLX instruction changes content of the index register X to the contents of the memory location at stack register plus 1 and also increments the stack register.</p>",
                "tooltip": "Pull Index Register X From Stack",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c02&tab=2#PLX",
            };

        case "PLY":
            return {
                "html": "<p>This instruction adds 1 to the current value of the stack pointer and uses it to address the stack and loads the contents of the stack into the Y register.</p><p>The PLY instruction does not affect the carry or overflow flags. It sets N if the bit 7 is on in index register Y as a result of instructions, otherwise it is reset. If index register Y is zero as a result of the PLA, then the Z flag is set, otherwise it is reset. The PLY instruction changes content of the index register Y to the contents of the memory location at stack register plus 1 and also increments the stack register.</p>",
                "tooltip": "Pull Index Register Y From Stack",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c02&tab=2#PLY",
            };

        case "STZ":
            return {
                "html": "<p>Transfers the value 0 to addressed memory location.</p><p>No flags or registers in the microprocessor are affected by the store operation.</p>",
                "tooltip": "Store Zero In Memory",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c02&tab=2#STZ",
            };

        case "TRB":
            return {
                "html": "<p>This instruction tests and resets bits in memory, using the accumulator for both a test mask, and a reset mask. It performs a logical AND between the inverted bits of the accumulator and the bits in memory, storing the result back into memory.</p><p>The zero flag is set if all bits of the result of the AND are zero, otherwise it is reset.</p>",
                "tooltip": "Test And Reset Memory Bits With Accumulator",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c02&tab=2#TRB",
            };

        case "TSB":
            return {
                "html": "<p>This instruction tests and sets bits in memory, using the accumulator for both a test mask, and a set mask. It performs a logical OR between the bits of the accumulator and the bits in memory, storing the result back into memory.</p><p>The zero flag is set if all bits of the result of the OR are zero, otherwise it is reset.</p>",
                "tooltip": "Test And Set Memory Bits With Accumulator",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c02&tab=2#TSB",
            };

    }
}