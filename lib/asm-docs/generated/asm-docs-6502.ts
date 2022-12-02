export function getAsmOpcode(opcode) {
    if (!opcode) return;
    switch (opcode.toUpperCase()) {
        case "ADC":
            return {
                "html": "<p>This instruction adds the value of memory and carry from the previous operation to the value of the accumulator and stores the result in the accumulator.</p>",
                "tooltip": "Add Memory to Accumulator with Carry",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#ADC",
            };

        case "AND":
            return {
                "html": "<p>The AND instruction transfer the accumulator and memory to the adder which performs a bit-by-bit AND operation and stores the result back in the accumulator.</p>",
                "tooltip": "\"AND\" Memory with Accumulator",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#AND",
            };

        case "ASL":
            return {
                "html": "<p>The shift left instruction shifts either the accumulator or the address memory location 1 bit to the left, with the bit 0 always being set to 0 and the the input bit 7 being stored in the carry flag. ASL either shifts the accumulator left 1 bit or is a read/modify/write instruction that affects only memory.</p>",
                "tooltip": "Arithmetic Shift Left",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#ASL",
            };

        case "BCC":
            return {
                "html": "<p>This instruction tests the state of the carry bit and takes a conditional branch if the carry bit is reset.</p>",
                "tooltip": "Branch on Carry Clear",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#BCC",
            };

        case "BCS":
            return {
                "html": "<p>This instruction takes the conditional branch if the carry flag is on.</p>",
                "tooltip": "Branch on Carry Set",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#BCS",
            };

        case "BEQ":
            return {
                "html": "<p>This instruction could also be called \"Branch on Equal.\"</p>",
                "tooltip": "Branch on Result Zero",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#BEQ",
            };

        case "BIT":
            return {
                "html": "<p>This instruction performs an AND between a memory location and the accumulator but does not store the result of the AND into the accumulator.</p>",
                "tooltip": "Test Bits in Memory with Accumulator",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#BIT",
            };

        case "BMI":
            return {
                "html": "<p>This instruction takes the conditional branch if the N bit is set.</p>",
                "tooltip": "Branch on Result Minus",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#BMI",
            };

        case "BNE":
            return {
                "html": "<p>This instruction could also be called \"Branch on Not Equal.\" It tests the Z flag and takes the conditional branch if the Z flag is not on, indicating that the previous result was not zero.</p>",
                "tooltip": "Branch on Result Not Zero",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#BNE",
            };

        case "BPL":
            return {
                "html": "<p>This instruction is the complementary branch to branch on result minus. It is a conditional branch which takes the branch when the N bit is reset (0). BPL is used to test if the previous result bit 7 was off (0) and branch on result minus is used to determine if the previous result was minus or bit 7 was on (1).</p>",
                "tooltip": "Branch on Result Plus",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#BPL",
            };

        case "BRK":
            return {
                "html": "<p>The break command causes the microprocessor to go through an inter rupt sequence under program control. This means that the program counter of the second byte after the BRK. is automatically stored on the stack along with the processor status at the beginning of the break instruction. The microprocessor then transfers control to the interrupt vector.</p>",
                "tooltip": "Break Command",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#BRK",
            };

        case "BVC":
            return {
                "html": "<p>This instruction tests the status of the V flag and takes the conditional branch if the flag is not set.</p>",
                "tooltip": "Branch on Overflow Clear",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#BVC",
            };

        case "BVS":
            return {
                "html": "<p>This instruction tests the V flag and takes the conditional branch if V is on.</p>",
                "tooltip": "Branch on Overflow Set",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#BVS",
            };

        case "CLC":
            return {
                "html": "<p>This instruction initializes the carry flag to a 0. This op eration should normally precede an ADC loop. It is also useful when used with a R0L instruction to clear a bit in memory.</p>",
                "tooltip": "Clear Carry Flag",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#CLC",
            };

        case "CLD":
            return {
                "html": "<p>This instruction sets the decimal mode flag to a 0. This all subsequent ADC and SBC instructions to operate as simple operations.</p>",
                "tooltip": "Clear Decimal Mode",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#CLD",
            };

        case "CLI":
            return {
                "html": "<p>This instruction initializes the interrupt disable to a 0. This allows the microprocessor to receive interrupts.</p>",
                "tooltip": "Clear Interrupt Disable",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#CLI",
            };

        case "CLV":
            return {
                "html": "<p>This instruction clears the overflow flag to a 0. This com mand is used in conjunction with the set overflow pin which can change the state of the overflow flag with an external signal.</p>",
                "tooltip": "Clear Overflow Flag",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#CLV",
            };

        case "CMP":
            return {
                "html": "<p>This instruction subtracts the contents of memory from the contents of the accumulator.</p>",
                "tooltip": "Compare Memory and Accumulator",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#CMP",
            };

        case "CPX":
            return {
                "html": "<p>This instruction subtracts the value of the addressed memory location from the content of index register X using the adder but does not store the result; therefore, its only use is to set the N, Z and C flags to allow for comparison between the index register X and the value in memory.</p>",
                "tooltip": "Compare Index Register X To Memory",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#CPX",
            };

        case "CPY":
            return {
                "html": "<p>This instruction performs a two's complement subtraction between the index register Y and the specified memory location. The results of the subtraction are not stored anywhere. The instruction is strictly used to set the flags.</p>",
                "tooltip": "Compare Index Register Y To Memory",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#CPY",
            };

        case "DEC":
            return {
                "html": "<p>This instruction subtracts 1, in two's complement, from the contents of the addressed memory location.</p>",
                "tooltip": "Decrement Memory By One",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#DEC",
            };

        case "DEX":
            return {
                "html": "<p>This instruction subtracts one from the current value of the index register X and stores the result in the index register X.</p>",
                "tooltip": "Decrement Index Register X By One",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#DEX",
            };

        case "DEY":
            return {
                "html": "<p>This instruction subtracts one from the current value in the in dex register Y and stores the result into the index register Y. The result does not affect or consider carry so that the value in the index register Y is decremented to 0 and then through 0 to FF.</p>",
                "tooltip": "Decrement Index Register Y By One",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#DEY",
            };

        case "EOR":
            return {
                "html": "<p>The EOR instruction transfers the memory and the accumulator to the adder which performs a binary \"EXCLUSIVE OR\" on a bit-by-bit basis and stores the result in the accumulator.</p>",
                "tooltip": "\"Exclusive OR\" Memory with Accumulator",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#EOR",
            };

        case "INC":
            return {
                "html": "<p>This instruction adds 1 to the contents of the addressed memory location.</p>",
                "tooltip": "Increment Memory By One",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#INC",
            };

        case "INX":
            return {
                "html": "<p>Increment X adds 1 to the current value of the X register. This is an 8-bit increment which does not affect the carry operation, therefore, if the value of X before the increment was FF, the resulting value is 00.</p>",
                "tooltip": "Increment Index Register X By One",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#INX",
            };

        case "INY":
            return {
                "html": "<p>Increment Y increments or adds one to the current value in the Y register, storing the result in the Y register. As in the case of INX the primary application is to step thru a set of values using the Y register.</p>",
                "tooltip": "Increment Index Register Y By One",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#INY",
            };

        case "JMP":
            return {
                "html": "<p>In this instruction, the data from the memory location located in the program sequence after the OP CODE is loaded into the low order byte of the program counter (PCL) and the data from the next memory location after that is loaded into the high order byte of the program counter (PCH).</p>",
                "tooltip": "JMP Indirect",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#JMP",
            };

        case "JSR":
            return {
                "html": "<p>This instruction transfers control of the program counter to a subroutine location but leaves a return pointer on the stack to allow the user to return to perform the next instruction in the main program after the subroutine is complete. To accomplish this, JSR instruction stores the program counter address which points to the last byte of the jump instruc tion onto the stack using the stack pointer. The stack byte contains the program count high first, followed by program count low. The JSR then transfers the addresses following the jump instruction to the program counter low and the program counter high, thereby directing the program to begin at that new address.</p>",
                "tooltip": "Jump To Subroutine",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#JSR",
            };

        case "LDA":
            return {
                "html": "<p>When instruction LDA is executed by the microprocessor, data is transferred from memory to the accumulator and stored in the accumulator.</p>",
                "tooltip": "Load Accumulator with Memory",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#LDA",
            };

        case "LDX":
            return {
                "html": "<p>Load the index register X from memory.</p>",
                "tooltip": "Load Index Register X From Memory",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#LDX",
            };

        case "LDY":
            return {
                "html": "<p>Load the index register Y from memory.</p>",
                "tooltip": "Load Index Register Y From Memory",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#LDY",
            };

        case "LSR":
            return {
                "html": "<p>This instruction shifts either the accumulator or a specified memory location 1 bit to the right, with the higher bit of the result always being set to 0, and the low bit which is shifted out of the field being stored in the carry flag.</p>",
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
                "html": "<p>The ORA instruction transfers the memory and the accumulator to the adder which performs a binary \"OR\" on a bit-by-bit basis and stores the result in the accumulator.</p>",
                "tooltip": "\"OR\" Memory with Accumulator",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#ORA",
            };

        case "PHA":
            return {
                "html": "<p>This instruction transfers the current value of the accumulator to the next location on the stack, automatically decrementing the stack to point to the next empty location.</p>",
                "tooltip": "Push Accumulator On Stack",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#PHA",
            };

        case "PHP":
            return {
                "html": "<p>This instruction transfers the contents of the processor status reg ister unchanged to the stack, as governed by the stack pointer.</p>",
                "tooltip": "Push Processor Status On Stack",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#PHP",
            };

        case "PLA":
            return {
                "html": "<p>This instruction adds 1 to the current value of the stack pointer and uses it to address the stack and loads the contents of the stack into the A register.</p>",
                "tooltip": "Pull Accumulator From Stack",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#PLA",
            };

        case "PLP":
            return {
                "html": "<p>This instruction transfers the next value on the stack to the Proces sor Status register, thereby changing all of the flags and setting the mode switches to the values from the stack.</p>",
                "tooltip": "Pull Processor Status From Stack",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#PLP",
            };

        case "ROL":
            return {
                "html": "<p>The rotate left instruction shifts either the accumulator or addressed memory left 1 bit, with the input carry being stored in bit 0 and with the input bit 7 being stored in the carry flags.</p>",
                "tooltip": "Rotate Left",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#ROL",
            };

        case "ROR":
            return {
                "html": "<p>The rotate right instruction shifts either the accumulator or addressed memory right 1 bit with bit 0 shifted into the carry and carry shifted into bit 7.</p>",
                "tooltip": "Rotate Right",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#ROR",
            };

        case "RTI":
            return {
                "html": "<p>This instruction transfers from the stack into the microprocessor the processor status and the program counter location for the instruction which was interrupted. By virtue of the interrupt having stored this data before executing the instruction and thei fact that the RTI reinitializes the microprocessor to the same state as when it was interrupted, the combination of interrupt plus RTI allows truly reentrant coding.</p>",
                "tooltip": "Return From Interrupt",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#RTI",
            };

        case "RTS":
            return {
                "html": "<p>This instruction loads the program count low and program count high from the stack into the program counter and increments the program counter so that it points to the instruction following the JSR. The stack pointer is adjusted by incrementing it twice.</p>",
                "tooltip": "Return From Subroutme",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#RTS",
            };

        case "SBC":
            return {
                "html": "<p>This instruction subtracts the value of memory and borrow from the value of the accumulator, using two's complement arithmetic, and stores the result in the accumulator. Borrow is defined as the carry flag complemented; therefore, a resultant carry flag indicates that a borrow has not occurred.</p>",
                "tooltip": "Subtract Memory from Accumulator with Borrow",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#SBC",
            };

        case "SEC":
            return {
                "html": "<p>This instruction initializes the carry flag to a 1. This op eration should normally precede a SBC loop. It is also useful when used with a ROL instruction to initialize a bit in memory to a 1.</p>",
                "tooltip": "Set Carry Flag",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#SEC",
            };

        case "SED":
            return {
                "html": "<p>This instruction sets the decimal mode flag D to a 1. This makes all subsequent ADC and SBC instructions operate as a decimal arithmetic operation.</p>",
                "tooltip": "Set Decimal Mode",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#SED",
            };

        case "SEI":
            return {
                "html": "<p>This instruction initializes the interrupt disable to a 1. It is used to mask interrupt requests during system reset operations and during interrupt commands.</p>",
                "tooltip": "Set Interrupt Disable",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#SEI",
            };

        case "STA":
            return {
                "html": "<p>This instruction transfers the contents of the accumulator to memory.</p>",
                "tooltip": "Store Accumulator in Memory",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#STA",
            };

        case "STX":
            return {
                "html": "<p>Transfers value of X register to addressed memory location.</p>",
                "tooltip": "Store Index Register X In Memory",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#STX",
            };

        case "STY":
            return {
                "html": "<p>Transfer the value of the Y register to the addressed memory location.</p>",
                "tooltip": "Store Index Register Y In Memory",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#STY",
            };

        case "TAX":
            return {
                "html": "<p>This instruction takes the value from accumulator A and trans fers or loads it into the index register X without disturbing the content of the accumulator A.</p>",
                "tooltip": "Transfer Accumulator To Index X",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#TAX",
            };

        case "TAY":
            return {
                "html": "<p>This instruction moves the value of the accumulator into index register Y without affecting the accumulator.</p>",
                "tooltip": "Transfer Accumula Tor To Index Y",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#TAY",
            };

        case "TSX":
            return {
                "html": "<p>This instruction transfers the value in the stack pointer to the index register X.</p>",
                "tooltip": "Transfer Stack Pointer To Index X",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#TSX",
            };

        case "TXA":
            return {
                "html": "<p>This instruction moves the value that is in the index register X to the accumulator A without disturbing the content of the index register X.</p>",
                "tooltip": "Transfer Index X To Accumulator",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#TXA",
            };

        case "TXS":
            return {
                "html": "<p>This instruction transfers the value in the index register X to the stack pointer.</p>",
                "tooltip": "Transfer Index X To Stack Pointer",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#TXS",
            };

        case "TYA":
            return {
                "html": "<p>This instruction moves the value that is in the index register Y to accumulator A without disturbing the content of the register Y.</p>",
                "tooltip": "Transfer Index Y To Accumulator",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#TYA",
            };

        case "SAX":
            return {
                "html": "<p>The undocumented SAX instruction performs a bit-by-bit AND operation of the accumulator and the X register and transfers the result to the addressed memory location.</p>",
                "tooltip": "Store Accumulator \"AND\" Index Register X in Memory",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#SAX",
            };

        case "SHA":
            return {
                "html": "<p>The undocumented SHA instruction performs a bit-by-bit AND operation of the following three operands: The first two are the accumulator and the index register X.</p>",
                "tooltip": "Store Accumulator \"AND\" Index Register X \"AND\" Value",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#SHA",
            };

        case "ASR":
            return {
                "html": "<p>The undocumented ASR instruction performs a bit-by-bit AND operation of the accumulator and memory, then shifts the accumulator 1 bit to the right, with the higher bit of the result always being set to 0, and the low bit which is shifted out of the field being stored in the carry flag.</p>",
                "tooltip": "\"AND\" then Logical Shift Right",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#ASR",
            };

        case "ANC":
            return {
                "html": "<p>The undocumented ANC instruction performs a bit-by-bit AND operation of the accumulator and memory and stores the result back in the accumulator.</p>",
                "tooltip": "\"AND\" Memory with Accumulator then Move Negative Flag to Carry Flag",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#ANC",
            };

        case "ARR":
            return {
                "html": "<p>The undocumented ARR instruction performs a bit-by-bit \"AND\" operation of the accumulator and memory, then shifts the result right 1 bit with bit 0 shifted into the carry and carry shifted into bit 7. It then stores the result back in the accumulator.</p>",
                "tooltip": "\"AND\" Accumulator then Rotate Right",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#ARR",
            };

        case "SBX":
            return {
                "html": "<p>This undocumented instruction performs a bit-by-bit \"AND\" of the value of the accumulator and the index register X and subtracts the value of memory from this result, using two's complement arithmetic, and stores the result in the index register X.</p>",
                "tooltip": "Subtract Memory from Accumulator \"AND\" Index Register X",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#SBX",
            };

        case "DCP":
            return {
                "html": "<p>This undocumented instruction subtracts 1, in two's complement, from the contents of the addressed memory location. It then subtracts the contents of memory from the contents of the accumulator.</p>",
                "tooltip": "Decrement Memory By One then Compare with Accumulator",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#DCP",
            };

        case "ISC":
            return {
                "html": "<p>This undocumented instruction adds 1 to the contents of the addressed memory location. It then subtracts the value of the result in memory and borrow from the value of the accumulator, using two's complement arithmetic, and stores the result in the accumulator.</p>",
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
                "html": "<p>This undocumented instruction performs a bit-by-bit \"AND\" operation of the stack pointer and memory and stores the result back in the accumulator, the index register X and the stack pointer.</p>",
                "tooltip": "\"AND\" Memory with Stack Pointer",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#LAS",
            };

        case "LAX":
            return {
                "html": "<p>The undocumented LAX instruction loads the accumulator and the index register X from memory.</p>",
                "tooltip": "Load Accumulator and Index Register X From Memory",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#LAX",
            };

        case "RLA":
            return {
                "html": "<p>The undocumented RLA instruction shifts the addressed memory left 1 bit, with the input carry being stored in bit 0 and with the input bit 7 being stored in the carry flags. It then performs a bit-by-bit AND operation of the result and the value of the accumulator and stores the result back in the accumulator.</p>",
                "tooltip": "Rotate Left then \"AND\" with Accumulator",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#RLA",
            };

        case "RRA":
            return {
                "html": "<p>The undocumented RRA instruction shifts the addressed memory right 1 bit with bit 0 shifted into the carry and carry shifted into bit 7. It then adds the result and generated carry to the value of the accumulator and stores the result in the accumulator.</p>",
                "tooltip": "Rotate Right and Add Memory to Accumulator",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#RRA",
            };

        case "SHX":
            return {
                "html": "<p>The undocumented SHX instruction performs a bit-by-bit AND operation of the index register X and the upper 8 bits of the given address (ignoring the the addressing mode's Y offset), plus 1. It then transfers the result to the addressed memory location.</p>",
                "tooltip": "Store Index Register X \"AND\" Value",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#SHX",
            };

        case "SHY":
            return {
                "html": "<p>The undocumented SHY instruction performs a bit-by-bit AND operation of the index register Y and the upper 8 bits of the given address (ignoring the the addressing mode's X offset), plus 1. It then transfers the result to the addressed memory location.</p>",
                "tooltip": "Store Index Register Y \"AND\" Value",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#SHY",
            };

        case "SLO":
            return {
                "html": "<p>The undocumented SLO instruction shifts the address memory location 1 bit to the left, with the bit 0 always being set to 0 and the bit 7 output always being contained in the carry flag. It then performs a bit-by-bit \"OR\" operation on the result and the accumulator and stores the result in the accumulator.</p>",
                "tooltip": "Arithmetic Shift Left then \"OR\" Memory with Accumulator",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#SLO",
            };

        case "SRE":
            return {
                "html": "<p>The undocumented SRE instruction shifts the specified memory location 1 bit to the right, with the higher bit of the result always being set to 0, and the low bit which is shifted out of the field being stored in the carry flag. It then performs a bit-by-bit \"EXCLUSIVE OR\" of the result and the value of the accumulator and stores the result in the accumulator.</p>",
                "tooltip": "Logical Shift Right then \"Exclusive OR\" Memory with Accumulator",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#SRE",
            };

        case "SHS":
            return {
                "html": "<p>The undocumented SHS instruction performs a bit-by-bit AND operation of the value of the accumulator and the value of the index register X and stores the result in the stack pointer. It then performs a bit-by-bit AND operation of the resulting stack pointer and the upper 8 bits of the given address (ignoring the addressing mode's Y offset), plus 1, and transfers the result to the addressed memory location.</p>",
                "tooltip": "Transfer Accumulator \"AND\" Index Register X to Stack Pointer then Store Stack Pointer \"AND\" Hi-Byte In Memory",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#SHS",
            };

        case "XAA":
            return {
                "html": "<p>The operation of the undocumented XAA instruction depends on the individual microprocessor. On most machines, it performs a bit-by-bit AND operation of the following three operands: The first two are the index register X and memory.</p>",
                "tooltip": "Non-deterministic Operation of Accumulator, Index Register X, Memory and Bus Contents",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=6502&tab=2#XAA",
            };

        case "BRA":
            return {
                "html": "<p>This instruction takes an unconditional branch.</p>",
                "tooltip": "Branch Always",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c02&tab=2#BRA",
            };

        case "PHX":
            return {
                "html": "<p>This instruction transfers the current value of the index register X to the next location on the stack, automatically decrementing the stack to point to the next empty location.</p>",
                "tooltip": "Push Index Register X On Stack",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c02&tab=2#PHX",
            };

        case "PHY":
            return {
                "html": "<p>This instruction transfers the current value of the index register Y to the next location on the stack, automatically decrementing the stack to point to the next empty location.</p>",
                "tooltip": "Push Index Register Y On Stack",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c02&tab=2#PHY",
            };

        case "PLX":
            return {
                "html": "<p>This instruction adds 1 to the current value of the stack pointer and uses it to address the stack and loads the contents of the stack into the X register.</p>",
                "tooltip": "Pull Index Register X From Stack",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c02&tab=2#PLX",
            };

        case "PLY":
            return {
                "html": "<p>This instruction adds 1 to the current value of the stack pointer and uses it to address the stack and loads the contents of the stack into the Y register.</p>",
                "tooltip": "Pull Index Register Y From Stack",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c02&tab=2#PLY",
            };

        case "STZ":
            return {
                "html": "<p>Transfers the value 0 to addressed memory location.</p>",
                "tooltip": "Store Zero In Memory",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c02&tab=2#STZ",
            };

        case "TRB":
            return {
                "html": "<p>This instruction tests and resets bits in memory, using the accumulator for both a test mask, and a reset mask. It performs a logical AND between the inverted bits of the accumulator and the bits in memory, storing the result back into memory.</p>",
                "tooltip": "Test And Reset Memory Bits With Accumulator",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c02&tab=2#TRB",
            };

        case "TSB":
            return {
                "html": "<p>This instruction tests and sets bits in memory, using the accumulator for both a test mask, and a set mask. It performs a logical OR between the bits of the accumulator and the bits in memory, storing the result back into memory.</p>",
                "tooltip": "Test And Set Memory Bits With Accumulator",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c02&tab=2#TSB",
            };

    }
}