import {AssemblyInstructionInfo} from '../base.js';

export function getAsmOpcode(opcode: string | undefined): AssemblyInstructionInfo | undefined {
    if (!opcode) return;
    switch (opcode.toUpperCase()) {
        case "ADC":
            return {
                "html": "<p>Add Memory to Accumulator with Carry</p>",
                "tooltip": "Add Memory to Accumulator with Carry",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#ADC",
            };

        case "AND":
            return {
                "html": "<p>\"AND\" Memory with Accumulator</p>",
                "tooltip": "\"AND\" Memory with Accumulator",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#AND",
            };

        case "ASL":
            return {
                "html": "<p>Arithmetic Shift Left</p>",
                "tooltip": "Arithmetic Shift Left",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#ASL",
            };

        case "BCC":
            return {
                "html": "<p>Branch on Carry Clear</p>",
                "tooltip": "Branch on Carry Clear",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#BCC",
            };

        case "BCS":
            return {
                "html": "<p>Branch on Carry Set</p>",
                "tooltip": "Branch on Carry Set",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#BCS",
            };

        case "BEQ":
            return {
                "html": "<p>Branch on Result Zero</p>",
                "tooltip": "Branch on Result Zero",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#BEQ",
            };

        case "BIT":
            return {
                "html": "<p>Test Bits in Memory with Accumulator</p>",
                "tooltip": "Test Bits in Memory with Accumulator",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#BIT",
            };

        case "BMI":
            return {
                "html": "<p>Branch on Result Minus</p>",
                "tooltip": "Branch on Result Minus",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#BMI",
            };

        case "BNE":
            return {
                "html": "<p>Branch on Result Not Zero</p>",
                "tooltip": "Branch on Result Not Zero",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#BNE",
            };

        case "BPL":
            return {
                "html": "<p>Branch on Result Plus</p>",
                "tooltip": "Branch on Result Plus",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#BPL",
            };

        case "BRK":
            return {
                "html": "<p>Break Command</p>",
                "tooltip": "Break Command",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#BRK",
            };

        case "BVC":
            return {
                "html": "<p>Branch on Overflow Clear</p>",
                "tooltip": "Branch on Overflow Clear",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#BVC",
            };

        case "BVS":
            return {
                "html": "<p>Branch on Overflow Set</p>",
                "tooltip": "Branch on Overflow Set",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#BVS",
            };

        case "CLC":
            return {
                "html": "<p>Clear Carry Flag</p>",
                "tooltip": "Clear Carry Flag",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#CLC",
            };

        case "CLD":
            return {
                "html": "<p>Clear Decimal Mode</p>",
                "tooltip": "Clear Decimal Mode",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#CLD",
            };

        case "CLI":
            return {
                "html": "<p>Clear Interrupt Disable</p>",
                "tooltip": "Clear Interrupt Disable",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#CLI",
            };

        case "CLV":
            return {
                "html": "<p>Clear Overflow Flag</p>",
                "tooltip": "Clear Overflow Flag",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#CLV",
            };

        case "CMP":
            return {
                "html": "<p>Compare Memory and Accumulator</p>",
                "tooltip": "Compare Memory and Accumulator",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#CMP",
            };

        case "CPX":
            return {
                "html": "<p>Compare Index Register X To Memory</p>",
                "tooltip": "Compare Index Register X To Memory",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#CPX",
            };

        case "CPY":
            return {
                "html": "<p>Compare Index Register Y To Memory</p>",
                "tooltip": "Compare Index Register Y To Memory",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#CPY",
            };

        case "DEC":
            return {
                "html": "<p>Decrement Memory By One</p>",
                "tooltip": "Decrement Memory By One",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#DEC",
            };

        case "DEX":
            return {
                "html": "<p>Decrement Index Register X By One</p>",
                "tooltip": "Decrement Index Register X By One",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#DEX",
            };

        case "DEY":
            return {
                "html": "<p>Decrement Index Register Y By One</p>",
                "tooltip": "Decrement Index Register Y By One",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#DEY",
            };

        case "EOR":
            return {
                "html": "<p>\"Exclusive OR\" Memory with Accumulator</p>",
                "tooltip": "\"Exclusive OR\" Memory with Accumulator",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#EOR",
            };

        case "INC":
            return {
                "html": "<p>Increment Memory By One</p>",
                "tooltip": "Increment Memory By One",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#INC",
            };

        case "INX":
            return {
                "html": "<p>Increment Index Register X By One</p>",
                "tooltip": "Increment Index Register X By One",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#INX",
            };

        case "INY":
            return {
                "html": "<p>Increment Index Register Y By One</p>",
                "tooltip": "Increment Index Register Y By One",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#INY",
            };

        case "JMP":
            return {
                "html": "<p>JMP Indirect</p>",
                "tooltip": "JMP Indirect",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#JMP",
            };

        case "JSR":
            return {
                "html": "<p>Jump To Subroutine</p>",
                "tooltip": "Jump To Subroutine",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#JSR",
            };

        case "LDA":
            return {
                "html": "<p>Load Accumulator with Memory</p>",
                "tooltip": "Load Accumulator with Memory",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#LDA",
            };

        case "LDX":
            return {
                "html": "<p>Load Index Register X From Memory</p>",
                "tooltip": "Load Index Register X From Memory",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#LDX",
            };

        case "LDY":
            return {
                "html": "<p>Load Index Register Y From Memory</p>",
                "tooltip": "Load Index Register Y From Memory",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#LDY",
            };

        case "LSR":
            return {
                "html": "<p>Logical Shift Right</p>",
                "tooltip": "Logical Shift Right",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#LSR",
            };

        case "NOP":
            return {
                "html": "<p>No Operation</p>",
                "tooltip": "No Operation",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#NOP",
            };

        case "ORA":
            return {
                "html": "<p>\"OR\" Memory with Accumulator</p>",
                "tooltip": "\"OR\" Memory with Accumulator",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#ORA",
            };

        case "PHA":
            return {
                "html": "<p>Push Accumulator On Stack</p>",
                "tooltip": "Push Accumulator On Stack",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#PHA",
            };

        case "PHP":
            return {
                "html": "<p>Push Processor Status On Stack</p>",
                "tooltip": "Push Processor Status On Stack",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#PHP",
            };

        case "PLA":
            return {
                "html": "<p>Pull Accumulator From Stack</p>",
                "tooltip": "Pull Accumulator From Stack",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#PLA",
            };

        case "PLP":
            return {
                "html": "<p>Pull Processor Status From Stack</p>",
                "tooltip": "Pull Processor Status From Stack",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#PLP",
            };

        case "ROL":
            return {
                "html": "<p>Rotate Left</p>",
                "tooltip": "Rotate Left",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#ROL",
            };

        case "ROR":
            return {
                "html": "<p>Rotate Right</p>",
                "tooltip": "Rotate Right",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#ROR",
            };

        case "RTI":
            return {
                "html": "<p>Return From Interrupt</p>",
                "tooltip": "Return From Interrupt",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#RTI",
            };

        case "RTS":
            return {
                "html": "<p>Return From Subroutine</p>",
                "tooltip": "Return From Subroutine",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#RTS",
            };

        case "SBC":
            return {
                "html": "<p>Subtract Memory from Accumulator with Borrow</p>",
                "tooltip": "Subtract Memory from Accumulator with Borrow",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#SBC",
            };

        case "SEC":
            return {
                "html": "<p>Set Carry Flag</p>",
                "tooltip": "Set Carry Flag",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#SEC",
            };

        case "SED":
            return {
                "html": "<p>Set Decimal Mode</p>",
                "tooltip": "Set Decimal Mode",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#SED",
            };

        case "SEI":
            return {
                "html": "<p>Set Interrupt Disable</p>",
                "tooltip": "Set Interrupt Disable",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#SEI",
            };

        case "STA":
            return {
                "html": "<p>Store Accumulator in Memory</p>",
                "tooltip": "Store Accumulator in Memory",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#STA",
            };

        case "STX":
            return {
                "html": "<p>Store Index Register X In Memory</p>",
                "tooltip": "Store Index Register X In Memory",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#STX",
            };

        case "STY":
            return {
                "html": "<p>Store Index Register Y In Memory</p>",
                "tooltip": "Store Index Register Y In Memory",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#STY",
            };

        case "TAX":
            return {
                "html": "<p>Transfer Accumulator To Index X</p>",
                "tooltip": "Transfer Accumulator To Index X",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#TAX",
            };

        case "TAY":
            return {
                "html": "<p>Transfer Accumulator To Index Y</p>",
                "tooltip": "Transfer Accumulator To Index Y",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#TAY",
            };

        case "TSX":
            return {
                "html": "<p>Transfer Stack Pointer To Index X</p>",
                "tooltip": "Transfer Stack Pointer To Index X",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#TSX",
            };

        case "TXA":
            return {
                "html": "<p>Transfer Index X To Accumulator</p>",
                "tooltip": "Transfer Index X To Accumulator",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#TXA",
            };

        case "TXS":
            return {
                "html": "<p>Transfer Index X To Stack Pointer</p>",
                "tooltip": "Transfer Index X To Stack Pointer",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#TXS",
            };

        case "TYA":
            return {
                "html": "<p>Transfer Index Y To Accumulator</p>",
                "tooltip": "Transfer Index Y To Accumulator",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#TYA",
            };

        case "BRA":
            return {
                "html": "<p>Branch Always</p>",
                "tooltip": "Branch Always",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#BRA",
            };

        case "PHX":
            return {
                "html": "<p>Push Index Register X On Stack</p>",
                "tooltip": "Push Index Register X On Stack",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#PHX",
            };

        case "PHY":
            return {
                "html": "<p>Push Index Register Y On Stack</p>",
                "tooltip": "Push Index Register Y On Stack",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#PHY",
            };

        case "PLX":
            return {
                "html": "<p>Pull Index Register X From Stack</p>",
                "tooltip": "Pull Index Register X From Stack",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#PLX",
            };

        case "PLY":
            return {
                "html": "<p>Pull Index Register Y From Stack</p>",
                "tooltip": "Pull Index Register Y From Stack",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#PLY",
            };

        case "STZ":
            return {
                "html": "<p>Store Zero In Memory</p>",
                "tooltip": "Store Zero In Memory",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#STZ",
            };

        case "TRB":
            return {
                "html": "<p>Test And Reset Memory Bits With Accumulator</p>",
                "tooltip": "Test And Reset Memory Bits With Accumulator",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#TRB",
            };

        case "TSB":
            return {
                "html": "<p>Test And Set Memory Bits With Accumulator</p>",
                "tooltip": "Test And Set Memory Bits With Accumulator",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#TSB",
            };

        case "BRL":
            return {
                "html": "<p>Branch Long</p>",
                "tooltip": "Branch Long",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#BRL",
            };

        case "COP":
            return {
                "html": "<p>Coprocessor</p>",
                "tooltip": "Coprocessor",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#COP",
            };

        case "JSL":
            return {
                "html": "<p>Jump to Subroutine Long</p>",
                "tooltip": "Jump to Subroutine Long",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#JSL",
            };

        case "MVN":
            return {
                "html": "<p>Move Memory Negative</p>",
                "tooltip": "Move Memory Negative",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#MVN",
            };

        case "MVP":
            return {
                "html": "<p>Move Memory Positive</p>",
                "tooltip": "Move Memory Positive",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#MVP",
            };

        case "PEA":
            return {
                "html": "<p>Push Effective Address</p>",
                "tooltip": "Push Effective Address",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#PEA",
            };

        case "PEI":
            return {
                "html": "<p>Push Effective Indirect Address</p>",
                "tooltip": "Push Effective Indirect Address",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#PEI",
            };

        case "PER":
            return {
                "html": "<p>Push Effective Relative Address</p>",
                "tooltip": "Push Effective Relative Address",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#PER",
            };

        case "PHB":
            return {
                "html": "<p>Push Data Bank Register</p>",
                "tooltip": "Push Data Bank Register",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#PHB",
            };

        case "PHD":
            return {
                "html": "<p>Push Direct Register</p>",
                "tooltip": "Push Direct Register",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#PHD",
            };

        case "PHK":
            return {
                "html": "<p>Push K Register</p>",
                "tooltip": "Push K Register",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#PHK",
            };

        case "PLB":
            return {
                "html": "<p>Pull Data Bank Register</p>",
                "tooltip": "Pull Data Bank Register",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#PLB",
            };

        case "PLD":
            return {
                "html": "<p>Pull Direct Register</p>",
                "tooltip": "Pull Direct Register",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#PLD",
            };

        case "REP":
            return {
                "html": "<p>Reset Processor Status Bits</p>",
                "tooltip": "Reset Processor Status Bits",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#REP",
            };

        case "RTL":
            return {
                "html": "<p>Return From Subroutine Long</p>",
                "tooltip": "Return From Subroutine Long",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#RTL",
            };

        case "SEP":
            return {
                "html": "<p>Set Processor Status Bits</p>",
                "tooltip": "Set Processor Status Bits",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#SEP",
            };

        case "STP":
            return {
                "html": "<p>Stop the Clock</p>",
                "tooltip": "Stop the Clock",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#STP",
            };

        case "TCD":
            return {
                "html": "<p>Transfer C Accumulator to Direct Register</p>",
                "tooltip": "Transfer C Accumulator to Direct Register",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#TCD",
            };

        case "TCS":
            return {
                "html": "<p>Transfer C Accumulator to Stack Pointer</p>",
                "tooltip": "Transfer C Accumulator to Stack Pointer",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#TCS",
            };

        case "TDC":
            return {
                "html": "<p>Transfer Direct Register to C Accumulator</p>",
                "tooltip": "Transfer Direct Register to C Accumulator",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#TDC",
            };

        case "TSC":
            return {
                "html": "<p>Transfer Stack Pointer to C Accumulator</p>",
                "tooltip": "Transfer Stack Pointer to C Accumulator",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#TSC",
            };

        case "TXY":
            return {
                "html": "<p>Transfer X to Y</p>",
                "tooltip": "Transfer X to Y",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#TXY",
            };

        case "TYX":
            return {
                "html": "<p>Transfer Y to X</p>",
                "tooltip": "Transfer Y to X",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#TYX",
            };

        case "WAI":
            return {
                "html": "<p>Wait for Interrupt</p>",
                "tooltip": "Wait for Interrupt",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#WAI",
            };

        case "WDM":
            return {
                "html": "<p>William D. Mensch, Jr.</p>",
                "tooltip": "William D. Mensch, Jr.",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#WDM",
            };

        case "XBA":
            return {
                "html": "<p>Exchange B and A Accumulator</p>",
                "tooltip": "Exchange B and A Accumulator",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#XBA",
            };

        case "XCE":
            return {
                "html": "<p>Exchange Carry and Emulation Flags</p>",
                "tooltip": "Exchange Carry and Emulation Flags",
                "url": "https://www.pagetable.com/c64ref/6502/?cpu=65c816&tab=2#XCE",
            };

    }
}