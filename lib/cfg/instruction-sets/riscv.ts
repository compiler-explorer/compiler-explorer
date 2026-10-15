// Copyright (c) 2026, Compiler Explorer Authors
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

import {InstructionSet} from '../../../types/instructionsets.js';
import {BaseInstructionSetInfo, InstructionType} from './base.js';

export class RiscvInstructionSetInfo extends BaseInstructionSetInfo {
    static override get key(): InstructionSet[] {
        return ['riscv32', 'riscv64'];
    }

    // beq, bne, blt, bge, bltu, bgeu, and the pseudo-instructions bgt, ble, bgtu, bleu,
    // beqz, bnez, bltz, bgez, bgtz, blez (plus the compressed c.beqz, c.bnez)
    static conditionalJumps = /^(?:c\.)?b(?:eq|ne|lt|ge|gt|le)[uz]?$/;
    // A tail call leaves the function, so like a return it has no successor within it
    static returnInstructions = /^(?:[ms]?ret|tail)$/;
    // Registers may be printed with ABI or architectural names (clang's -riscv-arch-reg-names)
    static zeroRegister = /^(?:zero|x0)$/;
    static returnAddressRegister = /^(?:ra|x1)$/;

    override isJmpInstruction(instruction: string) {
        const instype = this.getInstructionType(instruction);
        return instype === InstructionType.jmp || instype === InstructionType.conditionalJmpInst;
    }

    override getInstructionType(instruction: string) {
        const [opcode, ...operands] = instruction
            .trim()
            .toLowerCase()
            .split(/[\s,]+/);
        if (RiscvInstructionSetInfo.conditionalJumps.test(opcode)) return InstructionType.conditionalJmpInst;
        if (RiscvInstructionSetInfo.returnInstructions.test(opcode)) return InstructionType.retInst;
        // j and jr are aliases of jal and jalr that link to zero, i.e. discard the return address.
        // The single-operand forms of jal and jalr link to ra, so they are calls.
        const linksToZero = operands.length > 1 && RiscvInstructionSetInfo.zeroRegister.test(operands[0]);
        switch (opcode.replace(/^c\./, '')) {
            case 'j':
                return InstructionType.jmp;
            case 'jal':
                return linksToZero ? InstructionType.jmp : InstructionType.notRetInst;
            case 'jr':
                return this.getIndirectJumpType(operands[0]);
            case 'jalr':
                return linksToZero ? this.getIndirectJumpType(operands[1]) : InstructionType.notRetInst;
        }
        return InstructionType.notRetInst;
    }

    // A jump through ra is a return; through any other register it is an indirect jump (e.g. a jump table).
    // The register may be written as a base address, e.g. 0(ra).
    private getIndirectJumpType(target = '') {
        const register = target.replace(/^.*\((\w+)\)$/, '$1');
        return RiscvInstructionSetInfo.returnAddressRegister.test(register)
            ? InstructionType.retInst
            : InstructionType.jmp;
    }
}
