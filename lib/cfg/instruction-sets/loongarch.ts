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

export class LoongArchInstructionSetInfo extends BaseInstructionSetInfo {
    static override get key(): InstructionSet {
        return 'loongarch';
    }

    // beq, bne, blt, bge, bltu, bgeu, beqz, bnez, the pseudo-instructions bgt, ble, bgtu, bleu, bltz, bgez,
    // bgtz, blez, and the floating-point condition branches bceqz and bcnez
    static conditionalJumps = /^(?:b(?:eq|ne|lt|ge|gt|le)[uz]?|bc(?:eq|ne)z)$/;
    // A tail call leaves the function, so like a return it has no successor within it
    static returnInstructions = /^(?:ret|tail36)$/;
    // Registers may be printed with ABI names (clang) or as $rN (GCC)
    static zeroRegister = /^\$(?:zero|r0)$/;
    static returnAddressRegister = /^\$(?:ra|r1)$/;

    override isJmpInstruction(instruction: string) {
        const instype = this.getInstructionType(instruction);
        return instype === InstructionType.jmp || instype === InstructionType.conditionalJmpInst;
    }

    override getInstructionType(instruction: string) {
        const [opcode, ...operands] = instruction
            .trim()
            .toLowerCase()
            .split(/[\s,]+/);
        if (LoongArchInstructionSetInfo.conditionalJumps.test(opcode)) return InstructionType.conditionalJmpInst;
        if (LoongArchInstructionSetInfo.returnInstructions.test(opcode)) return InstructionType.retInst;
        switch (opcode) {
            case 'b':
                return InstructionType.jmp;
            case 'jr':
                return this.getIndirectJumpType(operands[0]);
            // jr and ret are aliases of jirl linking to $zero, i.e. discarding the return address; otherwise
            // jirl is a call
            case 'jirl':
                return LoongArchInstructionSetInfo.zeroRegister.test(operands[0])
                    ? this.getIndirectJumpType(operands[1])
                    : InstructionType.notRetInst;
        }
        return InstructionType.notRetInst;
    }

    // A jump through ra is a return; through any other register it is an indirect jump (e.g. a jump table)
    private getIndirectJumpType(target = '') {
        return LoongArchInstructionSetInfo.returnAddressRegister.test(target)
            ? InstructionType.retInst
            : InstructionType.jmp;
    }
}
