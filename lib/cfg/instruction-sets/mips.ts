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

export class MipsInstructionSetInfo extends BaseInstructionSetInfo {
    static override get key(): InstructionSet {
        return 'mips';
    }

    static conditionalJumps = new RegExp(
        '^(?:' +
            [
                // beq, bne, beqz, bnez, bltz, blez, bgtz, bgez, the pseudo-instructions blt, ble, bgt, bge and their
                // unsigned forms, each possibly branch-likely (beql) or compact (bnezc, bltuc), and nanoMIPS's
                // compares with an immediate (beqic, bgeiuc)
                'b(?:eq|ne|lt|le|gt|ge)(?:z|u|iu?)?[lc]?',
                // Floating-point and coprocessor 2 conditions: bc1t, bc1fl, release 6's bc1eqz
                'bc[12](?:[tf]l?|eqz|nez)',
                'b[no]vc',
                'bposge(?:32|64)',
                // MSA
                'bn?z\\.[bhwdv]',
                // Octeon and nanoMIPS bit tests
                'bbit[01](?:32)?',
                'bb(?:eq|ne)zc',
                // MIPS16 compares of the T register
                'bt(?:eq|ne)z',
            ].join('|') +
            ')$',
    );
    static unconditionalJumps = /^(?:b|bc|brsc)$/;
    // To a label or through a register; through $ra they return
    static jumps = /^(?:j|jr|jr\.hb|jrc|jic)$/;
    // Calls, unless they link to $zero, i.e. discard the return address
    static linkingJumps = /^(?:jalr|jalr\.hb|jalrc|jalrc\.hb)$/;
    static returnInstructions = /^(?:jraddiusp|jrcaddiusp|restore\.jrc|eret|eretnc|deret)$/;
    // Compact branches (release 6's bc, beqzc, jrc, jialc and the like), MIPS16's T register branches and
    // exception returns
    static noDelaySlot = /(?:c|c\.hb)$|^(?:jraddiusp|jrcaddiusp|eret|deret|bt(?:eq|ne)z)$/;
    static callsWithDelaySlot = /^(?:jal|jalr(?:\.hb)?|jalrs(?:\.hb)?|jals|jalx|bal|b(?:ge|lt)zall?)$/;
    static zeroRegister = /^\$(?:0|zero)$/;
    static returnAddressRegister = /^\$(?:31|ra)$/;

    // GCC and clang give o32 local labels a $ prefix ($L3, $BB0_2); n32 and n64 use .L like other targets
    override get localLabelPrefixes() {
        return ['.', '$'];
    }

    override isJmpInstruction(instruction: string) {
        const instype = this.getInstructionType(instruction);
        return instype === InstructionType.jmp || instype === InstructionType.conditionalJmpInst;
    }

    override getInstructionType(instruction: string) {
        const [opcode, ...operands] = this.parse(instruction);
        if (MipsInstructionSetInfo.conditionalJumps.test(opcode)) return InstructionType.conditionalJmpInst;
        if (MipsInstructionSetInfo.returnInstructions.test(opcode)) return InstructionType.retInst;
        if (MipsInstructionSetInfo.unconditionalJumps.test(opcode)) return InstructionType.jmp;
        if (MipsInstructionSetInfo.jumps.test(opcode)) return this.getIndirectJumpType(operands[0]);
        if (
            MipsInstructionSetInfo.linkingJumps.test(opcode) &&
            operands.length > 1 &&
            MipsInstructionSetInfo.zeroRegister.test(operands[0])
        ) {
            return this.getIndirectJumpType(operands[1]);
        }
        return InstructionType.notRetInst;
    }

    override hasDelaySlot(instruction: string) {
        const [opcode] = this.parse(instruction);
        if (MipsInstructionSetInfo.noDelaySlot.test(opcode)) return false;
        return (
            MipsInstructionSetInfo.callsWithDelaySlot.test(opcode) ||
            this.getInstructionType(instruction) !== InstructionType.notRetInst
        );
    }

    // GCC labels some PIC calls (`1:      jalr    $25`), and microMIPS's 16-bit encodings carry a 16 suffix (beqz16)
    private parse(instruction: string) {
        const [opcode, ...operands] = instruction
            .trim()
            .replace(/^\d+:\s*/, '')
            .toLowerCase()
            .split(/[\s,]+/);
        return [opcode.replace(/16$/, ''), ...operands];
    }

    private getIndirectJumpType(target = '') {
        return MipsInstructionSetInfo.returnAddressRegister.test(target)
            ? InstructionType.retInst
            : InstructionType.jmp;
    }
}
