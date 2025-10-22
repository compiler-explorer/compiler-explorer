// Copyright (c) 2025, Compiler Explorer Authors
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

export class XtensaInstructionSetInfo extends BaseInstructionSetInfo {
    static override get key(): InstructionSet {
        return 'xtensa';
    }

    // Full list:  BALL, BNALL, BANY, BNONE, BBC, BBCI, BBS, BBSI, BEQ, BEQI, BEQZ, BNE, BNEI, BNEZ
    // BGE, BGEI, BGEU, BGEUI, BGEZ, BLT, BLTI, BLTU, BLTUI, BLTZ
    static conditionalJumps = /^\s*b/;

    // whitespaces followed by 'j' or 'jx'
    static unconditionalJumps = /^\s*jx?/;
    static returnInstruction = /^\s*ret/;

    override isJmpInstruction(instruction: string) {
        const instype = this.getInstructionType(instruction);
        return instype === InstructionType.jmp || instype === InstructionType.conditionalJmpInst;
    }

    override getInstructionType(instruction: string) {
        if (XtensaInstructionSetInfo.conditionalJumps.test(instruction)) return InstructionType.conditionalJmpInst;
        if (XtensaInstructionSetInfo.unconditionalJumps.test(instruction)) return InstructionType.jmp;
        if (XtensaInstructionSetInfo.returnInstruction.test(instruction)) return InstructionType.retInst;
        return InstructionType.notRetInst;
    }
}
