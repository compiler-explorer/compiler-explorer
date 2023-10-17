// Copyright (c) 2023, Compiler Explorer Authors
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

export class ArmInstructionSetInfo extends BaseInstructionSetInfo {
    static conditions = `(?:${[
        'eq',
        'ne',
        'cs',
        'hs',
        'cc',
        'lo',
        'mi',
        'pl',
        'vs',
        'vc',
        'hi',
        'ls',
        'ge',
        'lt',
        'gt',
        'le',
        'al',
    ].join('|')})`;
    // handling:
    // bcc     label
    // bxcc    reg
    // popcc   {..., pc}
    // popcc   {..., tmp}; bxcc tmp
    // mov     pc, reg
    // currently not handling:
    // blcc    label
    // blxcc   label
    // blxcc   reg
    // movcc   pc, reg
    static conditionalJumps = new RegExp(
        '\\b(?:' +
            [
                `b\\.?${ArmInstructionSetInfo.conditions}(?:\\.w)?`,
                `bx${ArmInstructionSetInfo.conditions}`,
                `bxj${ArmInstructionSetInfo.conditions}`,
                `cbz`,
                `cbnz`,
                `tbz`,
                `tbnz`,
            ]
                .map(re => `(?:${re})`)
                .join('|') +
            ')\\b',
    );
    static unconditionalJumps = new RegExp(
        '\\b(?:' + [`b(?:\\.w)?`, `bx`, `bxj`].map(re => `(?:${re})`).join('|') + ')\\b',
    );
    static returnInstruction = new RegExp(
        '(?:' +
            [`bx`, `ret`].map(re => `(?:${re})`).join('|') +
            ')\\b.+' +
            `|pop\\s*\\{(?:r(?:\\d{2,}|[4-9]),\\s*)*pc\\}.+` +
            `|mov\\s*pc\\s*,.+`,
    );

    static override get key(): InstructionSet[] {
        return ['arm32', 'aarch64'];
    }

    override isJmpInstruction(instruction: string) {
        const opcode = instruction.trim().split(' ')[0].toLowerCase();
        return (
            !!opcode.match(ArmInstructionSetInfo.conditionalJumps) ||
            !!opcode.match(ArmInstructionSetInfo.unconditionalJumps)
        );
    }

    override getInstructionType(instruction: string) {
        const opcode = instruction.trim().split(' ')[0].toLowerCase();
        if (opcode.match(ArmInstructionSetInfo.unconditionalJumps)) return InstructionType.jmp;
        else if (opcode.match(ArmInstructionSetInfo.conditionalJumps)) return InstructionType.conditionalJmpInst;
        else if (instruction.trim().toLocaleLowerCase().match(ArmInstructionSetInfo.returnInstruction)) {
            return InstructionType.retInst;
        } else {
            return InstructionType.notRetInst;
        }
    }
}
