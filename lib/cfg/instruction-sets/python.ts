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

export class PythonInstructionSetInfo extends BaseInstructionSetInfo {
    static override get key(): InstructionSet {
        return 'python';
    }

    // Jump opcodes obtained on python 3.12 via:
    //   import dis
    //   import opcode
    //   print([opcode.opname[op] for op in dis.hasjump])

    static conditionalJumps = new RegExp(
        ['POP_JUMP_IF_FALSE', 'POP_JUMP_IF_NONE', 'POP_JUMP_IF_NOT_NONE', 'POP_JUMP_IF_TRUE'].join('|'),
    );
    static unconditionalJumps = new RegExp(
        ['JUMP_BACKWARD', 'JUMP_BACKWARD_NO_INTERRUPT', 'JUMP_FORWARD', 'JUMP', 'JUMP_NO_INTERRUPT'].join('|'),
    );
    static returnInstruction = new RegExp(
        ['RETURN_VALUE', 'RETURN_CONST', 'RETURN_GENERATOR', 'YIELD_VALUE', 'SEND'].join('|'),
    );
    override isJmpInstruction(instruction: string) {
        return instruction.includes('JUMP');
    }

    override getInstructionType(instruction: string) {
        if (PythonInstructionSetInfo.conditionalJumps.test(instruction)) return InstructionType.conditionalJmpInst;
        if (PythonInstructionSetInfo.unconditionalJumps.test(instruction)) return InstructionType.jmp;
        if (PythonInstructionSetInfo.returnInstruction.test(instruction)) return InstructionType.retInst;
        return InstructionType.notRetInst;
    }
}
