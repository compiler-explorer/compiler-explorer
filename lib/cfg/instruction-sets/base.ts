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

export enum InstructionType {
    jmp,
    conditionalJmpInst,
    notRetInst,
    retInst,
}

export class BaseInstructionSetInfo {
    static get key(): 'base' | InstructionSet | InstructionSet[] {
        return 'base';
    }

    isJmpInstruction(x: string) {
        return x.trim()[0] === 'j' || !!x.match(/\bb\.*(eq|ne|cs|hs|cc|lo|hi|ls|ge|lt|gt|le|rge|rlt)?\b/);
    }

    getInstructionType(inst: string) {
        if (inst.includes('jmp') || inst.includes(' b ')) return InstructionType.jmp;
        else if (this.isJmpInstruction(inst)) return InstructionType.conditionalJmpInst;
        else if (inst.includes(' ret')) {
            return InstructionType.retInst;
        } else {
            return InstructionType.notRetInst;
        }
    }
}
