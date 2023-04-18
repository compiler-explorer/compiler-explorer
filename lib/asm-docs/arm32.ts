// Copyright (c) 2021, Compiler Explorer Authors
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

import {AssemblyInstructionInfo, BaseAssemblyDocumentationProvider} from './base.js';
import {getAsmOpcode} from './generated/asm-docs-arm32.js';

export class Arm32DocumentationProvider extends BaseAssemblyDocumentationProvider {
    private static readonly CONDITIONAL_INSTRUCTION_REGEXP =
        /^([A-Za-z]+?)(EQ|NE|CS|CC|MI|PL|VS|VC|HI|LS|GE|LT|GT|LE|AL)$/;
    public static get key() {
        return 'arm32';
    }
    public override getInstructionInformation(instruction: string): AssemblyInstructionInfo | null {
        const info = getAsmOpcode(instruction) || Arm32DocumentationProvider.getConditionalOpcode(instruction);
        return info || null;
    }

    private static readonly CONDITIONAL_OPCODE_TAGS: Record<string, string> = {
        EQ: 'If equal, ',
        NE: 'If not equal, ',
        CS: 'If carry set, ',
        CC: 'If carry clear, ',
        MI: 'If negative, ',
        PL: 'If positive or zero, ',
        VS: 'If overflow, ',
        VC: 'If no overflow, ',
        HI: 'If unsigned higher, ',
        LS: 'If unsigned lower or same, ',
        GE: 'If signed greater than or equal, ',
        LT: 'If signed less than, ',
        GT: 'If signed greater than, ',
        LE: 'If signed less than or equal, ',
    };

    /** Add additional notes for conditional instructions */
    private static getConditionalOpcode(instruction: string): AssemblyInstructionInfo | null {
        // If the instruction is a conditional instruction
        const isConditionalOpcode = instruction.match(Arm32DocumentationProvider.CONDITIONAL_INSTRUCTION_REGEXP);
        if (!isConditionalOpcode) {
            return null;
        }
        const information = getAsmOpcode(isConditionalOpcode[1]);
        if (!information) return null;
        const text = Arm32DocumentationProvider.CONDITIONAL_OPCODE_TAGS[isConditionalOpcode[2]] || '';
        return {
            ...information,
            tooltip: text + information.tooltip,
            html: text + information.html,
        };
    }
}
