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

import type {AssemblyInstructionInfo} from '../../types/assembly-docs.interfaces.js';
import {BaseAssemblyDocumentationProvider} from './base.js';
import {getAsmOpcode} from './generated/asm-docs-amd64.js';

// Pseudo-op mnemonics that assemblers accept for an instruction plus a fixed immediate. They have no page of their
// own: Intel documents them in a table on the page of the instruction that takes the immediate.
// https://github.com/compiler-explorer/compiler-explorer/issues/1026
const COMPARE_PSEUDO_OPS: [RegExp, (match: RegExpExecArray) => string][] = [
    // cmpltps, cmpneqsd, vcmpunordss, ... (predicates 0-7)
    [/^(v?)cmp(?:eq|lt|le|unord|neq|nlt|nle|ord)([ps][sd])$/, m => `${m[1]}cmp${m[2]}`],
    // vcmpeq_uqpd, vcmplt_ossd, ... (the 32 AVX predicates)
    [/^vcmp(?:eq|lt|le|unord|neq|nlt|nle|ord|nge|ngt|false|ge|gt|true)_(?:oq|os|uq|us)([ps][sd])$/, m => `vcmp${m[1]}`],
    // vpcmpltb, vpcmpnled, ... (signed AVX-512 integer compares; vpcmpeqb and vpcmpgtb are instructions of their own)
    [/^vpcmp(?:lt|le|neq|nlt|nle)([bwdq])$/, m => `vpcmp${m[1]}`],
    // vpcmpequb, vpcmpltuw, ... (unsigned AVX-512 integer compares)
    [/^vpcmp(?:eq|lt|le|neq|nlt|nle)u([bwdq])$/, m => `vpcmpu${m[1]}`],
    // pclmullqlqdq, vpclmulhqhqdq, ...
    [/^(v?)pclmul[lh]q[lh]qdq$/, m => `${m[1]}pclmulqdq`],
];

function getComparePseudoOpInfo(instruction: string): AssemblyInstructionInfo | undefined {
    const lower = instruction.toLowerCase();
    for (const [pattern, parentOf] of COMPARE_PSEUDO_OPS) {
        const match = pattern.exec(lower);
        if (!match) continue;
        const parent = parentOf(match);
        // Not every VEX form is documented separately from its legacy SSE one (e.g. vcmpsd)
        return getAsmOpcode(parent) ?? getAsmOpcode(parent.replace(/^v/, ''));
    }
    return undefined;
}

export class Amd64DocumentationProvider extends BaseAssemblyDocumentationProvider {
    // abs is included as a suffix here for its use in movabs and movabsq
    // https://github.com/compiler-explorer/compiler-explorer/issues/4481
    // Mnemonics can contain digits before the suffix, e.g. cvtsi2sdl
    private static readonly ATT_SUFFIX_REMOVER = /^([a-z][a-z0-9]*?)(?:abs[blqw]?|[blqw])$/i;
    public static get key() {
        return 'amd64';
    }
    public override getInstructionInformation(instruction: string): AssemblyInstructionInfo | null {
        // Try both raw opcode and with AT&T suffix removed
        let info = getAsmOpcode(instruction);
        if (!info) {
            const alternativeInstruction = Amd64DocumentationProvider.ATT_SUFFIX_REMOVER.exec(instruction);
            if (alternativeInstruction) {
                info = getAsmOpcode(alternativeInstruction[1]);
            }
        }
        return info || getComparePseudoOpInfo(instruction) || null;
    }
}
