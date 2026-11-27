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

export class Amd64DocumentationProvider extends BaseAssemblyDocumentationProvider {
    // abs is included as a suffix here for its use in movabs and movabsq
    // https://github.com/compiler-explorer/compiler-explorer/issues/4481
    // Mnemonics can contain digits before the suffix, e.g. cvtsi2sdl
    private static readonly ATT_SUFFIX_REMOVER = /^([a-z][a-z0-9]*?)(?:abs[blqw]?|[blqw])$/i;
    // AT&T names whose Intel counterpart is spelled differently, not just a size suffix away
    private static readonly ATT_ALIASES = new Map([
        ['cbtw', 'cbw'],
        ['cwtl', 'cwde'],
        ['cltq', 'cdqe'],
        ['cwtd', 'cwd'],
        ['cltd', 'cdq'],
        ['cqto', 'cqo'],
        ['movslq', 'movsxd'],
    ]);
    // movz/movs plus source and destination size letters, e.g. movzbl, movswq. Not the string moves (movsb, movsl)
    private static readonly ATT_ZERO_EXTEND = /^movz(?:b[wlq]|w[lq])$/i;
    private static readonly ATT_SIGN_EXTEND = /^movs(?:b[wlq]|w[lq])$/i;
    public static get key() {
        return 'amd64';
    }
    public override getInstructionInformation(instruction: string): AssemblyInstructionInfo | null {
        const attName = Amd64DocumentationProvider.attToIntelName(instruction);
        if (attName) return getAsmOpcode(attName) || null;
        // Try both raw opcode and with AT&T suffix removed
        let info = getAsmOpcode(instruction);
        if (!info) {
            const alternativeInstruction = Amd64DocumentationProvider.ATT_SUFFIX_REMOVER.exec(instruction);
            if (alternativeInstruction) {
                info = getAsmOpcode(alternativeInstruction[1]);
            }
        }
        return info || null;
    }

    private static attToIntelName(instruction: string): string | undefined {
        const lower = instruction.toLowerCase();
        if (Amd64DocumentationProvider.ATT_ZERO_EXTEND.test(lower)) return 'movzx';
        if (Amd64DocumentationProvider.ATT_SIGN_EXTEND.test(lower)) return 'movsx';
        return Amd64DocumentationProvider.ATT_ALIASES.get(lower);
    }
}
