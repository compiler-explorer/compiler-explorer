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

import type {AssemblyInstructionInfo} from '../../types/assembly-docs.interfaces.js';
import {BaseAssemblyDocumentationProvider} from './base.js';
import {getAsmOpcode as getCdna1Opcode} from './generated/asm-docs-amd_cdna1.js';
import {getAsmOpcode as getCdna2Opcode} from './generated/asm-docs-amd_cdna2.js';
import {getAsmOpcode as getCdna3Opcode} from './generated/asm-docs-amd_cdna3.js';
import {getAsmOpcode as getCdna4Opcode} from './generated/asm-docs-amd_cdna4.js';
import {getAsmOpcode as getCdna5Opcode} from './generated/asm-docs-amd_cdna5.js';
import {getAsmOpcode as getRdna1Opcode} from './generated/asm-docs-amd_rdna1.js';
import {getAsmOpcode as getRdna2Opcode} from './generated/asm-docs-amd_rdna2.js';
import {getAsmOpcode as getRdna3Opcode} from './generated/asm-docs-amd_rdna3.js';
import {getAsmOpcode as getRdna35Opcode} from './generated/asm-docs-amd_rdna3_5.js';
import {getAsmOpcode as getRdna4Opcode} from './generated/asm-docs-amd_rdna4.js';

export class CDNA1DocumentationProvider extends BaseAssemblyDocumentationProvider {
    public static get key() {
        return 'amd_cdna1';
    }

    public override getInstructionInformation(instruction: string): AssemblyInstructionInfo | null {
        return getCdna1Opcode(instruction.toLowerCase()) || null;
    }
}

export class CDNA2DocumentationProvider extends BaseAssemblyDocumentationProvider {
    public static get key() {
        return 'amd_cdna2';
    }

    public override getInstructionInformation(instruction: string): AssemblyInstructionInfo | null {
        return getCdna2Opcode(instruction.toLowerCase()) || null;
    }
}

export class CDNA3DocumentationProvider extends BaseAssemblyDocumentationProvider {
    public static get key() {
        return 'amd_cdna3';
    }

    public override getInstructionInformation(instruction: string): AssemblyInstructionInfo | null {
        return getCdna3Opcode(instruction.toLowerCase()) || null;
    }
}

export class CDNA4DocumentationProvider extends BaseAssemblyDocumentationProvider {
    public static get key() {
        return 'amd_cdna4';
    }

    public override getInstructionInformation(instruction: string): AssemblyInstructionInfo | null {
        return getCdna4Opcode(instruction.toLowerCase()) || null;
    }
}

export class CDNA5DocumentationProvider extends BaseAssemblyDocumentationProvider {
    public static get key() {
        return 'amd_cdna5';
    }

    public override getInstructionInformation(instruction: string): AssemblyInstructionInfo | null {
        return getCdna5Opcode(instruction.toLowerCase()) || null;
    }
}

export class RDNA1DocumentationProvider extends BaseAssemblyDocumentationProvider {
    public static get key() {
        return 'amd_rdna1';
    }

    public override getInstructionInformation(instruction: string): AssemblyInstructionInfo | null {
        return getRdna1Opcode(instruction.toLowerCase()) || null;
    }
}

export class RDNA2DocumentationProvider extends BaseAssemblyDocumentationProvider {
    public static get key() {
        return 'amd_rdna2';
    }

    public override getInstructionInformation(instruction: string): AssemblyInstructionInfo | null {
        return getRdna2Opcode(instruction.toLowerCase()) || null;
    }
}

export class RDNA3DocumentationProvider extends BaseAssemblyDocumentationProvider {
    public static get key() {
        return 'amd_rdna3';
    }

    public override getInstructionInformation(instruction: string): AssemblyInstructionInfo | null {
        return getRdna3Opcode(instruction.toLowerCase()) || null;
    }
}

export class RDNA35DocumentationProvider extends BaseAssemblyDocumentationProvider {
    public static get key() {
        return 'amd_rdna3_5';
    }

    public override getInstructionInformation(instruction: string): AssemblyInstructionInfo | null {
        return getRdna35Opcode(instruction.toLowerCase()) || null;
    }
}

export class RDNA4DocumentationProvider extends BaseAssemblyDocumentationProvider {
    public static get key() {
        return 'amd_rdna4';
    }

    public override getInstructionInformation(instruction: string): AssemblyInstructionInfo | null {
        return getRdna4Opcode(instruction.toLowerCase()) || null;
    }
}
