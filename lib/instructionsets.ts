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

import os from 'node:os';

import {InstructionSet} from '../types/instructionsets.js';
import {CurrentHostExecHelper} from './execution/execution-triple.js';

type InstructionSetMethod = {
    target: string[];
};

export class InstructionSets {
    private defaultInstructionset: InstructionSet = CurrentHostExecHelper.getInstructionSetByNodeJSArch(os.arch());
    private supported: Record<InstructionSet, InstructionSetMethod>;

    constructor() {
        this.supported = {
            aarch64: {
                target: ['aarch64'],
            },
            arm32: {
                target: ['arm'],
            },
            avr: {
                target: ['avr'],
            },
            c6x: {
                target: ['c6x'],
            },
            dex: {
                target: [],
            },
            ebpf: {
                target: ['bpf'],
            },
            ez80: {
                target: ['ez80'],
            },
            hppa: {
                target: ['hppa'],
            },
            kvx: {
                target: ['kvx'],
            },
            loongarch: {
                target: ['loongarch'],
            },
            m68k: {
                target: ['m68k'],
            },
            mips: {
                target: ['mips'],
            },
            mrisc32: {
                target: ['mrisc32'],
            },
            msp430: {
                target: ['msp430'],
            },
            powerpc: {
                target: ['powerpc', 'ppc64', 'ppc'],
            },
            riscv64: {
                target: ['rv64', 'riscv64'],
            },
            riscv32: {
                target: ['rv32', 'riscv32'],
            },
            sh: {
                target: ['sh'],
            },
            sparc: {
                target: ['sparc', 'sparc64'],
            },
            s390x: {
                target: ['s390x'],
            },
            vax: {
                target: ['vax'],
            },
            wasm32: {
                target: ['wasm32'],
            },
            wasm64: {
                target: ['wasm64'],
            },
            xtensa: {
                target: ['xtensa'],
            },
            z180: {
                target: ['z180'],
            },
            z80: {
                target: ['z80'],
            },
            6502: {
                target: [],
            },
            wdc65c816: {
                target: [],
            },
            core: {
                target: [],
            },
            java: {
                target: [],
            },
            llvm: {
                target: [],
            },
            perl: {
                target: [],
            },
            python: {
                target: [],
            },
            mpy: {
                target: [],
            },
            ptx: {
                target: [],
            },
            x86: {
                target: [],
            },
            amd64: {
                target: ['x86_64'],
            },
            evm: {
                target: [],
            },
            eravm: {
                target: [],
            },
            mos6502: {
                target: [],
            },
            sass: {
                target: [],
            },
            beam: {
                target: [],
            },
            hook: {
                target: [],
            },
            spirv: {
                target: [],
            },
        };
    }

    // Return the first spelling of the target for the instruction set,
    // or null if data is missing from the 'supported' table.
    getInstructionSetTarget(instructionSet: InstructionSet): string | null {
        if (!(instructionSet in this.supported)) return null;
        if (this.supported[instructionSet].target.length === 0) return null;
        return this.supported[instructionSet].target[0];
    }

    getCompilerInstructionSetHint(compilerArch: string | boolean): InstructionSet {
        if (compilerArch && typeof compilerArch === 'string') {
            for (const [instructionSet, method] of Object.entries(this.supported) as [
                InstructionSet,
                InstructionSetMethod,
            ][]) {
                for (const target of method.target) {
                    if (compilerArch.includes(target)) {
                        return instructionSet;
                    }
                }
            }
        }

        return this.defaultInstructionset;
    }
}
