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

import {InstructionSet} from '../types/instructionsets.js';

type InstructionSetMethod = {
    target: string[];
    path: string[];
};

export class InstructionSets {
    private defaultInstructionset: InstructionSet = 'amd64';
    private supported: Record<InstructionSet, InstructionSetMethod>;

    constructor() {
        this.supported = {
            aarch64: {
                target: ['aarch64'],
                path: ['/aarch64-'],
            },
            arm32: {
                target: ['arm'],
                path: ['/arm-'],
            },
            avr: {
                target: ['avr'],
                path: ['/avr-'],
            },
            c6x: {
                target: ['c6x'],
                path: ['/c6x-'],
            },
            ebpf: {
                target: ['ebpf'],
                path: ['/bpf-'],
            },
            kvx: {
                target: ['kvx'],
                path: ['/kvx-', '/k1-'],
            },
            loongarch: {
                target: ['loongarch'],
                path: ['/loongarch64-'],
            },
            mips: {
                target: ['mips'],
                path: ['/mips', '/mipsel-', '/mips64el-', '/mips64-'],
            },
            mrisc32: {
                target: ['mrisc32'],
                path: [],
            },
            msp430: {
                target: ['msp430'],
                path: ['/msp430-'],
            },
            powerpc: {
                target: ['powerpc'],
                path: ['/powerpc-', '/powerpc64-', '/powerpc64le-'],
            },
            riscv64: {
                target: ['rv64'],
                path: ['/riscv64-'],
            },
            riscv32: {
                target: ['rv32'],
                path: ['/riscv32-'],
            },
            sh: {
                target: ['sh'],
                path: ['/sh-'],
            },
            sparc: {
                target: ['sparc', 'sparc64'],
                path: ['/sparc-', '/sparc64-'],
            },
            s390x: {
                target: ['s390x'],
                path: ['/s390x-'],
            },
            vax: {
                target: ['vax'],
                path: ['/vax-'],
            },
            wasm32: {
                target: ['wasm32'],
                path: [],
            },
            wasm64: {
                target: ['wasm64'],
                path: [],
            },
            xtensa: {
                target: ['xtensa'],
                path: ['/xtensa-'],
            },
            z80: {
                target: ['z80'],
                path: [],
            },
            6502: {
                target: [],
                path: [],
            },
            java: {
                target: [],
                path: [],
            },
            llvm: {
                target: [],
                path: [],
            },
            python: {
                target: [],
                path: [],
            },
            ptx: {
                target: [],
                path: [],
            },
            amd64: {
                target: [],
                path: [],
            },
            evm: {
                target: [],
                path: [],
            },
            mos6502: {
                target: [],
                path: [],
            },
            sass: {
                target: [],
                path: [],
            },
            beam: {
                target: [],
                path: [],
            },
            hook: {
                target: [],
                path: [],
            },
            spirv: {
                target: [],
                path: [],
            },
        };
    }

    async getCompilerInstructionSetHint(compilerArch: string | boolean, exe: string): Promise<InstructionSet> {
        return new Promise(resolve => {
            if (compilerArch && typeof compilerArch === 'string') {
                for (const [instructionSet, method] of Object.entries(this.supported) as [
                    InstructionSet,
                    InstructionSetMethod,
                ][]) {
                    for (const target of method.target) {
                        if (compilerArch.includes(target)) {
                            resolve(instructionSet);
                            return;
                        }
                    }
                }
            } else {
                for (const [instructionSet, method] of Object.entries(this.supported) as [
                    InstructionSet,
                    InstructionSetMethod,
                ][]) {
                    for (const path of method.path) {
                        if (exe.includes(path)) {
                            resolve(instructionSet);
                            return;
                        }
                    }
                }
            }

            resolve(this.defaultInstructionset);
        });
    }
}
