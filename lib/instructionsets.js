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

import _ from 'underscore';

export class InstructionSets {
    constructor() {
        this.default = 'amd64';

        this.supported = {
            aarch64: {
                target: ['aarch64'],
                path: ['/aarch64-'],
            },
            arm32: {
                target: ['arm'],
                path: ['/arm-'],
            },
            riscv64: {
                target: ['rv64'],
                path: ['/riscv64-'],
            },
            riscv32: {
                target: ['rv32'],
                path: ['/riscv32-'],
            },
            avr: {
                target: ['avr'],
                path: ['/avr-'],
            },
            msp430: {
                target: ['msp430'],
                path: ['/msp430-'],
            },
            mips: {
                target: ['mips'],
                path: ['/mips-'],
            },
            wasm32: {
                target: ['wasm32'],
                path: [],
            },
            wasm64: {
                target: ['wasm64'],
                path: [],
            },
            6502: {
                target: [],
                path: [],
            },
            powerpc64le: {
                target: ['powerpc64le'],
                path: ['/powerpc64le-'],
            },
            powerpc64: {
                target: ['powerpc64'],
                path: ['/powerpc64-'],
            },
        };
    }

    async getCompilerInstructionSetHint(compilerArch, exe) {
        return new Promise(resolve => {
            if (compilerArch) {
                _.each(this.supported, (method, instructionSet) => {
                    for (let target of method.target) {
                        if (compilerArch.includes(target)) {
                            resolve(instructionSet);
                        }
                    }
                });
            } else {
                _.each(this.supported, (method, instructionSet) => {
                    for (let path of method.path) {
                        if (exe.includes(path)) {
                            resolve(instructionSet);
                        }
                    }
                });
            }

            resolve(this.default);
        });
    }
}
