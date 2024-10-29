// Copyright (c) 2024, Compiler Explorer Authors
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

import os from 'os';

import {InstructionSet} from '../../types/instructionsets.js';
import {executeDirect} from '../exec.js';
import {logger} from '../logger.js';

export enum OSType {
    linux = 'linux',
    windows = 'win32',
}

export type BinaryInfo = {
    instructionSet: InstructionSet;
    os: OSType;
};

export class BinaryInfoLinux {
    static getInstructionSetForArchText(value: string): InstructionSet {
        switch (value) {
            case 'x86-64': {
                return 'amd64';
            }
            case 'intel 80386': {
                return 'x86';
            }
            case '80386': {
                return 'x86';
            }
            case 'arm aarch64': {
                return 'aarch64';
            }
            case 'aarch64': {
                return 'aarch64';
            }
            case 'arm': {
                return 'arm32';
            }
            case 'atmel avr 8-bit': {
                return 'avr';
            }
            case 'ucb risc-v': {
                return 'riscv64';
            }
            case '64-bit powerpc or cisco 7500': {
                return 'powerpc';
            }
            case 'powerpc or cisco 4500': {
                return 'powerpc';
            }
            case 'mips': {
                return 'mips';
            }
            case 'loongarch': {
                return 'loongarch';
            }
            default: {
                logger.error(`Unknown architecture text: ${value}`);
                return 'amd64';
            }
        }
    }

    static removeComments(value: string): string {
        let filtered: string = '';
        let inComment: boolean = false;
        for (const c of value) {
            if (!inComment && c === '(') {
                inComment = true;
            } else if (inComment && c === ')') {
                inComment = false;
            } else if (!inComment) {
                filtered += c;
            }
        }
        return filtered.trim();
    }

    static parseFileInfo(output: string): BinaryInfo | undefined {
        const csv: string[] = output.split(', ').map(val => val.trim().toLowerCase());
        const isElf = csv[0].startsWith('elf');
        const isPE = csv[0].startsWith('pe32');
        if (isElf) {
            return {
                os: OSType.linux,
                instructionSet: this.getInstructionSetForArchText(csv[1]),
            };
        } else if (isPE) {
            const filteredLine = this.removeComments(csv[0]);
            const lastWordPos = filteredLine.lastIndexOf(' ');
            const lastWord = filteredLine.substring(lastWordPos + 1);

            return {
                os: OSType.windows,
                instructionSet: this.getInstructionSetForArchText(lastWord),
            };
        }

        return undefined;
    }

    static async readFile(filepath: string): Promise<BinaryInfo | undefined> {
        if (os.platform() === 'win32') {
            return {
                os: OSType.windows,
                instructionSet: 'amd64',
            };
        } else {
            const info = await executeDirect('/usr/bin/file', ['-b', filepath], {});
            if (info.code === 0) return this.parseFileInfo(info.stdout);
        }
        return undefined;
    }
}
