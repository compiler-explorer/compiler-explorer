import {InstructionSet} from '../../types/instructionsets.js';
import {executeDirect} from '../exec.js';

import {os_linux, os_windows} from './base-execution-triple.js';

export type BinaryInfo = {
    instructionSet: InstructionSet;
    os: string;
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
            default: {
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
                os: os_linux,
                instructionSet: this.getInstructionSetForArchText(csv[1]),
            };
        } else if (isPE) {
            const filteredLine = this.removeComments(csv[0]);
            const lastWordPos = filteredLine.lastIndexOf(' ');
            const lastWord = filteredLine.substring(lastWordPos + 1);

            return {
                os: os_windows,
                instructionSet: this.getInstructionSetForArchText(lastWord),
            };
        }

        return undefined;
    }

    static async readFile(filepath: string): Promise<BinaryInfo | undefined> {
        const info = await executeDirect('/usr/bin/file', ['-b', filepath], {});
        if (info.code === 0) return this.parseFileInfo(info.stdout);
        return undefined;
    }
}
