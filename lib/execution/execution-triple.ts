import os from 'os';

import {InstructionSet} from '../../types/instructionsets.js';

import {BaseExecutionTriple, ExecutionSpecialty} from './base-execution-triple.js';

let _host_specialty = ExecutionSpecialty.cpu;

export function setHostSpecialty(value: ExecutionSpecialty) {
    _host_specialty = value;
}

export class ExecutionTriple extends BaseExecutionTriple {
    constructor() {
        super();
        this.initHostTriple();
    }

    private initHostTriple() {
        const hostArch = os.arch();
        switch (hostArch) {
            case 'x64': {
                this._instructionSet = 'amd64';
                break;
            }
            case 'arm64': {
                this._instructionSet = 'aarch64';
                break;
            }
            case 'arm': {
                this._instructionSet = 'arm32';
                break;
            }
            default: {
                this._instructionSet = os.arch() as InstructionSet;
                break;
            }
        }

        this._os = os.platform();
        this._specialty = _host_specialty;
    }

    protected archMatchesCurrentHost(value: InstructionSet): boolean {
        // os.arch() Possible values are `'arm'`, `'arm64'`, `'ia32'`, `'loong64'`,`'mips'`, `'mipsel'`, `'ppc'`, `'ppc64'`, `'riscv64'`, `'s390'`, `'s390x'`, and `'x64'`.

        const hostArch = os.arch();
        if (hostArch === 'arm64' && value in ['aarch64', 'arm32']) {
            return true;
        } else if (hostArch === 'arm' && value === 'arm32') {
            return true;
        } else if (hostArch === 'x64' && value === 'amd64') {
            // note: I think x86 32bits code is marked as amd64 as well... (probably shouldn't)
            return true;
        }

        return false;
    }

    protected osMatchesCurrentHost(value: string): boolean {
        // Possible values are `'aix'`, `'darwin'`, `'freebsd'`, `'linux'`, `'openbsd'`, `'sunos'`, and `'win32'`
        return value === os.platform();
    }

    protected specialtyMatchesCurrentHost(value: ExecutionSpecialty) {
        return value === _host_specialty;
    }

    matchesCurrentHost(): boolean {
        const matchesArch = this.archMatchesCurrentHost(this._instructionSet);
        const matchesOS = this.osMatchesCurrentHost(this._os);
        const matchesSpecialty = this.specialtyMatchesCurrentHost(this._specialty);

        return matchesArch && matchesOS && matchesSpecialty;
    }
}
