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

    private getInstructionSetByNodeJSArch(value: string): InstructionSet {
        switch (value) {
            case 'x64': {
                return 'amd64';
            }
            case 'ia32': {
                return 'x86';
            }
            case 'arm64': {
                return 'aarch64';
            }
            case 'arm': {
                return 'arm32';
            }
            default: {
                return (this._instructionSet = os.arch() as InstructionSet);
            }
        }
    }

    private initHostTriple() {
        this._instructionSet = this.getInstructionSetByNodeJSArch(os.arch());
        this._os = os.platform();
        this._specialty = _host_specialty;
    }

    protected isetCanRunOnCurrentHost(value: InstructionSet): boolean {
        // os.arch() Possible values are `'arm'`, `'arm64'`, `'ia32'`, `'loong64'`,`'mips'`, `'mipsel'`, `'ppc'`, `'ppc64'`, `'riscv64'`, `'s390'`, `'s390x'`, and `'x64'`.

        const hostArch = os.arch();
        if (hostArch === 'arm64' && value in ['aarch64', 'arm32']) {
            return true;
        } else if (hostArch === 'arm' && value === 'arm32') {
            return true;
        } else if (hostArch === 'x64' && (value === 'amd64' || value === 'x86')) {
            return true;
        } else if (hostArch === 'ia32' && value === 'x86') {
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
        const matchesArch = this.isetCanRunOnCurrentHost(this._instructionSet);
        const matchesOS = this.osMatchesCurrentHost(this._os);
        const matchesSpecialty = this.specialtyMatchesCurrentHost(this._specialty);

        return matchesArch && matchesOS && matchesSpecialty;
    }
}
