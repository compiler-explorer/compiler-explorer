import os from 'os';

import {BaseExecutionTriple, ExecutionSpecialty} from '../../types/execution/execution-triple.js';
import {InstructionSet} from '../../types/instructionsets.js';

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
        if (hostArch === 'x64') {
            this.instructionSet = 'amd64';
        } else if (hostArch === 'arm64') {
            this.instructionSet = 'aarch64';
        } else if (hostArch === 'arm') {
            this.instructionSet = 'arm32';
        } else {
            this.instructionSet = os.arch() as InstructionSet;
        }

        this.os = os.platform();
        this.specialty = _host_specialty;
    }

    matchesCurrentHost(): boolean {
        let matchesArch = false;
        let matchesOS = false;
        let matchesSpecialty = false;

        // os.arch() Possible values are `'arm'`, `'arm64'`, `'ia32'`, `'loong64'`,`'mips'`, `'mipsel'`, `'ppc'`, `'ppc64'`, `'riscv64'`, `'s390'`, `'s390x'`, and `'x64'`.

        if (this.instructionSet === 'aarch64' && os.arch() === 'arm64') {
            matchesArch = true;
        } else if (this.instructionSet === 'arm32' && os.arch() === 'arm64') {
            // todo: I'm assuming aarch64 can run arm32 code??
            matchesArch = true;
        } else if (this.instructionSet === 'arm32' && os.arch() === 'arm') {
            matchesArch = true;
        } else if (this.instructionSet === 'amd64' && os.arch() === 'x64') {
            // note: I think x86 32bits code is marked as amd64 as well... (probably shouldn't)
            matchesArch = true;
        }

        if (this.os === os.platform()) {
            // Possible values are `'aix'`, `'darwin'`, `'freebsd'`, `'linux'`, `'openbsd'`, `'sunos'`, and `'win32'`
            matchesOS = true;
        }

        if (this.specialty === _host_specialty) {
            matchesSpecialty = true;
        }

        return matchesArch && matchesOS && matchesSpecialty;
    }
}
