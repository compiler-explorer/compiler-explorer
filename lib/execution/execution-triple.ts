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
import {OSType} from '../binaries/binary-utils.js';
import {logger} from '../logger.js';
import * as utils from '../utils.js';

import {BaseExecutionTriple, ExecutionSpecialty} from './base-execution-triple.js';

// import fs from 'fs-extra';

let _host_specialty = ExecutionSpecialty.cpu;

function setHostSpecialty(value: ExecutionSpecialty) {
    _host_specialty = value;
}

export async function initHostSpecialties(): Promise<void> {
    if (os.platform() !== 'win32') {
        const nvidiaGpuExists = await utils.fileExists('/dev/nvidia0');
        if (nvidiaGpuExists) {
            setHostSpecialty(ExecutionSpecialty.nvgpu);
        }

        // const cpuInfo = await fs.readFile('/proc/cpuinfo');
        // if (cpuInfo.includes('GenuineIntel')) {

        // }
    }
}

class CurrentHostExecHelper {
    static isetCanRunOnCurrentHost(value: InstructionSet): boolean {
        // os.arch() Possible values are `'arm'`, `'arm64'`, `'ia32'`, `'loong64'`,`'mips'`, `'mipsel'`, `'ppc'`, `'ppc64'`, `'riscv64'`, `'s390'`, `'s390x'`, and `'x64'`.

        const hostArch = os.arch();
        if (hostArch === 'arm64' && value === 'aarch64') {
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

    static osMatchesCurrentHost(value: string): boolean {
        // Possible values are `'aix'`, `'darwin'`, `'freebsd'`, `'linux'`, `'openbsd'`, `'sunos'`, and `'win32'`
        return value === os.platform();
    }

    static specialtyMatchesCurrentHost(value: ExecutionSpecialty) {
        if (value === _host_specialty) return true;

        if (_host_specialty === ExecutionSpecialty.nvgpu && value === ExecutionSpecialty.cpu) return true;

        return _host_specialty === ExecutionSpecialty.amdgpu && value === ExecutionSpecialty.cpu;
    }

    static getInstructionSetByNodeJSArch(value: string): InstructionSet {
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
                return os.arch() as InstructionSet;
            }
        }
    }
}

// note: returns array of BaseExecutionTriple to prepare for the future of fulfilling multiple execution roles, not actually implemented
export function getExecutionTriplesForCurrentHost(): BaseExecutionTriple[] {
    const triple = new BaseExecutionTriple();
    triple.instructionSet = CurrentHostExecHelper.getInstructionSetByNodeJSArch(os.arch());

    const platform = os.platform() as string;
    if ((Object.values(OSType) as string[]).includes(platform)) {
        triple.os = platform as OSType;
    } else {
        logger.warning(`getExecutionTripleForCurrentHost - Unsupported platform ${platform}`);
    }

    triple.specialty = _host_specialty;

    return [triple];
}

export function matchesCurrentHost(triple: BaseExecutionTriple): boolean {
    const matchesArch = CurrentHostExecHelper.isetCanRunOnCurrentHost(triple.instructionSet);
    const matchesOS = CurrentHostExecHelper.osMatchesCurrentHost(triple.os);
    const matchesSpecialty = CurrentHostExecHelper.specialtyMatchesCurrentHost(triple.specialty);

    return matchesArch && matchesOS && matchesSpecialty;
}
