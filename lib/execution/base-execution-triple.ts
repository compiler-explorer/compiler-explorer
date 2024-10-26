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

import {InstructionSet} from '../../types/instructionsets.js';
import {OSType} from '../binaries/binary-utils.js';

export enum ExecutionSpecialty {
    cpu = 'cpu',
    nvgpu = 'nvgpu',
    amdgpu = 'amdgpu',
}

export class BaseExecutionTriple {
    protected _instructionSet: InstructionSet = 'amd64';
    protected _os: OSType = OSType.linux;
    protected _specialty: ExecutionSpecialty = ExecutionSpecialty.cpu;

    get instructionSet(): InstructionSet {
        return this._instructionSet;
    }

    set instructionSet(value: InstructionSet) {
        this._instructionSet = value;
    }

    set os(value: OSType) {
        this._os = value;
    }

    get os(): OSType {
        return this._os;
    }

    set specialty(value: ExecutionSpecialty) {
        this._specialty = value;
    }

    get specialty(): ExecutionSpecialty {
        return this._specialty;
    }

    toString(): string {
        return `${this._instructionSet}-${this._os}-${this._specialty}`;
    }

    parse(triple: string) {
        if (triple.includes('-')) {
            const reTriple = /(\w*)-(\w*)-(\w*)/;
            const match = triple.match(reTriple);
            if (match) {
                this._instructionSet = match[1] as InstructionSet;
                this._os = match[2] as OSType;
                this._specialty = match[3] as ExecutionSpecialty;
            }
        } else {
            this._instructionSet = triple as InstructionSet;
        }
    }
}
