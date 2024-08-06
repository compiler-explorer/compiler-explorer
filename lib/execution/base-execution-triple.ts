import {InstructionSet} from '../../types/instructionsets.js';

export enum ExecutionSpecialty {
    cpu = 'cpu',
    nvgpu = 'nvgpu',
    amdgpu = 'amdgpu',
}

export const os_linux: string = 'linux';
export const os_windows: string = 'win32';

export class BaseExecutionTriple {
    protected _instructionSet: InstructionSet = 'amd64';
    protected _os: string = os_linux;
    protected _specialty: ExecutionSpecialty = ExecutionSpecialty.cpu;

    get instructionSet(): InstructionSet {
        return this._instructionSet;
    }

    set instructionSet(value: InstructionSet) {
        this._instructionSet = value;
    }

    set os(value: string) {
        this._os = value;
    }

    get os(): string {
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
                this._os = match[2];
                this._specialty = match[3] as ExecutionSpecialty;
            }
        } else {
            this._instructionSet = triple as InstructionSet;
        }
    }
}
