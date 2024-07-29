import {InstructionSet} from '../instructionsets.js';

export enum ExecutionSpecialty {
    cpu = 'cpu',
    nvgpu = 'nvgpu',
    amdgpu = 'amdgpu',
}

export class BaseExecutionTriple {
    protected instructionSet: InstructionSet = 'amd64';
    protected os: string = 'linux';
    protected specialty: ExecutionSpecialty = ExecutionSpecialty.cpu;

    getInstructionSet(): InstructionSet {
        return this.instructionSet;
    }

    setInstructionSet(value: InstructionSet) {
        this.instructionSet = value;
    }

    setOS(value: string) {
        this.os = value;
    }

    setSpecialty(value: ExecutionSpecialty) {
        this.specialty = value;
    }

    toString(): string {
        return `${this.instructionSet}-${this.os}-${this.specialty}`;
    }

    parse(triple: string) {
        if (triple.includes('-')) {
            const reTriple = /(\w*)-(\w*)-(\w*)/;
            const match = triple.match(reTriple);
            if (match) {
                this.instructionSet = match[0] as InstructionSet;
                this.os = match[1];
                this.specialty = match[2] as ExecutionSpecialty;
            }
        } else {
            this.instructionSet = triple as InstructionSet;
        }
    }
}
