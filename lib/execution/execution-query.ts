import {BuildResult} from '../../types/compilation/compilation.interfaces.js';

import {ExecutionSpecialty, ExecutionTriple} from './execution-triple.js';

export class RemoteExecutionQuery {
    static async isPossible(triple: ExecutionTriple): Promise<boolean> {
        // todo: should request this from a database or something
        return triple.getInstructionSet() === 'aarch64';
    }

    static guessExecutionTripleForBuildresult(result: BuildResult): ExecutionTriple {
        const triple = new ExecutionTriple();
        if (result.instructionSet) {
            triple.parse(result.instructionSet);
            if (result.executableFilename && result.executableFilename.endsWith('.exe')) {
                triple.setOS('win32');
            }
            if (result.devices && Object.keys(result.devices).length > 0) {
                triple.setSpecialty(ExecutionSpecialty.nvgpu);
            }
        }

        return triple;
    }
}
