import {BuildResult} from '../../types/compilation/compilation.interfaces.js';
import {BinaryInfoLinux} from '../binaries/binary-utils.js';

import {BaseExecutionTriple, ExecutionSpecialty} from './base-execution-triple.js';
import {ExecutionTriple} from './execution-triple.js';

async function retrieveAllRemoteExecutionArchs(): Promise<string[]> {
    // eslint-disable-next-line n/no-unsupported-features/node-builtins
    const response = await fetch('https://api.compiler-explorer.com/get_remote_execution_archs');

    return await response.json();
}

const _available_remote_execution_archs: Promise<BaseExecutionTriple[]> = new Promise(resolve => {
    retrieveAllRemoteExecutionArchs().then(archs => {
        const triples: BaseExecutionTriple[] = [];
        for (const arch of archs) {
            const triple = new BaseExecutionTriple();
            triple.parse(arch);
            triples.push(triple);
        }
        resolve(triples);
    });
});

export class RemoteExecutionQuery {
    static async isPossible(triple: ExecutionTriple): Promise<boolean> {
        const triples = await _available_remote_execution_archs;
        return !!triples.find(remote => remote.toString() === triple.toString());
    }

    static async guessExecutionTripleForBuildresult(result: BuildResult): Promise<ExecutionTriple> {
        const triple = new ExecutionTriple();

        if (result.executableFilename) {
            const info = await BinaryInfoLinux.readFile(result.executableFilename);
            if (info) {
                triple.instructionSet = info.instructionSet;
                triple.os = info.os;
            }
        } else {
            if (result.instructionSet) triple.instructionSet = result.instructionSet;
        }

        if (result.devices && Object.keys(result.devices).length > 0) {
            triple.specialty = ExecutionSpecialty.nvgpu;
        }

        return triple;
    }
}
