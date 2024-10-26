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

import {BuildResult} from '../../types/compilation/compilation.interfaces.js';
import {BinaryInfoLinux} from '../binaries/binary-utils.js';
import {logger} from '../logger.js';
import {PropertyGetter} from '../properties.interfaces.js';

import {BaseExecutionTriple, ExecutionSpecialty} from './base-execution-triple.js';

async function retrieveAllRemoteExecutionArchs(ceProps: PropertyGetter, envs: string[]): Promise<string[]> {
    let url: string = ceProps<string>('execqueue.remote_archs_url', '');
    try {
        if (!url) return [];

        if (envs.find(env => env.includes('staging') || env === 'dev')) url += '?env=staging';

        // eslint-disable-next-line n/no-unsupported-features/node-builtins
        const response = await fetch(url);

        if (response.status !== 200) {
            throw new Error(`Status ${response.status} on GET ${url}`);
        }

        return await response.json();
    } catch (e) {
        logger.error('Error retrieving remote execution architectures');
        logger.error(e);
        process.exit(1);
    }
}

let _available_remote_execution_archs: Promise<BaseExecutionTriple[]>;

export class RemoteExecutionQuery {
    static initRemoteExecutionArchs(ceProps: PropertyGetter, envs: string[]) {
        _available_remote_execution_archs = new Promise(resolve => {
            retrieveAllRemoteExecutionArchs(ceProps, envs).then(archs => {
                const triples: BaseExecutionTriple[] = [];
                for (const arch of archs) {
                    const triple = new BaseExecutionTriple();
                    triple.parse(arch);
                    triples.push(triple);
                }
                resolve(triples);
            });
        });
    }

    static async isPossible(triple: BaseExecutionTriple): Promise<boolean> {
        const triples = await _available_remote_execution_archs;
        return !!triples.find(remote => remote.toString() === triple.toString());
    }

    static async guessExecutionTripleForBuildresult(result: BuildResult): Promise<BaseExecutionTriple> {
        const triple = new BaseExecutionTriple();

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
