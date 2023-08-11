// Copyright (c) 2018, Compiler Explorer Authors
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

import path from 'path';

import fs from 'fs-extra';

import type {ExecutionOptions} from '../../types/compilation/compilation.interfaces.js';
import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';
import {propsFor} from '../properties.js';
import * as utils from '../utils.js';

export class CleanCompiler extends BaseCompiler {
    static get key() {
        return 'clean';
    }

    executionType: string;

    constructor(compiler: PreliminaryCompilerInfo, env) {
        super(compiler, env);
        const execProps = propsFor('execution');
        this.executionType = execProps('executionType', 'none');
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions) {
        if (filters.binary) {
            return [];
        } else {
            return ['-S'];
        }
    }

    override getOutputFilename(dirPath: string) {
        return path.join(dirPath, 'Clean System Files/example.s');
    }

    override getExecutableFilename(dirPath: string) {
        return path.join(dirPath, 'a.out');
    }

    override getSharedLibraryPathsAsArguments() {
        return [];
    }

    preprocessOutput(output: string) {
        const errorRegex = /^error \[.*,(\d*),(.*)]:\s?(.*)/i;
        const errorLineRegex = /^error \[.*,(\d*)]:\s?(.*)/i;
        const parseerrorRegex = /^parse error \[.*,(\d*);(\d*),(.*)]:\s?(.*)/i;
        const typeeerrorRegex = /^type error \[.*,(\d*),(.*)]:\s?(.*)/i;
        return utils
            .splitLines(output)
            .map(line => {
                let matches = line.match(errorRegex);
                if (!matches) matches = line.match(typeeerrorRegex);

                if (matches) {
                    return '<source>:' + matches[1] + ',0: error: (' + matches[2] + ') ' + matches[3];
                }

                matches = line.match(errorLineRegex);
                if (matches) {
                    return '<source>:' + matches[1] + ',0: error: ' + matches[2];
                }

                matches = line.match(parseerrorRegex);
                if (matches) {
                    if (matches[3] === '') {
                        return '<source>:' + matches[1] + ',' + matches[2] + ': error: ' + matches[4];
                    } else {
                        return (
                            '<source>:' + matches[1] + ',' + matches[2] + ': error: (' + matches[3] + ') ' + matches[4]
                        );
                    }
                }

                return line;
            })
            .join('\n');
    }

    override async runCompiler(
        compiler: string,
        options: string[],
        inputFilename: string,
        execOptions: ExecutionOptions & {env: Record<string, string>},
    ) {
        const tmpDir = path.dirname(inputFilename);
        const moduleName = path.basename(inputFilename, '.icl');
        const compilerPath = path.dirname(compiler);

        execOptions = this.getDefaultExecOptions();
        execOptions.customCwd = tmpDir;
        if (await utils.dirExists(path.join(compilerPath, '../exe'))) {
            execOptions.env.CLEANLIB = path.join(compilerPath, '../exe');
        } else if (await utils.dirExists(path.join(compilerPath, '../lib/exe'))) {
            execOptions.env.CLEANLIB = path.join(compilerPath, '../lib/exe');
        }
        execOptions.env.CLEANPATH = this.compiler.libPath.join(':');
        execOptions.env.CLEANABCPATH = tmpDir + '/Clean System Files';
        execOptions.env.CLEANOPATH = tmpDir + '/obj';
        options.pop();
        options.push(moduleName);

        await fs.mkdir(execOptions.env.CLEANABCPATH);
        await fs.mkdir(execOptions.env.CLEANOPATH);

        if (this.executionType === 'nsjail') {
            execOptions.env.CLEANABCPATH = '/app/Clean System Files';
            execOptions.env.CLEANOPATH = '/app/obj';
        }

        const rawResult = await this.exec(compiler, options, execOptions);
        const result = {
            ...rawResult,
            inputFilename,
            stdout: utils.parseOutput(this.preprocessOutput(rawResult.stdout), inputFilename),
            stderr: utils.parseOutput(this.preprocessOutput(rawResult.stderr), inputFilename),
        };

        if (options.includes('-S')) {
            if (await fs.pathExists(this.getOutputFilename(tmpDir))) {
                result.code = 0;
            }
        } else {
            const aOut = path.join(tmpDir, 'a.out');
            if (await fs.pathExists(aOut)) {
                await fs.copyFile(aOut, this.getOutputFilename(tmpDir));
                result.code = 0;
            } else {
                result.code = 1;
            }
        }
        return result;
    }
}
