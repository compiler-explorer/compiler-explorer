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

import { BaseCompiler } from '../base-compiler';
import * as utils from '../utils';

export class CleanCompiler extends BaseCompiler {
    static get key() { return 'clean'; }

    optionsForFilter(filters) {
        if (filters.binary) {
            return [];
        } else {
            return ['-S'];
        }
    }

    getOutputFilename(dirPath) {
        return path.join(dirPath, 'Clean System Files/example.s');
    }

    preprocessOutput(output) {
        const errorRegex = /^error \[.*,(\d*),(.*)]:\s?(.*)/i;
        const errorLineRegex = /^error \[.*,(\d*)]:\s?(.*)/i;
        const parseerrorRegex = /^parse error \[.*,(\d*);(\d*),(.*)]:\s?(.*)/i;
        const typeeerrorRegex = /^type error \[.*,(\d*),(.*)]:\s?(.*)/i;
        return utils.splitLines(output).map(line => {
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
                    return '<source>:' + matches[1] + ',' + matches[2] + ': error: (' + matches[3] + ') ' + matches[4];
                }
            }

            return line;
        }).join('\n');
    }

    async runCompiler(compiler, options, inputFilename, execOptions) {
        const tmpDir = path.dirname(inputFilename);
        const moduleName = path.basename(inputFilename, '.icl');
        const compilerPath = path.dirname(compiler);
        execOptions = this.getDefaultExecOptions();
        execOptions.customCwd = tmpDir;
        execOptions.env.CLEANLIB = path.join(compilerPath, '../exe');
        execOptions.env.CLEANPATH = this.compiler.libPath;
        options.pop();
        options.push(moduleName);

        const result = await this.exec(compiler, options, execOptions);
        result.inputFilename = inputFilename;
        result.stdout = utils.parseOutput(this.preprocessOutput(result.stdout), inputFilename);
        result.stderr = utils.parseOutput(this.preprocessOutput(result.stderr), inputFilename);

        if (!options.includes('-S')) {
            const aOut = path.join(tmpDir, 'a.out');
            if (await fs.pathExists(aOut)) {
                await fs.copyFile(aOut, this.getOutputFilename(tmpDir));
                result.code = 0;
            } else {
                result.code = 1;
            }
        } else {
            if (await fs.pathExists(this.getOutputFilename(tmpDir))) {
                result.code = 0;
            }
        }
        return result;
    }
}
