// Copyright (c) 2025, Compiler Explorer Authors
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
import path from 'node:path';
import fs from 'fs-extra';

import {ExecutionOptions} from '../../types/compilation/compilation.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';

export class NixCompiler extends BaseCompiler {
    static get key() {
        return 'nix';
    }

    override async runCompiler(
        compiler: string,
        options: string[],
        inputFilename: string,
        execOptions: ExecutionOptions & {env: Record<string, string>},
    ) {
        if (!execOptions) {
            execOptions = this.getDefaultExecOptions();
        }

        execOptions.customCwd = path.dirname(inputFilename);
        const dirPath = path.dirname(inputFilename);
        if (!execOptions.customCwd) {
            execOptions.customCwd = dirPath;
        }
        const compilerExecResult = await super.runCompiler(compiler, options, inputFilename, execOptions);
        if (compilerExecResult.stdout.length > 0) {
            const outputFilename = this.getOutputFilename(dirPath, this.outputFilebase);
            const jsonOutput = compilerExecResult.stdout.map(line => line.text).join('\n');
            fs.writeFileSync(outputFilename, JSON.stringify(JSON.parse(jsonOutput), null, 2));
        }
        return compilerExecResult;
    }

    override optionsForFilter(): any[] {
        return ['eval', '--json', '--extra-experimental-features', 'nix-command', '--file'];
    }
}