// Copyright (c) 2021, Compiler Explorer Authors
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
import {ArtifactType} from '../../types/tool.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';
import {AsmParserBeebAsm} from '../parsers/asm-parser-beebasm.js';
import * as utils from '../utils.js';

export class BeebAsmCompiler extends BaseCompiler {
    static get key() {
        return 'beebasm';
    }

    constructor(compilerInfo: PreliminaryCompilerInfo, env) {
        super(compilerInfo, env);

        this.asm = new AsmParserBeebAsm(this.compilerProps);
    }

    override optionsForFilter() {
        return ['-v', '-do', 'disk.ssd'];
    }

    override getSharedLibraryPathsAsArguments() {
        return [];
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

        const dirPath = path.dirname(inputFilename);
        if (!execOptions.customCwd) {
            execOptions.customCwd = dirPath;
        }

        options.splice(-1, 0, '-i');

        const hasBootOption = options.some(opt => opt.includes('-boot'));

        const compilerExecResult = await this.exec(compiler, options, execOptions);

        if (compilerExecResult.stdout.length > 0) {
            const outputFilename = this.getOutputFilename(dirPath, this.outputFilebase);
            fs.writeFileSync(outputFilename, compilerExecResult.stdout);
            compilerExecResult.stdout = '';
        }

        const result = this.transformToCompilationResult(compilerExecResult, inputFilename);

        if (result.code === 0 && options.includes('-v')) {
            const diskfile = path.join(dirPath, 'disk.ssd');
            if (await utils.fileExists(diskfile)) {
                await this.addArtifactToResult(result, diskfile, ArtifactType.bbcdiskimage);

                if (!hasBootOption) {
                    if (!result.hints) result.hints = [];
                    result.hints.push(
                        'Try using the "-boot <filename>" option so you don\'t have to manually run your file',
                    );
                }
            }
        }

        const hasNoSaveError = result.stderr.some(opt => opt.text.includes('warning: no SAVE command in source file'));
        if (hasNoSaveError) {
            if (!result.hints) result.hints = [];
            result.hints.push(
                'You should SAVE your code to a file using\nSAVE "filename", start, end [, exec [, reload] ]',
            );
        }

        result.forceBinaryView = true;

        return result;
    }

    override isCfgCompiler(/*compilerVersion: string*/): boolean {
        return true;
    }
}
