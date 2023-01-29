// Copyright (c) 2020, Compiler Explorer Authors
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

import {ExecutionOptions} from '../../types/compilation/compilation.interfaces';
import {CompilerInfo} from '../../types/compiler.interfaces';
import {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces';
import {ResultLine} from '../../types/resultline/resultline.interfaces';
import {BaseCompiler} from '../base-compiler';
import {SassAsmParser} from '../parsers/asm-parser-sass';
import * as utils from '../utils';

import {BaseParser} from './argument-parsers';

export class PtxAssembler extends BaseCompiler {
    static get key() {
        return 'ptxas';
    }

    constructor(info: CompilerInfo, env) {
        super(info, env);
        this.compileFilename = 'example.ptxas';
        this.asm = new SassAsmParser();
    }

    parsePtxOutput(lines: string, inputFilename: string, pathPrefix: string) {
        const re = /^ptxas\s*<source>, line (\d+);(.*)/;
        const result: ResultLine[] = [];
        utils.eachLine(lines, function (line) {
            if (pathPrefix) line = line.replace(pathPrefix, '');
            if (inputFilename) {
                line = line.split(inputFilename).join('<source>');

                if (inputFilename.indexOf('./') === 0) {
                    line = line.split('/home/ubuntu/' + inputFilename.substr(2)).join('<source>');
                    line = line.split('/home/ce/' + inputFilename.substr(2)).join('<source>');
                }
            }
            if (line !== null) {
                const lineObj: ResultLine = {text: line};
                const match = line.replace(/\x1B\[[\d;]*[Km]/g, '').match(re);
                if (match) {
                    lineObj.text = `<source>:${match[1]} ${match[2].trim()}`;
                    lineObj.tag = {
                        severity: 0,
                        line: parseInt(match[1]),
                        column: 0,
                        text: match[2].trim(),
                    };
                }
                result.push(lineObj);
            }
        });
        return result;
    }

    override getSharedLibraryPathsAsArguments() {
        return [];
    }

    override getArgumentParser() {
        return BaseParser;
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions) {
        filters.binary = true;
        return [];
    }

    override async runCompiler(
        compiler: string,
        options: string[],
        inputFilename: string,
        execOptions: ExecutionOptions,
    ) {
        if (!execOptions) {
            execOptions = this.getDefaultExecOptions();
        }
        const dirPath = path.dirname(inputFilename);
        options.push('-o', this.getOutputFilename(dirPath, this.outputFilebase));
        execOptions.customCwd = path.dirname(inputFilename);

        const result = await this.exec(compiler, options, execOptions);
        return {
            ...result,
            inputFilename,
            stdout: this.parsePtxOutput(result.stdout, './' + this.compileFilename, 'no idea what to put here'),
            stderr: this.parsePtxOutput(result.stderr, './' + this.compileFilename, 'no idea what to put here'),
        };
    }

    override getOutputFilename(dirPath: string, outputFilebase: string) {
        return path.join(dirPath, `${outputFilebase}.cubin`);
    }

    override checkOutputFileAndDoPostProcess(asmResult, outputFilename, filters: ParseFiltersAndOutputOptions) {
        return this.postProcess(asmResult, outputFilename, filters);
    }

    override async objdump(outputFilename, result: any, maxSize: number) {
        const dirPath = path.dirname(outputFilename);
        const args = ['-c', '-g', '-hex', outputFilename];
        const objResult = await this.exec(this.compiler.objdumper, args, {maxOutput: maxSize, customCwd: dirPath});
        result.asm = objResult.stdout;
        if (objResult.code === 0) {
            result.objdumpTime = objResult.execTime;
        } else {
            result.asm = '<No output: objdump returned ' + objResult.code + '>';
        }
        return result;
    }
}
