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

import {BaseCompiler} from '../base-compiler';
import {SassAsmParser} from '../parsers/asm-parser-sass';
import * as utils from '../utils';

import {BaseParser} from './argument-parsers';

export class PtxAssembler extends BaseCompiler {
    static get key() {
        return 'ptxas';
    }

    constructor(info, env) {
        super(info, env);
        this.compileFilename = 'example.ptxas';
        this.asm = new SassAsmParser();
    }

    parsePtxOutput(lines, inputFilename, pathPrefix) {
        const re = /^ptxas\s*<source>, line (\d+);(.*)/;
        const result = [];
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
                const lineObj = {text: line};
                const match = line.replace(/\x1B\[[\d;]*[Km]/g, '').match(re);
                if (match) {
                    lineObj.text = `<source>:${match[1]} ${match[2].trim()}`;
                    lineObj.tag = {
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

    getSharedLibraryPathsAsArguments() {
        return [];
    }

    getArgumentParser() {
        return BaseParser;
    }

    optionsForFilter(filters) {
        filters.binary = true;
        return [];
    }

    async runCompiler(compiler, options, inputFilename, execOptions) {
        if (!execOptions) {
            execOptions = this.getDefaultExecOptions();
        }
        const dirPath = path.dirname(inputFilename);
        options.push('-o', this.getOutputFilename(dirPath, this.outputFilebase));
        execOptions.customCwd = path.dirname(inputFilename);

        const result = await this.exec(compiler, options, execOptions);
        result.inputFilename = inputFilename;
        result.stdout = this.parsePtxOutput(result.stdout, './' + this.compileFilename);
        result.stderr = this.parsePtxOutput(result.stderr, './' + this.compileFilename);
        return result;
    }

    getOutputFilename(dirPath, outputFilebase) {
        return path.join(dirPath, `${outputFilebase}.cubin`);
    }

    checkOutputFileAndDoPostProcess(asmResult, outputFilename, filters) {
        return this.postProcess(asmResult, outputFilename, filters);
    }

    async objdump(outputFilename, result, maxSize) {
        const dirPath = path.dirname(outputFilename);
        let args = ['-c', '-g', '-hex', outputFilename];
        const objResult = await this.exec(this.compiler.objdumper, args, {maxOutput: maxSize, customCwd: dirPath});
        result.asm = objResult.stdout;
        if (objResult.code !== 0) {
            result.asm = '<No output: objdump returned ' + objResult.code + '>';
        } else {
            result.objdumpTime = objResult.execTime;
        }
        return result;
    }
}
