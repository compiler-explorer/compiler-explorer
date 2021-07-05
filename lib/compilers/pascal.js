// Copyright (c) 2017, Compiler Explorer Authors
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
import _ from 'underscore';

import { BaseCompiler } from '../base-compiler';
import * as utils from '../utils';

import { PascalParser } from './argument-parsers';

export class FPCCompiler extends BaseCompiler {
    static get key() {
        return 'pascal';
    }

    constructor(info, env) {
        super(info, env);

        this.compileFilename = 'output.pas';
        this.supportsOptOutput = false;
        this.nasmPath = this.compilerProps('nasmpath');
    }

    getSharedLibraryPathsAsArguments() {
        return [];
    }

    processAsm(result, filters) {
        this.demangler = new this.demanglerClass(null, this);
        return this.asm.process(result.asm, filters);
    }

    postProcessAsm(result, filters) {
        if (!result.okToCache) return result;

        if (filters.binary) {
            for (let j = 0; j < result.asm.length; ++j) {
                this.demangler.addDemangleToCache(result.asm[j].text);
            }
        }

        for (let j = 0; j < result.asm.length; ++j)
            result.asm[j].text = this.demangler.demangleIfNeeded(result.asm[j].text);

        return result;
    }

    optionsForFilter(filters) {
        let options = ['-g', '-al'];

        if (this.compiler.intelAsm && filters.intel && !filters.binary) {
            options = options.concat(this.compiler.intelAsm.split(' '));
        }

        filters.preProcessLines = _.bind(this.preProcessLines, this);

        return options;
    }

    getOutputFilename(dirPath) {
        return path.join(dirPath, `${path.basename(this.compileFilename, this.lang.extensions[0])}.s`);
    }

    getExecutableFilename(dirPath) {
        return path.join(dirPath, 'prog');
    }

    static preProcessBinaryAsm(input) {
        const relevantAsmStartsAt = input.indexOf('<OUTPUT');
        if (relevantAsmStartsAt !== -1) {
            const lastLinefeedBeforeStart = input.lastIndexOf('\n', relevantAsmStartsAt);
            if (lastLinefeedBeforeStart !== -1) {
                input =
                    input.substr(0, input.indexOf('00000000004')) + '\n' +
                    input.substr(lastLinefeedBeforeStart + 1);
            } else {
                input =
                    input.substr(0, input.indexOf('00000000004')) + '\n' +
                    input.substr(relevantAsmStartsAt);
            }
        }
        return input;
    }

    getObjdumpOutputFilename(defaultOutputFilename) {
        return this.getExecutableFilename(path.dirname(defaultOutputFilename));
    }

    postProcessObjdumpOutput(output) {
        return FPCCompiler.preProcessBinaryAsm(output);
    }

    async saveDummyProjectFile(filename) {
        const unitName = path.basename(this.compileFilename, this.lang.extensions[0]);

        await fs.writeFile(filename,
            'program prog; ' +
            'uses ' + unitName + ' in \'' + this.compileFilename + '\'; ' +
            'begin ' +
            'end.');
    }

    async runCompiler(compiler, options, inputFilename, execOptions) {
        if (!execOptions) {
            execOptions = this.getDefaultExecOptions();
        }

        const dirPath = path.dirname(inputFilename);
        const projectFile = path.join(dirPath, 'prog.dpr');
        execOptions.customCwd = dirPath;
        if (this.nasmPath) {
            execOptions.env = _.clone(process.env);
            execOptions.env.PATH = execOptions.env.PATH + ':' + this.nasmPath;
        }
        await this.saveDummyProjectFile(projectFile);

        options.pop();
        options.push('-FE' + dirPath, '-B', projectFile);

        return this.parseOutput(await this.exec(compiler, options, execOptions), inputFilename, dirPath);
    }

    parseOutput(result, inputFilename, tempPath) {
        const fileWithoutPath = path.basename(inputFilename);
        result.inputFilename = fileWithoutPath;
        result.stdout = utils.parseOutput(result.stdout, fileWithoutPath, tempPath);
        result.stderr = utils.parseOutput(result.stderr, fileWithoutPath, tempPath);
        return result;
    }

    execBinary(executable, maxSize, executeParameters, homeDir) {
        executable = this.getExecutableFilename(path.dirname(executable));

        return super.execBinary(executable, maxSize, executeParameters, homeDir);
    }

    getArgumentParser() {
        return PascalParser;
    }

    getExtraAsmHint(asm) {
        if (asm.startsWith('# [')) {
            const bracketEndPos = asm.indexOf(']', 3);
            let valueInBrackets = asm.substr(3, bracketEndPos - 3);
            const colonPos = valueInBrackets.indexOf(':');
            if (colonPos !== -1) {
                valueInBrackets = valueInBrackets.substr(0, colonPos - 1);
            }

            if (!isNaN(valueInBrackets)) {
                return '  .loc 1 ' + valueInBrackets + ' 0';
            } else if (valueInBrackets.includes(this.compileFilename)) {
                return '  .file 1 "<stdin>"';
            } else {
                return false;
            }
        } else if (asm.startsWith('.Le')) {
            return '  .cfi_endproc';
        } else {
            return false;
        }
    }

    preProcessLines(asmLines) {
        let i = 0;

        while (i < asmLines.length) {
            const extraHint = this.getExtraAsmHint(asmLines[i]);
            if (extraHint) {
                i++;
                asmLines.splice(i, 0, extraHint);
            } else {
                this.demangler.addDemangleToCache(asmLines[i]);
            }

            i++;
        }

        return asmLines;
    }
}
