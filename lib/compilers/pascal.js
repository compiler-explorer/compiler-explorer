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

import {BaseCompiler} from '../base-compiler';
import * as utils from '../utils';

import {PascalParser} from './argument-parsers';
import {PascalUtils} from './pascal-utils';

export class FPCCompiler extends BaseCompiler {
    static get key() {
        return 'pascal';
    }

    constructor(info, env) {
        super(info, env);

        this.compileFilename = 'output.pas';
        this.dprFilename = 'prog.dpr';
        this.supportsOptOutput = false;
        this.nasmPath = this.compilerProps('nasmpath');
        this.pasUtils = new PascalUtils();
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

        if (filters.binary) {
            filters.dontMaskFilenames = true;
        }

        return options;
    }

    getOutputFilename(dirPath) {
        return path.join(dirPath, `${path.basename(this.compileFilename, this.lang.extensions[0])}.s`);
    }

    getExecutableFilename(dirPath) {
        return path.join(dirPath, 'prog');
    }

    async objdump(outputFilename, result, maxSize, intelAsm, demangle) {
        const dirPath = path.dirname(outputFilename);
        const execBinary = this.getExecutableFilename(dirPath);
        if (await utils.fileExists(execBinary)) {
            return super.objdump(execBinary, result, maxSize, intelAsm, demangle);
        }

        return super.objdump(outputFilename, result, maxSize, intelAsm, demangle);
    }

    static preProcessBinaryAsm(input) {
        const systemInitOffset = input.indexOf('<SYSTEM_$$_init$>');
        const relevantAsmStartsAt = input.indexOf('...', systemInitOffset);
        if (relevantAsmStartsAt !== -1) {
            const lastLinefeedBeforeStart = input.lastIndexOf('\n', relevantAsmStartsAt);
            if (lastLinefeedBeforeStart === -1) {
                input = input.substr(0, input.indexOf('00000000004')) + '\n' + input.substr(relevantAsmStartsAt);
            } else {
                input =
                    input.substr(0, input.indexOf('00000000004')) + '\n' + input.substr(lastLinefeedBeforeStart + 1);
            }
        }
        return input;
    }

    postProcessObjdumpOutput(output) {
        return FPCCompiler.preProcessBinaryAsm(output);
    }

    async saveDummyProjectFile(filename, unitName, unitPath) {
        await fs.writeFile(
            filename,
            // prettier-ignore
            'program prog;\n' +
            'uses ' + unitName + ' in \'' + unitPath + '\';\n' +
            'begin\n' +
            'end.\n',
        );
    }

    async writeAllFiles(dirPath, source, files, filters) {
        let inputFilename;
        if (this.pasUtils.isProgram(source)) {
            inputFilename = path.join(dirPath, this.dprFilename);
        } else {
            const unitName = this.pasUtils.getUnitname(source);
            if (unitName) {
                inputFilename = path.join(dirPath, unitName + '.pas');
            } else {
                inputFilename = path.join(dirPath, this.compileFilename);
            }
        }

        if (source !== '' || !files) {
            await fs.writeFile(inputFilename, source);
        }

        if (files && files.length > 0) {
            await this.writeMultipleFiles(files, dirPath);
        }

        return {
            inputFilename,
        };
    }

    async runCompiler(compiler, options, inputFilename, execOptions) {
        if (!execOptions) {
            execOptions = this.getDefaultExecOptions();
        }

        const alreadyHasDPR = path.basename(inputFilename) === this.dprFilename;
        const dirPath = path.dirname(inputFilename);

        const projectFile = path.join(dirPath, this.dprFilename);
        execOptions.customCwd = dirPath;
        if (this.nasmPath) {
            execOptions.env = _.clone(process.env);
            execOptions.env.PATH = execOptions.env.PATH + ':' + this.nasmPath;
        }

        if (!alreadyHasDPR) {
            const unitFilepath = path.basename(inputFilename);
            const unitName = unitFilepath.replace(/.pas$/i, '');
            await this.saveDummyProjectFile(projectFile, unitName, unitFilepath);
        }

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

    getArgumentParser() {
        return PascalParser;
    }

    getExtraAsmHint(asm, currentFileId) {
        if (asm.startsWith('# [')) {
            const bracketEndPos = asm.indexOf(']', 3);
            let valueInBrackets = asm.substr(3, bracketEndPos - 3);
            const colonPos = valueInBrackets.indexOf(':');
            if (colonPos !== -1) {
                valueInBrackets = valueInBrackets.substr(0, colonPos - 1);
            }

            if (valueInBrackets.startsWith('/')) {
                valueInBrackets = valueInBrackets.substr(1);
            }

            if (isNaN(valueInBrackets)) {
                return `  .file ${currentFileId} "${valueInBrackets}"`;
            } else {
                return `  .loc ${currentFileId} ${valueInBrackets} 0`;
            }
        } else if (asm.startsWith('.Le')) {
            return '  .cfi_endproc';
        } else {
            return false;
        }
    }

    tryGetFilenumber(asm, files) {
        if (asm.startsWith('# [')) {
            const bracketEndPos = asm.indexOf(']', 3);
            let valueInBrackets = asm.substr(3, bracketEndPos - 3);
            const colonPos = valueInBrackets.indexOf(':');
            if (colonPos !== -1) {
                valueInBrackets = valueInBrackets.substr(0, colonPos - 1);
            }

            if (valueInBrackets.startsWith('/')) {
                valueInBrackets = valueInBrackets.substr(1);
            }

            if (isNaN(valueInBrackets)) {
                if (!files[valueInBrackets]) {
                    let maxFileId = _.max(files);
                    if (maxFileId === -Infinity) {
                        maxFileId = 0;
                    }

                    files[valueInBrackets] = maxFileId + 1;
                    return maxFileId + 1;
                }
            }
        }

        return false;
    }

    preProcessLines(asmLines) {
        let i = 0;
        const files = {};
        let currentFileId = 1;

        while (i < asmLines.length) {
            const newFileId = this.tryGetFilenumber(asmLines[i], files);
            if (newFileId) currentFileId = newFileId;

            const extraHint = this.getExtraAsmHint(asmLines[i], currentFileId);
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
