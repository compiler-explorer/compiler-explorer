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

import { BaseCompiler } from '../base-compiler';
import { MapFileReaderDelphi } from '../map-file-delphi';
import { PELabelReconstructor } from '../pe32-support';
import * as utils from '../utils';

import { PascalUtils } from './pascal-utils';

export class PascalWinCompiler extends BaseCompiler {
    static get key() { return 'pascal-win'; }

    constructor(info, env) {
        super(info, env);
        info.supportsFiltersInBinary = true;

        this.mapFilename = false;
        this.compileFilename = 'output.pas';
        this.dprFilename = 'prog.dpr';
        this.pasUtils = new PascalUtils();
    }

    getSharedLibraryPathsAsArguments() {
        return [];
    }

    exec(command, args, options) {
        if (process.platform === 'linux' || process.platform === 'darwin') {
            const wine = this.env.gccProps('wine');

            args = args.slice(0);
            if (command.toLowerCase().endsWith('.exe')) {
                args.unshift(command);
                command = wine;
            }
        }

        return super.exec(command, args, options);
    }

    getExecutableFilename(dirPath) {
        return path.join(dirPath, 'prog.exe');
    }

    getOutputFilename(dirPath) {
        return path.join(dirPath, 'prog.exe');
    }

    filename(fn) {
        if (process.platform === 'linux' || process.platform === 'darwin') {
            return 'Z:' + fn;
        } else {
            return super.filename(fn);
        }
    }

    async objdump(outputFilename, result, maxSize, intelAsm) {
        const dirPath = path.dirname(outputFilename);
        const execBinary = this.getExecutableFilename(dirPath);
        if (await utils.fileExists(execBinary)) {
            outputFilename = execBinary;
        } else {
            outputFilename = this.getOutputFilename(path.dirname(outputFilename));
        }

        let args = ['-d', outputFilename];
        if (intelAsm) args = args.concat(['-M', 'intel']);
        return this.exec(this.compiler.objdumper, args, {maxOutput: 1024 * 1024 * 1024})
            .then((objResult) => {
                if (objResult.code !== 0) {
                    result.asm = '<No output: objdump returned ' + objResult.code + '>';
                } else {
                    result.asm = objResult.stdout;
                }

                return result;
            });
    }

    async saveDummyProjectFile(filename, unitName, unitPath) {
        await fs.writeFile(filename,
            'program prog;\n' +
            'uses ' + unitName + ' in \'' + unitPath + '\';\n' +
            'begin\n' +
            'end.\n');
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

        await fs.writeFile(inputFilename, source);

        if (files && files.length > 0) {
            filters.dontMaskFilenames = true;

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

        let alreadyHasDPR = path.basename(inputFilename) === this.dprFilename;

        const tempPath = path.dirname(inputFilename);
        const projectFile = path.join(tempPath, this.dprFilename);

        this.mapFilename = path.join(tempPath, 'prog.map');

        inputFilename = inputFilename.replace(/\//g, '\\');

        if (!alreadyHasDPR) {
            const unitFilepath = path.basename(inputFilename);
            const unitName = unitFilepath.replace(/.pas$/i, '');
            await this.saveDummyProjectFile(projectFile, unitName, unitFilepath);
        }

        options.pop();

        options.unshift(
            '-CC',
            '-W',
            '-H',
            '-GD',
            '-$D+',
            '-V',
            '-B');

        options.push(projectFile);
        execOptions.customCwd = tempPath;

        return this.exec(compiler, options, execOptions).then((result) => {
            result.inputFilename = inputFilename;
            result.stdout = utils.parseOutput(result.stdout, inputFilename);
            result.stderr = utils.parseOutput(result.stderr, inputFilename);
            return result;
        });
    }

    optionsForFilter(filters) {
        filters.binary = true;
        filters.dontMaskFilenames = true;
        filters.preProcessBinaryAsmLines = (asmLines) => {
            const mapFileReader = new MapFileReaderDelphi(this.mapFilename);
            const reconstructor = new PELabelReconstructor(asmLines, false, mapFileReader, false);
            reconstructor.run('output');
    
            return reconstructor.asmLines;
        };

        return [];
    }
}
