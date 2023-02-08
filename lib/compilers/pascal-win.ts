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

import {ExecutionOptions} from '../../types/compilation/compilation.interfaces';
import {CompilerInfo} from '../../types/compiler.interfaces';
import {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces';
import {unwrap} from '../assert';
import {BaseCompiler} from '../base-compiler';
import {MapFileReaderDelphi} from '../mapfiles/map-file-delphi';
import {PELabelReconstructor} from '../pe32-support';
import * as utils from '../utils';

import {PascalUtils} from './pascal-utils';

export class PascalWinCompiler extends BaseCompiler {
    static get key() {
        return 'pascal-win';
    }

    mapFilename: string | null;
    dprFilename: string;
    pasUtils: PascalUtils;

    constructor(info: CompilerInfo, env) {
        super(info, env);
        info.supportsFiltersInBinary = true;

        this.mapFilename = null;
        this.compileFilename = 'output.pas';
        this.dprFilename = 'prog.dpr';
        this.pasUtils = new PascalUtils();
    }

    override getSharedLibraryPathsAsArguments() {
        return [];
    }

    override exec(command: string, args: string[], options: ExecutionOptions) {
        if (process.platform === 'linux' || process.platform === 'darwin') {
            const wine = this.env.ceProps<string>('wine');

            args = args.slice(0);
            if (command.toLowerCase().endsWith('.exe')) {
                args.unshift(command);
                command = wine;
            }
        }

        return super.exec(command, args, options);
    }

    override getExecutableFilename(dirPath: string) {
        return path.join(dirPath, 'prog.exe');
    }

    override getOutputFilename(dirPath: string) {
        return path.join(dirPath, 'prog.exe');
    }

    override filename(fn) {
        if (process.platform === 'linux' || process.platform === 'darwin') {
            return 'Z:' + fn;
        } else {
            return super.filename(fn);
        }
    }

    override async objdump(outputFilename, result, maxSize: number, intelAsm) {
        const dirPath = path.dirname(outputFilename);
        const execBinary = this.getExecutableFilename(dirPath);
        if (await utils.fileExists(execBinary)) {
            outputFilename = execBinary;
        } else {
            outputFilename = this.getOutputFilename(path.dirname(outputFilename));
        }

        let args = ['-d', outputFilename];
        if (intelAsm) args = args.concat(['-M', 'intel']);
        return this.exec(this.compiler.objdumper, args, {maxOutput: 1024 * 1024 * 1024}).then(objResult => {
            if (objResult.code === 0) {
                result.asm = objResult.stdout;
            } else {
                result.asm = '<No output: objdump returned ' + objResult.code + '>';
            }

            return result;
        });
    }

    async saveDummyProjectFile(filename: string, unitName: string, unitPath: string) {
        await fs.writeFile(
            filename,
            // prettier-ignore
            'program prog;\n' +
            'uses ' + unitName + ' in \'' + unitPath + '\';\n' +
            'begin\n' +
            'end.\n',
        );
    }

    override async writeAllFiles(dirPath: string, source: string, files: any[], filters: ParseFiltersAndOutputOptions) {
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
            await this.writeMultipleFiles(files, dirPath);
        }

        return {
            inputFilename,
        };
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

        const alreadyHasDPR = path.basename(inputFilename) === this.dprFilename;

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

        options.unshift('-CC', '-W', '-H', '-GD', '-$D+', '-V', '-B');

        options.push(projectFile);
        execOptions.customCwd = tempPath;

        return this.exec(compiler, options, execOptions).then(result => {
            return {
                ...result,
                inputFilename,
                stdout: utils.parseOutput(result.stdout, inputFilename),
                stderr: utils.parseOutput(result.stderr, inputFilename),
            };
        });
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions) {
        filters.binary = true;
        filters.dontMaskFilenames = true;
        (filters as any).preProcessBinaryAsmLines = asmLines => {
            const mapFileReader = new MapFileReaderDelphi(unwrap(this.mapFilename));
            const reconstructor = new PELabelReconstructor(asmLines, false, mapFileReader, false);
            reconstructor.run('output');

            return reconstructor.asmLines;
        };

        return [];
    }
}
