// Copyright (c) 2022, Compiler Explorer Authors
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

import type {ExecutionOptions} from '../../types/compilation/compilation.interfaces.js';
import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {ArtifactType} from '../../types/tool.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';
import {logger} from '../logger.js';
import {AsmParserZ88dk} from '../parsers/asm-parser-z88dk.js';
import * as utils from '../utils.js';
import {Z88dkParser} from './argument-parsers.js';

export class z88dkCompiler extends BaseCompiler {
    static get key() {
        return 'z88dk';
    }

    constructor(compilerInfo: PreliminaryCompilerInfo, env) {
        super(compilerInfo, env);
        this.outputFilebase = 'example';
        this.asm = new AsmParserZ88dk(this.compilerProps);
    }

    protected override getArgumentParser() {
        return Z88dkParser;
    }

    override getTargetFlags(): string[] {
        return ['+<value>'];
    }

    public override getOutputFilename(dirPath: string, outputFilebase: string, key?: any): string {
        let filename;
        if (key && key.backendOptions && key.backendOptions.customOutputFilename) {
            filename = key.backendOptions.customOutputFilename;
        } else if (key && key.filters.binary) {
            filename = `${outputFilebase}`;
        } else {
            filename = `${outputFilebase}.c.asm`;
        }

        if (dirPath) {
            return path.join(dirPath, filename);
        } else {
            return filename;
        }
    }

    public override orderArguments(
        options: string[],
        inputFilename: string,
        libIncludes: string[],
        libOptions: string[],
        libPaths: string[],
        libLinks: string[],
        userOptions: string[],
        staticLibLinks: string[],
    ) {
        let targetOpt = options.filter(opt => opt.startsWith('+'));
        const withoutTarget = options.filter(opt => !opt.startsWith('+'));
        const withoutTargetUser = userOptions.filter(opt => !opt.startsWith('+'));

        if (targetOpt.length === 0) {
            targetOpt = userOptions.filter(opt => opt.startsWith('+'));
        }

        return targetOpt.concat(
            withoutTargetUser,
            withoutTarget,
            [this.filename(inputFilename)],
            libIncludes,
            libOptions,
            libPaths,
            libLinks,
            staticLibLinks,
        );
    }

    protected override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string): string[] {
        if (filters.binary) {
            return ['-o', outputFilename + '.s', '-create-app'];
        } else {
            return ['-S'];
        }
    }

    override getDefaultExecOptions(): ExecutionOptions & {env: Record<string, string>} {
        const opts = super.getDefaultExecOptions();
        opts.env.ZCCCFG = path.join(path.dirname(this.compiler.exe), '../share/z88dk/lib/config');
        opts.env.PATH = process.env.PATH + path.delimiter + path.dirname(this.compiler.exe);

        return opts;
    }

    override getObjdumpOutputFilename(defaultOutputFilename: string) {
        return defaultOutputFilename;
    }

    getTapefilename() {
        return `${this.outputFilebase}.tap`;
    }

    getSmsfilename() {
        return `${this.outputFilebase}.sms`;
    }

    override async objdump(
        outputFilename,
        result: any,
        maxSize: number,
        intelAsm,
        demangle,
        staticReloc: boolean,
        dynamicReloc: boolean,
        filters: ParseFiltersAndOutputOptions,
    ) {
        outputFilename = this.getObjdumpOutputFilename(outputFilename);

        // sometimes (with +z80 for example) the .bin file is written and the .s file is empty
        if (await utils.fileExists(outputFilename + '.bin')) {
            outputFilename += '.bin';
        } else {
            if (await utils.fileExists(outputFilename + '.s')) {
                outputFilename += '.s';
            } else {
                result.asm = '<No output file ' + outputFilename + '.s>';
                return result;
            }
        }

        const args = [...this.compiler.objdumperArgs, outputFilename];

        if (this.externalparser) {
            const objResult = await this.externalparser.objdumpAndParseAssembly(result.dirPath, args, filters);
            if (objResult.parsingTime !== undefined) {
                objResult.objdumpTime = parseInt(result.execTime) - parseInt(result.parsingTime);
                delete objResult.execTime;
            }

            result = {...result, ...objResult};
        } else {
            const execOptions: ExecutionOptions = {
                maxOutput: maxSize,
                customCwd: (result.dirPath as string) || path.dirname(outputFilename),
            };
            const objResult = await this.exec(this.compiler.objdumper, args, execOptions);

            if (objResult.code === 0) {
                result.objdumpTime = objResult.execTime;
                result.asm = this.postProcessObjdumpOutput(objResult.stdout);
            } else {
                logger.error(`Error executing objdump ${this.compiler.objdumper}`, objResult);
                result.asm = `<No output: objdump returned ${objResult.code}>`;
            }
        }

        if (result.code === 0 && filters.binary) {
            const tapeFilepath = path.join(result.dirPath, this.getTapefilename());
            if (await utils.fileExists(tapeFilepath)) {
                await this.addArtifactToResult(result, tapeFilepath, ArtifactType.zxtape);
            }

            const smsFilepath = path.join(result.dirPath, this.getSmsfilename());
            if (await utils.fileExists(smsFilepath)) {
                await this.addArtifactToResult(result, smsFilepath, ArtifactType.smsrom);
            }
        }

        return result;
    }
}
