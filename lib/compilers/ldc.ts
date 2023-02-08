// Copyright (c) 2016, Compiler Explorer Authors
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
import semverParser from 'semver';

import {CompilerInfo} from '../../types/compiler.interfaces';
import {BaseCompiler} from '../base-compiler';
import {logger} from '../logger';

import {LDCParser} from './argument-parsers';

export class LDCCompiler extends BaseCompiler {
    static get key() {
        return 'ldc';
    }

    asanSymbolizerPath: string;

    constructor(info: CompilerInfo, env) {
        super(info, env);
        this.compiler.supportsIntel = true;
        this.compiler.supportsIrView = true;
        this.compiler.irArg = ['-output-ll'];

        this.asanSymbolizerPath = this.compilerProps<string>('llvmSymbolizer');
    }

    override runExecutable(executable, executeParameters, homeDir) {
        if (this.asanSymbolizerPath) {
            executeParameters.env = {
                ASAN_SYMBOLIZER_PATH: this.asanSymbolizerPath,
                ...executeParameters.env,
            };
        }
        return super.runExecutable(executable, executeParameters, homeDir);
    }

    override getSharedLibraryPathsAsArguments() {
        return [];
    }

    override getOutputFilename(dirPath: string, outputFilebase: string, key?: any): string {
        if (key && key.filters && key.filters.binary) {
            return path.join(dirPath, 'output');
        } else if (key && key.filters && key.filters.binaryObject) {
            return path.join(dirPath, 'output.o');
        } else {
            return path.join(dirPath, 'output.s');
        }
    }

    override optionsForFilter(filters, outputFilename) {
        const options = ['-gline-tables-only', '-of', this.filename(outputFilename)];
        if (filters.intel && !filters.binary) options.push('-x86-asm-syntax=intel');
        if (!filters.binary && !filters.binaryObject) options.push('-output-s');
        else if (filters.binaryObject) options.push('-c');
        return options;
    }

    override getArgumentParser() {
        return LDCParser;
    }

    override filterUserOptions(userOptions) {
        return userOptions.filter(option => option !== '-run');
    }

    override isCfgCompiler() {
        return true;
    }

    override couldSupportASTDump(version) {
        const versionRegex = /\((\d\.\d+)\.\d+/;
        const versionMatch = versionRegex.exec(version);
        return versionMatch ? semverParser.compare(versionMatch[1] + '.0', '1.4.0', true) >= 0 : false;
    }

    override async generateAST(inputFilename, options) {
        // These options make LDC produce an AST dump in a separate file `<inputFilename>.cg`.
        const newOptions = options.concat('-vcg-ast');
        const execOptions = this.getDefaultExecOptions();
        // TODO(#4654) generateAST expects to return a ResultLine[] not a string
        return this.loadASTOutput(
            await this.runCompiler(this.compiler.exe, newOptions, this.filename(inputFilename), execOptions),
        ) as any;
    }

    async loadASTOutput(output) {
        if (output.code !== 0) {
            return `Error generating AST: ${output.code}`;
        }
        // Load the AST output from the `.cg` file.
        // Demangling is not needed.
        const astFilename = output.inputFilename.concat('.cg');
        try {
            return await fs.readFile(astFilename, 'utf8');
        } catch (e) {
            // TODO(jeremy-rifkin) why does e have .code here
            if (e instanceof Error && (e as any).code === 'ENOENT') {
                logger.warn(`LDC AST file ${astFilename} requested but it does not exist`);
                return '';
            }
            throw e;
        }
    }

    // Override the IR file name method for LDC because the output file is different from clang.
    override getIrOutputFilename(inputFilename) {
        return this.getOutputFilename(path.dirname(inputFilename), this.outputFilebase).replace('.s', '.ll');
    }
}
