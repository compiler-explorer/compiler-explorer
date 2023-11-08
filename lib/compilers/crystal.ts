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

import semverParser from 'semver';
import _ from 'underscore';

import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {unwrap} from '../assert.js';
import {BaseCompiler} from '../base-compiler.js';
import {CrystalAsmParser} from '../parsers/asm-parser-crystal.js';

import {CrystalParser} from './argument-parsers.js';

export class CrystalCompiler extends BaseCompiler {
    static get key() {
        return 'crystal';
    }

    ccPath: string;

    constructor(compiler: PreliminaryCompilerInfo, env) {
        super(compiler, env);
        this.asm = new CrystalAsmParser();
        this.compiler.supportsIrView = true;
        this.compiler.irArg = ['--emit', 'llvm-ir'];
        this.ccPath = this.compilerProps<string>(`compiler.${this.compiler.id}.cc`);
    }

    override getSharedLibraryPathsAsArguments() {
        return [];
    }

    override getDefaultExecOptions() {
        const execOptions = super.getDefaultExecOptions();
        if (this.ccPath) {
            execOptions.env.CC = this.ccPath;
        }
        return execOptions;
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string, userOptions?: string[]) {
        const output = this.filename(this.getExecutableFilename(path.dirname(outputFilename), this.outputFilebase));
        let options = ['build', '-o', output];

        const userRequestedEmit = _.any(unwrap(userOptions), opt => opt.includes('--emit'));
        if (!filters.binary) {
            if (!userRequestedEmit) {
                options = options.concat('--emit', 'asm');
            }
        }

        return options;
    }

    override getIrOutputFilename(inputFilename: string, filters: ParseFiltersAndOutputOptions): string {
        if (this.usesNewEmitFilenames()) {
            return this.getOutputFilename(path.dirname(inputFilename), this.outputFilebase).replace('.s', '.ll');
        } else {
            return super.getIrOutputFilename(inputFilename, filters);
        }
    }

    override getOutputFilename(dirPath: string, outputFilebase: string) {
        if (this.usesNewEmitFilenames()) {
            return path.join(dirPath, `${outputFilebase}.s`);
        } else {
            return path.join(dirPath, `${path.basename(this.compileFilename, this.lang.extensions[0])}.s`);
        }
    }

    override getExecutableFilename(dirPath: string, outputFilebase: string) {
        return path.join(dirPath, outputFilebase);
    }

    override getObjdumpOutputFilename(defaultOutputFilename: string) {
        return this.getExecutableFilename(path.dirname(defaultOutputFilename), this.outputFilebase);
    }

    override getArgumentParser() {
        return CrystalParser;
    }

    private usesNewEmitFilenames(): boolean {
        const versionRegex = /Crystal (\d+\.\d+\.\d+)/;
        const versionMatch = versionRegex.exec(this.compiler.version);
        return versionMatch ? semverParser.compare(versionMatch[1], '1.9.0', true) >= 0 : false;
    }
}
