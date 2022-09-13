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

import _ from 'underscore';

import {BaseCompiler} from '../base-compiler';
import {CrystalAsmParser} from '../parsers/asm-parser-crystal';

import {CrystalParser} from './argument-parsers';

export class CrystalCompiler extends BaseCompiler {
    static get key() {
        return 'crystal';
    }

    constructor(compiler, env) {
        super(compiler, env);
        this.asm = new CrystalAsmParser();
        this.compiler.supportsIrView = true;
        this.compiler.irArg = ['--emit', 'llvm-ir'];
        this.ccPath = this.compilerProps(`compiler.${this.compiler.id}.cc`);
    }

    getSharedLibraryPathsAsArguments() {
        return [];
    }

    getDefaultExecOptions() {
        const execOptions = super.getDefaultExecOptions();
        if (this.ccPath) {
            execOptions.env.CC = this.ccPath;
        }
        return execOptions;
    }

    optionsForFilter(filters, outputFilename, userOptions) {
        const output = this.filename(this.getExecutableFilename(path.dirname(outputFilename), this.outputFilebase));
        let options = ['build', '-o', output];

        const userRequestedEmit = _.any(userOptions, opt => opt.includes('--emit'));
        if (!filters.binary) {
            if (!userRequestedEmit) {
                options = options.concat('--emit', 'asm');
            }
        }

        return options;
    }

    getOutputFilename(dirPath) {
        return path.join(dirPath, `${path.basename(this.compileFilename, this.lang.extensions[0])}.s`);
    }

    getExecutableFilename(dirPath, outputFilebase) {
        return path.join(dirPath, outputFilebase);
    }

    getObjdumpOutputFilename(defaultOutputFilename) {
        return this.getExecutableFilename(path.dirname(defaultOutputFilename), this.outputFilebase);
    }

    getArgumentParser() {
        return CrystalParser;
    }
}
