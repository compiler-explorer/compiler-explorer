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

import _ from 'underscore';

import { BaseCompiler } from '../base-compiler';

import { RustParser } from './argument-parsers';

export class RustCompiler extends BaseCompiler {
    static get key() { return 'rust'; }

    constructor(info, env) {
        super(info, env);
        this.compiler.supportsIntel = true;
        this.compiler.supportsIrView = true;
        this.compiler.irArg = ['--emit', 'llvm-ir'];
        this.linker = this.compilerProps('linker');
    }

    getSharedLibraryPathsAsArguments() {
        return [];
    }

    optionsForFilter(filters, outputFilename, userOptions) {
        let options = ['-C', 'debuginfo=1', '-o', this.filename(outputFilename)];

        const userRequestedEmit = _.any(userOptions, opt => opt.includes('--emit'));
        if (filters.binary) {
            options = options.concat(['--crate-type', 'bin']);
            if (this.linker) {
                options = options.concat(`-Clinker=${this.linker}`);
            }
        } else if (!filters.binary) {
            if (!userRequestedEmit) {
                options = options.concat('--emit', 'asm');
            }
            if (filters.intel) options = options.concat('-Cllvm-args=--x86-asm-syntax=intel');
            options = options.concat(['--crate-type', 'rlib']);
        }
        return options;
    }

    // Override the IR file name method for rustc because the output file is different from clang.
    getIrOutputFilename(inputFilename) {
        return this.getOutputFilename(path.dirname(inputFilename), this.outputFilebase)
            .replace('.s', '.ll');
    }

    getArgumentParser() {
        return RustParser;
    }

    isCfgCompiler(/*compilerVersion*/) {
        return true;
    }
}
