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

import Semver from 'semver';
import _ from 'underscore';

import {PreliminaryCompilerInfo} from '../../types/compiler.interfaces';
import {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces';
import {BaseCompiler} from '../base-compiler';
import {asSafeVer} from '../utils';

import {ISPCParser} from './argument-parsers';
import {unwrap} from '../assert';

export class ISPCCompiler extends BaseCompiler {
    static get key() {
        return 'ispc';
    }

    constructor(info: PreliminaryCompilerInfo, env) {
        super(info, env);
        this.compiler.supportsIrView = true;
        this.compiler.irArg = ['--emit-llvm-text'];
    }

    override couldSupportASTDump(version: string) {
        return Semver.gte(asSafeVer(this.compiler.semver), '1.18.0', true);
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string) {
        let options = ['--target=avx2-i32x8', '--emit-asm', '-g', '-o', this.filename(outputFilename)];
        if (this.compiler.intelAsm && filters.intel && !filters.binary) {
            options = options.concat(this.compiler.intelAsm.split(' '));
        }
        return options;
    }

    override async generateIR(inputFilename: string, options: string[], filters: ParseFiltersAndOutputOptions) {
        const newOptions = [...options, ...unwrap(this.compiler.irArg), '-o', this.getIrOutputFilename(inputFilename)];
        return super.generateIR(inputFilename, newOptions, filters);
    }

    override getArgumentParser() {
        return ISPCParser;
    }

    override async generateAST(inputFilename, options) {
        // These options make Clang produce an AST dump
        const newOptions = _.filter(options, option => option !== '--colored-output').concat(['--ast-dump']);

        const execOptions = this.getDefaultExecOptions();
        // A higher max output is needed for when the user includes headers
        execOptions.maxOutput = 1024 * 1024 * 1024;

        return this.llvmAst.processAst(
            await this.runCompiler(this.compiler.exe, newOptions, this.filename(inputFilename), execOptions)
        );
    }

    override isCfgCompiler(/*compilerVersion*/) {
        return true;
    }
}
