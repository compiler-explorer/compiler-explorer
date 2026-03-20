// Copyright (c) 2023, Compiler Explorer Authors
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

import path from 'node:path';

import Semver from 'semver';

import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {unwrap} from '../assert.js';
import {BaseCompiler} from '../base-compiler.js';
import {CompilationEnvironment} from '../compilation-env.js';
import {SassAsmParser} from '../parsers/asm-parser-sass.js';
import {asSafeVer} from '../utils.js';
import {ClangParser} from './argument-parsers.js';

export class NVVMCompiler extends BaseCompiler {
    static get key() {
        return 'nvvm';
    }

    constructor(info: PreliminaryCompilerInfo, env: CompilationEnvironment) {
        super(info, env);
        this.asm = new SassAsmParser(this.compilerProps);
    }

    override getOutputFilename(dirPath) {
        return path.join(dirPath, 'example.n001.ptx');
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string, userOptions?: string[]) {
        return ['--nvptx64-nvidia-cuda', '-o', this.filename(outputFilename)];
    }

    override async objdump(outputFilename: string, result: any, maxSize: number) {
        // ptx -> cubin
        const ptxas_args = ['-o', 'example.n001.bin'];
        const execOptions = {maxOutput: maxSize, customCwd: path.dirname(outputFilename)};
        await this.exec('ptxas', ptxas_args, execOptions);

        // nvdisasm cubin -> SASS
        const {nvdisasm, semver} = this.compiler;

        const args = Semver.lt(asSafeVer(semver), '11.0.0', true)
            ? [outputFilename, '-c', '-g']
            : [outputFilename, '-c', '-g', '-hex'];

        const {code, execTime, stdout} = await this.exec(unwrap(nvdisasm), args, {
            maxOutput: maxSize,
            customCwd: result.dirPath,
        });

        if (code === 0) {
            result.objdumpTime = execTime;
            result.asm = stdout;
        } else {
            result.asm = `<No output: ${path.basename(unwrap(nvdisasm))} returned ${code}>`;
        }
        return result;
    }

    override getArgumentParserClass() {
        return ClangParser;
    }
}
