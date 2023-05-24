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

import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';

import {Win32Compiler} from './win32.js';
import {unwrap} from '../assert.js';

export class ClangCLCompiler extends Win32Compiler {
    static override get key() {
        return 'clang-cl';
    }

    constructor(info: PreliminaryCompilerInfo, env) {
        super(info, env);

        this.compiler.supportsIrView = true;
        this.compiler.irArg = ['-Xclang', '-emit-llvm'];
        this.compiler.minIrArgs = ['-emit-llvm'];
        this.compiler.supportsIntel = false;
        this.compiler.includeFlag = '/clang:-isystem';
    }

    override async generateIR(inputFilename: string, options: string[], filters: ParseFiltersAndOutputOptions) {
        // These options make Clang produce an IR
        const newOptions = options
            .filter(option => option !== '/FA' && !option.startsWith('/Fa'))
            .concat(unwrap(this.compiler.irArg));

        const execOptions = this.getDefaultExecOptions();
        // A higher max output is needed for when the user includes headers
        execOptions.maxOutput = 1024 * 1024 * 1024;

        const output = await this.runCompiler(this.compiler.exe, newOptions, this.filename(inputFilename), execOptions);
        if (output.code !== 0) {
            return [{text: 'Failed to run compiler to get IR code'}];
        }
        const ir = await this.processIrOutput(output, filters);
        return ir.asm;
    }

    override getIrOutputFilename(inputFilename: string, filters: ParseFiltersAndOutputOptions): string {
        return this.filename(path.dirname(inputFilename) + '/output.s.obj');
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string) {
        const options = super.optionsForFilter(filters, outputFilename);

        // Force the debugging info flag or we can't source locations.
        return ['/Z7'].concat(options);
    }
}
