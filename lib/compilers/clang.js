// Copyright (c) 2018, Compiler Explorer Authors
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

import { SassAsmParser } from '../asm-parser-sass';
import { BaseCompiler } from '../base-compiler';

export class ClangCompiler extends BaseCompiler {
    static get key() { return 'clang'; }

    runCompiler(compiler, options, inputFilename, execOptions) {
        if (!execOptions) {
            execOptions = this.getDefaultExecOptions();
        }

        execOptions.customCwd = path.dirname(inputFilename);

        return super.runCompiler(compiler, options, inputFilename, execOptions);
    }
}

export class ClangCudaCompiler extends ClangCompiler {
    static get key() { return 'clang-cuda'; }
    constructor(info, env) {
        super(info, env);

        this.asm = new SassAsmParser();
    }

    optionsForFilter(filters, outputFilename) {
        return ['-o', this.filename(outputFilename), '-g1', filters.binary ? '-c' : '-S'];
    }

    async objdump(outputFilename, result, maxSize) {
        // For nvdisasm.
        const args = [outputFilename, '-c', '-g', '-hex'];
        const execOptions = {maxOutput: maxSize, customCwd: path.dirname(outputFilename)};

        const objResult = await this.exec(this.compiler.objdumper, args, execOptions);
        result.asm = objResult.stdout;
        if (objResult.code !== 0) {
            result.asm = `<No output: nvdisasm returned ${objResult.code}>`;
        } else {
            result.objdumpTime = objResult.execTime;
        }
        return result;
    }
}
