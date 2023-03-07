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

import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';

import {RustCompiler} from './rust.js';

export class RustcCgGCCCompiler extends RustCompiler {
    static override get key() {
        return 'rustc-cg-gcc';
    }

    constructor(info: PreliminaryCompilerInfo, env) {
        super(info, env);
        this.compiler.supportsIrView = false;

        // Enable GCC dump code,
        this.compiler.supportsGccDump = true;

        // but do not try to filter out empty dumps:
        // This filtering needs to dump all files but currently libgccjit (used by
        // this rustc backend) and compiler-explorer's use of jails makes this task a bit tricky.
        // The user will be presented more dumps than effectively created.
        // Turning this to 'true' will break the dumps.

        this.compiler.removeEmptyGccDump = false;
    }

    override getGccDumpOptions(gccDumpOptions, outputFilename: string) {
        return ['-C', 'llvm-args=' + super.getGccDumpOptions(gccDumpOptions, outputFilename).join(' ')];
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string, userOptions?: string[]) {
        // these options are direcly taken from rustc_codegen_gcc doc.
        // See https://github.com/antoyo/rustc_codegen_gcc
        const toolroot = path.resolve(path.dirname(this.compiler.exe), '..');

        let options = [
            '-C',
            'panic=abort',
            '-Z',
            'codegen-backend=librustc_codegen_gcc.so',
            '--sysroot',
            toolroot + '/sysroot',
        ];

        // rust.js makes the asumption that LLVM is used. This may go away when cranelift is available.
        // Until this is the case and the super() class is refactored, simply ditch -Cllvm arg.
        const super_options = super
            .optionsForFilter(filters, outputFilename, userOptions)
            .filter(arg => !/-Cllvm.*/.test(arg));
        options = options.concat(super_options);
        return options;
    }
}
