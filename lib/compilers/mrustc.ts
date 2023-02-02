// Copyright (c) 2021, Marc Poulhiès
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

import {ExecutionOptions} from '../../types/compilation/compilation.interfaces';
import {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces';
import {BaseCompiler} from '../base-compiler';

import {MrustcParser} from './argument-parsers';

export class MrustcCompiler extends BaseCompiler {
    static get key() {
        return 'mrustc';
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string) {
        // mrustc always dumps the C code for <baseout> target in the <baseout>.c file.
        // In our case, the actual file in -o is not even created because we are
        // faking the last step (C compilation).
        // Craft the 'outname' to have the intermediate .c file writen in outputFilename.
        const outname = path.join(
            path.dirname(this.filename(outputFilename)),
            path.basename(this.filename(outputFilename), '.c'),
        );

        // Currently always targets a rlib, no binary support at the moment.
        return [
            '--crate-type',
            'rlib',
            '-o',
            outname,
            '-L',
            path.join(path.dirname(this.compiler.exe), '..', 'output'),
        ];
    }

    override async runCompiler(
        compiler: string,
        options: string[],
        inputFilename: string,
        execOptions: ExecutionOptions,
    ) {
        // mrustc will always invoke a C compiler on its C output to create a final exec/object.
        // There's no easy way to disable this last step, so simply faking it with 'true' works.
        execOptions.env.CC = 'true';

        return super.runCompiler(compiler, options, inputFilename, execOptions);
    }

    override getOutputFilename(dirPath: string) {
        return path.join(dirPath, `${path.basename(this.compileFilename, this.lang.extensions[0])}.c`);
    }

    override getArgumentParser() {
        return MrustcParser;
    }
}
