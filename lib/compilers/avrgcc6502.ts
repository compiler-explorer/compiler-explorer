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

import type {ExecutionOptions} from '../../types/compilation/compilation.interfaces.js';
import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';

export class AvrGcc6502Compiler extends BaseCompiler {
    private readonly avrgccpath: string;
    private readonly xapath: string;
    private readonly avrlibstdcpppath: string;

    static get key() {
        return 'avrgcc6502';
    }

    constructor(compilerInfo: PreliminaryCompilerInfo, env) {
        super(compilerInfo, env);

        this.avrgccpath = this.compilerProps<string>(`compiler.${this.compiler.id}.avrgccpath`);
        this.xapath = this.compilerProps<string>(`compiler.${this.compiler.id}.xapath`);
        this.avrlibstdcpppath = this.compilerProps<string>(`compiler.${this.compiler.id}.avrlibstdcpppath`);
        this.outputFilebase = 'example';
    }

    public override getOutputFilename(dirPath: string, outputFilebase: string, key?: any) {
        let filename;
        if (key && key.backendOptions && key.backendOptions.customOutputFilename) {
            filename = key.backendOptions.customOutputFilename;
        } else {
            filename = `${outputFilebase}.6502.asm`;
        }

        if (dirPath) {
            return path.join(dirPath, filename);
        } else {
            return filename;
        }
    }

    protected override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string): string[] {
        return [`-I${this.avrlibstdcpppath}`];
    }

    public override async runCompiler(
        compiler: string,
        options: string[],
        inputFilename: string,
        execOptions: ExecutionOptions & {env: Record<string, string>},
    ) {
        if (!execOptions) {
            execOptions = this.getDefaultExecOptions();
        }

        execOptions.env = {
            PATH: path.join(this.avrgccpath, 'bin') + path.delimiter + this.xapath,
        };

        return await super.runCompiler(compiler, options, inputFilename, execOptions);
    }
}
