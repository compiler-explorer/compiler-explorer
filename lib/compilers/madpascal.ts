// Copyright (c) 2024, Compiler Explorer Authors
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

import {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {BaseCompiler, c_value_placeholder} from '../base-compiler.js';
import {CompilationEnvironment} from '../compilation-env.js';
import {MadpascalParser} from './argument-parsers.js';
import * as path from 'path';
import {MadsAsmParser} from '../parsers/asm-parser-mads.js';

export class MadPascalCompiler extends BaseCompiler {
    protected madsExe: any;

    static get key() {
        return 'madpascal';
    }

    constructor(info: PreliminaryCompilerInfo, env: CompilationEnvironment) {
        super(info, env);

        this.compileFilename = 'output.pas';
        this.madsExe = this.compilerProps<string>(`compiler.${info.id}.madsexe`);
        this.asm = new MadsAsmParser(this.compilerProps);
    }

    override getOutputFilename(dirPath: string, outputFilebase: string, key?: any): string {
        let filename;
        if (key && key.backendOptions && key.backendOptions.customOutputFilename) {
            filename = key.backendOptions.customOutputFilename;
        } else {
            filename = `${outputFilebase}.a65`;
        }

        if (dirPath) {
            return path.join(dirPath, filename);
        } else {
            return filename;
        }
    }

    protected override getArgumentParser(): any {
        return MadpascalParser;
    }

    protected override optionsForFilter(
        filters: ParseFiltersAndOutputOptions,
        outputFilename: string,
        userOptions?: string[],
    ): string[] {
        filters.demangle = false;
        return [];
    }

    override getTargetFlags(): string[] {
        return [`-target:${c_value_placeholder}`];
    }

    override orderArguments(
        options: string[],
        inputFilename: string,
        libIncludes: string[],
        libOptions: string[],
        libPaths: string[],
        libLinks: string[],
        userOptions: string[],
        staticLibLinks: string[],
    ) {
        return options.concat(
            [this.filename(inputFilename)],
            libIncludes,
            libOptions,
            libPaths,
            libLinks,
            staticLibLinks,
            userOptions,
        );
    }
}
