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

import _ from 'underscore';

import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {unwrap} from '../assert.js';
import {BaseCompiler} from '../base-compiler.js';

export class SnowballCompiler extends BaseCompiler {
    linker: string;

    static get key() {
        return 'snowball';
    }

    constructor(info: PreliminaryCompilerInfo, env) {
        super(info, env);
        this.compiler.supportsIntel = false;
        this.compiler.supportsIrView = true;
        this.compiler.supportsLLVMOptPipelineView = true;
        this.compiler.supportsCfg = true;

        this.compiler.irArg = ['--emit', 'llvm-ir'];
        this.linker = this.compilerProps<string>('linker');
    }

    override getSharedLibraryPathsAsArguments(libraries, libDownloadPath) {
        return [];
    }

    override getSharedLibraryLinks(libraries: any[]): string[] {
        return [];
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
        return options.concat(userOptions, libIncludes, libOptions, libPaths, libLinks, staticLibLinks, [
            '--file',
            this.filename(inputFilename),
        ]);
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string, userOptions?: string[]) {
        let options = ['build', '--silent', '--output', this.filename(outputFilename)];

        const userRequestedEmit = _.any(unwrap(userOptions), opt => opt.includes('--emit'));
        if (filters.binary) {
            options = options.concat(['--emit', 'exec']);
        } else if (filters.binaryObject) {
            options = options.concat(['--emit', 'lib']);
        } else {
            if (!userRequestedEmit) {
                options = options.concat('--emit', 'asm');
            }
            // TODO:
            // if (filters.intel) options = options.concat('--llvm-args', '--x86-asm-syntax=intel');
        }
        return options;
    }

    override isCfgCompiler(/*compilerVersion*/) {
        return true;
    }
}
