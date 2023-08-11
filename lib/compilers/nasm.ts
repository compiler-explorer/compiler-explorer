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

import type {ConfiguredOverrides} from '../../types/compilation/compiler-overrides.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';

import {AssemblyCompiler} from './assembly.js';

export class NasmCompiler extends AssemblyCompiler {
    static override get key() {
        return 'nasm';
    }

    override prepareArguments(
        userOptions: string[],
        filters: ParseFiltersAndOutputOptions,
        backendOptions: Record<string, any>,
        inputFilename: string,
        outputFilename: string,
        libraries,
        overrides: ConfiguredOverrides,
    ) {
        let options = super.prepareArguments(
            userOptions,
            filters,
            backendOptions,
            inputFilename,
            outputFilename,
            libraries,
            overrides,
        );

        let fmode;
        if (options.includes('-felf')) {
            fmode = 'elf';
        } else if (options.includes('-felf64')) {
            fmode = 'elf64';
        } else if (options.includes('-fbin')) {
            fmode = 'bin';
        } else if (options.includes('-f')) {
            const idx = options.indexOf('-f');
            fmode = options[idx + 1];
        }

        if (!fmode) {
            options = ['-g', '-f', 'elf', '-F', 'stabs'].concat(options);
        }

        return options;
    }
}
