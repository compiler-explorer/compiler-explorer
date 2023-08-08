// Copyright (c) 2018, Filipe Cabecinhas
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

import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';

import {AnalysisTool} from './analysis-tool.js';
import {ClangParser} from './argument-parsers.js';

// Plain compiler, which just runs the tool and returns whatever the output was
export class LLVMmcaTool extends AnalysisTool {
    static override get key() {
        return 'llvm-mca';
    }

    override supportsObjdump() {
        return false;
    }

    getOutputFilenameArgs(filename: string) {
        // TODO: Some tools might require an argument of the form -arg=filename
        return ['-o', filename];
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string) {
        let options = this.getOutputFilenameArgs(outputFilename);
        // Some tools might require more than one arg, so split(' ')
        if (filters.intel) options = options.concat(this.compiler.intelAsm.split(' '));
        return options;
    }

    override getArgumentParser() {
        return ClangParser;
    }
}
