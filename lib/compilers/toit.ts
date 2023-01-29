// Copyright (c) 2022, Serzhan Nasredin
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

import {CompilerInfo} from '../../types/compiler.interfaces';
import {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces';
import {BaseCompiler} from '../base-compiler';

import {ToitParser} from './argument-parsers';

export class ToitCompiler extends BaseCompiler {
    static get key() {
        return 'toit';
    }

    constructor(info: CompilerInfo, env) {
        super(info, env);
        this.compiler.supportsIntel = true;
    }

    cacheDir(outputFilename: string) {
        return outputFilename + '.cache';
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename, userOptions?): string[] {
        if (!filters.binary) return ['execute', outputFilename];
        return [outputFilename];
    }

    override getSharedLibraryPathsAsArguments(libraries: object[], libDownloadPath: string) {
        return [];
    }
    override getArgumentParser() {
        return ToitParser;
    }
    override isCfgCompiler() {
        return true;
    }
}
