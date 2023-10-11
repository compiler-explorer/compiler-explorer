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

import Semver from 'semver';

import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';
import {DartAsmParser} from '../parsers/asm-parser-dart.js';
import * as utils from '../utils.js';

import {BaseParser} from './argument-parsers.js';
import type {ConfiguredOverrides} from '../../types/compilation/compiler-overrides.interfaces.js';

export class DartCompiler extends BaseCompiler {
    constructor(info: PreliminaryCompilerInfo, env) {
        super(info, env);
        this.asm = new DartAsmParser();
    }

    static get key() {
        return 'dart';
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
        let options = this.optionsForFilter(filters, outputFilename, userOptions);

        if (this.compiler.options) {
            options = options.concat(utils.splitArguments(this.compiler.options));
        }

        const libIncludes = this.getIncludeArguments(libraries);
        const libOptions = this.getLibraryOptions(libraries);

        userOptions = this.filterUserOptions(userOptions) || [];
        return options.concat(libIncludes, libOptions, userOptions, [this.filename(inputFilename)]);
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string, userOptions?: string[]) {
        // Dart includes way too much of the standard library (even for simple programs)
        // to show all of it without truncation
        filters.libraryCode = true;
        // Dart doesn't support emitting assembly
        filters.binary = true;

        const dartCompileIntroduction = '2.10.0';
        if (Semver.lt(utils.asSafeVer(this.compiler.semver), dartCompileIntroduction, true)) {
            return ['-k', 'aot', '-o', this.filename(outputFilename)];
        } else {
            return ['compile', 'aot-snapshot', '-o', this.filename(outputFilename)];
        }
    }

    override getArgumentParser() {
        return BaseParser;
    }
}
