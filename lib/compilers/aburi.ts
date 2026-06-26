// Copyright (c) 2025, Compiler Explorer Authors
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
import * as utils from '../utils.js';

import type {LLVMIrBackendOptions} from '../../types/llvm-ir/backend.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {unwrap} from '../assert.js';
import {BaseCompiler} from '../base-compiler.js';

export class AburiCompiler extends BaseCompiler {
    static override get key() {
        return 'aburi';
    }

    constructor(info, env) {
        super(info, env);
        this.compiler.supportsIrView = true;
        this.compiler.irArg = ['--emit-llvm'];
    }

    override optionsForFilter(
        filters: ParseFiltersAndOutputOptions,
        outputFilename: string,
        userOptions?: string[],
    ) {
        if (_.some(unwrap(userOptions), opt => opt === '--help' || opt === '-h' || opt === '-hh')) {
            return [];
        }
        const options = ['-g', '-o', this.filename(outputFilename)];
        if (!filters.binary) {
            options.unshift('-S');
        }
        return options;
    }

    override filterUserOptions(userOptions: string[]) {
        return userOptions.filter(opt => opt !== '-run');
    }

    override getIrOutputFilename(inputFilename: string): string {
        return utils.changeExtension(inputFilename, '.ll');
    }

    override generateIR(
        inputFilename: string,
        options: string[],
        irOptions: LLVMIrBackendOptions,
        produceCfg: boolean,
        filters: ParseFiltersAndOutputOptions,
    ) {
        const irPath = this.getIrOutputFilename(inputFilename);
        const irOptions_ = options
            .filter(opt => opt !== '-S')
            .map((opt) => {
                if (opt === '-o') return '__OFLAG__';
                return opt;
            });
        const oIdx = irOptions_.indexOf('__OFLAG__');
        if (oIdx !== -1 && oIdx + 1 < irOptions_.length) {
            irOptions_[oIdx] = '-o';
            irOptions_[oIdx + 1] = irPath;
        }
        return super.generateIR(inputFilename, irOptions_, irOptions, produceCfg, filters);
    }
}
