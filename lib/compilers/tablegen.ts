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

import {CompilerOverrideType} from '../../types/compilation/compiler-overrides.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';
import {TableGenParser} from './argument-parsers.js';

export class TableGenCompiler extends BaseCompiler {
    static get key() {
        return 'tablegen';
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string) {
        const options: string[] = ['-o', outputFilename];
        if (this.compiler.includePath) {
            options.push(`-I${this.compiler.includePath}`);
        }
        return options;
    }

    override isCfgCompiler() {
        return false;
    }

    override getArgumentParserClass() {
        return TableGenParser;
    }

    override async populatePossibleOverrides() {
        const possibleActions = await this.argParser.getPossibleActions();
        if (possibleActions.length > 0) {
            this.compiler.possibleOverrides?.push({
                type: 'options',
                name: CompilerOverrideType.action,
                display_title: 'Action',
                description:
                    'The action to perform, which is the backend you wish to ' +
                    'run. By default, the records are just printed as text. ' +
                    'Many backends expect to find certain classes and defnitions ' +
                    'in your source code. You may find details of those in the ' +
                    '<a href="https://llvm.org/docs/TableGen/BackEnds.html" target="_blank">documentation</a>, ' +
                    'but if not, refer to use of the backend in the ' +
                    '<a href="https://github.com/llvm/llvm-project" target="_blank">LLVM Project</a> ' +
                    'by searching for the command line name e.g. "gen-attrs".',
                flags: ['<value>'],
                values: possibleActions,
                default: '--print-records',
            });
        }

        await super.populatePossibleOverrides();
    }
}
