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

import {splitArguments} from '../../shared/common-utils.js';
import type {CompilationResult} from '../../types/compilation/compilation.interfaces.js';
import {LLVMIrBackendOptions} from '../../types/compilation/ir.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import type {ResultLine} from '../../types/resultline/resultline.interfaces.js';
import {unwrap} from '../assert.js';
import {FlangParser} from './argument-parsers.js';
import {FortranCompiler} from './fortran.js';

// `flang -fc1` accepts only one of these flags per run, so existing ones are removed before the parse tree
// dump is added.
const FC1_ACTIONS = new Set([
    '-E',
    '-S',
    '-emit-obj',
    '-emit-llvm',
    '-emit-llvm-bc',
    '-emit-fir',
    '-emit-hlfir',
    '-emit-mlir',
    '-fsyntax-only',
    '-fdebug-unparse',
    '-fdebug-unparse-no-sema',
    '-fdebug-unparse-with-symbols',
    '-fdebug-unparse-with-modules',
    '-fdebug-dump-symbols',
    '-fdebug-dump-parse-tree',
    '-fdebug-dump-parse-tree-no-sema',
    '-fdebug-dump-pft',
    '-fdebug-dump-all',
    '-fdebug-dump-provenance',
    '-fdebug-dump-parsing-log',
    '-fdebug-measure-parse-tree',
    '-fdebug-pre-fir-tree',
]);

// LLVM Flang's version string starts with `flang-new version` (up to 19), `flang version` (20 onward) or, in AOCC,
// `AMD aof version`. AOCC's classic Flang prints `AMD clang version` or `AMD flang version`, and has no `-fc1`.
export function isLlvmFlang(version: string): boolean {
    return /^(?:flang-new|flang|AMD aof) version /m.test(version);
}

// Rewrites `flang -fc1` arguments to print the parse tree, dropping the output file and any action flag.
export function parseTreeArguments(fc1Arguments: string[]): string[] {
    const result: string[] = [];
    for (let i = 0; i < fc1Arguments.length; i++) {
        if (fc1Arguments[i] === '-o') {
            i++;
        } else if (!FC1_ACTIONS.has(fc1Arguments[i])) {
            result.push(fc1Arguments[i]);
        }
    }
    return [...result, '-fdebug-dump-parse-tree'];
}

export function parseTreeOutput(result: CompilationResult): ResultLine[] {
    // Flang prints no tree when the source has syntax or semantic errors, so show its errors instead.
    if (result.code !== 0) return [{text: '<Flang could not produce a parse tree>'}, ...result.stderr];
    return result.stdout.filter(line => !/^=+ Flang: parse tree dump =+$/.test(line.text));
}

export class FlangCompiler extends FortranCompiler {
    static override get key() {
        return 'flang';
    }

    protected override getArgumentParserClass(): any {
        return FlangParser;
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string) {
        let options = ['-o', this.filename(outputFilename)];
        if (this.compiler.intelAsm && filters.intel && !filters.binary) {
            options = options.concat(this.compiler.intelAsm.split(' '));
        }
        if (filters.binary) {
            options = options.concat('-g');
        } else {
            options = options.concat('-S');
        }
        return options;
    }

    override async generateIR(
        inputFilename: string,
        options: string[],
        irOptions: LLVMIrBackendOptions,
        produceCfg: boolean,
        filters: ParseFiltersAndOutputOptions,
    ) {
        // -emit-llvm is passed directly to flang, rather than via -Xflang -emit-llvm,
        // which is the style used for clang. This means the output filename is not
        // derived from the input file name, it instead follows the -o option as a
        // normal compilation would. These options will end up with 2 -o, but the
        // last one in wins, which is the one that we add here.
        const newOptions = [
            ...options,
            ...unwrap(this.compiler.irArg),
            '-o',
            this.getIrOutputFilename(inputFilename, filters),
        ];
        return super.generateIR(inputFilename, newOptions, irOptions, produceCfg, filters);
    }

    override couldSupportASTDump(version: string) {
        return isLlvmFlang(version);
    }

    override async generateAST(inputFilename: string, options: string[]): Promise<ResultLine[]> {
        // The driver rejects `-fdebug-dump-parse-tree`; only the frontend (`flang -fc1`) accepts it. `-###` makes the
        // driver print the frontend command it would run, including the include and module paths and preprocessing
        // options. That command is then run with the dump as its action.
        const driver = await this.runCompilerRawOutput(
            this.compiler.exe,
            [...options, '-###'],
            this.filename(inputFilename),
            this.getDefaultExecOptions(),
        );
        const fc1Line = driver.code === 0 ? driver.stderr.split('\n').find(line => line.includes('"-fc1"')) : undefined;
        if (!fc1Line) return [{text: '<Flang did not report a frontend command, so there is no parse tree>'}];

        // The command starts with the path to flang itself. Keep only the arguments after it.
        const fc1Arguments = splitArguments(fc1Line).slice(1);
        const execOptions = this.getDefaultExecOptions();
        execOptions.maxOutput = 1024 * 1024 * 1024;
        return parseTreeOutput(
            await this.runCompiler(
                this.compiler.exe,
                parseTreeArguments(fc1Arguments),
                this.filename(inputFilename),
                execOptions,
            ),
        );
    }
}
