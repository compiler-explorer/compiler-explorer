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

import fs from 'fs';
import path from 'path';

import { BaseCompiler } from '../base-compiler';

import { KotlinNativeParser } from './argument-parsers';

export class KotlinNativeCompiler extends BaseCompiler {
    static get key() {
        return 'kotlin-native';
    }

    constructor(compilerInfo, env) {
        if (!compilerInfo.disabledFilters) {
            compilerInfo.disabledFilters = ['labels', 'directives', 'commentOnly', 'trim'];
        }
        super(compilerInfo, env);
    }

    optionsForFilter(filters) {
        // Force CE to disassemble bitcode file
        filters.binary = true;
        return ['-p', 'bitcode', '-nopack'];
    }

    filterUserOptions(userOptions) {
        const filteredOptions = [];
        let toSkip = 0;

        const forbiddenFlags = new Set([
            '-nomain', '-nopack', '-progressive', '-script',
        ]);
        const oneArgForbiddenList = new Set([
            '-o', '--output',
            '-p', '--produce',
            '-kotlin-home',
        ]);

        for (const userOption of userOptions) {
            if (toSkip > 0) {
                toSkip--;
                continue;
            }
            if (forbiddenFlags.has(userOption)) {
                continue;
            }
            if (oneArgForbiddenList.has(userOption)) {
                toSkip = 1;
                continue;
            }

            filteredOptions.push(userOption);
        }

        return filteredOptions;
    }

    getOutputFilename(dirPath) {
        return path.join(dirPath, 'bitcode.bc');
    }

    async objdump(outputFilename) {
        const outputDirectory = path.dirname(outputFilename);
        const bitcodeFile = path.resolve(outputDirectory, './bitcode.bc');
        const llvmIrFile = path.resolve(outputDirectory, './bitcode.ll');
        // The kotlinc-native compiler statically links the stdlib LLVM module which is 200kloc IR
        // we are only interested in the user defined functions. We can llvm-extract to get the user defined ones
        const args = [
            // Input file produced by kotlinc-native
            bitcodeFile,
            // Extract all functions which match the following regexp:
            '-rfunc', 'kfun:#.+',
            // Output readable LLVM IR instead of bitcode
            '-S',
            // Decide output file
            '-o', llvmIrFile,
        ];
        const process = await this.exec(this.compiler.objdumper, args, {
            customCwd: outputDirectory,
        });
        const result = {
            asm: process.stdout,
            dirPath: outputDirectory,
        };

        if (process.code !== 0) {
            result.asm = `<No output: ${this.compiler.objdumper} returned ${process.code}>`;
        } else {
            result.asm = await fs.promises.readFile(llvmIrFile, 'utf8');
            result.objdumpTime = process.execTime;
        }

        return result;
    }

    getSharedLibraryPathsAsArguments() {
        return [];
    }

    getArgumentParser() {
        return KotlinNativeParser;
    }
}
