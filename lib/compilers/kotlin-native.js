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

import path from 'path';

import { BaseCompiler } from '../base-compiler';

import { KotlinNativeParser } from './argument-parsers';

export class KotlinNativeCompiler extends BaseCompiler {
    static get key() {
        return 'kotlin-native';
    }

    constructor(compilerInfo, env) {
        super(compilerInfo, env);
        this.llvmExtract = this.compilerProps('llvmExtract');
        this.llvmCompiler = this.compilerProps('llvmCompiler');

        this.compiler.supportsIrView = this.llvmExtract !== undefined;
    }

    getDefaultExecOptions() {
        // TODO: add JAVA_HOME
        return super.getDefaultExecOptions();
    }

    getOutputFilename(dirPath) {
        return path.join(dirPath, 'program.bc');
    }

    getExecutableFilename(dirPath) {
        return path.join(dirPath, 'program.kexe');
    }

    getObjdumpOutputFilename(defaultOutputFilename) {
        return path.join(path.dirname(defaultOutputFilename), 'output.o');
    }

    getIrOutputFilename(inputFilename) {
        return path.join(path.dirname(inputFilename), 'output.ll');
    }

    async objdump(outputFilename, result, maxSize, intelAsm, demangle) {
        const outputDirectory = path.dirname(outputFilename);
        const inputFile = path.resolve(outputDirectory, './program.bc');
        const objectFile = path.resolve(outputDirectory, './output.o');
        const bitcodeFile = path.resolve(outputDirectory, './output.bc');
        const execOptions = this.getDefaultExecOptions();
        // First we extract the truncated bitcode
        const extractArgs = [inputFile, '-rfunc', 'kfun:#.+', '-S', '-o', bitcodeFile];
        const extractProcess = await this.exec(this.llvmExtract, extractArgs, execOptions);
        result.code = extractProcess.code;
        if (extractProcess.code !== 0) {
            result.asm = `<No output: ${this.llvmExtract} returned ${extractProcess.code}>`;
            result.objdumpTime = extractProcess.execTime;
            return result;
        }
        // Then we compile into shared object
        const llcArgs = [bitcodeFile, '-filetype', 'obj', '-o', objectFile];
        const llcProcess = await this.exec(this.llvmCompiler, llcArgs, execOptions);
        result.code = llcProcess.code;
        if (llcProcess.code !== 0) {
            result.asm = `<No output: ${this.llvmCompiler} returned ${llcProcess.code}>`;
            result.objdumpTime = extractProcess.execTime + llcProcess.execTime;
            return result;
        }

        return super.objdump(objectFile, result, maxSize, intelAsm, demangle);

        //
        // const extractProcess = await this.exec(this.compiler.objdumper, extractArgs, execOptions);
        // result.asm = extractProcess.stdout;
        // result.code = extractProcess.code;
        // result.objdumpTime = extractProcess.execTime;
        //
        // if (extractProcess.code !== 0) {
        //     result.asm = `<No output: ${this.llvmExtract} returned ${extractProcess.code}>`;
        // } else {
        //     result.asm = await fs.readFile(outputFile, 'utf8');
        // }
        // return result;
    }

    // processAsm(result, filters) {
    //     if (!result.asm.includes('\n') && result.asm[0] === '<') {
    //         return [{text: result.asm, source: null}];
    //     }
    //     return this.llvmIr.process(result.asm, filters);
    // }

    async generateIR(inputFilename, _options, filters) {
        const outputDirectory = path.dirname(inputFilename);
        const outputFile = path.resolve(outputDirectory, './output.ll');
        const inputFile = path.resolve(outputDirectory, './program.bc');
        const execOptions = this.getDefaultExecOptions();
        // First we run kotlinc-native
        const kotlincArgs = [inputFilename, '-p', 'bitcode', '-g', '-o', 'program.bc', '-nopack'];
        const kotlincProcess = await this.runCompiler(this.compiler.exe, kotlincArgs, inputFilename, execOptions);
        if (kotlincProcess.code !== 0) {
            return [{text: 'Failed to run kotlinc-native to get LLVM IR'}];
        }
        // Then we run llvm-extract to produce LLVM IR
        const extractArgs = [inputFile, '-rfunc', 'kfun:#.+', '-S', '-o', outputFile];
        const extractProcess = await this.runCompiler(this.llvmExtract, extractArgs, inputFile, execOptions);
        if (extractProcess.code !== 0) {
            return [{text: 'Failed to run llvm-extract to get LLVM IR'}];
        }
        const ir = await this.processIrOutput(extractProcess, filters);
        return ir.asm;
    }

    getSharedLibraryPathsAsArguments() {
        return [];
    }

    getArgumentParser() {
        return KotlinNativeParser;
    }

    optionsForFilter(filters) {
        filters.binary = true;
        if (filters.execute) {
            return ['-p', 'program', '-g', '-o', 'program.kexe', '-nopack'];
        }
        return ['-p', 'bitcode', '-g', '-o', 'program.bc', '-nopack'];
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
}
