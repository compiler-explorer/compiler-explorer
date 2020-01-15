// Copyright (c) 2018, Patrick Quist
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
"use strict";

const BaseCompiler = require('../base-compiler'),
    AsmRaw = require('../asm-raw').AsmParser,
    utils = require('../utils'),
    fs = require("fs"),
    path = require("path"),
    argumentParsers = require("./argument-parsers");

class AssemblyCompiler extends BaseCompiler {
    constructor(info, env) {
        super(info, env);
        this.asm = new AsmRaw();
    }

    getSharedLibraryPathsAsArguments() {
        return [];
    }

    getArgumentParser() {
        return argumentParsers.Base;
    }

    optionsForFilter(filters) {
        filters.binary = true;
        return [];
    }

    async runCompiler(compiler, options, inputFilename, execOptions) {
        if (!execOptions) {
            execOptions = this.getDefaultExecOptions();
        }
        execOptions.customCwd = path.dirname(inputFilename);

        const result = await this.exec(compiler, options, execOptions);
        result.inputFilename = inputFilename;
        result.stdout = utils.parseOutput(result.stdout, inputFilename);
        result.stderr = utils.parseOutput(result.stderr, inputFilename);
        return result;
    }

    checkOutputFileAndDoPostProcess(asmResult, outputFilename, filters) {
        return this.postProcess(asmResult, outputFilename, filters);
    }

    getGeneratedOutputFilename(inputFilename) {
        const outputFolder = path.dirname(inputFilename);

        return new Promise((resolve, reject) => {
            fs.readdir(outputFolder, (err, files) => {
                files.forEach(file => {
                    if (file[0] !== '.' && file !== this.compileFilename) {
                        resolve(path.join(outputFolder, file));
                    }
                });
                reject("No output file was generated");
            });
        });
    }

    async objdump(outputFilename, result, maxSize, intelAsm, demangle) {
        const realOutputFilename = await this.getGeneratedOutputFilename(outputFilename);
        const dirPath = path.dirname(realOutputFilename);
        let args = ["-d", realOutputFilename, "-l", "--insn-width=16"];
        if (demangle) args = args.concat("-C");
        if (intelAsm) args = args.concat(["-M", "intel"]);
        const objResult = await this.exec(
            this.compiler.objdumper, args, {maxOutput: maxSize, customCwd: dirPath});
        result.asm = objResult.stdout;
        if (objResult.code !== 0) {
            result.asm = "<No output: objdump returned " + objResult.code + ">";
        }
        return result;
    }
}

module.exports = AssemblyCompiler;
