// Copyright (c) 2018, Mitch Kennedy
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

const BaseCompiler = require('../base-compiler'),
    utils = require('../utils'),
    path = require("path");

class AdaCompiler extends BaseCompiler {
    constructor(info, env) {
        super(info, env);
        this.supportsOptOutput = false;
        this.compiler.supportsIntel = true;
    }

    optionsForFilter(filters, outputFilename) {
        let options = ['compile',
            '-g', // enable debugging
            '-c', // Compile only
            '-S', // Generate ASM 
            '-fdiagnostics-color=always',
            '-fverbose-asm', // Geneate verbose ASM showing variables
            '-cargs', // Compiler Switches for gcc.
            '-o', // Set the output executable name
            outputFilename
        ];
        if (this.compiler.intelAsm && filters.intel && !filters.binary) {
            options = options.concat(this.compiler.intelAsm.split(" "));
        }
        return options;
    }

    // I have left the overloaded preProcess method in case there is a 
    // need to any actual pre-processing of the input.
    // As Ada expects the outermost function name to match the source file name.
    // The initial solution was to wrap any input in a dummy example procedure,
    // this however restricts users from including standard library packages, as
    // Ada mandates that 'with' clauses are placed in the context clause,
    // which in the case of a single subprogram is outside of its declaration and body.
    preProcess(source) {
        return source;
    }

    async runCompiler(compiler, options, inputFilename, execOptions) {
        if (!execOptions) {
            execOptions = this.getDefaultExecOptions();
        }
        // Set the working directory to be the temp directory that has been created
        execOptions.customCwd = path.dirname(inputFilename);
        // As the file name is always appended to the end of the options array we need to 
        // find where the '-cargs' flag is in options. This is to allow us to set the 
        // output as 'output.s' and not end up with 'example.s'. If the output is left
        // as 'example.s' CE can't find it and thus you get no output.
        const inputFileName = options.pop();
        for (let i = 0; i < options.length; i++) {
            if (options[i] === '-cargs') {
                options.splice(i, 0, inputFileName);
                break;
            }
        }
        const result = await this.exec(compiler, options, execOptions);
        result.inputFilename = inputFilename;
        result.stdout = utils.parseOutput(result.stdout, inputFilename);
        result.stderr = utils.parseOutput(result.stderr, inputFilename);
        return result;
    }
}

module.exports = AdaCompiler;
