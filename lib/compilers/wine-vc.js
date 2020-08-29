// Copyright (c) 2016, Matt Godbolt
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
    AsmParser = require('../asm-parser-vc'),
    argumentParsers = require('./argument-parsers'),
    path = require('path'),
    PELabelReconstructor = require('../pe32-support').labelReconstructor;

class WineVcCompiler extends BaseCompiler {
    constructor(info, env) {
        info.supportsFiltersInBinary = true;
        super(info, env);
        this.asm = new AsmParser();
    }

    filename(fn) {
        return 'Z:' + fn;
    }

    runCompiler(compiler, options, inputFilename, execOptions) {
        if (!execOptions) {
            execOptions = this.getDefaultExecOptions();
        }

        execOptions.customCwd = path.dirname(inputFilename).substr(2);

        return super.runCompiler(compiler, options, inputFilename, execOptions);
    }

    getArgumentParser() {
        return argumentParsers.VC;
    }

    getExecutableFilename(dirPath, outputFilebase) {
        return this.getOutputFilename(dirPath, outputFilebase) + '.exe';
    }

    async objdump(outputFilename, result, maxSize, intelAsm) {
        const dirPath = path.dirname(outputFilename);
        outputFilename = this.getExecutableFilename(dirPath, 'output');

        let args = ['-d', outputFilename];
        if (intelAsm) args = args.concat(['-M', 'intel']);
        const objResult = await this.exec(this.compiler.objdumper, args, {maxOutput: 0, customCwd: dirPath});
        if (objResult.code !== 0) {
            result.asm = '<No output: objdump returned ' + objResult.code + '>';
        } else {
            result.asm = objResult.stdout;
        }

        return result;
    }

    optionsForFilter(filters, outputFilename) {
        if (filters.binary) {
            const mapFilename = outputFilename + '.map';

            filters.preProcessBinaryAsmLines = (asmLines) => {
                const reconstructor = new PELabelReconstructor(asmLines, mapFilename, false, 'vs');
                reconstructor.run('output.s.obj');

                return reconstructor.asmLines;
            };

            return [
                '/nologo',
                '/FA',
                '/Fa' + this.filename(outputFilename),
                '/Fo' + this.filename(outputFilename + '.obj'),
                '/Fm' + this.filename(mapFilename),
                '/Fe' + this.filename(this.getExecutableFilename(path.dirname(outputFilename), 'output')),
            ];
        } else {
            return [
                '/nologo',
                '/FA',
                '/c',
                '/Fa' + this.filename(outputFilename),
                '/Fo' + this.filename(outputFilename + '.obj'),
            ];
        }
    }
}

module.exports = WineVcCompiler;
