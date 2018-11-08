// Copyright (c) 2018, Microsoft Corporation
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
    temp = require('temp'),
    path = require('path'),
    PELabelReconstructor = require("../pe32-support").labelReconstructor;

class Win32Compiler extends BaseCompiler {
    newTempDir() {
        return new Promise((resolve, reject) => {
            temp.mkdir({prefix: 'compiler-explorer-compiler', dir: process.env.TMP}, (err, dirPath) => {
                if (err)
                    reject(`Unable to open temp file: ${err}`);
                else
                    resolve(dirPath);
            });
        });
    }

    getExecutableFilename(dirPath, outputFilebase) {
        return this.getOutputFilename(dirPath, outputFilebase) + ".exe";
    }

    objdump(outputFilename, result, maxSize, intelAsm) {
        outputFilename = this.getExecutableFilename(path.dirname(outputFilename), "output");

        let args = ["-d", outputFilename];
        if (intelAsm) args = args.concat(["-M", "intel"]);
        return this.exec(this.compiler.objdumper, args, {maxOutput: 0})
            .then((objResult) => {
                if (objResult.code !== 0) {
                    result.asm = "<No output: objdump returned " + objResult.code + ">";
                } else {
                    result.asm = objResult.stdout;
                }

                return result;
            });
    }

    optionsForFilter(filters, outputFilename) {
        if (filters.binary) {
            const mapFilename = outputFilename + '.map';

            filters.preProcessBinaryAsmLines = (asmLines) => {
                const reconstructor = new PELabelReconstructor(asmLines, mapFilename, false, "vs");
                reconstructor.run("output.s.obj");

                return reconstructor.asmLines;
            };

            return [
                '/nologo',
                '/FA',
                '/Fa' + this.filename(outputFilename),
                '/Fo' + this.filename(outputFilename + '.obj'),
                '/Fm' + this.filename(mapFilename),
                '/Fe' + this.filename(this.getExecutableFilename(path.dirname(outputFilename), "output"))
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

    exec(compiler, args, options_) {
        let options = Object.assign({}, options_);
        options.env = Object.assign({}, options.env);

        if (this.compiler.includePath) {
            options.env['INCLUDE'] = this.compiler.includePath;
        }
        if (this.compiler.libPath) {
            options.env['LIB'] = this.compiler.libPath;
        }
        for (const [env, to] of this.compiler.envVars) {
            options.env[env] = to;
        }

        return super.exec(compiler, args, options);
    }
}

module.exports = Win32Compiler;
