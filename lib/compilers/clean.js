// Copyright (c) 2019, Compiler Explorer Team
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

const
    BaseCompiler = require('../base-compiler'),
    fs = require('fs-extra'),
    path = require('path'),
    logger = require('../logger').logger;

class CleanCompiler extends BaseCompiler {
    optionsForFilter(filters, outputFilename) {
        if (filters.binary) {
			return [];
        } else {
            return ['-S'];
        }
    }

    getOutputFilename(dirPath, outputFilebase) {
        return path.join(dirPath, "Clean System Files/example.s");
    }

    objdump(outputFilename, result, maxSize, intelAsm, demangle) {
        logger.info(outputFilename);
        let args = ["-d", outputFilename, "-l", "--insn-width=16"];
        if (demangle) args = args.concat("-C");
        if (intelAsm) args = args.concat(["-M", "intel"]);
        return this.exec(this.compiler.objdumper, args, {maxOutput: maxSize})
            .then(objResult => {
                result.asm = objResult.stdout;
                if (objResult.code !== 0) {
                    result.asm = `<No output: objdump returned ${objResult.code}>`;
                }
                return result;
            });
    }

    runCompiler(compiler, options, inputFilename, execOptions) {
        const tmpDir = path.dirname(inputFilename);
        const moduleName = path.basename(inputFilename, '.icl');
        const compilerPath = path.dirname(compiler);
        execOptions = this.getDefaultExecOptions();
        execOptions.customCwd = tmpDir;
        execOptions.env.CLEANLIB = path.join(compilerPath, "../exe");
        execOptions.env.CLEANPATH = this.compiler.libPath;
        options.pop();
        options.push(moduleName);
        return super.runCompiler(compiler, options, moduleName, execOptions).then(result => {
            return new Promise((resolve, reject) => {
                if (!options.includes("-S")) {
                    const aout = path.join(tmpDir, "a.out");
                    fs.stat(aout).then(stats => {
                        fs.copyFile(aout, this.getOutputFilename(tmpDir)).then(() => {
                            result.code = 0;
                            resolve(result);
                        });
                    }).catch(err => {
                        result.code = 1;
                        resolve(result);
                    });
                } else {
                    fs.stat(this.getOutputFilename(tmpDir)).then(stats => {
                        result.code = 0;
                        resolve(result);
                    }).catch(err => {
                        resolve(result);
                    });
                }
            });
        });
    }
}

module.exports = CleanCompiler;
