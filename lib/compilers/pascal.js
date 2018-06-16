// Copyright (c) 2017, Patrick Quist
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
    utils = require('../utils'),
    _ = require('underscore'),
    fs = require("fs"),
    path = require("path"),
    argumentParsers = require("./argument-parsers");

class FPCCompiler extends BaseCompiler {
    constructor(info, env) {
        super(info, env);

        let demanglerClassFile = this.compiler.demanglerClassFile;
        if (!demanglerClassFile) demanglerClassFile = "../pascal-support";
        const demanglerClass = require(demanglerClassFile).Demangler;

        this.demangler = new demanglerClass(null, null, this);
        this.compileFilename = 'output.pas';
        this.supportsOptOutput = false;
    }

    postProcessAsm(result, filters) {
        if (!result.okToCache) return result;

        if (filters.binary) {
            for (let j = 0; j < result.asm.length; ++j) {
                this.demangler.addDemangleToCache(result.asm[j].text);
            }
        }

        for (let j = 0; j < result.asm.length; ++j)
            result.asm[j].text = this.demangler.demangleIfNeeded(result.asm[j].text);

        return result;
    }

    optionsForFilter(filters) {
        let options = ['-g', '-al'];

        if (this.compiler.intelAsm && filters.intel && !filters.binary) {
            options = options.concat(this.compiler.intelAsm.split(" "));
        }

        filters.preProcessLines = _.bind(this.preProcessLines, this);

        return options;
    }

    getOutputFilename(dirPath) {
        return path.join(dirPath, `${path.basename(this.compileFilename, this.lang.extensions[0])}.s`);
    }

    static preProcessBinaryAsm(input) {
        const relevantAsmStartsAt = input.indexOf("<OUTPUT");
        if (relevantAsmStartsAt !== -1) {
            const lastLinefeedBeforeStart = input.lastIndexOf("\n", relevantAsmStartsAt);
            if (lastLinefeedBeforeStart !== -1) {
                input =
                    input.substr(0, input.indexOf("00000000004")) + "\n" +
                    input.substr(lastLinefeedBeforeStart + 1);
            } else {
                input =
                    input.substr(0, input.indexOf("00000000004")) + "\n" +
                    input.substr(relevantAsmStartsAt);
            }
        }
        return input;
    }

    objdump(outputFilename, result, maxSize, intelAsm, demangle) {
        outputFilename = path.join(path.dirname(outputFilename), "prog");
        let args = ["-d", outputFilename, "-l", "--insn-width=16"];
        if (demangle) args = args.concat(["-C"]);
        if (intelAsm) args = args.concat(["-M", "intel"]);
        return this.exec(this.compiler.objdumper, args, {maxOutput: maxSize}).then(objResult => {
            if (objResult.code !== 0) {
                result.asm = "<No output: objdump returned " + objResult.code + ">";
            } else {
                result.asm = FPCCompiler.preProcessBinaryAsm(objResult.stdout);
            }
            return result;
        });
    }

    saveDummyProjectFile(filename) {
        const unitName = path.basename(this.compileFilename, this.lang.extensions[0]);

        fs.writeFileSync(filename,
            "program prog; " +
            "uses " + unitName + " in '" + this.compileFilename + "'; " +
            "begin " +
            "end.", () => {
            });
    }

    runCompiler(compiler, options, inputFilename, execOptions) {
        if (!execOptions) {
            execOptions = this.getDefaultExecOptions();
        }

        const tempPath = path.dirname(inputFilename);
        const projectFile = path.join(tempPath, "prog.dpr");

        this.saveDummyProjectFile(projectFile);

        options.pop();
        options.push('-FE' + tempPath);
        options.push('-B');
        options.push(projectFile);

        return this.exec(compiler, options, execOptions).then(result => {
            return this.parseOutput(result, inputFilename, tempPath);
        });
    }

    parseOutput(result, inputFilename, tempPath) {
        const fileWithoutPath = path.basename(inputFilename);
        result.inputFilename = fileWithoutPath;
        result.stdout = utils.parseOutput(result.stdout, fileWithoutPath, tempPath);
        result.stderr = utils.parseOutput(result.stderr, fileWithoutPath, tempPath);
        return result;
    }

    execBinary(executable, result, maxSize) {
        executable = path.join(path.dirname(executable), "prog");

        super.execBinary(executable, result, maxSize);
    }

    getArgumentParser() {
        return argumentParsers.Base;
    }

    getExtraAsmHint(asm) {
        if (asm.startsWith("# [")) {
            const bracketEndPos = asm.indexOf("]", 3);
            let valueInBrackets = asm.substr(3, bracketEndPos - 3);
            const colonPos = valueInBrackets.indexOf(":");
            if (colonPos !== -1) {
                valueInBrackets = valueInBrackets.substr(0, colonPos - 1);
            }

            if (!isNaN(valueInBrackets)) {
                return "  .loc 1 " + valueInBrackets + " 0";
            } else if (valueInBrackets.includes(this.compileFilename)) {
                return "  .file 1 \"<stdin>\"";
            } else {
                return false;
            }
        } else if (asm.startsWith(".Le")) {
            return "  .cfi_endproc";
        } else {
            return false;
        }
    }

    preProcessLines(asmLines) {
        let i = 0;

        while (i < asmLines.length) {
            const extraHint = this.getExtraAsmHint(asmLines[i]);
            if (extraHint) {
                i++;
                asmLines.splice(i, 0, extraHint);
            } else {
                this.demangler.addDemangleToCache(asmLines[i]);
            }

            i++;
        }

        return asmLines;
    }
}

module.exports = FPCCompiler;
