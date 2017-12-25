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

var Compile = require('../base-compiler'),
    PascalDemangler = require('../pascal-support').demangler,
    utils = require('../utils'),
    fs = require("fs"),
    path = require("path");

function compileFPC(info, env) {
    var demangler = new PascalDemangler();
    var compiler = new Compile(info, env);
    compiler.compileFilename = "output.pas";
    compiler.supportsOptOutput = false;

    var originalExecBinary = compiler.execBinary;
    var currentlyActiveFilters = {};

    compiler.postProcessAsm = function (result) {
        if (!result.okToCache) return result;

        if (currentlyActiveFilters.binary) {
            preProcessAsm(result.asm);
        }

        for (var j = 0; j < result.asm.length; ++j)
            result.asm[j].text = demangler.demangleIfNeeded(result.asm[j].text);

        return result;
    };

    compiler.optionsForFilter = function (filters, outputFilename, userOptions) {
        var options = ['-g'];

        if (this.compiler.intelAsm && filters.intel && !filters.binary) {
            options = options.concat(this.compiler.intelAsm.split(" "));
        }

        currentlyActiveFilters = filters;
        filters.preProcessLines = preProcessLines;

        return options;
    };

    compiler.getOutputFilename = function (dirPath, outputFilebase) {
        return path.join(dirPath, path.basename(this.compileFilename, this.langInfo.extensions[0]) + ".s");
    };

    var saveDummyProjectFile = function (filename) {
        const unitName = path.basename(compiler.compileFilename, compiler.langInfo.extensions[0]);

        fs.writeFileSync(filename,
            "program prog; " +
            "uses " + unitName + " in '" + compiler.compileFilename + "'; " +
            "begin " +
            "end.", function() {});
    };

    var preProcessBinaryAsm = function (input) {
        var relevantAsmStartsAt = input.indexOf("<OUTPUT");
        if (relevantAsmStartsAt != -1) {
            var lastLinefeedBeforeStart = input.lastIndexOf("\n", relevantAsmStartsAt);
            if (lastLinefeedBeforeStart != -1) {
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
    };

    var getExtraAsmHint = function (asm) {
        if (asm.startsWith("# [")) {
            var bracketEndPos = asm.indexOf("]", 3);
            var valueInBrackets = asm.substr(3, bracketEndPos - 3);
            var colonPos = valueInBrackets.indexOf(":");
            if (colonPos != -1) {
                valueInBrackets = valueInBrackets.substr(0, colonPos - 1);
            }

            if (!isNaN(valueInBrackets)) {
                return "  .loc 1 " + valueInBrackets + " 0";
            } else if (valueInBrackets.includes(compiler.compileFilename)) {
                return "  .file 1 \"<stdin>\"";
            } else {
                return false;
            }
        } else if (asm.startsWith(".Le")) {
            return "  .cfi_endproc";
        } else {
            return false;
        }
    };

    var preProcessLines = function(asmLines) {
        var i = 0;

        while (i < asmLines.length) {
            var extraHint = getExtraAsmHint(asmLines[i]);
            if (extraHint) {
                i++;
                asmLines.splice(i, 0, extraHint);
            } else {
                demangler.addDemangleToCache(asmLines[i]);
            }

            i++;
        }

        return asmLines;
    };

    var preProcessAsm = function(asm) {
        for (var j = 0; j < asm.length; ++j) demangler.addDemangleToCache(asm[j].text);
    };

    compiler.objdump = function (outputFilename, result, maxSize, intelAsm, demangle) {
         outputFilename = path.join(path.dirname(outputFilename), "prog");

         var args = ["-d", outputFilename, "-l", "--insn-width=16"];
         if (demangle) args = args.concat(["-C"]);
         if (intelAsm) args = args.concat(["-M", "intel"]);
         return this.exec(this.compiler.objdumper, args, {maxOutput: maxSize})
            .then(function (objResult) {
                if (objResult.code !== 0) {
                    result.asm = "<No output: objdump returned " + objResult.code + ">";
                } else {
                    result.asm = preProcessBinaryAsm(objResult.stdout);
                }

                return result;
            });
    };

    compiler.runCompiler = function (compiler, options, inputFilename, execOptions) {
        if (!execOptions) {
            execOptions = this.getDefaultExecOptions();
        }

        var tempPath = path.dirname(inputFilename);
        var projectFile = path.join(tempPath, "prog.dpr");

        saveDummyProjectFile(projectFile);

        options.pop();
        options.push('-FE' + tempPath);
        options.push('-B');
        options.push(projectFile);

        return this.exec(compiler, options, execOptions).then(function (result) {
            result.inputFilename = inputFilename;
            result.stdout = utils.parseOutput(result.stdout, inputFilename);
            result.stderr = utils.parseOutput(result.stderr, inputFilename);
            return result;
        });
    };

    compiler.execBinary = function (executable, result, maxSize) {
        executable = path.join(path.dirname(executable), "prog");

        originalExecBinary(executable, result, maxSize);
    };

    if (info.unitTestMode) {
        compiler.initialise();
        return compiler;
    } else
        return compiler.initialise();
}

module.exports = compileFPC;
