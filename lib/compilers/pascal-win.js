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

var Compile = require('../base-compiler');
var asm = require('../asm-cl');
var path = require('path');
var utils = require('../utils');
var fs = require('fs');
var logger = require('../logger').logger;
var PascalDemangler = require('../pascal-support').demangler,
    PascalLabelReconstructor = require('../pe32-support').labelReconstructor;

function compilePascalWin32(info, env) {
    var compile = new Compile(info, env);

    info.supportsFiltersInBinary = true;
    if (process.platform == "linux") {
        var wine = env.gccProps("wine");
        var origExec = compile.exec;
        compile.exec = function (command, args, options) {
            if (command.toLowerCase().endsWith(".exe")) {
                args.unshift(command);
                command = wine;
            }
            return origExec(command, args, options);
        };
        compile.filename = function (fn) {
            return 'Z:' + fn;
        };
    }

    var currentlyActiveFilters = [];
    var mapFilename = false;

    compile.objdump = function (outputFilename, result, maxSize, intelAsm, demangle) {
        outputFilename = path.join(path.dirname(outputFilename), "prog.exe");

        var args = ["-d", outputFilename];
        if (intelAsm) args = args.concat(["-M", "intel"]);
        return this.exec(this.compiler.objdumper, args, {maxOutput: maxSize})
           .then(function (objResult) {
               if (objResult.code !== 0) {
                   result.asm = "<No output: objdump returned " + objResult.code + ">";
               } else {
                   result.asm = objResult.stdout;
               }

               return result;
           });
    };

    var saveDummyProjectFile = function (dprfile, sourcefile) {
        if (dprfile.startsWith("Z:")) {
            dprfile = dprfile.substr(2);
        }

        fs.writeFile(dprfile,
            "program prog; " +
            "uses output in '" + sourcefile + "'; " +
            "begin " +
            "end.");
    };

    compile.runCompiler = function (compiler, options, inputFilename, execOptions) {
        if (!execOptions) {
            execOptions = this.getDefaultExecOptions();
        }

        var tempPath = path.dirname(inputFilename);
        var projectFile = path.join(tempPath, "prog.dpr");

        mapFilename = path.join(tempPath.substr(2), "prog.map");

        inputFilename = inputFilename.replace(/\//g, '\\');
        saveDummyProjectFile(projectFile, inputFilename);

        options.pop();
        options.push('-CC');
        options.push('-W');
        options.push('-H');
        options.push('-GD');
        options.push('-$D+');
        options.push('-V');
        options.push('-B');
        options.push(projectFile.replace(/\//g, '\\'));

        return this.exec(compiler, options, execOptions).then(function (result) {
            result.inputFilename = inputFilename;
            result.stdout = utils.parseOutput(result.stdout, inputFilename);
            result.stderr = utils.parseOutput(result.stderr, inputFilename);
            return result;
        });
    };

    var preProcessBinaryAsmLines = function(asmLines) {
        var reconstructor = new PascalLabelReconstructor(asmLines, mapFilename, false);
        reconstructor.Run();

        return reconstructor.asmLines;
    };

    compile.optionsForFilter = function (filters, outputFilename, userOptions) {
        currentlyActiveFilters = filters;
        filters.preProcessBinaryAsmLines = preProcessBinaryAsmLines;

        return [];
    };
    return compile.initialise();
}

module.exports = compilePascalWin32;
