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

var Compile = require('../base-compiler'),
    logger = require('../logger').logger,
    utils = require('../utils'),
    fs = require("fs"),
    path = require("path");

function compileFPC(info, env) {
    var compiler = new Compile(info, env);
    compiler.supportsOptOutput = false;

    var originalExecBinary = compiler.execBinary;

    var symbolcache = {};
    var sortedsymbolcache = [];
    var fixedsymbols = {};
    var ignoredsymbols = [];

    fixedsymbols.OUTPUT_$$_init = 'unit_initialization';
    fixedsymbols.OUTPUT_$$_finalize = 'unit_finalization';
    fixedsymbols.OUTPUT_$$_init_implicit = 'unit_initialization_implicit';
    fixedsymbols.OUTPUT_$$_finalize_implicit ='unit_finalization_implicit';

    ignoredsymbols = [
        ".L", "VMT_$", "INIT_$", "INIT$_$", "FINALIZE$_$", "RTTI_$", "_$",
        "DEBUGSTART_$", "DEBUGEND_$", "DBG_$", "DBG2_$", "DBGREF_$"];

    var addDemangleToCache = function(text) {
        if (text.endsWith(':')) {
            if (shouldIgnoreSymbol(text)) {
                return;
            }

            for (var k in fixedsymbols) {
                if (text.startsWith(k)) {
                    text = text.replace(k, fixedsymbols[k]);
                    symbolcache[k] = fixedsymbols[k];
                    return;
                }
            }

            text = text.substr(0, text.length - 1);

            if (text.startsWith("U_$OUTPUT_$$_")) {
                var unmangledglobalvar = text.substr(13).toLowerCase();
                symbolcache[text] = unmangledglobalvar;
                return;
            }

            var idx, paramtype = "", signature = "", phase = 0;
            var unitname = "", classname = "", methodname = "", params = "", resulttype = "";

            idx = text.indexOf("$_$");
            if (idx != -1) {
                unitname = text.substr(0, idx - 1);
                classname = text.substr(idx + 3, text.indexOf("_$_", idx + 2) - idx - 3);
            }

            signature = "";
            idx = text.indexOf("_$$_");
            if (idx != -1) {
                if (unitname == "") unitname = text.substr(0, idx - 1);
                signature =  text.substr(idx + 3);
            }

            if (unitname == "") {
                idx = text.indexOf("OUTPUT_");
                if (idx != -1) {
                    unitname = "OUTPUT";

                    idx = text.indexOf("_$__");
                    if (idx != -1) {
                        classname = text.substr(7, idx - 7);
                        signature = text.substr(idx + 3);
                    } else {
                        signature = text.substr(6);
                    }
                }
            }

            if (signature != "") {
                for (idx = 1; idx < signature.length; idx++) {
                    if (signature[idx] == '$') {
                        if (phase == 0) phase = 1;
                        else if (phase == 1) {
                            if (paramtype == "") phase = 2;
                            else if (params != "") {
                                params = params + "," + paramtype;
                                paramtype = "";
                            } else if (params == "") {
                                params = paramtype;
                                paramtype = "";
                            }
                        }
                    } else {
                        if (phase == 0) methodname = methodname + signature[idx];
                        else if (phase == 1) paramtype = paramtype + signature[idx];
                        else if (phase == 2) resulttype = resulttype + signature[idx];
                    }
                }

                if (paramtype != "") {
                    if (params != "") params = params + "," + paramtype;
                    else params = paramtype;
                }
            }

            symbolcache[text] = composeReadableMethodSignature(unitname, classname, methodname, params);
        }
    };

    var composeReadableMethodSignature = function (unitname, classname, methodname, params) {
        var signature = "";

        if (classname != "") signature = classname.toLowerCase() + ".";

        signature = signature + methodname.toLowerCase();
        signature = signature + "(" + params.toLowerCase() + ")";

        return signature;
    };

    var shouldIgnoreSymbol = function(text) {
        for (var k in ignoredsymbols) {
            if (text.startsWith(ignoredsymbols[k])) {
                return true;
            }
        }

        return false;
    };

    var demangleIfNeeded = function(text) {
        if (text.includes('$')) {
            if (shouldIgnoreSymbol(text)) {
                return text;
            }

            for (var idx in sortedsymbolcache) {
                text = text.replace(sortedsymbolcache[idx][0], sortedsymbolcache[idx][1]);
            }

            return text;
        } else {
            return text;
        }
    };

    var buildOrderedCache = function () {
        sortedsymbolcache = [];
        for (var symbol in symbolcache) {
           sortedsymbolcache.push([symbol, symbolcache[symbol]]);
        }

        sortedsymbolcache = sortedsymbolcache.sort(function(a, b) {
            return b[0].length - a[0].length;
        });

        symbolcache = {};
    };

    compiler.postProcessAsm = function (result) {
        if (!result.okToCache) return result;

        buildOrderedCache();

        for (var j = 0; j < result.asm.length; ++j)
            result.asm[j].text = demangleIfNeeded(result.asm[j].text);

        return result;
    };

    compiler.optionsForFilter = function (filters, outputFilename, userOptions) {
        options = ['-g'];

        if (this.compiler.intelAsm && filters.intel && !filters.binary) {
            options = options.concat(this.compiler.intelAsm.split(" "));
        }

        filters.preProcessLines = preProcessLines;

        return options;
    };

    var saveDummyProjectFile = function (filename) {
        fs.writeFile(filename,
            "program prog; " +
            "uses output in 'output.pas'; " +
            "begin " +
            "end.");
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
                    input.substr(0, result.asm.indexOf("00000000004")) + "\n" +
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
            } else if (valueInBrackets.includes("output.pas")) {
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
                addDemangleToCache(asmLines[i]);
            }

            i++;
        }

        return asmLines;
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

        var projectFile = path.join(path.dirname(inputFilename), "prog.dpr");

        saveDummyProjectFile(projectFile);

        options.pop();
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

    return compiler.initialise();
}

module.exports = compileFPC;
