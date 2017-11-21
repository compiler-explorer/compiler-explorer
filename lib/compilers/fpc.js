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
    path = require("path");

function compileFPC(info, env) {
    var compiler = new Compile(info, env);
    compiler.supportsOptOutput = false;
    compiler.originalPostProcess = compiler.postProcess;
    compiler.symbolcache = [];

    compiler.postProcess = function (result, outputFilename, filters) {
        if (filters.binary && this.supportsObjdump()) {
            outputFilename = outputFilename.replace(".s", ".o");
        }

        return this.originalPostProcess(result, outputFilename, filters);
    };

    compiler.demangleIfNeeded = function(text) {
        var line = text;

        var p1 = -1, p2 = -1;

        if (line.indexOf('$') != -1) {
            if (line[line.length - 1] == ':') {
                if ((line.indexOf("VMT_$") == 0) ||
                    (line.indexOf("INIT_$") == 0) ||
                    (line.indexOf("RTTI_$") == 0) ||
                    (line.indexOf("U_$") == 0)) {
                    return line;
                } else {
                    p1 = 0;
                    p2 = line.length - 1;
                    text = line.substr(0, line.length - 1);
                }
            } else {
                var sortedsymbols = this.symbolcache.sort(function(a,b) {
                    return b.length - a.length;
                });

                for (var k in sortedsymbols) {
                    text = text.replace(k, this.symbolcache[k]);
                }

                return text;
            }
        }

        if ((p1 != -1) && (p2 != -1)) {
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
                        if (phase == 0) phase = 1
                        else if (phase == 1) {
                            if (paramtype == "") phase = 2
                            else if (params != "") {
                                params = params + "," + paramtype;
                                paramtype = "";
                            } else if (params == "") {
                                params = paramtype;
                                paramtype = "";
                            }
                        }
                    } else {
                        if (phase == 0) methodname = methodname + signature[idx]
                        else if (phase == 1) paramtype = paramtype + signature[idx]
                        else if (phase == 2) resulttype = resulttype + signature[idx];
                    }
                }

                if (paramtype != "") {
                    if (params != "") params = params + "," + paramtype
                    else params = paramtype;
                }
            }

            var symbol = text;

            var unmangledsymbol = "";
            if (classname != "") unmangledsymbol = unmangledsymbol + classname.toLowerCase() + ".";
            unmangledsymbol = unmangledsymbol + methodname.toLowerCase();

            unmangledsymbol = unmangledsymbol + "(" + params.toLowerCase() + ")";

            this.symbolcache[symbol] = unmangledsymbol;

            return line.substr(0, p1) + unmangledsymbol + line.substr(p2);
        } else {
            return text;
        }
    };

    compiler.postProcessAsm = function (result) {
        if (!result.okToCache) return result;

        for (var i = 0; i < result.asm.length; ++i)
            result.asm[i].text = this.demangleIfNeeded(result.asm[i].text);

        return result;
    };

    compiler.optionsForFilter = function (filters, outputFilename, userOptions) {
        filters.execute = false;

        return ['@/opt/compiler-explorer/fpc/fpc.cfg', '-a'];
    };

    return compiler.initialise();
}

module.exports = compileFPC;
