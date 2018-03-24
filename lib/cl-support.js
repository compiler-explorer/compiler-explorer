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

const fs = require('fs-extra'),
    path = require('path'),
    DemanglerCPP = require("./demangler-cpp").DemanglerCPP;

class CLDemangler extends DemanglerCPP {
    constructor(demanglerExe, symbolstore, compiler) {
        super(demanglerExe, symbolstore);

        this.compiler = compiler;

        this.labelDef = /^([.a-z_$?]+[a-z0-9$_.@]*)\sPROC.*/i;
        this.jumpDef = /j.+\s+([.a-z_$?]+[a-z0-9$_.@]*)/i;
        this.callDef = /(call|mov|lea|EXTRN).+\s+([.a-z_$?]+[a-z0-9$_.@]*)/i;
    }

    execDemangler(options) {
        if (this.demanglerExe.toLowerCase().endsWith("undname.exe")) {
            return this.compiler.newTempDir().then(tmpDir => {
                const tmpfile = path.join(tmpDir, "output.s");
                fs.writeFileSync(tmpfile, options.input);

                const tmpFileAsArgument = this.compiler.filename(tmpfile);

                return this.compiler.exec(this.demanglerExe, [tmpFileAsArgument], this.compiler.getDefaultExecOptions())
                    .then(demangleResult => {
                        fs.unlink(tmpfile, () => {
                            fs.remove(tmpDir, () => {});
                        });
                        return demangleResult;
                    });
            });
        } else {
            return this.compiler.exec(this.demanglerExe, [], {input: options.input});
        }
    }
}

exports.Demangler = CLDemangler;
