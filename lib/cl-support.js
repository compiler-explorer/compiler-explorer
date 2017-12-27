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

var fs = require('fs-extra'),
    path = require('path'),
    _ = require('underscore-node'),
    utils = require('./utils');

function OverwriteAsmLines(result, demangleOutput) {
    var lines = utils.splitLines(demangleOutput);
    for (var i = 0; i < result.asm.length; ++i)
        result.asm[i].text = lines[i];
    return result;
}

function RunCLDemangler(compiler, result) {
    if (!result.okToCache) return result;
    var demangler = compiler.compiler.demangler;
    if (!demangler) return result;

    var asmFileContent = _.pluck(result.asm, 'text').join("\n");

    if (demangler.toLowerCase().endsWith("undname.exe")) {
        return compiler.newTempDir().then(function (tmpDir) {
            var tmpfile = path.join(tmpDir, "output.s");
            fs.writeFileSync(tmpfile, asmFileContent);
    
            var tmpFileAsArgument = compiler.filename(tmpfile);

            return compiler.exec(demangler, [tmpFileAsArgument], compiler.getDefaultExecOptions())
                .then(function (demangleResult) {
                fs.unlink(tmpfile, function() {
                    fs.remove(tmpDir, function() {});
                });
    
                return OverwriteAsmLines(result, demangleResult.stdout);
            });
        });
    } else {
        return compiler.exec(demangler, [], {input: asmFileContent})
            .then(function (demangleResult) {
                return OverwriteAsmLines(result, demangleResult.stdout);
            });
    }
}

exports.RunCLDemangler = RunCLDemangler;
