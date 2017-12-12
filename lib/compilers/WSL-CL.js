// Copyright (c) 2017, Andrew Pardoe
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


// AP: WSL-CL.js is a WSL-specific copy of CL.js (renamed to Wine-CL.js). 
// Main differences: 
//   Don't run under Wine (obviously)
//   Translate compiler path from Unix mounted volume (/mnt/c/tmp) to Windows (c:/tmp)

var Compile = require('../base-compiler');
var asm = require('../asm-cl');
var RunCLDemangler = require('../cl-support').RunCLDemangler;

function compileCl(info, env) {
    var compile = new Compile(info, env);
    compile.asm = new asm.AsmParser(env.compilerProps);
    info.supportsFiltersInBinary = true;
    if (process.platform == "linux") {
        var origExec = compile.exec;
        compile.exec = function (command, args, options) {
            return origExec(command, args, options);
        };
        compile.filename = function (fn) {
            // AP: Need to translate compiler paths from what the Node.js process sees 
            // on a Unix mounted volume (/mnt/c/tmp) to what CL sees on Windows (c:/tmp)
            // We know process.env.tmpDir is of format /mnt/X/dir where X is drive letter.
            var driveLetter = process.env.tmpDir.substring(5, 6);
            var directoryPath = process.env.tmpDir.substring(7);
            var windowsStyle = driveLetter.concat(":/", directoryPath);
            return fn.replace(process.env.tmpDir, windowsStyle);
        };
    }
    compile.supportsObjdump = function () {
        return false;
    };

    compile.postProcessAsm = function(result) {
        return RunCLDemangler(this, result);
    };

    compile.optionsForFilter = function (filters, outputFilename) {
        return [
            '/FAsc',
            '/c',
            '/Fa' + this.filename(outputFilename),
            '/Fo' + this.filename(outputFilename + '.obj')
        ];
    };
    return compile.initialise();
}

module.exports = compileCl;
