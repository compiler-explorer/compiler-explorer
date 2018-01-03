// Copyright (c) 2012-2018, Andrew Pardoe
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

const Compile = require('../base-compiler'),
    asm = require('../asm-cl'),
    temp = require('temp'),
    RunCLDemangler = require('../cl-support').RunCLDemangler;

function compileCl(info, env) {
    var compile = new Compile(info, env);
    compile.asm = new asm.AsmParser(compile.compilerProps);
    info.supportsFiltersInBinary = true;
    if ((process.platform === "linux") || (process.platform === "darwin")) {
        const origExec = compile.exec;
        compile.exec = function (command, args, options) {
            return origExec(command, args, options);
        };
        compile.filename = function (fn) {
            // AP: Need to translate compiler paths from what the Node.js process sees 
            // on a Unix mounted volume (/mnt/c/tmp) to what CL sees on Windows (c:/tmp)
            // We know process.env.tmpDir is of format /mnt/X/dir where X is drive letter.
            const driveLetter = process.env.winTmp.substring(5, 6);
            const directoryPath = process.env.winTmp.substring(7);
            const windowsStyle = driveLetter.concat(":/", directoryPath);
            return fn.replace(process.env.winTmp, windowsStyle);
        };
    }
    // AP: Create CE temp directory in winTmp directory instead of the tmpDir directory.
    // NPM temp package: https://www.npmjs.com/package/temp, see Affixes
    compile.newTempDir = function () {
        return new Promise(function (resolve, reject) {
            temp.mkdir({prefix: 'compiler-explorer-compiler', dir: process.env.winTmp}, function (err, dirPath) {
                if (err)
                    reject("Unable to open temp file: " + err);
                else
                    resolve(dirPath);
            });
        });
    };
    compile.supportsObjdump = function () {
        return false;
    };

    compile.getArgumentParser = () => (compiler) => compiler;

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

    if (info.unitTestMode) {
        compile.initialise();

        return compile;
    } else {
        return compile.initialise();
    }
}

module.exports = compileCl;
