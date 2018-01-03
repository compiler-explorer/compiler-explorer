// Copyright (c) 2012-2018, Patrick Quist
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
    Compile = require('../base-compiler'),
    path = require('path');

function compileBCC(info, env) {
    var compile = new Compile(info, env);
    if ((process.platform === "linux") || (process.platform === "darwin")) {
        var wine = env.ceProps("wine");
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

    compile.optionsForFilter = function (filters, outputFilename, userOptions) {
        let options = [
            '-o' + this.filename(outputFilename),
            '-n' + this.filename(path.dirname(outputFilename)),
            '-v'
        ];
        if (!filters.binary) options = options.concat('-S');
        return options;
    };

    if (info.unitTestMode) {
        compile.initialise();

        return compile;
    } else {
        return compile.initialise();
    }
}

module.exports = compileBCC;
