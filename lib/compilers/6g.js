// Copyright (c) 2012-2016, Matt Godbolt
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

function compile6g(info, env) {
    function convert6g(code) {
        var re = /^[0-9]+\s*\(([^:]+):([0-9]+)\)\s*([A-Z]+)(.*)/;
        var prevLine = null;
        var file = null;
        return code.map(function (obj) {
            var line = obj.line;
            var match = line.match(re);
            if (match) {
                var res = "";
                if (file === null) {
                    res += "\t.file 1 \"" + match[1] + "\"\n";
                    file = match[1];
                }
                if (prevLine != match[2]) {
                    res += "\t.loc 1 " + match[2] + "\n";
                    prevLine = match[2];
                }
                return res + "\t" + match[3].toLowerCase() + match[4];
            } else
                return null;
        }).filter(_.identity).join("\n");
    }

    var compiler = new Compile(info, env);
    compiler.postProcess = function (result, outputFilename, filters) {
        result.asm = this.convert6g(result.stdout);
        result.stdout = [];
        return Promise.resolve(result);
    };
    return compiler.initialise();
}

module.exports = compile6g;