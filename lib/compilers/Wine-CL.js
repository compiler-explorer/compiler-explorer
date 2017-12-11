// Copyright (c) 2012-2017, Matt Godbolt
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
var utils = require('../utils');
var fs = require('fs-extra'),
    path = require('path'),
    _ = require('underscore-node');

function compileCl(info, env) {
    var compile = new Compile(info, env);
    compile.asm = new asm.AsmParser(env.compilerProps);
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
    compile.supportsObjdump = function () {
        return false;
    };

    compile.postProcessAsm = function (result) {
        if (!result.okToCache) return result;
        var demangler = this.compiler.demangler;
        if (!demangler) return result;

        if (demangler.toLowerCase().endsWith("undname.exe")) {
            return this.newTempDir().then(function (tmpDir) {
                var tmpfile = path.join(tmpDir, "output.s");
                fs.writeFileSync(tmpfile, _.pluck(result.asm, 'text').join("\r\n"));

                return this.exec(demangler, [tmpfile], {})
                    .then(_.bind(function (demangleResult) {
                    fs.unlink(tmpfile, function() {
                        fs.remove(tmpDir, function() {});
                    });

                    var lines = utils.splitLines(demangleResult.stdout);
                    for (var i = 0; i < result.asm.length; ++i)
                        result.asm[i].text = lines[i];
                    return result;
                }, this));
            }.bind(this));
        } else {
            return this.exec(demangler, [], {input: _.pluck(result.asm, 'text').join("\n")})
                .then(_.bind(function (demangleResult) {
                    var lines = utils.splitLines(demangleResult.stdout);
                    for (var i = 0; i < result.asm.length; ++i)
                        result.asm[i].text = lines[i];
                    return result;
                }, this));
        }
    };

    compile.optionsForFilter = function (filters, outputFilename, userOptions) {
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