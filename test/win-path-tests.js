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

var chai = require('chai');
var should = chai.should();
var assert = chai.assert;
var WslCL = require('../lib/compilers/WSL-CL');
var WineCL = require('../lib/compilers/Wine-CL');
var logger = require('../lib/logger').logger;
var CompilationEnvironment = require('../lib/compilation-env').CompilationEnvironment;

describe('Paths', function () {
    it('Linux -> Wine path', function () {
        var info = {
            "exe": null,
            "remote": true,
            "unitTestMode": true
        };
        var envprops = function (key, deflt) {
            return deflt;
        };

        var env = new CompilationEnvironment(envprops);
        env.compilerProps = function () {};

        var compiler = new WineCL(info, env);        
        compiler.filename("/tmp/123456/output.s").should.equal("Z:/tmp/123456/output.s");
    });

    it('Linux -> Windows path', function () {
        var info = {
            "exe": null,
            "remote": true,
            "unitTestMode": true
        };
        var envprops = function (key, deflt) {
            return deflt;
        };

        var env = new CompilationEnvironment(envprops);
        env.compilerProps = function () {};

        process.env.winTmp = "/mnt/c/tmp";

        var compiler = new WslCL(info, env);
        compiler.filename("/mnt/c/tmp/123456/output.s").should.equal("c:/tmp/123456/output.s");
    });
});
