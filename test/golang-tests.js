// Copyright (c) 2018, Patrick Quist
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

const chai = require('chai'),
    chaiAsPromised = require("chai-as-promised"),
    fs = require('fs-extra'),
    utils = require('../lib/utils'),
    logger = require('../lib/logger').logger,
    CompilationEnvironment = require('../lib/compilation-env'),
    GoCompiler = require('../lib/compilers/golang');

chai.use(chaiAsPromised);
chai.should();

const props = (key, deflt) => deflt;

const ce = new CompilationEnvironment(props);
const info = {
    "exe": null,
    "remote": true,
    "lang": "pascal"
};

ce.compilerPropsL = function (lang, property, defaultValue) {
    return "";
};

function testGoAsm(basefilename) {
    const compiler = new GoCompiler(info, ce);

    const asmLines = utils.splitLines(fs.readFileSync(basefilename + ".asm").toString());

    const result = {
        stdout: asmLines.map((line) => {
            return {
                text: line
            };
        })
    };

    return compiler.postProcess(result).then((output) => {
        const expectedOutput = utils.splitLines(fs.readFileSync(basefilename + ".output.asm").toString()).join("\n");

        return output.should.deep.equal({
            asm: expectedOutput,
            stdout: []
        });
    });
}

describe('GO asm tests', () => {
    it('Handles unknown line number correctly', () => {
        return testGoAsm("test/golang/bug-901");
    });
});
