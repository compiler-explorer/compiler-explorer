// Copyright (c) 2017, Compiler Explorer Authors
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

import child_process from 'child_process';

import {WineVcCompiler} from '../lib/compilers/wine-vc';
import {WslVcCompiler} from '../lib/compilers/wsl-vc';

import {makeCompilationEnvironment} from './utils';

const languages = {
    'c++': {id: 'c++'},
};

const info = {
    lang: languages['c++'].id,
    exe: null,
    remote: true,
};

describe('Paths', () => {
    let env;

    before(() => {
        env = makeCompilationEnvironment({languages});
    });

    it('Linux -> Wine path', () => {
        const compiler = new WineVcCompiler(info, env);
        compiler.filename('/tmp/123456/output.s').should.equal('Z:/tmp/123456/output.s');
    });

    it('Linux -> Windows path', function () {
        process.env.winTmp = '/mnt/c/tmp';

        const compiler = new WslVcCompiler(info, env);
        compiler.filename('/mnt/c/tmp/123456/output.s').should.equal('c:/tmp/123456/output.s');
    });
});

function testExecOutput(x) {
    // Work around chai not being able to deepEquals with a function
    x.filenameTransform.should.be.a('function');
    delete x.filenameTransform;
    return x;
}

let ce;

function createCompiler(compiler) {
    if (ce === undefined) {
        ce = makeCompilationEnvironment({languages});
    }

    const info = {
        lang: languages['c++'].id,
        envVars: [],
    };

    return new compiler(info, ce);
}

if (process.platform === 'linux' && child_process.execSync('uname -a').toString().includes('Microsoft')) {
    // WSL
    describe('Wsl compiler', () => {
        let compiler;

        before(() => {
            compiler = createCompiler(WslVcCompiler);
        });

        it('Can set working directory', () => {
            return compiler
                .runCompiler('pwd', [], 'c:/this-should-be-run-in-mnt-c')
                .then(testExecOutput)
                .should.eventually.deep.equals({
                    code: 0,
                    inputFilename: 'c:/this-should-be-run-in-mnt-c',
                    okToCache: true,
                    stderr: [],
                    stdout: [{text: '/mnt/c'}],
                });
        });
    });
}
