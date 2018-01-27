// Copyright (c) 2012-2018, Matt Godbolt
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

const Compile = require('../base-compiler'),
    exec = require('../exec'),
    logger = require('../logger').logger;

function compilePpci(info, env) {
    const compiler = new Compile(info, env);
    const originalExec = compiler.exec;

    compiler.exec = function (compiler, args, options) {
        if (compiler.endsWith('.py')) {
            const python = env.ceProps("python3");
            options = options || {};

            const matches = compiler.match(/^(.*)(\/ppci\/)(.*).py/);
            if (matches) {
                const pythonpath = matches[1];
                const ppciname = "ppci." + matches[3].replace('/', '.');
                options.env = {'PYTHONPATH': pythonpath};
                let python_args = ['-m', ppciname].concat(args);
                return exec.execute(python, python_args, options);
            } else {
                logger.error('invalid ppci path');
            }
        } else {
            return originalExec(compiler, args, options);
        }
    };

    return compiler.initialise();
}

module.exports = compilePpci;
