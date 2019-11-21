// Copyright (c) 2018, Elliot Saba
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

const BaseCompiler = require('../base-compiler'),
      utils = require('../utils'),
      path = require('path');

class JuliaCompiler extends BaseCompiler {
    constructor(info, env) {
        super(info, env);
    }

    // No demangling for now
    postProcessAsm(result) {
        return result;
    }

    optionsForFilter(filters, outputFilename) {
        let opts = [
            "--output=" + outputFilename,
        ]
        if (filters.optOutput) {
            opts += [
                "--optimize",
            ]
        }
        return opts;
    }

    runCompiler(compiler, options, inputFilename, execOptions) {
        if (!execOptions) {
            execOptions = this.getDefaultExecOptions();
        }

        if (!execOptions.customCwd) {
            execOptions.customCwd = path.dirname(inputFilename);
        }

        // compiler wrapper, then input should be first argument, not last
        options.unshift(options.pop());
        options.unshift(path.resolve(__dirname, '..', '..', 'etc', 'scripts', 'compiler_wrapper.jl'))

        return this.exec(compiler, options, execOptions).then(result => {
            result.inputFilename = inputFilename;
            const transformedInput = result.filenameTransform(inputFilename);
            result.stdout = utils.parseOutput(result.stdout, transformedInput);
            result.stderr = utils.parseOutput(result.stderr, transformedInput);
            return result;
        });
    }
}

module.exports = JuliaCompiler;
