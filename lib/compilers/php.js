// Copyright (c) 2017, Mike Cochrane
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

var Compile = require('../base-compiler'),
    fs = require('fs-extra'),
    php = require('../php'),
    utils = require('../utils');

function compilePHP(info, env) {
    var compiler = new Compile(info, env);
    compiler.supportsIntel = false;
    compiler.couldSupportASTDump = () => true;
    compiler.asm = new php.PHPByteCodeParser(env.compilerProps);
    compiler.optionsForFilter = function (filters, outputFilename, userOptions) {
        // Add options for vld extension.
        return ['-dvld.active=1','-dvld.execute=0', '-dvld.format=1', '-dopcache.file_update_protection=0'];
    };

    compiler.prepareArguments = function (userOptions, filters, backendOptions, inputFilename, outputFilename) {
        let options = this.optionsForFilter(filters, outputFilename, userOptions);
        backendOptions = backendOptions || {};

        if (this.compiler.options) {
            options = options.concat(this.compiler.options.split(" "));
        }

        if (this.compiler.supportsOptOutput && backendOptions.produceOptInfo) {
            options = options.concat(this.compiler.optArg);
        }

        // Only allow "-e" and enabling the opcode cache option, everthing else looks dangerous
        userOptions = (userOptions || []).filter(option => option == '-e' || option == '-dopcache.enable_cli=1');

        // Hack: Pass output filename as the last option
        return options.concat(userOptions || []).concat([this.filename(inputFilename), outputFilename]);
    };

    compiler.runCompiler = function (compiler, options, inputFilename, execOptions) {
        // Hack: Get output filename from last option
        let outputFilename = options.pop();

        return this.exec(compiler, options, execOptions).then(function (result) {
            result.inputFilename = inputFilename;
            result.stdout = utils.parseOutput(result.stdout, inputFilename);
            if (result.code === 0) {
                // Actual bytecode is emitted to stderror, write it to the output file
                fs.writeFileSync(outputFilename, result.stderr);
                result.stderr = "";
            } else {
                result.stderr = utils.parseOutput(result.stderr, inputFilename);
            }
            return result;
        });
    };

    compiler.generateAST = function (inputFilename, options) {
        let newOptions = ['etc/scripts/dump-php-ast.php', inputFilename];
        let execOptions = this.getDefaultExecOptions();
        return this.exec(this.compiler.exe, newOptions, execOptions).then(result => result.stdout);
    };

    return compiler.initialise();
}

module.exports = compilePHP;
