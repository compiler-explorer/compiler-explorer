// Copyright (c) 2012-2018, Jared Wyles
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

const _ = require('underscore-node'),
    logger = require('../logger').logger,
    utils = require('../utils');

const getOptions = function (compiler, helpArg) {
    return compiler.exec(compiler.compiler.exe, [helpArg])
        .then(result => {
            const options = {};
            if (result.code === 0) {
                const optionFinder = /^\s*(--?[-a-zA-Z]+)/;

                utils.eachLine(result.stdout + result.stderr, line => {
                    const match = line.match(optionFinder);
                    if (!match) return;
                    options[match[1]] = true;
                });
            }
            return options;
        });
};

const gccParser = function (compiler) {
    return Promise.all([
        getOptions(compiler, "--target-help"),
        getOptions(compiler, "--help=common")])
        .then(results => {
            const options = _.extend.apply(_.extend, results);
            compiler.compiler.supportsGccDump = true;
            logger.debug("gcc-like compiler options: ", _.keys(options).join(" "));
            if (options['-masm']) {
                compiler.compiler.intelAsm = "-masm=intel";
                compiler.compiler.supportsIntel = true;
            }
            if (options['-fdiagnostics-color']) {
                if (compiler.compiler.options) compiler.compiler.options += " ";
                else compiler.compiler.options = "";

                compiler.compiler.options += "-fdiagnostics-color=always";
            }
            return compiler;
        });
};

const clangParser = function (compiler) {
    return getOptions(compiler, "--help").then(options => {
        logger.debug("clang-like compiler options: ", _.keys(options).join(" "));
        if (options['-fsave-optimization-record']) {
            compiler.compiler.optArg = "-fsave-optimization-record";
            compiler.compiler.supportsOptOutput = true;
        }
        if (options['-fcolor-diagnostics']) {
            if (compiler.compiler.options) compiler.compiler.options += " ";
            compiler.compiler.options += "-fcolor-diagnostics";
        }
        return compiler;
    });
};


module.exports = {
    getOptions: getOptions,
    clang: clangParser,
    gcc: gccParser
};