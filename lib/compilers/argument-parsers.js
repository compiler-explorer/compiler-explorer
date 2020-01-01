// Copyright (c) 2017, Jared Wyles
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

const _ = require('underscore'),
    logger = require('../logger').logger,
    utils = require('../utils');

class BaseParser {
    static hasSupport(options, forOption) {
        return _.keys(options).find(option => option.includes(forOption));
    }

    static parseLines(stdout, optionRegex) {
        let previousOption = false;
        let options = {};

        utils.eachLine(stdout, line => {
            const match = line.match(optionRegex);
            if (!match) {
                if (previousOption && (line.trim().length !== 0)) {
                    if (options[previousOption].description.endsWith("-"))
                        options[previousOption].description += line.trim();
                    else {
                        if (options[previousOption].description.length !== 0)
                            options[previousOption].description += " " + line.trim();
                        else
                            options[previousOption].description = line.trim();
                    }
                } else {
                    previousOption = false;
                }
                return;
            }

            if (match) previousOption = match[1];
            if (previousOption) {
                options[previousOption] = {
                    description: match[2].trim(),
                    timesused: 0
                };
            }
        });

        return options;
    }

    static getOptions(compiler, helpArg) {
        return compiler.execCompilerCached(compiler.compiler.exe, [helpArg]).then(result => {
            let options = {};
            if (result.code === 0) {
                const optionFinder = /^\s*(--?[a-z0-9=+,[\]<>|-]*)\s*(.*)/i;

                options = BaseParser.parseLines(result.stdout + result.stderr, optionFinder);
            }
            compiler.possibleArguments.populateOptions(options);
            return options;
        });
    }

    static parse(compiler) {
        return compiler;
    }
}

class GCCParser extends BaseParser {
    static parse(compiler) {
        return Promise.all([
            GCCParser.getOptions(compiler, "--target-help"),
            GCCParser.getOptions(compiler, "--help=common"),
            GCCParser.getOptions(compiler, "--help=optimizers")
        ]).then(results => {
            const options = _.extend.apply(_.extend, results);
            const keys =  _.keys(options);
            logger.debug(`gcc-like compiler options: ${keys.join(" ")}`);
            if (BaseParser.hasSupport(options, "-masm=")) {
                compiler.compiler.intelAsm = "-masm=intel";
                compiler.compiler.supportsIntel = true;
            }
            if (BaseParser.hasSupport(options, "-fdiagnostics-color")) {
                if (compiler.compiler.options) compiler.compiler.options += " ";
                compiler.compiler.options += "-fdiagnostics-color=always";
            }
            // This check is not infallible, but takes care of Rust and Swift being picked up :)
            if (_.find(keys, key => key.startsWith('-fdump-'))) {
                compiler.compiler.supportsGccDump = true;
            }
            return compiler;
        });
    }
}

class ClangParser extends BaseParser {
    static parse(compiler) {
        return ClangParser.getOptions(compiler, "--help").then(options => {
            logger.debug(`clang-like compiler options: ${_.keys(options).join(" ")}`);
            if (BaseParser.hasSupport(options, '-fsave-optimization-record')) {
                compiler.compiler.optArg = "-fsave-optimization-record";
                compiler.compiler.supportsOptOutput = true;
            }
            if (BaseParser.hasSupport(options, "-fcolor-diagnostics")) {
                if (compiler.compiler.options) compiler.compiler.options += " ";
                compiler.compiler.options += "-fcolor-diagnostics";
            }
            if (BaseParser.hasSupport(options, "-emit-llvm")) {
                compiler.compiler.supportsIrView = true;
                compiler.compiler.irArg = ['-Xclang', '-emit-llvm', '-fsyntax-only'];
            }
            if (BaseParser.hasSupport(options, "-fno-crash-diagnostics")) {
                if (compiler.compiler.options) compiler.compiler.options += " ";
                compiler.compiler.options += "-fno-crash-diagnostics";
            }
            return compiler;
        });
    }
}

class PascalParser extends BaseParser {
    static parse(compiler) {
        return PascalParser.getOptions(compiler, "-help").then(() => compiler);
    }
}

class ISPCParser extends BaseParser {
    static parse(compiler) {
        return ISPCParser.getOptions(compiler, "--help").then(options => {
            if (BaseParser.hasSupport(options, "--x86-asm-syntax")) {
                compiler.compiler.intelAsm = "--x86-asm-syntax=intel";
                compiler.compiler.supportsIntel = true;
            }
            return compiler;
        });
    }

    static getOptions(compiler, helpArg) {
        return compiler.execCompilerCached(compiler.compiler.exe, [helpArg]).then(result => {
            let options = {};
            if (result.code === 0) {
                const optionFinder = /^\s*\[(--?[a-z0-9=+,{}()\s<>/|-]*)\]\s*(.*)/i;

                options = BaseParser.parseLines(result.stdout + result.stderr, optionFinder);
            }
            compiler.possibleArguments.populateOptions(options);
            return options;
        });
    }
}

class JavaParser extends BaseParser {

    static parse(compiler) {
        return JavaParser.getOptions(compiler, "-help").then(() => compiler);
    }
}

class VCParser extends BaseParser {
    static parse(compiler) {
        return Promise.all([
            VCParser.getOptions(compiler, "/help")
        ]).then(() => {
            return compiler;
        });
    }

    static parseLines(stdout, optionRegex) {
        let previousOption = false;
        let options = {};

        const matchLine = (line) => {
            if (line.startsWith("/?")) return;

            const match = line.match(optionRegex);
            if (!match) {
                if (previousOption && (line.trim().length !== 0)) {
                    if (options[previousOption].description.endsWith(":"))
                        options[previousOption].description += " " + line.trim();
                    else {
                        if (options[previousOption].description.length !== 0)
                            options[previousOption].description += ", " + line.trim();
                        else
                            options[previousOption].description = line.trim();
                    }
                } else {
                    previousOption = false;
                }
                return;
            }

            if (match) previousOption = match[1];
            if (previousOption) {
                options[previousOption] = {
                    description: match[2].trim(),
                    timesused: 0
                };
            }
        };
    
        utils.eachLine(stdout, line => {
            if (line.length === 0) return;
            if (line.includes("C/C++ COMPILER OPTIONS")) return;
            if (line.match(/^\s\s*-.*-$/)) return;

            let col1;
            let col2;
            if ((line.length > 39) && (line[40] === '/')) {
                col1 = line.substr(0, 39);
                col2 = line.substr(40);
            } else {
                col1 = line;
                col2 = "";
            }

            if (col1) matchLine(col1);
            if (col2) matchLine(col2);
        });

        return options;
    }

    static getOptions(compiler, helpArg) {
        return compiler.execCompilerCached(compiler.compiler.exe, [helpArg]).then(result => {
            let options = {};
            if (result.code === 0) {
                const optionFinder = /^\s*(\/[a-z0-9=:+#.,[\]{}<>|_-]*)\s*(.*)/i;

                options = this.parseLines(result.stdout, optionFinder);
            }
            compiler.possibleArguments.populateOptions(options);
            return options;
        });
    }
}

class RustParser extends BaseParser {
    static parse(compiler) {
        return Promise.all([
            RustParser.getOptions(compiler, "--help"),
            RustParser.getOptions(compiler, "-C help"),
            RustParser.getOptions(compiler, "--help -v")
        ]).then(results => {
            const options = _.extend.apply(_.extend, results);
            if (BaseParser.hasSupport(options, "--color")) {
                if (compiler.compiler.options) compiler.compiler.options += " ";
                compiler.compiler.options += "--color=always";
            }
            return compiler;
        });
    }

    static getOptions(compiler, helpArg) {
        return compiler.execCompilerCached(compiler.compiler.exe, helpArg.split(' ')).then(result => {
            let options = {};
            if (result.code === 0) {
                if (helpArg === "-C help") {
                    const optionFinder = /^\s*(-C\s*[a-z0-9=-]*)\s--\s(.*)/i;

                    options = BaseParser.parseLines(result.stdout + result.stderr, optionFinder);
                } else {
                    const optionFinder = /^\s*(--?[a-z0-9=+,[\]<>|-]*)\s*(.*)/i;

                    options = BaseParser.parseLines(result.stdout + result.stderr, optionFinder);
                }
            }
            compiler.possibleArguments.populateOptions(options);
            return options;
        });
    }
}

class NimParser extends BaseParser {

    static parse(compiler) {
        return NimParser.getOptions(compiler, "-help").then(() => compiler);
    }
}

module.exports = {
    Base: BaseParser,
    Clang: ClangParser,
    GCC: GCCParser,
    Java: JavaParser,
    VC: VCParser,
    Pascal: PascalParser,
    ISPC: ISPCParser,
    Rust: RustParser,
    Nim: NimParser,
};
