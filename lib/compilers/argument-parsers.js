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

import path from 'path';
import process from 'process';

import _ from 'underscore';

import {logger} from '../logger';
import * as props from '../properties';
import * as utils from '../utils';

export class BaseParser {
    static hasSupport(options, forOption) {
        return _.keys(options).find(option => option.includes(forOption));
    }

    static parseLines(stdout, optionRegex) {
        let previousOption = false;
        let options = {};

        utils.eachLine(stdout, line => {
            const match = line.match(optionRegex);
            if (!match) {
                if (previousOption && line.trim().length > 0) {
                    if (options[previousOption].description.endsWith('-'))
                        options[previousOption].description += line.trim();
                    else {
                        if (options[previousOption].description.length > 0)
                            options[previousOption].description += ' ' + line.trim();
                        else options[previousOption].description = line.trim();
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
                    timesused: 0,
                };
            }
        });

        return options;
    }

    static async getOptions(compiler, helpArg) {
        const optionFinder = /^\s*(--?[\d+,<=>[\]a-z|-]*)\s*(.*)/i;
        const result = await compiler.execCompilerCached(compiler.compiler.exe, [helpArg]);
        const options = result.code === 0 ? BaseParser.parseLines(result.stdout + result.stderr, optionFinder) : {};
        compiler.possibleArguments.populateOptions(options);
        return options;
    }

    static parse(compiler) {
        return compiler;
    }
}

export class GCCParser extends BaseParser {
    static async setCompilerSettingsFromOptions(compiler, options) {
        const keys = _.keys(options);
        logger.debug(`gcc-like compiler options: ${keys.join(' ')}`);
        if (BaseParser.hasSupport(options, '-masm=')) {
            // -masm= may be available but unsupported by the compiler.
            const res = await compiler.execCompilerCached(compiler.compiler.exe, [
                '-fsyntax-only',
                '--target-help',
                '-masm=intel',
            ]);
            if (res.code === 0) {
                compiler.compiler.intelAsm = '-masm=intel';
                compiler.compiler.supportsIntel = true;
            }
        }
        if (BaseParser.hasSupport(options, '-fdiagnostics-color')) {
            if (compiler.compiler.options) compiler.compiler.options += ' ';
            compiler.compiler.options += '-fdiagnostics-color=always';
        }
        // This check is not infallible, but takes care of Rust and Swift being picked up :)
        if (_.find(keys, key => key.startsWith('-fdump-'))) {
            compiler.compiler.supportsGccDump = true;

            // By default, consider the compiler to be a regular GCC (eg. gcc,
            // g++) and do the extra work of filtering out enabled pass that did
            // not produce anything.
            compiler.compiler.removeEmptyGccDump = true;
        }
    }

    static async parse(compiler) {
        const results = await Promise.all([
            GCCParser.getOptions(compiler, '-fsyntax-only --help'),
            GCCParser.getOptions(compiler, '-fsyntax-only --target-help'),
            GCCParser.getOptions(compiler, '-fsyntax-only --help=common'),
            GCCParser.getOptions(compiler, '-fsyntax-only --help=warnings'),
            GCCParser.getOptions(compiler, '-fsyntax-only --help=optimizers'),
        ]);
        const options = Object.assign({}, ...results);
        await this.setCompilerSettingsFromOptions(compiler, options);
        return compiler;
    }

    static async getOptions(compiler, helpArg) {
        const optionFinder = /^\s*(--?[\d+,<=>[\]a-z|-]*)\s*(.*)/i;
        const result = await compiler.execCompilerCached(compiler.compiler.exe, helpArg.split(' '));
        const options = result.code === 0 ? BaseParser.parseLines(result.stdout + result.stderr, optionFinder) : {};
        compiler.possibleArguments.populateOptions(options);
        return options;
    }
}

export class ClangParser extends BaseParser {
    static setCompilerSettingsFromOptions(compiler, options) {
        logger.debug(`clang-like compiler options: ${_.keys(options).join(' ')}`);
        if (BaseParser.hasSupport(options, '-fsave-optimization-record')) {
            compiler.compiler.optArg = '-fsave-optimization-record';
            compiler.compiler.supportsOptOutput = true;
        }
        if (BaseParser.hasSupport(options, '-emit-llvm')) {
            compiler.compiler.supportsIrView = true;
            compiler.compiler.irArg = ['-Xclang', '-emit-llvm', '-fsyntax-only'];
        }

        if (
            BaseParser.hasSupport(options, '-mllvm') &&
            this.mllvmOptions.has('--print-before-all') &&
            this.mllvmOptions.has('--print-after-all')
        ) {
            compiler.compiler.supportsLLVMOptPipelineView = true;
            compiler.compiler.llvmOptArg = ['-mllvm', '--print-before-all', '-mllvm', '--print-after-all'];
            compiler.compiler.llvmOptModuleScopeArg = [];
            compiler.compiler.llvmOptNoDiscardValueNamesArg = [];
            if (this.mllvmOptions.has('--print-module-scope')) {
                compiler.compiler.llvmOptModuleScopeArg = ['-mllvm', '-print-module-scope'];
            }
            if (BaseParser.hasSupport(options, '-fno-discard-value-names')) {
                compiler.compiler.llvmOptNoDiscardValueNamesArg = ['-fno-discard-value-names'];
            }
        }

        if (BaseParser.hasSupport(options, '-fcolor-diagnostics')) compiler.compiler.options += ' -fcolor-diagnostics';
        if (BaseParser.hasSupport(options, '-fno-crash-diagnostics'))
            compiler.compiler.options += ' -fno-crash-diagnostics';
    }

    static async parse(compiler) {
        try {
            const options = await ClangParser.getOptions(compiler, '--help');

            const EXAMPLES_PATH = props.get('builtin', 'sourcePath', './examples/');
            let filename = path.join(EXAMPLES_PATH, 'c++/default.cpp');
            if (!path.isAbsolute(filename)) filename = path.join(process.cwd(), filename);

            this.mllvmOptions = new Set(
                _.keys(await ClangParser.getOptions(compiler, `-mllvm --help-list-hidden ${filename} -c`, false)),
            );
            this.setCompilerSettingsFromOptions(compiler, options);
            return compiler;
        } catch (error) {
            logger.error('Error while trying to generate llvm backend arguments');
            logger.debug(error);
        }
    }

    static async getOptions(compiler, helpArg, populate = true) {
        const optionFinder = /^\s*(--?[\d+,<=>[\]a-z|-]*)\s*(.*)/i;
        const result = await compiler.execCompilerCached(compiler.compiler.exe, helpArg.split(' '));
        const options = result.code === 0 ? BaseParser.parseLines(result.stdout + result.stderr, optionFinder) : {};
        if (populate) {
            compiler.possibleArguments.populateOptions(options);
        }
        return options;
    }
}

export class ErlangParser extends BaseParser {
    static async parse(compiler) {
        await ErlangParser.getOptions(compiler, '-help');
        return compiler;
    }
}

export class PascalParser extends BaseParser {
    static async parse(compiler) {
        await PascalParser.getOptions(compiler, '-help');
        return compiler;
    }
}

export class ISPCParser extends BaseParser {
    static async setCompilerSettingsFromOptions(compiler, options) {
        if (BaseParser.hasSupport(options, '--x86-asm-syntax')) {
            compiler.compiler.intelAsm = '--x86-asm-syntax=intel';
            compiler.compiler.supportsIntel = true;
        }
    }

    static async parse(compiler) {
        const options = await ISPCParser.getOptions(compiler, '--help');
        await this.setCompilerSettingsFromOptions(compiler, options);
        return compiler;
    }

    static async getOptions(compiler, helpArg) {
        const result = await compiler.execCompilerCached(compiler.compiler.exe, [helpArg]);
        const optionFinder = /^\s*\[(--?[\d\s()+,/<=>a-z{|}-]*)]\s*(.*)/i;
        const options = result.code === 0 ? BaseParser.parseLines(result.stdout + result.stderr, optionFinder) : {};
        compiler.possibleArguments.populateOptions(options);
        return options;
    }
}

export class JavaParser extends BaseParser {
    static async parse(compiler) {
        await JavaParser.getOptions(compiler, '-help');
        return compiler;
    }
}

export class KotlinParser extends BaseParser {
    static async parse(compiler) {
        await KotlinParser.getOptions(compiler, '-help');
        return compiler;
    }
}

export class ScalaParser extends BaseParser {
    static async parse(compiler) {
        await ScalaParser.getOptions(compiler, '-help');
        return compiler;
    }
}

export class VCParser extends BaseParser {
    static async parse(compiler) {
        await VCParser.getOptions(compiler, '/help');
        return compiler;
    }

    static parseLines(stdout, optionRegex) {
        let previousOption = false;
        let options = {};

        const matchLine = line => {
            if (line.startsWith('/?')) return;

            const match = line.match(optionRegex);
            if (!match) {
                if (previousOption && line.trim().length > 0) {
                    if (options[previousOption].description.endsWith(':'))
                        options[previousOption].description += ' ' + line.trim();
                    else {
                        if (options[previousOption].description.length > 0)
                            options[previousOption].description += ', ' + line.trim();
                        else options[previousOption].description = line.trim();
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
                    timesused: 0,
                };
            }
        };

        utils.eachLine(stdout, line => {
            if (line.length === 0) return;
            if (line.includes('C/C++ COMPILER OPTIONS')) return;
            if (/^\s+-.*-$/.test(line)) return;

            let col1;
            let col2;
            if (line.length > 39 && line[40] === '/') {
                col1 = line.substr(0, 39);
                col2 = line.substr(40);
            } else {
                col1 = line;
                col2 = '';
            }

            if (col1) matchLine(col1);
            if (col2) matchLine(col2);
        });

        return options;
    }

    static async getOptions(compiler, helpArg) {
        const result = await compiler.execCompilerCached(compiler.compiler.exe, [helpArg]);
        const optionFinder = /^\s*(\/[\w#+,.:<=>[\]{|}-]*)\s*(.*)/i;
        const options = result.code === 0 ? this.parseLines(result.stdout, optionFinder) : {};
        compiler.possibleArguments.populateOptions(options);
        return options;
    }
}

export class RustParser extends BaseParser {
    static async setCompilerSettingsFromOptions(compiler, options) {
        if (BaseParser.hasSupport(options, '--color')) {
            if (compiler.compiler.options) compiler.compiler.options += ' ';
            compiler.compiler.options += '--color=always';
        }
    }

    static async parse(compiler) {
        const results = await Promise.all([
            RustParser.getOptions(compiler, '--help'),
            RustParser.getOptions(compiler, '-C help'),
            RustParser.getOptions(compiler, '--help -v'),
        ]);
        const options = Object.assign({}, ...results);
        await this.setCompilerSettingsFromOptions(compiler, options);
        return compiler;
    }

    static async getOptions(compiler, helpArg) {
        const result = await compiler.execCompilerCached(compiler.compiler.exe, helpArg.split(' '));
        let options = {};
        if (result.code === 0) {
            if (helpArg === '-C help') {
                const optionFinder = /^\s*(-c\s*[\d=a-z-]*)\s--\s(.*)/i;

                options = BaseParser.parseLines(result.stdout + result.stderr, optionFinder);
            } else {
                const optionFinder = /^\s*(--?[\d+,<=>[\]a-z|-]*)\s*(.*)/i;

                options = BaseParser.parseLines(result.stdout + result.stderr, optionFinder);
            }
        }
        compiler.possibleArguments.populateOptions(options);
        return options;
    }
}

export class MrustcParser extends BaseParser {
    static async parse(compiler) {
        await MrustcParser.getOptions(compiler, '--help');
        return compiler;
    }
}

export class NimParser extends BaseParser {
    static async parse(compiler) {
        await NimParser.getOptions(compiler, '-help');
        return compiler;
    }
}

export class CrystalParser extends BaseParser {
    static async parse(compiler) {
        await CrystalParser.getOptions(compiler, 'build');
        return compiler;
    }
}

export class TypeScriptNativeParser extends BaseParser {
    static async parse(compiler) {
        await TypeScriptNativeParser.getOptions(compiler, '--help');
        return compiler;
    }
}

export class TurboCParser extends BaseParser {
    static async parse(compiler) {
        await TurboCParser.getOptions(compiler, '');
        return compiler;
    }
}

export class ToitParser extends BaseParser {
    static async parse(compiler) {
        await ToitParser.getOptions(compiler, '-help');
        return compiler;
    }
}
