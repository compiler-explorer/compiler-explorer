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

import {logger} from '../logger.js';
import * as props from '../properties.js';
import * as utils from '../utils.js';
import fs from 'fs-extra';
import {CompilerOverrideOptions} from '../../types/compilation/compiler-overrides.interfaces.js';

export class BaseParser {
    static setCompilerSettingsFromOptions(compiler, options) {}

    static hasSupport(options, forOption) {
        return _.keys(options).find(option => option.includes(forOption));
    }

    static hasSupportStartsWith(options, forOption) {
        return _.keys(options).find(option => option.startsWith(forOption));
    }

    static getExamplesRoot(): string {
        return props.get('builtin', 'sourcePath', './examples/');
    }

    static getDefaultExampleFilename() {
        return 'c++/default.cpp';
    }

    static getExampleFilepath(): string {
        let filename = path.join(this.getExamplesRoot(), this.getDefaultExampleFilename());
        if (!path.isAbsolute(filename)) filename = path.join(process.cwd(), filename);

        return filename;
    }

    static parseLines(stdout, optionWithDescRegex: RegExp, optionWithoutDescRegex?: RegExp) {
        let previousOption: false | string = false;
        const options = {};

        utils.eachLine(stdout, line => {
            const match1 = line.match(optionWithDescRegex);
            if (match1 && match1[1] && match1[2]) {
                previousOption = match1[1].trim();
                if (previousOption) {
                    options[previousOption] = {
                        description: this.spaceCompress(match1[2].trim()),
                        timesused: 0,
                    };
                }
                return;
            } else if (optionWithoutDescRegex) {
                const match2 = line.match(optionWithoutDescRegex);
                if (match2 && match2[1]) {
                    previousOption = match2[1].trim();

                    if (previousOption) {
                        options[previousOption] = {
                            description: '',
                            timesused: 0,
                        };
                    }
                    return;
                }
            }

            if (previousOption && line.trim().length > 0) {
                if (options[previousOption].description.endsWith('-'))
                    options[previousOption].description += line.trim();
                else {
                    if (options[previousOption].description.length > 0) {
                        const combined = options[previousOption].description + ' ' + line.trim();
                        options[previousOption].description = combined;
                    } else {
                        options[previousOption].description = line.trim();
                    }
                }

                options[previousOption].description = this.spaceCompress(options[previousOption].description);
            } else {
                previousOption = false;
            }
        });

        return options;
    }

    static spaceCompress(text: string): string {
        return text.replaceAll('  ', ' ');
    }

    static async getPossibleTargets(compiler): Promise<string[]> {
        return [];
    }

    static async getPossibleStdvers(compiler): Promise<CompilerOverrideOptions> {
        return [];
    }

    static async getPossibleEditions(compiler): Promise<string[]> {
        return [];
    }

    static async getOptions(compiler, helpArg) {
        const optionFinder1 = /^ *(--?[#\d+,<=>[\]a-z|-]* ?[\d+,<=>[\]a-z|-]*)  +(.*)/i;
        const optionFinder2 = /^ *(--?[#\d+,<=>[\]a-z|-]* ?[\d+,<=>[\]a-z|-]*)/i;
        const result = await compiler.execCompilerCached(compiler.compiler.exe, [helpArg]);
        const options =
            result.code === 0 ? this.parseLines(result.stdout + result.stderr, optionFinder1, optionFinder2) : {};
        compiler.possibleArguments.populateOptions(options);
        return options;
    }

    static parse(compiler) {
        return compiler;
    }
}

export class GCCParser extends BaseParser {
    static async checkAndSetMasmIntelIfSupported(compiler) {
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

    static override async setCompilerSettingsFromOptions(compiler, options) {
        const keys = _.keys(options);
        logger.debug(`gcc-like compiler options: ${keys.join(' ')}`);
        if (this.hasSupport(options, '-masm=')) {
            await this.checkAndSetMasmIntelIfSupported(compiler);
        }
        if (this.hasSupport(options, '-fstack-usage')) {
            compiler.compiler.stackUsageArg = '-fstack-usage';
            compiler.compiler.supportsStackUsageOutput = true;
        }
        if (this.hasSupport(options, '-fdiagnostics-color')) {
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
        if (this.hasSupportStartsWith(options, '-march=')) compiler.compiler.supportsMarch = true;
        if (this.hasSupportStartsWith(options, '--target=')) compiler.compiler.supportsTargetIs = true;
        if (this.hasSupportStartsWith(options, '--target ')) compiler.compiler.supportsTarget = true;
    }

    static override async parse(compiler) {
        const results = await Promise.all([
            this.getOptions(compiler, '-fsyntax-only --help'),
            this.getOptions(compiler, '-fsyntax-only --target-help'),
            this.getOptions(compiler, '-fsyntax-only --help=common'),
            this.getOptions(compiler, '-fsyntax-only --help=warnings'),
            this.getOptions(compiler, '-fsyntax-only --help=optimizers'),
            this.getOptions(compiler, '-fsyntax-only --help=target'),
        ]);
        const options = Object.assign({}, ...results);
        await this.setCompilerSettingsFromOptions(compiler, options);
        return compiler;
    }

    static override async getPossibleTargets(compiler): Promise<string[]> {
        const re = /Known valid arguments for -march= option:\s+(.*)/;
        const result = await compiler.execCompilerCached(compiler.compiler.exe, ['-fsyntax-only', '--target-help']);
        const match = result.stdout.match(re);
        if (match) {
            return match[1].split(' ');
        } else {
            return [];
        }
    }

    static getLanguageSpecificHelpFlags(): string[] {
        return ['-fsyntax-only', '--help=c++'];
    }

    static override async getPossibleStdvers(compiler): Promise<CompilerOverrideOptions> {
        const possible: CompilerOverrideOptions = [];
        const options = await this.getOptionsStrict(compiler, this.getLanguageSpecificHelpFlags());
        for (const opt in options) {
            if (opt.startsWith('-std=') && !options[opt].description?.startsWith('Deprecated')) {
                const stdver = opt.substring(5);
                possible.push({
                    name: stdver + ': ' + options[opt].description,
                    value: stdver,
                });
            }
        }
        return possible;
    }

    static override async getOptions(compiler, helpArg) {
        const optionFinder1 = /^ *(--?[#\d+,<=>[\]a-z|-]* ?[\d+,<=>[\]a-z|-]*)  +(.*)/i;
        const optionFinder2 = /^ *(--?[#\d+,<=>[\]a-z|-]* ?[\d+,<=>[\]a-z|-]*)/i;
        const result = await compiler.execCompilerCached(compiler.compiler.exe, helpArg.split(' '));
        const options =
            result.code === 0 ? this.parseLines(result.stdout + result.stderr, optionFinder1, optionFinder2) : {};
        compiler.possibleArguments.populateOptions(options);
        return options;
    }

    static async getOptionsStrict(compiler, helpArgs: string[]) {
        const optionFinder = /^ {2}(--?[\d+,<=>[\]a-z|-]*) *(.*)/i;
        const result = await compiler.execCompilerCached(compiler.compiler.exe, helpArgs);
        return result.code === 0 ? this.parseLines(result.stdout + result.stderr, optionFinder) : {};
    }
}

export class ClangParser extends BaseParser {
    static mllvmOptions = new Set<string>();

    static override setCompilerSettingsFromOptions(compiler, options) {
        const keys = _.keys(options);
        logger.debug(`clang-like compiler options: ${keys.join(' ')}`);

        if (keys.length === 0) {
            logger.error(`compiler options appear empty for ${compiler.compiler.id}`);
        }

        if (this.hasSupport(options, '-fsave-optimization-record')) {
            compiler.compiler.optArg = '-fsave-optimization-record';
            compiler.compiler.supportsOptOutput = true;
        }
        if (this.hasSupport(options, '-fstack-usage')) {
            compiler.compiler.stackUsageArg = '-fstack-usage';
            compiler.compiler.supportsStackUsageOutput = true;
        }

        if (this.hasSupport(options, '-emit-llvm')) {
            compiler.compiler.supportsIrView = true;
            compiler.compiler.irArg = ['-Xclang', '-emit-llvm', '-fsyntax-only'];
            compiler.compiler.minIrArgs = ['-emit-llvm'];
        }

        if (
            this.hasSupport(options, '-mllvm') &&
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
            if (this.hasSupport(options, '-fno-discard-value-names')) {
                compiler.compiler.llvmOptNoDiscardValueNamesArg = ['-fno-discard-value-names'];
            }
        }

        if (this.hasSupport(options, '-fcolor-diagnostics')) compiler.compiler.options += ' -fcolor-diagnostics';
        if (this.hasSupport(options, '-fno-crash-diagnostics')) compiler.compiler.options += ' -fno-crash-diagnostics';

        if (this.hasSupportStartsWith(options, '--target=')) compiler.compiler.supportsTargetIs = true;
        if (this.hasSupportStartsWith(options, '--target ')) compiler.compiler.supportsTarget = true;
    }

    static getMainHelpOptions(): string[] {
        return ['--help'];
    }

    static getHiddenHelpOptions(exampleFile: string): string[] {
        return ['-mllvm', '--help-list-hidden', exampleFile, '-c'];
    }

    static getStdVersHelpOptions(exampleFile: string): string[] {
        return ['-std=c++9999999', exampleFile, '-c'];
    }

    static getTargetsHelpOptions(): string[] {
        return ['--print-targets'];
    }

    static override async parse(compiler) {
        try {
            const options = await this.getOptions(compiler, this.getMainHelpOptions().join(' '));

            const filename = this.getExampleFilepath();

            this.mllvmOptions = new Set(
                _.keys(await this.getOptions(compiler, this.getHiddenHelpOptions(filename).join(' '), false, true)),
            );
            this.setCompilerSettingsFromOptions(compiler, options);
            return compiler;
        } catch (error) {
            logger.error('Error while trying to generate llvm backend arguments');
            logger.debug(error);
        }
    }

    static getRegexMatchesAsStdver(match, maxToMatch): CompilerOverrideOptions {
        if (!match) return [];
        if (!match[maxToMatch]) return [];

        const arr: CompilerOverrideOptions = [];

        for (let i = 1; i < maxToMatch; i++) {
            if (!match[i]) return [];

            arr.push({
                name: match[i] + ': ' + match[maxToMatch],
                value: match[i],
            });
        }

        return arr;
    }

    static extractPossibleStdvers(lines: string[]): CompilerOverrideOptions {
        const possible: CompilerOverrideOptions = [];
        const re1 = /note: use '([\w\d+:]*)' for '(.*)' standard/;
        const re2 = /note: use '([\w\d+:]*)' or '([\w\d+:]*)' for '(.*)' standard/;
        const re3 = /note: use '([\w\d+:]*)', '([\w\d+:]*)', or '([\w\d+:]*)' for '(.*)' standard/;
        const re4 = /note: use '([\w\d+:]*)', '([\w\d+:]*)', '([\w\d+:]*)', or '([\w\d+:]*)' for '(.*)' standard/;
        for (const line of lines) {
            let match = line.match(re1);
            let stdvers = this.getRegexMatchesAsStdver(match, 2);
            possible.push(...stdvers);
            if (stdvers.length > 0) continue;

            match = line.match(re2);
            stdvers = this.getRegexMatchesAsStdver(match, 3);
            possible.push(...stdvers);
            if (stdvers.length > 0) continue;

            match = line.match(re3);
            stdvers = this.getRegexMatchesAsStdver(match, 4);
            possible.push(...stdvers);
            if (stdvers.length > 0) continue;

            match = line.match(re4);
            stdvers = this.getRegexMatchesAsStdver(match, 5);
            possible.push(...stdvers);
        }
        return possible;
    }

    static override async getPossibleStdvers(compiler): Promise<CompilerOverrideOptions> {
        let possible: CompilerOverrideOptions = [];

        // clang doesn't have a --help option to get the std versions, we'll have to compile with a fictional stdversion to coax a response
        const filename = this.getExampleFilepath();

        const result = await compiler.execCompilerCached(compiler.compiler.exe, this.getStdVersHelpOptions(filename), {
            ...compiler.getDefaultExecOptions(),
            createAndUseTempDir: true,
        });
        if (result.stderr) {
            const lines = utils.splitLines(result.stderr);

            possible = this.extractPossibleStdvers(lines);
            possible.sort((a, b) => {
                return a.value === b.value ? 0 : a.value > b.value ? 1 : -1;
            });
        }

        return possible;
    }

    static extractPossibleTargets(lines: string[]): string[] {
        const re = /\s+([\w-]*)\s*-\s.*/;
        return lines
            .map(line => {
                const match = line.match(re);
                if (match) {
                    return match[1];
                } else {
                    return false;
                }
            })
            .filter(Boolean) as string[];
    }

    static override async getPossibleTargets(compiler): Promise<string[]> {
        const result = await compiler.execCompilerCached(compiler.compiler.exe, this.getTargetsHelpOptions());
        return this.extractPossibleTargets(utils.splitLines(result.stdout));
    }

    static override async getOptions(compiler, helpArg, populate = true, isolate = false) {
        const optionFinderWithDesc = /^ {2}?(--?[#\d+,<=>[\]a-zA-Z|-]*\s?[\d+,<=>[\]a-zA-Z|-]*)\s+([A-Z].*)/;
        const optionFinderWithoutDesc = /^ {2}?(--?[#\d+,<=>[\]a-z|-]*\s?[\d+,<=>[\]a-z|-]*)/i;
        const execOptions = isolate ?? {...compiler.getDefaultExecOptions(), createAndUseTempDir: true};
        const result = await compiler.execCompilerCached(compiler.compiler.exe, helpArg.split(' '), execOptions);
        const options =
            result.code === 0
                ? this.parseLines(result.stdout + result.stderr, optionFinderWithDesc, optionFinderWithoutDesc)
                : {};
        if (populate) compiler.possibleArguments.populateOptions(options);
        return options;
    }
}

export class GCCCParser extends GCCParser {
    static override getLanguageSpecificHelpFlags(): string[] {
        return ['-fsyntax-only', '--help=c'];
    }

    static override getDefaultExampleFilename() {
        return 'c/default.c';
    }
}

export class ClangCParser extends ClangParser {
    static override getDefaultExampleFilename() {
        return 'c/default.c';
    }

    static override getStdVersHelpOptions(exampleFile: string): string[] {
        return ['-std=c9999999', exampleFile, '-c'];
    }
}

export class CircleParser extends ClangParser {
    static override async getOptions(compiler, helpArg) {
        const optionFinder1 = /^ +(--?[#\d,<=>[\]a-z|_.-]*)  +- (.*)/i;
        const optionFinder2 = /^ +(--?[#\d,<=>[\]a-z|_.-]*)/i;
        const result = await compiler.execCompilerCached(compiler.compiler.exe, helpArg.split(' '));
        const options = result.code === 0 ? this.parseLines(result.stdout, optionFinder1, optionFinder2) : {};
        compiler.possibleArguments.populateOptions(options);
        return options;
    }

    static override async getPossibleStdvers(compiler): Promise<CompilerOverrideOptions> {
        const possible: CompilerOverrideOptions = [];
        const optionFinder = /^ {4}=([\w+]*) +- +(.*)/i;
        const result = await compiler.execCompilerCached(compiler.compiler.exe, ['--help']);
        let isInStdVerSection = false;
        for (const line of utils.splitLines(result.stdout)) {
            if (!isInStdVerSection && line.startsWith('  --std=')) {
                isInStdVerSection = true;
                continue;
            } else if (isInStdVerSection && line.startsWith('  --')) {
                break;
            }

            if (!isInStdVerSection) continue;

            const match = line.match(optionFinder);
            if (match) {
                const stdver = match[1];
                const desc = match[2];
                possible.push({
                    name: stdver + ': ' + desc,
                    value: stdver,
                });
            }
        }
        return possible;
    }
}

export class LDCParser extends BaseParser {
    static override setCompilerSettingsFromOptions(compiler, options) {
        if (this.hasSupport(options, '--fsave-optimization-record')) {
            compiler.compiler.optArg = '--fsave-optimization-record';
            compiler.compiler.supportsOptOutput = true;
        }

        if (this.hasSupport(options, '--print-before-all') && this.hasSupport(options, '--print-after-all')) {
            compiler.compiler.supportsLLVMOptPipelineView = true;
            compiler.compiler.llvmOptArg = ['--print-before-all', '--print-after-all'];
            compiler.compiler.llvmOptModuleScopeArg = [];
            compiler.compiler.llvmOptNoDiscardValueNamesArg = [];
            if (this.hasSupport(options, '--print-module-scope')) {
                compiler.compiler.llvmOptModuleScopeArg = ['--print-module-scope'];
            }
            if (this.hasSupport(options, '--fno-discard-value-names')) {
                compiler.compiler.llvmOptNoDiscardValueNamesArg = ['--fno-discard-value-names'];
            }
        }

        if (this.hasSupport(options, '--enable-color')) {
            compiler.compiler.options += ' --enable-color';
        }
    }

    static override async parse(compiler) {
        const options = await this.getOptions(compiler, '--help-hidden');
        this.setCompilerSettingsFromOptions(compiler, options);
        return compiler;
    }

    static override async getOptions(compiler, helpArg, populate = true) {
        const optionFinder = /^\s*(--?[\d+,<=>[\]a-z|-]*)\s*(.*)/i;
        const result = await compiler.execCompilerCached(compiler.compiler.exe, helpArg.split(' '));
        const options = result.code === 0 ? this.parseLines(result.stdout + result.stderr, optionFinder) : {};
        if (populate) {
            compiler.possibleArguments.populateOptions(options);
        }
        return options;
    }
}

export class ErlangParser extends BaseParser {
    static override async parse(compiler) {
        await this.getOptions(compiler, '-help');
        return compiler;
    }
}

export class PascalParser extends BaseParser {
    static override async parse(compiler) {
        await this.getOptions(compiler, '-help');
        return compiler;
    }
}

export class ICCParser extends GCCParser {
    static override async setCompilerSettingsFromOptions(compiler, options) {
        const keys = _.keys(options);
        if (this.hasSupport(options, '-masm=')) {
            compiler.compiler.intelAsm = '-masm=intel';
            compiler.compiler.supportsIntel = true;
        }
        if (this.hasSupport(options, '-fdiagnostics-color')) {
            if (compiler.compiler.options) compiler.compiler.options += ' ';
            compiler.compiler.options += '-fdiagnostics-color=always';
        }
        if (_.find(keys, key => key.startsWith('-fdump-'))) {
            compiler.compiler.supportsGccDump = true;
            compiler.compiler.removeEmptyGccDump = true;
        }
        if (this.hasSupportStartsWith(options, '-march=')) compiler.compiler.supportsMarch = true;
        if (this.hasSupportStartsWith(options, '--target=')) compiler.compiler.supportsTargetIs = true;
        if (this.hasSupportStartsWith(options, '--target ')) compiler.compiler.supportsTarget = true;
    }

    static extractPossibleStdvers(lines: string[]): CompilerOverrideOptions {
        const stdverRe = /-std=<std>/;
        const descRe = /^\s{12}([\w\d+]*)\s+(.*)/;
        const possible: CompilerOverrideOptions = [];
        let foundStdver = false;
        let skipLine = false;
        for (const line of lines) {
            if (skipLine) {
                skipLine = false;
                continue;
            }

            if (!foundStdver) {
                const match = line.match(stdverRe);
                if (match) {
                    foundStdver = true;
                    skipLine = true;
                }
            } else {
                const descMatch = line.match(descRe);
                if (descMatch) {
                    possible.push({
                        name: descMatch[1] + ': ' + descMatch[2],
                        value: descMatch[1],
                    });
                } else {
                    break;
                }
            }
        }
        return possible;
    }

    static override async getPossibleStdvers(compiler): Promise<CompilerOverrideOptions> {
        const result = await compiler.execCompilerCached(compiler.compiler.exe, ['--help']);
        const lines = utils.splitLines(result.stdout);

        return this.extractPossibleStdvers(lines);
    }

    static override async parse(compiler) {
        const results = await Promise.all([this.getOptions(compiler, '-fsyntax-only --help')]);
        const options = Object.assign({}, ...results);
        await this.setCompilerSettingsFromOptions(compiler, options);
        return compiler;
    }
}

export class ISPCParser extends BaseParser {
    static override async setCompilerSettingsFromOptions(compiler, options) {
        if (this.hasSupport(options, '--x86-asm-syntax')) {
            compiler.compiler.intelAsm = '--x86-asm-syntax=intel';
            compiler.compiler.supportsIntel = true;
        }
    }

    static override async parse(compiler) {
        const options = await this.getOptions(compiler, '--help');
        await this.setCompilerSettingsFromOptions(compiler, options);
        return compiler;
    }

    static override async getOptions(compiler, helpArg) {
        const result = await compiler.execCompilerCached(compiler.compiler.exe, [helpArg]);
        const optionFinder = /^\s*\[(--?[\d\s()+,/<=>a-z{|}-]*)]\s*(.*)/i;
        const options = result.code === 0 ? this.parseLines(result.stdout + result.stderr, optionFinder) : {};
        compiler.possibleArguments.populateOptions(options);
        return options;
    }
}

export class JavaParser extends BaseParser {
    static override async parse(compiler) {
        await this.getOptions(compiler, '-help');
        return compiler;
    }
}

export class KotlinParser extends BaseParser {
    static override async parse(compiler) {
        await this.getOptions(compiler, '-help');
        return compiler;
    }
}

export class ScalaParser extends BaseParser {
    static override async parse(compiler) {
        await this.getOptions(compiler, '-help');
        return compiler;
    }
}

export class VCParser extends BaseParser {
    static override async parse(compiler) {
        await this.getOptions(compiler, '/help');
        return compiler;
    }

    static override parseLines(stdout, optionRegex) {
        let previousOption: string | false = false;
        const options = {};

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

    static extractPossibleStdvers(lines: string[]): CompilerOverrideOptions {
        const stdverRe = /\/std:<(.*)>\s.*/;
        const descRe = /(c\+\+.*) - (.*)/;
        const possible: CompilerOverrideOptions = [];
        const stdverValues: string[] = [];
        for (const line of lines) {
            if (stdverValues.length === 0) {
                const match = line.match(stdverRe);
                if (match) {
                    stdverValues.push(...match[1].split('|'));
                }
            } else {
                const descMatch = line.match(descRe);
                if (descMatch) {
                    if (stdverValues.includes(descMatch[1])) {
                        possible.push({
                            name: descMatch[1] + ': ' + descMatch[2],
                            value: descMatch[1],
                        });
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
        }
        return possible;
    }

    static override async getPossibleStdvers(compiler): Promise<CompilerOverrideOptions> {
        const result = await compiler.execCompilerCached(compiler.compiler.exe, ['/help']);
        const lines = utils.splitLines(result.stdout);

        return this.extractPossibleStdvers(lines);
    }

    static override async getOptions(compiler, helpArg) {
        const result = await compiler.execCompilerCached(compiler.compiler.exe, [helpArg]);
        const optionFinder = /^\s*(\/[\w#+,.:<=>[\]{|}-]*)\s*(.*)/i;
        const options = result.code === 0 ? this.parseLines(result.stdout, optionFinder) : {};
        compiler.possibleArguments.populateOptions(options);
        return options;
    }
}

export class RustParser extends BaseParser {
    static override async setCompilerSettingsFromOptions(compiler, options) {
        if (this.hasSupport(options, '--color')) {
            if (compiler.compiler.options) compiler.compiler.options += ' ';
            compiler.compiler.options += '--color=always';
        }
        if (this.hasSupportStartsWith(options, '--target=')) compiler.compiler.supportsTargetIs = true;
        if (this.hasSupportStartsWith(options, '--target ')) compiler.compiler.supportsTarget = true;
    }

    static override async parse(compiler) {
        const results = await Promise.all([
            this.getOptions(compiler, '--help'),
            this.getOptions(compiler, '-C help'),
            this.getOptions(compiler, '--help -v'),
        ]);
        const options = Object.assign({}, ...results);
        await this.setCompilerSettingsFromOptions(compiler, options);
        return compiler;
    }

    static override async getPossibleEditions(compiler): Promise<string[]> {
        const result = await compiler.execCompilerCached(compiler.compiler.exe, ['--help']);
        const re = /--edition ([\d|]*)/;

        const match = result.stdout.match(re);
        if (match && match[1]) {
            return match[1].split('|');
        }

        return [];
    }

    static override async getPossibleTargets(compiler): Promise<string[]> {
        const result = await compiler.execCompilerCached(compiler.compiler.exe, ['--print', 'target-list']);
        return utils.splitLines(result.stdout).filter(Boolean);
    }

    static parseRustHelpLines(stdout) {
        let previousOption: false | string = false;
        const options = {};

        const doubleOptionFinder = /^\s{4}(-\w, --\w*\s?[\w[\]:=]*)\s*(.*)/i;
        const singleOptionFinder = /^\s{8}(--[\w-]*\s?[\w[\]:=|-]*)\s*(.*)/i;
        const singleComplexOptionFinder = /^\s{4}(-\w*\s?[\w[\]:=]*)\s*(.*)/i;

        utils.eachLine(stdout, line => {
            let description = '';

            const match1 = line.match(doubleOptionFinder);
            const match2 = line.match(singleOptionFinder);
            const match3 = line.match(singleComplexOptionFinder);
            if (!match1 && !match2 && !match3) {
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
            } else {
                if (match1) {
                    previousOption = match1[1].trim();
                    if (match1[2]) description = match1[2].trim();
                } else if (match2) {
                    previousOption = match2[1].trim();
                    if (match2[2]) description = match2[2].trim();
                } else if (match3) {
                    previousOption = match3[1].trim();
                    if (match3[2]) description = match3[2].trim();
                }
            }

            if (previousOption) {
                options[previousOption] = {
                    description: description,
                    timesused: 0,
                };
            }
        });

        return options;
    }

    static override async getOptions(compiler, helpArg) {
        const result = await compiler.execCompilerCached(compiler.compiler.exe, helpArg.split(' '));
        let options = {};
        if (result.code === 0) {
            if (helpArg === '-C help') {
                const optionFinder = /^\s*(-c\s*[\d=a-z-]*)\s--\s(.*)/i;

                options = this.parseLines(result.stdout + result.stderr, optionFinder);
            } else {
                options = this.parseRustHelpLines(result.stdout + result.stderr);
            }
        }
        compiler.possibleArguments.populateOptions(options);
        return options;
    }
}

export class MrustcParser extends BaseParser {
    static override async parse(compiler) {
        await this.getOptions(compiler, '--help');
        return compiler;
    }
}

export class NimParser extends BaseParser {
    static override async parse(compiler) {
        await this.getOptions(compiler, '-help');
        return compiler;
    }
}

export class CrystalParser extends BaseParser {
    static override async parse(compiler) {
        await this.getOptions(compiler, 'build');
        return compiler;
    }
}

export class TypeScriptNativeParser extends BaseParser {
    static override async parse(compiler) {
        await this.getOptions(compiler, '--help');
        return compiler;
    }
}

export class TurboCParser extends BaseParser {
    static override async parse(compiler) {
        await this.getOptions(compiler, '');
        return compiler;
    }
}

export class ToitParser extends BaseParser {
    static override async parse(compiler) {
        await this.getOptions(compiler, '-help');
        return compiler;
    }
}

export class JuliaParser extends BaseParser {
    // Get help line from wrapper not Julia runtime
    static override async getOptions(compiler, helpArg) {
        const optionFinder = /^\s*(--?[\d+,<=>[\]a-z|-]*)\s*(.*)/i;
        const result = await compiler.execCompilerCached(compiler.compiler.exe, [
            compiler.compilerWrapperPath,
            helpArg,
        ]);
        const options = result.code === 0 ? this.parseLines(result.stdout + result.stderr, optionFinder) : {};
        compiler.possibleArguments.populateOptions(options);
        return options;
    }

    static override async parse(compiler) {
        await this.getOptions(compiler, '--help');
        return compiler;
    }
}

export class Z88dkParser extends BaseParser {
    static override async getPossibleTargets(compiler): Promise<string[]> {
        const configPath = path.join(path.dirname(compiler.compiler.exe), '../share/z88dk/lib/config');
        const targets: string[] = [];
        const dir = await fs.readdir(configPath);
        dir.forEach(filename => {
            if (filename.toLowerCase().endsWith('.cfg')) {
                targets.push(filename.substring(0, filename.length - 4));
            }
        });
        return targets;
    }
}

export class ZigCxxParser extends ClangParser {
    static override getMainHelpOptions(): string[] {
        return ['c++', '--help'];
    }

    static override getHiddenHelpOptions(exampleFile: string): string[] {
        return ['c++', '-mllvm', '--help-list-hidden', exampleFile, '-S', '-o', '/tmp/output.s'];
    }

    static override getStdVersHelpOptions(exampleFile: string): string[] {
        return ['c++', '-std=c++9999999', exampleFile, '-S', '-o', '/tmp/output.s'];
    }

    static override getTargetsHelpOptions(): string[] {
        return ['c++', '--print-targets'];
    }
}

export class GccFortranParser extends GCCParser {
    static override getDefaultExampleFilename() {
        return 'fortran/default.f90';
    }

    static override getLanguageSpecificHelpFlags(): string[] {
        return ['-fsyntax-only', '--help=fortran'];
    }
}

export class FlangParser extends ClangParser {
    static override getDefaultExampleFilename() {
        return 'fortran/default.f90';
    }

    static override hasSupport(options, param) {
        // param is available but we get a warning, so lets not use it
        if (param === '-fcolor-diagnostics') return undefined;

        return BaseParser.hasSupport(options, param);
    }

    static override extractPossibleStdvers(lines: string[]): CompilerOverrideOptions {
        const possible: CompilerOverrideOptions = [];
        const re1 = /error: Only -std=([\w\d+]*) is allowed currently./;
        for (const line of lines) {
            const match = line.match(re1);
            if (match && match[1]) {
                possible.push({
                    name: match[1],
                    value: match[1],
                });
            }
        }
        return possible;
    }
}

export class GHCParser extends GCCParser {
    static override async parse(compiler) {
        const results = await Promise.all([this.getOptions(compiler, '--help')]);
        const options = Object.assign({}, ...results);
        await this.setCompilerSettingsFromOptions(compiler, options);
        return compiler;
    }

    static override async getOptions(compiler, helpArg) {
        const optionFinder1 = /^ {4}(-[\w[\]]+)\s+(.*)/i;
        const optionFinder2 = /^ {4}(-[\w[\]]+)/;
        const result = await compiler.execCompilerCached(compiler.compiler.exe, helpArg.split(' '));
        const options = result.code === 0 ? this.parseLines(result.stdout, optionFinder1, optionFinder2) : {};

        compiler.possibleArguments.populateOptions(options);
        return options;
    }
}

export class SwiftParser extends ClangParser {
    static override async parse(compiler) {
        const results = await Promise.all([this.getOptions(compiler, '--help')]);
        const options = Object.assign({}, ...results);
        this.setCompilerSettingsFromOptions(compiler, options);
        return compiler;
    }

    static override async getPossibleStdvers(compiler): Promise<CompilerOverrideOptions> {
        return [];
    }

    static override async getPossibleTargets(compiler): Promise<string[]> {
        return [];
    }
}

export class TendraParser extends GCCParser {
    static override async parse(compiler) {
        const results = await Promise.all([this.getOptions(compiler, '--help')]);
        const options = Object.assign({}, ...results);
        await this.setCompilerSettingsFromOptions(compiler, options);
        return compiler;
    }

    static override async getOptions(compiler, helpArg) {
        const optionFinder = /^ *(-[#\d+,<=>[\]a-z|-]* ?[\d+,<=>[\]a-z|-]*) : +(.*)/i;
        const result = await compiler.execCompilerCached(compiler.compiler.exe, helpArg.split(' '));
        const options = this.parseLines(result.stdout + result.stderr, optionFinder);
        compiler.possibleArguments.populateOptions(options);
        return options;
    }

    static override async getPossibleStdvers(compiler): Promise<CompilerOverrideOptions> {
        return [];
    }

    static override async getPossibleTargets(compiler): Promise<string[]> {
        return [];
    }
}

export class GolangParser extends GCCParser {
    static override getDefaultExampleFilename() {
        return 'go/default.go';
    }

    static override async parse(compiler) {
        const results = await Promise.all([
            this.getOptions(compiler, 'build -o ./output.s "-gcflags=-S --help" ' + this.getExampleFilepath()),
        ]);
        const options = Object.assign({}, ...results);
        await this.setCompilerSettingsFromOptions(compiler, options);
        return compiler;
    }

    static override async getOptions(compiler, helpArg) {
        const optionFinder1 = /^\s*(--?[#\d+,<=>[\]a-z|-]* ?[\d+,<=>[\]a-z|-]*)\s+(.*)/i;
        const optionFinder2 = /^\s*(--?[#\d+,<=>[\]a-z|-]* ?[\d+,<=>[\]a-z|-]*)/i;
        const result = await compiler.execCompilerCached(compiler.compiler.exe, utils.splitArguments(helpArg), {
            ...compiler.getDefaultExecOptions(),
            createAndUseTempDir: true,
        });
        const options = this.parseLines(result.stdout + result.stderr, optionFinder1, optionFinder2);
        compiler.possibleArguments.populateOptions(options);
        return options;
    }
}
