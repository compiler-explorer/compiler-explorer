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

import path from 'node:path';
import process from 'node:process';

import fs from 'node:fs/promises';
import * as Sentry from '@sentry/node';
import _ from 'underscore';

import {splitArguments} from '../../shared/common-utils.js';
import {CompilerOverrideOptions} from '../../types/compilation/compiler-overrides.interfaces.js';
import {Argument} from '../../types/compiler-arguments.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';
import {logger} from '../logger.js';
import * as props from '../properties.js';
import * as utils from '../utils.js';

import {JuliaCompiler} from './julia.js';

export class BaseParser {
    static setCompilerSettingsFromOptions(compiler: BaseCompiler, options: Record<string, Argument>) {}

    static hasSupport(options: Record<string, Argument>, forOption: string) {
        return _.keys(options).find(option => option.includes(forOption));
    }

    static hasSupportStartsWith(options: Record<string, Argument>, forOption: string) {
        return _.keys(options).find(option => option.startsWith(forOption));
    }

    static getExamplesRoot(): string {
        return props.get<string>('builtin', 'sourcePath', './examples/');
    }

    static getDefaultExampleFilename() {
        return 'c++/default.cpp';
    }

    static getExampleFilepath(): string {
        let filename = path.join(BaseParser.getExamplesRoot(), BaseParser.getDefaultExampleFilename());
        if (!path.isAbsolute(filename)) filename = path.join(process.cwd(), filename);

        return filename;
    }

    static parseLines(stdout: string, optionWithDescRegex: RegExp, optionWithoutDescRegex?: RegExp) {
        let previousOption: false | string = false;
        const options: Record<string, Argument> = {};

        utils.eachLine(stdout, line => {
            const match1 = line.match(optionWithDescRegex);
            if (match1?.[1] && match1[2]) {
                previousOption = match1[1].trim();
                if (previousOption) {
                    options[previousOption] = {
                        description: BaseParser.spaceCompress(match1[2].trim()),
                        timesused: 0,
                    };
                }
                return;
            }
            if (optionWithoutDescRegex) {
                const match2 = line.match(optionWithoutDescRegex);
                if (match2?.[1]) {
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

                options[previousOption].description = BaseParser.spaceCompress(options[previousOption].description);
            } else {
                previousOption = false;
            }
        });

        return options;
    }

    static spaceCompress(text: string): string {
        return text.replaceAll('  ', ' ');
    }

    static async getPossibleTargets(compiler: BaseCompiler): Promise<string[]> {
        return [];
    }

    static async getPossibleStdvers(compiler: BaseCompiler): Promise<CompilerOverrideOptions> {
        return [];
    }

    // Currently used only for Rust
    static async getPossibleEditions(compiler: BaseCompiler): Promise<string[]> {
        return [];
    }

    static async getOptions(compiler: BaseCompiler, helpArg: string) {
        const optionFinder1 = /^ *(--?[\d#+,<=>[\]a-z|-]* ?[\d+,<=>[\]a-z|-]*) {2,}(.*)/i;
        const optionFinder2 = /^ *(--?[\d#+,<=>[\]a-z|-]* ?[\d+,<=>[\]a-z|-]*)/i;
        const result = await compiler.execCompilerCached(compiler.compiler.exe, [helpArg]);
        const options =
            result.code === 0 ? BaseParser.parseLines(result.stdout + result.stderr, optionFinder1, optionFinder2) : {};
        compiler.possibleArguments.populateOptions(options);
        return options;
    }

    // async for compatibility with children, who call getOptions
    static async parse(compiler: BaseCompiler) {
        return compiler;
    }
}

export class GCCParser extends BaseParser {
    static async checkAndSetMasmIntelIfSupported(compiler: BaseCompiler) {
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

    static override async setCompilerSettingsFromOptions(compiler: BaseCompiler, options: Record<string, Argument>) {
        const keys = _.keys(options);
        logger.debug(`gcc-like compiler options: ${keys.join(' ')}`);
        if (GCCParser.hasSupport(options, '-masm=')) {
            await GCCParser.checkAndSetMasmIntelIfSupported(compiler);
        }
        if (GCCParser.hasSupport(options, '-fstack-usage')) {
            compiler.compiler.stackUsageArg = '-fstack-usage';
            compiler.compiler.supportsStackUsageOutput = true;
        }
        if (GCCParser.hasSupport(options, '-fdiagnostics-color')) {
            if (compiler.compiler.options) compiler.compiler.options += ' ';
            compiler.compiler.options += '-fdiagnostics-color=always';
        }
        if (GCCParser.hasSupport(options, '-fverbose-asm')) {
            compiler.compiler.supportsVerboseAsm = true;
        }
        if (GCCParser.hasSupport(options, '-fopt-info')) {
            compiler.compiler.optArg = '-fopt-info-all';
            compiler.compiler.supportsOptOutput = true;
        }
        // This check is not infallible, but takes care of Rust and Swift being picked up :)
        if (_.find(keys, key => key.startsWith('-fdump-'))) {
            compiler.compiler.supportsGccDump = true;

            // By default, consider the compiler to be a regular GCC (eg. gcc,
            // g++) and do the extra work of filtering out enabled pass that did
            // not produce anything.
            compiler.compiler.removeEmptyGccDump = true;
        }
        if (GCCParser.hasSupportStartsWith(options, '-march=')) compiler.compiler.supportsMarch = true;
        if (GCCParser.hasSupportStartsWith(options, '--target=')) compiler.compiler.supportsTargetIs = true;
        if (GCCParser.hasSupportStartsWith(options, '--target ')) compiler.compiler.supportsTarget = true;
    }

    static override async parse(compiler: BaseCompiler) {
        const results = await Promise.all([
            GCCParser.getOptions(compiler, '-fsyntax-only --help'),
            GCCParser.getOptions(compiler, '-fsyntax-only --target-help'),
            GCCParser.getOptions(compiler, '-fsyntax-only --help=common'),
            GCCParser.getOptions(compiler, '-fsyntax-only --help=warnings'),
            GCCParser.getOptions(compiler, '-fsyntax-only --help=optimizers'),
            GCCParser.getOptions(compiler, '-fsyntax-only --help=target'),
        ]);
        const options = Object.assign({}, ...results);
        await GCCParser.setCompilerSettingsFromOptions(compiler, options);
        return compiler;
    }

    static override async getPossibleTargets(compiler: BaseCompiler): Promise<string[]> {
        const re = /Known valid arguments for -march= option:\s+(.*)/;
        const result = await compiler.execCompilerCached(compiler.compiler.exe, ['-fsyntax-only', '--target-help']);
        const match = result.stdout.match(re);
        if (match) {
            return match[1].split(' ');
        }
        return [];
    }

    static getLanguageSpecificHelpFlags(): string[] {
        return ['-fsyntax-only', '--help=c++'];
    }

    static override async getPossibleStdvers(compiler: BaseCompiler): Promise<CompilerOverrideOptions> {
        const possible: CompilerOverrideOptions = [];
        const options = await GCCParser.getOptionsStrict(compiler, GCCParser.getLanguageSpecificHelpFlags());
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

    static override async getOptions(compiler: BaseCompiler, helpArg: string) {
        const optionFinder1 = /^ *(--?[\d#+,<=>[\]a-z|-]* ?[\d+,<=>[\]a-z|-]*) {2,}(.*)/i;
        const optionFinder2 = /^ *(--?[\d#+,<=>[\]a-z|-]* ?[\d+,<=>[\]a-z|-]*)/i;
        const result = await compiler.execCompilerCached(compiler.compiler.exe, helpArg.split(' '));
        const options =
            result.code === 0 ? GCCParser.parseLines(result.stdout + result.stderr, optionFinder1, optionFinder2) : {};
        compiler.possibleArguments.populateOptions(options);
        return options;
    }

    static async getOptionsStrict(compiler: BaseCompiler, helpArgs: string[]) {
        const optionFinder = /^ {2}(--?[\d+,<=>[\]a-z|-]*) *(.*)/i;
        const result = await compiler.execCompilerCached(compiler.compiler.exe, helpArgs);
        return result.code === 0 ? GCCParser.parseLines(result.stdout + result.stderr, optionFinder) : {};
    }
}

export class ClangParser extends BaseParser {
    static mllvmOptions = new Set<string>();

    static override async setCompilerSettingsFromOptions(compiler: BaseCompiler, options: Record<string, Argument>) {
        const keys = _.keys(options);
        logger.debug(`clang-like compiler options: ${keys.join(' ')}`);

        if (keys.length === 0) {
            logger.error(`compiler options appear empty for ${compiler.compiler.id}`);
        }

        if (ClangParser.hasSupport(options, '-fsave-optimization-record')) {
            compiler.compiler.optArg = '-fsave-optimization-record';
            compiler.compiler.supportsOptOutput = true;
        }
        if (ClangParser.hasSupport(options, '-fstack-usage')) {
            compiler.compiler.stackUsageArg = '-fstack-usage';
            compiler.compiler.supportsStackUsageOutput = true;
        }
        if (ClangParser.hasSupport(options, '-fverbose-asm')) {
            compiler.compiler.supportsVerboseAsm = true;
        }

        if (ClangParser.hasSupport(options, '-emit-llvm')) {
            compiler.compiler.supportsIrView = true;
            compiler.compiler.irArg = ['-Xclang', '-emit-llvm', '-fsyntax-only'];
            compiler.compiler.minIrArgs = ['-emit-llvm'];
        }

        // if (ClangParser.hasSupport(options, '-emit-cir')) {
        // #7265: clang-trunk supposedly has '-emit-cir', but it's not doing much. Checking explicitly
        // for clangir in the compiler name instead.
        if (compiler.compiler.name?.includes('clangir')) {
            compiler.compiler.supportsClangirView = true;
        }

        if (
            ClangParser.hasSupport(options, '-mllvm') &&
            ClangParser.mllvmOptions.has('--print-before-all') &&
            ClangParser.mllvmOptions.has('--print-after-all')
        ) {
            compiler.compiler.optPipeline = {
                arg: ['-mllvm', '--print-before-all', '-mllvm', '--print-after-all'],
                moduleScopeArg: [],
                noDiscardValueNamesArg: [],
            };
            if (ClangParser.mllvmOptions.has('--print-module-scope')) {
                compiler.compiler.optPipeline.moduleScopeArg = ['-mllvm', '-print-module-scope'];
            }
            if (ClangParser.hasSupport(options, '-fno-discard-value-names')) {
                compiler.compiler.optPipeline.noDiscardValueNamesArg = ['-fno-discard-value-names'];
            }
        }

        if (ClangParser.hasSupport(options, '-fcolor-diagnostics')) compiler.compiler.options += ' -fcolor-diagnostics';
        if (ClangParser.hasSupport(options, '-fno-crash-diagnostics'))
            compiler.compiler.options += ' -fno-crash-diagnostics';

        if (ClangParser.hasSupportStartsWith(options, '--target=')) compiler.compiler.supportsTargetIs = true;
        if (ClangParser.hasSupportStartsWith(options, '--target ')) compiler.compiler.supportsTarget = true;
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

    static override async parse(compiler: BaseCompiler) {
        try {
            const options = await ClangParser.getOptions(compiler, ClangParser.getMainHelpOptions().join(' '));

            const filename = ClangParser.getExampleFilepath();

            ClangParser.mllvmOptions = new Set(
                _.keys(
                    await ClangParser.getOptions(
                        compiler,
                        ClangParser.getHiddenHelpOptions(filename).join(' '),
                        false,
                        true,
                    ),
                ),
            );
            ClangParser.setCompilerSettingsFromOptions(compiler, options);
        } catch (error) {
            const err = `Error while trying to generate llvm backend arguments for ${compiler.compiler.id}: ${error}`;
            logger.error(err);
            Sentry.captureMessage(err);
        }
        return compiler;
    }

    static getRegexMatchesAsStdver(match: RegExpMatchArray | null, maxToMatch: number): CompilerOverrideOptions {
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
        const re1 = /note: use '([\w+:]*)' for '(.*)' standard/;
        const re2 = /note: use '([\w+:]*)' or '([\w+:]*)' for '(.*)' standard/;
        const re3 = /note: use '([\w+:]*)', '([\w+:]*)', or '([\w+:]*)' for '(.*)' standard/;
        const re4 = /note: use '([\w+:]*)', '([\w+:]*)', '([\w+:]*)', or '([\w+:]*)' for '(.*)' standard/;
        for (const line of lines) {
            let match = line.match(re1);
            let stdvers = ClangParser.getRegexMatchesAsStdver(match, 2);
            possible.push(...stdvers);
            if (stdvers.length > 0) continue;

            match = line.match(re2);
            stdvers = ClangParser.getRegexMatchesAsStdver(match, 3);
            possible.push(...stdvers);
            if (stdvers.length > 0) continue;

            match = line.match(re3);
            stdvers = ClangParser.getRegexMatchesAsStdver(match, 4);
            possible.push(...stdvers);
            if (stdvers.length > 0) continue;

            match = line.match(re4);
            stdvers = ClangParser.getRegexMatchesAsStdver(match, 5);
            possible.push(...stdvers);
        }
        return possible;
    }

    static override async getPossibleStdvers(compiler: BaseCompiler): Promise<CompilerOverrideOptions> {
        let possible: CompilerOverrideOptions = [];

        // clang doesn't have a --help option to get the std versions, we'll have to compile with a fictional stdversion to coax a response
        const filename = ClangParser.getExampleFilepath();

        const result = await compiler.execCompilerCached(
            compiler.compiler.exe,
            ClangParser.getStdVersHelpOptions(filename),
            {
                ...compiler.getDefaultExecOptions(),
                createAndUseTempDir: true,
            },
        );
        if (result.stderr) {
            const lines = utils.splitLines(result.stderr);

            possible = ClangParser.extractPossibleStdvers(lines);
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
                }
                return false;
            })
            .filter(Boolean) as string[];
    }

    static override async getPossibleTargets(compiler: BaseCompiler): Promise<string[]> {
        const result = await compiler.execCompilerCached(compiler.compiler.exe, ClangParser.getTargetsHelpOptions());
        return ClangParser.extractPossibleTargets(utils.splitLines(result.stdout));
    }

    static override async getOptions(compiler: BaseCompiler, helpArg: string, populate = true, isolate = false) {
        const optionFinderWithDesc = /^ {2}?(--?[\d#+,<=>A-Z[\]a-z|-]*\s?[\d+,<=>A-Z[\]a-z|-]*)\s+([A-Z].*)/;
        const optionFinderWithoutDesc = /^ {2}?(--?[\d#+,<=>[\]a-z|-]*\s?[\d+,<=>[\]a-z|-]*)/i;
        const execOptions = {...compiler.getDefaultExecOptions()};
        if (isolate) execOptions.createAndUseTempDir = true;
        const result = await compiler.execCompilerCached(compiler.compiler.exe, helpArg.split(' '), execOptions);
        const options =
            result.code === 0
                ? ClangParser.parseLines(result.stdout + result.stderr, optionFinderWithDesc, optionFinderWithoutDesc)
                : {};
        if (populate) compiler.possibleArguments.populateOptions(options);
        return options;
    }
}

export class ClangirParser extends ClangParser {
    static override async setCompilerSettingsFromOptions(compiler: BaseCompiler, options: Record<string, Argument>) {
        ClangParser.setCompilerSettingsFromOptions(compiler, options);

        compiler.compiler.optPipeline = {
            arg: [],
            moduleScopeArg: ['-mmlir', '--mlir-print-ir-before-all', '-mmlir', '--mlir-print-ir-after-all'],
            noDiscardValueNamesArg: [],
            supportedOptions: ['demangle-symbols'],
            supportedFilters: [],
            initialOptionsState: {
                'dump-full-module': true,
                'demangle-symbols': true,
                '-fno-discard-value-names': false,
            },
            initialFiltersState: {'filter-debug-info': false, 'filter-instruction-metadata': false},
        };
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
    static override async getOptions(compiler: BaseCompiler, helpArg: string) {
        const optionFinder1 = /^ +(--?[\w#,.<=>[\]|-]*) {2,}- (.*)/i;
        const optionFinder2 = /^ +(--?[\w#,.<=>[\]|-]*)/i;
        const result = await compiler.execCompilerCached(compiler.compiler.exe, helpArg.split(' '));
        const options = result.code === 0 ? CircleParser.parseLines(result.stdout, optionFinder1, optionFinder2) : {};
        compiler.possibleArguments.populateOptions(options);
        return options;
    }

    static override async getPossibleStdvers(compiler: BaseCompiler): Promise<CompilerOverrideOptions> {
        const possible: CompilerOverrideOptions = [];
        const optionFinder = /^ {4}=([\w+]*) +- +(.*)/i;
        const result = await compiler.execCompilerCached(compiler.compiler.exe, ['--help']);
        let isInStdVerSection = false;
        for (const line of utils.splitLines(result.stdout)) {
            if (!isInStdVerSection && line.startsWith('  --std=')) {
                isInStdVerSection = true;
                continue;
            }
            if (isInStdVerSection && line.startsWith('  --')) {
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
    static override async setCompilerSettingsFromOptions(compiler: BaseCompiler, options: Record<string, Argument>) {
        if (LDCParser.hasSupport(options, '--fsave-optimization-record')) {
            compiler.compiler.optArg = '--fsave-optimization-record';
            compiler.compiler.supportsOptOutput = true;
        }

        if (LDCParser.hasSupport(options, '-fverbose-asm')) {
            compiler.compiler.supportsVerboseAsm = true;
        }

        if (LDCParser.hasSupport(options, '--print-before-all') && LDCParser.hasSupport(options, '--print-after-all')) {
            compiler.compiler.optPipeline = {
                arg: ['--print-before-all', '--print-after-all'],
                moduleScopeArg: [],
                noDiscardValueNamesArg: [],
            };
            if (LDCParser.hasSupport(options, '--print-module-scope')) {
                compiler.compiler.optPipeline.moduleScopeArg = ['--print-module-scope'];
            }
            if (LDCParser.hasSupport(options, '--fno-discard-value-names')) {
                compiler.compiler.optPipeline.noDiscardValueNamesArg = ['--fno-discard-value-names'];
            }
        }

        if (LDCParser.hasSupport(options, '--enable-color')) {
            compiler.compiler.options += ' --enable-color';
        }
    }

    static override async parse(compiler: BaseCompiler) {
        const options = await LDCParser.getOptions(compiler, '--help-hidden');
        LDCParser.setCompilerSettingsFromOptions(compiler, options);
        return compiler;
    }

    static override async getOptions(compiler: BaseCompiler, helpArg: string, populate = true) {
        const optionFinder = /^\s*(--?[\d+,<=>[\]a-z|-]*)\s*(.*)/i;
        const result = await compiler.execCompilerCached(compiler.compiler.exe, helpArg.split(' '));
        const options = result.code === 0 ? LDCParser.parseLines(result.stdout + result.stderr, optionFinder) : {};
        if (populate) {
            compiler.possibleArguments.populateOptions(options);
        }
        return options;
    }
}

export class ElixirParser extends BaseParser {
    static override async parse(compiler: BaseCompiler) {
        await ElixirParser.getOptions(compiler, '--help');
        return compiler;
    }
}

export class ErlangParser extends BaseParser {
    static override async parse(compiler: BaseCompiler) {
        await ErlangParser.getOptions(compiler, '-help');
        return compiler;
    }
}

export class PascalParser extends BaseParser {
    static override async parse(compiler: BaseCompiler) {
        await PascalParser.getOptions(compiler, '-help');
        return compiler;
    }
}

export class ICCParser extends GCCParser {
    static override async setCompilerSettingsFromOptions(compiler: BaseCompiler, options: Record<string, Argument>) {
        const keys = _.keys(options);
        if (ICCParser.hasSupport(options, '-masm=')) {
            compiler.compiler.intelAsm = '-masm=intel';
            compiler.compiler.supportsIntel = true;
        }
        if (ICCParser.hasSupport(options, '-fdiagnostics-color')) {
            if (compiler.compiler.options) compiler.compiler.options += ' ';
            compiler.compiler.options += '-fdiagnostics-color=always';
        }
        if (_.find(keys, key => key.startsWith('-fdump-'))) {
            compiler.compiler.supportsGccDump = true;
            compiler.compiler.removeEmptyGccDump = true;
        }
        if (ICCParser.hasSupportStartsWith(options, '-march=')) compiler.compiler.supportsMarch = true;
        if (ICCParser.hasSupportStartsWith(options, '--target=')) compiler.compiler.supportsTargetIs = true;
        if (ICCParser.hasSupportStartsWith(options, '--target ')) compiler.compiler.supportsTarget = true;
    }

    static extractPossibleStdvers(lines: string[]): CompilerOverrideOptions {
        const stdverRe = /-std=<std>/;
        const descRe = /^\s{12}([\w+]*)\s+(.*)/;
        const possible: CompilerOverrideOptions = [];
        let foundStdver = false;
        let skipLine = false;
        for (const line of lines) {
            if (skipLine) {
                skipLine = false;
                continue;
            }

            if (foundStdver) {
                const descMatch = line.match(descRe);
                if (descMatch) {
                    possible.push({
                        name: descMatch[1] + ': ' + descMatch[2],
                        value: descMatch[1],
                    });
                } else {
                    break;
                }
            } else {
                const match = line.match(stdverRe);
                if (match) {
                    foundStdver = true;
                    skipLine = true;
                }
            }
        }
        return possible;
    }

    static override async getPossibleStdvers(compiler: BaseCompiler): Promise<CompilerOverrideOptions> {
        const result = await compiler.execCompilerCached(compiler.compiler.exe, ['--help']);
        const lines = utils.splitLines(result.stdout);

        return ICCParser.extractPossibleStdvers(lines);
    }

    static override async parse(compiler: BaseCompiler) {
        const results = await Promise.all([ICCParser.getOptions(compiler, '-fsyntax-only --help')]);
        const options = Object.assign({}, ...results);
        await ICCParser.setCompilerSettingsFromOptions(compiler, options);
        return compiler;
    }
}

export class ISPCParser extends BaseParser {
    static override async setCompilerSettingsFromOptions(compiler: BaseCompiler, options: Record<string, Argument>) {
        if (ISPCParser.hasSupport(options, '--x86-asm-syntax')) {
            compiler.compiler.intelAsm = '--x86-asm-syntax=intel';
            compiler.compiler.supportsIntel = true;
        }
    }

    static override async parse(compiler: BaseCompiler) {
        const options = await ISPCParser.getOptions(compiler, '--help');
        await ISPCParser.setCompilerSettingsFromOptions(compiler, options);
        return compiler;
    }

    static override async getOptions(compiler: BaseCompiler, helpArg: string) {
        const result = await compiler.execCompilerCached(compiler.compiler.exe, [helpArg]);
        const optionFinder = /^\s*\[(--?[\d\s()+,/<=>a-z{|}-]*)]\s*(.*)/i;
        const options = result.code === 0 ? ISPCParser.parseLines(result.stdout + result.stderr, optionFinder) : {};
        compiler.possibleArguments.populateOptions(options);
        return options;
    }
}

export class JavaParser extends BaseParser {
    static override async parse(compiler: BaseCompiler) {
        await JavaParser.getOptions(compiler, '-help');
        return compiler;
    }
}

export class KotlinParser extends BaseParser {
    static override async parse(compiler: BaseCompiler) {
        await KotlinParser.getOptions(compiler, '-help');
        return compiler;
    }
}

export class ScalaParser extends BaseParser {
    static override async parse(compiler: BaseCompiler) {
        await ScalaParser.getOptions(compiler, '-help');
        return compiler;
    }
}

export class VCParser extends BaseParser {
    static override async parse(compiler: BaseCompiler) {
        await VCParser.getOptions(compiler, '/help');
        return compiler;
    }

    static override parseLines(stdout: string, optionRegex: RegExp) {
        let previousOption: string | false = false;
        const options: Record<string, Argument> = {};

        const matchLine = (line: string) => {
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
                col1 = line.substring(0, 39);
                col2 = line.substring(40);
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

    static override async getPossibleStdvers(compiler: BaseCompiler): Promise<CompilerOverrideOptions> {
        const result = await compiler.execCompilerCached(compiler.compiler.exe, ['/help']);
        const lines = utils.splitLines(result.stdout);

        return VCParser.extractPossibleStdvers(lines);
    }

    static override async getOptions(compiler: BaseCompiler, helpArg: string) {
        const result = await compiler.execCompilerCached(compiler.compiler.exe, [helpArg]);
        const optionFinder = /^\s*(\/[\w#+,.:<=>[\]{|}-]*)\s*(.*)/i;
        const options = result.code === 0 ? VCParser.parseLines(result.stdout, optionFinder) : {};
        compiler.possibleArguments.populateOptions(options);
        return options;
    }
}

export class RustParser extends BaseParser {
    static override async setCompilerSettingsFromOptions(compiler: BaseCompiler, options: Record<string, Argument>) {
        if (RustParser.hasSupport(options, '--color')) {
            if (compiler.compiler.options) compiler.compiler.options += ' ';
            compiler.compiler.options += '--color=always';
        }
        if (RustParser.hasSupportStartsWith(options, '--target=')) compiler.compiler.supportsTargetIs = true;
        if (RustParser.hasSupportStartsWith(options, '--target ')) compiler.compiler.supportsTarget = true;
    }

    static override async parse(compiler: BaseCompiler) {
        const results = await Promise.all([
            RustParser.getOptions(compiler, '--help'),
            RustParser.getOptions(compiler, '-C help'),
            RustParser.getOptions(compiler, '--help -v'),
        ]);
        const options = Object.assign({}, ...results);
        await RustParser.setCompilerSettingsFromOptions(compiler, options);
        return compiler;
    }

    static override async getPossibleEditions(compiler: BaseCompiler): Promise<string[]> {
        const result = await compiler.execCompilerCached(compiler.compiler.exe, ['--help']);
        const re = /--edition ([\d|]*)/;

        const match = result.stdout.match(re);
        if (match?.[1]) {
            return match[1].split('|');
        }

        return [];
    }

    static override async getPossibleTargets(compiler: BaseCompiler): Promise<string[]> {
        const result = await compiler.execCompilerCached(compiler.compiler.exe, ['--print', 'target-list']);
        return utils.splitLines(result.stdout).filter(Boolean);
    }

    static parseRustHelpLines(stdout: string) {
        let previousOption: false | string = false;
        const options: Record<string, Argument> = {};

        const doubleOptionFinder = /^\s{4}(-\w, --\w*\s?[\w:=[\]]*)\s*(.*)/i;
        const singleOptionFinder = /^\s{8}(--[\w-]*\s?[\w:=[\]|-]*)\s*(.*)/i;
        const singleComplexOptionFinder = /^\s{4}(-\w*\s?[\w:=[\]]*)\s*(.*)/i;

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
            }
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

            if (previousOption) {
                options[previousOption] = {
                    description: description,
                    timesused: 0,
                };
            }
        });

        return options;
    }

    static override async getOptions(compiler: BaseCompiler, helpArg: string) {
        const result = await compiler.execCompilerCached(compiler.compiler.exe, helpArg.split(' '));
        let options = {};
        if (result.code === 0) {
            if (helpArg === '-C help') {
                const optionFinder = /^\s*(-c\s*[\d=a-z-]*)\s--\s(.*)/i;

                options = RustParser.parseLines(result.stdout + result.stderr, optionFinder);
            } else {
                options = RustParser.parseRustHelpLines(result.stdout + result.stderr);
            }
        }
        compiler.possibleArguments.populateOptions(options);
        return options;
    }
}

export class ZksolcParser extends RustParser {
    static override async parse(compiler: BaseCompiler) {
        const options = await ZksolcParser.getOptions(compiler, '--help');
        await ZksolcParser.setCompilerSettingsFromOptions(compiler, options);
        return compiler;
    }
}

export class MrustcParser extends BaseParser {
    static override async parse(compiler: BaseCompiler) {
        await MrustcParser.getOptions(compiler, '--help');
        return compiler;
    }
}

export class NimParser extends BaseParser {
    static override async parse(compiler: BaseCompiler) {
        await NimParser.getOptions(compiler, '-help');
        return compiler;
    }
}

export class CrystalParser extends BaseParser {
    static override async parse(compiler: BaseCompiler) {
        await CrystalParser.getOptions(compiler, 'build');
        return compiler;
    }
}

export class TableGenParser extends BaseParser {
    static async getPossibleActions(compiler: BaseCompiler): Promise<CompilerOverrideOptions> {
        const result = await compiler.execCompilerCached(compiler.compiler.exe, ['--help']);
        return TableGenParser.extractPossibleActions(utils.splitLines(result.stdout));
    }

    static extractPossibleActions(lines: string[]): CompilerOverrideOptions {
        const actions: CompilerOverrideOptions = [];
        let found_actions = false;

        for (const line of lines) {
            // Action options are in a section with this header.
            if (line.includes('Action to perform:')) {
                found_actions = true;
            } else if (found_actions) {
                // Actions are indented 6 spaces. The description follows after
                // a dash, for example:
                // <6 spaces>--do-thing  - Description of thing.
                const action_match = line.match(/^ {6}(--\S+)\s+-\s+(.+)$/);
                // The end of the option section is an option indented only 2 spaces.
                if (action_match == null) {
                    break;
                }

                actions.push({
                    name: action_match[1].substring(2) + ': ' + action_match[2],
                    value: action_match[1],
                });
            }
        }

        return actions;
    }
}

export class TypeScriptNativeParser extends BaseParser {
    static override async parse(compiler: BaseCompiler) {
        await TypeScriptNativeParser.getOptions(compiler, '--help');
        return compiler;
    }
}

export class TurboCParser extends BaseParser {
    static override async parse(compiler: BaseCompiler) {
        await TurboCParser.getOptions(compiler, '');
        return compiler;
    }
}

export class ToitParser extends BaseParser {
    static override async parse(compiler: BaseCompiler) {
        await ToitParser.getOptions(compiler, '-help');
        return compiler;
    }
}

export class JuliaParser extends BaseParser {
    // Get help line from wrapper not Julia runtime
    static override async getOptions(compiler: JuliaCompiler, helpArg: string) {
        const optionFinder = /^\s*(--?[\d+,<=>[\]a-z|-]*)\s*(.*)/i;
        const result = await compiler.execCompilerCached(compiler.compiler.exe, [
            compiler.compilerWrapperPath,
            helpArg,
        ]);
        const options = result.code === 0 ? JuliaParser.parseLines(result.stdout + result.stderr, optionFinder) : {};
        compiler.possibleArguments.populateOptions(options);
        return options;
    }

    static override async parse(compiler: JuliaCompiler) {
        await JuliaParser.getOptions(compiler, '--help');
        return compiler;
    }
}

export class Z88dkParser extends BaseParser {
    static override async getPossibleTargets(compiler: BaseCompiler): Promise<string[]> {
        const configPath = path.join(path.dirname(compiler.compiler.exe), '../share/z88dk/lib/config');
        const targets: string[] = [];
        const dir = await fs.readdir(configPath);
        for (const filename of dir) {
            if (filename.toLowerCase().endsWith('.cfg')) {
                targets.push(filename.substring(0, filename.length - 4));
            }
        }
        return targets;
    }
}

export class WasmtimeParser extends BaseParser {
    static override async parse(compiler: BaseCompiler) {
        await WasmtimeParser.getOptions(compiler, '--help');
        return compiler;
    }
}

export class ZigParser extends GCCParser {
    static override async parse(compiler: BaseCompiler) {
        const results = await Promise.all([ZigParser.getOptions(compiler, 'build-obj --help')]);
        const options = Object.assign({}, ...results);
        await GCCParser.setCompilerSettingsFromOptions(compiler, options);
        if (GCCParser.hasSupportStartsWith(options, '-target ')) compiler.compiler.supportsHyphenTarget = true;
        return compiler;
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

    static override async setCompilerSettingsFromOptions(compiler: BaseCompiler, options: Record<string, Argument>) {
        ClangParser.setCompilerSettingsFromOptions(compiler, options);

        // flang does not allow -emit-llvm to be used as it is with clang
        // as -Xflang -emit-llvm. Instead you just give -emit-llvm to flang
        // directly.
        if (FlangParser.hasSupport(options, '-emit-llvm')) {
            compiler.compiler.supportsIrView = true;
            compiler.compiler.irArg = ['-emit-llvm'];
            compiler.compiler.minIrArgs = ['-emit-llvm'];
        }

        compiler.compiler.supportsIntel = true;
        compiler.compiler.intelAsm = '-masm=intel';
    }

    static override hasSupport(options: Record<string, Argument>, param: string) {
        // param is available but we get a warning, so lets not use it
        if (param === '-fcolor-diagnostics') return;

        return BaseParser.hasSupport(options, param);
    }

    static override extractPossibleStdvers(lines: string[]): CompilerOverrideOptions {
        const possible: CompilerOverrideOptions = [];
        const re1 = /error: Only -std=([\w+]*) is allowed currently./;
        for (const line of lines) {
            const match = line.match(re1);
            if (match?.[1]) {
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
    static override async parse(compiler: BaseCompiler) {
        const results = await Promise.all([GHCParser.getOptions(compiler, '--help')]);
        const options = Object.assign({}, ...results);
        await GHCParser.setCompilerSettingsFromOptions(compiler, options);
        return compiler;
    }

    static override async getOptions(compiler: BaseCompiler, helpArg: string) {
        const optionFinder1 = /^ {4}(-[\w[\]]+)\s+(.*)/i;
        const optionFinder2 = /^ {4}(-[\w[\]]+)/;
        const result = await compiler.execCompilerCached(compiler.compiler.exe, helpArg.split(' '));
        const options = result.code === 0 ? GHCParser.parseLines(result.stdout, optionFinder1, optionFinder2) : {};

        compiler.possibleArguments.populateOptions(options);
        return options;
    }
}

export class SwiftParser extends ClangParser {
    static override async parse(compiler: BaseCompiler) {
        const results = await Promise.all([SwiftParser.getOptions(compiler, '--help')]);
        const options = Object.assign({}, ...results);
        SwiftParser.setCompilerSettingsFromOptions(compiler, options);
        return compiler;
    }

    static override async getPossibleStdvers(compiler: BaseCompiler): Promise<CompilerOverrideOptions> {
        return [];
    }

    static override async getPossibleTargets(compiler: BaseCompiler): Promise<string[]> {
        return [];
    }
}

export class TendraParser extends GCCParser {
    static override async parse(compiler: BaseCompiler) {
        const results = await Promise.all([TendraParser.getOptions(compiler, '--help')]);
        const options = Object.assign({}, ...results);
        await TendraParser.setCompilerSettingsFromOptions(compiler, options);
        return compiler;
    }

    static override async getOptions(compiler: BaseCompiler, helpArg: string) {
        const optionFinder = /^ *(-[\d#+,<=>[\]a-z|-]* ?[\d+,<=>[\]a-z|-]*) : +(.*)/i;
        const result = await compiler.execCompilerCached(compiler.compiler.exe, helpArg.split(' '));
        const options = TendraParser.parseLines(result.stdout + result.stderr, optionFinder);
        compiler.possibleArguments.populateOptions(options);
        return options;
    }

    static override async getPossibleStdvers(compiler: BaseCompiler): Promise<CompilerOverrideOptions> {
        return [];
    }

    static override async getPossibleTargets(compiler: BaseCompiler): Promise<string[]> {
        return [];
    }
}

export class GolangParser extends GCCParser {
    static override getDefaultExampleFilename() {
        return 'go/default.go';
    }

    static override async parse(compiler: BaseCompiler) {
        const results = await Promise.all([
            GolangParser.getOptions(
                compiler,
                'build -o ./output.s "-gcflags=-S --help" ' + GolangParser.getExampleFilepath(),
            ),
        ]);
        const options = Object.assign({}, ...results);
        await GolangParser.setCompilerSettingsFromOptions(compiler, options);
        return compiler;
    }

    static override async getOptions(compiler: BaseCompiler, helpArg: string) {
        const optionFinder1 = /^\s*(--?[\d#+,<=>[\]a-z|-]* ?[\d+,<=>[\]a-z|-]*)\s+(.*)/i;
        const optionFinder2 = /^\s*(--?[\d#+,<=>[\]a-z|-]* ?[\d+,<=>[\]a-z|-]*)/i;
        const result = await compiler.execCompilerCached(compiler.compiler.exe, splitArguments(helpArg), {
            ...compiler.getDefaultExecOptions(),
            createAndUseTempDir: true,
        });
        const options = GolangParser.parseLines(result.stdout + result.stderr, optionFinder1, optionFinder2);
        compiler.possibleArguments.populateOptions(options);
        return options;
    }
}

export class GnuCobolParser extends GCCParser {
    static override getLanguageSpecificHelpFlags(): string[] {
        return ['--help'];
    }

    static override async getPossibleStdvers(compiler: BaseCompiler): Promise<CompilerOverrideOptions> {
        const possible: CompilerOverrideOptions = [];
        const options = await GnuCobolParser.getOptionsStrict(compiler, GnuCobolParser.getLanguageSpecificHelpFlags());
        for (const opt in options) {
            if (opt.startsWith('-std=')) {
                const vers = options[opt].description
                    .split(':')[1]
                    .split(',')
                    .map(v => v.trim());
                vers[vers.length - 1] = vers[vers.length - 1].split(';')[0];
                for (const ver of vers) {
                    possible.push({
                        name: ver,
                        value: ver,
                    });
                }
                break;
            }
        }
        return possible;
    }
}

export class MadpascalParser extends GCCParser {
    static override async parse(compiler: BaseCompiler) {
        const results = await Promise.all([MadpascalParser.getOptions(compiler, '')]);
        const options = Object.assign({}, ...results);
        await MadpascalParser.setCompilerSettingsFromOptions(compiler, options);
        return compiler;
    }

    static override async getOptions(compiler: BaseCompiler, helpArg: string) {
        const optionFinder = /^(-[\w:<>]*) *(.*)/i;
        const result = await compiler.execCompilerCached(compiler.compiler.exe, []);
        const options = MadpascalParser.parseLines(result.stdout + result.stderr, optionFinder);
        compiler.possibleArguments.populateOptions(options);
        return options;
    }

    static override async getPossibleStdvers(compiler: BaseCompiler): Promise<CompilerOverrideOptions> {
        return [];
    }

    static override async getPossibleTargets(compiler: BaseCompiler): Promise<string[]> {
        return ['a8', 'c64', 'c4p', 'raw', 'neo'];
    }
}

export class GlslangParser extends BaseParser {
    static override async parse(compiler: BaseCompiler) {
        await GlslangParser.getOptions(compiler, '--help');
        return compiler;
    }

    static override async getOptions(compiler: BaseCompiler, helpArg: string) {
        const optionFinder1 = /^ *(--?[\d#+,<=>[\]a-z|-]* ?[\d+,<=>[\]a-z|-]*) {2,}(.*)/i;
        const optionFinder2 = /^ *(--?[\d#+,<=>[\]a-z|-]* ?[\d+,<=>[\]a-z|-]*)/i;
        const result = await compiler.execCompilerCached(compiler.compiler.exe, [helpArg]);
        // glslang will return a return code of 1 when calling --help (since it means nothing was compiled)
        const options = GlslangParser.parseLines(result.stdout + result.stderr, optionFinder1, optionFinder2);
        compiler.possibleArguments.populateOptions(options);
        return options;
    }
}
