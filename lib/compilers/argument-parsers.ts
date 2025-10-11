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

import fs from 'node:fs/promises';
import path from 'node:path';
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
    protected readonly compiler: BaseCompiler;

    constructor(compiler: BaseCompiler) {
        this.compiler = compiler;
    }
    async setCompilerSettingsFromOptions(options: Record<string, Argument>) {}

    hasSupport(options: Record<string, Argument>, forOption: string) {
        return _.keys(options).find(option => option.includes(forOption));
    }

    hasSupportStartsWith(options: Record<string, Argument>, forOption: string) {
        return _.keys(options).find(option => option.startsWith(forOption));
    }

    parseLines(stdout: string, optionWithDescRegex: RegExp, optionWithoutDescRegex?: RegExp) {
        let previousOption: false | string = false;
        const options: Record<string, Argument> = {};

        utils.eachLine(stdout, line => {
            const match1 = line.match(optionWithDescRegex);
            if (match1?.[1] && match1[2]) {
                previousOption = match1[1].trim();
                if (previousOption) {
                    options[previousOption] = {
                        description: this.spaceCompress(match1[2].trim()),
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

                options[previousOption].description = this.spaceCompress(options[previousOption].description);
            } else {
                previousOption = false;
            }
        });

        return options;
    }

    spaceCompress(text: string): string {
        return text.replaceAll('  ', ' ');
    }

    async getPossibleTargets(): Promise<string[]> {
        return [];
    }

    async getPossibleStdvers(): Promise<CompilerOverrideOptions> {
        return [];
    }

    // Currently used only for Rust
    async getPossibleEditions(): Promise<string[]> {
        return [];
    }

    // Currently used only for TableGen
    async getPossibleActions(): Promise<CompilerOverrideOptions> {
        return [];
    }

    async getOptions(helpArg: string) {
        const optionFinder1 = /^ *(--?[\d#+,<=>[\]a-z|-]* ?[\d+,<=>[\]a-z|-]*) {2,}(.*)/i;
        const optionFinder2 = /^ *(--?[\d#+,<=>[\]a-z|-]* ?[\d+,<=>[\]a-z|-]*)/i;
        const result = await this.compiler.execCompilerCached(this.compiler.compiler.exe, [helpArg]);
        const options =
            result.code === 0 ? this.parseLines(result.stdout + result.stderr, optionFinder1, optionFinder2) : {};
        this.compiler.possibleArguments.populateOptions(options);
        return options;
    }

    // async for compatibility with children, who call getOptions
    async parse() {
        return this.compiler;
    }
}

export class GCCParser extends BaseParser {
    async checkAndSetMasmIntelIfSupported() {
        // -masm= may be available but unsupported by the compiler.
        const res = await this.compiler.execCompilerCached(this.compiler.compiler.exe, [
            '-fsyntax-only',
            '--target-help',
            '-masm=intel',
        ]);
        if (res.code === 0) {
            this.compiler.compiler.intelAsm = '-masm=intel';
            this.compiler.compiler.supportsIntel = true;
        }
    }

    override async setCompilerSettingsFromOptions(options: Record<string, Argument>) {
        const keys = _.keys(options);
        logger.debug(`gcc-like compiler options: ${keys.join(' ')}`);
        if (this.hasSupport(options, '-masm=')) {
            await this.checkAndSetMasmIntelIfSupported();
        }
        if (this.hasSupport(options, '-fstack-usage')) {
            this.compiler.compiler.stackUsageArg = '-fstack-usage';
            this.compiler.compiler.supportsStackUsageOutput = true;
        }
        if (this.hasSupport(options, '-fdiagnostics-color')) {
            if (this.compiler.compiler.options) this.compiler.compiler.options += ' ';
            this.compiler.compiler.options += '-fdiagnostics-color=always';
        }
        if (this.hasSupport(options, '-fverbose-asm')) {
            this.compiler.compiler.supportsVerboseAsm = true;
        }
        if (this.hasSupport(options, '-fopt-info')) {
            this.compiler.compiler.optArg = '-fopt-info-all=all.opt';
            this.compiler.compiler.supportsOptOutput = true;
        }
        // This check is not infallible, but takes care of Rust and Swift being picked up :)
        if (_.find(keys, key => key.startsWith('-fdump-'))) {
            this.compiler.compiler.supportsGccDump = true;

            // By default, consider the compiler to be a regular GCC (eg. gcc,
            // g++) and do the extra work of filtering out enabled pass that did
            // not produce anything.
            this.compiler.compiler.removeEmptyGccDump = true;
        }
        if (this.hasSupportStartsWith(options, '-march=')) this.compiler.compiler.supportsMarch = true;
        if (this.hasSupportStartsWith(options, '--target=')) this.compiler.compiler.supportsTargetIs = true;
        if (this.hasSupportStartsWith(options, '--target ')) this.compiler.compiler.supportsTarget = true;
    }

    override async parse() {
        const results = await Promise.all([
            this.getOptions('-fsyntax-only --help'),
            this.getOptions('-fsyntax-only --target-help'),
            this.getOptions('-fsyntax-only --help=common'),
            this.getOptions('-fsyntax-only --help=warnings'),
            this.getOptions('-fsyntax-only --help=optimizers'),
            this.getOptions('-fsyntax-only --help=target'),
        ]);
        const options = Object.assign({}, ...results);
        await this.setCompilerSettingsFromOptions(options);
        return this.compiler;
    }

    override async getPossibleTargets(): Promise<string[]> {
        const re = /Known valid arguments for -march= option:\s+(.*)/;
        const result = await this.compiler.execCompilerCached(this.compiler.compiler.exe, [
            '-fsyntax-only',
            '--target-help',
        ]);
        const match = result.stdout.match(re);
        if (match) {
            return match[1].split(' ');
        }
        return [];
    }

    getLanguageSpecificHelpFlags(): string[] {
        return ['-fsyntax-only', '--help=c++'];
    }

    override async getPossibleStdvers(): Promise<CompilerOverrideOptions> {
        const possible: CompilerOverrideOptions = [];
        const options = await this.getOptionsStrict(this.getLanguageSpecificHelpFlags());
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

    override async getOptions(helpArg: string) {
        const optionFinder1 = /^ *(--?[\d#+,<=>[\]a-z|-]* ?[\d+,<=>[\]a-z|-]*) {2,}(.*)/i;
        const optionFinder2 = /^ *(--?[\d#+,<=>[\]a-z|-]* ?[\d+,<=>[\]a-z|-]*)/i;
        const result = await this.compiler.execCompilerCached(this.compiler.compiler.exe, splitArguments(helpArg));
        const options =
            result.code === 0 ? this.parseLines(result.stdout + result.stderr, optionFinder1, optionFinder2) : {};
        this.compiler.possibleArguments.populateOptions(options);
        return options;
    }

    async getOptionsStrict(helpArgs: string[]) {
        const optionFinder = /^ {2}(--?[\d+,<=>[\]a-z|-]*) *(.*)/i;
        const result = await this.compiler.execCompilerCached(this.compiler.compiler.exe, helpArgs);
        return result.code === 0 ? this.parseLines(result.stdout + result.stderr, optionFinder) : {};
    }
}

export class ClangParser extends BaseParser {
    mllvmOptions = new Set<string>();

    override async setCompilerSettingsFromOptions(options: Record<string, Argument>) {
        const keys = _.keys(options);
        logger.debug(`clang-like compiler options: ${keys.join(' ')}`);

        if (keys.length === 0) {
            logger.error(`compiler options appear empty for ${this.compiler.compiler.id}`);
        }

        if (this.hasSupport(options, '-fsave-optimization-record')) {
            this.compiler.compiler.optArg = '-fsave-optimization-record';
            this.compiler.compiler.supportsOptOutput = true;
        }
        if (this.hasSupport(options, '-fstack-usage')) {
            this.compiler.compiler.stackUsageArg = '-fstack-usage';
            this.compiler.compiler.supportsStackUsageOutput = true;
        }
        if (this.hasSupport(options, '-fverbose-asm')) {
            this.compiler.compiler.supportsVerboseAsm = true;
        }

        if (this.hasSupport(options, '-emit-llvm')) {
            this.compiler.compiler.supportsIrView = true;
            this.compiler.compiler.irArg = ['-Xclang', '-emit-llvm', '-fsyntax-only'];
            this.compiler.compiler.minIrArgs = ['-emit-llvm'];
        }

        // if (this.hasSupport(options, '-emit-cir')) {
        // #7265: clang-trunk supposedly has '-emit-cir', but it's not doing much. Checking explicitly
        // for clangir in the compiler name instead.
        if (this.compiler.compiler.name?.includes('clangir')) {
            this.compiler.compiler.supportsClangirView = true;
        }

        if (
            this.hasSupport(options, '-mllvm') &&
            this.mllvmOptions.has('--print-before-all') &&
            this.mllvmOptions.has('--print-after-all')
        ) {
            this.compiler.compiler.optPipeline = {
                arg: ['-mllvm', '--print-before-all', '-mllvm', '--print-after-all'],
                moduleScopeArg: [],
                noDiscardValueNamesArg: [],
            };
            if (this.mllvmOptions.has('--print-module-scope')) {
                this.compiler.compiler.optPipeline.moduleScopeArg = ['-mllvm', '-print-module-scope'];
            }
            if (this.hasSupport(options, '-fno-discard-value-names')) {
                this.compiler.compiler.optPipeline.noDiscardValueNamesArg = ['-fno-discard-value-names'];
            }
        }

        if (this.hasSupport(options, '-fcolor-diagnostics')) this.compiler.compiler.options += ' -fcolor-diagnostics';
        if (this.hasSupport(options, '-fno-crash-diagnostics'))
            this.compiler.compiler.options += ' -fno-crash-diagnostics';

        if (this.hasSupportStartsWith(options, '--target=')) this.compiler.compiler.supportsTargetIs = true;
        if (this.hasSupportStartsWith(options, '--target ')) this.compiler.compiler.supportsTarget = true;
    }

    getMainHelpOptions(): string[] {
        return ['--help'];
    }

    getHiddenHelpOptions(): string[] {
        return ['-mllvm', '--help-list-hidden', '-x', 'c++', '/dev/null', '-c'];
    }

    getStdVersHelpOptions(): string[] {
        return ['-std=c++9999999', '-x', 'c++', '/dev/null', '-c'];
    }

    getTargetsHelpOptions(): string[] {
        return ['--print-targets'];
    }

    override async parse() {
        try {
            const options = await this.getOptions(this.getMainHelpOptions().join(' '));

            this.mllvmOptions = new Set(
                _.keys(await this.getOptions(this.getHiddenHelpOptions().join(' '), false, true)),
            );
            await this.setCompilerSettingsFromOptions(options);
        } catch (error) {
            const err = `Error while trying to generate llvm backend arguments for ${this.compiler.compiler.id}: ${error}`;
            logger.error(err);
            Sentry.captureMessage(err);
        }
        return this.compiler;
    }

    getRegexMatchesAsStdver(match: RegExpMatchArray | null, maxToMatch: number): CompilerOverrideOptions {
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

    extractPossibleStdvers(lines: string[]): CompilerOverrideOptions {
        const possible: CompilerOverrideOptions = [];
        const re1 = /note: use '([\w+:]*)' for '(.*)' standard/;
        const re2 = /note: use '([\w+:]*)' or '([\w+:]*)' for '(.*)' standard/;
        const re3 = /note: use '([\w+:]*)', '([\w+:]*)', or '([\w+:]*)' for '(.*)' standard/;
        const re4 = /note: use '([\w+:]*)', '([\w+:]*)', '([\w+:]*)', or '([\w+:]*)' for '(.*)' standard/;
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

    override async getPossibleStdvers(): Promise<CompilerOverrideOptions> {
        let possible: CompilerOverrideOptions = [];

        const result = await this.compiler.execCompilerCached(
            this.compiler.compiler.exe,
            this.getStdVersHelpOptions(),
            {
                ...this.compiler.getDefaultExecOptions(),
                createAndUseTempDir: true,
            },
        );
        if (result.stderr) {
            const lines = utils.splitLines(result.stderr);

            possible = this.extractPossibleStdvers(lines);
            possible.sort((a, b) => {
                return a.value === b.value ? 0 : a.value > b.value ? 1 : -1;
            });
        }

        return possible;
    }

    extractPossibleTargets(lines: string[]): string[] {
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

    override async getPossibleTargets(): Promise<string[]> {
        const result = await this.compiler.execCompilerCached(this.compiler.compiler.exe, this.getTargetsHelpOptions());
        return this.extractPossibleTargets(utils.splitLines(result.stdout));
    }

    override async getOptions(helpArg: string, populate = true, isolate = false) {
        const optionFinderWithDesc = /^ {2}?(--?[\d#+,<=>A-Z[\]a-z|-]*\s?[\d+,<=>A-Z[\]a-z|-]*)\s+([A-Z].*)/;
        const optionFinderWithoutDesc = /^ {2}?(--?[\d#+,<=>[\]a-z|-]*\s?[\d+,<=>[\]a-z|-]*)/i;
        const execOptions = {...this.compiler.getDefaultExecOptions()};
        if (isolate) execOptions.createAndUseTempDir = true;
        const result = await this.compiler.execCompilerCached(
            this.compiler.compiler.exe,
            splitArguments(helpArg),
            execOptions,
        );
        const options =
            result.code === 0
                ? this.parseLines(result.stdout + result.stderr, optionFinderWithDesc, optionFinderWithoutDesc)
                : {};
        if (populate) this.compiler.possibleArguments.populateOptions(options);
        return options;
    }
}

export class ClangirParser extends ClangParser {
    override async setCompilerSettingsFromOptions(options: Record<string, Argument>) {
        await super.setCompilerSettingsFromOptions(options);

        this.compiler.compiler.optPipeline = {
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
    override getLanguageSpecificHelpFlags(): string[] {
        return ['-fsyntax-only', '--help=c'];
    }
}

export class ClangCParser extends ClangParser {
    override getStdVersHelpOptions(): string[] {
        return ['-std=c9999999', '-x', 'c', '/dev/null', '-c'];
    }
}

export class CircleParser extends ClangParser {
    override async getOptions(helpArg: string) {
        const optionFinder1 = /^ +(--?[\w#,.<=>[\]|-]*) {2,}- (.*)/i;
        const optionFinder2 = /^ +(--?[\w#,.<=>[\]|-]*)/i;
        const result = await this.compiler.execCompilerCached(this.compiler.compiler.exe, splitArguments(helpArg));
        const options = result.code === 0 ? this.parseLines(result.stdout, optionFinder1, optionFinder2) : {};
        this.compiler.possibleArguments.populateOptions(options);
        return options;
    }

    override async getPossibleStdvers(): Promise<CompilerOverrideOptions> {
        const possible: CompilerOverrideOptions = [];
        const optionFinder = /^ {4}=([\w+]*) +- +(.*)/i;
        const result = await this.compiler.execCompilerCached(this.compiler.compiler.exe, ['--help']);
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
    override async setCompilerSettingsFromOptions(options: Record<string, Argument>) {
        if (this.hasSupport(options, '--fsave-optimization-record')) {
            this.compiler.compiler.optArg = '--fsave-optimization-record';
            this.compiler.compiler.supportsOptOutput = true;
        }

        if (this.hasSupport(options, '-fverbose-asm')) {
            this.compiler.compiler.supportsVerboseAsm = true;
        }

        if (this.hasSupport(options, '--print-before-all') && this.hasSupport(options, '--print-after-all')) {
            this.compiler.compiler.optPipeline = {
                arg: ['--print-before-all', '--print-after-all'],
                moduleScopeArg: [],
                noDiscardValueNamesArg: [],
            };
            if (this.hasSupport(options, '--print-module-scope')) {
                this.compiler.compiler.optPipeline.moduleScopeArg = ['--print-module-scope'];
            }
            if (this.hasSupport(options, '--fno-discard-value-names')) {
                this.compiler.compiler.optPipeline.noDiscardValueNamesArg = ['--fno-discard-value-names'];
            }
        }

        if (this.hasSupport(options, '--enable-color')) {
            this.compiler.compiler.options += ' --enable-color';
        }
    }

    override async parse() {
        const options = await this.getOptions('--help-hidden');
        await this.setCompilerSettingsFromOptions(options);
        return this.compiler;
    }

    override async getOptions(helpArg: string, populate = true) {
        const optionFinder = /^\s*(--?[\d+,<=>[\]a-z|-]*)\s*(.*)/i;
        const result = await this.compiler.execCompilerCached(this.compiler.compiler.exe, splitArguments(helpArg));
        const options = result.code === 0 ? this.parseLines(result.stdout + result.stderr, optionFinder) : {};
        if (populate) {
            this.compiler.possibleArguments.populateOptions(options);
        }
        return options;
    }
}

export class ElixirParser extends BaseParser {
    override async parse() {
        await this.getOptions('--help');
        return this.compiler;
    }
}

export class ErlangParser extends BaseParser {
    override async parse() {
        await this.getOptions('-help');
        return this.compiler;
    }
}

export class PascalParser extends BaseParser {
    override async parse() {
        await this.getOptions('-help');
        return this.compiler;
    }
}

export class MojoParser extends BaseParser {
    override async parse() {
        await this.getOptions('-help');
        return this.compiler;
    }
}

export class ICCParser extends GCCParser {
    override async setCompilerSettingsFromOptions(options: Record<string, Argument>) {
        const keys = _.keys(options);
        if (this.hasSupport(options, '-masm=')) {
            this.compiler.compiler.intelAsm = '-masm=intel';
            this.compiler.compiler.supportsIntel = true;
        }
        if (this.hasSupport(options, '-fdiagnostics-color')) {
            if (this.compiler.compiler.options) this.compiler.compiler.options += ' ';
            this.compiler.compiler.options += '-fdiagnostics-color=always';
        }
        if (_.find(keys, key => key.startsWith('-fdump-'))) {
            this.compiler.compiler.supportsGccDump = true;
            this.compiler.compiler.removeEmptyGccDump = true;
        }
        if (this.hasSupportStartsWith(options, '-march=')) this.compiler.compiler.supportsMarch = true;
        if (this.hasSupportStartsWith(options, '--target=')) this.compiler.compiler.supportsTargetIs = true;
        if (this.hasSupportStartsWith(options, '--target ')) this.compiler.compiler.supportsTarget = true;
    }

    extractPossibleStdvers(lines: string[]): CompilerOverrideOptions {
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

    override async getPossibleStdvers(): Promise<CompilerOverrideOptions> {
        const result = await this.compiler.execCompilerCached(this.compiler.compiler.exe, ['--help']);
        const lines = utils.splitLines(result.stdout);

        return this.extractPossibleStdvers(lines);
    }

    override async parse() {
        const results = await Promise.all([this.getOptions('-fsyntax-only --help')]);
        const options = Object.assign({}, ...results);
        await this.setCompilerSettingsFromOptions(options);
        return this.compiler;
    }
}

export class ISPCParser extends BaseParser {
    override async setCompilerSettingsFromOptions(options: Record<string, Argument>) {
        if (this.hasSupport(options, '--x86-asm-syntax')) {
            this.compiler.compiler.intelAsm = '--x86-asm-syntax=intel';
            this.compiler.compiler.supportsIntel = true;
        }
    }

    override async parse() {
        const options = await this.getOptions('--help');
        await this.setCompilerSettingsFromOptions(options);
        return this.compiler;
    }

    override async getOptions(helpArg: string) {
        const result = await this.compiler.execCompilerCached(this.compiler.compiler.exe, [helpArg]);
        const optionFinder = /^\s*\[(--?[\d\s()+,/<=>a-z{|}-]*)]\s*(.*)/i;
        const options = result.code === 0 ? this.parseLines(result.stdout + result.stderr, optionFinder) : {};
        this.compiler.possibleArguments.populateOptions(options);
        return options;
    }
}

export class JavaParser extends BaseParser {
    override async parse() {
        await this.getOptions('-help');
        return this.compiler;
    }
}

export class KotlinParser extends BaseParser {
    override async parse() {
        await this.getOptions('-help');
        return this.compiler;
    }
}

export class ScalaParser extends BaseParser {
    override async parse() {
        await this.getOptions('-help');
        return this.compiler;
    }
}

export class VCParser extends BaseParser {
    override async parse() {
        await this.getOptions('/help');
        return this.compiler;
    }

    override parseLines(stdout: string, optionRegex: RegExp) {
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

    extractPossibleStdvers(lines: string[]): CompilerOverrideOptions {
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

    override async getPossibleStdvers(): Promise<CompilerOverrideOptions> {
        const result = await this.compiler.execCompilerCached(this.compiler.compiler.exe, ['/help']);
        const lines = utils.splitLines(result.stdout);

        return this.extractPossibleStdvers(lines);
    }

    override async getOptions(helpArg: string) {
        const result = await this.compiler.execCompilerCached(this.compiler.compiler.exe, [helpArg]);
        const optionFinder = /^\s*(\/[\w#+,.:<=>[\]{|}-]*)\s*(.*)/i;
        const options = result.code === 0 ? this.parseLines(result.stdout, optionFinder) : {};
        this.compiler.possibleArguments.populateOptions(options);
        return options;
    }
}

export class RustParser extends BaseParser {
    override async setCompilerSettingsFromOptions(options: Record<string, Argument>) {
        if (this.hasSupport(options, '--color')) {
            if (this.compiler.compiler.options) this.compiler.compiler.options += ' ';
            this.compiler.compiler.options += '--color=always';
        }
        if (this.hasSupportStartsWith(options, '--target=')) this.compiler.compiler.supportsTargetIs = true;
        if (this.hasSupportStartsWith(options, '--target ')) this.compiler.compiler.supportsTarget = true;
    }

    override async parse() {
        const results = await Promise.all([
            this.getOptions('--help'),
            this.getOptions('-C help'),
            this.getOptions('--help -v'),
        ]);
        const options = Object.assign({}, ...results);
        await this.setCompilerSettingsFromOptions(options);
        return this.compiler;
    }

    override async getPossibleEditions(): Promise<string[]> {
        const result = await this.compiler.execCompilerCached(this.compiler.compiler.exe, ['--help', '-v']);
        const re = /--edition <?([\w|]*)>?/;

        const match = result.stdout.match(re);
        if (match?.[1]) {
            return match[1].split('|');
        }

        return [];
    }

    override async getPossibleTargets(): Promise<string[]> {
        const result = await this.compiler.execCompilerCached(this.compiler.compiler.exe, ['--print', 'target-list']);
        return utils.splitLines(result.stdout).filter(Boolean);
    }

    parseRustHelpLines(stdout: string) {
        let previousOption: false | string = false;
        const options: Record<string, Argument> = {};

        const doubleOptionFinder = /^\s{4}(-\w, --\w*\s?[\w:=[\]<>]*)\s*(.*)/i;
        const singleOptionFinder = /^\s{8}(--[\w-]*\s?[\w:=[\]|<>-]*)\s*(.*)/i;
        const singleComplexOptionFinder = /^\s{4}(-\w*\s?[\w:=[\]<>]*)\s*(.*)/i;

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

    override async getOptions(helpArg: string) {
        const result = await this.compiler.execCompilerCached(this.compiler.compiler.exe, splitArguments(helpArg));
        let options = {};
        if (result.code === 0) {
            if (helpArg === '-C help') {
                const optionFinder = /^\s*(-c\s*[\d=a-z-]*)\s--\s(.*)/i;

                options = this.parseLines(result.stdout + result.stderr, optionFinder);
            } else {
                options = this.parseRustHelpLines(result.stdout + result.stderr);
            }
        }
        this.compiler.possibleArguments.populateOptions(options);
        return options;
    }
}

export class ZksolcParser extends RustParser {
    override async parse() {
        const options = await this.getOptions('--help');
        await this.setCompilerSettingsFromOptions(options);
        return this.compiler;
    }
}

export class SolxParser extends RustParser {
    override async parse() {
        const options = await this.getOptions('--help');
        await this.setCompilerSettingsFromOptions(options);
        return this.compiler;
    }
}

export class ResolcParser extends BaseParser {
    override async parse() {
        await this.getOptions('--help');
        return this.compiler;
    }
}

export class MrustcParser extends BaseParser {
    override async parse() {
        await this.getOptions('--help');
        return this.compiler;
    }
}

export class C2RustParser extends BaseParser {
    override async parse() {
        await this.getOptions('--help');
        return this.compiler;
    }
}

export class NimParser extends BaseParser {
    override async parse() {
        await this.getOptions('-help');
        return this.compiler;
    }
}

export class CrystalParser extends BaseParser {
    override async parse() {
        await this.getOptions('build');
        return this.compiler;
    }
}

export class TableGenParser extends BaseParser {
    override async getPossibleActions(): Promise<CompilerOverrideOptions> {
        const result = await this.compiler.execCompilerCached(this.compiler.compiler.exe, ['--help']);
        return this.extractPossibleActions(utils.splitLines(result.stdout));
    }

    extractPossibleActions(lines: string[]): CompilerOverrideOptions {
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
    override async parse() {
        await this.getOptions('--help');
        return this.compiler;
    }
}

export class TurboCParser extends BaseParser {
    override async parse() {
        await this.getOptions('');
        return this.compiler;
    }
}

export class ToitParser extends BaseParser {
    override async parse() {
        await this.getOptions('-help');
        return this.compiler;
    }
}

export class JuliaParser extends BaseParser {
    // Get help line from wrapper not Julia runtime
    override async getOptions(helpArg: string) {
        const optionFinder = /^\s*(--?[\d+,<=>[\]a-z|-]*)\s*(.*)/i;
        const juliaCompiler = this.compiler as JuliaCompiler;
        const result = await this.compiler.execCompilerCached(this.compiler.compiler.exe, [
            juliaCompiler.compilerWrapperPath,
            helpArg,
        ]);
        const options = result.code === 0 ? this.parseLines(result.stdout + result.stderr, optionFinder) : {};
        this.compiler.possibleArguments.populateOptions(options);
        return options;
    }

    override async parse() {
        await this.getOptions('--help');
        return this.compiler;
    }
}

export class Z88dkParser extends BaseParser {
    override async getPossibleTargets(): Promise<string[]> {
        const configPath = path.join(path.dirname(this.compiler.compiler.exe), '../share/z88dk/lib/config');
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
    override async parse() {
        await this.getOptions('--help');
        return this.compiler;
    }
}

export class ZigParser extends GCCParser {
    override async parse() {
        const results = await Promise.all([this.getOptions('build-obj --help')]);
        const options = Object.assign({}, ...results);
        await this.setCompilerSettingsFromOptions(options);
        if (this.hasSupportStartsWith(options, '-target ')) this.compiler.compiler.supportsHyphenTarget = true;
        return this.compiler;
    }
}

export class ZigCxxParser extends ClangParser {
    override getMainHelpOptions(): string[] {
        return ['c++', '--help'];
    }

    override getHiddenHelpOptions(): string[] {
        return ['c++', '-mllvm', '--help-list-hidden', '-x', 'c++', '/dev/null', '-S', '-o', '/tmp/output.s'];
    }

    override getStdVersHelpOptions(): string[] {
        return ['c++', '-std=c++9999999', '-x', 'c++', '/dev/null', '-S', '-o', '/tmp/output.s'];
    }

    override getTargetsHelpOptions(): string[] {
        return ['c++', '--print-targets'];
    }
}

export class GccFortranParser extends GCCParser {
    override getLanguageSpecificHelpFlags(): string[] {
        return ['-fsyntax-only', '--help=fortran'];
    }
}

export class FlangParser extends ClangParser {
    override async setCompilerSettingsFromOptions(options: Record<string, Argument>) {
        await super.setCompilerSettingsFromOptions(options);

        // flang does not allow -emit-llvm to be used as it is with clang
        // as -Xflang -emit-llvm. Instead you just give -emit-llvm to flang
        // directly.
        if (this.hasSupport(options, '-emit-llvm')) {
            this.compiler.compiler.supportsIrView = true;
            this.compiler.compiler.irArg = ['-emit-llvm'];
            this.compiler.compiler.minIrArgs = ['-emit-llvm'];
        }

        this.compiler.compiler.supportsIntel = true;
        this.compiler.compiler.intelAsm = '-masm=intel';
    }

    override hasSupport(options: Record<string, Argument>, param: string) {
        // param is available but we get a warning, so lets not use it
        if (param === '-fcolor-diagnostics') return;

        return super.hasSupport(options, param);
    }

    override extractPossibleStdvers(lines: string[]): CompilerOverrideOptions {
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
    override async parse() {
        const results = await Promise.all([this.getOptions('--help')]);
        const options = Object.assign({}, ...results);
        await this.setCompilerSettingsFromOptions(options);
        return this.compiler;
    }

    override async getOptions(helpArg: string) {
        const optionFinder1 = /^ {4}(-[\w[\]]+)\s+(.*)/i;
        const optionFinder2 = /^ {4}(-[\w[\]]+)/;
        const result = await this.compiler.execCompilerCached(this.compiler.compiler.exe, splitArguments(helpArg));
        const options = result.code === 0 ? this.parseLines(result.stdout, optionFinder1, optionFinder2) : {};

        this.compiler.possibleArguments.populateOptions(options);
        return options;
    }
}

export class SwiftParser extends ClangParser {
    override async parse() {
        const results = await Promise.all([this.getOptions('--help')]);
        const options = Object.assign({}, ...results);
        await this.setCompilerSettingsFromOptions(options);
        return this.compiler;
    }

    override async getPossibleStdvers(): Promise<CompilerOverrideOptions> {
        return [];
    }

    override async getPossibleTargets(): Promise<string[]> {
        return [];
    }
}

export class TendraParser extends GCCParser {
    override async parse() {
        const results = await Promise.all([this.getOptions('--help')]);
        const options = Object.assign({}, ...results);
        await this.setCompilerSettingsFromOptions(options);
        return this.compiler;
    }

    override async getOptions(helpArg: string) {
        const optionFinder = /^ *(-[\d#+,<=>[\]a-z|-]* ?[\d+,<=>[\]a-z|-]*) : +(.*)/i;
        const result = await this.compiler.execCompilerCached(this.compiler.compiler.exe, splitArguments(helpArg));
        const options = this.parseLines(result.stdout + result.stderr, optionFinder);
        this.compiler.possibleArguments.populateOptions(options);
        return options;
    }

    override async getPossibleStdvers(): Promise<CompilerOverrideOptions> {
        return [];
    }

    override async getPossibleTargets(): Promise<string[]> {
        return [];
    }
}

export class GolangParser extends GCCParser {
    override async parse() {
        // NB this file _must_ be visible to the jail, if you're using one. This may bite on a local install when your
        // example path may not match paths available in the jail (e.g. `/infra/.deploy/examples`)
        // TODO: find a way to invoke GoLang without needing a real example Go file.
        const examplesRoot = props.get<string>('builtin', 'sourcePath', './examples/');
        const exampleFilepath = path.resolve(path.join(examplesRoot, 'go/default.go'));
        const results = await Promise.all([
            this.getOptions('build -o /tmp/output.s "-gcflags=-S --help" ' + exampleFilepath),
        ]);
        const options = Object.assign({}, ...results);
        await this.setCompilerSettingsFromOptions(options);
        return this.compiler;
    }

    override async getOptions(helpArg: string) {
        const optionFinder1 = /^\s*(--?[\d#+,<=>[\]a-z|-]* ?[\d+,<=>[\]a-z|-]*)\s+(.*)/i;
        const optionFinder2 = /^\s*(--?[\d#+,<=>[\]a-z|-]* ?[\d+,<=>[\]a-z|-]*)/i;
        const result = await this.compiler.execCompilerCached(this.compiler.compiler.exe, splitArguments(helpArg), {
            ...this.compiler.getDefaultExecOptions(),
            createAndUseTempDir: true,
        });
        const options = this.parseLines(result.stdout + result.stderr, optionFinder1, optionFinder2);
        this.compiler.possibleArguments.populateOptions(options);
        return options;
    }
}

export class GnuCobolParser extends GCCParser {
    override getLanguageSpecificHelpFlags(): string[] {
        return ['--help'];
    }

    override async getPossibleStdvers(): Promise<CompilerOverrideOptions> {
        const possible: CompilerOverrideOptions = [];
        const options = await this.getOptionsStrict(this.getLanguageSpecificHelpFlags());
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
    override async parse() {
        const results = await Promise.all([this.getOptions('')]);
        const options = Object.assign({}, ...results);
        await this.setCompilerSettingsFromOptions(options);
        return this.compiler;
    }

    override async getOptions(helpArg: string) {
        const optionFinder = /^(-[\w:<>]*) *(.*)/i;
        const result = await this.compiler.execCompilerCached(this.compiler.compiler.exe, []);
        const options = this.parseLines(result.stdout + result.stderr, optionFinder);
        this.compiler.possibleArguments.populateOptions(options);
        return options;
    }

    override async getPossibleStdvers(): Promise<CompilerOverrideOptions> {
        return [];
    }

    override async getPossibleTargets(): Promise<string[]> {
        return ['a8', 'c64', 'c4p', 'raw', 'neo'];
    }
}

export class GlslangParser extends BaseParser {
    override async parse() {
        await this.getOptions('--help');
        return this.compiler;
    }

    override async getOptions(helpArg: string) {
        const optionFinder1 = /^ *(--?[\d#+,<=>[\]a-z|-]* ?[\d+,<=>[\]a-z|-]*) {2,}(.*)/i;
        const optionFinder2 = /^ *(--?[\d#+,<=>[\]a-z|-]* ?[\d+,<=>[\]a-z|-]*)/i;
        const result = await this.compiler.execCompilerCached(this.compiler.compiler.exe, [helpArg]);
        // glslang will return a return code of 1 when calling --help (since it means nothing was compiled)
        const options = this.parseLines(result.stdout + result.stderr, optionFinder1, optionFinder2);
        this.compiler.possibleArguments.populateOptions(options);
        return options;
    }
}
