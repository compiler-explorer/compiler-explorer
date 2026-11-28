// Copyright (c) 2026, Compiler Explorer Authors
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

import type {
    CompilationResult,
    ExecutionOptionsWithEnv,
    FiledataPair,
} from '../../types/compilation/compilation.interfaces.js';
import type {BasicExecutionResult, UnprocessedExecResult} from '../../types/execution/execution.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import type {ResultLine} from '../../types/resultline/resultline.interfaces.js';
import {BaseCompiler, c_value_placeholder} from '../base-compiler.js';
import * as utils from '../utils.js';

const forbiddenSubcommands = new Set([
    'build',
    'bump',
    'bundle',
    'check',
    'deps',
    'dev',
    'docs',
    'experimental-lsp',
    'fmt',
    'glue',
    'help',
    'install',
    'licenses',
    'loc',
    'repl',
    'run',
    'test',
    'unbundle',
    'version',
]);

const rocDiagnosticHeaderRe = /^──\s+(\S+)\s+(.+?)\s+─+\s+(?:<source>|(?:\.\/)?([^\s:]+)):(\d+):(\d+)\s*$/;
const objdumpSymbolHeaderRe = /^[0-9a-fA-F]+\s+<([^>]+)>:$/;
const objdumpSectionHeaderRe = /^Disassembly of section /;

export class RocCompiler extends BaseCompiler {
    static get key() {
        return 'roc';
    }

    override fixFiltersBeforeCacheKey(filters: ParseFiltersAndOutputOptions, options: string[], files: FiledataPair[]) {
        // `roc build` produces linked binaries rather than textual assembly.
        filters.binary = true;
        super.fixFiltersBeforeCacheKey(filters, options, files);
    }

    override getOutputFilename(dirPath: string, outputFilebase: string): string {
        return dirPath ? path.join(dirPath, outputFilebase) : outputFilebase;
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string): string[] {
        filters.binary = true;
        return ['build', '--no-cache', `--output=${this.filename(outputFilename)}`];
    }

    override filterUserOptions(userOptions: string[]): string[] {
        const filtered: string[] = [];
        for (let i = 0; i < userOptions.length; i++) {
            const option = userOptions[i];
            if (i === 0 && forbiddenSubcommands.has(option)) {
                continue;
            }
            if (option === '--') {
                break;
            }
            if (option === '--watch' || option === '--no-cache' || option === '--keep-temp') {
                continue;
            }
            if (option === '--output' || option === '-o') {
                i += 1;
                continue;
            }
            if (option.startsWith('--output=')) {
                continue;
            }
            if (option === '--replace-dep') {
                i += 2;
                continue;
            }
            filtered.push(option);
        }
        return filtered;
    }

    override getSharedLibraryPathsAsArguments(): string[] {
        return [];
    }

    override getSharedLibraryLinks(): string[] {
        return [];
    }

    override getStaticLibraryLinks(): string[] {
        return [];
    }

    override getAllPossibleTargetFlags(): string[][] {
        return [[`--target=${c_value_placeholder}`], ['--target', c_value_placeholder]];
    }

    override async runCompiler(
        compiler: string,
        options: string[],
        inputFilename: string,
        execOptions: ExecutionOptionsWithEnv,
        filters?: ParseFiltersAndOutputOptions,
    ): Promise<CompilationResult> {
        if (!execOptions) {
            execOptions = this.getDefaultExecOptions();
        }
        const dir = path.dirname(inputFilename);
        if (!execOptions.customCwd) {
            execOptions.customCwd = dir;
        }
        // Roc creates temporary build directories under $XDG_CACHE_HOME/roc/tmp (falling back to $HOME/.cache)
        // and aborts with NoHomeDirectory if neither points to a writable directory.
        execOptions = {
            ...execOptions,
            env: {
                ...execOptions.env,
                HOME: dir,
                XDG_CACHE_HOME: path.join(dir, '.cache'),
            },
        };
        return super.runCompiler(compiler, options, inputFilename, execOptions, filters);
    }

    protected override processExecutionResult(
        input: UnprocessedExecResult,
        inputFilename?: string,
    ): BasicExecutionResult {
        const result = super.processExecutionResult(input, inputFilename);
        this.attachRocDiagnostics(result.stdout, inputFilename);
        this.attachRocDiagnostics(result.stderr, inputFilename);
        return result;
    }

    private attachRocDiagnostics(lines: ResultLine[], inputFilename?: string) {
        for (const lineObj of lines) {
            if (lineObj.tag) continue;
            const plain = utils.filterEscapeSequences(lineObj.text);
            const match = plain.match(rocDiagnosticHeaderRe);
            if (!match) continue;
            const [, marker, rawTitle, matchedFile, lineStr, colStr] = match;
            const title = rawTitle.trim();
            const file = matchedFile
                ? path.basename(matchedFile)
                : inputFilename
                  ? path.basename(inputFilename)
                  : undefined;
            lineObj.tag = {
                line: Number.parseInt(lineStr, 10),
                column: Number.parseInt(colStr, 10),
                text: title,
                severity: marker === '✗' ? 3 : 2,
                file,
            };
        }
    }

    override async processAsm(result: any, filters: ParseFiltersAndOutputOptions, options: string[]) {
        const asm =
            filters.libraryCode && typeof result.asm === 'string' ? this.filterRuntimeSymbols(result.asm) : result.asm;
        // Roc binaries do not emit per-line DWARF records referencing the `.roc` file, so AsmParser's DWARF-based
        // libraryCode check would discard user instructions if left enabled.
        return super.processAsm({...result, asm}, {...filters, binary: true, libraryCode: false}, options);
    }

    private filterRuntimeSymbols(asm: string): string {
        const lines = utils.splitLines(asm);
        const kept: string[] = [];
        let inSymbol = false;
        let keepSymbol = true;
        let foundUserSymbol = false;

        for (const line of lines) {
            if (objdumpSectionHeaderRe.test(line)) {
                inSymbol = false;
                keepSymbol = true;
                kept.push(line);
                continue;
            }
            const match = line.match(objdumpSymbolHeaderRe);
            if (match) {
                inSymbol = true;
                const symbol = match[1].replace(/^_/, '');
                keepSymbol =
                    symbol.startsWith('roc__') && !symbol.startsWith('roc__num_') && !symbol.startsWith('roc__rc_');
                if (keepSymbol) {
                    foundUserSymbol = true;
                }
            }
            if (!inSymbol || keepSymbol) {
                kept.push(line);
            }
        }

        return foundUserSymbol ? kept.join('\n') : asm;
    }
}
