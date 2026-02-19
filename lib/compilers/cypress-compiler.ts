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

import type {
    ActiveTool,
    BypassCache,
    CompilationResult,
    ExecutionParams,
    FiledataPair,
} from '../../types/compilation/compilation.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import type {SelectedLibraryVersion} from '../../types/libraries/libraries.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';

/**
 * A fake compiler for Cypress E2E tests that never executes a real binary.
 * Parses directives from source comments to produce deterministic results.
 *
 * ## Directives (in source comments)
 *
 *   // @cypress-fake: asm push rbp; mov rbp, rsp; ret
 *   // @cypress-fake: stdout Hello from cypress
 *   // @cypress-fake: stderr Some warning
 *   // @cypress-fake: exitcode 42
 *   // @cypress-fake: pp #expanded preprocessor output
 *   // @cypress-fake: opt missed loop not vectorised
 *   // @cypress-fake: gccdump-pass 001t.original
 *   // @cypress-fake: gccdump ;; Function square
 *
 * Without directives, generates assembly from function names found in source.
 * Handles #ifdef/#else/#endif with -D flags.
 *
 * Properties: compilerType=cypress-compiler, exe=/bin/true, version=1.0.0
 */
export class CypressCompiler extends BaseCompiler {
    static get key() {
        return 'cypress-compiler';
    }

    override async compile(
        source: string,
        options: string[],
        backendOptions: Record<string, any>,
        filters: ParseFiltersAndOutputOptions,
        _bypassCache: BypassCache,
        _tools: ActiveTool[],
        _executeParameters: ExecutionParams,
        _libraries: SelectedLibraryVersion[],
        _files: FiledataPair[],
    ): Promise<CompilationResult> {
        const directives = this.parseDirectives(source);
        const defines = this.extractDefines(options);
        const processed = this.applyPreprocessor(source, defines);

        if (backendOptions.executorRequest) {
            return this.buildExecutionResult(directives);
        }

        const asmLines = directives.asm.length > 0 ? directives.asm : this.generateDefaultAsm(processed, filters);

        const result: CompilationResult = {
            code: directives.exitcode,
            timedOut: false,
            okToCache: true,
            stdout: directives.stdout.map(text => ({text})),
            stderr: directives.stderr.map(text => ({text})),
            inputFilename: 'example.cpp',
            compilationOptions: options,
            downloads: [],
            tools: [],
            asm: asmLines.map((text, i) => ({
                text,
                source: {file: null, line: i < asmLines.length - 1 ? Math.floor(i / 2) + 1 : null, mainsource: true},
                labels: [],
            })),
            languageId: 'c++',
        };

        if (backendOptions.producePp) {
            result.ppOutput =
                directives.pp.length > 0
                    ? {numberOfLinesFiltered: 0, output: directives.pp.join('\n')}
                    : {numberOfLinesFiltered: 0, output: this.generateDefaultPp(processed, defines)};
        }

        if (backendOptions.produceOptInfo) {
            result.optOutput = directives.opt.map((text, i) => ({
                optType: (text.startsWith('passed') ? 'Passed' : text.startsWith('missed') ? 'Missed' : 'Analysis') as
                    | 'Passed'
                    | 'Missed'
                    | 'Analysis',
                displayString: text,
                Pass: 'fake-pass',
                Name: 'fake-remark',
                Function: 'test_function',
                DebugLoc: {File: 'example.cpp', Line: i + 1, Column: 1},
                Args: [{String: text}],
            }));
        }

        if (backendOptions.produceGccDump) {
            const passes =
                directives.gccdumpPasses.length > 0
                    ? directives.gccdumpPasses
                    : ['001t.original', '002t.gimple', '003t.eh'];
            result.gccDumpOutput = {
                all: passes.map(name => ({name, header: name})),
                currentPassOutput:
                    directives.gccdump.length > 0
                        ? directives.gccdump.join('\n')
                        : this.generateDefaultGccDump(processed),
                selectedPass: passes[0],
                treeDumpEnabled: true,
                rtlDumpEnabled: true,
                ipaDumpEnabled: true,
            };
        }

        return result;
    }

    private parseDirectives(source: string) {
        const directives = {
            asm: [] as string[],
            stdout: [] as string[],
            stderr: [] as string[],
            exitcode: 0,
            pp: [] as string[],
            opt: [] as string[],
            gccdumpPasses: [] as string[],
            gccdump: [] as string[],
        };
        for (const line of source.split('\n')) {
            const match = line.match(/\/\/\s*@cypress-fake:\s*(\S+)\s*(.*)/);
            if (!match) continue;
            const [, directive, value] = match;
            switch (directive) {
                case 'asm':
                    directives.asm.push(...value.split(';').map(s => s.trim()));
                    break;
                case 'stdout':
                    directives.stdout.push(value);
                    break;
                case 'stderr':
                    directives.stderr.push(value);
                    break;
                case 'exitcode':
                    directives.exitcode = parseInt(value, 10);
                    break;
                case 'pp':
                    directives.pp.push(value);
                    break;
                case 'opt':
                    directives.opt.push(value);
                    break;
                case 'gccdump-pass':
                    directives.gccdumpPasses.push(value);
                    break;
                case 'gccdump':
                    directives.gccdump.push(value);
                    break;
            }
        }
        return directives;
    }

    private extractDefines(options: string[]): Map<string, string> {
        const defines = new Map<string, string>();
        for (const opt of options) {
            const match = opt.match(/^-D(\w+)(?:=(.*))?$/);
            if (match) defines.set(match[1], match[2] ?? '1');
        }
        return defines;
    }

    private applyPreprocessor(source: string, defines: Map<string, string>): string {
        const lines = source.split('\n');
        const output: string[] = [];
        const stack: boolean[] = [];
        let active = true;
        for (const line of lines) {
            const ifdefMatch = line.match(/^\s*#ifdef\s+(\w+)/);
            const ifndefMatch = line.match(/^\s*#ifndef\s+(\w+)/);
            const elifMatch = line.match(/^\s*#elif\s+defined\((\w+)\)/);
            const directive = line.match(/^\s*#(else|endif)/);
            if (ifdefMatch) {
                stack.push(active);
                active = active && defines.has(ifdefMatch[1]);
            } else if (ifndefMatch) {
                stack.push(active);
                active = active && !defines.has(ifndefMatch[1]);
            } else if (elifMatch) {
                active = (stack[stack.length - 1] ?? true) && defines.has(elifMatch[1]);
            } else if (directive) {
                if (directive[1] === 'else') {
                    active = (stack[stack.length - 1] ?? true) && !active;
                } else {
                    active = stack.pop() ?? true;
                }
            } else if (active && !line.match(/^\s*#/)) {
                output.push(line);
            }
        }
        return output.join('\n');
    }

    private extractFunctionNames(source: string): string[] {
        const funcRegex = /\b(?:int|void|bool|char|float|double|auto|long|short|unsigned)\s+(\w+)\s*\(/g;
        const names: string[] = [];
        let match;
        while ((match = funcRegex.exec(source)) !== null) names.push(match[1]);
        return names.length > 0 ? names : ['main'];
    }

    private generateDefaultAsm(source: string, filters: ParseFiltersAndOutputOptions): string[] {
        const functions = this.extractFunctionNames(source);
        const lines: string[] = [];
        for (const fn of functions) {
            if (!filters.directives) {
                lines.push(`.globl ${fn}`, `.type ${fn}, @function`);
            }
            lines.push(`${fn}:`, '  push rbp', '  mov rbp, rsp', '  xor eax, eax', '  pop rbp', '  ret');
        }
        return lines;
    }

    private generateDefaultPp(source: string, defines: Map<string, string>): string {
        const lines = ['// Preprocessor output from fake compiler'];
        for (const [key, value] of defines) lines.push(`// #define ${key} ${value}`);
        lines.push(source);
        return lines.join('\n');
    }

    private generateDefaultGccDump(source: string): string {
        const fn = this.extractFunctionNames(source)[0] || 'main';
        return [
            `;; Function ${fn}`,
            '(note 1 0 2 NOTE_INSN_DELETED)',
            '(insn 2 1 3 (set (reg:SI 0 eax) (const_int 0)))',
            '(insn 3 2 0 (use (reg:SI 0 eax)))',
        ].join('\n');
    }

    private buildExecutionResult(directives: ReturnType<CypressCompiler['parseDirectives']>): CompilationResult {
        return {
            code: 0,
            timedOut: false,
            stdout: [],
            stderr: [],
            inputFilename: 'example.cpp',
            compilationOptions: [],
            downloads: [],
            tools: [],
            asm: [],
            languageId: 'c++',
            execResult: {
                didExecute: true,
                code: directives.exitcode,
                stdout: directives.stdout.map(text => ({text})),
                stderr: directives.stderr.map(text => ({text})),
                timedOut: false,
                buildResult: {
                    code: 0,
                    timedOut: false,
                    stdout: [],
                    stderr: [],
                    inputFilename: 'example.cpp',
                    compilationOptions: [],
                    downloads: [],
                    executableFilename: '',
                    tools: [],
                    asm: [],
                    languageId: 'c++',
                },
            },
        };
    }
}
