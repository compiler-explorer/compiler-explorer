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
 * A fake compiler for Cypress E2E tests. Never executes a real binary.
 *
 * By default, echoes source lines back as "assembly", with each output line
 * mapped to its corresponding source line (so line highlighting works).
 *
 * Control output via magic comments in the source:
 *
 *   // FAKE: asm mov eax, 1
 *   // FAKE: asm ret
 *   // FAKE: stdout Hello world
 *   // FAKE: stderr warning: something
 *   // FAKE: exitcode 42
 *   // FAKE: pp #define FOO 1
 *   // FAKE: opt missed: loop not vectorised
 *   // FAKE: gccdump-pass 001t.original
 *   // FAKE: gccdump ;; Function main
 *
 * When asm directives are present, they replace the default echo behaviour.
 * Each asm directive line maps back to the source line the comment was on.
 *
 * Properties: compilerType=cypress-compiler, exe=/bin/true, version=1.0.0
 */
export class CypressCompiler extends BaseCompiler {
    static get key() {
        return 'cypress-compiler';
    }

    override async initialise() {
        // Skip real binary execution entirely — set capabilities directly
        this.compiler.version = '1.0.0 (fake)';
        this.compiler.fullVersion = '1.0.0 (fake)';
        this.compiler.supportsPpView = true;
        this.compiler.supportsOptOutput = true;
        this.compiler.supportsGccDump = true;
        this.compiler.supportsCfg = true;
        this.compiler.supportsExecute = true;
        return this;
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
        this.applyOptionOverrides(options, directives);

        if (backendOptions.executorRequest) {
            return this.buildExecutionResult(directives);
        }

        const asm = directives.asm.length > 0 ? directives.asm : this.echoSource(source, options);

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
            asm: asm.map(({text, sourceLine}) => ({
                text,
                source: sourceLine ? {file: null, line: sourceLine, mainsource: true} : null,
                labels: [],
            })),
            languageId: 'c++',
        };

        if (backendOptions.producePp) {
            result.ppOutput = {
                numberOfLinesFiltered: 0,
                output: directives.pp.length > 0 ? directives.pp.join('\n') : `// Preprocessed\n${source}`,
            };
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
                Function: 'main',
                DebugLoc: {File: 'example.cpp', Line: i + 1, Column: 1},
                Args: [{String: text}],
            }));
        }

        if (backendOptions.produceGccDump) {
            const passes = directives.gccdumpPasses.length > 0 ? directives.gccdumpPasses : ['001t.original'];
            result.gccDumpOutput = {
                all: passes.map(name => ({name, header: name})),
                currentPassOutput:
                    directives.gccdump.length > 0 ? directives.gccdump.join('\n') : ';; Function main\n(nop)',
                selectedPass: passes[0],
                treeDumpEnabled: true,
                rtlDumpEnabled: true,
                ipaDumpEnabled: true,
            };
        }

        return result;
    }

    private echoSource(source: string, options: string[]): Array<{text: string; sourceLine: number | null}> {
        const result: Array<{text: string; sourceLine: number | null}> = [];
        const displayOptions = options.filter(o => !o.startsWith('--fake-'));
        if (displayOptions.length > 0) {
            result.push({text: `; Options: ${displayOptions.join(' ')}`, sourceLine: null});
        }
        for (const [i, line] of source.split('\n').entries()) {
            result.push({text: line, sourceLine: line.trim() ? i + 1 : null});
        }
        return result;
    }

    private applyOptionOverrides(options: string[], directives: ReturnType<CypressCompiler['parseDirectives']>) {
        for (const opt of options) {
            const match = opt.match(/^--fake-(\w+)=(.*)$/);
            if (!match) continue;
            const [, key, value] = match;
            switch (key) {
                case 'exitcode':
                    directives.exitcode = parseInt(value, 10);
                    break;
                case 'stdout':
                    directives.stdout.push(value);
                    break;
                case 'stderr':
                    directives.stderr.push(value);
                    break;
                case 'asm':
                    directives.asm.push({text: value, sourceLine: null});
                    break;
            }
        }
    }

    private parseDirectives(source: string) {
        const directives = {
            asm: [] as Array<{text: string; sourceLine: number | null}>,
            stdout: [] as string[],
            stderr: [] as string[],
            exitcode: 0,
            pp: [] as string[],
            opt: [] as string[],
            gccdumpPasses: [] as string[],
            gccdump: [] as string[],
        };

        const lines = source.split('\n');
        for (let i = 0; i < lines.length; i++) {
            const match = lines[i].match(/\/\/\s*FAKE:\s*(\S+)\s*(.*)/);
            if (!match) continue;
            const [, directive, value] = match;
            switch (directive) {
                case 'asm':
                    directives.asm.push({text: value, sourceLine: i + 1});
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

    private buildExecutionResult(directives: ReturnType<CypressCompiler['parseDirectives']>): CompilationResult {
        return {
            code: 0,
            didExecute: true,
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
