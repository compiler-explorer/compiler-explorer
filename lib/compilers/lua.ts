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

import fs from 'node:fs/promises';
import path from 'node:path';

import type {AsmResultSource, ParsedAsmResultLine} from '../../types/asmresult/asmresult.interfaces.js';
import type {
    CacheKey,
    CompilationResult,
    ExecutionOptionsWithEnv,
} from '../../types/compilation/compilation.interfaces.js';
import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {ExecutableExecutionOptions} from '../../types/execution/execution.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';
import {CompilationEnvironment} from '../compilation-env.js';
import {BaseParser} from './argument-parsers.js';

// Compiler for the reference (PUC-Rio) Lua interpreter.
//
// The configured `compiler.exe` is expected to be `luac`; it is invoked with
// `-l -l -p` to produce a verbose bytecode listing on stdout. Execution uses
// the `lua` interpreter configured via the `interpreter` property.
//
// Alternative Lua implementations (e.g. LuaJIT) can extend this class and
// override `getDisassemblyArgs` to plug in a different bytecode dumper.
export class LuaCompiler extends BaseCompiler {
    private readonly interpreterExe: string;

    static get key() {
        return 'lua';
    }

    constructor(compilerInfo: PreliminaryCompilerInfo, env: CompilationEnvironment) {
        super(compilerInfo, env);
        this.compiler.demangler = '';
        this.demanglerClass = null;
        // Per-compiler properties aren't inherited through the normal cascade
        // (TODO(#7150)), so resolve the interpreter explicitly: compiler-specific,
        // then group, then the language-level default. Reading plain 'interpreter'
        // would only ever see the top-level value, so every version would execute
        // with the same Lua.
        this.interpreterExe =
            this.compilerProps<string>(`compiler.${this.compiler.id}.interpreter`, '') ||
            this.compilerProps<string>(`group.${this.compiler.group}.interpreter`, '') ||
            this.compilerProps<string>('interpreter', '');
    }

    override getArgumentParserClass() {
        return BaseParser;
    }

    override optionsForFilter(_filters: ParseFiltersAndOutputOptions, _outputFilename: string): string[] {
        // luac's listing always goes to stdout; runCompiler handles capturing it.
        return [];
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
        if (!execOptions.customCwd) {
            execOptions.customCwd = path.dirname(inputFilename);
        }

        const luacArgs = this.getDisassemblyArgs(options, inputFilename);
        const result = await this.exec(compiler, luacArgs, execOptions);

        const outputFilename = this.getOutputFilename(path.dirname(inputFilename), this.outputFilebase);
        await fs.writeFile(outputFilename, result.stdout);
        result.stdout = '';

        return {
            ...this.transformToCompilationResult(result, inputFilename),
            languageId: this.getCompilerResultLanguageId(filters),
            instructionSet: this.getInstructionSetFromCompilerArgs(luacArgs),
        };
    }

    override async handleInterpreting(
        key: CacheKey,
        executeParameters: ExecutableExecutionOptions,
    ): Promise<CompilationResult> {
        if (!this.interpreterExe) {
            return super.handleInterpreting(key, executeParameters);
        }
        // `compiler.exe` points at luac for compilation; swap it to the lua
        // interpreter for the duration of the script run.
        const originalExe = this.compiler.exe;
        this.compiler.exe = this.interpreterExe;
        try {
            return await super.handleInterpreting(key, executeParameters);
        } finally {
            this.compiler.exe = originalExe;
        }
    }

    override async processAsm(result: {asm: string}) {
        // luac listing format (Lua 5.x):
        //   "main <name:0,0> (N instructions at 0x...)"
        //   "        1       [3]     VARARGPREP      0"
        // The bracketed value is the source line for that bytecode instruction.
        const instructionLineRe = /^\s*\d+\s+\[(\d+)\]/;
        const functionHeaderRe = /<[^:>]*:(\d+),\d+>/;

        const bytecodeLines = result.asm.split('\n');
        const bytecodeResult: ParsedAsmResultLine[] = [];
        let lastLineNo: number | null = null;
        let sourceLoc: AsmResultSource | null = null;

        for (const line of bytecodeLines) {
            const insnMatch = line.match(instructionLineRe);
            const fnMatch = line.match(functionHeaderRe);

            if (insnMatch) {
                const lineno = Number.parseInt(insnMatch[1], 10);
                lastLineNo = lineno;
                sourceLoc = {line: lineno, file: null};
            } else if (fnMatch) {
                const lineno = Number.parseInt(fnMatch[1], 10);
                lastLineNo = lineno > 0 ? lineno : null;
                sourceLoc = {line: lastLineNo, file: null};
            } else if (line) {
                sourceLoc = {line: lastLineNo, file: null};
            } else {
                lastLineNo = null;
                sourceLoc = {line: null, file: null};
            }

            bytecodeResult.push({text: line, source: sourceLoc});
        }

        return {asm: bytecodeResult};
    }

    /** Arguments passed to the bytecode dumper. */
    protected getDisassemblyArgs(_options: string[], inputFilename: string): string[] {
        // -l -l : verbose listing (constants, locals, upvalues)
        // -p    : parse only, do not write a bytecode file
        return ['-l', '-l', '-p', inputFilename];
    }
}
