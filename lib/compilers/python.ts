// Copyright (c) 2026, Compiler Explorer Authors
// Copyright (c) 2019, Sebastian Rath
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

import type {AsmResultSource, ParsedAsmResultLine} from '../../types/asmresult/asmresult.interfaces.js';
import type {CompilationResult} from '../../types/compilation/compilation.interfaces.js';
import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import type {ResultLine} from '../../types/resultline/resultline.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';
import {CompilationEnvironment} from '../compilation-env.js';
import {PythonAstParser} from '../python-ast.js';
import {resolvePathFromAppRoot} from '../utils.js';
import {BaseParser} from './argument-parsers.js';

export class PythonCompiler extends BaseCompiler {
    private readonly disasmScriptPath: string;
    private readonly astScriptPath: string;
    private readonly pythonAstParser: PythonAstParser;

    static get key() {
        return 'python';
    }

    constructor(compilerInfo: PreliminaryCompilerInfo, env: CompilationEnvironment) {
        super(compilerInfo, env);
        this.compiler.demangler = '';
        this.demanglerClass = null;
        this.disasmScriptPath =
            this.compilerProps<string>('disasmScript') ||
            resolvePathFromAppRoot('etc', 'scripts', 'disasms', 'dis_all.py');
        this.astScriptPath =
            this.compilerProps<string>('astScript') || resolvePathFromAppRoot('etc', 'scripts', 'ast_dump.py');
        this.pythonAstParser = new PythonAstParser(this.compilerProps);
    }

    override async processAsm(result) {
        const lineRe = /^\s{0,4}(\d+)(.*)/;

        const bytecodeLines = result.asm.split('\n');

        const bytecodeResult: ParsedAsmResultLine[] = [];
        let lastLineNo: number | null = null;
        let sourceLoc: AsmResultSource | null = null;

        for (const line of bytecodeLines) {
            const match = line.match(lineRe);

            if (match) {
                const lineno = Number.parseInt(match[1], 10);
                sourceLoc = {line: lineno, file: null};
                lastLineNo = lineno;
            } else if (line) {
                sourceLoc = {line: lastLineNo, file: null};
            } else {
                sourceLoc = {line: null, file: null};
                lastLineNo = null;
            }

            bytecodeResult.push({text: line, source: sourceLoc});
        }

        return {asm: bytecodeResult};
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string) {
        return ['-I', this.disasmScriptPath, '--outputfile', outputFilename, '--inputfile'];
    }

    override couldSupportASTDump(version: string) {
        // Python's ast module is available since Python 2.6. However, the
        // formatted dump of Python AST via `ast.dump()` is available since
        // Python 3.9. Therefore, we return true for all versions >= 3.9.

        const versionRegex = /Python (\d+)\.(\d+)\.(\d+)$/;
        const versionMatch = versionRegex.exec(version);

        if (versionMatch === null) {
            return false;
        }

        const major = Number.parseInt(versionMatch[1]);
        const minor = Number.parseInt(versionMatch[2]);
        return major > 3 || (major === 3 && minor >= 9);
    }

    override async generateAST(inputFilename: string, options: string[]): Promise<ResultLine[]> {
        const astOptions = ['-I', this.astScriptPath, this.filename(inputFilename)];
        const execOptions = this.getDefaultExecOptions();
        const result: CompilationResult = await this.runCompiler(
            this.compiler.exe,
            astOptions,
            this.filename(inputFilename),
            execOptions,
        );
        return this.pythonAstParser.processAst(result);
    }

    override getArgumentParserClass() {
        return BaseParser;
    }

    override orderArguments(
        options: string[],
        inputFilename: string,
        libIncludes: string[],
        libOptions: string[],
        libPaths: string[],
        libLinks: string[],
        userOptions: string[],
        staticLibLinks: string[],
    ) {
        return options.concat(
            [this.filename(inputFilename)],
            libIncludes,
            libOptions,
            libPaths,
            libLinks,
            userOptions,
            staticLibLinks,
        );
    }
}
