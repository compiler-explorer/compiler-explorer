// Copyright (c) 2025, Compiler Explorer Authors
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

import type {CompilationResult} from '../../types/compilation/compilation.interfaces.js';
import type {
    OptPipelineBackendOptions,
    OptPipelineOutput,
} from '../../types/compilation/opt-pipeline-output.interfaces.js';
import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';
import {CompilationEnvironment} from '../compilation-env.js';
import {AsmParser} from '../parsers/asm-parser.js';
import {NumbaPassDumpParser} from '../parsers/numba-pass-dump-parser.js';
import {resolvePathFromAppRoot} from '../utils.js';
import {BaseParser} from './argument-parsers.js';

export class NumbaCompiler extends BaseCompiler {
    private compilerWrapperPath: string;
    private passDumpParser: NumbaPassDumpParser;

    static get key() {
        return 'numba';
    }

    constructor(compilerInfo: PreliminaryCompilerInfo, env: CompilationEnvironment) {
        super(compilerInfo, env);
        this.compilerWrapperPath =
            this.compilerProps('compilerWrapper', '') || resolvePathFromAppRoot('etc', 'scripts', 'numba_wrapper.py');
        this.passDumpParser = new NumbaPassDumpParser();
        this.compiler.optPipeline = {
            groupName: 'Function',
            supportedOptions: [],
            supportedFilters: [],
            monacoLanguage: 'python',
        };
    }

    override async processAsm(result, filters, options: string[]) {
        const processed = await super.processAsm(result, filters, options);
        // Numba's function-end labels survive standard filtering.
        if (filters.labels) {
            processed.asm = processed.asm.filter(item => !item.text.startsWith('.Lfunc_end'));
        }
        if (!(this.asm instanceof AsmParser)) return processed;
        for (const item of processed.asm) {
            // We receive line numbers as comments to line ends.
            const match = item.text.match(/;(\d+)$/);
            if (!match) continue;
            item.text = item.text.slice(0, match.index);
            const inNvccCode = false;
            if (this.asm.hasOpcode(item.text, inNvccCode))
                item.source = {line: Number.parseInt(match[1], 10), file: null};
        }
        return processed;
    }

    override async postProcessAsm(result, filters?: ParseFiltersAndOutputOptions) {
        result = await super.postProcessAsm(result, filters);
        for (const item of result.asm) {
            let line = item.text;
            // Numba includes long and noisy abi tags.
            line = line.replaceAll(/\[abi:\w+]/g, '');
            // Numba's custom name mangling is not invertible. It escapes symbols to
            // valid Python identifiers in a "_%02x" format. Because users can write
            // coinciding identifiers, we cannot perfectly demangle. Python qualifies
            // scoped function names with "<locals>". There is little risk from
            // collisions with user-defined symbols including `_3clocals_3e`.
            line = line.replaceAll('::_3clocals_3e::', '::<locals>::');
            // Numba's generators have many escaped symbols in their argument listings.
            line = line.replace(/::next\(\w+_20generator_28\w+\)/, decode_symbols);
            item.text = line;
        }
        return result;
    }

    override async generateOptPipeline(
        inputFilename: string,
        options: string[],
        filters: ParseFiltersAndOutputOptions,
        optPipelineOptions: OptPipelineBackendOptions,
    ): Promise<OptPipelineOutput | undefined> {
        const pipelineDir = await this.newTempDir();
        const inputFile = this.filename(inputFilename);
        const pipelineFile = path.join(pipelineDir, path.basename(inputFile));
        await fs.copyFile(inputFile, pipelineFile);

        const execOptions = this.getDefaultExecOptions();
        execOptions.maxOutput = 1024 * 1024 * 1024;
        execOptions.env.NUMBA_DEBUG_PRINT_AFTER = 'all';

        const compileStart = performance.now();
        const output = await this.runCompiler(this.compiler.exe, options, pipelineFile, execOptions);
        const compileEnd = performance.now();

        if (output.timedOut) {
            return {
                error: 'Invocation timed out',
                results: {},
                compileTime: output.execTime || compileEnd - compileStart,
            };
        }

        if (output.code !== 0) {
            return;
        }

        try {
            const parseStart = performance.now();
            const results = await this.processOptPipeline(output, filters, optPipelineOptions);
            const parseEnd = performance.now();
            return {
                results,
                compileTime: compileEnd - compileStart,
                parseTime: parseEnd - parseStart,
            };
        } catch (e: any) {
            return {
                error: e.toString(),
                results: {},
                compileTime: compileEnd - compileStart,
            };
        }
    }

    override async processOptPipeline(
        output: CompilationResult,
        filters: ParseFiltersAndOutputOptions,
        optPipelineOptions: OptPipelineBackendOptions,
    ) {
        // Numba writes pass dumps to stdout via print()
        return this.passDumpParser.process(output.stdout, filters, optPipelineOptions);
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string): string[] {
        return ['-I', this.compilerWrapperPath, '--outputfile', outputFilename, '--inputfile'];
    }

    override getArgumentParserClass() {
        return BaseParser;
    }
}

export function decode_symbols(text: string): string {
    // Numba escapes /[^a-z0-9_]/ characters to "_%02x"-formatted strings.
    return text.replaceAll(/_([\da-f]{2})/g, (_, hex) => String.fromCodePoint(Number.parseInt(hex, 16)));
}
