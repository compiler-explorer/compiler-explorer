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

import type {CompilationResult, ExecutionOptionsWithEnv} from '../../types/compilation/compilation.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import type {SelectedLibraryVersion} from '../../types/libraries/libraries.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';

// Nexium (https://londopy.github.io/nexium/) compiles to C and drives a C
// compiler. Without the binary filter, the output pane shows the C that
// `nx emit-c` writes, with the runtime it inlines folded away unless the
// "library code" filter is on; with the binary filter, `nx build` produces
// an executable that Compiler Explorer disassembles as usual.
export class NexiumCompiler extends BaseCompiler {
    static get key() {
        return 'nexium';
    }

    // the emitted C is the whole translation unit: the runtime, then the
    // program's sections, each opened by a four-dash marker
    static readonly programStart = /^\/\* ---- (imported C headers|forward declarations|types) ---- \*\/$/;

    override getOutputFilename(dirPath: string, outputFilebase: string, key?: any): string {
        const binary = key?.filters?.binary;
        return path.join(dirPath, binary ? outputFilebase : 'output.c');
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string): string[] {
        // the input file name is appended by the base class
        if (filters.binary) {
            return ['build', '--mode', 'fast', '-o', outputFilename, '--out-dir', path.dirname(outputFilename)];
        }
        return ['emit-c', '--mode', 'fast'];
    }

    override filterUserOptions(userOptions: string[]): string[] {
        // the command is ours; a user picks the mode or the CPU
        return userOptions.filter(option => !['build', 'run', 'emit-c', 'check', '-o'].includes(option));
    }

    override async runCompiler(
        compiler: string,
        options: string[],
        inputFilename: string,
        execOptions: ExecutionOptionsWithEnv,
        filters?: ParseFiltersAndOutputOptions,
    ): Promise<CompilationResult> {
        if (!execOptions) {
            execOptions = super.getDefaultExecOptions();
        }
        const dir = path.dirname(inputFilename);
        if (!execOptions.customCwd) execOptions.customCwd = dir;
        // nx keeps its caches and generated C beside the program
        execOptions.env['NX_NO_UPDATE_CHECK'] = '1';
        execOptions.env['NX_OFFLINE'] = '1';

        // `nx <command> [options] file` : the base class appends the file last,
        // but nx wants it right after the command
        const command = options[0];
        const file = options[options.length - 1];
        const reordered = [command, file, ...options.slice(1, -1)];

        const result = await this.exec(compiler, reordered, execOptions);
        if (!filters?.binary && result.code === 0) {
            // emit-c prints the C: it becomes the output file the base class reads
            const outputFilename = this.getOutputFilename(dir, this.outputFilebase);
            await fs.writeFile(outputFilename, result.stdout);
            result.stdout = '';
        }
        return {
            ...this.transformToCompilationResult(result, inputFilename),
            languageId: this.getCompilerResultLanguageId(filters),
        };
    }

    override getCompilerResultLanguageId(filters?: ParseFiltersAndOutputOptions): string | undefined {
        return filters?.binary ? undefined : 'c';
    }

    override async processAsm(result: any, filters: ParseFiltersAndOutputOptions, options: string[]): Promise<any> {
        if (filters.binary) {
            return this.asm.process(result.asm, filters);
        }
        let lines: string[] = result.asm.split('\n');
        // the inlined runtime is library code
        if (!filters.libraryCode) {
            const start = lines.findIndex(line => NexiumCompiler.programStart.test(line));
            if (start > 0) lines = lines.slice(start);
        }
        if (!filters.commentOnly) {
            lines = lines.filter(line => !line.trimStart().startsWith('/*') || !line.trimEnd().endsWith('*/'));
        }
        if (!filters.directives) lines = lines.filter(line => !line.trimStart().startsWith('#'));
        if (filters.trim) lines = lines.map(line => line.trimStart()).filter(line => line !== '');
        const out: string[] = [];
        let blank = false;
        for (const line of lines) {
            if (line.trim() === '') {
                if (blank) continue;
                blank = true;
            } else {
                blank = false;
            }
            out.push(line);
        }
        return {asm: out.map(text => ({text}))};
    }

    override getSharedLibraryPathsAsArguments(libraries: SelectedLibraryVersion[], libDownloadPath?: string) {
        return [];
    }

    override getSharedLibraryLinks(libraries: SelectedLibraryVersion[]): string[] {
        return [];
    }

    override isCfgCompiler() {
        return false;
    }
}
