// Copyright (c) 2022, Compiler Explorer Authors
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

import {CompilationResult, ExecutionOptions} from '../../types/compilation/compilation.interfaces';
import {CompilerInfo} from '../../types/compiler.interfaces';
import {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces';
import {BaseCompiler} from '../base-compiler';
import {logger} from '../logger';

export class RacketCompiler extends BaseCompiler {
    private raco: string;

    static get key() {
        return 'racket';
    }

    constructor(info: CompilerInfo, env) {
        // Disable output filters, as they currently don't do anything
        if (!info.disabledFilters) {
            info.disabledFilters = ['labels', 'directives', 'commentOnly', 'trim'];
        }
        super(info, env);
        this.raco = this.compilerProps<string>(`compiler.${this.compiler.id}.raco`);
    }

    override optionsForFilter(
        filters: ParseFiltersAndOutputOptions,
        outputFilename: string,
        userOptions?: string[],
    ): string[] {
        // We currently always compile to bytecode first and then decompile.
        // Forcing `binary` on like this ensures `objdump` will be called for
        // the decompilation phase.
        filters.binary = true;

        return [];
    }

    override supportsObjdump(): boolean {
        return true;
    }

    override getSharedLibraryPathsAsArguments(libraries: object[], libDownloadPath?: string): string[] {
        return [];
    }

    override async runCompiler(
        compiler: string,
        options: string[],
        inputFilename: string,
        execOptions: ExecutionOptions,
    ): Promise<CompilationResult> {
        if (!execOptions) {
            execOptions = this.getDefaultExecOptions();
        }

        if (!execOptions.customCwd) {
            execOptions.customCwd = path.dirname(inputFilename);
        }

        // Compile to bytecode via `raco make`
        options.unshift('make');
        const makeResult = await this.exec(this.raco, options, execOptions);

        return this.transformToCompilationResult(makeResult, inputFilename);
    }

    override getOutputFilename(dirPath: string, outputFilebase: string, key?: any): string {
        return path.join(dirPath, 'compiled', `${this.compileFilename.replace('.', '_')}.zo`);
    }

    override async objdump(
        outputFilename: any,
        result: any,
        maxSize: number,
        intelAsm: any,
        demangle: any,
        staticReloc,
        dynamicReloc,
        filters: ParseFiltersAndOutputOptions,
    ): Promise<any> {
        // Decompile to assembly via `raco decompile` with `disassemble` package
        const execOptions: ExecutionOptions = {
            maxOutput: maxSize,
            customCwd: (result.dirPath as string) || path.dirname(outputFilename),
        };
        const decompileResult = await this.exec(this.raco, ['decompile', outputFilename], execOptions);

        if (decompileResult.code) {
            logger.error('Error decompiling via `raco decompile`', decompileResult);
            result.asm = `<No output: \`raco decompile\` returned ${decompileResult.code}>`;
        }

        result.objdumpTime = decompileResult.execTime;
        result.asm = this.postProcessObjdumpOutput(decompileResult.stdout);

        return result;
    }

    override processAsm(result: any, filters: any, options: any) {
        // TODO: Process and highlight decompiled output
        return {
            asm: [{text: result.asm}],
        };
    }
}
