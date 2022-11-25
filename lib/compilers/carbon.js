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

import {BaseCompiler} from '../base-compiler';

import {BaseParser} from './argument-parsers';

export class CarbonCompiler extends BaseCompiler {
    static get key() {
        return 'carbon';
    }

    constructor(compilerInfo, env) {
        super(compilerInfo, env);
        this.compiler.demangler = '';
        this.demanglerClass = null;
    }

    optionsForFilter(filters, outputFilename) {
        return ['--color', `--trace_file=${outputFilename}`];
    }

    processAsm(result, filters, options) {
        // Really should write a custom parser, but for now just don't filter anything.
        return super.processAsm(result, {}, options);
    }

    async afterCompilation(
        result,
        doExecute,
        key,
        executeParameters,
        tools,
        backendOptions,
        filters,
        options,
        optOutput,
        customBuildPath,
    ) {
        result = await super.afterCompilation(
            result,
            doExecute,
            key,
            executeParameters,
            tools,
            backendOptions,
            filters,
            options,
            optOutput,
            customBuildPath,
        );
        if (result.code === 0) {
            // Hook to parse out the "result: 123" line at the end of the interpreted execution run.
            const re = /^result: (\d+)$/;
            const match = re.exec(result.asm.at(-1).text);
            const code = match ? parseInt(match[1]) : -1;
            result.execResult = {
                stdout: result.stdout,
                stderr: [],
                code: code,
                didExecute: true,
                buildResult: {code: 0},
            };
            result.stdout = [];
        }
        return result;
    }

    getArgumentParser() {
        // TODO: may need a custom one, based on/borrowing from ClangParser
        return BaseParser;
    }
}
