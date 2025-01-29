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

import path from 'path';

import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';

import {ElixirParser} from './argument-parsers.js';

export class ElixirAsmCompiler extends BaseCompiler {
    static get key() {
        return 'elixirasm';
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string): string[] {
        return [
            '--eval',
            '[input] = Code.required_files();' +
                'code = Code.compile_file(input);' +
                ':erts_debug.set_internal_state(:available_internal_state, true);' +
                ':erts_debug.set_internal_state(:jit_asm_dump, true);' +
                'code = Code.compile_file(input)' +
                '|> Enum.map(fn {module,_} -> File.read!(:erlang.atom_to_list(module) ++ ".asm") end) ' +
                '|> Enum.join("\n"); ' +
                `:ok = File.write("${outputFilename}", code);`,
        ];
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
    ): string[] {
        return ['-r', inputFilename]
            .concat(options)
            .concat(libIncludes, libOptions, libPaths, libLinks, userOptions, staticLibLinks);
    }

    override getOutputFilename(dirPath: string, outputFilebase: string): string {
        return path.join(dirPath, `${outputFilebase}.asm`);
    }

    override getArgumentParserClass() {
        return ElixirParser;
    }
}
