// Copyright (c) 2021, Compiler Explorer Authors
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

import {ParseFilters} from '../../types/features/filters.interfaces';
import {BaseCompiler} from '../base-compiler';

import {ErlangParser} from './argument-parsers';

export class ErlangCompiler extends BaseCompiler {
    static get key() {
        return 'erlang';
    }

    override optionsForFilter(filters: ParseFilters, outputFilename: string): string[] {
        return [
            '-noshell',
            '-eval',
            '{ok, Input} = init:get_argument(input),' +
                "{ok, _, Output} = compile:file(Input, ['S', binary, no_line_info, report])," +
                `{ok,Fd} = file:open("${outputFilename}", [write]),` +
                'beam_listing:module(Fd, Output),' +
                'file:close(Fd),' +
                'halt().',
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
        options.push('-input', inputFilename);
        return options.concat(libIncludes, libOptions, libPaths, libLinks, userOptions, staticLibLinks);
    }

    override getOutputFilename(dirPath: string, outputFilebase: string): string {
        return path.join(dirPath, `${outputFilebase}.S`);
    }

    override getArgumentParser() {
        return ErlangParser;
    }
}
