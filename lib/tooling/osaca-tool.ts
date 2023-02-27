// Copyright (c) 2023, Compiler Explorer Authors
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

import fs from 'fs-extra';

import {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {AsmParser} from '../parsers/asm-parser.js';

import {BaseTool} from './base-tool.js';

export class OSACATool extends BaseTool {
    static get key() {
        return 'osaca-tool';
    }

    async writeAsmFile(asmParser: AsmParser, asm: string, filters: ParseFiltersAndOutputOptions, destination: string) {
        // Applying same filters as applied to compiler outpu
        const filteredAsm = asmParser.process(asm, filters).asm.reduce(function (acc, line) {
            return acc + line.text + '\n';
        }, '');
        return fs.writeFile(destination, filteredAsm);
    }

    override async runTool(compilationInfo: Record<any, any>, inputFilepath?: string, args?: string[]) {
        if (compilationInfo.filters.binary) {
            return this.createErrorResponse('<cannot run analysis on binary>');
        }

        if (compilationInfo.filters.intel) {
            return this.createErrorResponse('<cannot run analysis on Intel assembly>');
        }

        const rewrittenOutputFilename = compilationInfo.outputFilename + '.osaca';
        await this.writeAsmFile(
            compilationInfo.asmParser,
            compilationInfo.asm,
            compilationInfo.filters,
            rewrittenOutputFilename,
        );
        return super.runTool(compilationInfo, rewrittenOutputFilename, args);
    }
}
