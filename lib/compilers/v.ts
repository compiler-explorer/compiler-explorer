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

import path from 'path';
import _ from 'underscore';

import {unwrap} from '../assert.js';
import {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';
import {ParsedAsmResultLine} from '../../types/asmresult/asmresult.interfaces.js';

export class VCompiler extends BaseCompiler {
    static get key() {
        return 'v';
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string, userOptions?: string[]) {
        if (_.some(unwrap(userOptions), opt => opt === '--help' || opt === '-h')) {
            return [];
        } else if (!filters.binary) {
            return ['-o', this.filename(outputFilename), '-skip-unused'];
        }
        return ['-skip-unused'];
    }

    override async objdump(
        outputFilename,
        result: any,
        maxSize: number,
        intelAsm,
        demangle,
        staticReloc: boolean,
        dynamicReloc: boolean,
        filters: ParseFiltersAndOutputOptions,
    ) {
        const objdumpResult = await super.objdump(
            outputFilename,
            result,
            maxSize,
            intelAsm,
            demangle,
            staticReloc,
            dynamicReloc,
            filters,
        );

        objdumpResult.languageId = 'C';
        return objdumpResult;
    }

    override async processAsm(result: any, filters: any, options: any): Promise<any> {
        const lineRe = /^.*main__.*$/;
        const mainFunctionCall = '\tmain__main();';

        const cCodeLines = result.asm.split('\n');
        const cCodeResult: ParsedAsmResultLine[] = [];

        let scopeDepth = 0;
        let insertNewLine = false;

        for (const lineNo in cCodeLines) {
            const line = cCodeLines[lineNo];
            if (!line) continue;

            if (insertNewLine) {
                cCodeResult.push({text: ''});
                insertNewLine = false;
            }

            if ((scopeDepth === 0 && line.match(lineRe) && line !== mainFunctionCall) || scopeDepth > 0) {
                const opening = (line.match(/{/g) || []).length - 1;
                const closing = (line.match(/}/g) || []).length - 1;
                scopeDepth += opening - closing;

                cCodeResult.push({text: line});

                insertNewLine = scopeDepth === 0;
            }
        }

        return {asm: cCodeResult};
    }

    override getSharedLibraryPathsAsArguments(libraries, libDownloadPath) {
        return [];
    }

    override getSharedLibraryLinks(libraries: any[]): string[] {
        return [];
    }

    override getOutputFilename(dirPath: string, outputFilebase: string, key?: any): string {
        return path.join(dirPath, 'example.c');
    }
}
