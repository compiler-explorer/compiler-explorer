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

import {AsmResultSource, ParsedAsmResult, ParsedAsmResultLine} from '../../types/asmresult/asmresult.interfaces';
import {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces';
import {PropertyGetter} from '../properties.interfaces';
import * as utils from '../utils';

import {AsmParser} from './asm-parser';

export class TurboCAsmParser extends AsmParser {
    private readonly asmBinaryParser: AsmParser;
    private readonly filestart = /^\s+\?debug\s+S\s"(.+)"/;
    private readonly linestart = /^;\s+\?debug\s+L\s(\d+)/;
    private readonly procbegin = /^(\w+)\sproc\s*near/;
    private readonly procend = /^(\w+)\sendp/;

    constructor(compilerProps: PropertyGetter) {
        super(compilerProps);
        this.asmBinaryParser = new AsmParser(compilerProps);
    }

    override processAsm(asm: string, filters: ParseFiltersAndOutputOptions): ParsedAsmResult {
        if (filters.binary) return this.asmBinaryParser.processBinaryAsm(asm, filters);

        let currentfile = '';
        let currentline: string | undefined;
        let currentproc = '';

        const asmLines: ParsedAsmResultLine[] = [];
        let isDirective = true;
        asm = asm.replace(/\u001A$/, '');

        utils.eachLine(asm, line => {
            const procmatch = line.match(this.procbegin);
            if (procmatch) {
                currentproc = procmatch[1];
                isDirective = false;
            }

            const endprocmatch = line.match(this.procend);
            if (endprocmatch) {
                currentproc = '';
                currentline = undefined;
                isDirective = false;
            }

            const filematch = line.match(this.filestart);
            if (filematch) {
                currentfile = filematch[1];
                isDirective = true;
            }

            const linematch = line.match(this.linestart);
            if (linematch) {
                currentline = linematch[1];
                isDirective = true;
            }

            let source: AsmResultSource | null = null;
            if (currentfile && currentline) {
                if (filters.dontMaskFilenames) {
                    source = {
                        file: currentfile,
                        line: parseInt(currentline),
                    };
                } else {
                    source = {
                        file: null,
                        line: parseInt(currentline),
                    };
                }
            }

            if (currentproc) {
                if (filters.directives && isDirective) {
                    isDirective = false;
                    return;
                }

                asmLines.push({
                    text: line,
                    source,
                });
            } else if (!filters.directives || !isDirective) {
                isDirective = true;

                asmLines.push({
                    text: line,
                    source,
                });
            }
        });

        return {
            asm: asmLines,
        };
    }
}
