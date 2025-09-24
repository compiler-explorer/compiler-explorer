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

import _ from 'underscore';

import {AssemblyLine, BaseCFGParser, Range} from './base.js';

export class VcCFGParser extends BaseCFGParser {
    static override get key() {
        return 'vc';
    }

    override filterData(assembly: AssemblyLine[]): AssemblyLine[] {
        // Keep only lines between a line that starts with '... PROC' and a line that ends with '... ENDP'.
        // Remove lines that start with ';'
        const removeComment = (line: AssemblyLine) => {
            const pos = line.text.indexOf(';');
            let newText = line.text;
            if (pos !== -1) {
                newText = line.text.substring(0, pos).trimEnd();
            }
            return {...line, text: newText};
        };
        const noCommentLines = assembly.map(removeComment).filter(line => line.text.length > 0);
        const isFuncStart = (line: string) => {
            return line.endsWith(' PROC');
        };
        const isFuncEnd = (line: string) => {
            return line.endsWith(' ENDP');
        };

        const newRes: AssemblyLine[] = [];
        let inFunction = false;
        for (const line of noCommentLines) {
            if (isFuncStart(line.text)) {
                inFunction = true;
            }
            if (inFunction) {
                newRes.push(line);
            }
            if (isFuncEnd(line.text)) {
                inFunction = false;
            }
        }
        return newRes;
    }

    override splitToFunctions(asmArr: AssemblyLine[]): Range[] {
        if (asmArr.length === 0) return [];
        const result: Range[] = [];
        let cur = 1;
        const last = asmArr.length;
        const fnRange: Range = {start: 0, end: 0};
        do {
            if (this.isFunctionEnd(asmArr[cur].text)) {
                fnRange.end = cur + 1;
                result.push(_.clone(fnRange));
                fnRange.start = cur + 1;
            }
            ++cur;
        } while (cur < last);

        fnRange.end = last;
        if (fnRange.end > fnRange.start + 1) result.push(_.clone(fnRange));
        return result;
    }

    override isFunctionEnd(x: string) {
        return x.endsWith('ENDP');
    }
    override isBasicBlockEnd(inst: string, prevInst: string) {
        // Keep ENDP line in the same block as prevInst. Might drop it entirely.
        if (this.isFunctionEnd(inst)) return false;
        return inst[0] === '$';
    }

    override extractJmpTargetName(inst: string) {
        return inst.match(/\$.*/) + ':';
    }

    override getLabelSeparator() {
        // `@` is used natively by MSVC labels, so we use `#` instead
        return '#';
    }
}
