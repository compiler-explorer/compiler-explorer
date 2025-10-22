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

import {AssemblyLine, BaseCFGParser} from './base.js';

export class GccCFGParser extends BaseCFGParser {
    static override get key() {
        return 'gcc';
    }

    override filterData(assembly: AssemblyLine[]) {
        const isInstruction = (x: string) => !x.startsWith('#') && !x.match(/^\s+\./) && !x.match(/^\s*$/);
        const isFunctionName = (x: string) => x.endsWith(':') && x.includes('(') && x.includes(')');
        const res: AssemblyLine[] = [];
        // preserve only labels that contain actual instructions, and not just comments/directives
        let sawInstInLabel = false;
        let curLabel: AssemblyLine | null = null;
        for (const line of assembly) {
            if (isFunctionName(line.text)) {
                res.push(line);
                curLabel = line;
                sawInstInLabel = false;
                continue;
            }
            if (line.text.trim().endsWith(':')) {
                // A label
                curLabel = line;
                sawInstInLabel = false;
                continue;
            }
            if (isInstruction(line.text)) {
                if (!sawInstInLabel) {
                    // First actual instruction in this label
                    res.push(curLabel!);
                    sawInstInLabel = true;
                }
                res.push(line);
            }
        }
        return res;
    }

    override extractJmpTargetName(inst: string) {
        return inst.match(/\.L\d+/) + ':';
    }
}
