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

import type {OptPipelineResults, Pass} from '../../types/compilation/opt-pipeline-output.interfaces.js';
import type {ResultLine} from '../../types/resultline/resultline.interfaces.js';

const passHeader = /^-*(.+): (?:[^:]+): AFTER (.+?)-*$/;
const firstPass = 'translate_bytecode';

function sameLines(left: ResultLine[], right: ResultLine[]): boolean {
    if (left.length !== right.length) return false;
    for (let i = 0; i < left.length; i++) {
        if (left[i].text !== right[i].text) return false;
    }
    return true;
}

function freshKey(results: OptPipelineResults, name: string): string {
    if (!(name in results)) return name;
    let n = 2;
    let key = `${name} [${n}]`;
    while (key in results) {
        n++;
        key = `${name} [${n}]`;
    }
    return key;
}

export class NumbaPassDumpParser {
    process(output: ResultLine[]): OptPipelineResults {
        const results: OptPipelineResults = Object.create(null);
        const groupFor = new Map<string, string>();
        let functionName: string | undefined;
        let passName: string | undefined;
        let lines: ResultLine[] = [];

        const flush = () => {
            if (!functionName || !passName) return;
            const key = groupFor.get(functionName);
            if (!key) return;
            const passes = results[key];
            const before = passes.length > 0 ? passes[passes.length - 1].after : [];
            const pass: Pass = {
                name: passName,
                machine: false,
                before,
                after: lines,
                irChanged: !sameLines(before, lines),
            };
            passes.push(pass);
        };

        for (const line of output) {
            const match = passHeader.exec(line.text);
            if (!match) {
                if (passName) lines.push(line);
                continue;
            }
            flush();
            functionName = match[1];
            passName = match[2];
            lines = [];
            let key = groupFor.get(functionName);
            if (key && passName === firstPass && results[key].length > 0) key = undefined;
            if (!key) {
                key = freshKey(results, functionName);
                groupFor.set(functionName, key);
                results[key] = [];
            }
        }
        flush();
        return results;
    }
}
