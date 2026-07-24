// Copyright (c) 2026, Compiler Explorer Authors
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

import fs from 'node:fs/promises';
import path from 'node:path';

import {OptPipelineResults, Pass} from '../../types/compilation/opt-pipeline-output.interfaces.js';
import * as utils from '../utils.js';

const read = async filename => utils.splitLines(await fs.readFile(filename, 'utf8'));
const beforeMir = '.before.mir';

export async function parseMirPassDump(mirDumpDir: string): Promise<OptPipelineResults> {
    const results: Record<string, (Pass & {passId: string})[]> = {};
    for await (const filename of fs.glob(path.join(mirDumpDir, `/*${beforeMir}`))) {
        const fullName = path.basename(filename, beforeMir);
        const passName = path.extname(fullName);
        const {name: functionName, ext: passId} = path.parse(fullName.slice(0, -passName.length));
        const before = await read(filename);
        const after = await read(`${filename.slice(0, -beforeMir.length)}.after.mir`);

        let func = results[functionName];
        if (func === undefined) {
            func = results[functionName] = [];
        }
        func.push({
            passId: passId.slice(1),
            name: passName.slice(1),
            machine: false,
            before: before.map(line => ({text: line})),
            after: after.map(line => ({text: line})),
            irChanged: before.slice(1).join('\n') != after.slice(1).join('\n'),
        });
    }

    // sort by pass sequence number because `fs.glob` does not document an order (even though it seems to be alphabetically ordered)
    for (const passes of Object.values(results)) {
        passes.sort(({passId: lhs}, {passId: rhs}) => {
            if (lhs < rhs) return -1;
            if (lhs > rhs) return 1;
            return 0;
        });
    }

    return results;
}
