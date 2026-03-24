// Copyright (c) 2016, Compiler Explorer Authors
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

import path from 'node:path';

import {OptRemark} from '../../static/panes/opt-view.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';
import * as utils from '../utils.js';

export class GCCCompiler extends BaseCompiler {
    static get key() {
        return 'gcc';
    }

    override getOptFilePath(dirPath: string, outputFilebase: string): string {
        return path.join(dirPath, 'all.opt');
    }

    override processRawOptRemarks(buffer: string, compileFileName = ''): OptRemark[] {
        const remarks: OptRemark[] = [];

        // example stderr lines:
        // <source>:3:20: optimized: loop vectorized using 8 byte vectors
        // <source>: 2: 6: note: vectorized 1 loops in function.
        // <source>:11:13: missed: statement clobbers memory: somefunc (&i);
        // BB 3 is always executed in loop 1

        const remarkRegex = /^(.*?):\s*(\d+):\s*(\d+): (.*?): (.*)$/;

        const mapOptType = (type: string): 'Missed' | 'Passed' | 'Analysis' => {
            if (type === 'missed') return 'Missed';
            if (type === 'optimized') return 'Passed';
            return 'Analysis'; // for type 'note' or empty
        };

        utils.eachLine(buffer, line => {
            const match = line.match(remarkRegex);
            if (match) {
                const [_, file, lineNum, colNum, type, message] = match;
                // Filter out opt-remarks for included header files
                if (file.includes(compileFileName) || file.includes('<source>')) {
                    // opt-remark with line/col info
                    // convert to llvm-emitted OptRemark format, just because it was here first
                    remarks.push({
                        DebugLoc: {
                            File: file,
                            // Could use line.tag for these too:
                            Line: Number.parseInt(lineNum, 10),
                            Column: Number.parseInt(colNum, 10),
                        },
                        optType: mapOptType(type),
                        displayString: message,
                        // TODO: make these optional?
                        Function: '',
                        Pass: '',
                        Name: '',
                        Args: [],
                    });
                }
            } else {
                // opt-remark without line/col info
                remarks.push({
                    DebugLoc: {File: '', Line: -1, Column: -1},
                    optType: 'Analysis',
                    displayString: line,
                    Function: '',
                    Pass: '',
                    Name: '',
                    Args: [],
                });
            }
        });

        return remarks;
    }
}
