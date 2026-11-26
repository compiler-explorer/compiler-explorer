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

import {CompilationInfo} from '../../types/compilation/compilation.interfaces.js';
import {fileExists} from '../utils.js';
import {BaseTool} from './base-tool.js';

export class LLVMPDBUtilTool extends BaseTool {
    static get key() {
        return 'llvm-pdbutil-tool';
    }

    override async runTool(compilationInfo: CompilationInfo, inputFilepath?: string, args?: string[]) {
        if (!compilationInfo.filters.binary) {
            return this.createErrorResponse(`${this.tool.name ?? 'llvm-pdbutil'} requires an executable`);
        }

        let filename = compilationInfo.executableFilename;
        if (!(await fileExists(filename))) {
            filename = compilationInfo.outputFilename;
        }

        // If a PDB is generated, it will be named after the executable.
        const dotIdx = filename.lastIndexOf('.');
        if (dotIdx >= 0) {
            filename = filename.slice(0, dotIdx) + '.pdb';
        }

        return super.runTool(compilationInfo, filename, args);
    }
}
