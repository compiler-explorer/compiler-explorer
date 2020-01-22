// Copyright (c) 2018, Compiler Explorer Authors
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
"use strict";

const
    BaseTool = require('./base-tool'),
    fs = require('fs-extra');

class LLVMMcaTool extends BaseTool {
    constructor(toolInfo, env) {
        super(toolInfo, env);
    }

    rewriteAsm(asm) {
        return asm
            .replace(/.hword\s/gim, '.short ')
            .replace(/OFFSET FLAT:/gim, '')
            .replace(/PTR\s%fs/gim, 'PTR fs');
    }

    writeAsmFile(data, destination) {
        return fs.writeFile(destination, this.rewriteAsm(data));
    }

    async runTool(compilationInfo, inputFilepath, args) {
        const extraArgs = [];
        if (compilationInfo.filters.intel) {
            extraArgs.push("--x86-asm-syntax=intel");
            extraArgs.push("-output-asm-variant=1");
        }

        if (compilationInfo.filters.binary) {
            return this.createErrorResponse("<cannot run analysis on binary>");
        }

        const rewrittenOutputFilename = compilationInfo.outputFilename + ".mca";
        await this.writeAsmFile(compilationInfo.asm, rewrittenOutputFilename);
        return super.runTool(compilationInfo, rewrittenOutputFilename, args);
    }
}

module.exports = LLVMMcaTool;
