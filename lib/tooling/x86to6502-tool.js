// Copyright (c) 2019, Compiler Explorer Team
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
    _ = require('underscore'),
    AsmParser = require('../asm-parser.js');

class x86to6502Tool extends BaseTool {
    constructor(toolInfo, env) {
        super(toolInfo, env);
    }

    runTool(compilationInfo, sourcefile, args) {
        if (compilationInfo.filters.intel) {
            return new Promise((resolve) => {
                resolve(this.createErrorResponse("<need AT&T notation assembly>"));
            });
        }

        if (compilationInfo.filters.binary) {
            return new Promise((resolve) => {
                resolve(this.createErrorResponse("<cannot run x86to6502 on binary>"));
            });
        }

        const parser = new AsmParser();
        const filters = Object.assign({}, compilationInfo.filters);

        const result = parser.process(compilationInfo.asm, filters);

        const asm = _.map(result.asm, (obj) => {
            if (typeof obj.text !== "string" || obj.text.trim() === "") {
                return "";
            } else if (obj.text.match(/.*:/)) {
                return obj.text.replace(/^\s*/, "");
            } else {
                return obj.text.replace(/^\s*/, "\t");
            }
        }).join("\n");

        return super.runTool(compilationInfo, false, args, asm);
    }
}

module.exports = x86to6502Tool;
