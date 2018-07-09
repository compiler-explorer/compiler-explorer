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

const utils = require('./utils');

class LlvmIrParser {
    constructor(compilerProps) {
        // @todo More settings/compiler properties

        this.maxIrLines = 500;
        if (compilerProps) {
            this.maxIrLines = compilerProps('maxLinesOfAsm', this.maxIrLines);
        }

        this.debugReference = /!dbg (!\d+)/;
        this.metaNodeRe = /^(!\d+) = (distinct )?!DI([A-Za-z]+)\(([^)]+?)\)/;
    }

    getFileName(debugInfo, scope) {
        const stdInLooking = /.*<stdin>|^-$|example\.[^/]+$|<source>/;

        if (!debugInfo[scope]) {
            // No such meta info.
            return null;
        }
        // MetaInfo is a file node
        if (debugInfo[scope].filename) {
            const filename = debugInfo[scope].filename;
            return !filename.match(stdInLooking) ? filename : null;
        }
        // MetaInfo has a file reference.
        if (debugInfo[scope].file) {
            return this.getFileName(debugInfo, debugInfo[scope].file);
        }
        if (!debugInfo[scope].scope) {
            // No higher scope => can't find file.
            return null;
        }
        // "Bubbling" up.
        return this.getFileName(debugInfo, debugInfo[scope].scope);
    }

    getSourceLineNumber(debugInfo, scope) {
        if (!debugInfo[scope]) {
            return null;
        }
        if (debugInfo[scope].line) {
            return debugInfo[scope].line;
        }
        if (debugInfo[scope].scope) {
            return this.getSourceLineNumber(debugInfo, debugInfo[scope].scope);
        }
        return null;
    }

    processIr(ir, filters) {
        const result = [];
        let irLines = utils.splitLines(ir);
        let debugInfo = {};
        let prevLineEmpty = false;

        // Filters
        const commentOnly = /^\s*(;.*)$/;

        irLines.forEach(line => {
            let source = null;
            let match;

            if (!line.trim().length) {
                // Avoid multiple successive empty lines.
                if (!prevLineEmpty) {
                    result.push({text: "", source: null});
                }
                prevLineEmpty = true;
                return;
            }

            if (filters.commentOnly && line.match(commentOnly)) {
                return;
            }

            // Non-Meta IR line. Metadata is attached to it using "!dbg !123"
            match = line.match(this.debugReference);
            if (match) {
                result.push({text: (filters.trim ? utils.trimLine(line) : line), source: source, scope: match[1]});
                prevLineEmpty = false;
                return;
            }

            // Metadata Nodes
            // See: https://llvm.org/docs/LangRef.html#metadata
            // @todo: Multiline Metadata?
            // @todo: Named Metadata?
            match = line.match(this.metaNodeRe);
            if (match) {
                const metaId = match[1];
                const metaName = match[3];

                let options = {};
                const optionWithValue = match[4].split(',');
                for (let i = 0; i < optionWithValue.length; ++i) {
                    const pair = optionWithValue[i].split(':');
                    options[pair[0].trim()] = pair[1].trim();
                }
                options.metaId = match[1];
                debugInfo[metaId] = options;

                if (metaName === "File" && options.filename) {
                    // Remove quotes
                    debugInfo[metaId].filename = options.filename.substr(1, options.filename.length - 2);
                }
            }

            if (filters.directives && this.isLineLlvmDirective(line)) {
                return;
            }
            result.push({text: (filters.trim ? utils.trimLine(line) : line), source: source});
            prevLineEmpty = false;
        });

        if (result.length >= this.maxIrLines) {
            result.length = this.maxIrLines + 1;
            result[this.maxIrLines] = {text: "[truncated; too many lines]", source: null};
        }
       
        result.forEach(line => {
            if (!line.scope) return;
            // @todo: Nested Scopes?
            line.source = {
                file: this.getFileName(debugInfo, line.scope),
                line: this.getSourceLineNumber(debugInfo, line.scope)
            };
        });

        return result;
    }

    process(ir, filters) {
        return this.processIr(ir, filters);
    }

    isLineLlvmDirective(line) {
        return line.match(/^!\d+ = !{/)
            || line.match(/^!\d+ = (distinct )?!DI/)
            || line.match(/^!llvm./)
            || line.match(/^source_filename = /)
            || line.match(/^target datalayout = /)
            || line.match(/^target triple = /);
    }

    isLlvmIr(code) {
        return code.includes("@llvm") && code.includes("!DI") && code.includes("!dbg");
    }
}

module.exports = {
    LlvmIrParser: LlvmIrParser
};
