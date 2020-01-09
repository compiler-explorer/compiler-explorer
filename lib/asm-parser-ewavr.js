// Copyright (c) 2019, Ethan Slattery
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

const AsmParserBase = require('./asm-parser'),
    utils = require('./utils'),
    AsmRegex = require('./asmregex').AsmRegex;

class AsmEWAVRParser extends AsmParserBase {
    processAsm(asmResult, filters) {
        if (filters.binary) return this.processBinaryAsm(asmResult, filters);

        if (filters.commentOnly) {
            // Remove any block comments that start and end on a line if we're removing comment-only lines.
            const blockComments = /^[ \t]*\/\*(\*(?!\/)|[^*])*\*\/\s*/mg;
            asmResult = asmResult.replace(blockComments, "");
        }

        const asm = [];
        const labelDefinitions = {};

        let asmLines = utils.splitLines(asmResult);
        if (filters.preProcessLines !== undefined) {
            asmLines = filters.preProcessLines(asmLines);
        }

        const labelsUsed = this.findUsedLabels(asmLines, filters.directives);
        const files = this.parseFiles(asmLines);
        let prevLabel = "";

        // Lines matching the following pattern are considered comments:
        // - starts with '#', '@', '//' or a single ';' (non repeated)
        // - starts with ';;' and has non-whitespace before end of line
        const commentOnly = /^\s*(((#|@|\/\/).*)|(\/\*.*\*\/)|(;\s*)|(;[^;].*)|(;;.*\S.*))$/;

        const commentOnlyNvcc = /^\s*(((#|;|\/\/).*)|(\/\*.*\*\/))$/;
        const sourceTag = /^\/\/\s*([0-9]+)\s(?!bytes).*/;
        const source6502Dbg = /^\s*\.dbg\s+line,\s*"([^"]+)",\s*(\d+)/;
        const source6502DbgEnd = /^\s*\.dbg\s+line[^,]/;
        const sourceStab = /^\s*\.stabn\s+(\d+),0,(\d+),.*/;
        const stdInLooking = /<stdin>|^-$|example\.[^/]+$|<source>/;
        const endBlock = /\.(cfi_endproc|data|text|section)/;
        let source = null;
        let mayRemovePreviousLabel = true;
        let keepInlineCode = false;

        let lastOwnSource = null;

        function maybeAddBlank() {
            const lastBlank = asm.length === 0 || asm[asm.length - 1].text === "";
            if (!lastBlank)
                asm.push({text: "", source: null, labels: []});
        }

        function handleSource(line) {
            const match = line.match(sourceTag);
            if (match) {
                const sourceLine = parseInt(match[1]);
                source = {
                    file: null,
                    line: sourceLine
                };
            }
        }

        function handleStabs(line) {
            const match = line.match(sourceStab);
            if (!match) return;
            // cf http://www.math.utah.edu/docs/info/stabs_11.html#SEC48
            switch (parseInt(match[1])) {
                case 68:
                    source = {file: null, line: parseInt(match[2])};
                    break;
                case 132:
                case 100:
                    source = null;
                    prevLabel = null;
                    break;
            }
        }

        function handle6502(line) {
            const match = line.match(source6502Dbg);
            if (match) {
                const file = match[1];
                const sourceLine = parseInt(match[2]);
                source = {
                    file: !file.match(stdInLooking) ? file : null,
                    line: sourceLine
                };
            } else if (line.match(source6502DbgEnd)) {
                source = null;
            }
        }

        let inNvccDef = false;
        let inNvccCode = false;

        let inCustomAssembly = 0;
        asmLines.forEach(line => {
            if (line.trim() === "") return maybeAddBlank();

            if (line.match(this.startAppBlock) || line.match(this.startAsmNesting)) {
                inCustomAssembly++;
            } else if (line.match(this.endAppBlock) || line.match(this.endAsmNesting)) {
                inCustomAssembly--;
            }

            handleSource(line);
            handleStabs(line);
            handle6502(line);

            if (source && source.file === null) {
                lastOwnSource = source;
            }

            if (line.match(endBlock) || (inNvccCode && line.match(/}/))) {
                source = null;
                prevLabel = null;
                lastOwnSource = null;
            }

            if (filters.libraryCode && !lastOwnSource && source && source.file !== null) {
                if (mayRemovePreviousLabel && asm.length > 0) {
                    const lastLine = asm[asm.length - 1];

                    const labelDef = lastLine.text
                        ? lastLine.text.match(this.labelDef) : null;

                    if (labelDef) {
                        asm.pop();
                        keepInlineCode = false;
                        delete labelDefinitions[labelDef[1]];
                    } else {
                        keepInlineCode = true;
                    }
                    mayRemovePreviousLabel = false;
                }

                if (!keepInlineCode) return;
            } else {
                mayRemovePreviousLabel = true;
            }

            if (filters.commentOnly &&
                ((line.match(commentOnly) && !inNvccCode) ||
                    (line.match(commentOnlyNvcc) && inNvccCode))
            ) {
                return;
            }

            if (inCustomAssembly > 0)
                line = this.fixLabelIndentation(line);

            let match = line.match(this.labelDef);
            if (!match) match = line.match(this.assignmentDef);
            if (!match) {
                match = line.match(this.cudaBeginDef);
                if (match) {
                    inNvccDef = true;
                    inNvccCode = true;
                }
            }
            if (match) {
                // It's a label definition.
                if (labelsUsed[match[1]] === undefined) {
                    // It's an unused label.
                    if (filters.labels) return;
                } else {
                    // A used label.
                    prevLabel = match;
                    labelDefinitions[match[1]] = asm.length + 1;
                }
            }
            if (inNvccDef) {
                if (line.match(this.cudaEndDef))
                    inNvccDef = false;
            } else if (!match && filters.directives) {
                // Check for directives only if it wasn't a label; the regexp would
                // otherwise misinterpret labels as directives.
                if (line.match(this.dataDefn) && prevLabel) {
                    // We're defining data that's being used somewhere.
                } else {
                    if (line.match(this.directive)) return;
                }
            }

            line = utils.expandTabs(line);
            const text = AsmRegex.filterAsmLine(line, filters);

            const labelsInLine = match ? [] : this.getUsedLabelsInLine(text);

            asm.push({
                text: text,
                source: this.hasOpcode(line, inNvccCode) ? source : null,
                labels: labelsInLine
            });
        });

        this.removeLabelsWithoutDefinition(asm, labelDefinitions);

        return {
            asm: asm,
            labelDefinitions: labelDefinitions
        };
    }
}

module.exports = AsmEWAVRParser;
