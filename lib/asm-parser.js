// Copyright (c) 2015, Matt Godbolt
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

const _ = require('underscore'),
    utils = require('./utils'),
    AsmRegex = require('./asmregex').AsmRegex;

class AsmParser extends AsmRegex {
    constructor(compilerProps) {
        super();

        this.labelFindNonMips = /[.a-zA-Z_][a-zA-Z0-9$_.]*/g;
        // MIPS labels can start with a $ sign, but other assemblers use $ to mean literal.
        this.labelFindMips = /[$.a-zA-Z_][a-zA-Z0-9$_.]*/g;
        this.mipsLabelDefinition = /^\$[a-zA-Z0-9$_.]+:/;
        this.dataDefn = /^\s*\.(string|asciz|ascii|[1248]?byte|short|x?word|long|quad|value|zero)/;
        this.fileFind = /^\s*\.file\s+(\d+)\s+"([^"]+)"(\s+"([^"]+)")?.*/;
        this.hasOpcodeRe = /^\s*[a-zA-Z]/;
        this.definesFunction = /^\s*\.type.*,\s*[@%]function$/;
        this.definesGlobal = /^\s*\.globa?l\s*([.a-zA-Z_][a-zA-Z0-9$_.]*)/;
        this.indentedLabelDef = /^\s*([.a-zA-Z_$][a-zA-Z0-9$_.]*):/;
        this.assignmentDef = /^\s*([.a-zA-Z_$][a-zA-Z0-9$_.]+)\s*=/;
        this.directive = /^\s*\..*$/;
        this.startAppBlock = /\s*#APP.*/;
        this.endAppBlock = /\s*#NO_APP.*/;
        this.startAsmNesting = /\s*# Begin ASM.*/;
        this.endAsmNesting = /\s*# End ASM.*/;
        this.cudaBeginDef = /\.(entry|func)\s+(?:\([^)]*\)\s*)?([.a-zA-Z_$][a-zA-Z0-9$_.]*)\($/;
        this.cudaEndDef = /^\s*\)\s*$/;

        this.binaryHideFuncRe = null;
        this.maxAsmLines = 500;
        this.inNvccDef = false;
        if (compilerProps) {
            const binaryHideFuncReValue = compilerProps('binaryHideFuncRe');
            if (binaryHideFuncReValue) {
                this.binaryHideFuncRe = new RegExp(binaryHideFuncReValue);
            }

            this.maxAsmLines = compilerProps('maxLinesOfAsm', this.maxAsmLines);
        }

        this.asmOpcodeRe = /^\s*([0-9a-f]+):\s*(([0-9a-f][0-9a-f] ?)+)\s*(.*)/;
        this.lineRe = /^(\/[^:]+):([0-9]+).*/;
        this.labelRe = /^([0-9a-f]+)\s+<([^>]+)>:$/;
        this.destRe = /\s([0-9a-f]+)\s+<([^>]+)>$/;
    }

    hasOpcode(line) {
        // Remove any leading label definition...
        const match = line.match(this.labelDef);
        if (match) {
            line = line.substr(match[0].length);
        }
        // Strip any comments
        line = line.split(/[#;]/, 1)[0];
        // Detect assignment, that's not an opcode...
        if (line.match(this.assignmentDef)) return false;
        return !!line.match(this.hasOpcodeRe);
    }

    labelFindFor(asmLines) {
        const isMips = _.any(asmLines, line => !!line.match(this.mipsLabelDefinition));
        return isMips ? this.labelFindMips : this.labelFindNonMips;
    }

    findUsedLabels(asmLines, filterDirectives) {
        const labelsUsed = {};
        const weakUsages = {};
        const labelFind = this.labelFindFor(asmLines);
        // The current label set is the set of labels all pointing at the current code, so:
        // foo:
        // bar:
        //    add r0, r0, #1
        // in this case [foo, bar] would be the label set for the add instruction.
        let currentLabelSet = [];
        let inLabelGroup = false;
        let inCustomAssembly = 0;

        // Scan through looking for definite label usages (ones used by opcodes),
        // and ones that are weakly used: that is, their use is conditional on another label.
        // For example:
        // .foo: .string "moo"
        // .baz: .quad .foo
        //       mov eax, .baz
        // In this case, the '.baz' is used by an opcode, and so is strongly used.
        // The '.foo' is weakly used by .baz.
        asmLines.forEach(line => {
            if (line.match(this.startAppBlock) || line.match(this.startAsmNesting)) {
                inCustomAssembly++;
            } else if (line.match(this.endAppBlock) || line.match(this.endAsmNesting)) {
                inCustomAssembly--;
            }

            if (inCustomAssembly > 0)
                line = this.fixLabelIndentation(line);

            let match = line.match(this.labelDef);
            if (match) {
                if (inLabelGroup)
                    currentLabelSet.push(match[1]);
                else
                    currentLabelSet = [match[1]];
                inLabelGroup = true;
            } else {
                inLabelGroup = false;
            }
            match = line.match(this.definesGlobal);
            if (!match)
                match = line.match(this.cudaBeginDef);
            if (match) {
                labelsUsed[match[1]] = true;
            }

            if (!line || line[0] === '.') return;

            match = line.match(labelFind);
            if (!match) return;

            if (!filterDirectives || this.hasOpcode(line) || line.match(this.definesFunction)) {
                // Only count a label as used if it's used by an opcode, or else we're not filtering directives.
                match.forEach(label => labelsUsed[label] = true);
            } else {
                // If we have a current label, then any subsequent opcode or data definition's labels are referred to
                // weakly by that label.
                const isDataDefinition = !!line.match(this.dataDefn);
                const isOpcode = this.hasOpcode(line);
                if (isDataDefinition || isOpcode) {
                    currentLabelSet.forEach(currentLabel => {
                        if (!weakUsages[currentLabel]) weakUsages[currentLabel] = [];
                        match.forEach(label => weakUsages[currentLabel].push(label));
                    });
                }
            }
        });

        // Now follow the chains of used labels, marking any weak references they refer
        // to as also used. We iteratively do this until either no new labels are found,
        // or we hit a limit (only here to prevent a pathological case from hanging).
        function markUsed(label) {
            labelsUsed[label] = true;
        }

        const MaxLabelIterations = 10;
        for (let iter = 0; iter < MaxLabelIterations; ++iter) {
            let toAdd = [];
            _.each(labelsUsed, (t, label) => { // jshint ignore:line
                _.each(weakUsages[label], nowused => {
                    if (labelsUsed[nowused]) return;
                    toAdd.push(nowused);
                });
            });
            if (!toAdd) break;
            _.each(toAdd, markUsed);
        }
        return labelsUsed;
    }

    parseFiles(asmLines) {
        const files = {};
        asmLines.forEach(line => {
            const match = line.match(this.fileFind);
            if (match) {
                const lineNum = parseInt(match[1]);
                if (match[3]) {
                    // Clang-style file directive '.file X "dir" "filename"'
                    files[lineNum] = match[2] + "/" + match[3];
                } else {
                    files[lineNum] = match[2];
                }
            }
        });
        return files;
    }

    processAsm(asm, filters) {
        if (filters.binary) return this.processBinaryAsm(asm, filters);

        if (filters.commentOnly) {
            // Remove any block comments that start and end on a line if we're removing comment-only lines.
            const blockComments = /^\s*\/\*(\*(?!\/)|[^*])*\*\/\s*\r?\n/mg;
            asm = asm.replace(blockComments, "");
        }

        const result = [];
        let asmLines = utils.splitLines(asm);
        if (filters.preProcessLines !== undefined) {
            asmLines = filters.preProcessLines(asmLines);
        }

        const labelsUsed = this.findUsedLabels(asmLines, filters.directives);
        const files = this.parseFiles(asmLines);
        let prevLabel = "";

        const commentOnly = /^\s*(((#|@|;|\/\/).*)|(\/\*.*\*\/))$/;
        const sourceTag = /^\s*\.loc\s+(\d+)\s+(\d+).*/;
        const sourceStab = /^\s*\.stabn\s+(\d+),0,(\d+),.*/;
        const stdInLooking = /<stdin>|^-$|example\.[^/]+$|<source>/;
        const endBlock = /\.(cfi_endproc|data|text|section)/;
        let source = null;

        let inCustomAssembly = 0;
        asmLines.forEach(line => {
            let match;
            if (line.trim() === "") {
                result.push({text: "", source: null});
                return;
            }

            if (line.match(this.startAppBlock) || line.match(this.startAsmNesting)) {
                inCustomAssembly++;
            } else if (line.match(this.endAppBlock) || line.match(this.endAsmNesting)) {
                inCustomAssembly--;
            }
            match = line.match(sourceTag);
            if (match) {
                const file = files[parseInt(match[1])];
                const sourceLine = parseInt(match[2]);
                if (file) {
                    source = {
                        file: !file.match(stdInLooking) ? file : null,
                        line: sourceLine
                    };
                } else {
                    source = null;
                }
            }
            match = line.match(sourceStab);
            if (match) {
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
            if (line.match(endBlock)) {
                source = null;
                prevLabel = null;
            }

            if (filters.commentOnly && line.match(commentOnly)) return;

            if (inCustomAssembly > 0)
                line = this.fixLabelIndentation(line);

            match = line.match(this.labelDef);
            if (!match) match = line.match(this.assignmentDef);
            if (!match) {
                match = line.match(this.cudaBeginDef);
                if (match)
                    this.inNvccDef = true;
            }
            if (match) {
                // It's a label definition.
                if (labelsUsed[match[1]] === undefined) {
                    // It's an unused label.
                    if (filters.labels) return;
                } else {
                    // A used label.
                    prevLabel = match;
                }
            }
            if (this.inNvccDef) {
                if (line.match(this.cudaEndDef))
                    this.inNvccDef = false;
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
            result.push({text: AsmRegex.filterAsmLine(line, filters), source: this.hasOpcode(line) ? source : null});
        });
        return result;
    }

    fixLabelIndentation(line) {
        const match = line.match(this.indentedLabelDef);
        if (match) {
            return line.replace(/^\s+/, "");
        } else {
            return line;
        }
    }

    isUserFunction(func) {
        if (this.binaryHideFuncRe === null) return true;

        return !func.match(this.binaryHideFuncRe);
    }

    processBinaryAsm(asm, filters) {
        const result = [];
        let asmLines = asm.split("\n");
        let source = null;
        let func = null;

        // Handle "error" documents.
        if (asmLines.length === 1 && asmLines[0][0] === '<') {
            return [{text: asmLines[0], source: null}];
        }

        if (filters.preProcessBinaryAsmLines !== undefined) {
            asmLines = filters.preProcessBinaryAsmLines(asmLines);
        }

        asmLines.forEach(line => {
            if (result.length >= this.maxAsmLines) {
                if (result.length === this.maxAsmLines) {
                    result.push({text: "[truncated; too many lines]", source: null});
                }
                return;
            }
            let match = line.match(this.lineRe);
            if (match) {
                source = {file: null, line: parseInt(match[2])};
                return;
            }

            match = line.match(this.labelRe);
            if (match) {
                func = match[2];
                if (this.isUserFunction(func)) {
                    result.push({text: func + ":", source: null});
                }
                return;
            }

            if (!func || !this.isUserFunction(func)) return;

            match = line.match(this.asmOpcodeRe);
            if (match) {
                const address = parseInt(match[1], 16);
                const opcodes = match[2].split(" ").filter(x => !!x);
                const disassembly = " " + AsmRegex.filterAsmLine(match[4], filters);
                let links = null;
                const destMatch = line.match(this.destRe);
                if (destMatch) {
                    links = [{
                        offset: disassembly.indexOf(destMatch[1]),
                        length: destMatch[1].length,
                        to: parseInt(destMatch[1], 16)
                    }];
                }
                result.push({opcodes: opcodes, address: address, text: disassembly, source: source, links: links});
            }
        });
        return result;
    }

    process(asm, filters) {
        return this.processAsm(asm, filters);
    }
}

module.exports = AsmParser;
