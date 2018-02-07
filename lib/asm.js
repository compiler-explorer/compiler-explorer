// Copyright (c) 2012-2018, Matt Godbolt
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

const _ = require('underscore-node'),
    utils = require('./utils');

function AsmParser(compilerProps) {
    const labelFindNonMips = /[.a-zA-Z_][a-zA-Z0-9$_.]*/g;
    // MIPS labels can start with a $ sign, but other assemblersuse $ to mean literal.
    const labelFindMips = /[$.a-zA-Z_][a-zA-Z0-9$_.]*/g;
    const mipsLabelDefinition = /^\$[a-zA-Z0-9$_.]+:/;
    const dataDefn = /^\s*\.(string|asciz|ascii|[1248]?byte|short|word|long|quad|value|zero)/;
    const fileFind = /^\s*\.file\s+(\d+)\s+"([^"]+)"(\s+"([^"]+)")?.*/;
    const hasOpcodeRe = /^\s*[a-zA-Z]/;
    const definesFunction = /^\s*\.type.*,\s*[@%]function$/;
    const definesGlobal = /^\s*\.globa?l\s*([.a-zA-Z_][a-zA-Z0-9$_.]*)/;
    const labelDef = /^([.a-zA-Z_$][a-zA-Z0-9$_.]*):/;
    const indentedLabelDef = /^\s*([.a-zA-Z_$][a-zA-Z0-9$_.]*):/;
    const assignmentDef = /^\s*([.a-zA-Z_$][a-zA-Z0-9$_.]+)\s*=/;
    const directive = /^\s*\..*$/;
    const startAppBlock = /\s*\#APP.*/;
    const endAppBlock = /\s*\#NO_APP.*/;
    const startAsmNesting = /\s*\# Begin ASM.*/;
    const endAsmNesting = /\s*\# End ASM.*/;

    function hasOpcode(line) {
        // Remove any leading label definition...
        const match = line.match(labelDef);
        if (match) {
            line = line.substr(match[0].length);
        }
        // Strip any comments
        line = line.split(/[#;]/, 1)[0];
        // Detect assignment, that's not an opcode...
        if (line.match(assignmentDef)) return false;
        return !!line.match(hasOpcodeRe);
    }

    function filterAsmLine(line, filters) {
        if (!filters.trim) return line;
        const splat = line.split(/\s+/);
        if (splat[0] === "") {
            // An indented line: preserve a two-space indent
            return "  " + splat.slice(1).join(" ");
        } else {
            return splat.join(" ");
        }
    }

    function labelFindFor(asmLines) {
        const isMips = _.any(asmLines, line => !!line.match(mipsLabelDefinition));
        return isMips ? labelFindMips : labelFindNonMips;
    }

    function findUsedLabels(asmLines, filterDirectives) {
        const labelsUsed = {};
        const weakUsages = {};
        const labelFind = labelFindFor(asmLines);
        let currentLabel = "";
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
            if (line.match(startAppBlock) || line.match(startAsmNesting)) {
                inCustomAssembly++;
            } else if (line.match(endAppBlock) || line.match(endAsmNesting)) {
                inCustomAssembly--;
            }

            if (inCustomAssembly > 0)
                line = fixLabelIndentation(line);

            let match = line.match(labelDef);
            if (match)
                currentLabel = match[1];
            match = line.match(definesGlobal);
            if (match) {
                labelsUsed[match[1]] = true;
            }
            if (!line || line[0] === '.') return;

            match = line.match(labelFind);
            if (!match) return;

            if (!filterDirectives || hasOpcode(line) || line.match(definesFunction)) {
                // Only count a label as used if it's used by an opcode, or else we're not filtering directives.
                match.forEach(label => labelsUsed[label] = true);
            } else if (currentLabel) {
                // If we have a current label, then any subsequent opcode or data definition's labels are refered to
                // weakly by that label.
                const isDataDefinition = !!line.match(dataDefn);
                const isOpcode = hasOpcode(line);
                if (isDataDefinition || isOpcode) {
                    if (!weakUsages[currentLabel]) weakUsages[currentLabel] = [];
                    match.forEach(label => weakUsages[currentLabel].push(label));
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

    function parseFiles(asmLines) {
        const files = {};
        asmLines.forEach(line => {
            const match = line.match(fileFind);
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

    function processAsm(asm, filters) {
        if (filters.binary) return processBinaryAsm(asm, filters);

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

        const labelsUsed = findUsedLabels(asmLines, filters.directives);
        const files = parseFiles(asmLines);
        let prevLabel = "";

        const commentOnly = /^\s*(((#|@|;|\/\/).*)|(\/\*.*\*\/))$/;
        const sourceTag = /^\s*\.loc\s+(\d+)\s+(\d+).*/;
        const sourceStab = /^\s*\.stabn\s+(\d+),0,(\d+),.*/;
        const stdInLooking = /.*<stdin>|^-$|example\.[^/]+$|<source>/;
        const endBlock = /\.(cfi_endproc|data|text|section)/;
        let source = null;

        let inCustomAssembly = 0;
        asmLines.forEach(line => {
            let match;
            if (line.trim() === "") {
                result.push({text: "", source: null});
                return;
            }

            if (line.match(startAppBlock) || line.match(startAsmNesting)) {
                inCustomAssembly++;
            } else if (line.match(endAppBlock) || line.match(endAsmNesting)) {
                inCustomAssembly--;
            }

            if (!!(match = line.match(sourceTag))) {
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
            if (!!(match = line.match(sourceStab))) {
                // cf http://www.math.utah.edu/docs/info/stabs_11.html#SEC48
                switch (parseInt(match[1])) {
                    case 68:
                        source = {'file': null, 'line': parseInt(match[2])};
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
                line = fixLabelIndentation(line);

            match = line.match(labelDef);
            if (!match) match = line.match(assignmentDef);
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
            if (!match && filters.directives) {
                // Check for directives only if it wasn't a label; the regexp would
                // otherwise misinterpret labels as directives.
                if (line.match(dataDefn) && prevLabel) {
                    // We're defining data that's being used somewhere.
                } else {
                    if (line.match(directive)) return;
                }
            }

            line = utils.expandTabs(line);
            result.push({text: filterAsmLine(line, filters), source: hasOpcode(line) ? source : null});
        });
        return result;
    }

    function fixLabelIndentation(line) {
        const match = line.match(indentedLabelDef);
        if (match) {
            return line.replace(/^\s+/, "");
        } else {
            return line;
        }
    }

    let binaryHideFuncRe = null;
    let maxAsmLines = 500;

    function isUserFunction(func) {
        if (binaryHideFuncRe === null) return true;

        return !func.match(binaryHideFuncRe);
    }

    function processBinaryAsm(asm, filters) {
        const result = [];
        const asmLines = asm.split("\n");
        const asmOpcodeRe = /^\s*([0-9a-f]+):\s*(([0-9a-f][0-9a-f] ?)+)\s*(.*)/;
        const lineRe = /^(\/[^:]+):([0-9]+).*/;
        const labelRe = /^([0-9a-f]+)\s+<([^>]+)>:$/;
        const destRe = /.*\s([0-9a-f]+)\s+<([^>]+)>$/;
        let source = null;
        let func = null;

        // Handle "error" documents.
        if (asmLines.length === 1 && asmLines[0][0] === '<') {
            return [{text: asmLines[0], source: null}];
        }

        asmLines.forEach(line => {
            if (result.length >= maxAsmLines) {
                if (result.length === maxAsmLines) {
                    result.push({text: "[truncated; too many lines]", source: null});
                }
                return;
            }
            let match = line.match(lineRe);
            if (match) {
                source = {'file': null, 'line': parseInt(match[2])};
                return;
            }

            match = line.match(labelRe);
            if (match) {
                func = match[2];
                if (isUserFunction(func)) {
                    result.push({text: func + ":", source: null});
                }
                return;
            }

            if (!func || !isUserFunction(func)) return;

            match = line.match(asmOpcodeRe);
            if (match) {
                const address = parseInt(match[1], 16);
                const opcodes = match[2].split(" ").filter(x => !!x);
                const disassembly = " " + filterAsmLine(match[4], filters);
                let links = null;
                const destMatch = line.match(destRe);
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

    if (compilerProps) {
        const binaryHideFuncReValue = compilerProps('binaryHideFuncRe');
        if (binaryHideFuncReValue) binaryHideFuncRe = new RegExp(binaryHideFuncReValue);

        maxAsmLines = compilerProps('maxLinesOfAsm', maxAsmLines);
    }
    this.process = function (asm, filters) {
        return processAsm(asm, filters);
    };
}

module.exports = {
    AsmParser: AsmParser
};
