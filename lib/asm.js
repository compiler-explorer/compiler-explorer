// Copyright (c) 2012-2017, Matt Godbolt
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

var _ = require('underscore-node');
var utils = require('./utils');

function AsmParser(compilerProps) {
    var labelFind = /[.a-zA-Z_][a-zA-Z0-9$_.]*/g;
    var dataDefn = /\.(string|asciz|ascii|[1248]?byte|short|word|long|quad|value|zero)/;
    var fileFind = /^\s*\.file\s+(\d+)\s+"([^"]+)".*/;
    var hasOpcode = /^\s*([a-zA-Z$_][a-zA-Z0-9$_.]*:\s*)?[a-zA-Z].*/;
    var definesFunction = /^\s*\.type.*,\s*[@%]function$/;
    var labelDef = /^([.a-zA-Z_$][a-zA-Z0-9$_.]+):/;
    var directive = /^\s*\..*$/;

    function findUsedLabels(asmLines, filterDirectives) {
        var labelsUsed = {};
        var weakUsages = {};
        var currentLabel = "";

        // Scan through looking for definite label usages (ones used by opcodes),
        // and ones that are weakly used: that is, their use is conditional on another label.
        // For example:
        // .foo: .string "moo"
        // .baz: .quad .foo
        //       mov eax, .baz
        // In this case, the '.baz' is used by an opcode, and so is strongly used.
        // The '.foo' is weakly used by .baz.
        asmLines.forEach(function (line) {
            var match = line.match(labelDef);
            if (match)
                currentLabel = match[1];
            if (!line || line[0] === '.') return;

            match = line.match(labelFind);
            if (!match) return;

            if (!filterDirectives || line.match(hasOpcode) || line.match(definesFunction)) {
                // Only count a label as used if it's used by an opcode, or else we're not filtering directives.
                match.forEach(function (label) {
                    labelsUsed[label] = true;
                });
            } else if (currentLabel) {
                // If we have a current label, then any subsequent opcode or data definition's labels are refered to
                // weakly by that label.
                var isDataDefinition = !!line.match(dataDefn);
                var isOpcode = !line.match(directive); // We assume anything that's not a directive is an opcode.
                if (isDataDefinition || isOpcode) {
                    if (!weakUsages[currentLabel]) weakUsages[currentLabel] = [];
                    match.forEach(function (label) {
                        weakUsages[currentLabel].push(label);
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

        var MaxLabelIterations = 10;
        for (var iter = 0; iter < MaxLabelIterations; ++iter) {
            var toAdd = [];
            _.each(labelsUsed, function (t, label) { // jshint ignore:line
                _.each(weakUsages[label], function (nowused) {
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
        var files = {};
        asmLines.forEach(function (line) {
            var match = line.match(fileFind);
            if (match) {
                files[parseInt(match[1])] = match[2];
            }
        });
        return files;
    }

    function processAsm(asm, filters) {
        if (filters.binary) return processBinaryAsm(asm, filters);

        var result = [];
        var asmLines = utils.splitLines(asm);
        var labelsUsed = findUsedLabels(asmLines, filters.directives);
        var files = parseFiles(asmLines);
        var prevLabel = "";

        var commentOnly = /^\s*(((#|@|\/\/).*)|(\/\*.*\*\/))$/;
        var sourceTag = /^\s*\.loc\s+(\d+)\s+(\d+).*/;
        var sourceStab = /^\s*\.stabn\s+(\d+),0,(\d+),.*/;
        var stdInLooking = /.*<stdin>|-|example/;
        var endBlock = /\.(cfi_endproc|data|text|section)/;
        var source = null;
        asmLines.forEach(function (line) {
            var match;
            if (line.trim() === "") {
                result.push({text: "", source: null});
                return;
            }
            if (!!(match = line.match(sourceTag))) {
                source = null;
                var file = files[parseInt(match[1])];
                if (file && file.match(stdInLooking)) {
                    source = parseInt(match[2]);
                }
            }
            if (!!(match = line.match(sourceStab))) {
                // cf http://www.math.utah.edu/docs/info/stabs_11.html#SEC48
                switch (parseInt(match[1])) {
                    case 68:
                        source = parseInt(match[2]);
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

            match = line.match(labelDef);
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

            var hasOpcodeMatch = line.match(hasOpcode);
            line = utils.expandTabs(line);
            result.push({text: line, source: hasOpcodeMatch ? source : null});
        });
        return result;
    }

    var binaryHideFuncRe = null;
    var maxAsmLines = 500;

    function isUserFunction(func) {
        return !func.match(binaryHideFuncRe);
    }

    function processBinaryAsm(asm, filters) {
        var result = [];
        var asmLines = asm.split("\n");
        var asmOpcodeRe = /^\s*([0-9a-f]+):\s*(([0-9a-f][0-9a-f] ?)+)\s*(.*)/;
        var lineRe = /^(\/[^:]+):([0-9]+).*/;
        var labelRe = /^([0-9a-f]+)\s+<([^>]+)>:$/;
        var destRe = /.*\s([0-9a-f]+)\s+<([^>]+)>$/;
        var source = null;
        var func = null;

        // Handle "error" documents.
        if (asmLines.length === 1 && asmLines[0][0] === '<') {
            return [{text: asmLines[0], source: null}];
        }

        asmLines.forEach(function (line) {
            if (result.length >= maxAsmLines) {
                if (result.length == maxAsmLines) {
                    result.push({text: "[truncated; too many lines]", source: null});
                }
                return;
            }
            var match = line.match(lineRe);
            if (match) {
                source = parseInt(match[2]);
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
                var address = parseInt(match[1], 16);
                var opcodes = match[2].split(" ").filter(function (x) {
                    return x;
                });
                var disassembly = " " + match[4];
                var links = null;
                var destMatch = line.match(destRe);
                if (destMatch) {
                    links = [{
                        offset: disassembly.indexOf(destMatch[1]),
                        length: destMatch[1].length,
                        to: parseInt(destMatch[1], 16)
                    }];
                }
                result.push({opcodes: opcodes, address: address, text: disassembly, source: source, links: links});
            } else {
                //result.push({text: line, source: null});
            }
        });
        return result;
    }

    if (compilerProps) {
        binaryHideFuncRe = new RegExp(compilerProps('binaryHideFuncRe'));
        maxAsmLines = compilerProps('maxLinesOfAsm', maxAsmLines);
    }
    this.process = function (asm, filters) {
        return processAsm(asm, filters);
    };
}

module.exports = {
    AsmParser: AsmParser
};
