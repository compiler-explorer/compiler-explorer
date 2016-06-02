// Copyright (c) 2012-2016, Matt Godbolt
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

(function () {
    var tabsRe = /\t/g;

    function expandTabs(line) {
        var extraChars = 0;
        return line.replace(tabsRe, function (match, offset, string) {
            var total = offset + extraChars;
            var spacesNeeded = (total + 8) & 7;
            extraChars += spacesNeeded - 1;
            return "        ".substr(spacesNeeded);
        });
    }

    function processAsm(asm, filters) {
        if (filters.binary) return processBinaryAsm(asm, filters);

        var result = [];
        var asmLines = asm.split(/\r?\n/);
        var labelsUsed = {};
        var labelFind = /[.a-zA-Z0-9_][a-zA-Z0-9$_.]*/g;
        var files = {};
        var prevLabel = "";
        var dataDefn = /\.(string|asciz|ascii|[1248]?byte|short|word|long|quad|value|zero)/;
        var fileFind = /^\s*\.file\s+(\d+)\s+"([^"]+)".*/;
        var hasOpcode = /^\s*([a-zA-Z0-9$_][a-zA-Z0-9$_.]*:\s*)?[a-zA-Z].*/;
        asmLines.forEach(function (line) {
            if (line === "" || line[0] === ".") return;
            var match = line.match(labelFind);
            if (match && (!filters.directives || line.match(hasOpcode))) {
                // Only count a label as used if it's used by an opcode, or else we're not filtering directives.
                match.forEach(function (label) {
                    labelsUsed[label] = true;
                });
            }
            match = line.match(fileFind);
            if (match) {
                files[parseInt(match[1])] = match[2];
            }
        });

        var directive = /^\s*(\.|([_A-Z]+\b))/;
        var labelDefinition = /^([a-zA-Z0-9$_.]+):/;
        var commentOnly = /^\s*([#@;]|\/\/).*/;
        var sourceTag = /^\s*\.loc\s+(\d+)\s+(\d+).*/;
        var stdInLooking = /.*<stdin>|-/;
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
            if (line.match(endBlock)) {
                source = null;
                prevLabel = null;
            }

            if (filters.commentOnly && line.match(commentOnly)) return;

            match = line.match(labelDefinition);
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
                    console.log("moo", line);
                    if (line.match(directive)) return;
                }
            }

            var hasOpcodeMatch = line.match(hasOpcode);
            line = expandTabs(line);
            result.push({text: line, source: hasOpcodeMatch ? source : null});
        });
        return result;
    }

    var binaryHideFuncRe = null;
    var maxAsmLines = 500;

    function initialise(compilerProps) {
        var pattern = compilerProps('binaryHideFuncRe');
        console.log("asm: binary re = " + pattern);
        binaryHideFuncRe = new RegExp(pattern);
        maxAsmLines = compilerProps('maxLinesOfAsm', maxAsmLines);
    }

    function isUserFunction(func) {
        return !func.match(binaryHideFuncRe);
    }

    function processBinaryAsm(asm, filters) {
        var result = [];
        var asmLines = asm.split(/\r?\n/);
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
                }).map(function (x) {
                    return parseInt(x, 16);
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

    exports.processAsm = processAsm;
    exports.initialise = initialise;

}).call(this);
