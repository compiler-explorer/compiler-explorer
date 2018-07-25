// Copyright (c) 2018, Microsoft Corporation
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
    logger = require('./logger').logger,
    utils = require('./utils');

class AsmParser extends AsmParserBase {
    constructor(compilerProps) {
        super(compilerProps);
        this.miscDirective = /^\s*(include|INCLUDELIB|TITLE|\.|THUMB|ARM64|TTL|END$)/;
        this.localLabelDef = /^([a-zA-Z$_]+) =/;
        this.commentOnly = /^;/;
        this.filenameComment = /^; File (.+)/;
        this.lineNumberComment = /^; Line ([0-9]+)/;
        this.beginSegment = /^(CONST|_BSS|\.?[prx]?data(\$[a-zA-Z]+)?|CRT(\$[a-zA-Z]+)?|_TEXT|\.?text(\$[a-zA-Z]+)?)\s+SEGMENT|\s*AREA/;
        this.endSegment = /^(CONST|_BSS|[prx]?data(\$[a-zA-Z]+)?|CRT(\$[a-zA-Z]+)?|_TEXT|text(\$[a-zA-Z]+)?)\s+ENDS/;
        this.beginFunction = /^; Function compile flags: /;
        this.endProc = /^([a-zA-Z@$?_][a-zA-Z0-9@$?_<>]*)?\s+ENDP/;
        // on x86, we use the end of the segment to end a function
        // on arm, we use ENDP
        this.endFunction = /^(_TEXT\s+ENDS|\s+ENDP)/;

        this.labelDef = /^\|?([a-zA-Z@$?_][a-zA-Z0-9@$?_<>]*)\|?\s+(PROC|=|D[BWDQ])/;
        this.definesGlobal = /^\s*(PUBLIC|EXTRN|EXPORT)\s+/;
        this.definesFunction = /^\|?([a-zA-Z@$?_][a-zA-Z0-9@$?_<>]*)\|?\s+PROC/;
        this.labelFind = /[a-zA-Z@$?_][a-zA-Z0-9@$?_<>]*/g;
        this.dataDefn = /^(\|?[a-zA-Z@$?_][a-zA-Z0-9@$?_<>]*\|? DC?[BWDQ]\s|\s+DC?[BWDQ]\s|\s+ORG)/;

        // these are set to an impossible regex, because VC doesn't have inline assembly
        this.startAppBlock = this.startAsmNesting = /a^/;
        this.endAppBLock = this.endAsmNesting = /a^/;
        // same, but for CUDA
        this.cudaBeginDef = /a^/;
    }

    hasOpcode(line) {
        // note: cl doesn't output leading labels
        // strip comments
        line = line.split(';', 1)[0];
        // check for empty lines
        if (line.length === 0) return false;
        // check for a local label definition
        if (line.match(this.localLabelDef)) return false;
        // check for global label definitions
        if (line.match(this.definesGlobal)) return false;
        // check for data definitions
        if (line.match(this.dataDefn)) return false;
        // check for segment begin and end
        if (line.match(this.beginSegment) || line.match(this.endSegment)) return false;
        // check for function begin and end
        // note: functionBegin is used for the function compile flags comment
        if (line.match(this.definesFunction) || line.match(this.endProc)) return false;
        // check for miscellaneous directives
        if (line.match(this.miscDirective)) return false;

        return !!line.match(this.hasOpcodeRe);
    }

    labelFindFor() {
        return this.labelFind;
    }

    processAsm(asm, filters) {
        const getFilenameFromComment = line => {
            const matches = line.match(this.filenameComment);
            if (!matches) {
                return null;
            } else {
                return matches[1];
            }
        };
        const getLineNumberFromComment = line => {
            const matches = line.match(this.lineNumberComment);
            if (!matches) {
                return null;
            } else {
                return parseInt(matches[1]);
            }
        };

        if (filters.binary) {
            logger.error("Binary mode should be disabled with the Visual C++ assembler parser");
        }

        const asmLines = utils.splitLines(asm);
        // note: VC doesn't output unused labels, afaict

        const stdInLooking = /<stdin>|^-$|example\.[^/]+$|<source>/;

        // type source = {file: string option; line: int}
        let source = null; // source
        // type line = {line: string; source: source option}
        // type func =
        //   { lines: line array
        //   ; name: string | undefined
        //   ; initialLine: int
        //   ; file: string option | undefined }
        let resultObject = {
            prefix: [],    // line array
            functions: [], // func array
            postfix: null    // line?
        };

        let currentFunction = null; // func option

        let seenEnd = false;

        asmLines.forEach(line => {
            if (line.trim() === "END") {
                seenEnd = true;
                if (!filters.directives) {
                    resultObject.postfix = line;
                }
                return;
            }
            if (line.trim() === "") {
                if (seenEnd) return;

                const emptyLine = {text: "", source: null};
                if (currentFunction === null) {
                    resultObject.prefix.push(emptyLine);
                } else {
                    currentFunction.lines.push(emptyLine);
                }
                return;
            }
            if (seenEnd) {
                // this should never happen
                throw Error("Visual C++: text after the end statement");
            }

            let tmp = null;
            tmp = getFilenameFromComment(line);
            if (tmp !== null) {
                if (currentFunction === null) {
                    logger.error("We have a file comment outside of a function: ",
                        line);
                }
                // if the file is the "main file", give it the file `null`
                if (tmp.match(stdInLooking)) {
                    currentFunction.file = null;
                } else {
                    currentFunction.file = tmp;
                }
                source = {file: currentFunction.file, line: 0};
            } else {
                tmp = getLineNumberFromComment(line);
                if (tmp !== null) {
                    if (source === null) {
                        logger.error("Somehow, we have a line number comment without a file comment: ",
                            line);
                    } else {
                        if (tmp < currentFunction.initialLine) {
                            currentFunction.initialLine = tmp;
                        }
                        source = {file: source.file, line: tmp};
                    }
                }
            }

            if (line.match(this.beginFunction)) {
                currentFunction = {
                    lines: [],
                    initialLine: Infinity,
                    name: undefined,
                    file: undefined
                };
                resultObject.functions.push(currentFunction);
            }

            const functionName = line.match(this.definesFunction);
            if (functionName) {
                if (asmLines.length === 0) { return; }
                currentFunction.name = functionName[1];
            }

            if (filters.commentOnly && line.match(this.commentOnly)) return;

            const shouldSkip = filters.directives && (
                line.match(this.endSegment) ||
                line.match(this.dataDefn) ||
                line.match(this.definesGlobal) ||
                line.match(this.miscDirective) ||
                line.match(this.beginSegment));

            if (shouldSkip) {
                return;
            }

            line = utils.expandTabs(line);
            const textAndSource = {
                text: this.filterAsmLine(line, filters),
                source: this.hasOpcode(line) ? source : null
            };
            if (currentFunction === null) {
                resultObject.prefix.push(textAndSource);
            } else if (!shouldSkip) {
                currentFunction.lines.push(textAndSource);
            }
        });

        return this.resultObjectIntoArray(resultObject);
    }

    resultObjectIntoArray(obj) {
        obj.functions.sort((f1, f2) => {
            // order the main file above all others
            if (f1.file === null && f2.file !== null) {
                return -1;
            }
            if (f1.file !== null && f2.file === null) {
                return 1;
            }
            // order no-file below all others
            if (f1.file === undefined && f2.file !== undefined) {
                return 1;
            }
            if (f1.file !== undefined && f2.file === undefined) {
                return -1;
            }

            // if the files are the same, use line number ordering
            if (f1.file === f2.file) {
                // if the lines are the same as well, it's either:
                //   - two template instantiations, or
                //   - two compiler generated functions
                // order by name
                if (f1.initialLine === f2.initialLine) {
                    return f1.name.localeCompare(f2.name);
                } else {
                    return f1.initialLine - f2.initialLine;
                }
            }

            // else, order by file
            return f1.file.localeCompare(f2.file);
        });

        let result = [];
        let lastLineWasWhitespace = true;
        let pushLine = line => {
            if (line.text.trim() === '') {
                if (!lastLineWasWhitespace) {
                    result.push({text: "", source: null});
                    lastLineWasWhitespace = true;
                }
            } else {
                result.push(line);
                lastLineWasWhitespace = false;
            }
        };

        for (const line of obj.prefix) {
            pushLine(line);
        }

        for (const func of obj.functions) {
            pushLine({text: "", source: null});
            for (const line of func.lines) {
                pushLine(line);
            }
        }

        if (obj.postfix !== null) {
            pushLine(obj.postfix);
        }

        return result;
    }
}

module.exports = AsmParser;
