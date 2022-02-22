// Copyright (c) 2018, 2021, Microsoft Corporation & Compiler Explorer Authors
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

import { logger } from '../logger';
import * as utils from '../utils';

import { AsmParser } from './asm-parser';
import { AsmRegex } from './asmregex';

export class Vc6AsmParser extends AsmParser {
    constructor(compilerProps) {
        super(compilerProps);
        this.asmBinaryParser = new AsmParser(compilerProps);
        this.miscDirective = /^\s*(include|INCLUDELIB|TITLE|\.|else$|endif$|if @Version|FLAT|ASSUME|THUMB|ARM64|TTL|END$)/;
        this.localLabelDef = /^([$A-Z_a-z]+) =/;
        this.commentOnly = /^;/;
        this.filenameComment = /^; File (.+)/;
        this.lineNumberComment = /^; Line (\d+)/;
        this.beginSegment = /^(CONST|_BSS|_DATA|_TLS|\.?[prx]?data(\$[A-Za-z]+)?|CRT(\$[A-Za-z]+)?|_TEXT|\.?text(\$[A-Za-z]+)?)\s+SEGMENT|\s*AREA/;
        this.endSegment = /^(CONST|_BSS|_DATA|_TLS|[prx]?data(\$[A-Za-z]+)?|CRT(\$[A-Za-z]+)?|_TEXT|text(\$[A-Za-z]+)?)\s+ENDS/;
        this.beginFunction = /^; Function compile flags: /;
        this.endProc = /^([$?@A-Z_a-z][\w$<>?@]*)?\s+ENDP/;
        // on x86, we use the end of the segment to end a function
        // on arm, we use ENDP
        this.endFunction = /^(_TEXT\s+ENDS|\s+ENDP)/;

        this.labelDef = /^\|?([$?@A-Z_a-z][\w$<>?@]*)\|?\s+(PROC|=|D[BDQW])/;
        this.definesGlobal = /^\s*(PUBLIC|EXTRN|EXPORT)\s+/;
        this.definesFunction = /^\|?([$?@A-Z_a-z][\w$<>?@]*)\|?\s+PROC/;
        this.labelFind = /[$?@A-Z_a-z][\w$<>?@]*/g;
        this.dataDefn = /^(\|?[$?@A-Z_a-z][\w$<>?@]*\|?)\sDC?[BDQW]\s|\s+DC?[BDQW]\s|\s+ORG/;

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
        if (this.localLabelDef.test(line)) return false;
        // check for global label definitions
        if (this.definesGlobal.test(line)) return false;
        // check for data definitions
        if (this.dataDefn.test(line)) return false;
        // check for segment begin and end
        if (this.beginSegment.test(line) || this.endSegment.test(line)) return false;
        // check for function begin and end
        // note: functionBegin is used for the function compile flags comment
        if (this.definesFunction.test(line) || this.endProc.test(line)) return false;
        // check for miscellaneous directives
        if (this.miscDirective.test(line)) return false;

        return !!this.hasOpcodeRe.test(line);
    }

    labelFindFor() {
        return this.labelFind;
    }

    processAsm(asm, filters) {
        if (filters.binary) {
            return this.asmBinaryParser.processAsm(asm, filters);
        }

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

        const asmLines = utils.splitLines(asm);
        // note: VC doesn't output unused labels, afaict

        const stdInLooking = /<stdin>|^-$|example\.[^/]+$|<source>/;

        // type source = {file: string option; line: int}
        // type line = {line: string; source: source option}
        // type func =
        //   { lines: line array
        //   ; name: string | undefined
        //   ; initialLine: int
        //   ; file: string option | undefined }
        let resultObject = {
            prefix: [],    // line array
            functions: [], // func array
            postfix: null,    // line?
        };

        let currentFunction = null; // func option
        let currentFile;
        let currentLine;

        let seenEnd = false;

        const datadefLabels = [];
        const datadefLabelsUsed = [];

        const createSourceFor = (hasopc, currentFile, currentLine) => {
            if (hasopc && (currentFile || currentLine)) {
                return {
                    file: (currentFile ? currentFile : null),
                    line: (currentLine ? currentLine : null),
                };
            }

            return null;
        };

        const checkUsedDatadefLabels = (line) => {
            const labels = line.match(this.labelFind);
            if (!labels) return;
            labels.splice(0, 1);
            for (const item of labels) {
                if (datadefLabels.find(l => item === l)) {
                    datadefLabelsUsed.push(item);
                }
            }
        };

        const checkBeginFunction = (line) => {
            if (this.definesFunction.test(line)) {
                currentFunction = {
                    lines: [],
                    initialLine: undefined,
                    name: undefined,
                    file: currentFile,
                };
                resultObject.functions.push(currentFunction);
            }
        };

        const checkForDdefLabel = (line) => {
            const ddef = line.match(this.dataDefn);
            if (ddef && ddef[1]) {
                datadefLabels.push(ddef[1]);
            }
        };

        for (let line of asmLines) {
            if (line.trim() === 'END') {
                seenEnd = true;
                if (!filters.directives) {
                    resultObject.postfix = {text: line, source: null};
                }
                continue;
            }
            if (line.trim() === '') {
                if (seenEnd) continue;

                const emptyLine = {text: '', source: null};
                if (currentFunction === null) {
                    resultObject.prefix.push(emptyLine);
                } else {
                    currentFunction.lines.push(emptyLine);
                }
                continue;
            }
            if (seenEnd) {
                // this should never happen
                throw new Error('Visual C++: text after the end statement');
            }

            let tmp = getFilenameFromComment(line);
            if (tmp !== null) {
                if (currentFunction === null) {
                    logger.error('We have a file comment outside of a function: %s',
                        line);
                }
                // if the file is the "main file", give it the file `null`
                if (stdInLooking.test(tmp)) {
                    currentFile = null;
                } else {
                    currentFile = tmp;
                }
                if (currentFunction.file === undefined) {
                    currentFunction.file = currentFile;
                }
            } else {
                tmp = getLineNumberFromComment(line);
                if (tmp !== null) {
                    if (currentFile === undefined) {
                        logger.error('Somehow, we have a line number comment without a file comment: %s',
                            line);
                    }
                    if (currentFunction.initialLine === undefined) {
                        currentFunction.initialLine = tmp;
                    }
                    currentLine = tmp;
                }
            }

            checkBeginFunction(line);

            const functionName = line.match(this.definesFunction);
            if (functionName) {
                if (asmLines.length === 0) {
                    continue;
                }
                currentFunction.name = functionName[1];
            }

            if (filters.commentOnly && this.commentOnly.test(line)) continue;

            const shouldSkip = filters.directives && (
                line.match(this.endSegment) ||
                line.match(this.definesGlobal) ||
                line.match(this.miscDirective) ||
                line.match(this.beginSegment));

            if (shouldSkip) {
                continue;
            }

            checkForDdefLabel(line);

            line = utils.expandTabs(line);
            const hasopc = this.hasOpcode(line);
            const textAndSource = {
                text: AsmRegex.filterAsmLine(line, filters),
                source: createSourceFor(hasopc, currentFile, currentLine),
            };
            if (currentFunction === null) {
                resultObject.prefix.push(textAndSource);
            } else if (!shouldSkip) {
                currentFunction.lines.push(textAndSource);
            }

            checkUsedDatadefLabels(line);
        }

        return this.resultObjectIntoArray(resultObject, filters, datadefLabelsUsed);
    }

    resultObjectIntoArray(obj, filters, ddefLabelsUsed) {
        const collator = new Intl.Collator();

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
                    return collator.compare(f1.name, f2.name);
                } else {
                    return f1.initialLine - f2.initialLine;
                }
            }

            // else, order by file
            return collator.compare(f1.file, f2.file);
        });

        let result = [];
        let lastLineWasWhitespace = true;
        let pushLine = line => {
            if (line.text.trim() === '') {
                if (!lastLineWasWhitespace) {
                    result.push({text: '', source: null});
                    lastLineWasWhitespace = true;
                }
            } else {
                result.push(line);
                lastLineWasWhitespace = false;
            }
        };

        if (filters.labels) {
            let currentDdef = false;
            let isUsed = false;
            for (const line of obj.prefix) {
                const matches = line.text.match(this.dataDefn);
                if (matches) {
                    if (matches[1]) {
                        currentDdef = matches[1];
                        isUsed = ddefLabelsUsed.find(label => currentDdef === label);
                    }

                    if (isUsed) {
                        pushLine(line);
                    }
                } else {
                    currentDdef = false;
                    pushLine(line);
                }
            }
        } else {
            for (const line of obj.prefix) {
                pushLine(line);
            }
        }

        for (const func of obj.functions) {
            if (!filters.libraryCode || func.file === null) {
                pushLine({text: '', source: null});
                for (const line of func.lines) {
                    pushLine(line);
                }
            }
        }

        if (obj.postfix !== null) {
            pushLine(obj.postfix);
        }

        return {
            asm: result,
        };
    }
}
