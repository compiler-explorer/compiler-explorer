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

import { AsmParser } from './asm-parser';
import { AsmRegex } from './asmregex';
import { logger } from './logger';
import * as utils from './utils';

export class AsmEWAVRParser extends AsmParser {
    constructor(compilerProps) {
        super(compilerProps);
        this.commentOnly = /^\s*(((#|@|\$|\/\/).*)|(\/\*.*\*\/))$/;
        this.filenameComment = /^\/\/\s[A-Za-z]?:?(\\\\?([^/\\]*[/\\])*)([^/\\]+)$/;
        this.lineNumberComment = /^\/\/\s*(\d+)\s(?!bytes).*/;

        // categories of directives. remove if filters.directives set
        this.segmentBegin = /^\s*(ASEG|ASEGN|COMMON|RSEG|STACK)\s*([A-Z_a-z][\w():]*)/;
        this.segmentControl = /^\s*(ALIGN|EVEN|ODD|ORG)/;
        this.definesGlobal = /^\s*(EXTERN|EXTRN|IMPORT|EXPORT|PUBWEAK|PUBLIC)\s+(.+)$/;
        this.definesLocal = /^\s*((ASSIGN|DEFINE|LOCAL|ALIAS|EQU|VAR)|(([A-Z_a-z]\w*)=.+))$/;
        this.miscDirective = /^\s*(NAME|MODULE|PROGRAM|LIBRARY|ERROR|END|CASEOFF|CASEON|CFI|COL|RADIX)/;

        // NOTE: Compiler generated labels can have spaces in them, but are quoted and in <>
        this.labelDef = /^`?\?*<?([A-Z_a-z][\w :]*)>?`?:$/;
        this.dataStatement = /^\s*(DB|DC16|DC24|DC32|DC8|DD|DP|DS|DS16|DS24|DS32|DW)/;
        this.requireStatement = /^\s*REQUIRE\s+`?\?*<?([A-Z_a-z][\w ]*)>?`?/;
        this.beginFunctionMaybe = /^\s*RSEG\s*CODE:CODE:(.+)$/;
    }

    hasOpcode(line) {
        // check for empty lines
        if (line.length === 0) return false;
        // check for a local label definition
        if (this.definesLocal.test(line)) return false;
        // check for global label definitions
        if (this.definesGlobal.test(line)) return false;
        // check for data definitions
        if (this.dataStatement.test(line)) return false;
        // check for segment begin and end
        if (this.segmentBegin.test(line) || this.segmentControl.test(line)) return false;
        // check for label definition
        if (this.labelDef.test(line)) return false;
        // check for miscellaneous directives
        if (this.miscDirective.test(line)) return false;
        // check for requre statement
        if (this.requireStatement.test(line)) return false;

        return !!this.hasOpcodeRe.test(line);
    }

    labelFindFor() {
        return this.labelDef;
    }

    processAsm(asm, filters) {
        // NOTE: EWAVR assembly seems to be closest to visual studio
        const getFilenameFromComment = line => {
            const matches = line.match(this.filenameComment);
            if (!matches) {
                return null;
            } else {
                return matches[3];
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

        let asmLines = utils.splitLines(asm);

        const stdInLooking = /<stdin>|^-$|example\.[^/]+$|<source>/;

        // type source = {file: string option; line: int}
        // type line = {line: string; source: source option}
        // type label =
        //   { lines: line array
        //   ; name: string | undefined
        //   ; initialLine: int
        //   ; file: string option
        //   ; require: string array }
        let resultObject = {
            prefix: [],    // line array
            labels: [],    // label array
            postfix: [],    // line array
        };

        let currentLabel = null; // func option
        let currentFile;
        let currentLine;

        let seenEnd = false;

        const definedLabels = {};

        const createSourceFor = (line, currentFile, currentLine) => {
            const hasopc = this.hasOpcode(line);
            const createsData = line.match(this.dataStatement);
            if ((hasopc || createsData) && (currentFile || currentLine)) {
                return {
                    file: (currentFile ? currentFile : null),
                    line: (currentLine ? currentLine : null),
                };
            }

            return null;
        };

        const checkBeginLabel = (line) => {
            const matches = line.match(this.labelDef);
            if (matches) {
                currentLabel = {
                    lines: [],
                    initialLine: currentLine,
                    name: matches[1],
                    file: currentFile,
                };
                definedLabels[matches[1]] =  currentLine;
                resultObject.labels.push(currentLabel);
            }
        };

        const checkRequiresStatement = (line) => {
            const matches = line.match(this.requireStatement);
            if (matches && currentLabel != null) {
                if (currentLabel.require != null) {
                    currentLabel.require.push(matches[1]);
                } else {
                    currentLabel.require = [ matches[1] ];
                }
            }
        };

        asmLines.forEach(line => {
            if (line.trim() === 'END') {
                seenEnd = true;
                if (!filters.directives) {
                    resultObject.postfix.push({text: line, source: null});
                }
                return;
            }

            if (line.trim() === '') {
                const emptyLine = {text: '', source: null};
                if (seenEnd) {
                    resultObject.postfix.push(emptyLine);
                }
                else if (currentLabel === null) {
                    resultObject.prefix.push(emptyLine);
                }
                else {
                    currentLabel.lines.push(emptyLine);
                }
                return;
            }
            if (seenEnd && !this.commentOnly.test(line)) {
                // There should be nothing but comments after END directive
                throw new Error('EWAVR: non-comment line after the end statement');
            }

            let tmp = getFilenameFromComment(line);
            if (tmp !== null) {
                // if the file is the "main file", give it the file `null`
                if (stdInLooking.test(tmp)) {
                    currentFile = null;
                } else {
                    currentFile = tmp;
                }
                if (currentLabel != null && currentLabel.file === undefined) {
                    currentLabel.file = currentFile;
                }
            } else {
                tmp = getLineNumberFromComment(line);
                if (tmp !== null) {
                    if (currentFile === undefined) {
                        logger.error('Somehow, we have a line number comment without a file comment: %s',
                            line);
                    }
                    if (currentLabel != null && currentLabel.initialLine === undefined) {
                        currentLabel.initialLine = tmp;
                    }
                    currentLine = tmp;
                }
            }

            checkBeginLabel(line);
            checkRequiresStatement(line);

            if (filters.commentOnly && this.commentOnly.test(line)) {
                return;
            }

            const shouldSkip = filters.directives && (
                line.match(this.segmentBegin) ||
                line.match(this.segmentControl) ||
                line.match(this.definesGlobal) ||
                line.match(this.definesLocal) ||
                line.match(this.miscDirective) ||
                line.match(this.requireStatement)
            );

            if (shouldSkip) {
                return;
            }

            line = utils.expandTabs(line);
            const textAndSource = {
                text: AsmRegex.filterAsmLine(line, filters),
                source: createSourceFor(line, currentFile, currentLine),
            };
            if (currentLabel === null) {
                resultObject.prefix.push(textAndSource);
            } else if (!shouldSkip) {
                currentLabel.lines.push(textAndSource);
            }
        });

        return this.resultObjectIntoArray(resultObject, filters, definedLabels);
    }

    resultObjectIntoArray(obj, filters, ddefLabels) {
        // NOTES on EWAVR function and labels:
        // - template functions are not mangled with type info.
        //   Instead they simply have a `_#` appended to the end, with #
        //   incrementing for each instantiation.
        // - labels for variables, functions, and code fragments are all the same.
        // - One exception.. functions SEEM to always have a segment command
        //   with a few lines before the label. is this reliable?
        
        // NOTES: compiler generated labels
        //    'Initializer for' is used to init variables. usually at end of file
        //    'Segment init:' is used to init sections. One per many 'Initializer for' labels.
        const compilerGeneratedLabel = /^(initializer for |segment init: )([\w :]*)$/i;
        const segInitLabel = /^segment init: ([\w :]*)$/i;

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

        for (const line of obj.prefix) {
            pushLine(line);
        }

        for (const label of obj.labels) {
            if (!filters.libraryCode || label.file === null) {
                const match = label.name.match(compilerGeneratedLabel);
                const segInitMatch = label.name.match(segInitLabel);
                pushLine({text: '', source: null});
                for (const line of label.lines) {
                    // Match variable inits to the source line of declaration.
                    // No source line for global section initilization
                    if(match && line.source != null) {
                        line.source.line = ddefLabels[match[2]];
                    }
                    // Filter section inits as directives
                    if(segInitMatch && filters.directives) {
                        continue;
                    }
                    pushLine(line);
                }
            }
        }

        for (const line of obj.postfix) {
            pushLine(line);
        }

        return {
            asm: result,
        };
    }
}
