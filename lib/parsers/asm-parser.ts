// Copyright (c) 2015, Compiler Explorer Authors
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

import _ from 'underscore';

import {
    AsmResultLabel,
    AsmResultSource,
    ParsedAsmResult,
    ParsedAsmResultLine,
} from '../../types/asmresult/asmresult.interfaces';
import {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces';
import {assert} from '../assert';
import {isString} from '../common-utils';
import {PropertyGetter} from '../properties.interfaces';
import * as utils from '../utils';

import {IAsmParser} from './asm-parser.interfaces';
import {AsmRegex} from './asmregex';

export class AsmParser extends AsmRegex implements IAsmParser {
    labelFindNonMips: RegExp;
    labelFindMips: RegExp;
    mipsLabelDefinition: RegExp;
    dataDefn: RegExp;
    fileFind: RegExp;
    hasOpcodeRe: RegExp;
    instructionRe: RegExp;
    identifierFindRe: RegExp;
    hasNvccOpcodeRe: RegExp;
    definesFunction: RegExp;
    definesGlobal: RegExp;
    definesWeak: RegExp;
    indentedLabelDef: RegExp;
    assignmentDef: RegExp;
    directive: RegExp;
    startAppBlock: RegExp;
    endAppBlock: RegExp;
    startAsmNesting: RegExp;
    endAsmNesting: RegExp;
    cudaBeginDef: RegExp;
    cudaEndDef: RegExp;
    binaryHideFuncRe: RegExp | null;
    maxAsmLines: number;
    asmOpcodeRe: RegExp;
    relocationRe: RegExp;
    relocDataSymNameRe: RegExp;
    lineRe: RegExp;
    labelRe: RegExp;
    destRe: RegExp;
    commentRe: RegExp;
    instOpcodeRe: RegExp;
    commentOnly: RegExp;
    commentOnlyNvcc: RegExp;
    sourceTag: RegExp;
    sourceD2Tag: RegExp;
    sourceCVTag: RegExp;
    source6502Dbg: RegExp;
    source6502DbgEnd: RegExp;
    sourceStab: RegExp;
    stdInLooking: RegExp;
    endBlock: RegExp;
    blockComments: RegExp;

    constructor(compilerProps?: PropertyGetter) {
        super();

        this.labelFindNonMips = /[.A-Z_a-z][\w$.]*/g;
        // MIPS labels can start with a $ sign, but other assemblers use $ to mean literal.
        this.labelFindMips = /[$.A-Z_a-z][\w$.]*/g;
        this.mipsLabelDefinition = /^\$[\w$.]+:/;
        this.dataDefn = /^\s*\.(string|asciz|ascii|[1248]?byte|short|half|[dhx]?word|long|quad|octa|value|zero)/;
        this.fileFind = /^\s*\.(?:cv_)?file\s+(\d+)\s+"([^"]+)"(\s+"([^"]+)")?.*/;
        // Opcode expression here matches LLVM-style opcodes of the form `%blah = opcode`
        this.hasOpcodeRe = /^\s*(%[$.A-Z_a-z][\w$.]*\s*=\s*)?[A-Za-z]/;
        this.instructionRe = /^\s*[A-Za-z]+/;
        this.identifierFindRe = /[$.@A-Z_a-z]\w*/g;
        this.hasNvccOpcodeRe = /^\s*[@A-Za-z|]/;
        this.definesFunction = /^\s*\.(type.*,\s*[#%@]function|proc\s+[.A-Z_a-z][\w$.]*:.*)$/;
        this.definesGlobal = /^\s*\.(?:globa?l|GLB|export)\s*([.A-Z_a-z][\w$.]*)/;
        this.definesWeak = /^\s*\.(?:weakext|weak)\s*([.A-Z_a-z][\w$.]*)/;
        this.indentedLabelDef = /^\s*([$.A-Z_a-z][\w$.]*):/;
        this.assignmentDef = /^\s*([$.A-Z_a-z][\w$.]*)\s*=/;
        this.directive = /^\s*\..*$/;
        this.startAppBlock = /\s*#APP.*/;
        this.endAppBlock = /\s*#NO_APP.*/;
        this.startAsmNesting = /\s*# Begin ASM.*/;
        this.endAsmNesting = /\s*# End ASM.*/;
        this.cudaBeginDef = /\.(entry|func)\s+(?:\([^)]*\)\s*)?([$.A-Z_a-z][\w$.]*)\($/;
        this.cudaEndDef = /^\s*\)\s*$/;

        this.binaryHideFuncRe = null;
        this.maxAsmLines = 5000;
        if (compilerProps) {
            const binaryHideFuncReValue = compilerProps('binaryHideFuncRe');
            if (binaryHideFuncReValue) {
                assert(isString(binaryHideFuncReValue));
                this.binaryHideFuncRe = new RegExp(binaryHideFuncReValue);
            }

            this.maxAsmLines = compilerProps('maxLinesOfAsm', this.maxAsmLines);
        }

        this.asmOpcodeRe = /^\s*(?<address>[\da-f]+):\s*(?<opcodes>([\da-f]{2} ?)+)\s*(?<disasm>.*)/;
        this.relocationRe = /^\s*(?<address>[\da-f]+):\s*(?<relocname>(R_[\dA-Z_]+))\s*(?<relocdata>.*)/;
        this.relocDataSymNameRe = /^(?<symname>[^\d-+][\w.]*)?\s*(?<addend_or_value>.*)$/;
        this.lineRe = /^(\/[^:]+):(?<line>\d+).*/;

        // labelRe is made very greedy as it's also used with demangled objdump
        // output (eg. it can have c++ template with <>).
        this.labelRe = /^([\da-f]+)\s+<(.+)>:$/;
        this.destRe = /\s([\da-f]+)\s+<([^+>]+)(\+0x[\da-f]+)?>$/;
        this.commentRe = /[#;]/;
        this.instOpcodeRe = /(\.inst\.?\w?)\s*(.*)/;

        // Lines matching the following pattern are considered comments:
        // - starts with '#', '@', '//' or a single ';' (non repeated)
        // - starts with ';;' and the first non-whitespace before end of line is not #
        this.commentOnly = /^\s*(((#|@|\/\/).*)|(\/\*.*\*\/)|(;\s*)|(;[^;].*)|(;;\s*[^\s#].*))$/;
        this.commentOnlyNvcc = /^\s*(((#|;|\/\/).*)|(\/\*.*\*\/))$/;
        this.sourceTag = /^\s*\.loc\s+(\d+)\s+(\d+)\s+(.*)/;
        this.sourceD2Tag = /^\s*\.d2line\s+(\d+),?\s*(\d*).*/;
        this.sourceCVTag = /^\s*\.cv_loc\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+).*/;
        this.source6502Dbg = /^\s*\.dbg\s+line,\s*"([^"]+)",\s*(\d+)/;
        this.source6502DbgEnd = /^\s*\.dbg\s+line[^,]/;
        this.sourceStab = /^\s*\.stabn\s+(\d+),0,(\d+),.*/;
        this.stdInLooking = /<stdin>|^-$|example\.[^/]+$|<source>/;
        this.endBlock = /\.(cfi_endproc|data|text|section)/;
        this.blockComments = /^[\t ]*\/\*(\*(?!\/)|[^*])*\*\/\s*/gm;
    }

    hasOpcode(line, inNvccCode) {
        // Remove any leading label definition...
        const match = line.match(this.labelDef);
        if (match) {
            line = line.substr(match[0].length);
        }
        // Strip any comments
        line = line.split(this.commentRe, 1)[0];
        // .inst generates an opcode, so also counts
        if (this.instOpcodeRe.test(line)) return true;
        // Detect assignment, that's not an opcode...
        if (this.assignmentDef.test(line)) return false;
        if (inNvccCode) {
            return !!this.hasNvccOpcodeRe.test(line);
        }
        return !!this.hasOpcodeRe.test(line);
    }

    labelFindFor(asmLines) {
        const isMips = _.any(asmLines, line => !!this.mipsLabelDefinition.test(line));
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
        let currentLabelSet: string[] = [];
        let inLabelGroup = false;
        let inCustomAssembly = 0;
        const startBlock = /\.cfi_startproc/;
        const endBlock = /\.cfi_endproc/;
        let inFunction = false;
        let inNvccCode = false;

        // Scan through looking for definite label usages (ones used by opcodes),
        // and ones that are weakly used: that is, their use is conditional on another label.
        // For example:
        // .foo: .string "moo"
        // .baz: .quad .foo
        //       mov eax, .baz
        // In this case, the '.baz' is used by an opcode, and so is strongly used.
        // The '.foo' is weakly used by .baz.
        // Also, if we have random data definitions within a block of a function (between
        // cfi_startproc and cfi_endproc), we assume they are strong usages. This covers things
        // like jump tables embedded in ARM code.
        // See https://github.com/compiler-explorer/compiler-explorer/issues/2788
        for (let line of asmLines) {
            if (this.startAppBlock.test(line) || this.startAsmNesting.test(line)) {
                inCustomAssembly++;
            } else if (this.endAppBlock.test(line) || this.endAsmNesting.test(line)) {
                inCustomAssembly--;
            } else if (startBlock.test(line)) {
                inFunction = true;
            } else if (endBlock.test(line)) {
                inFunction = false;
            } else if (this.cudaBeginDef.test(line)) {
                inNvccCode = true;
            }

            if (inCustomAssembly > 0) line = this.fixLabelIndentation(line);

            let match = line.match(this.labelDef);
            if (match) {
                if (inLabelGroup) currentLabelSet.push(match[1]);
                else currentLabelSet = [match[1]];
                inLabelGroup = true;
            } else {
                inLabelGroup = false;
            }
            match = line.match(this.definesGlobal);
            if (!match) match = line.match(this.definesWeak);
            if (!match) match = line.match(this.cudaBeginDef);
            if (match) {
                labelsUsed[match[1]] = true;
            }

            const definesFunction = line.match(this.definesFunction);
            if (!definesFunction && (!line || line[0] === '.')) continue;

            match = line.match(labelFind);
            if (!match) continue;

            if (!filterDirectives || this.hasOpcode(line, inNvccCode) || definesFunction) {
                // Only count a label as used if it's used by an opcode, or else we're not filtering directives.
                for (const label of match) labelsUsed[label] = true;
            } else {
                // If we have a current label, then any subsequent opcode or data definition's labels are referred to
                // weakly by that label.
                const isDataDefinition = !!this.dataDefn.test(line);
                const isOpcode = this.hasOpcode(line, inNvccCode);
                if (isDataDefinition || isOpcode) {
                    for (const currentLabel of currentLabelSet) {
                        if (inFunction && isDataDefinition) {
                            // Data definitions in the middle of code should be treated as if they were used strongly.
                            for (const label of match) labelsUsed[label] = true;
                        } else {
                            if (!weakUsages[currentLabel]) weakUsages[currentLabel] = [];
                            for (const label of match) weakUsages[currentLabel].push(label);
                        }
                    }
                }
            }
        }

        // Now follow the chains of used labels, marking any weak references they refer
        // to as also used. We iteratively do this until either no new labels are found,
        // or we hit a limit (only here to prevent a pathological case from hanging).
        function markUsed(label) {
            labelsUsed[label] = true;
        }

        const MaxLabelIterations = 10;
        for (let iter = 0; iter < MaxLabelIterations; ++iter) {
            const toAdd: string[] = [];
            _.each(labelsUsed, (t, label) => {
                // jshint ignore:line
                _.each(weakUsages[label], (nowused: string) => {
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
        for (const line of asmLines) {
            const match = line.match(this.fileFind);
            if (match) {
                const lineNum = parseInt(match[1]);
                if (match[4] && !line.includes('.cv_file')) {
                    // Clang-style file directive '.file X "dir" "filename"'
                    files[lineNum] = match[2] + '/' + match[4];
                } else {
                    files[lineNum] = match[2];
                }
            }
        }
        return files;
    }

    // Remove labels which do not have a definition.
    removeLabelsWithoutDefinition(asm, labelDefinitions) {
        for (const obj of asm) {
            if (obj.labels) {
                obj.labels = obj.labels.filter(label => labelDefinitions[label.name]);
            }
        }
    }

    // Get labels which are used in the given line.
    getUsedLabelsInLine(line) {
        const labelsInLine: AsmResultLabel[] = [];

        // Strip any comments
        const instruction = line.split(this.commentRe, 1)[0];

        // Remove the instruction.
        const params = instruction.replace(this.instructionRe, '');

        const removedCol = instruction.length - params.length + 1;
        params.replace(this.identifierFindRe, (label, index) => {
            const startCol = removedCol + index;
            labelsInLine.push({
                name: label,
                range: {
                    startCol: startCol,
                    endCol: startCol + label.length,
                },
            });
        });

        return labelsInLine;
    }

    processAsm(asmResult: string, filters: ParseFiltersAndOutputOptions): ParsedAsmResult {
        if (filters.binary || filters.binaryObject) return this.processBinaryAsm(asmResult, filters);

        const startTime = process.hrtime.bigint();

        if (filters.commentOnly) {
            // Remove any block comments that start and end on a line if we're removing comment-only lines.
            asmResult = asmResult.replace(this.blockComments, '');
        }

        const asm: ParsedAsmResultLine[] = [];
        const labelDefinitions: Record<string, number> = {};

        let asmLines = utils.splitLines(asmResult);
        const startingLineCount = asmLines.length;
        if (filters.preProcessLines !== undefined) {
            asmLines = filters.preProcessLines(asmLines);
        }

        const labelsUsed = this.findUsedLabels(asmLines, filters.directives);
        const files = this.parseFiles(asmLines);
        let prevLabel = '';

        let source: AsmResultSource | undefined | null;
        let mayRemovePreviousLabel = true;
        let keepInlineCode = false;

        let lastOwnSource: AsmResultSource | undefined | null;
        const dontMaskFilenames = filters.dontMaskFilenames;

        function maybeAddBlank() {
            const lastBlank = asm.length === 0 || asm[asm.length - 1].text === '';
            if (!lastBlank) asm.push({text: '', source: null, labels: []});
        }

        const handleSource = line => {
            let match = line.match(this.sourceTag);
            if (match) {
                const file = utils.maskRootdir(files[parseInt(match[1])]);
                const sourceLine = parseInt(match[2]);
                if (file) {
                    if (dontMaskFilenames) {
                        source = {
                            file: file,
                            line: sourceLine,
                            mainsource: !!this.stdInLooking.test(file),
                        };
                    } else {
                        source = {
                            file: this.stdInLooking.test(file) ? null : file,
                            line: sourceLine,
                        };
                    }
                    const sourceCol = parseInt(match[3]);
                    if (!isNaN(sourceCol) && sourceCol !== 0) {
                        source.column = sourceCol;
                    }
                } else {
                    source = null;
                }
            } else {
                match = line.match(this.sourceD2Tag);
                if (match) {
                    const sourceLine = parseInt(match[1]);
                    source = {
                        file: null,
                        line: sourceLine,
                    };
                } else {
                    match = line.match(this.sourceCVTag);
                    if (match) {
                        // cv_loc reports: function file line column
                        const sourceLine = parseInt(match[3]);
                        const file = utils.maskRootdir(files[parseInt(match[2])]);
                        if (dontMaskFilenames) {
                            source = {
                                file: file,
                                line: sourceLine,
                                mainsource: !!this.stdInLooking.test(file),
                            };
                        } else {
                            source = {
                                file: this.stdInLooking.test(file) ? null : file,
                                line: sourceLine,
                            };
                        }
                        const sourceCol = parseInt(match[4]);
                        if (!isNaN(sourceCol) && sourceCol !== 0) {
                            source.column = sourceCol;
                        }
                    }
                }
            }
        };

        const handleStabs = line => {
            const match = line.match(this.sourceStab);
            if (!match) return;
            // cf http://www.math.utah.edu/docs/info/stabs_11.html#SEC48
            switch (parseInt(match[1])) {
                case 68: {
                    source = {file: null, line: parseInt(match[2])};
                    break;
                }
                case 132:
                case 100: {
                    source = null;
                    prevLabel = '';
                    break;
                }
            }
        };

        const handle6502 = line => {
            const match = line.match(this.source6502Dbg);
            if (match) {
                const file = utils.maskRootdir(match[1]);
                const sourceLine = parseInt(match[2]);
                if (dontMaskFilenames) {
                    source = {
                        file: file,
                        line: sourceLine,
                        mainsource: !!this.stdInLooking.test(file),
                    };
                } else {
                    source = {
                        file: this.stdInLooking.test(file) ? null : file,
                        line: sourceLine,
                    };
                }
            } else if (this.source6502DbgEnd.test(line)) {
                source = null;
            }
        };

        let inNvccDef = false;
        let inNvccCode = false;

        let inCustomAssembly = 0;

        // TODO: Make this function smaller
        // eslint-disable-next-line max-statements
        for (let line of asmLines) {
            if (line.trim() === '') {
                maybeAddBlank();
                continue;
            }

            if (this.startAppBlock.test(line) || this.startAsmNesting.test(line)) {
                inCustomAssembly++;
            } else if (this.endAppBlock.test(line) || this.endAsmNesting.test(line)) {
                inCustomAssembly--;
            }

            handleSource(line);
            handleStabs(line);
            handle6502(line);

            if (source && (source.file === null || source.mainsource)) {
                lastOwnSource = source;
            }

            if (this.endBlock.test(line) || (inNvccCode && /}/.test(line))) {
                source = null;
                prevLabel = '';
                lastOwnSource = null;
            }

            if (filters.libraryCode && !lastOwnSource && source && source.file !== null && !source.mainsource) {
                if (mayRemovePreviousLabel && asm.length > 0) {
                    const lastLine = asm[asm.length - 1];

                    const labelDef = lastLine.text ? lastLine.text.match(this.labelDef) : null;

                    if (labelDef) {
                        asm.pop();
                        keepInlineCode = false;
                        delete labelDefinitions[labelDef[1]];
                    } else {
                        keepInlineCode = true;
                    }
                    mayRemovePreviousLabel = false;
                }

                if (!keepInlineCode) {
                    continue;
                }
            } else {
                mayRemovePreviousLabel = true;
            }

            if (
                filters.commentOnly &&
                ((this.commentOnly.test(line) && !inNvccCode) || (this.commentOnlyNvcc.test(line) && inNvccCode))
            ) {
                continue;
            }

            if (inCustomAssembly > 0) line = this.fixLabelIndentation(line);

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
                    if (filters.labels) {
                        continue;
                    }
                } else {
                    // A used label.
                    prevLabel = match[1];
                    labelDefinitions[match[1]] = asm.length + 1;
                }
            }
            if (inNvccDef) {
                if (this.cudaEndDef.test(line)) inNvccDef = false;
            } else if (!match && filters.directives) {
                // Check for directives only if it wasn't a label; the regexp would
                // otherwise misinterpret labels as directives.
                if (this.dataDefn.test(line) && prevLabel) {
                    // We're defining data that's being used somewhere.
                } else {
                    // .inst generates an opcode, so does not count as a directive
                    if (this.directive.test(line) && !this.instOpcodeRe.test(line)) {
                        continue;
                    }
                }
            }

            line = utils.expandTabs(line);
            const text = AsmRegex.filterAsmLine(line, filters);

            const labelsInLine = match ? [] : this.getUsedLabelsInLine(text);

            asm.push({
                text: text,
                source: this.hasOpcode(line, inNvccCode) ? source || null : null,
                labels: labelsInLine,
            });
        }

        this.removeLabelsWithoutDefinition(asm, labelDefinitions);

        const endTime = process.hrtime.bigint();
        return {
            asm: asm,
            labelDefinitions: labelDefinitions,
            parsingTime: ((endTime - startTime) / BigInt(1000000)).toString(),
            filteredCount: startingLineCount - asm.length,
        };
    }

    fixLabelIndentation(line) {
        const match = line.match(this.indentedLabelDef);
        if (match) {
            return line.replace(/^\s+/, '');
        } else {
            return line;
        }
    }

    isUserFunction(func) {
        if (this.binaryHideFuncRe === null) return true;

        return !this.binaryHideFuncRe.test(func);
    }

    processBinaryAsm(asmResult: string, filters: ParseFiltersAndOutputOptions): ParsedAsmResult {
        const startTime = process.hrtime.bigint();
        const asm: ParsedAsmResultLine[] = [];
        const labelDefinitions: Record<string, number> = {};
        const dontMaskFilenames = filters.dontMaskFilenames;

        let asmLines = asmResult.split('\n');
        const startingLineCount = asmLines.length;
        let source: AsmResultSource | undefined | null = null;
        let func: string | null = null;
        let mayRemovePreviousLabel = true;

        // Handle "error" documents.
        if (asmLines.length === 1 && asmLines[0][0] === '<') {
            return {
                asm: [{text: asmLines[0], source: null}],
            };
        }

        if (filters.preProcessBinaryAsmLines !== undefined) {
            asmLines = filters.preProcessBinaryAsmLines(asmLines);
        }

        for (const line of asmLines) {
            const labelsInLine: AsmResultLabel[] = [];

            if (asm.length >= this.maxAsmLines) {
                if (asm.length === this.maxAsmLines) {
                    asm.push({
                        text: '[truncated; too many lines]',
                        source: null,
                        labels: labelsInLine,
                    });
                }
                continue;
            }
            let match = line.match(this.lineRe);
            if (match) {
                assert(match.groups);
                if (dontMaskFilenames) {
                    source = {
                        file: utils.maskRootdir(match[1]),
                        line: parseInt(match.groups.line),
                        mainsource: true,
                    };
                } else {
                    source = {file: null, line: parseInt(match.groups.line), mainsource: true};
                }
                continue;
            }

            match = line.match(this.labelRe);
            if (match) {
                func = match[2];
                if (func && this.isUserFunction(func)) {
                    asm.push({
                        text: func + ':',
                        source: null,
                        labels: labelsInLine,
                    });
                    labelDefinitions[func] = asm.length;
                }
                continue;
            }

            if (func && line === `${func}():`) continue;

            if (!func || !this.isUserFunction(func)) continue;

            // note: normally the source.file will be null if it's code from example.ext
            //  but with filters.dontMaskFilenames it will be filled with the actual filename
            //  instead we can test source.mainsource in that situation
            const isMainsource = source && (source.file === null || source.mainsource);
            if (filters.libraryCode && !isMainsource) {
                if (mayRemovePreviousLabel && asm.length > 0) {
                    const lastLine = asm[asm.length - 1];
                    if (lastLine.text && this.labelDef.test(lastLine.text)) {
                        asm.pop();
                    }
                    mayRemovePreviousLabel = false;
                }
                continue;
            } else {
                mayRemovePreviousLabel = true;
            }

            match = line.match(this.asmOpcodeRe);
            if (match) {
                assert(match.groups);
                const address = parseInt(match.groups.address, 16);
                const opcodes = (match.groups.opcodes || '').split(' ').filter(x => !!x);
                const disassembly = ' ' + AsmRegex.filterAsmLine(match.groups.disasm, filters);
                const destMatch = line.match(this.destRe);
                if (destMatch) {
                    const labelName = destMatch[2];
                    const startCol = disassembly.indexOf(labelName) + 1;
                    labelsInLine.push({
                        name: labelName,
                        range: {
                            startCol: startCol,
                            endCol: startCol + labelName.length,
                        },
                    });
                }
                asm.push({
                    opcodes: opcodes,
                    address: address,
                    text: disassembly,
                    source: source,
                    labels: labelsInLine,
                });
            }

            match = line.match(this.relocationRe);
            if (match) {
                assert(match.groups);
                const address = parseInt(match.groups.address, 16);
                const relocname = match.groups.relocname;
                const relocdata = match.groups.relocdata;
                // value/addend matched but not used yet.
                const match_value = relocdata.match(this.relocDataSymNameRe);
                asm.push({
                    text: `   ${relocname} ${relocdata}`,
                    address: address,
                });
            }
        }

        this.removeLabelsWithoutDefinition(asm, labelDefinitions);

        const endTime = process.hrtime.bigint();

        return {
            asm: asm,
            labelDefinitions: labelDefinitions,
            parsingTime: ((endTime - startTime) / BigInt(1000000)).toString(),
            filteredCount: startingLineCount - asm.length,
        };
    }

    process(asm, filters) {
        return this.processAsm(asm, filters);
    }
}
