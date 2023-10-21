import {
    AsmResultLabel,
    AsmResultSource,
    ParsedAsmResult,
    ParsedAsmResultLine,
} from '../../types/asmresult/asmresult.interfaces.js';
import {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {assert} from '../assert.js';
import {PropertyGetter} from '../properties.interfaces.js';
import * as utils from '../utils.js';

import {AsmParser} from './asm-parser.js';
import {AsmRegex} from './asmregex.js';

export class AsmParserZ88dk extends AsmParser {
    constructor(compilerProps: PropertyGetter) {
        super(compilerProps);

        this.asmOpcodeRe = /^(?<disasm>.*);\[(?<address>[\da-f]+)]\s*(?<opcodes>([\da-f]{2} ?)+)/;

        this.sourceTag = /^\s+C_LINE\s*(\d+),"([^"]+)(::\w*::\d*::\d*)?"/;
        this.labelDef = /^\.([\w$.@]+)$/i;
        this.definesGlobal = /^\s*GLOBAL\s*([.A-Z_a-z][\w$.]*)/;
        this.directive = /^\s*(C_LINE|MODULE|INCLUDE|SECTION|GLOBAL|EXTERN|IF|ENDIF)/;
    }

    override processAsm(asmResult: string, filters: ParseFiltersAndOutputOptions): ParsedAsmResult {
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
            const match = line.match(this.sourceTag);
            if (match) {
                const sourceLine = parseInt(match[1]);
                const file = utils.maskRootdir(match[2]);
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
            }
        };

        let inCustomAssembly = 0;

        for (let line of asmLines) {
            if (line.trim() === '') {
                maybeAddBlank();
                continue;
            }

            if (this.startAppBlock.test(line.trim()) || this.startAsmNesting.test(line.trim())) {
                inCustomAssembly++;
            } else if (this.endAppBlock.test(line.trim()) || this.endAsmNesting.test(line.trim())) {
                inCustomAssembly--;
            }

            handleSource(line);

            if (source && (source.file === null || source.mainsource)) {
                lastOwnSource = source;
            }

            if (this.endBlock.test(line)) {
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

            if (filters.commentOnly && this.commentOnly.test(line)) {
                continue;
            }

            if (inCustomAssembly > 0) line = this.fixLabelIndentation(line);

            let match = line.match(this.labelDef);
            if (!match) match = line.match(this.assignmentDef);
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

            if (!match && filters.directives) {
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
                source: this.hasOpcode(line, false) ? source || null : null,
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

    override processBinaryAsm(asmResult: string, filters: ParseFiltersAndOutputOptions): ParsedAsmResult {
        const startTime = process.hrtime.bigint();
        const asm: ParsedAsmResultLine[] = [];
        const labelDefinitions = {};

        let asmLines = asmResult.split('\n');
        const startingLineCount = asmLines.length;
        const source = null;

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

            const match = line.match(this.asmOpcodeRe);
            if (match) {
                assert(match.groups);
                const address = parseInt(match.groups.address, 16);
                const opcodes = (match.groups.opcodes || '').split(' ').filter(x => !!x);
                const disassembly = ' ' + AsmRegex.filterAsmLine(match.groups.disasm.trim(), filters);
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
        }

        const endTime = process.hrtime.bigint();

        return {
            asm: asm,
            labelDefinitions: labelDefinitions,
            parsingTime: ((endTime - startTime) / BigInt(1000000)).toString(),
            filteredCount: startingLineCount - asm.length,
        };
    }
}
