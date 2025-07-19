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

import {isString} from '../../shared/common-utils.js';
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

import {IAsmParser} from './asm-parser.interfaces.js';
import {AsmRegex} from './asmregex.js';
import {LabelContext, LabelProcessor} from './label-processor.js';
import {ParsingState} from './parsing-state.js';
import {SourceHandlerContext, SourceLineHandler} from './source-line-handler.js';

function maybeAddBlank(asm: ParsedAsmResultLine[]) {
    const lastBlank = asm.length === 0 || asm[asm.length - 1].text === '';
    if (!lastBlank) asm.push({text: '', source: null, labels: []});
}

export type ParsingContext = {
    files: Record<number, string>;
    source: AsmResultSource | undefined | null;
    dontMaskFilenames: boolean;
    prevLabel: string;
    prevLabelIsUserFunction: boolean;
};

export class AsmParser extends AsmRegex implements IAsmParser {
    protected sourceLineHandler: SourceLineHandler;
    protected labelProcessor: LabelProcessor;
    protected parsingState: ParsingState;

    protected maxAsmLines: number;

    protected labelFindNonMips: RegExp;
    protected labelFindMips: RegExp;
    protected mipsLabelDefinition: RegExp;
    protected dataDefn: RegExp;
    protected fileFind: RegExp;
    protected hasOpcodeRe: RegExp;
    protected instructionRe: RegExp;
    protected identifierFindRe: RegExp;
    protected hasNvccOpcodeRe: RegExp;
    protected definesFunction: RegExp;
    protected definesGlobal: RegExp;
    protected definesWeak: RegExp;
    protected definesAlias: RegExp;
    protected indentedLabelDef: RegExp;
    protected assignmentDef: RegExp;
    protected directive: RegExp;
    protected startAppBlock: RegExp;
    protected endAppBlock: RegExp;
    protected startAsmNesting: RegExp;
    protected endAsmNesting: RegExp;
    protected cudaBeginDef: RegExp;
    protected cudaEndDef: RegExp;
    protected binaryHideFuncRe: RegExp | null;
    protected asmOpcodeRe: RegExp;
    protected relocationRe: RegExp;
    protected relocDataSymNameRe: RegExp;
    protected lineRe: RegExp;
    protected labelRe: RegExp;
    protected destRe: RegExp;
    protected commentRe: RegExp;
    protected instOpcodeRe: RegExp;
    protected commentOnly: RegExp;
    protected commentOnlyNvcc: RegExp;
    protected sourceTag: RegExp;
    protected sourceD2Tag: RegExp;
    protected sourceCVTag: RegExp;
    protected source6502Dbg: RegExp;
    protected source6502DbgEnd: RegExp;
    protected sourceStab: RegExp;
    protected stdInLooking: RegExp;
    protected endBlock: RegExp;
    protected blockComments: RegExp;

    private updateParsingState(line: string, context: ParsingContext) {
        if (this.startAppBlock.test(line.trim()) || this.startAsmNesting.test(line.trim())) {
            this.parsingState.enterCustomAssembly();
        } else if (this.endAppBlock.test(line.trim()) || this.endAsmNesting.test(line.trim())) {
            this.parsingState.exitCustomAssembly();
        } else {
            this.parsingState.setVLIWPacket(this.checkVLIWpacket(line, this.parsingState.inVLIWpacket));
        }

        this.handleSource(context, line);
        this.handleStabs(context, line);
        this.handle6502(context, line);

        this.parsingState.updateSource(context.source);

        if (this.endBlock.test(line) || (this.parsingState.inNvccCode && /}/.test(line))) {
            context.source = null;
            context.prevLabel = '';
            this.parsingState.resetToBlockEnd();
        }
    }

    private shouldSkipDirective(
        line: string,
        filters: ParseFiltersAndOutputOptions,
        context: ParsingContext,
        match: RegExpMatchArray | null,
    ): boolean {
        if (this.parsingState.inNvccDef) {
            if (this.cudaEndDef.test(line)) this.parsingState.exitNvccDef();
            return false;
        }

        if (!match && filters.directives) {
            // Check for directives only if it wasn't a label; the regexp would otherwise misinterpret labels as directives.
            if (this.dataDefn.test(line) && context.prevLabel) {
                // We're defining data that's being used somewhere.
                return false;
            }
            // .inst generates an opcode, so does not count as a directive, nor does an alias definition that's used.
            if (this.directive.test(line) && !this.instOpcodeRe.test(line) && !this.definesAlias.test(line)) {
                return true;
            }
        }

        return false;
    }

    private processLabelDefinition(
        line: string,
        filters: ParseFiltersAndOutputOptions,
        context: ParsingContext,
        asmLines: string[],
        labelsUsed: Set<string>,
        labelDefinitions: Record<string, number>,
        asmLength: number,
    ): {match: RegExpMatchArray | null; skipLine: boolean} {
        let match = line.match(this.labelDef);
        if (!match) match = line.match(this.assignmentDef);
        if (!match) {
            match = line.match(this.cudaBeginDef);
            if (match) {
                this.parsingState.enterNvccDef();
            }
        }

        if (!match) {
            return {match: null, skipLine: false};
        }

        // It's a label definition. g-as shows local labels as eg: "1:  call  mcount". We characterize such a label
        // as "the label-matching part doesn't equal the whole line" and treat it as used. As a special case,
        // consider assignments of the form "symbol = ." to be labels.
        if (!labelsUsed.has(match[1]) && match[0] === line && (match[2] === undefined || match[2].trim() === '.')) {
            // It's an unused label.
            if (filters.labels) {
                context.prevLabel = '';
                return {match, skipLine: true};
            }
        } else {
            // A used label.
            context.prevLabel = match[1];
            labelDefinitions[match[1]] = asmLength + 1;

            if (!this.parsingState.inNvccDef && !this.parsingState.inNvccCode && filters.libraryCode) {
                context.prevLabelIsUserFunction = this.isUserFunctionByLookingAhead(
                    context,
                    asmLines,
                    this.parsingState.getCurrentLineIndex(),
                );
            }
        }

        return {match, skipLine: false};
    }

    private processAllLines(
        filters: ParseFiltersAndOutputOptions,
        context: ParsingContext,
        asmLines: string[],
        labelsUsed: Set<string>,
    ): {asm: ParsedAsmResultLine[]; labelDefinitions: Record<string, number>} {
        const asm: ParsedAsmResultLine[] = [];
        const labelDefinitions: Record<string, number> = {};
        for (let line of this.parsingState) {
            if (line.trim() === '') {
                maybeAddBlank(asm);
                continue;
            }

            this.updateParsingState(line, context);

            if (this.shouldSkipLibraryCode(filters, context, asm, labelDefinitions)) {
                continue;
            }

            if (this.shouldSkipCommentOnlyLine(filters, line)) {
                continue;
            }

            if (this.parsingState.isInCustomAssembly()) line = this.fixLabelIndentation(line);

            const labelResult = this.processLabelDefinition(
                line,
                filters,
                context,
                asmLines,
                labelsUsed,
                labelDefinitions,
                asm.length,
            );
            const match = labelResult.match;
            if (labelResult.skipLine) {
                continue;
            }

            if (this.shouldSkipDirective(line, filters, context, match)) {
                continue;
            }

            line = utils.expandTabs(line);
            const text = AsmRegex.filterAsmLine(line, filters);

            const labelsInLine = match ? [] : this.getUsedLabelsInLine(text);

            asm.push({
                text: text,
                source: this.hasOpcode(line, this.parsingState.inNvccCode, this.parsingState.inVLIWpacket)
                    ? context.source || null
                    : null,
                labels: labelsInLine,
            });
        }

        return {asm, labelDefinitions};
    }

    private shouldSkipCommentOnlyLine(filters: ParseFiltersAndOutputOptions, line: string): boolean {
        if (this.labelDef.test(line)) {
            return false;
        }
        return Boolean(
            filters.commentOnly &&
                ((this.commentOnly.test(line) && !this.parsingState.inNvccCode) ||
                    (this.commentOnlyNvcc.test(line) && this.parsingState.inNvccCode)),
        );
    }

    private shouldSkipLibraryCode(
        filters: ParseFiltersAndOutputOptions,
        context: ParsingContext,
        asm: ParsedAsmResultLine[],
        labelDefinitions: Record<string, number>,
    ): boolean {
        // Only filter library code if user enabled it AND we're not currently in a user function
        const doLibraryFilterCheck = filters.libraryCode && !context.prevLabelIsUserFunction;

        // Don't skip if any of these conditions indicate this is user code or filtering is disabled
        if (
            !doLibraryFilterCheck || // Library filtering disabled or we're in user function
            this.parsingState.lastOwnSource || // We recently processed user source code
            !context.source || // No source information available
            context.source.file === null || // Main source file (user code)
            context.source.mainsource // Explicitly marked as main source
        ) {
            // We're in user code, so future labels might need removal if we transition to library code
            this.parsingState.setMayRemovePreviousLabel(true);
            return false;
        }

        // We're in library code that should be filtered. Handle "orphaned labels" that precede filtered code.
        // When we start filtering library code, we might have just output a label that will now be orphaned.
        if (this.parsingState.shouldRemovePreviousLabel() && asm.length > 0) {
            const lastLine = asm[asm.length - 1];
            const labelDef = lastLine.text ? lastLine.text.match(this.labelDef) : null;

            if (labelDef) {
                // Last line was a label - it's now orphaned, so remove it retroactively
                asm.pop();
                this.parsingState.setKeepInlineCode(false);
                delete labelDefinitions[labelDef[1]];
            } else {
                // Last line wasn't a label - there's user code mixed in, so keep showing library code
                this.parsingState.setKeepInlineCode(true);
            }
            // Don't try to remove labels again until we transition back to user code
            this.parsingState.setMayRemovePreviousLabel(false);
        }

        // Skip this line unless we determined there's user code mixed in (keepInlineCode=true)
        return !this.parsingState.shouldKeepInlineCode();
    }

    constructor(compilerProps?: PropertyGetter) {
        super();

        this.sourceLineHandler = new SourceLineHandler();
        this.labelProcessor = new LabelProcessor();
        this.parsingState = new ParsingState({}, null, '', false, false, []);

        this.labelFindNonMips = /[.A-Z_a-z][\w$.]*|"[.A-Z_a-z][\w$.]*"/g;
        // MIPS labels can start with a $ sign, but other assemblers use $ to mean literal.
        this.labelFindMips = /[$.A-Z_a-z][\w$.]*|"[$.A-Z_a-z][\w$.]*"/g;
        this.mipsLabelDefinition = /^\$[\w$.]+:/;
        this.dataDefn =
            /^\s*\.(ascii|asciz|base64|[1248]?byte|dc(?:\.[abdlswx])?|dcb(?:\.[bdlswx])?|ds(?:\.[bdlpswx])?|double|dword|fill|float|half|hword|int|long|octa|quad|short|single|skip|space|string(?:8|16|32|64)?|value|word|xword|zero)/;
        this.fileFind = /^\s*\.(?:cv_)?file\s+(\d+)\s+"([^"]+)"(\s+"([^"]+)")?.*/;
        // Opcode expression here matches LLVM-style opcodes of the form `%blah = opcode`
        this.hasOpcodeRe = /^\s*(%[$.A-Z_a-z][\w$.]*\s*=\s*)?[A-Za-z]/;
        this.instructionRe = /^\s*[A-Za-z]+/;
        this.identifierFindRe = /([$.@A-Z_a-z]\w*)(?:@\w+)*/g;
        this.hasNvccOpcodeRe = /^\s*[@A-Za-z|]/;
        this.definesFunction = /^\s*\.(type.*,\s*[#%@]function|proc\s+[.A-Z_a-z][\w$.]*:.*)$/;
        this.definesGlobal = /^\s*\.(?:globa?l|GLB|export)\s*([.A-Z_a-z][\w$.]*|"[.A-Z_a-z][\w$.]*")/;
        this.definesWeak = /^\s*\.(?:weakext|weak)\s*([.A-Z_a-z][\w$.]*|"[.A-Z_a-z][\w$.]*")/;
        this.definesAlias = /^\s*\.set\s*((?:[.A-Z_a-z][\w$.]*|"[.A-Z_a-z][\w$.]*")\s*),\s*\.\s*(\+\s*0)?$/;
        this.indentedLabelDef = /^\s*([$.A-Z_a-z][\w$.]*|"[$.A-Z_a-z][\w$.]*"):/;
        this.assignmentDef = /^\s*([$.A-Z_a-z][\w$.]*)\s*=\s*(.*)/;
        this.directive = /^\s*\..*$/;
        // These four regexes when phrased as /\s*#APP.*/ etc exhibit costly polynomial backtracking. Instead use ^$ and
        // test with regex.test(line.trim()), more robust anyway
        this.startAppBlock = /^#APP.*$/;
        this.endAppBlock = /^#NO_APP.*$/;
        this.startAsmNesting = /^# Begin ASM.*$/;
        this.endAsmNesting = /^# End ASM.*$/;
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
        if (process.platform === 'win32') {
            this.lineRe = /^([A-Z]:\/[^:]+):(?<line>\d+).*/;
        } else {
            this.lineRe = /^(\/[^:]+):(?<line>\d+).*/;
        }

        // labelRe is made very greedy as it's also used with demangled objdump output (eg. it can have c++ template with <>).
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

    checkVLIWpacket(_line: string, inVLIWpacket: boolean) {
        return inVLIWpacket;
    }

    hasOpcode(line: string, inNvccCode = false, _inVLIWpacket = false) {
        // Remove any leading label definition...
        const match = line.match(this.labelDef);
        if (match) {
            line = line.substring(match[0].length);
        }
        // Strip any comments
        line = line.split(this.commentRe, 1)[0];
        // .inst generates an opcode, so also counts
        if (this.instOpcodeRe.test(line)) return true;
        // Detect assignment, that's not an opcode...
        if (this.assignmentDef.test(line)) return false;
        if (inNvccCode) {
            return this.hasNvccOpcodeRe.test(line);
        }
        return this.hasOpcodeRe.test(line);
    }

    private createLabelContext(): LabelContext {
        return {
            hasOpcode: this.hasOpcode.bind(this),
            checkVLIWpacket: this.checkVLIWpacket.bind(this),
            labelDef: this.labelDef,
            dataDefn: this.dataDefn,
            commentRe: this.commentRe,
            instructionRe: this.instructionRe,
            identifierFindRe: this.identifierFindRe,
            definesGlobal: this.definesGlobal,
            definesWeak: this.definesWeak,
            definesAlias: this.definesAlias,
            definesFunction: this.definesFunction,
            cudaBeginDef: this.cudaBeginDef,
            startAppBlock: this.startAppBlock,
            endAppBlock: this.endAppBlock,
            startAsmNesting: this.startAsmNesting,
            endAsmNesting: this.endAsmNesting,
            mipsLabelDefinition: this.mipsLabelDefinition,
            labelFindNonMips: this.labelFindNonMips,
            labelFindMips: this.labelFindMips,
            fixLabelIndentation: this.fixLabelIndentation.bind(this),
        };
    }

    labelFindFor(asmLines: string[]) {
        return this.labelProcessor.getLabelFind(asmLines, this.createLabelContext());
    }

    findUsedLabels(asmLines: string[], filterDirectives?: boolean): Set<string> {
        return this.labelProcessor.findUsedLabels(asmLines, filterDirectives || false, this.createLabelContext());
    }

    parseFiles(asmLines: string[]) {
        const files: Record<number, string> = {};
        for (const line of asmLines) {
            const match = line.match(this.fileFind);
            if (!match) continue;

            const lineNum = Number.parseInt(match[1]);
            if (match[4] && !line.includes('.cv_file')) {
                // Clang-style file directive '.file X "dir" "filename"'
                if (match[4].startsWith('/')) {
                    files[lineNum] = match[4];
                } else {
                    files[lineNum] = match[2] + '/' + match[4];
                }
            } else {
                files[lineNum] = match[2];
            }
        }
        return files;
    }

    removeLabelsWithoutDefinition(asm: ParsedAsmResultLine[], labelDefinitions: Record<string, number>) {
        this.labelProcessor.removeLabelsWithoutDefinition(asm, labelDefinitions);
    }

    getUsedLabelsInLine(line: string): AsmResultLabel[] {
        return this.labelProcessor.getUsedLabelsInLine(line, this.createLabelContext());
    }

    protected isUserFunctionByLookingAhead(context: ParsingContext, asmLines: string[], idxFrom: number): boolean {
        const funcContext: ParsingContext = {
            files: context.files,
            source: undefined,
            dontMaskFilenames: true,
            prevLabelIsUserFunction: false,
            prevLabel: '',
        };

        for (let idx = idxFrom; idx < asmLines.length; idx++) {
            const line = asmLines[idx];

            const endprocMatch = line.match(this.endBlock);
            if (endprocMatch) return false;

            this.handleSource(funcContext, line);
            this.handleStabs(funcContext, line);
            this.handle6502(funcContext, line);

            if (funcContext.source?.mainsource) return true;
        }

        return false;
    }

    protected handleSource(context: ParsingContext, line: string) {
        const sourceContext: SourceHandlerContext = {
            files: context.files,
            dontMaskFilenames: context.dontMaskFilenames,
        };

        const result = this.sourceLineHandler.processSourceLine(line, sourceContext);
        if (result.source !== undefined) context.source = result.source;
        if (result.resetPrevLabel) context.prevLabel = '';
    }

    protected handleStabs(context: ParsingContext, line: string) {
        const sourceContext: SourceHandlerContext = {
            files: context.files,
            dontMaskFilenames: context.dontMaskFilenames,
        };

        const result = this.sourceLineHandler.processSourceLine(line, sourceContext);
        if (result.source !== undefined) context.source = result.source;
        if (result.resetPrevLabel) context.prevLabel = '';
    }

    protected handle6502(context: ParsingContext, line: string) {
        const sourceContext: SourceHandlerContext = {
            files: context.files,
            dontMaskFilenames: context.dontMaskFilenames,
        };

        const result = this.sourceLineHandler.processSourceLine(line, sourceContext);
        if (result.source !== undefined) context.source = result.source;
        if (result.resetPrevLabel) context.prevLabel = '';
    }

    processAsm(asmResult: string, filters: ParseFiltersAndOutputOptions): ParsedAsmResult {
        if (filters.binary || filters.binaryObject) return this.processBinaryAsm(asmResult, filters);

        const startTime = process.hrtime.bigint();

        if (filters.commentOnly) {
            // Remove any block comments that start and end on a line if we're removing comment-only lines.
            asmResult = asmResult.replace(this.blockComments, '');
        }

        let asmLines = utils.splitLines(asmResult);
        const startingLineCount = asmLines.length;
        if (filters.preProcessLines) asmLines = filters.preProcessLines(asmLines);

        const labelsUsed = this.findUsedLabels(asmLines, filters.directives);

        const files = this.parseFiles(asmLines);
        this.parsingState = new ParsingState(files, null, '', false, filters.dontMaskFilenames || false, asmLines);

        const context: ParsingContext = {
            files: files,
            source: null,
            prevLabel: '',
            prevLabelIsUserFunction: false,
            dontMaskFilenames: filters.dontMaskFilenames || false,
        };

        const {asm, labelDefinitions} = this.processAllLines(filters, context, asmLines, labelsUsed);

        this.removeLabelsWithoutDefinition(asm, labelDefinitions);

        const endTime = process.hrtime.bigint();
        return {
            asm: asm,
            labelDefinitions: labelDefinitions,
            parsingTime: utils.deltaTimeNanoToMili(startTime, endTime),
            filteredCount: startingLineCount - asm.length,
        };
    }

    fixLabelIndentation(line: string) {
        const match = line.match(this.indentedLabelDef);
        return match ? line.replace(/^\s+/, '') : line;
    }

    isUserFunction(func: string) {
        if (this.binaryHideFuncRe === null) return true;

        return !this.binaryHideFuncRe.test(func);
    }

    processBinaryAsm(asmResult: string, filters: ParseFiltersAndOutputOptions): ParsedAsmResult {
        const startTime = process.hrtime.bigint();
        const asm: ParsedAsmResultLine[] = [];
        const labelDefinitions: Record<string, number> = {};
        const dontMaskFilenames = filters.dontMaskFilenames;

        let asmLines = utils.splitLines(asmResult);
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

        if (filters.preProcessBinaryAsmLines) asmLines = filters.preProcessBinaryAsmLines(asmLines);

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
                        line: Number.parseInt(match.groups.line),
                        mainsource: true,
                    };
                } else {
                    source = {file: null, line: Number.parseInt(match.groups.line), mainsource: true};
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
                    if (process.platform === 'win32') source = null;
                }
                continue;
            }

            if (func && line === `${func}():`) continue;

            if (!func || !this.isUserFunction(func)) continue;

            // note: normally the source.file will be null if it's code from example.ext but with
            //  filters.dontMaskFilenames it will be filled with the actual filename instead we can test
            //  source.mainsource in that situation
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
            }
            mayRemovePreviousLabel = true;

            match = line.match(this.asmOpcodeRe);
            if (match) {
                assert(match.groups);
                const address = Number.parseInt(match.groups.address, 16);
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
                const address = Number.parseInt(match.groups.address, 16);
                const relocname = match.groups.relocname;
                const relocdata = match.groups.relocdata;
                // value/addend matched but not used yet.
                // const match_value = relocdata.match(this.relocDataSymNameRe);
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
            parsingTime: utils.deltaTimeNanoToMili(startTime, endTime),
            filteredCount: startingLineCount - asm.length,
        };
    }

    process(asm: string, filters: ParseFiltersAndOutputOptions) {
        return this.processAsm(asm, filters);
    }
}
