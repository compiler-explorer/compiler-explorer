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

// A function that binaryHideFuncRe hides, parsed on the side. `at` is where it goes in the shown lines and `labels`
// has line indices relative to its own first line.
interface HiddenFunction {
    name: string;
    at: number;
    lines: ParsedAsmResultLine[];
    labels: Record<string, number>;
}

export class AsmParser extends AsmRegex implements IAsmParser {
    protected sourceLineHandler: SourceLineHandler;
    protected labelProcessor: LabelProcessor;
    protected parsingState: ParsingState;

    protected maxAsmLines: number;

    protected labelFindNonMips: RegExp;
    protected labelFindMips: RegExp;
    protected numericLocalLabelRef: RegExp;
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
    protected indentedLocalLabelDef: RegExp;
    protected assignmentDef: RegExp;
    protected setDef: RegExp;
    protected directive: RegExp;
    protected startAppBlock: RegExp;
    protected endAppBlock: RegExp;
    protected startAsmNesting: RegExp;
    protected endAsmNesting: RegExp;
    protected endOfUnit: RegExp;
    protected cudaBeginDef: RegExp;
    protected cudaEndDef: RegExp;
    protected binaryHideFuncRe: RegExp | null;
    protected asmOpcodeRe: RegExp;
    protected relocationRe: RegExp;
    protected relocDataSymNameRe: RegExp;
    protected lineRe: RegExp;
    protected labelRe: RegExp;
    protected destRe: RegExp;
    private binaryReferenceStartRe: RegExp;
    protected binaryCommentRe: RegExp;
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
    protected startBlock: RegExp;
    protected endBlock: RegExp;
    protected blockComments: RegExp;

    private updateParsingState(line: string, context: ParsingContext) {
        if (this.startAppBlock.test(line.trim()) || this.startAsmNesting.test(line.trim())) {
            this.parsingState.enterCustomAssembly();
        } else if (this.endAppBlock.test(line.trim()) || this.endAsmNesting.test(line.trim())) {
            this.parsingState.exitCustomAssembly();
        } else if (this.parsingState.isInCustomAssembly() && this.endOfUnit.test(line.trim())) {
            this.parsingState.closeCustomAssembly();
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
            // Between #APP and #NO_APP the directives are the user's own inline asm, and
            // dropping them changes what the block assembles to: `.rept 5` / `.endr` around
            // a `nop` collapses to a single `nop`. Only the source-location markers the
            // compiler interleaves in there are ours to drop.
            if (this.parsingState.isInCustomAssembly() && !this.sourceLineHandler.isSourceLocationDirective(line)) {
                return false;
            }
            // .inst generates an opcode, so does not count as a directive, nor does an alias definition that's used.
            if (
                this.directive.test(line) &&
                !this.instOpcodeRe.test(line) &&
                !this.definesAlias.test(line) &&
                !this.setDef.test(line)
            ) {
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
        let isSetDef = false;
        if (!match) {
            match = line.match(this.setDef);
            isSetDef = !!match;
        }
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
            labelDefinitions[match[1]] = asmLength + 1;

            if (isSetDef) {
                // `.set` does not start a new function
                return {match: null, skipLine: false};
            }

            context.prevLabel = match[1];

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
            else line = this.fixLocalLabelIndentation(line);

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
        // GNU as numeric local labels are defined as `1:` and referred to as `1f` (forwards) or
        // `1b` (backwards). The finders above only match names that start with a letter, a dot or
        // an underscore, so those references need a pattern of their own. The lookbehind keeps
        // NEON element sizes out of it: the `16b` of `add v0.16b, v1.16b, v2.16b` is not a label.
        this.numericLocalLabelRef = /(?<![\w.$])(\d+)[bf]\b/g;
        this.mipsLabelDefinition = /^\$[\w$.]+:/;
        this.dataDefn =
            /^\s*\.(ascii|asciz|base64|[1248]?byte|dc(?:\.[abdlswx])?|dcb(?:\.[bdlswx])?|ds(?:\.[bdlpswx])?|double|dword|fill|float|half|hword|int|long|octa|quad|short|single|skip|space|string(?:8|16|32|64)?|value|word|xword|zero)/;
        this.fileFind = /^\s*\.(?:cv_)?file\s+(\d+)\s+"([^"]+)"(\s+"([^"]+)")?.*/;
        // Opcode expression here matches LLVM-style opcodes of the form `%blah = opcode`
        this.hasOpcodeRe = /^\s*(%[$.A-Z_a-z][\w$.]*\s*=\s*)?[A-Za-z]/;
        this.instructionRe = /^\s*[A-Za-z]+/;
        this.identifierFindRe = /((?!\$\.)[$.@A-Z_a-z"][\w$.]*"?)(?:@\w+)*/g;
        this.hasNvccOpcodeRe = /^\s*[@A-Za-z|]/;
        this.definesFunction = /^\s*\.(type.*,\s*[#%@]function|proc\s+[.A-Z_a-z][\w$.]*:.*)$/;
        this.definesGlobal = /^\s*\.(?:globa?l|GLB|export)\s*([.@A-Z_a-z][\w$.@]*|"[.@A-Z_a-z][\w$.@]*")/;
        this.definesWeak = /^\s*\.(?:weakext|weak)\s*([.@A-Z_a-z][\w$.@]*|"[.@A-Z_a-z][\w$.@]*")/;
        this.definesAlias = /^\s*\.set\s*((?:[.A-Z_a-z][\w$.]*|"[.A-Z_a-z][\w$.]*")\s*),\s*\.\s*(\+\s*0)?$/;
        this.indentedLabelDef = /^\s*([$.A-Z_a-z][\w$.]*|"[$.A-Z_a-z][\w$.]*"):/;
        // Outside #APP only `.`-prefixed local labels are de-indented: some output still carries source text
        // (V8 echoes the JS source, for example) where an indented `key: value` is not a label definition.
        this.indentedLocalLabelDef = /^\s+\.[\w$.]+:/;
        this.assignmentDef = /^\s*([$.A-Z_a-z][\w$.]*)\s*=\s*(.*)/;
        // ".set label, label" where "label" is `this.labelFindNonMips`
        this.setDef =
            /^\s*\.set\s+([.A-Z_a-z][\w$.]*|"[.A-Z_a-z][\w$.]*"),\s*(?:[.A-Z_a-z][\w$.]*|"[.A-Z_a-z][\w$.]*")\s*$/;
        this.directive = /^\s*\..*$/;
        // These four regexes when phrased as /\s*#APP.*/ etc exhibit costly polynomial backtracking. Instead use ^$ and
        // test with regex.test(line.trim()), more robust anyway
        this.startAppBlock = /^#APP.*$/;
        this.endAppBlock = /^#NO_APP.*$/;
        this.startAsmNesting = /^# Begin ASM.*$/;
        this.endAsmNesting = /^# End ASM.*$/;
        // GCC writes #NO_APP only when it next emits a function or a variable, so asm that ends the unit leaves its
        // #APP block open. What follows is GCC's own end-of-unit output: the debug info, which opens with .Letext0,
        // or the .ident when there is none.
        this.endOfUnit = /^(\.Letext\d+:|\.ident\s)/;
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
        // The path is whatever objdump printed, not the host: `/file:N`, `C:/file:N`, or `C:\file:N`, or
        // `./file:N` from a build with trimmed paths (`go build -trimpath`).
        // llvm-objdump prefixes that same record with `; `.
        this.lineRe = /^(?:; )?(?<file>(?:[A-Z]:[\\/]|\.?\/)[^:]+):(?<line>\d+).*/;

        // labelRe is made very greedy as it's also used with demangled objdump output (eg. it can have c++ template with <>).
        this.labelRe = /^([\da-f]+)\s+<(.+)>:$/;
        this.destRe = /\s([\da-f]+)\s+<([^+>]+)(\+0x[\da-f]+)?>$/;
        // RISC-V, PowerPC and s390x print the address straight after a register operand: `jal ra,10008 <sym>`
        this.binaryReferenceStartRe = /[\s,][\da-f]+\s+</;
        // An x86 objdump comment about data an instruction reads (`# 404028 <sym>`). Elsewhere `#` is an immediate
        // prefix (`tbz w0, #0, <target>`), which is never followed by whitespace.
        this.binaryCommentRe = /\s#\s/;
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
        this.startBlock = /\.cfi_startproc\b/;
        this.endBlock = /\.(cfi_endproc|data|text|section)\b/;
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
        if (this.setDef.test(line)) return false;
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
            numericLocalLabelRef: this.numericLocalLabelRef,
            startBlock: this.startBlock,
            endBlock: this.endBlock,
            fixLabelIndentation: this.fixLabelIndentation.bind(this),
            fixLocalLabelIndentation: this.fixLocalLabelIndentation.bind(this),
        };
    }

    findUsedLabels(asmLines: string[], filterDirectives?: boolean): Set<string> {
        return this.labelProcessor.findUsedLabels(asmLines, filterDirectives || false, this.createLabelContext());
    }

    parseFiles(asmLines: string[]) {
        const files: Record<number, string> = {};
        for (const line of asmLines) {
            const match = line.match(this.fileFind);
            if (!match) continue;

            const lineNum = Number.parseInt(match[1], 10);
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

    fixLocalLabelIndentation(line: string) {
        return this.indentedLocalLabelDef.test(line) ? line.replace(/^\s+/, '') : line;
    }

    isUserFunction(func: string) {
        if (this.binaryHideFuncRe === null) return true;

        return !this.binaryHideFuncRe.test(func);
    }

    // PLT stubs and the .plt section they jump to stay hidden even when something shown refers to them
    private isPltName(name: string) {
        return name.startsWith('.plt') || name.includes('@plt');
    }

    // `.L` labels are local labels inside a function (RISC-V keeps `.Lpcrel_hi*` in the symbol table), not new functions
    private startsFunction(label: string) {
        return !label.startsWith('.L');
    }

    // The symbol a direct call or jump refers to, as in `call 401100 <name+0x4>`. Demangled names can hold `<`, `>` and
    // `+`, so the name runs from the first `<` that follows an address to the last `>`, less any `+0x` offset.
    private referenceTarget(line: string): string | undefined {
        const commentStart = line.search(this.binaryCommentRe);
        const instruction = commentStart === -1 ? line : line.slice(0, commentStart);
        const end = instruction.lastIndexOf('>');
        const addressMatch = this.binaryReferenceStartRe.exec(instruction);
        if (end === -1 || !addressMatch) return undefined;
        const start = addressMatch.index + addressMatch[0].length;
        if (start >= end) return undefined;
        const name = instruction.slice(start, end).replace(/\+0x[\da-f]+$/, '');
        return name || undefined;
    }

    // The hidden function that a direct call or jump on this line refers to, e.g. libgcc's __divdc3 in a statically
    // linked binary.
    private hiddenReferenceTarget(line: string): string | undefined {
        const target = this.referenceTarget(line);
        if (!target) return undefined;
        if (this.isPltName(target)) return undefined;
        return this.isUserFunction(target) ? undefined : target;
    }

    // Which of the hidden functions to show, given shown code that may already be at the limit. The output is the shown
    // lines and the chosen functions merged in file order and cut at the limit, so every chosen function has to end
    // before the cut and be referenced from a line before it. Start with everything that a shown line refers to and
    // drop what doesn't qualify until nothing does. Shown code never moves out of the output for the sake of a
    // function that isn't referenced from it, and showing a function pushes out only the shown code that follows it.
    private chooseHiddenFunctions(
        shownCount: number,
        hiddenFunctions: HiddenFunction[],
        firstReference: Map<string, number>,
    ): HiddenFunction[] {
        const limit = this.maxAsmLines;
        const kept = hiddenFunctions.filter(func => firstReference.has(func.name) && func.at <= shownCount);
        for (;;) {
            // heldBefore[i] is how many lines the kept functions ahead of kept[i] hold
            const heldBefore = [0];
            for (const func of kept) heldBefore.push(heldBefore[heldBefore.length - 1] + func.lines.length);
            // The first reference is the one that matters: if it is cut, so are the rest
            const positionOfShownLine = (index: number) => {
                let low = 0;
                let high = kept.length;
                while (low < high) {
                    const mid = (low + high) >> 1;
                    if (kept[mid].at <= index) low = mid + 1;
                    else high = mid;
                }
                return index + heldBefore[low];
            };
            // Only the last failing function goes each time: dropping one pulls what follows it back into the output,
            // which can rescue the others
            let failing = -1;
            for (let i = kept.length - 1; i >= 0 && failing === -1; i--) {
                const fits = kept[i].at + heldBefore[i] + kept[i].lines.length <= limit;
                if (!fits || positionOfShownLine(firstReference.get(kept[i].name)!) >= limit) failing = i;
            }
            if (failing === -1) return kept;
            kept.splice(failing, 1);
        }
    }

    // Shows the functions that binaryHideFuncRe hides but that a line of the output calls or jumps to. Only direct
    // references count: following them transitively would pull in all of stdio from a call to printf.
    private addReferencedHiddenFunctions(
        asm: ParsedAsmResultLine[],
        labelDefinitions: Record<string, number>,
        hiddenFunctions: HiddenFunction[],
        referenceTargets: Map<ParsedAsmResultLine, string>,
    ): ParsedAsmResultLine[] {
        const firstReference = new Map<string, number>();
        if (hiddenFunctions.length > 0) {
            asm.forEach((line, index) => {
                const target = referenceTargets.get(line);
                if (target && !firstReference.has(target)) firstReference.set(target, index);
            });
        }
        const kept = this.chooseHiddenFunctions(asm.length, hiddenFunctions, firstReference);
        if (kept.length === 0) return asm;

        // insertedBefore[i] is how many lines are inserted ahead of shown line i
        const insertedBefore: number[] = new Array(asm.length + 1).fill(0);
        for (const func of kept) insertedBefore[func.at] += func.lines.length;
        for (let i = 1; i <= asm.length; i++) insertedBefore[i] += insertedBefore[i - 1];

        // A label's index is 1-based: it names the line after the one before it
        for (const [label, index] of Object.entries(labelDefinitions)) {
            labelDefinitions[label] = index + insertedBefore[index - 1];
        }

        const merged: ParsedAsmResultLine[] = [];
        let nextKept = 0;
        for (let i = 0; i <= asm.length; i++) {
            while (nextKept < kept.length && kept[nextKept].at === i) {
                const func = kept[nextKept++];
                for (const [label, index] of Object.entries(func.labels)) {
                    labelDefinitions[label] = merged.length + index;
                }
                merged.push(...func.lines);
            }
            if (i < asm.length) merged.push(asm[i]);
        }
        return merged;
    }

    // The temp dir matcher only knows forward slashes, but a Windows build records e.g. C:\Temp\<tempdir>\src\helper.cpp
    private isProgramFile(file: string) {
        const normalised = file.replaceAll('\\', '/');
        return utils.maskRootdir(normalised) !== normalised;
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
        // A function is the user's if any of its code comes from a file of the program, i.e. from the directory it was
        // built in. Anything else is library code, including a statically linked runtime that keeps the paths it was
        // built from, e.g. clang's asan runtime refers to /root/llvm-project/compiler-rt/lib/asan/asan_allocator.cpp
        let functionStart = 0;
        let functionLabels: string[] = [];
        let functionHasUserCode = false;
        let truncated = false;
        let hiddenFunction: HiddenFunction | null = null;
        const hiddenFunctions: HiddenFunction[] = [];
        // The lines held for hiddenFunctions, referenced or not, are capped at the limit so that a binary with a lot of
        // hidden code can't use unbounded memory. A function that takes the pool past the cap is dropped, and once the
        // cap has been reached no later function is held, however small: at most the limit's worth of them could be
        // shown anyway, and the later ones are the first to go.
        let heldHiddenLines = 0;
        let hiddenCapReached = false;
        // Set for a hidden function that isn't being held, so that its local labels don't start showing its code
        let skippingFunction = false;
        const referenceTargets = new Map<ParsedAsmResultLine, string>();
        // Hidden functions' lines go into their own buffer so they never count towards the limit
        let out = asm;
        const mayRemoveFunction = () => filters.libraryCode && !functionHasUserCode;
        const removeFunctionIfLibrary = () => {
            if (!mayRemoveFunction()) return;
            if (hiddenFunction) {
                heldHiddenLines -= hiddenFunction.lines.length;
                hiddenFunctions.pop();
                return;
            }
            asm.splice(functionStart);
            for (const label of functionLabels) delete labelDefinitions[label];
        };
        const abandonHiddenFunction = () => {
            if (!hiddenFunction) return;
            heldHiddenLines -= hiddenFunction.lines.length;
            hiddenFunctions.pop();
            hiddenFunction = null;
            hiddenCapReached = true;
            skippingFunction = true;
        };
        // Whether the line was kept: a line of a hidden function that is over the cap drops the whole function
        const pushLine = (line: ParsedAsmResultLine) => {
            if (hiddenFunction) {
                if (heldHiddenLines >= this.maxAsmLines) {
                    abandonHiddenFunction();
                    return false;
                }
                heldHiddenLines++;
            }
            out.push(line);
            return true;
        };
        const markTruncated = () => {
            if (truncated) return;
            truncated = true;
        };
        // A function kept for its user code may have run past the limit while it could still be removed
        const keepFunction = () => {
            functionHasUserCode = true;
            if (!hiddenFunction && asm.length > this.maxAsmLines) {
                asm.splice(this.maxAsmLines);
                for (const label of functionLabels) {
                    if (labelDefinitions[label] > this.maxAsmLines) delete labelDefinitions[label];
                }
                markTruncated();
            }
        };

        // Handle "error" documents.
        if (asmLines.length === 1 && asmLines[0][0] === '<') {
            return {
                asm: [{text: asmLines[0], source: null}],
            };
        }

        if (filters.preProcessBinaryAsmLines) asmLines = filters.preProcessBinaryAsmLines(asmLines);

        for (const line of asmLines) {
            const labelsInLine: AsmResultLabel[] = [];

            // A function that may still be removed doesn't use up the limit, otherwise a large statically linked
            // runtime would hide the user's functions behind it
            if (asm.length >= this.maxAsmLines && !mayRemoveFunction()) {
                markTruncated();
                continue;
            }
            let match = line.match(this.lineRe);
            if (match) {
                assert(match.groups);
                // Subclasses that override lineRe (SASS) capture the path as group 1 and do not name it `file`.
                const file = match.groups.file ?? match[1];
                assert(file);
                const mainsource = this.stdInLooking.test(file);
                source = {
                    file: dontMaskFilenames || !mainsource ? utils.maskRootdir(file) : null,
                    line: Number.parseInt(match.groups.line, 10),
                    mainsource,
                };
                if (!functionHasUserCode && (mainsource || this.isProgramFile(file))) keepFunction();
                continue;
            }

            match = line.match(this.labelRe);
            if (match) {
                func = match[2];
                // Don't let a function with no line records (e.g. CRT code without debug info) inherit the previous
                // function's line; objdump prints a function's first record after its label.
                if (this.startsFunction(func)) {
                    source = null;
                    removeFunctionIfLibrary();
                    functionStart = asm.length;
                    functionLabels = [];
                    functionHasUserCode = false;
                    hiddenFunction = null;
                    skippingFunction = false;
                    if (!this.isUserFunction(func)) {
                        // A PLT name is never promoted, so holding it would only use up the cap
                        if (!this.isPltName(func) && heldHiddenLines >= this.maxAsmLines) hiddenCapReached = true;
                        if (this.isPltName(func) || hiddenCapReached) {
                            skippingFunction = true;
                        } else {
                            hiddenFunction = {name: func, at: asm.length, lines: [], labels: {}};
                            hiddenFunctions.push(hiddenFunction);
                        }
                    }
                    out = hiddenFunction ? hiddenFunction.lines : asm;
                }
                if (func && !skippingFunction && (hiddenFunction || this.isUserFunction(func))) {
                    const pushed = pushLine({
                        text: func + ':',
                        source: null,
                        labels: labelsInLine,
                    });
                    if (hiddenFunction) {
                        if (pushed) hiddenFunction.labels[func] = out.length;
                    } else if (pushed) {
                        labelDefinitions[func] = out.length;
                        functionLabels.push(func);
                    }
                }
                continue;
            }

            if (func && (line === `${func}():` || line === `; ${func}():`)) continue;

            if (!func || skippingFunction || !(hiddenFunction || this.isUserFunction(func))) continue;

            // Within the user's functions, library code is code without line information: code inlined from another
            // file is still the user's, even though it doesn't belong to the main source
            if (filters.libraryCode && !source) continue;

            match = line.match(this.asmOpcodeRe);
            if (match) {
                assert(match.groups);
                const address = Number.parseInt(match.groups.address, 16);
                const opcodes = (match.groups.opcodes || '').split(' ').filter(x => !!x);
                const disassembly = ' ' + AsmRegex.filterAsmLine(match.groups.disasm, filters);
                const labelName = this.referenceTarget(line);
                if (labelName) {
                    const startCol = disassembly.lastIndexOf(labelName) + 1;
                    labelsInLine.push({
                        name: labelName,
                        range: {
                            startCol: startCol,
                            endCol: startCol + labelName.length,
                        },
                    });
                }
                const instruction: ParsedAsmResultLine = {
                    opcodes: opcodes,
                    address: address,
                    text: disassembly,
                    source: source,
                    labels: labelsInLine,
                };
                if (!pushLine(instruction)) continue;
                if (!hiddenFunction && this.binaryHideFuncRe !== null) {
                    const target = this.hiddenReferenceTarget(line);
                    if (target) referenceTargets.set(instruction, target);
                }
            }

            match = line.match(this.relocationRe);
            if (match) {
                assert(match.groups);
                const address = Number.parseInt(match.groups.address, 16);
                const relocname = match.groups.relocname;
                const relocdata = match.groups.relocdata;
                // value/addend matched but not used yet.
                // const match_value = relocdata.match(this.relocDataSymNameRe);
                pushLine({
                    text: `   ${relocname} ${relocdata}`,
                    address: address,
                });
            }
        }
        removeFunctionIfLibrary();

        const shownAsm = this.addReferencedHiddenFunctions(asm, labelDefinitions, hiddenFunctions, referenceTargets);
        if (shownAsm.length > this.maxAsmLines) {
            shownAsm.splice(this.maxAsmLines);
            truncated = true;
        }
        if (truncated) {
            for (const [label, index] of Object.entries(labelDefinitions)) {
                if (index > this.maxAsmLines) delete labelDefinitions[label];
            }
            shownAsm.push({text: '[truncated; too many lines]', source: null, labels: []});
        }
        this.removeLabelsWithoutDefinition(shownAsm, labelDefinitions);

        const endTime = process.hrtime.bigint();

        return {
            asm: shownAsm,
            labelDefinitions: labelDefinitions,
            parsingTime: utils.deltaTimeNanoToMili(startTime, endTime),
            filteredCount: startingLineCount - shownAsm.length,
        };
    }

    process(asm: string, filters: ParseFiltersAndOutputOptions) {
        return this.processAsm(asm, filters);
    }
}
