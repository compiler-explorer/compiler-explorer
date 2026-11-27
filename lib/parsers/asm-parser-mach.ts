// Copyright (c) 2026, Compiler Explorer Authors
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

import type {ParsedAsmResult, ParsedAsmResultLine} from '../../types/asmresult/asmresult.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import type {PropertyGetter} from '../properties.interfaces.js';
import {AsmParser, type ParsingContext} from './asm-parser.js';
import {SPIRVAsmParser} from './asm-parser-spirv.js';

export const MACH_MODULE_SEP = '\n# __mach_module__\n';
export const MACH_SPIRV_HEADER = '; __mach_spirv__\n';

/**
 * Parses Mach assembly across masc text listings (`--emit asm`), linked/Wasm objdump disassembly, and SPIR-V
 * disassembly. Each module's `.s` listing owns its own `.file 0` table and `.Ltmp*` labels, so multi-module listings
 * are parsed per module and merged while keeping every user source under `src/` named as the tree pane names it.
 */
export class MachAsmParser extends AsmParser {
    private readonly sourcePrefix: string;
    private readonly spirvParser: SPIRVAsmParser;

    constructor(
        compilerProps: PropertyGetter | undefined,
        sourceDir: string,
        private readonly mainFilename = 'example.mach',
    ) {
        super(compilerProps);
        this.sourcePrefix = `${sourceDir}/`;
        this.spirvParser = new SPIRVAsmParser(compilerProps);
    }

    private isOwnSourceFile(file: string | null | undefined): boolean {
        if (!file) return false;
        const clean = file.replace(/^\.\//, '');
        return clean.startsWith(this.sourcePrefix);
    }

    private normaliseLines(lines: ParsedAsmResultLine[], dontMaskFilenames?: boolean) {
        for (const line of lines) {
            const source = line.source;
            if (!source || source.file === undefined || source.file === null) continue;
            const clean = source.file.replaceAll('\\', '/').replace(/^\.\//, '');
            if (clean.startsWith(this.sourcePrefix)) {
                const rel = clean.slice(this.sourcePrefix.length);
                const isMain = rel === this.mainFilename;
                line.source = {
                    ...source,
                    file: dontMaskFilenames || !isMain ? rel : null,
                    mainsource: isMain,
                };
            } else {
                line.source = {...source, mainsource: false};
            }
        }
    }

    protected override isUserFunctionByLookingAhead(
        context: ParsingContext,
        asmLines: string[],
        idxFrom: number,
    ): boolean {
        const funcContext: ParsingContext = {
            files: context.files,
            source: undefined,
            dontMaskFilenames: true,
            prevLabelIsUserFunction: false,
            prevLabel: '',
        };

        for (let idx = idxFrom; idx < asmLines.length; idx++) {
            const line = asmLines[idx];
            if (this.endBlock.test(line)) return false;

            this.handleSource(funcContext, line);
            this.handleStabs(funcContext, line);
            this.handle6502(funcContext, line);

            if (funcContext.source?.mainsource || this.isOwnSourceFile(funcContext.source?.file)) {
                return true;
            }
        }
        return false;
    }

    override processAsm(asmResult: string, filters: ParseFiltersAndOutputOptions): ParsedAsmResult {
        if (asmResult.startsWith(MACH_SPIRV_HEADER)) {
            return this.processSpirvModules(asmResult.slice(MACH_SPIRV_HEADER.length), filters);
        }
        if (filters.binary || filters.binaryObject) {
            return this.processBinaryAsm(asmResult, filters);
        }

        const modules = asmResult.split(MACH_MODULE_SEP);
        const mergedAsm: ParsedAsmResultLine[] = [];
        const mergedLabels: Record<string, number> = {};
        let parsingTime = 0;
        let filteredCount = 0;

        for (const chunk of modules) {
            // parse with dontMaskFilenames so secondary `sub/example.mach` never aliases `src/example.mach` before normaliseLines
            const subResult = super.processAsm(chunk, {...filters, dontMaskFilenames: true});
            this.normaliseLines(subResult.asm, filters.dontMaskFilenames);

            const lineOffset = mergedAsm.length;
            for (const [label, lineNum] of Object.entries(subResult.labelDefinitions ?? {})) {
                if (!label.startsWith('.L') || !(label in mergedLabels)) {
                    mergedLabels[label] = lineOffset + lineNum;
                }
            }
            mergedAsm.push(...subResult.asm);
            parsingTime += subResult.parsingTime ?? 0;
            filteredCount += subResult.filteredCount ?? 0;
        }

        if (modules.length > 1) {
            for (const line of mergedAsm) {
                if (!line.text) continue;
                const used = this.getUsedLabelsInLine(line.text).filter(
                    l => !l.name.startsWith('.L') && l.name in mergedLabels,
                );
                if (used.length > 0) {
                    const existing = new Set((line.labels ?? []).map(l => l.name));
                    line.labels = [...(line.labels ?? []), ...used.filter(l => !existing.has(l.name))];
                }
            }
        }

        return {
            asm: mergedAsm,
            labelDefinitions: mergedLabels,
            parsingTime,
            filteredCount,
        };
    }

    private processSpirvModules(asmResult: string, filters: ParseFiltersAndOutputOptions): ParsedAsmResult {
        const modules = asmResult.split(MACH_MODULE_SEP);
        const mergedAsm: ParsedAsmResultLine[] = [];
        const mergedLabels: Record<string, number> = {};
        let parsingTime = 0;
        let filteredCount = 0;

        for (const chunk of modules) {
            const subResult = this.spirvParser.processAsm(chunk, filters);
            this.normaliseLines(subResult.asm, filters.dontMaskFilenames);
            const lineOffset = mergedAsm.length;
            for (const [label, lineNum] of Object.entries(subResult.labelDefinitions ?? {})) {
                if (!(label in mergedLabels)) mergedLabels[label] = lineOffset + lineNum;
            }
            mergedAsm.push(...subResult.asm);
            parsingTime += Number(subResult.parsingTime ?? 0);
            filteredCount += subResult.filteredCount ?? 0;
        }

        return {
            asm: mergedAsm,
            labelDefinitions: mergedLabels,
            languageId: 'spirv',
            parsingTime,
            filteredCount,
        };
    }

    override processBinaryAsm(asmResult: string, filters: ParseFiltersAndOutputOptions): ParsedAsmResult {
        const result = super.processBinaryAsm(asmResult, filters);
        this.normaliseLines(result.asm, filters.dontMaskFilenames);
        return result;
    }
}
