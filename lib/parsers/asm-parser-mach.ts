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

/** Separates the modules of one compilation in the text handed to the parser. */
export const MACH_MODULE_SEP = '\n# __mach_module__\n';

/**
 * Parses Mach assembly across masc text listings (`--emit asm`), objdump disassembly of objects and linked binaries,
 * and SPIR-V disassembly. Each module's `.s` listing owns its own `.file 0` table and `.Ltmp*` labels, so multi-module
 * listings are parsed per module and merged while keeping every user source under `src/` named as the tree pane names
 * it.
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
        return file.replace(/^\.\//, '').startsWith(this.sourcePrefix);
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

    /**
     * Parses each module on its own and joins the results into one listing. A label a later module defines again
     * keeps its first definition unless `redefines` lets the later one take it.
     */
    private mergeModules(
        asmResult: string,
        filters: ParseFiltersAndOutputOptions,
        parse: (module: string) => ParsedAsmResult,
        redefines: (label: string) => boolean,
    ): ParsedAsmResult {
        const asm: ParsedAsmResultLine[] = [];
        const labelDefinitions: Record<string, number> = {};
        let parsingTime = 0;
        let filteredCount = 0;

        for (const module of asmResult.split(MACH_MODULE_SEP)) {
            const parsed = parse(module);
            this.normaliseLines(parsed.asm, filters.dontMaskFilenames);
            for (const [label, line] of Object.entries(parsed.labelDefinitions ?? {})) {
                if (redefines(label) || !(label in labelDefinitions)) labelDefinitions[label] = asm.length + line;
            }
            asm.push(...parsed.asm);
            parsingTime += Number(parsed.parsingTime ?? 0);
            filteredCount += parsed.filteredCount ?? 0;
        }
        return {asm, labelDefinitions, parsingTime, filteredCount};
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
        if (filters.binary || filters.binaryObject) {
            return this.processBinaryAsm(asmResult, filters);
        }

        // parsed unmasked, so that a module's `sub/example.mach` is not taken for the entry before normaliseLines
        const result = this.mergeModules(
            asmResult,
            filters,
            module => super.processAsm(module, {...filters, dontMaskFilenames: true}),
            label => !label.startsWith('.L'),
        );

        if (!asmResult.includes(MACH_MODULE_SEP)) return result;

        // a module only links the labels it defines itself, which leaves a call into another module unlinked
        const defined = result.labelDefinitions ?? {};
        for (const line of result.asm) {
            if (!line.text) continue;
            const linked = new Set((line.labels ?? []).map(label => label.name));
            const crossModule = this.getUsedLabelsInLine(line.text).filter(
                label => !label.name.startsWith('.L') && label.name in defined && !linked.has(label.name),
            );
            if (crossModule.length > 0) line.labels = [...(line.labels ?? []), ...crossModule];
        }
        return result;
    }

    /** The disassembly of one or more SPIR-V modules, as `spirv-dis` writes it. */
    processSpirv(asmResult: string, filters: ParseFiltersAndOutputOptions): ParsedAsmResult {
        return {
            ...this.mergeModules(
                asmResult,
                filters,
                module => this.spirvParser.processAsm(module, filters),
                () => false,
            ),
            languageId: 'spirv',
        };
    }

    override processBinaryAsm(asmResult: string, filters: ParseFiltersAndOutputOptions): ParsedAsmResult {
        const result = super.processBinaryAsm(asmResult, filters);
        this.normaliseLines(result.asm, filters.dontMaskFilenames);
        return result;
    }
}
