// Copyright (c) 2018, Compiler Explorer Authors
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

import type {ParsedAsmResult} from '../../types/asmresult/asmresult.interfaces.js';
import type {ExecutionOptions} from '../../types/compilation/compilation.interfaces.js';
import type {UnprocessedExecResult} from '../../types/execution/execution.interfaces.js';
import {unwrap} from '../assert.js';
import {BaseCompiler} from '../base-compiler.js';
import {logger} from '../logger.js';
import {AsmRegex} from '../parsers/asmregex.js';
import {SymbolStore} from '../symbol-store.js';
import * as utils from '../utils.js';

import {PrefixTree} from './prefix-tree.js';

export class BaseDemangler extends AsmRegex {
    readonly demanglerExe: string;
    readonly demanglerArguments: string[];
    symbolstore: SymbolStore | null;
    othersymbols: SymbolStore;
    result: ParsedAsmResult;
    input: string[];
    readonly includeMetadata: boolean;
    readonly compiler: BaseCompiler;

    readonly jumpDef = /(j\w+|b|bl|blx)\s+([$_a-z][\w$@]*)/i;
    readonly callDef = /callq?\s+([$._a-z][\w$.@]*)/i;
    readonly callPtrDef1 = /callq?.*ptr\s\[[a-z]*\s\+\s([$._a-z][\w$.@]*)]/i;
    readonly callPtrDef2 = /callq?\s+([$*._a-z][\w$.@]*)/i;
    readonly callPtrDef3 = /callq?.*\[qword ptr\s([$._a-z][\w$.@]*).*]/i;
    readonly callPtrDef4 = /callq?.*qword\sptr\s\[[a-z]*\s\+\s([$._a-z][\w$.@]*)\+?\d?]/i;

    // symbols in a mov or lea command starting with an underscore
    readonly movUnderscoreDef = /mov.*\s(_[\w$.@]*)/i;
    readonly leaUnderscoreDef = /lea.*\s(_[\w$.@]*)/i;
    readonly quadUnderscoreDef = /\.quad\s*(_[\w$.@]*)/i;

    // E.g., ".entry _Z6squarePii("
    // E.g., ".func  (.param .b32 func_retval0) bar("
    readonly ptxFuncDef = /\.(entry|func)\s+(?:\([^)]*\)\s*)?([$.A-Z_a-z][\w$.]*)\(/;
    // E.g., ".const .attribute(.managed) .align 4 .v4 .u32 myvar"
    // E.g., ".global .texref mytex"
    readonly ptxVarDef =
        /^\.(global|const)\s+(?:\.(tex|sampler|surf)ref\s+)?(?:\.attribute\([^)]*\)\s+)?(?:\.align\s+\d+\s+)?(?:\.v\d+\s+)?(?:\.[a-z]\d+\s+)?([$.A-Z_a-z][\w$.]*)/;

    constructor(demanglerExe: string, compiler: BaseCompiler, demanglerArguments: string[] = []) {
        super();

        this.demanglerExe = demanglerExe;
        this.demanglerArguments = demanglerArguments;
        this.symbolstore = null;
        this.othersymbols = new SymbolStore();
        this.result = {
            asm: [],
        };
        this.input = [];
        this.includeMetadata = true;
        this.compiler = compiler;
    }

    // Iterates over the labels, demangle the label names and updates the start and
    // end position of the label.
    protected demangleLabels(labels, tree: PrefixTree) {
        if (!Array.isArray(labels) || labels.length === 0) return;

        for (const [index, label] of labels.entries()) {
            const value = label.name;
            const newValue = tree.findExact(value);
            if (newValue) {
                label.name = newValue;
                label.range.endCol = label.range.startCol + newValue.length;

                // Update the startCol value for each further labels.
                for (let j = index + 1; j < labels.length; j++) {
                    labels[j].range.startCol += newValue.length - value.length;
                }
            }
        }
    }

    protected demangleLabelDefinitions(labelDefinitions, translations: [string, string][]) {
        if (!labelDefinitions) return;

        for (const [oldValue, newValue] of translations) {
            const value = labelDefinitions[oldValue];
            if (value && oldValue !== newValue) {
                labelDefinitions[newValue] = value;
                delete labelDefinitions[oldValue];
            }
        }
    }

    protected collectLabels() {
        const symbolMatchers = [
            this.jumpDef,
            this.callPtrDef4,
            this.callPtrDef3,
            this.callPtrDef2,
            this.callPtrDef1,
            this.callDef,
            this.movUnderscoreDef,
            this.leaUnderscoreDef,
            this.quadUnderscoreDef,
            this.ptxFuncDef,
            this.ptxVarDef,
        ];
        for (let j = 0; j < this.result.asm.length; ++j) {
            const line = this.result.asm[j].text;
            if (!line) continue;

            const labelMatch = line.match(this.labelDef);
            if (labelMatch) unwrap(this.symbolstore).add(labelMatch[labelMatch.length - 1]);

            for (const reToMatch of symbolMatchers) {
                const matches = line.match(reToMatch);
                if (matches) {
                    this.othersymbols.add(matches[matches.length - 1]);
                    break;
                }
            }
        }

        this.othersymbols.exclude(unwrap(this.symbolstore));
    }

    protected getInput() {
        this.input = [];
        this.input = this.input.concat(unwrap(this.symbolstore).listSymbols());
        this.input = this.input.concat(this.othersymbols.listSymbols());

        return this.input.join('\n');
    }

    protected getMetadata(symbol: string): {ident: RegExp; description: string}[] {
        return [];
    }

    protected addTranslation(symbol: string, translation: string) {
        if (this.includeMetadata) {
            translation += this.getMetadata(symbol)
                .map(meta => ` [${meta.description}]`)
                .join(',');
        }

        if (unwrap(this.symbolstore).contains(symbol)) {
            unwrap(this.symbolstore).add(symbol, translation);
        } else {
            this.othersymbols.add(symbol, translation);
        }
    }

    protected processOutput(output: UnprocessedExecResult) {
        if (output.stdout.length === 0 && output.stderr.length > 0) {
            logger.error(`Error executing demangler ${this.demanglerExe}`, output);
            return this.result;
        }

        const lines = utils.splitLines(output.stdout);
        if (lines.length > this.input.length) {
            logger.error(`Demangler output issue ${lines.length} > ${this.input.length}`, this.input, output);
            throw new Error('Internal issue in demangler');
        }
        for (let i = 0; i < lines.length; ++i) this.addTranslation(this.input[i], lines[i]);

        const translations = [
            ...unwrap(this.symbolstore).listTranslations(),
            ...this.othersymbols.listTranslations(),
        ].filter(elem => elem[0] !== elem[1]);
        if (translations.length > 0) {
            const tree = new PrefixTree(translations);

            for (const asm of this.result.asm) {
                asm.text = tree.replaceAll(asm.text);
                this.demangleLabels(asm.labels, tree);
            }

            this.demangleLabelDefinitions(this.result.labelDefinitions, translations);
        }
        return this.result;
    }

    protected execDemangler(options: ExecutionOptions) {
        options.maxOutput = -1;

        return this.compiler.exec(this.demanglerExe, this.demanglerArguments, options);
    }

    public async process(result: ParsedAsmResult, execOptions?: ExecutionOptions) {
        const options = execOptions || {};
        this.result = result;

        if (!this.symbolstore) {
            this.symbolstore = new SymbolStore();
        }

        this.collectLabels();

        options.input = this.getInput();

        if (options.input === '') {
            return this.result;
        }

        return this.processOutput(await this.execDemangler(options));
    }
}
