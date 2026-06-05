// Copyright (c) 2022, Compiler Explorer Authors
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

import {ParsedAsmResult} from '../../types/asmresult/asmresult.interfaces.js';
import {OptPipelineResults} from '../../types/compilation/opt-pipeline-output.interfaces.js';
import {ResultLine} from '../../types/resultline/resultline.interfaces.js';
import {BaseDemangler} from './base.js';
import {PrefixTree} from './prefix-tree.js';

export class LLVMIRDemangler {
    // Identifiers can be quoted: https://llvm.org/docs/LangRef.html#identifiers
    llvmSymbolRE = /@(?<symbol>[\w$.]+)/gi;
    llvmQuotedSymbolRE = /@"(?<symbol>[^"]+)"/gi;

    symbolDemangler: BaseDemangler;
    result: ParsedAsmResult;
    labels: string[];

    static get key() {
        return 'llvm-ir';
    }

    constructor(symbolDemangler: BaseDemangler) {
        this.symbolDemangler = symbolDemangler;
        this.result = {
            asm: [],
        };
        this.labels = [];
    }

    public canDemangle() {
        return this.symbolDemangler.canDemangle();
    }

    private collectLabels() {
        const labels = new Set<string>();
        for (const line of this.result.asm) {
            const text = line.text;
            if (!text) continue;

            const matches = [...text.matchAll(this.llvmSymbolRE), ...text.matchAll(this.llvmQuotedSymbolRE)];
            for (const match of matches) {
                labels.add(match.groups!.symbol);
            }
        }
        this.labels = [...labels];
    }

    public collect(result: {asm: ResultLine[]}) {
        this.result = result;
        this.collectLabels();
    }

    protected processPassOutput(passOutput: OptPipelineResults, translations: [string, string][]) {
        if (translations.length > 0) {
            const tree = new PrefixTree(translations);
            for (const [functionName, passes] of Object.entries(passOutput)) {
                const demangledFunctionName = tree.replaceAll(functionName).newText;
                for (const pass of passes) {
                    pass.name = tree.replaceAll(pass.name).newText; // needed at least for full module mode
                    for (const dump of [pass.before, pass.after]) {
                        for (const line of dump) {
                            line.text = tree.replaceAll(line.text).newText;
                        }
                    }
                }
                delete passOutput[functionName];
                passOutput[demangledFunctionName] = passes;
            }
        }
        return passOutput;
    }

    public async demangleLLVMPasses(passOutput: OptPipelineResults) {
        if (this.labels.length === 0) {
            return passOutput;
        }
        await this.symbolDemangler.process({asm: []}, {overrideSymbols: this.labels, skipTranslation: true});
        return this.processPassOutput(passOutput, this.symbolDemangler.getTranslations());
    }

    public async process(result: ParsedAsmResult) {
        this.result = result;
        this.collectLabels();
        if (this.labels.length === 0) {
            return result;
        }
        return await this.symbolDemangler.process(result, {overrideSymbols: this.labels});
    }
}
