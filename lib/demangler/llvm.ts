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

import {OptPipelineResults} from '../../types/compilation/opt-pipeline-output.interfaces.js';
import {ResultLine} from '../../types/resultline/resultline.interfaces.js';
import {logger} from '../logger.js';
import {SymbolStore} from '../symbol-store.js';
import * as utils from '../utils.js';

import {BaseDemangler} from './base.js';
import {PrefixTree} from './prefix-tree.js';

export class LLVMIRDemangler extends BaseDemangler {
    llvmSymbolRE = /@([\w$.]+)/gi;

    static get key() {
        return 'llvm-ir';
    }

    protected override collectLabels() {
        for (const line of this.result.asm) {
            const text = line.text;
            if (!text) continue;

            const matches = [...text.matchAll(this.llvmSymbolRE)];
            for (const match of matches) {
                // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
                this.symbolstore!.add(match[1]);
            }
        }
    }

    public collect(result: {asm: ResultLine[]}) {
        this.result = result;
        if (!this.symbolstore) {
            this.symbolstore = new SymbolStore();
            this.collectLabels();
        }
    }

    protected processPassOutput(passOutput: OptPipelineResults, demanglerOutput) {
        if (demanglerOutput.stdout.length === 0 && demanglerOutput.stderr.length > 0) {
            logger.error(`Error executing demangler ${this.demanglerExe}`, demanglerOutput);
            return passOutput;
        }

        const lines = utils.splitLines(demanglerOutput.stdout);
        if (lines.length > this.input.length) {
            logger.error(`Demangler output issue ${lines.length} > ${this.input.length}`, this.input, demanglerOutput);
            throw new Error('Internal issue in demangler');
        }
        for (let i = 0; i < lines.length; ++i) this.addTranslation(this.input[i], lines[i]);

        // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
        const translations = [...this.symbolstore!.listTranslations(), ...this.othersymbols.listTranslations()].filter(
            elem => elem[0] !== elem[1],
        );
        if (translations.length > 0) {
            const tree = new PrefixTree(translations);
            for (const [functionName, passes] of Object.entries(passOutput)) {
                const demangledFunctionName = tree.replaceAll(functionName);
                for (const pass of passes) {
                    pass.name = tree.replaceAll(pass.name); // needed at least for full module mode
                    for (const dump of [pass.before, pass.after]) {
                        for (const line of dump) {
                            line.text = tree.replaceAll(line.text);
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
        const options = {
            input: this.getInput(),
        };

        if (options.input === '') {
            return passOutput;
        }
        return this.processPassOutput(passOutput, await this.execDemangler(options));
    }
}
