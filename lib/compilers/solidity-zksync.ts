// Copyright (c) 2024, Compiler Explorer Authors
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

import path from 'path';

import {BaseCompiler} from '../base-compiler.js';

import {ZksolcParser} from './argument-parsers.js';

export class SolidityZKsyncCompiler extends BaseCompiler {
    static get key() {
        return 'solidity-eravm';
    }

    override getSharedLibraryPathsAsArguments() {
        return [];
    }

    override getArgumentParserClass() {
        return ZksolcParser;
    }

    override optionsForFilter(): string[] {
        return ['--combined-json', 'asm', '-o', 'contracts'];
    }

    override isCfgCompiler() {
        return false;
    }

    override getOutputFilename(dirPath: string) {
        return path.join(dirPath, 'contracts/combined.json');
    }

    override async processAsm(result) {
        // Handle "error" documents.
        if (!result.asm.includes('\n') && result.asm[0] === '<') {
            return {asm: [{text: result.asm}]};
        }

        const combinedJson = JSON.parse(result.asm);
        const asm: any[] = [];
        for (const build of Object.values(combinedJson.contracts) as JSON[]) {
            asm.push({text: build['asm']});
        }
        return {asm};
    }
}
