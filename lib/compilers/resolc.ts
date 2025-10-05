// Copyright (c) 2025, Compiler Explorer Authors
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

import fs from 'node:fs';
import path from 'node:path';

import type {ActiveTool, CacheKey} from '../../types/compilation/compilation.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import type {SelectedLibraryVersion} from '../../types/libraries/libraries.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';
import {changeExtension} from '../utils.js';
import {type BaseParser, ResolcParser} from './argument-parsers.js';

/**
 * The kind of output requested by the user.
 * Defaults to RISC-V, but can be changed via compiler options.
 */
enum OutputKind {
    RiscV,
    PolkaVM,
}

export class ResolcCompiler extends BaseCompiler {
    private outputKind = OutputKind.RiscV;

    static get key() {
        return 'resolc';
    }

    override getSharedLibraryPathsAsArguments(): string[] {
        return [];
    }

    override getArgumentParserClass(): typeof BaseParser {
        return ResolcParser;
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, _outputFilename: string): string[] {
        // For RISC-V output the binary object will be passed to llvm objdump.
        filters.binaryObject = this.outputKind === OutputKind.RiscV;
        // Keep the PolkaVM assembly header comments, such as the number of instructions and code size.
        filters.commentOnly = false;

        const options = ['-g', '--overwrite', '--debug-output-dir', 'artifacts'];

        return options;
    }

    override isCfgCompiler(): boolean {
        return false;
    }

    override getOutputFilename(dirPath: string): string {
        const artifactExtension = '.pvmasm';
        const basenamePrefix = dirPath.replaceAll('/', '_');
        const contractName = this.getSolidityContractName(dirPath);
        const outputFilename = path.join(
            dirPath,
            `artifacts/${basenamePrefix}_${this.compileFilename}.${contractName}${artifactExtension}`,
        );

        return outputFilename;
    }

    override getObjdumpOutputFilename(defaultOutputFilename: string): string {
        return changeExtension(defaultOutputFilename, '.o');
    }

    override async doCompilation(
        inputFilename: string,
        dirPath: string,
        key: CacheKey,
        options: string[],
        filters: ParseFiltersAndOutputOptions,
        backendOptions: Record<string, any>,
        libraries: SelectedLibraryVersion[],
        tools: ActiveTool[],
    ) {
        this.outputKind = options.includes('--asm') ? OutputKind.PolkaVM : OutputKind.RiscV;

        return super.doCompilation(inputFilename, dirPath, key, options, filters, backendOptions, libraries, tools);
    }

    /**
     * Get the Solidity component/contract name used in the compile file.
     *
     * Example:
     * ```solidity
     * contract Square { ... } // Name = Square
     * ```
     */
    private getSolidityContractName(dirPath: string): string {
        return this.getContractName(dirPath, 'contract');
    }

    private getContractName(dirPath: string, preceedingKeyword: string): string {
        const source = fs.readFileSync(`${dirPath}/${this.compileFilename}`, {encoding: 'utf8'});
        const whitespace = /\s+/;
        const sourceParts = source.split(whitespace);
        const contractStart = sourceParts.indexOf(preceedingKeyword);

        return contractStart >= 0 ? sourceParts[contractStart + 1] : 'contract_not_found';
    }
}
