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

import fs from 'node:fs/promises';
import path from 'node:path';

import type {CompilationResult} from '../../types/compilation/compilation.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import type {SelectedLibraryVersion} from '../../types/libraries/libraries.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';
import {ZksolcParser} from './argument-parsers.js';

export class SolidityZKsyncCompiler extends BaseCompiler {
    static get key() {
        return 'solidity-eravm';
    }

    override getSharedLibraryPathsAsArguments() {
        return [];
    }

    override getIncludeArguments(libraries: SelectedLibraryVersion[]) {
        const libraryPaths = libraries.flatMap(selectedLib => this.findLibVersion(selectedLib)?.path ?? []);
        if (libraryPaths.length === 0) return [];

        return ['--allow-paths', [...new Set(libraryPaths)].join(',')];
    }

    override getArgumentParserClass() {
        return ZksolcParser;
    }

    override optionsForFilter(): string[] {
        return ['--asm', '-o', 'contracts'];
    }

    override isCfgCompiler() {
        return false;
    }

    override getOutputFilename(dirPath: string) {
        return path.join(dirPath, 'contracts');
    }

    override async checkOutputFileAndDoPostProcess(
        asmResult: CompilationResult,
        outputDirectory: string,
        filters: ParseFiltersAndOutputOptions,
        produceOptRemarks = false,
    ) {
        const artifacts = await fs.readdir(outputDirectory, {recursive: true}).catch(() => []);
        const sourceArtifactPrefix = `${this.compileFilename}${path.sep}`;
        const legacySourceArtifactPrefix = `${this.compileFilename}:`;

        const isSourceArtifact = (artifact: string) =>
            artifact.startsWith(sourceArtifactPrefix) || artifact.startsWith(legacySourceArtifactPrefix);

        const zasmArtifacts = artifacts.filter(artifact => artifact.endsWith('.zasm'));
        const sourceArtifacts = zasmArtifacts.filter(isSourceArtifact).sort();
        const importedArtifacts = zasmArtifacts.filter(artifact => !isSourceArtifact(artifact)).sort();
        const orderedArtifacts = [...sourceArtifacts, ...importedArtifacts];

        const outputFilename = path.join(outputDirectory, 'combined.zasm');

        if (orderedArtifacts.length > 0) {
            const output = await Promise.all(
                orderedArtifacts.map(artifact => fs.readFile(path.join(outputDirectory, artifact), 'utf8')),
            );
            await fs.writeFile(outputFilename, output.join('\n'));
        }

        return super.checkOutputFileAndDoPostProcess(asmResult, outputFilename, filters, produceOptRemarks);
    }

    override async processAsm(result: CompilationResult) {
        return {asm: [{text: result.asm as string}]};
    }
}
