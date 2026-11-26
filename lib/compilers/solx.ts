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

import path from 'node:path';

import type {SelectedLibraryVersion} from '../../types/libraries/libraries.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';
import {resultLinesToText} from '../utils.js';
import {SolxParser} from './argument-parsers.js';

export class SolxCompiler extends BaseCompiler {
    static get key() {
        return 'solx';
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
        return SolxParser;
    }

    override optionsForFilter(): string[] {
        return ['--asm'];
    }

    override isCfgCompiler() {
        return false;
    }

    override async processAsm(result) {
        const assembly = resultLinesToText(result.stdout);
        const sectionHeaders = [...assembly.matchAll(/^======= .+ =======$/gm)];
        const sections = sectionHeaders.map((header, index) =>
            assembly.slice(header.index, sectionHeaders[index + 1]?.index),
        );
        const sourceSections = sections.filter(section => this.isSourceSection(section));

        return {
            asm: [
                {
                    text:
                        sourceSections.length === 0
                            ? assembly
                            : this.orderSourceSections(assembly, sections, sourceSections),
                },
            ],
        };
    }

    private orderSourceSections(assembly: string, sections: string[], sourceSections: string[]): string {
        const preamble = assembly.slice(0, assembly.indexOf(sections[0]));
        const importedSections = sections.filter(section => !sourceSections.includes(section));
        return `${preamble}${[...sourceSections, ...importedSections].join('')}`;
    }

    private isSourceSection(section: string): boolean {
        const header = /^======= (?<source>.+):[^:]+ =======$/.exec(section.split('\n', 1)[0]);
        const source = header?.groups?.source ?? '';
        return source === '<source>' || path.basename(source) === this.compileFilename;
    }
}
