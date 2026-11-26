// Copyright (c) 2023, Compiler Explorer Authors
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

import Semver from 'semver';

import type {CompilationResult} from '../../types/compilation/compilation.interfaces.js';
import type {LLVMIrBackendOptions} from '../../types/compilation/ir.interfaces.js';
import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';
import {CompilationEnvironment} from '../compilation-env.js';
import * as utils from '../utils.js';
import {asSafeVer} from '../utils.js';

// c3c writes one file per module section into the `--asm-out`/`--llvm-out` directory, named after the
// module with `::` replaced by `.`, never to the single path CE expects, so we join the user's files
// ourselves. Driven by the declarations, so the compiler_rt and generic instantiation files c3c writes
// alongside are never joined in.
const MODULE_DECL_RE = /^[^\S\n]*module\s+([a-zA-Z_]\w*(?:\s*::\s*[a-zA-Z_]\w*)*)/gm;

// c3c replaces anything that isn't a word character when deriving a module name from a file name.
const FILENAME_TO_MODULE_RE = /\W/g;

export class C3Compiler extends BaseCompiler {
    static get key() {
        return 'c3c';
    }

    constructor(info: PreliminaryCompilerInfo, env: CompilationEnvironment) {
        super(info, env);
        this.compiler.supportsIrView = true;
        this.compiler.irArg = ['--emit-llvm'];
    }

    override optionsForFilter() {
        const options = ['compile-only', '-g', '-l', 'pthread', '--no-obj', '--emit-asm'];
        if (Semver.gte(asSafeVer(this.compiler.semver), '0.6.8', true)) {
            options.push('--llvm-out', '.', '--asm-out', '.');
        }
        // c3c strips unused code by default, tracing only from main, `@export` and `@nostrip`, so a lone
        // uncalled function emits nothing at all: the common case on CE. Not stripping also re-emits the whole
        // standard library, which is several times slower, so skip writing that out: the pair is then faster
        // than c3c's default. 0.4 spells both as --no-*. User options come after these and c3c takes the last
        // occurrence, so either can be overridden.
        if (Semver.gte(asSafeVer(this.compiler.semver), '0.5.0', true)) {
            options.push('--strip-unused=no', '--emit-stdlib=no');
        } else {
            options.push('--no-strip-unused', '--no-emit-stdlib');
        }
        return options;
    }

    // The IR run otherwise reuses the asm options wholesale, re-emitting the asm alongside the IR.
    override async generateIR(
        inputFilename: string,
        options: string[],
        irOptions: LLVMIrBackendOptions,
        produceCfg: boolean,
        filters: ParseFiltersAndOutputOptions,
    ) {
        const irRunOptions = options.filter(
            (option, index) => option !== '--emit-asm' && option !== '--asm-out' && options[index - 1] !== '--asm-out',
        );
        return super.generateIR(inputFilename, irRunOptions, irOptions, produceCfg, filters);
    }

    override getIrOutputFilename(inputFilename: string): string {
        return this.filename(path.dirname(inputFilename) + '/output.ll');
    }

    // Blanks comments and string and character literals, keeping newlines so line anchors still match.
    // Mirrors c3c's lexer: `/* */` nests, `<* *>` doc comments don't, and a raw string escapes ` as ``.
    static blankNonCode(source: string): string {
        const out = source.split('');
        const blank = (from: number, to: number) => {
            for (let i = from; i < to; i++) if (out[i] !== '\n') out[i] = ' ';
        };
        let i = 0;
        while (i < source.length) {
            const start = i;
            const two = source.slice(i, i + 2);
            if (two === '//') {
                const end = source.indexOf('\n', i);
                i = end < 0 ? source.length : end;
            } else if (two === '/*') {
                let depth = 1;
                i += 2;
                while (i < source.length && depth > 0) {
                    const pair = source.slice(i, i + 2);
                    if (pair === '/*') {
                        depth++;
                        i += 2;
                    } else if (pair === '*/') {
                        depth--;
                        i += 2;
                    } else {
                        i++;
                    }
                }
            } else if (two === '<*') {
                const end = source.indexOf('*>', i + 2);
                i = end < 0 ? source.length : end + 2;
            } else if (source[i] === '`') {
                i++;
                while (i < source.length) {
                    if (source[i] === '`' && source[i + 1] === '`') i += 2;
                    else if (source[i++] === '`') break;
                }
            } else if (source[i] === '"' || source[i] === "'") {
                const quote = source[i++];
                // An unterminated literal is a compile error; stop at the line end so the rest is still read.
                while (i < source.length && source[i] !== '\n') {
                    if (source[i] === '\\') i += 2;
                    else if (source[i++] === quote) break;
                }
            } else {
                i++;
                continue;
            }
            blank(start, i);
        }
        return out.join('');
    }

    // The module name c3c derives for a file that declares none.
    static moduleNameForFile(inputFilename: string): string {
        return path.parse(inputFilename).name.replace(FILENAME_TO_MODULE_RE, '_');
    }

    // Module names as c3c would name its output files, in declaration order, or the derived name.
    async getModuleBaseNames(inputFilename: string): Promise<string[]> {
        let source: string;
        try {
            source = await fs.readFile(inputFilename, 'utf8');
        } catch {
            return [C3Compiler.moduleNameForFile(inputFilename)];
        }
        const code = C3Compiler.blankNonCode(source);
        const names = [...code.matchAll(MODULE_DECL_RE)].map(match => match[1].replaceAll(/\s*::\s*/g, '.'));
        // A module may be declared in several sections, all appending to one file: join it in once.
        const unique = [...new Set(names)];
        return unique.length > 0 ? unique : [C3Compiler.moduleNameForFile(inputFilename)];
    }

    async joinModuleOutputs(dirPath: string, baseNames: string[], extension: string, destination: string) {
        const parts: {name: string; text: string}[] = [];
        for (const baseName of baseNames) {
            const modulePath = path.join(dirPath, baseName + extension);
            // A `module output;` would otherwise have us copy the destination onto itself.
            if (modulePath !== destination && (await utils.fileExists(modulePath))) {
                parts.push({name: baseName, text: await fs.readFile(modulePath, 'utf8')});
            }
        }
        // Leave the destination absent, so the usual "no output file" reporting applies, not an empty pane.
        if (parts.length === 0) return;
        // Only label the chunks when there is more than one, so single-module output is unchanged.
        const comment = extension === '.ll' ? ';' : '#';
        const body =
            parts.length === 1
                ? parts[0].text
                : parts.map(part => `${comment} module ${part.name}\n${part.text}`).join('\n');
        await fs.writeFile(destination, body);
    }

    async joinModulesFor(inputFilename: string, extension: string, destination: string) {
        await this.joinModuleOutputs(
            path.dirname(inputFilename),
            await this.getModuleBaseNames(inputFilename),
            extension,
            destination,
        );
    }

    // Runs after every compiler invocation for this request has finished, handed the caller's own
    // outputFilename, so there is no second writer and no name to guess.
    override async postProcess(
        result: CompilationResult,
        outputFilename: string,
        filters: ParseFiltersAndOutputOptions,
        produceOptRemarks = false,
    ) {
        if (result.code === 0 && result.inputFilename) {
            await this.joinModulesFor(result.inputFilename, '.s', outputFilename);
            // checkOutputFileAndDoPostProcess sizes the output before calling us, i.e. before the file
            // exists at all. Without refreshing it, super reports "<No output file>" regardless.
            try {
                result.asmSize = (await fs.stat(outputFilename)).size;
            } catch {
                // Nothing written; leave asmSize unset so the usual reporting applies.
            }
        }
        return super.postProcess(result, outputFilename, filters, produceOptRemarks);
    }

    // The IR pane reads its own file, written by a separate --emit-llvm invocation, so join it here
    // rather than in postProcess. Only that invocation writes .ll files, so the two never collide.
    override async processIrOutput(
        output: CompilationResult,
        irOptions: LLVMIrBackendOptions,
        filters: ParseFiltersAndOutputOptions,
    ) {
        if (output.inputFilename)
            await this.joinModulesFor(output.inputFilename, '.ll', this.getIrOutputFilename(output.inputFilename));
        return super.processIrOutput(output, irOptions, filters);
    }
}
