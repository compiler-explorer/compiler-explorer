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

// c3c emits one file per module section, named after the module with `::` replaced by `.`, into the
// `--asm-out`/`--llvm-out` directory. A file with no `module` declaration gets a module named after
// the file. Nothing is ever written to the single output path CE expects, so we work out which files
// belong to the user's source and join them together ourselves.
//
// Deliberately driven by the module declarations rather than by "every .s that isn't the standard
// library": it is legal to declare `module std::io` in user code, in which case c3c appends to the
// standard library's own file and a name-based exclusion would discard the user's output.
const MODULE_DECL_RE = /^[^\S\n]*module\s+([a-zA-Z_]\w*(?:\s*::\s*[a-zA-Z_]\w*)*)/gm;

// Raw strings and the three comment forms, blanked before looking for module declarations so that a
// commented-out or quoted `module std::io;` doesn't have us serve up the standard library's assembly.
const NON_CODE_RE = /`[^`]*`|<\*[\s\S]*?\*>|\/\*[\s\S]*?\*\/|\/\/[^\n]*/g;

// c3c replaces anything that isn't a word character when it derives a module name from a file name.
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

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string) {
        const options = ['compile-only', '-g', '-l', 'pthread', '--no-obj', '--emit-asm'];
        if (Semver.gte(asSafeVer(this.compiler.semver), '0.6.8', true)) {
            options.push('--llvm-out', '.', '--asm-out', '.');
        }
        // c3c defaults to --strip-unused=yes, tracing only from main, `@export` and `@nostrip`, so a
        // lone uncalled function produces no output at all. That is the common case on CE, so ask for
        // everything. Users can put --strip-unused=yes back: user options are appended after these,
        // and c3c takes the last occurrence.
        if (Semver.gte(asSafeVer(this.compiler.semver), '0.5.0', true)) {
            options.push('--strip-unused=no');
        }
        return options;
    }

    override getIrOutputFilename(
        inputFilename: string,
        filters?: ParseFiltersAndOutputOptions,
        irOptions?: LLVMIrBackendOptions,
    ): string {
        return this.filename(path.dirname(inputFilename) + '/output.ll');
    }

    // The module name c3c derives for a file that declares none.
    static moduleNameForFile(inputFilename: string): string {
        return path.parse(inputFilename).name.replace(FILENAME_TO_MODULE_RE, '_');
    }

    // Module names for the sections declared in the source, in declaration order, as c3c would name
    // the files it writes. Falls back to the name derived from the file, matching what c3c does when
    // the source declares no module at all.
    async getModuleBaseNames(inputFilename: string): Promise<string[]> {
        let source: string;
        try {
            source = await fs.readFile(inputFilename, 'utf8');
        } catch {
            return [C3Compiler.moduleNameForFile(inputFilename)];
        }
        // Blank out non-code, preserving newlines so the line anchors below still line up.
        const code = source.replaceAll(NON_CODE_RE, match => match.replaceAll(/[^\n]/g, ' '));
        const names = [...code.matchAll(MODULE_DECL_RE)].map(match => match[1].replaceAll(/\s*::\s*/g, '.'));
        // Order matters, but a module may be declared more than once in one file and each declaration
        // appends to the same output file, so only join it in once.
        const unique = [...new Set(names)];
        return unique.length > 0 ? unique : [C3Compiler.moduleNameForFile(inputFilename)];
    }

    async joinModuleOutputs(dirPath: string, baseNames: string[], extension: string, destination: string) {
        const parts: {name: string; text: string}[] = [];
        for (const baseName of baseNames) {
            const modulePath = path.join(dirPath, baseName + extension);
            // Read everything before writing: a `module output;` would otherwise have us copy the
            // destination onto itself.
            if (modulePath !== destination && (await utils.fileExists(modulePath))) {
                parts.push({name: baseName, text: await fs.readFile(modulePath, 'utf8')});
            }
        }
        // Leave the destination absent when there is nothing to write, so the usual "no output"
        // reporting still happens rather than showing an empty pane.
        if (parts.length === 0) return;
        // One module is overwhelmingly the common case; only label the chunks when there is more than
        // one to tell apart, so ordinary output is untouched.
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

    // Runs once, after every compiler invocation for this request has finished, and is handed the
    // caller's own outputFilename, so there is no second writer and no need to guess the name.
    override async postProcess(
        result: CompilationResult,
        outputFilename: string,
        filters: ParseFiltersAndOutputOptions,
        produceOptRemarks = false,
    ) {
        if (result.code === 0 && result.inputFilename) {
            await this.joinModulesFor(result.inputFilename, '.s', outputFilename);
            // checkOutputFileAndDoPostProcess sizes the output before calling us, which is before the
            // file exists at all. Without refreshing it, super reports "<No output file>" regardless.
            try {
                result.asmSize = (await fs.stat(outputFilename)).size;
            } catch {
                // Nothing was written; leave asmSize unset so the usual reporting applies.
            }
        }
        return super.postProcess(result, outputFilename, filters, produceOptRemarks);
    }

    // The IR pane reads its own file, written by a separate --emit-llvm invocation, so it is joined
    // here rather than in postProcess. Only that invocation writes .ll files, so the two never collide.
    override async processIrOutput(
        output: CompilationResult,
        irOptions: LLVMIrBackendOptions,
        filters: ParseFiltersAndOutputOptions,
    ) {
        if (output.inputFilename)
            await this.joinModulesFor(
                output.inputFilename,
                '.ll',
                this.getIrOutputFilename(output.inputFilename, filters, irOptions),
            );
        return super.processIrOutput(output, irOptions, filters);
    }
}
