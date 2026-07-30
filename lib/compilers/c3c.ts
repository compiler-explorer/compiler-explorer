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

import type {CompilationResult, ExecutionOptionsWithEnv} from '../../types/compilation/compilation.interfaces.js';
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

    override getIrOutputFilename(inputFilename: string): string {
        return this.filename(path.dirname(inputFilename) + '/output.ll');
    }

    // Module names for the sections declared in the source, in declaration order, as c3c would name
    // the files it writes. Falls back to the source's own basename, matching how c3c names the module
    // for a file that declares none.
    async getModuleBaseNames(inputFilename: string): Promise<string[]> {
        let source: string;
        try {
            source = await fs.readFile(inputFilename, 'utf8');
        } catch {
            return [];
        }
        const names = [...source.matchAll(MODULE_DECL_RE)].map(match => match[1].replaceAll(/\s*::\s*/g, '.'));
        // Order matters, but a module may be declared more than once in one file and each declaration
        // appends to the same output file, so only join it in once.
        const unique = [...new Set(names)];
        return unique.length > 0 ? unique : [path.parse(inputFilename).name];
    }

    async joinModuleOutputs(dirPath: string, baseNames: string[], extension: string, destination: string) {
        const parts: string[] = [];
        for (const baseName of baseNames) {
            const modulePath = path.join(dirPath, baseName + extension);
            // Read everything before writing: a `module output;` would otherwise have us copy the
            // destination onto itself.
            if (modulePath !== destination && (await utils.fileExists(modulePath))) {
                parts.push(await fs.readFile(modulePath, 'utf8'));
            }
        }
        // Leave the destination absent when there is nothing to write, so the usual "no output"
        // reporting still happens rather than showing an empty pane.
        if (parts.length > 0) await fs.writeFile(destination, parts.join('\n'));
    }

    override async runCompiler(
        compiler: string,
        options: string[],
        inputFilename: string,
        execOptions: ExecutionOptionsWithEnv,
        filters?: ParseFiltersAndOutputOptions,
    ): Promise<CompilationResult> {
        const result = await super.runCompiler(compiler, options, inputFilename, execOptions, filters);
        if (result.code === 0) {
            const dirPath = path.dirname(inputFilename);
            const baseNames = await this.getModuleBaseNames(inputFilename);
            // Both panes are fed from per-module files, and which one this invocation produced depends
            // on whether --emit-llvm was added, so gather up whatever is there.
            await this.joinModuleOutputs(
                dirPath,
                baseNames,
                '.s',
                this.getOutputFilename(dirPath, this.outputFilebase),
            );
            await this.joinModuleOutputs(dirPath, baseNames, '.ll', this.getIrOutputFilename(inputFilename));
        }
        return result;
    }
}
