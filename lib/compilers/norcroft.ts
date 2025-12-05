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

import _ from 'underscore';
import type {ExecutionOptions} from '../../types/compilation/compilation.interfaces.js';
import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {unwrap} from '../assert.js';
import {BaseCompiler} from '../base-compiler.js';
import {CompilationEnvironment} from '../compilation-env.js';
import {logger} from '../logger.js';
import {NorcroftObjAsmParser} from '../parsers/asm-parser-norcroft.js';

export class NorcroftCompiler extends BaseCompiler {
    constructor(compilerInfo: PreliminaryCompilerInfo, env: CompilationEnvironment) {
        super(compilerInfo, env);

        this.asm = new NorcroftObjAsmParser(this.compilerProps);
    }

    static get key() {
        return 'norcroft';
    }

    override async getVersion() {
        logger.info(`Gathering ${this.compiler.id} version information on ${this.compiler.exe}...`);
        if (this.compiler.explicitVersion) {
            logger.debug(`${this.compiler.id} has forced version output: ${this.compiler.explicitVersion}`);
            return {stdout: this.compiler.explicitVersion, stderr: '', code: 0};
        }
        const execOptions = this.getDefaultExecOptions();
        const versionFlag: string[] = [];
        execOptions.timeoutMs = 0;
        execOptions.ldPath = this.getSharedLibraryPathsAsLdLibraryPaths([]);

        try {
            const res = await this.execCompilerCached(this.compiler.exe, versionFlag, execOptions);
            return {stdout: res.stdout, stderr: res.stderr, code: res.code};
        } catch (err) {
            logger.error(`Unable to get version for compiler '${this.compiler.exe}' - ${err}`);
            return null;
        }
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string, userOptions?: string[]) {
        filters.binary = false;

        if (_.some(unwrap(userOptions), opt => opt === '-help' || opt === '-h')) {
            // Let the compiler print its own help text
            return [];
        }
        if (!_.some(unwrap(userOptions), opt => opt === '-E')) {
            filters.binary = false;
        }
        return ['-c', '-S', '--asm-includes-location', '-o', this.filename(outputFilename)];
    }

    override filterUserOptions(userOptions: string[]) {
        return userOptions.filter(opt => opt !== '-run' && !opt.startsWith('-W'));
    }

    override async runCompiler(
        compiler: string,
        options: string[],
        inputFilename: string,
        execOptions: ExecutionOptions & {env: Record<string, string>},
    ) {
        const result = await this.exec(compiler, options, execOptions);

        // Norcroft diagnostics look like:
        //   "<source>", line 11: Warning: message...
        //   "<source>", line 37: Error: message...
        // The generic applyParse_SourceWithLine expects:
        //   <source>:11:1: warning: message...
        //   <source>:37:1: error: message...
        //
        // Rewrite stderr to that shape so the generic parser can
        // extract file/line/severity and drive hover markers.
        if (typeof result.stderr === 'string') {
            result.stderr = result.stderr
                // Warnings
                .replace(/^"([^"]+)",\s*line\s+(\d+):\s*Warning:\s*/gm, '$1:$2:1: warning: ')
                // Errors and serious errors
                .replace(/^"([^"]+)",\s*line\s+(\d+):\s*(Error|Serious error):\s*/gm, '$1:$2:1: error: ');
        }

        return this.transformToCompilationResult(result, inputFilename);
    }
}
