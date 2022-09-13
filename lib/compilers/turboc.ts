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

import path from 'path';

import {logger} from '../logger';

import {TurboCParser} from './argument-parsers';
import {DosboxCompiler} from './dosbox-compiler';

export class TurboCCompiler extends DosboxCompiler {
    static get key() {
        return 'turboc';
    }

    override optionsForFilter() {
        return ['-B'];
    }

    override getSharedLibraryPathsAsArguments(libraries: object[], libDownloadPath: string) {
        return [];
    }

    override getSharedLibraryPathsAsLdLibraryPaths(libraries: object[]) {
        return [];
    }

    override async getVersion() {
        logger.info(`Gathering ${this.compiler.id} version information on ${this.compiler.exe}...`);
        if (this.compiler.explicitVersion) {
            logger.debug(`${this.compiler.id} has forced version output: ${this.compiler.explicitVersion}`);
            return {stdout: [this.compiler.explicitVersion], stderr: [], code: 0};
        }
        const execOptions = this.getDefaultExecOptions();
        const versionFlag = '';
        execOptions.timeoutMs = 0;
        execOptions.ldPath = this.getSharedLibraryPathsAsLdLibraryPaths([]);

        try {
            return this.execCompilerCached(this.compiler.exe, [versionFlag], execOptions);
        } catch (err) {
            logger.error(`Unable to get version for compiler '${this.compiler.exe}' - ${err}`);
            return null;
        }
    }

    override getOutputFilename(dirPath: string, outputFilebase: string, key?: object) {
        return path.join(dirPath, 'EXAMPLE.ASM');
    }

    override getArgumentParser() {
        return TurboCParser;
    }
}
