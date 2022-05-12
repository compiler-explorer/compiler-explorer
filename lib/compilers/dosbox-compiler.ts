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

import fs from 'fs-extra';

import {BaseCompiler} from '../base-compiler';
import * as exec from '../exec';
import {logger} from '../logger';
import {TurboCAsmParser} from '../parsers/asm-parser-turboc';

export class DosboxCompiler extends BaseCompiler {
    private readonly dosbox: string;
    private readonly root: string;

    constructor(compilerInfo, env) {
        super(compilerInfo, env);

        this.dosbox = this.compilerProps(`compiler.${this.compiler.id}.dosbox`);
        this.root = this.compilerProps(`compiler.${this.compiler.id}.root`);
        this.asm = new TurboCAsmParser(this.compilerProps);
    }

    protected override async writeMultipleFiles(files: any[], dirPath: string): Promise<any[]> {
        const filesToWrite: any[] = [];

        for (const file of files) {
            if (!file.filename) throw new Error('One of more files do not have a filename');

            const fullpath = this.getExtraFilepath(dirPath, file.filename);
            const contents = file.contents.replaceAll(/\n/g, '\r\n');
            filesToWrite.push(fs.outputFile(fullpath, contents));
        }

        return Promise.all(filesToWrite);
    }

    protected override async writeAllFiles(dirPath: string, source: string, files: any[], filters: object) {
        if (!source) throw new Error(`File ${this.compileFilename} has no content or file is missing`);

        const inputFilename = path.join(dirPath, this.compileFilename);
        await fs.writeFile(inputFilename, source.replaceAll(/\n/g, '\r\n'));

        if (files && files.length > 0) {
            (filters as any).dontMaskFilenames = true;

            await this.writeMultipleFiles(files, dirPath);
        }

        return {
            inputFilename,
        };
    }

    private getDosboxArgs(tempDir: string, compileArgs: string[]) {
        const binPath = path.relative(this.root, path.dirname(this.compiler.exe));
        const exeName = path.basename(this.compiler.exe).replace(/\.exe$/i, '');
        return [
            '-c',
            `mount c ${this.root}`,
            '-c',
            `mount d ${tempDir}`,
            '-c',
            `PATH=%PATH%;C:\\${binPath}`,
            '-c',
            'd:',
            '-c',
            `${exeName} ${compileArgs.join(' ')} > STDOUT.TXT`,
            '-c',
            'exit',
        ];
    }

    private getDosboxEnv() {
        return {
            SDL_VIDEODRIVER: 'dummy',
        };
    }

    protected override async execCompilerCached(compiler, args, options) {
        if (this.mtime === null) {
            throw new Error('Attempt to access cached compiler before initialise() called');
        }
        if (!options) {
            options = this.getDefaultExecOptions();
            options.timeoutMs = 0;
            options.ldPath = this.getSharedLibraryPathsAsLdLibraryPaths([]);
        }

        const key = this.getCompilerCacheKey(compiler, args, options);
        let result = await this.env.compilerCacheGet(key);
        if (!result) {
            result = await this.env.enqueue(async () => this.exec(compiler, args, options));
            if (result.okToCache) {
                this.env
                    .compilerCachePut(key, result)
                    .then(() => {
                        // Do nothing, but we don't await here.
                    })
                    .catch(e => {
                        logger.info('Uncaught exception caching compilation results', e);
                    });
            }
        }

        return result;
    }

    public override async exec(filepath: string, args: string[], execOptions: any) {
        if (!execOptions) {
            execOptions = this.getDefaultExecOptions();
        }

        execOptions.env = this.getDosboxEnv();

        if (!execOptions.customCwd) {
            execOptions.customCwd = await this.newTempDir();
        }

        const tempDir = execOptions.customCwd;
        const fullArgs = this.getDosboxArgs(tempDir, args);

        const result = await exec.executeDirect(this.dosbox, fullArgs, execOptions);

        const stdoutFilename = path.join(tempDir, 'STDOUT.TXT');
        const stdout = await fs.readFile(stdoutFilename);
        (result as any).stdout = stdout.toString('utf8');

        return result;
    }

    public override async runCompiler(compiler, options, inputFilename, execOptions) {
        return super.runCompiler(
            compiler,
            options.map(option => {
                if (option === inputFilename) {
                    return path.basename(option);
                } else {
                    return option;
                }
            }),
            inputFilename,
            execOptions,
        );
    }
}
