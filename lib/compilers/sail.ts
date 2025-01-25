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

import fs from 'fs-extra';
import path from 'path';

import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';
import {CompilationEnvironment} from '../compilation-env.js';
import { CompilationResult, ExecutionOptionsWithEnv } from '../../types/compilation/compilation.interfaces.js';
import { SelectedLibraryVersion } from '../../types/libraries/libraries.interfaces.js';

export class SailCompiler extends BaseCompiler {
    static get key() {
        return 'sail';
    }

    constructor(info: PreliminaryCompilerInfo, env: CompilationEnvironment) {
        super(info, env);

        this.outputFilebase = 'model';
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: any) {
        // By default this adds C compiler options (-g etc).

        // Add a fake option so we know to compile to binary in `runCompiler()`.
        // This is how most other compilers work (with something like `-S`), and
        // the `BaseCompiler` isn't really set up to allow anything else.
        if (filters.binary) {
            return ["--link-to-binary"];
        }
        return [];
    }

    override async runCompiler(
        compiler: string,
        options: string[],
        inputFilename: string,
        execOptions: ExecutionOptionsWithEnv,
        filters?: ParseFiltersAndOutputOptions,
    ): Promise<CompilationResult> {
        if (!execOptions) {
            execOptions = this.getDefaultExecOptions();
        }

        const tmpDir = path.dirname(inputFilename);

        if (!execOptions.customCwd) {
            execOptions.customCwd = tmpDir;
        }

        // We can't use `filters` to decide whether to compile to binary
        // since it isn't always set (e.g. when this is called from
        // buildExecutable()). Instead we use a fake command line option.
        const binary = options.includes("--link-to-binary");
        options = options.filter(option => option !== "--link-to-binary");

        const fullResult: CompilationResult = {
            code: 0,
            timedOut: false,
            stdout: [],
            stderr: [],
            buildsteps: [],
            inputFilename,
        };

        const sailResult = await this.doBuildstepAndAddToResult(
            fullResult,
            'Sail to C',
            compiler,
            [...options, '-o', this.outputFilebase],
            execOptions,
        );

        if (sailResult.code !== 0 || !binary) {
            return fullResult;
        }

        fullResult.executableFilename = inputFilename + '.exe';

        // Query the sail compiler via `sail -dir` to find out where the
        // C runtime files are (`sail.c` etc).
        const sailDirResult = await this.doBuildstepAndAddToResult(
            fullResult,
            'Get Sail dir',
            compiler,
            ['-dir'],
            execOptions,
        );

        if (sailDirResult.code !== 0) {
            return fullResult;
        }

        const sailDir = sailDirResult.stdout.map(line => line.text).join('\n').trim();

        // Now compile the C file to an executable.
        const outputFilenameC = this.getOutputFilename(tmpDir, this.outputFilebase);
        const outputFilenameExe = outputFilenameC + '.exe';

        const compileResult = await this.doBuildstepAndAddToResult(
            fullResult,
            'C to binary',
            'cc',
            [
                outputFilenameC,
                // Sail C support files.
                '-I', `${sailDir}/lib`,
                `${sailDir}/lib/elf.c`,
                `${sailDir}/lib/rts.c`,
                `${sailDir}/lib/sail.c`,
                `${sailDir}/lib/sail_failure.c`,
                // Support for .elf.gz files. This has been removed in Sail 0.19
                // so it isn't needed for future versions.
                '-lz',
                // For arbitrary precision integer types.
                '-lgmp',
                // Enable optimisations so the assembly isn't hilariously verbose.
                // Ideally this would be user configurable but we'd need
                // something like `-Wl,..` to pass options through Sail to
                // here and that doesn't really exist.
                '-O',
                '-o', outputFilenameExe,
            ],
            execOptions,
        );

        // This is weird, but as far as I can tell CE expects the same
        // output file name when compiling to IR (C in this case)
        // and when compiling to a binary. If you don't do this it
        // tries to do nonsensical things like objdumping the C, so we
        // copy the binary back over the C file.
        if (compileResult.code === 0 && await fs.pathExists(outputFilenameExe)) {
            console.log(`Copying ${outputFilenameExe} to ${outputFilenameC}`);
            await fs.copyFile(outputFilenameExe, outputFilenameC);
        }

        return fullResult;
    }

    override getOutputFilename(dirPath: string, outputFilebase: string, key?: any): string {
        return path.join(dirPath, `${outputFilebase}.c`);
    }

    override getSharedLibraryPathsAsArguments(
        libraries: SelectedLibraryVersion[],
        libDownloadPath: string | undefined,
        toolchainPath: string | undefined,
        dirPath: string,
    ): string[] {
        // By default this adds -L./lib which isn't understood by the Sail compiler.
        return [];
    }
}
