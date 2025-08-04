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

import {CompilerMultiVersionInfo, type PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import {ClangCompiler} from './clang.js';
import {CompilationEnvironment} from '../compilation-env.js';
import fs from 'node:fs/promises';
import {logger} from '../logger.js';
import {CompilationResult, ExecutionOptionsWithEnv} from '../../types/compilation/compilation.interfaces.js';

export class ManyClangsCompiler extends ClangCompiler {
    private reifiedCompilerPath: string| undefined;
    static override get key() {
        return 'manyclangs';
    }

    constructor(info: PreliminaryCompilerInfo, env: CompilationEnvironment) {
        super(info, env);
        this.reifiedCompilerPath = undefined;
    }

    // things we may have to override:
    // getCompilerCacheKey to include the version. (though that may not be necessary if we specialiseForVersion)

    // Unpacking and running the compiler is tricky as we use `this.compiler.exe` everywhere in the base class.
    // We may need to get rid of that to point it at the unpacked compiler executable.
    // Or - can we have a "specialise" method that returns a new instance of the compiler with the correct executable?
    // The trick there is we want to enqueue the request higher up so we can't specialise until the request is dequeued.
    // It _might_ be enough to lazily unpack.
    // Another qpproach: we override doCompilation. We still need a way to get the version info down though.
    // backendOptions may be a way to smuggle the value through.

    // Claude code suggestion:
    override specialiseForVersion(version: string): ManyClangsCompiler {
        logger.info('Specialising ManyClangsCompiler for version:', version);
        // Create a shallow copy of this instance
        const specialised = Object.create(Object.getPrototypeOf(this));
        Object.assign(specialised, this);

        // TODO lots of decisions here about where we unpack; and how to handle temp dirs, etc
        // need to consider how that works in the sandbox and we may have to map in another temp directory.
        // `/compiler/bin/clang` etc
        // this.compilerService.ensureCompilerUnpacked(this.compiler, version); //
        specialised.reifiedCompilerPath = 'moo';

        // Now override the bits that vary per-request
        specialised.compiler = {
            ...this.compiler,
            exe: /* TODO!... this.ensureClangUnpacked(gitSha)*/ '/home/moose',
            version: version,
            // whatever else varies by version
        };

        // TODO how to reify at the right time?
        // need to be careful, this shallow copy isn't quite the same as the original

        return specialised;
    }

    override async runCompiler(compiler: string, options: string[], inputFilename: string, execOptions: ExecutionOptionsWithEnv): Promise<CompilationResult> {
        // TODO maybe here ensure reified compiler is unpacked?
        return await super.runCompiler(compiler, options, inputFilename, execOptions);
    }

    override supportsMultipleVersions(): boolean {
        return true;
    }

    override async queryVersions(query: string): Promise<CompilerMultiVersionInfo[]> {
        // TODO hacky version for now:
        const commitList = '/home/matthew/dev/ce/commit-list.txt';
        // parse the file to get a list of versions; format is 'VERSION<whitespace>DESCRTIPION\n';
        const commitFileContents = await fs.readFile(commitList, 'utf8');
        return commitFileContents.split('\n')
            .filter((line) => line.trim() !== '')
            .filter((line) => line.includes(query))
            .slice(0, 100).map((line: string): CompilerMultiVersionInfo => {
                const [version, ...descriptionParts] = line.split(/\s+/);
                const description = descriptionParts.join(' ');
                return {version, description};
            });
    }
}
