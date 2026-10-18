// Copyright (c) 2026, Compiler Explorer Authors
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

import _ from 'underscore';

import type {BuildSystemDescriptor, BuildSystemId} from '../../shared/build-systems.js';
import type {CompilationResult} from '../../types/compilation/compilation.interfaces.js';
import type {BaseCompiler} from '../base-compiler.js';
import type {ParsedRequest} from '../handlers/compile.js';
import * as utils from '../utils.js';
import type {BuildContext, BuildPlan, BuildSystemDriver, ExecutionPlan} from './build-system.interfaces.js';

/**
 * What every build system does the same way, so that a driver is left saying only what makes its build system
 * different: what to run, and why a compiler might not be able to run it. Override the rest only where a build system
 * genuinely disagrees -- CMake building in a subdirectory, Maven having no artifact to disassemble.
 */
export abstract class BaseBuildSystem implements BuildSystemDriver {
    abstract readonly id: BuildSystemId;
    abstract readonly descriptor: BuildSystemDescriptor;

    abstract getUnsupportedReason(compiler: BaseCompiler): Promise<string | undefined>;

    abstract getBuildPlan(ctx: BuildContext): Promise<BuildPlan>;

    applyRequestDefaults(compiler: BaseCompiler, parsedRequest: ParsedRequest): void {
        _.defaults(parsedRequest.filters, compiler.getDefaultFilters());
        // A project build produces an artifact to disassemble, whatever the language calls one, and the filenames in
        // its diagnostics are the user's own.
        parsedRequest.filters.binary = true;
        parsedRequest.filters.dontMaskFilenames = true;
    }

    getBuildPath(dirPath: string): string {
        // Most build systems run in the project root and put their output wherever they see fit under it.
        return dirPath;
    }

    async writeProjectFiles(ctx: BuildContext): Promise<{inputFilename: string}> {
        return await ctx.compiler.writeProjectFiles(
            ctx.dirPath,
            this.descriptor.manifestFilename,
            ctx.key.source,
            ctx.files,
        );
    }

    async prepareBuildDirectory(_ctx: BuildContext): Promise<void> {
        // Most build systems create what they need themselves.
    }

    getArtifactFilename(ctx: BuildContext): string {
        const customName = ctx.key.backendOptions?.customOutputFilename;
        // customName is user input, and finaliseArtifact copies to what this returns.
        return utils.resolveWithinDir(ctx.dirPath, customName || this.descriptor.defaultArtifactName);
    }

    async finaliseArtifact(
        _ctx: BuildContext,
        _result: CompilationResult,
        _artifactFilename: string,
    ): Promise<string | undefined> {
        // Nothing to reconcile: the build put its artifact at the path it was told to.
        return undefined;
    }

    async prepareExecution(_ctx: BuildContext, artifactFilename: string): Promise<ExecutionPlan> {
        // The build succeeded and yet the artifact is not there, which means it was built under another name: the
        // build system chose one, and the only one we know to look for is the one we were told to expect.
        if (!(await utils.fileExists(artifactFilename))) {
            const wanted = path.basename(artifactFilename);
            return {
                cannotRun:
                    `Nothing to run: the build produced no ${wanted}. Either build something by that name, or put ` +
                    'the name it does build in the output file box.',
            };
        }
        return {executable: artifactFilename};
    }

    async postProcessArtifact(
        ctx: BuildContext,
        result: CompilationResult,
        artifactFilename: string,
    ): Promise<CompilationResult> {
        const [asmResult] = await ctx.compiler.checkOutputFileAndDoPostProcess(
            result,
            artifactFilename,
            ctx.key.filters,
        );
        return asmResult;
    }
}
