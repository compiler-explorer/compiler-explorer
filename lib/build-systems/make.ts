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

import {BuildSystems} from '../../shared/build-systems.js';
import type {CompilationResult} from '../../types/compilation/compilation.interfaces.js';
import {assert} from '../assert.js';
import type {BaseCompiler} from '../base-compiler.js';
import type {ParsedRequest} from '../handlers/compile.js';
import * as utils from '../utils.js';
import type {BuildContext, BuildPlan, BuildSystemDriver} from './build-system.interfaces.js';

/** What the artifact is called when the project has not said, which a Makefile is free to disagree with. */
const DEFAULT_ARTIFACT_NAME = 'output';

export class MakeBuildSystem implements BuildSystemDriver {
    readonly id = 'make' as const;
    readonly descriptor = BuildSystems.make;

    async getUnsupportedReason(_compiler: BaseCompiler): Promise<string | undefined> {
        // A Makefile decides for itself what to run, so there is nothing about a compiler that rules it out. What it
        // is given -- CXX, CXXFLAGS and the rest -- is whatever the selected compiler would have been given anyway.
        return undefined;
    }

    applyRequestDefaults(compiler: BaseCompiler, parsedRequest: ParsedRequest): void {
        _.defaults(parsedRequest.filters, compiler.getDefaultFilters());
        parsedRequest.filters.binary = true;
        parsedRequest.filters.dontMaskFilenames = true;
    }

    getBuildPath(dirPath: string): string {
        // make builds where the Makefile is, and puts things wherever that says.
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

    async prepareBuildDirectory(): Promise<void> {
        // Whatever directories the build wants are the Makefile's business.
    }

    async getBuildPlan(ctx: BuildContext): Promise<BuildPlan> {
        const compiler = ctx.compiler;

        const execParams = compiler.getDefaultExecOptions();
        execParams.appHome = ctx.dirPath;
        execParams.customCwd = ctx.dirPath;
        compiler.applyOverridesToExecOptions(execParams, ctx.parsedRequest.backendOptions.overrides || []);

        // The same environment CMake is handed, and for the same reason: a Makefile that says `$(CXX) $(CXXFLAGS)`
        // should get the compiler that was selected here, with the options and libraries that were chosen with it.
        const makeExecParams = compiler.createCmakeExecParams(
            execParams,
            ctx.dirPath,
            ctx.libsAndOptions,
            ctx.toolchainPath,
        );

        // A CUDA Makefile conventionally reaches for $(NVCC), which make has no built-in for and CMake has no need
        // of, so it would otherwise expand to nothing and the recipe would run without a compiler at all. Only when
        // this really is nvcc: clang and others compile CUDA too, and naming one of those NVCC would be a lie the
        // Makefile has no way to see through.
        if (compiler.compiler.compilerType === 'nvcc') makeExecParams.env.NVCC = compiler.compiler.exe;

        const make = ctx.env.ceProps('make') as string;
        assert(make, 'No make command found');

        return {
            getCompilationOptions: () => compiler.getUsedEnvironmentVariableFlags(makeExecParams),
            steps: [
                {
                    name: 'make',
                    exe: make,
                    // Nothing of ours: a bare `make` is the default target, and anything else the user asks for --
                    // a target, -j, a variable override -- is theirs to say.
                    args: ctx.buildSystemArgs,
                    execParams: makeExecParams,
                    failureMessage: '<Make failed>',
                    reportsCompilationOptions: true,
                },
            ],
        };
    }

    getArtifactFilename(ctx: BuildContext): string {
        const customName = ctx.key.backendOptions?.customOutputFilename;
        return path.join(ctx.dirPath, customName || DEFAULT_ARTIFACT_NAME);
    }

    async finaliseArtifact(
        ctx: BuildContext,
        result: CompilationResult,
        artifactFilename: string,
    ): Promise<string | undefined> {
        // Only the Makefile knows what it built, so say plainly when nothing is where we were told to look rather
        // than leaving the disassembler to fail on a file that is not there.
        if (!(await utils.fileExists(artifactFilename))) {
            return (
                `Nothing was built at ${path.basename(artifactFilename)}. A Makefile names its own output, so ` +
                'either build something by that name or put the name it does build in the output file box.'
            );
        }
        return undefined;
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
