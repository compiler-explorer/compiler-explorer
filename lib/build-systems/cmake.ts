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

import fs from 'node:fs/promises';
import path from 'node:path';

import _ from 'underscore';

import {BuildSystems} from '../../shared/build-systems.js';
import type {CompilationResult} from '../../types/compilation/compilation.interfaces.js';
import {assert} from '../assert.js';
import type {BaseCompiler} from '../base-compiler.js';
import type {ParsedRequest} from '../handlers/compile.js';
import type {BuildContext, BuildPlan, BuildSystemDriver} from './build-system.interfaces.js';

export class CMakeBuildSystem implements BuildSystemDriver {
    readonly id = 'cmake' as const;
    readonly descriptor = BuildSystems.cmake;

    async getUnsupportedReason(compiler: BaseCompiler): Promise<string | undefined> {
        if (!compiler.compiler.supportsBinary) return 'Compiler does not support compiling to binaries';
        return undefined;
    }

    applyRequestDefaults(compiler: BaseCompiler, parsedRequest: ParsedRequest): void {
        _.defaults(parsedRequest.filters, compiler.getDefaultFilters());
        parsedRequest.filters.binary = true;
        parsedRequest.filters.dontMaskFilenames = true;
    }

    getBuildPath(dirPath: string): string {
        return path.join(dirPath, 'build');
    }

    async writeProjectFiles(ctx: BuildContext): Promise<{inputFilename: string}> {
        return await ctx.compiler.writeProjectFiles(
            ctx.dirPath,
            this.descriptor.manifestFilename,
            ctx.key.source,
            ctx.files,
        );
    }

    async prepareBuildDirectory(ctx: BuildContext): Promise<void> {
        await fs.mkdir(ctx.buildPath);
    }

    async getBuildPlan(ctx: BuildContext): Promise<BuildPlan> {
        const compiler = ctx.compiler;

        const execParams = compiler.getDefaultExecOptions();
        execParams.appHome = ctx.dirPath;
        execParams.customCwd = ctx.buildPath;

        // Note this shares its `env` with execParams, so both steps see the compiler flags it adds.
        const makeExecParams = compiler.createCmakeExecParams(
            execParams,
            ctx.dirPath,
            ctx.libsAndOptions,
            ctx.toolchainPath,
        );

        const toolchainparam = compiler.getCMakeExtToolchainParam(ctx.parsedRequest.backendOptions.overrides || []);

        const partArgs: string[] = [
            toolchainparam,
            ...compiler.getExtraCMakeArgs(ctx.parsedRequest),
            ...ctx.buildSystemArgs,
            '..',
        ].filter(Boolean);
        const useNinja = ctx.env.ceProps('useninja');
        const fullArgs: string[] = useNinja ? ['-GNinja'].concat(partArgs) : partArgs;

        const cmd = ctx.env.ceProps('cmake') as string;
        assert(cmd, 'No cmake command found');

        return {
            getCompilationOptions: () => compiler.getUsedEnvironmentVariableFlags(makeExecParams),
            steps: [
                {
                    name: 'cmake',
                    exe: cmd,
                    args: fullArgs,
                    execParams: makeExecParams,
                    failureMessage: '<CMake configure step failed>',
                    reportsCompilationOptions: true,
                },
                {
                    name: 'build',
                    exe: cmd,
                    args: ['--build', '.'],
                    execParams: execParams,
                    failureMessage: '<CMake build step failed>',
                },
            ],
        };
    }

    getArtifactFilename(ctx: BuildContext): string {
        return ctx.compiler.getExecutableFilename(ctx.buildPath, ctx.compiler.outputFilebase, ctx.key);
    }

    async finaliseArtifact(): Promise<void> {
        // CMake builds to the path we asked for, so there is nothing to reconcile.
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
