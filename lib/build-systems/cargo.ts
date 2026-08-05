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
import type {BaseCompiler} from '../base-compiler.js';
import type {ParsedRequest} from '../handlers/compile.js';
import * as utils from '../utils.js';
import type {BuildContext, BuildPlan, BuildSystemDriver} from './build-system.interfaces.js';

/** The name the built binary is copied to, so the rest of the compilation has a path it knew in advance. */
const DEFAULT_ARTIFACT_NAME = 'output';

/** One `compiler-artifact` record of `cargo --message-format=json...`. Only the parts we use. */
type CargoArtifact = {
    reason: string;
    executable?: string | null;
    target?: {name?: string};
};

export class CargoBuildSystem implements BuildSystemDriver {
    readonly id = 'cargo' as const;
    readonly descriptor = BuildSystems.cargo;

    /**
     * Cargo is part of a Rust toolchain, not a standalone tool like CMake, so it has to be the one belonging to the
     * selected compiler: a 1.80 cargo driving a 1.91 rustc disagree about lockfile and edition features.
     */
    static getCargoPath(compiler: BaseCompiler): string {
        return path.join(path.dirname(compiler.compiler.exe), 'cargo');
    }

    async getUnsupportedReason(compiler: BaseCompiler): Promise<string | undefined> {
        if (!compiler.compiler.exe) return 'Compiler has no executable to find a cargo alongside';
        // Plenty of Rust compilers are not rustc — gccrs, mrustc — and ship no cargo.
        if (!(await utils.fileExists(CargoBuildSystem.getCargoPath(compiler)))) {
            return 'This compiler does not come with cargo';
        }
        return undefined;
    }

    applyRequestDefaults(compiler: BaseCompiler, parsedRequest: ParsedRequest): void {
        _.defaults(parsedRequest.filters, compiler.getDefaultFilters());
        parsedRequest.filters.binary = true;
        parsedRequest.filters.dontMaskFilenames = true;
    }

    getBuildPath(dirPath: string): string {
        // cargo runs in the project root and puts its output under target/ itself.
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
        // cargo creates target/ itself.
    }

    async getBuildPlan(ctx: BuildContext): Promise<BuildPlan> {
        const compiler = ctx.compiler;

        const execParams = compiler.getDefaultExecOptions();
        execParams.appHome = ctx.dirPath;
        execParams.customCwd = ctx.dirPath;

        execParams.env.RUSTC = compiler.compiler.exe;
        // Keep cargo's registry and caches inside the sandbox: build nodes have no network, and nothing it writes
        // should outlive the compilation.
        execParams.env.CARGO_HOME = path.join(ctx.dirPath, '.cargo-home');
        const rustFlags = ctx.parsedRequest.options.join(' ').trim();
        if (rustFlags) execParams.env.RUSTFLAGS = rustFlags;

        return {
            getCompilationOptions: () => ctx.parsedRequest.options,
            steps: [
                {
                    name: 'cargo',
                    exe: CargoBuildSystem.getCargoPath(compiler),
                    args: [
                        'build',
                        // No network on a build node, so fail fast and legibly rather than hang resolving crates.
                        '--offline',
                        // Diagnostics render to stderr for the user; stdout is left as records we can parse.
                        '--message-format=json-render-diagnostics',
                        ...ctx.buildSystemArgs,
                    ],
                    execParams: execParams,
                    failureMessage: '<Cargo build failed>',
                    reportsCompilationOptions: true,
                },
            ],
        };
    }

    getArtifactFilename(ctx: BuildContext): string {
        const customName = ctx.key.backendOptions?.customOutputFilename;
        return path.join(ctx.dirPath, customName || DEFAULT_ARTIFACT_NAME);
    }

    /**
     * cargo reports where it put things, but under whichever spelling of the project root it saw: the sandbox
     * bind-mounts it as /app, and without a sandbox it is the real temp directory. maskRootdir reduces both to a path
     * relative to that root; anything it leaves absolute is not ours to rebase.
     */
    static toHostPath(reportedPath: string, dirPath: string): string {
        const relative = utils.maskRootdir(reportedPath);
        return path.isAbsolute(relative) ? reportedPath : path.join(dirPath, relative);
    }

    /** The `compiler-artifact` records cargo wrote to stdout, in order. */
    private static parseArtifacts(stdout: {text: string}[]): CargoArtifact[] {
        const artifacts: CargoArtifact[] = [];
        for (const line of stdout) {
            const text = line.text.trim();
            if (!text.startsWith('{')) continue;
            try {
                const record = JSON.parse(text) as CargoArtifact;
                if (record.reason === 'compiler-artifact') artifacts.push(record);
            } catch {
                // Not a record we understand; the user sees cargo's own output regardless.
            }
        }
        return artifacts;
    }

    /**
     * cargo names its output after the manifest, so copy what it built to the path the rest of the compilation was
     * told to expect. `customOutputFilename` picks between artifacts when a project builds more than one.
     */
    async finaliseArtifact(
        ctx: BuildContext,
        result: CompilationResult,
        artifactFilename: string,
        copyFile: (from: string, to: string) => Promise<void> = fs.copyFile,
    ): Promise<void> {
        const cargoStep = result.buildsteps?.find(step => step.step === 'cargo');
        const artifacts = CargoBuildSystem.parseArtifacts(cargoStep?.stdout ?? []);

        // The records are machine-readable noise; the user reads cargo's stderr instead.
        if (cargoStep) cargoStep.stdout = [];

        const executables = artifacts
            .filter(artifact => artifact.executable)
            .map(artifact => ({
                ...artifact,
                executable: CargoBuildSystem.toHostPath(artifact.executable!, ctx.dirPath),
            }));
        if (executables.length === 0) {
            throw new Error('Cargo built no executable. Does the project have a [[bin]] target?');
        }

        const wanted = path.basename(artifactFilename);
        const chosen =
            executables.find(artifact => path.basename(artifact.executable as string) === wanted) ??
            executables[executables.length - 1];

        if (chosen.executable !== artifactFilename) {
            await copyFile(chosen.executable as string, artifactFilename);
        }
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
