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
import type {ResultLine} from '../../types/resultline/resultline.interfaces.js';
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

/** Parse one line of cargo's stdout as a record, or undefined if it is ordinary output. */
function CargoArtifactRecord(text: string): CargoArtifact | undefined {
    const trimmed = text.trim();
    if (!trimmed.startsWith('{')) return undefined;
    try {
        const record = JSON.parse(trimmed) as CargoArtifact;
        return record.reason ? record : undefined;
    } catch {
        return undefined;
    }
}

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

        // Libraries the user picked are already unpacked into the project as prebuilt rlibs, so hand them to rustc
        // directly. cargo could not resolve them itself: it wants sources from a registry, and there is no network.
        const libIncludes = compiler.getIncludeArguments(ctx.libsAndOptions.libraries, ctx.dirPath);
        const rustcArgs = [...libIncludes, ...ctx.libsAndOptions.options];
        // The encoded form is separator-delimited, so it survives paths and flags containing spaces.
        if (rustcArgs.length > 0) execParams.env.CARGO_ENCODED_RUSTFLAGS = rustcArgs.join('\x1f');

        return {
            getCompilationOptions: () => ctx.libsAndOptions.options,
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
                    explainFailure: CargoBuildSystem.explainFailure,
                    reportsCompilationOptions: true,
                },
            ],
        };
    }

    /**
     * Compiler Explorer cannot let cargo fetch crates, so a [dependencies] entry always fails to resolve. The
     * libraries it does have are prebuilt and selected in the Libraries pane, which is not something cargo's own
     * message could know to suggest.
     */
    static explainFailure(stderr: string): string | undefined {
        // A slash means a dependency's feature, e.g. --features rand/small_rng, which cannot apply to a crate that is
        // not a dependency. Without the slash it is a feature of their own package, and cargo's message is the truth.
        if (/does not contain this feature: \S+\//.test(stderr)) {
            return (
                'Features of a library cannot be chosen here: its rlib is prebuilt, with every feature enabled ' +
                'where that built cleanly and none at all where it did not. --features applies only to the ' +
                'features your own Cargo.toml declares.'
            );
        }
        if (/no matching package|failed to (?:select a version|get|resolve)|offline mode/.test(stderr)) {
            return (
                'Crates cannot be downloaded here, so [dependencies] cannot be resolved. ' +
                'Pick the library in the Libraries pane instead and leave it out of Cargo.toml — ' +
                'it is passed to rustc directly, so `use <crate>::...` works without declaring it.'
            );
        }
        return undefined;
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

    /**
     * Split cargo's stdout into the machine-readable records we asked for and everything else. Anything else is
     * genuine output the user wants — `--help` prints there, for one — so it must survive.
     */
    private static partitionStdout(stdout: ResultLine[]): {artifacts: CargoArtifact[]; output: ResultLine[]} {
        const artifacts: CargoArtifact[] = [];
        const output: ResultLine[] = [];
        for (const line of stdout) {
            const record = CargoArtifactRecord(line.text);
            if (!record) {
                output.push(line);
            } else if (record.reason === 'compiler-artifact') {
                artifacts.push(record);
            }
        }
        return {artifacts, output};
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
    ): Promise<string | undefined> {
        const cargoStep = result.buildsteps?.find(step => step.step === 'cargo');
        const {artifacts, output} = CargoBuildSystem.partitionStdout(cargoStep?.stdout ?? []);

        // Drop the records we asked for, which mean nothing to the user, but keep the rest of what cargo said.
        if (cargoStep) cargoStep.stdout = output;

        const executables = artifacts
            .filter(artifact => artifact.executable)
            .map(artifact => ({
                ...artifact,
                executable: CargoBuildSystem.toHostPath(artifact.executable!, ctx.dirPath),
            }));
        if (executables.length === 0) {
            // Not an error as such: cargo can succeed without building a binary, for a library-only crate or when an
            // argument like --help means it never builds at all.
            return 'Cargo did not build an executable, so there is nothing to show here';
        }

        const wanted = path.basename(artifactFilename);
        const chosen =
            executables.find(artifact => path.basename(artifact.executable as string) === wanted) ??
            executables[executables.length - 1];

        if (chosen.executable !== artifactFilename) {
            await copyFile(chosen.executable as string, artifactFilename);
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
