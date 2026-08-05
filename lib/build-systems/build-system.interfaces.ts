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

import type {
    CmakeCacheKey,
    CompilationResult,
    ExecutionOptionsWithEnv,
    FiledataPair,
    LibsAndOptions,
} from '../../types/compilation/compilation.interfaces.js';
import type {BaseCompiler} from '../base-compiler.js';
import type {CompilationEnvironment} from '../compilation-env.js';
import type {ParsedRequest} from '../handlers/compile.js';

/**
 * Identifies a build system. This value also ends up in the compilation cache key (as `api`), so existing ids must
 * never change or every cached build for them is invalidated.
 */
export type BuildSystemId = 'cmake';

/** A single command in a build system's plan, for example CMake's configure step. */
export type BuildSystemStep = {
    /** Shown to the user as the name of this build step */
    name: string;
    exe: string;
    args: string[];
    execParams: ExecutionOptionsWithEnv;
    /** Placeholder shown in the assembly pane if this step fails */
    failureMessage: string;
    /** Whether a failure of this step reports the effective compilation options back to the user */
    reportsCompilationOptions?: boolean;
};

export type BuildPlan = {
    steps: BuildSystemStep[];
    /**
     * The flags the build system ends up handing the compiler, shown to the user. Resolved lazily because a build
     * system may only know them once its steps have run.
     */
    getCompilationOptions: () => string[];
};

/** Everything a driver needs to know about the build it has been asked to run. */
export type BuildContext = {
    compiler: BaseCompiler;
    env: CompilationEnvironment;
    /** The project root: where the manifest and all the user's files are written */
    dirPath: string;
    /** Where the build steps run, and where the artifact is looked for */
    buildPath: string;
    key: CmakeCacheKey;
    parsedRequest: ParsedRequest;
    files: FiledataPair[];
    libsAndOptions: LibsAndOptions;
    toolchainPath: string | undefined;
};

/**
 * A build system, as far as {@link BaseCompiler.buildProject} is concerned. Everything that is *not* specific to a
 * build system — temp directories, caching, the compilation queue, execution, post-compilation tools — stays in the
 * orchestrator; this interface covers only the parts that differ between CMake, Cargo, Maven and friends.
 */
export interface BuildSystemDriver {
    readonly id: BuildSystemId;

    /** The file the tree's main source is written to, e.g. `CMakeLists.txt` */
    readonly manifestFilename: string;

    /** Why this compiler cannot be built with this build system, or undefined if it can. */
    getUnsupportedReason(compiler: BaseCompiler): string | undefined;

    /** Coerce the request into the shape this build system needs. Runs before the cache key is computed. */
    applyRequestDefaults(compiler: BaseCompiler, parsedRequest: ParsedRequest): void;

    /** Where builds happen, given the project root. */
    getBuildPath(dirPath: string): string;

    /** Write the manifest and the rest of the project into the project root. */
    writeProjectFiles(ctx: BuildContext): Promise<{inputFilename: string}>;

    /** Create whatever the build steps expect to already exist. Runs after the files are written. */
    prepareBuildDirectory(ctx: BuildContext): Promise<void>;

    /** The commands to run, in order. */
    getBuildPlan(ctx: BuildContext): Promise<BuildPlan>;

    /** The artifact to post-process and, if asked, execute. */
    getArtifactFilename(ctx: BuildContext): string;

    /** Turn the built artifact into something to show the user, usually by disassembling it. */
    postProcessArtifact(
        ctx: BuildContext,
        result: CompilationResult,
        artifactFilename: string,
    ): Promise<CompilationResult>;
}
