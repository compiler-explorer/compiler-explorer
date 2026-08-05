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
import type {ExecutableExecutionOptions} from '../../types/execution/execution.interfaces.js';
import type {BaseCompiler} from '../base-compiler.js';
import type {ParsedRequest} from '../handlers/compile.js';
import * as utils from '../utils.js';
import type {BuildContext, BuildPlan, BuildSystemDriver} from './build-system.interfaces.js';

/** Where maven puts compiled classes, relative to the project. */
const CLASSES_DIR = path.join('target', 'classes');

/** The name the built jar is copied to, so the rest of the compilation has a path it knew in advance. */
const DEFAULT_ARTIFACT_NAME = 'output.jar';

export class MavenBuildSystem implements BuildSystemDriver {
    readonly id = 'maven' as const;
    readonly descriptor = BuildSystems.maven;

    /** The JDK the selected compiler belongs to: its exe is `<jdk>/bin/javac`. */
    static getJavaHome(compiler: BaseCompiler): string {
        return path.resolve(path.dirname(compiler.compiler.exe), '..');
    }

    async getUnsupportedReason(compiler: BaseCompiler): Promise<string | undefined> {
        if (!compiler.compiler.exe) return 'Compiler has no executable to find a JDK from';
        if (!(await utils.fileExists(path.join(MavenBuildSystem.getJavaHome(compiler), 'bin', 'java')))) {
            return 'This compiler is not part of a JDK, so maven cannot run against it';
        }
        return undefined;
    }

    applyRequestDefaults(compiler: BaseCompiler, parsedRequest: ParsedRequest): void {
        _.defaults(parsedRequest.filters, compiler.getDefaultFilters());
        // Java reads this as "run javap", which is how bytecode gets shown; there is no native binary involved.
        parsedRequest.filters.binary = true;
        parsedRequest.filters.dontMaskFilenames = true;
    }

    getBuildPath(dirPath: string): string {
        // maven runs in the project root and makes target/ itself.
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
        // maven creates target/ itself.
    }

    async getBuildPlan(ctx: BuildContext): Promise<BuildPlan> {
        const compiler = ctx.compiler;

        const execParams = compiler.getDefaultExecOptions();
        execParams.appHome = ctx.dirPath;
        execParams.customCwd = ctx.dirPath;
        execParams.env.JAVA_HOME = MavenBuildSystem.getJavaHome(compiler);

        const mvn = ctx.env.ceProps('maven') as string;
        // Everything maven needs was fetched into this repository when it was installed; see infra's tools.yaml.
        const repository = path.join(path.dirname(path.dirname(mvn)), 'repository');

        return {
            getCompilationOptions: () => ctx.libsAndOptions.options,
            steps: [
                {
                    name: 'maven',
                    exe: mvn,
                    args: [
                        // Offline because build nodes have no network, and batch mode so there is no progress spam.
                        '-o',
                        '-B',
                        // The sandbox mounts /tmp noexec, where jansi would otherwise unpack a native library.
                        '-Djansi.passthrough=true',
                        `-Dmaven.repo.local=${repository}`,
                        ...ctx.buildSystemArgs,
                        // Only if the user has not asked for particular goals themselves.
                        ...(ctx.buildSystemArgs.some(arg => !arg.startsWith('-')) ? [] : ['package']),
                    ],
                    execParams: execParams,
                    failureMessage: '<Maven build failed>',
                    explainFailure: MavenBuildSystem.explainFailure,
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
     * Maven's offline failures name an artifact rather than the reason it is missing, which in Compiler Explorer is
     * always the same reason.
     */
    static explainFailure(stderr: string): string | undefined {
        if (/Cannot access .* in offline mode|could not be resolved|has not been downloaded/.test(stderr)) {
            return (
                'Compiler Explorer cannot download from Maven Central, so only the plugins bundled with maven here ' +
                'are available, and a <dependencies> entry cannot be resolved at all.'
            );
        }
        return undefined;
    }

    /**
     * maven names the jar after the pom, so copy it to the path the rest of the compilation was told to expect, the
     * same way the Cargo driver does. `customOutputFilename` picks between jars when a build produces several.
     */
    async finaliseArtifact(
        ctx: BuildContext,
        result: CompilationResult,
        artifactFilename: string,
        copyFile: (from: string, to: string) => Promise<void> = fs.copyFile,
    ): Promise<string | undefined> {
        if (!(await utils.dirExists(path.join(ctx.dirPath, CLASSES_DIR)))) {
            return 'Maven compiled no classes, so there is nothing to show here';
        }

        const targetDir = path.join(ctx.dirPath, 'target');
        const jars = (await fs.readdir(targetDir).catch(() => [])).filter(name => name.endsWith('.jar'));
        if (jars.length === 0) {
            // The classes are still there to disassemble, so this is only a problem if they wanted to run it.
            return undefined;
        }

        const wanted = path.basename(artifactFilename);
        const chosen = jars.find(name => name === wanted) ?? jars[jars.length - 1];
        await copyFile(path.join(targetDir, chosen), artifactFilename);
        return undefined;
    }

    /** A jar is not executable on its own: start a JVM on the classes maven produced. */
    async prepareExecution(
        ctx: BuildContext,
        artifactFilename: string,
        executeOptions: ExecutableExecutionOptions,
    ): Promise<string | undefined> {
        const classesDir = path.join(ctx.dirPath, CLASSES_DIR);
        const mainClass = await MavenBuildSystem.findMainClass(classesDir);
        if (!mainClass) return undefined;

        executeOptions.args = [
            // The sandbox is tight on threads and memory, and a JVM left to itself asks for far more than it gets.
            // These are the same concessions the Java compiler makes when it runs a class directly.
            '-Xss136K',
            '-XX:CICompilerCount=2',
            '-XX:-UseDynamicNumberOfCompilerThreads',
            '-XX:-UseDynamicNumberOfGCThreads',
            '-XX:+UseSerialGC',
            '-cp',
            classesDir,
            mainClass,
            ...executeOptions.args,
        ];
        return path.join(MavenBuildSystem.getJavaHome(ctx.compiler), 'bin', 'java');
    }

    /** The first class with a `main`, named the way the JVM wants it: package-qualified, dots not slashes. */
    private static async findMainClass(classesDir: string): Promise<string | undefined> {
        const entries = await fs.readdir(classesDir, {recursive: true, withFileTypes: true});
        for (const entry of entries) {
            if (!entry.isFile() || !entry.name.endsWith('.class')) continue;
            const full = path.join(entry.parentPath ?? classesDir, entry.name);
            const contents = await fs.readFile(full);
            // Every class carrying a main has the name and descriptor in its constant pool.
            if (!contents.includes('([Ljava/lang/String;)V') || !contents.includes('main')) continue;
            return path
                .relative(classesDir, full)
                .replace(/\.class$/, '')
                .replaceAll(path.sep, '.');
        }
        return undefined;
    }

    async postProcessArtifact(ctx: BuildContext, result: CompilationResult): Promise<CompilationResult> {
        // Reuse the Java compiler's javap handling by pointing it at maven's class output.
        const classesDir = path.join(ctx.dirPath, CLASSES_DIR);
        const [asmResult] = await ctx.compiler.checkOutputFileAndDoPostProcess(
            result,
            path.join(classesDir, 'unused.class'),
            ctx.key.filters,
        );
        return asmResult;
    }
}
