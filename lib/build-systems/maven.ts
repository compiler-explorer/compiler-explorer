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
import {maybeRemapJailedDir} from '../exec.js';
import type {ParsedRequest} from '../handlers/compile.js';
import * as utils from '../utils.js';
import type {BuildContext, BuildPlan, BuildSystemDriver} from './build-system.interfaces.js';

/** Where jansi is told to unpack its native library, relative to the project. */
const JANSI_DIR = '.jansi';

/** Where maven puts compiled classes, relative to the project. */
const CLASSES_DIR = path.join('target', 'classes');

/** Where maven-dependency-plugin gathers the jars needed to run, relative to the project. */
const DEPENDENCY_DIR = path.join('target', 'dependency');

/** A repository of this build's own, layered in front of the shared one, relative to the project. */
const HEAD_REPOSITORY_DIR = '.m2';

/** Where the Kotlin artifacts live in a repository, and in a Kotlin installation. */
const KOTLIN_GROUP_PATH = path.join('org', 'jetbrains', 'kotlin');

/** What a Kotlin's repository says its installation has to answer for, one artifact a line. Written by infra. */
const SUPPLIED_RECORD = '.ce-supplied-by-installation';

/** The name the built jar is copied to, so the rest of the compilation has a path it knew in advance. */
const DEFAULT_ARTIFACT_NAME = 'output.jar';

/**
 * What the Java-family compilers know about the JDK they belong to, neither of which BaseCompiler has any concept of:
 * `javaHome` is the JDK a Kotlin compiler targets, `javaRuntime` the `java` that runs a Java compiler's output.
 */
type JavaLikeCompiler = BaseCompiler & {javaHome?: string; javaRuntime?: string};

export class MavenBuildSystem implements BuildSystemDriver {
    readonly id = 'maven' as const;
    readonly descriptor = BuildSystems.maven;

    /**
     * The JDK maven should run against. A Kotlin compiler is not part of one at all -- it lives in its own
     * kotlin-jvm-x.y.z -- but it says which JDK it targets, and a Java compiler says which one runs its output, so
     * ask the compiler before reading a JDK off its own path.
     */
    static getJavaHome(compiler: BaseCompiler): string {
        const javaLike = compiler as JavaLikeCompiler;
        if (javaLike.javaHome) return javaLike.javaHome;
        if (javaLike.javaRuntime) return path.resolve(path.dirname(javaLike.javaRuntime), '..');
        return MavenBuildSystem.getCompilerRoot(compiler);
    }

    /** Where a compiler is installed, its exe being in the bin of it. */
    private static getCompilerRoot(compiler: BaseCompiler): string {
        return path.resolve(path.dirname(compiler.compiler.exe), '..');
    }

    /**
     * The repository of Maven artifacts belonging to a Kotlin: what its own installation does not carry, installed
     * beside it as `<installation>-maven` so that a new Kotlin brings its Maven support with it. See infra's
     * `tools/kotlin-maven`.
     */
    private static getKotlinRepository(compiler: BaseCompiler): string {
        return `${MavenBuildSystem.getCompilerRoot(compiler)}-maven`;
    }

    /** The Kotlins that can be built with maven here: those whose Maven artifacts are installed beside them. */
    private static async getMavenCapableKotlinVersions(compiler: BaseCompiler): Promise<string[]> {
        const root = MavenBuildSystem.getCompilerRoot(compiler);
        const siblings = await fs.readdir(path.dirname(root)).catch(() => []);
        return siblings
            .map(name => /^kotlin-jvm-(.+)-maven$/.exec(name)?.[1])
            .filter((version): version is string => version !== undefined)
            .sort((a, b) => a.localeCompare(b, undefined, {numeric: true}));
    }

    async getUnsupportedReason(compiler: BaseCompiler): Promise<string | undefined> {
        if (!compiler.compiler.exe) return 'Compiler has no executable to find a JDK from';
        if (!(await utils.fileExists(path.join(MavenBuildSystem.getJavaHome(compiler), 'bin', 'java')))) {
            return 'No JDK could be found for this compiler, so maven cannot run against it';
        }
        // Kotlin is built by a plugin that has to be fetched for exactly that version, which is done when the
        // compiler is installed. Saying so here is better than letting maven fail on a resolution it cannot explain.
        if (
            compiler.lang?.id === 'kotlin' &&
            !(await utils.dirExists(MavenBuildSystem.getKotlinRepository(compiler)))
        ) {
            const available = await MavenBuildSystem.getMavenCapableKotlinVersions(compiler);
            const alternatives = available.length > 0 ? ` Maven is available for ${available.join(', ')}.` : '';
            return `The Maven artifacts for this Kotlin are not installed on this machine.${alternatives}`;
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

    async prepareBuildDirectory(ctx: BuildContext): Promise<void> {
        // maven creates target/ itself, but jansi will not create the directory it is pointed at, and a repository
        // maven is told to write to has to exist before it will read the ones behind it.
        await fs.mkdir(path.join(ctx.dirPath, JANSI_DIR), {recursive: true});
        await fs.mkdir(path.join(ctx.dirPath, HEAD_REPOSITORY_DIR), {recursive: true});
    }

    /** Where maven is, and the repository holding everything it was given when it was installed. */
    private static getMavenPaths(ctx: BuildContext): {mvn: string; repository: string} {
        const mvn = ctx.env.ceProps('maven') as string;
        // Everything maven needs was fetched into this repository when it was installed; see infra's tools.yaml.
        return {mvn, repository: path.join(path.dirname(path.dirname(mvn)), 'repository')};
    }

    /** How to run maven at all: which JDK, and where it may write. Shared by the build and by running the result. */
    private static getExecParams(ctx: BuildContext) {
        const compiler = ctx.compiler;

        const execParams = compiler.getDefaultExecOptions();
        execParams.appHome = ctx.dirPath;
        execParams.customCwd = ctx.dirPath;
        compiler.applyOverridesToExecOptions(execParams, ctx.parsedRequest.backendOptions.overrides || []);
        execParams.env.JAVA_HOME = MavenBuildSystem.getJavaHome(compiler);
        // jansi unpacks a native library into java.io.tmpdir and maps it, and the sandbox mounts /tmp noexec. Point
        // it at the project instead, which is executable because built binaries run from there. This has to reach the
        // JVM at launch: jansi initialises before maven turns its own -D arguments into system properties.
        // Spelled as the build will see it: /app under a sandbox, the real path without one. Ours goes first so
        // that anything the user set through an environment override wins, the JVM taking the last of a duplicate.
        const jansiTmpdir = `-Djansi.tmpdir=${path.join(maybeRemapJailedDir(ctx.dirPath), JANSI_DIR)}`;
        execParams.env.MAVEN_OPTS = [jansiTmpdir, execParams.env.MAVEN_OPTS].filter(Boolean).join(' ');

        return execParams;
    }

    /** Where a Kotlin installation keeps its jars, which are also the artifacts Maven knows it by. */
    private static getKotlinLib(compiler: BaseCompiler): string {
        return path.join(MavenBuildSystem.getCompilerRoot(compiler), 'lib');
    }

    /**
     * kotlin-maven-plugin compiles with a compiler it resolves from a repository rather than with anything installed
     * on the machine, which on its own would make the compiler picked here decorative. A Kotlin installation's jars
     * are the very artifacts it would resolve, byte for byte, so link them into a repository of this build's own and
     * put it in front: the plugin then loads the compiler that was actually selected. Its own repository beside it
     * holds only what the installation does not carry -- the plugin, the poms, and from 2.2 the embeddable compiler.
     *
     * Returns the artifacts this version needs from the installation but which it does not have, which is nothing at
     * all unless the compiler is not really installed on this machine.
     */
    private static async linkSelectedKotlin(
        ctx: BuildContext,
        kotlinRepository: string,
        version: string,
    ): Promise<string[]> {
        // Which artifacts to link is not ours to work out: the repository was primed with them taken out and says
        // which those were. Guessing from what has a pom would miss the ones maven itself already had a copy of.
        const record = await fs.readFile(path.join(kotlinRepository, SUPPLIED_RECORD), 'utf8').catch(() => '');
        const artifacts = record.split('\n').filter(Boolean);
        const lib = MavenBuildSystem.getKotlinLib(ctx.compiler);
        const headRepository = path.join(ctx.dirPath, HEAD_REPOSITORY_DIR);
        const missing: string[] = [];

        for (const artifact of artifacts) {
            // A distribution names most of its jars after the artifact, and a few without the leading `kotlin-`.
            const candidates = [`${artifact}.jar`, `${artifact.replace(/^kotlin-/, '')}.jar`];
            const source = await MavenBuildSystem.findFirst(candidates.map(name => path.join(lib, name)));
            if (!source) {
                missing.push(artifact);
                continue;
            }

            const target = path.join(headRepository, KOTLIN_GROUP_PATH, artifact, version);
            await fs.mkdir(target, {recursive: true});
            await fs.symlink(source, path.join(target, `${artifact}-${version}.jar`)).catch(() => {});
        }
        return missing;
    }

    /** The Kotlin that the selected compiler is, when it is one at all. */
    private static getKotlinVersion(ctx: BuildContext): string | undefined {
        return ctx.compiler.lang?.id === 'kotlin' ? ctx.compiler.compiler.semver : undefined;
    }

    /**
     * What every maven run within one build has to agree on, or the ones that do not will fail to resolve what the
     * others did. Three repositories chained, nearest first: this build's own, holding the selected compiler; the
     * one installed beside that compiler, holding what it does not carry; and the shared one maven was installed
     * with. And which Kotlin, since a pom naming its version once expects to be told what that is.
     */
    private static async getCommonArgs(ctx: BuildContext, repository: string): Promise<string[]> {
        const kotlinVersion = MavenBuildSystem.getKotlinVersion(ctx);
        const tail = [...(kotlinVersion ? [MavenBuildSystem.getKotlinRepository(ctx.compiler)] : []), repository];
        return [
            // The head is ours to write to and so has to be spelled as the build will see it: /app under a sandbox.
            // The others are read where they are installed.
            `-Dmaven.repo.local=${path.join(maybeRemapJailedDir(ctx.dirPath), HEAD_REPOSITORY_DIR)}`,
            `-Dmaven.repo.local.tail=${tail.join(',')}`,
            ...(kotlinVersion
                ? [
                      `-Dkotlin.version=${kotlinVersion}`,
                      // From 2.2 the plugin compiles in a daemon of its own by default, which is a second JVM on top
                      // of maven's: more than the sandbox allows in processes or memory, and it fails to start a
                      // thread. Nothing here is long-lived enough for a daemon to pay for itself anyway. Older
                      // plugins have no such option and ignore it.
                      '-Dkotlin.compiler.daemon=false',
                  ]
                : []),
        ];
    }

    /** The first of these paths that is there, if any. */
    private static async findFirst(paths: string[]): Promise<string | undefined> {
        for (const candidate of paths) {
            if (await utils.fileExists(candidate)) return candidate;
        }
        return undefined;
    }

    /** Which versions of a plugin maven itself was given when it was installed, oldest first. */
    private static async getBundledVersions(
        repository: string,
        groupId: string,
        artifactId: string,
    ): Promise<string[]> {
        const artifactDir = path.join(repository, ...groupId.split('.'), artifactId);
        return (await fs.readdir(artifactDir).catch(() => [])).sort();
    }

    async getBuildPlan(ctx: BuildContext): Promise<BuildPlan> {
        const execParams = MavenBuildSystem.getExecParams(ctx);
        const {mvn, repository} = MavenBuildSystem.getMavenPaths(ctx);

        // Kotlin is built by a plugin that resolves its own compiler, so the selected one has to be put where the
        // plugin will look, and the plugin told to ask for that version.
        const kotlinVersion = MavenBuildSystem.getKotlinVersion(ctx);
        const missing = kotlinVersion
            ? await MavenBuildSystem.linkSelectedKotlin(
                  ctx,
                  MavenBuildSystem.getKotlinRepository(ctx.compiler),
                  kotlinVersion,
              )
            : [];
        const commonArgs = await MavenBuildSystem.getCommonArgs(ctx, repository);

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
                        // Which repositories, and which Kotlin: a pom following the convention of naming its version
                        // once picks the latter up for both the plugin and the standard library. Before the user's
                        // own arguments, so that saying otherwise still wins.
                        ...commonArgs,
                        // Without an encoding maven warns that the build is platform dependent, three times over.
                        // Compiler Explorer writes these files as UTF-8, so saying so is both true and what makes
                        // the build reproducible -- but only when the project has not spoken for itself, because a
                        // -D would override what its pom says.
                        ...(/encoding/i.test(ctx.key.source) ? [] : ['-Dproject.build.sourceEncoding=UTF-8']),
                        ...ctx.buildSystemArgs,
                        // Only if the user has not asked for particular goals themselves.
                        ...(ctx.buildSystemArgs.some(arg => !arg.startsWith('-')) ? [] : ['package']),
                    ],
                    execParams: execParams,
                    failureMessage: '<Maven build failed>',
                    explainFailure: async output =>
                        MavenBuildSystem.explainFailure(
                            output,
                            // Only worth looking at what else is installed once something has actually gone wrong.
                            kotlinVersion ? await MavenBuildSystem.getMavenCapableKotlinVersions(ctx.compiler) : [],
                            kotlinVersion,
                            missing,
                        ),
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
    static explainFailure(
        output: string,
        mavenCapableKotlinVersions: string[] = [],
        selectedKotlinVersion?: string,
        missingFromInstallation: string[] = [],
    ): string | undefined {
        // The compiler is meant to come from the selected Kotlin's own installation, so if that is not really on this
        // machine, no amount of choosing a different version in the pom is the answer.
        if (missingFromInstallation.length > 0 && /org\.jetbrains\.kotlin/.test(output)) {
            return (
                `Kotlin ${selectedKotlinVersion} does not appear to be fully installed on this machine, so maven ` +
                `cannot be given the compiler to build with: ${missingFromInstallation.join(', ')} are missing from ` +
                'it. This is something for Compiler Explorer to fix, not your project.'
            );
        }
        // A pom naming a Kotlin of its own overrides what was selected, and only the installed ones can be built.
        if (mavenCapableKotlinVersions.length > 0 && /org\.jetbrains\.kotlin/.test(output)) {
            return (
                'Maven builds Kotlin through kotlin-maven-plugin, which is installed here for ' +
                `${mavenCapableKotlinVersions.join(', ')}. Select one of those Kotlin compilers, and leave the ` +
                'version out of the pom: the compiler selected is the one the build is told to use.'
            );
        }
        if (/Cannot access .* in offline mode|could not be resolved|has not been downloaded/.test(output)) {
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

    /**
     * Whatever the project depends on has to be on the classpath to run: Kotlin cannot get through its own standard
     * library's intrinsics without one. Maven leaves those jars in the repository, which is not part of the project
     * the execution sandbox is given, so have it copy them in. Only worth a maven run when the project asked for
     * something -- a plain Java project here can depend on nothing, there being no network to fetch it from.
     */
    private static async gatherRuntimeDependencies(ctx: BuildContext): Promise<string[]> {
        if (!/<dependency>/.test(ctx.key.source)) return [];

        const dependencyDir = path.join(ctx.dirPath, DEPENDENCY_DIR);
        if (!(await utils.dirExists(dependencyDir))) {
            const {mvn, repository} = MavenBuildSystem.getMavenPaths(ctx);
            // Named in full: resolving the `dependency:` prefix instead would need metadata from Maven Central,
            // which offline maven has no way to read.
            const versions = await MavenBuildSystem.getBundledVersions(
                repository,
                'org.apache.maven.plugins',
                'maven-dependency-plugin',
            );
            const version = versions.at(-1);
            if (!version) return [];
            await ctx.compiler.exec(
                mvn,
                [
                    '-o',
                    '-B',
                    ...(await MavenBuildSystem.getCommonArgs(ctx, repository)),
                    `org.apache.maven.plugins:maven-dependency-plugin:${version}:copy-dependencies`,
                    '-DincludeScope=runtime',
                ],
                MavenBuildSystem.getExecParams(ctx),
            );
        }

        const jars = (await fs.readdir(dependencyDir).catch(() => [])).filter(name => name.endsWith('.jar'));
        return jars.map(jar => path.join(dependencyDir, jar));
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

        const classpath = [classesDir, ...(await MavenBuildSystem.gatherRuntimeDependencies(ctx))].join(path.delimiter);

        executeOptions.args = [
            // The sandbox is tight on threads and memory, and a JVM left to itself asks for far more than it gets:
            // it sizes its heap and its pools from the machine's memory, which it sees rather than the cgroup it is
            // held to (200 MiB, in etc/nsjail/user-execution.cfg). Saying so outright is what makes a Kotlin program
            // start reliably; a Java one is light enough to have got away with it.
            '-XX:MaxRAM=192m',
            // The rest are the concessions the Java compiler makes when it runs a class directly, except for the
            // stack: it uses 136K, which is not enough to get through the nested class loading a Kotlin program
            // starts with -- reaching its standard library's collections overflows it before main. Still tiny.
            '-Xss512K',
            '-XX:CICompilerCount=2',
            '-XX:-UseDynamicNumberOfCompilerThreads',
            '-XX:-UseDynamicNumberOfGCThreads',
            '-XX:+UseSerialGC',
            '-cp',
            classpath,
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
