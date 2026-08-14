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

import fsSync from 'node:fs';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';

import {describe, expect, it, vi} from 'vitest';

import {BaseCompiler} from '../lib/base-compiler.js';
import {CargoBuildSystem} from '../lib/build-systems/cargo.js';
import type {BuildContext, BuildPlan} from '../lib/build-systems/index.js';
import {
    cargoBuildSystem,
    cmakeBuildSystem,
    getBuildSystem,
    getBuildSystemArgs,
    makeBuildSystem,
    mavenBuildSystem,
} from '../lib/build-systems/index.js';
import {MakeBuildSystem} from '../lib/build-systems/make.js';
import {MavenBuildSystem} from '../lib/build-systems/maven.js';
import {CompilationEnvironment} from '../lib/compilation-env.js';
import {RustCompiler} from '../lib/compilers/rust.js';
import {ParsedRequest} from '../lib/handlers/compile.js';
import {
    BuildSystems,
    getBuildSystemByManifestFilename,
    getBuildSystemByManifestLanguageId,
    getBuildSystemsForLanguage,
    isBuildSystemId,
    isManifestLanguageId,
} from '../shared/build-systems.js';
import {BypassCache} from '../types/compilation/compilation.interfaces.js';
import {CompilerInfo} from '../types/compiler.interfaces.js';
import {makeCompilationEnvironment, makeFakeCompilerInfo} from './utils.js';

const languages = {
    'c++': {id: 'c++'},
} as const;

const CMAKE_EXE = '/opt/compiler-explorer/cmake/bin/cmake';

function makeEnv(props: Record<string, any> = {}): CompilationEnvironment {
    return makeCompilationEnvironment({languages, props: {cmake: CMAKE_EXE, useninja: false, ...props}});
}

function makeCompiler(env: CompilationEnvironment, info: Partial<CompilerInfo> = {}): BaseCompiler {
    return new BaseCompiler(
        makeFakeCompilerInfo({
            exe: '/usr/bin/g++',
            lang: 'c++',
            ldPath: [],
            libPath: [],
            supportsBinary: true,
            ...info,
        }),
        env,
    );
}

/** An `env` compiler override, as the overrides widget produces. */
function envOverride(vars: Record<string, string>) {
    return {name: 'env', values: Object.entries(vars).map(([name, value]) => ({name, value}))};
}

function makeParsedRequest(backendOptions: Record<string, any> = {}): ParsedRequest {
    return {
        source: 'project(test)',
        options: [],
        backendOptions,
        filters: {},
        bypassCache: BypassCache.None,
        tools: [],
        executeParameters: {args: [], stdin: '', runtimeTools: []},
        libraries: [],
    } as unknown as ParsedRequest;
}

function makeContext(compiler: BaseCompiler, env: CompilationEnvironment, parsedRequest: ParsedRequest): BuildContext {
    const dirPath = '/tmp/ce-fake-build';
    return {
        compiler,
        env,
        dirPath,
        buildPath: cmakeBuildSystem.getBuildPath(dirPath),
        key: compiler.getBuildProjectCacheKey(cmakeBuildSystem, parsedRequest, []),
        parsedRequest,
        files: [],
        libsAndOptions: {libraries: [], options: []},
        toolchainPath: undefined,
        buildSystemArgs: getBuildSystemArgs(parsedRequest.backendOptions),
    };
}

describe('Build system registry', () => {
    it('resolves cmake', () => {
        expect(isBuildSystemId('cmake')).toBe(true);
        expect(getBuildSystem('cmake')).toBe(cmakeBuildSystem);
    });

    it('resolves cargo', () => {
        expect(isBuildSystemId('cargo')).toBe(true);
        expect(getBuildSystem('cargo')).toBe(cargoBuildSystem);
    });

    it('resolves make', () => {
        expect(isBuildSystemId('make')).toBe(true);
        expect(getBuildSystem('make')).toBe(makeBuildSystem);
    });

    it('resolves maven', () => {
        expect(isBuildSystemId('maven')).toBe(true);
        expect(getBuildSystem('maven')).toBe(mavenBuildSystem);
    });

    it('does not resolve unknown build systems', () => {
        expect(isBuildSystemId('gradle')).toBe(false);
        expect(getBuildSystem('gradle')).toBeUndefined();
        expect(getBuildSystem('toString')).toBeUndefined();
    });

    it('takes execute out of the very filters it was handed when making a cache key', () => {
        // Load-bearing, and easy to trip over: the key is built from the request's own filters rather than a copy,
        // so anything that wants to know whether to execute has to ask before the key is made, not after.
        const env = makeCompilationEnvironment({languages: {'c++': {id: 'c++'}}});
        const compiler = new BaseCompiler(
            makeFakeCompilerInfo({exe: '/usr/bin/g++', lang: 'c++', ldPath: [], libPath: []}),
            env,
        );
        const req = makeParsedRequest();
        req.filters.execute = true;

        compiler.getBuildProjectCacheKey(cmakeBuildSystem, req, []);

        expect(req.filters.execute).toBeUndefined();
    });

    it('recognises each build system by its manifest filename', () => {
        // Makefile is the reason this exists: it has no extension, so nothing else identifies it.
        expect(getBuildSystemByManifestFilename('Makefile')?.id).toEqual('make');
        expect(getBuildSystemByManifestFilename('CMakeLists.txt')?.id).toEqual('cmake');
        expect(getBuildSystemByManifestFilename('Cargo.toml')?.id).toEqual('cargo');
        expect(getBuildSystemByManifestFilename('pom.xml')?.id).toEqual('maven');

        // A manifest in a subdirectory is still one, however the path is spelled.
        expect(getBuildSystemByManifestFilename('sub/dir/Makefile')?.id).toEqual('make');
        expect(getBuildSystemByManifestFilename('sub\\dir\\Makefile')?.id).toEqual('make');

        expect(getBuildSystemByManifestFilename('makefile')).toBeUndefined();
        expect(getBuildSystemByManifestFilename('Makefile.am')).toBeUndefined();
        expect(getBuildSystemByManifestFilename('main.cpp')).toBeUndefined();
        expect(getBuildSystemByManifestFilename('')).toBeUndefined();
    });

    it('names the manifest a given language is written in', () => {
        // What the rename dialog offers, which is nothing the language's own extension would suggest.
        expect(getBuildSystemByManifestLanguageId('makefile')?.manifestFilename).toEqual('Makefile');
        expect(getBuildSystemByManifestLanguageId('cmake')?.manifestFilename).toEqual('CMakeLists.txt');
        expect(getBuildSystemByManifestLanguageId('cargo')?.manifestFilename).toEqual('Cargo.toml');
        expect(getBuildSystemByManifestLanguageId('maven')?.manifestFilename).toEqual('pom.xml');

        expect(getBuildSystemByManifestLanguageId('c++')).toBeUndefined();
        expect(getBuildSystemByManifestLanguageId('rust')).toBeUndefined();
    });

    it('knows which languages are a manifest rather than something to compile', () => {
        expect(isManifestLanguageId('makefile')).toBe(true);
        expect(isManifestLanguageId('cmake')).toBe(true);
        expect(isManifestLanguageId('cargo')).toBe(true);
        expect(isManifestLanguageId('maven')).toBe(true);

        expect(isManifestLanguageId('c++')).toBe(false);
        expect(isManifestLanguageId('rust')).toBe(false);
        expect(isManifestLanguageId('make')).toBe(false);
    });

    it('offers each build system only for the languages it can build', () => {
        // Make is offered everywhere, a Makefile being able to drive anything, so it accompanies each of these.
        expect(getBuildSystemsForLanguage('c++').map(bs => bs.id)).toEqual(['cmake', 'make']);
        // CMake builds Rust too, given a compiler it can drive; whether a particular one can is the driver's business.
        expect(getBuildSystemsForLanguage('rust').map(bs => bs.id)).toEqual(['cmake', 'cargo', 'make']);
        expect(getBuildSystemsForLanguage('java').map(bs => bs.id)).toEqual(['make', 'maven']);
        expect(getBuildSystemsForLanguage('kotlin').map(bs => bs.id)).toEqual(['make', 'maven']);
        expect(getBuildSystemsForLanguage('haskell').map(bs => bs.id)).toEqual(['make']);
    });
});

describe('Build system arguments', () => {
    it('takes buildSystemArgs when given', () => {
        expect(getBuildSystemArgs({buildSystemArgs: '-DA=1 -DB=2'})).toEqual(['-DA=1', '-DB=2']);
    });

    it('falls back to the original cmakeArgs, as sent by the frontend and stored in shared links', () => {
        expect(getBuildSystemArgs({cmakeArgs: '-DA=1'})).toEqual(['-DA=1']);
        expect(getBuildSystemArgs({buildSystemArgs: '-DNew=1', cmakeArgs: '-DOld=1'})).toEqual(['-DNew=1']);
    });

    it('copes with neither being given', () => {
        expect(getBuildSystemArgs({})).toEqual([]);
    });
});

describe('CMake build system', () => {
    it('builds into a build subdirectory of the project', () => {
        expect(cmakeBuildSystem.descriptor.manifestFilename).toEqual('CMakeLists.txt');
        expect(cmakeBuildSystem.getBuildPath('/tmp/project')).toEqual(path.join('/tmp/project', 'build'));
    });

    it('refuses compilers that cannot produce binaries', async () => {
        const env = makeEnv();
        expect(await cmakeBuildSystem.getUnsupportedReason(makeCompiler(env))).toBeUndefined();
        expect(await cmakeBuildSystem.getUnsupportedReason(makeCompiler(env, {supportsBinary: false}))).toEqual(
            'Compiler does not support compiling to binaries',
        );
    });

    it('forces a binary, unmasked build', () => {
        const env = makeEnv();
        const compiler = makeCompiler(env);
        const parsedRequest = makeParsedRequest();

        cmakeBuildSystem.applyRequestDefaults(compiler, parsedRequest);

        expect(parsedRequest.filters.binary).toBe(true);
        expect(parsedRequest.filters.dontMaskFilenames).toBe(true);
    });

    it('names the artifact after the output filebase, in the build directory', () => {
        const env = makeEnv();
        const compiler = makeCompiler(env);
        const ctx = makeContext(compiler, env, makeParsedRequest());

        // Note this is what a CMakeLists.txt in tree mode has to name its target.
        expect(cmakeBuildSystem.getArtifactFilename(ctx)).toEqual(
            path.join('/tmp/ce-fake-build', 'build', `${compiler.outputFilebase}.s`),
        );
    });

    it('honours a custom output filename', () => {
        const env = makeEnv();
        const compiler = makeCompiler(env);
        const ctx = makeContext(compiler, env, makeParsedRequest({customOutputFilename: 'thing.elf'}));

        expect(cmakeBuildSystem.getArtifactFilename(ctx)).toEqual(
            path.join('/tmp/ce-fake-build', 'build', 'thing.elf'),
        );
    });

    it('hands CMake a Rust compiler and its flags, which it will not find for itself', async () => {
        const env = makeCompilationEnvironment({languages: {rust: {id: 'rust'}}, props: {cmake: CMAKE_EXE}});
        const rustc = new RustCompiler(
            makeFakeCompilerInfo({
                exe: '/opt/compiler-explorer/rust-1.91.0/bin/rustc',
                lang: 'rust',
                semver: '1.91.0',
                ldPath: [],
                libPath: [],
                supportsBinary: true,
                options: '-C debuginfo=2',
            }),
            env,
        );
        const req = makeParsedRequest();
        req.options = ['--edition', '2021'];

        expect(rustc.getExtraCMakeArgs(req)).toEqual([
            '-DCMAKE_Rust_COMPILER=/opt/compiler-explorer/rust-1.91.0/bin/rustc',
            '-DCMAKE_Rust_FLAGS=-C debuginfo=2 --edition 2021',
        ]);
        expect(await cmakeBuildSystem.getUnsupportedReason(rustc)).toBeUndefined();
    });

    it('plans a configure step followed by a build step', async () => {
        const env = makeEnv();
        const compiler = makeCompiler(env);
        const ctx = makeContext(compiler, env, makeParsedRequest({cmakeArgs: '-DCMAKE_BUILD_TYPE=Debug'}));

        const plan = await cmakeBuildSystem.getBuildPlan(ctx);

        expect(plan.steps.map(step => step.name)).toEqual(['cmake', 'build']);
        expect(plan.steps.map(step => step.exe)).toEqual([CMAKE_EXE, CMAKE_EXE]);
        // The toolchain path is derived from the compiler's exe, so its spelling is platform-dependent.
        expect(plan.steps[0].args[0]).toMatch(/^-DCMAKE_CXX_COMPILER_EXTERNAL_TOOLCHAIN=/);
        expect(plan.steps[0].args.slice(1)).toEqual(['-DCMAKE_BUILD_TYPE=Debug', '..']);
        expect(plan.steps[1].args).toEqual(['--build', '.']);
    });

    it('only reports the compilation options if the configure step fails', async () => {
        const env = makeEnv();
        const compiler = makeCompiler(env);
        const plan = await cmakeBuildSystem.getBuildPlan(makeContext(compiler, env, makeParsedRequest()));

        expect(plan.steps[0].reportsCompilationOptions).toBe(true);
        expect(plan.steps[1].reportsCompilationOptions).toBeFalsy();
        expect(plan.steps.map(step => step.failureMessage)).toEqual([
            '<CMake configure step failed>',
            '<CMake build step failed>',
        ]);
    });

    it('passes the environment the user asked for to the build', async () => {
        const env = makeEnv();
        const compiler = makeCompiler(env);
        const req = makeParsedRequest({overrides: [envOverride({CE_PROBE: 'reached'})]});
        const plan = await cmakeBuildSystem.getBuildPlan(makeContext(compiler, env, req));

        expect(plan.steps[0].execParams.env.CE_PROBE).toEqual('reached');
    });

    it('runs both steps in the build directory, sharing the compiler environment', async () => {
        const env = makeEnv();
        const compiler = makeCompiler(env, {options: '-fsome-flag'});
        const ctx = makeContext(compiler, env, makeParsedRequest());

        const plan = await cmakeBuildSystem.getBuildPlan(ctx);

        for (const step of plan.steps) {
            expect(step.execParams.customCwd).toEqual(ctx.buildPath);
            expect(step.execParams.appHome).toEqual(ctx.dirPath);
            // The compiler flags are set up for the configure step, but the build step must see them too.
            expect(step.execParams.env.CXXFLAGS).toContain('-fsome-flag');
        }
        expect(plan.getCompilationOptions()).toContain('-fsome-flag');
    });

    it('asks for the ninja generator when configured to', async () => {
        const env = makeEnv({useninja: true});
        const compiler = makeCompiler(env);
        const plan = await cmakeBuildSystem.getBuildPlan(makeContext(compiler, env, makeParsedRequest()));

        expect(plan.steps[0].args[0]).toEqual('-GNinja');
    });

    it('passes on compiler-specific cmake arguments', async () => {
        const env = makeEnv();
        const compiler = makeCompiler(env);
        compiler.getExtraCMakeArgs = () => ['-DSOMETHING=1'];
        const ctx = makeContext(compiler, env, makeParsedRequest({cmakeArgs: '-DUSER=2'}));

        const plan = await cmakeBuildSystem.getBuildPlan(ctx);

        expect(plan.steps[0].args[0]).toMatch(/^-DCMAKE_CXX_COMPILER_EXTERNAL_TOOLCHAIN=/);
        expect(plan.steps[0].args.slice(1)).toEqual(['-DSOMETHING=1', '-DUSER=2', '..']);
    });
});

describe('Cargo build system', () => {
    const rustLanguages = {rust: {id: 'rust'}} as const;
    // The real cargo of a real CE toolchain, so the sibling-of-rustc lookup is checked against something that exists.
    const RUSTC_EXE = '/opt/compiler-explorer/rust-1.91.0/bin/rustc';

    function makeRustEnv(): CompilationEnvironment {
        return makeCompilationEnvironment({languages: rustLanguages, props: {}});
    }

    function makeRustCompiler(env: CompilationEnvironment, info: Partial<CompilerInfo> = {}): BaseCompiler {
        return new BaseCompiler(
            makeFakeCompilerInfo({exe: RUSTC_EXE, lang: 'rust', ldPath: [], libPath: [], ...info}),
            env,
        );
    }

    function makeCargoContext(compiler: BaseCompiler, env: CompilationEnvironment, req: ParsedRequest): BuildContext {
        const dirPath = '/tmp/ce-fake-cargo';
        return {
            compiler,
            env,
            dirPath,
            buildPath: cargoBuildSystem.getBuildPath(dirPath),
            key: compiler.getBuildProjectCacheKey(cargoBuildSystem, req, []),
            parsedRequest: req,
            files: [],
            libsAndOptions: {libraries: req.libraries, options: req.options},
            toolchainPath: undefined,
            buildSystemArgs: getBuildSystemArgs(req.backendOptions),
        };
    }

    it('builds in the project root, where cargo makes its own target directory', () => {
        expect(cargoBuildSystem.descriptor.manifestFilename).toEqual('Cargo.toml');
        expect(cargoBuildSystem.getBuildPath('/tmp/project')).toEqual('/tmp/project');
    });

    it('takes cargo from the selected compiler own toolchain, not a global one', () => {
        const compiler = makeRustCompiler(makeRustEnv());
        expect(CargoBuildSystem.getCargoPath(compiler)).toEqual(path.join(path.dirname(RUSTC_EXE), 'cargo'));
    });

    it('refuses Rust compilers that ship no cargo', async () => {
        const env = makeRustEnv();
        // gccrs and mrustc are Rust compilers with no cargo beside them.
        const gccrs = makeRustCompiler(env, {exe: '/opt/compiler-explorer/gcc-16.1.0/bin/gccrs'});
        expect(await cargoBuildSystem.getUnsupportedReason(gccrs)).toEqual('This compiler does not come with cargo');
        expect(await cargoBuildSystem.getUnsupportedReason(makeRustCompiler(env, {exe: ''}))).toEqual(
            'Compiler has no executable to find a cargo alongside',
        );
    });

    it('plans one offline build step that reports artifacts as JSON', async () => {
        const env = makeRustEnv();
        const compiler = makeRustCompiler(env);
        const plan = await cargoBuildSystem.getBuildPlan(makeCargoContext(compiler, env, makeParsedRequest()));

        expect(plan.steps.map(step => step.name)).toEqual(['cargo']);
        expect(plan.steps[0].exe).toEqual(path.join(path.dirname(RUSTC_EXE), 'cargo'));
        expect(plan.steps[0].args).toEqual(['build', '--offline', '--message-format=json-render-diagnostics']);
    });

    it('hands prebuilt libraries to rustc, since cargo cannot resolve them offline', async () => {
        const env = makeRustEnv();
        const compiler = makeRustCompiler(env);
        const ctx = makeCargoContext(compiler, env, makeParsedRequest());
        // Stand in for a library the user picked, already unpacked into the project by setupBuildEnvironment.
        compiler.getIncludeArguments = () => ['--extern', 'rand=rand/build/debug/librand.rlib'];

        const plan = await cargoBuildSystem.getBuildPlan(ctx);

        expect(plan.steps[0].execParams.env.CARGO_ENCODED_RUSTFLAGS).toEqual(
            '--extern\x1frand=rand/build/debug/librand.rlib',
        );
    });

    it('explains that crates cannot be declared in Cargo.toml when resolution fails', () => {
        const offline = [
            'error: no matching package named `rand` found',
            'location searched: crates.io index',
            "As a reminder, you're using offline mode (--offline)",
        ].join('\n');
        expect(CargoBuildSystem.explainFailure(offline)).toMatch(/Libraries pane/);
        // An ordinary compile error is cargo's to explain, not ours.
        expect(CargoBuildSystem.explainFailure('error[E0425]: cannot find value `x` in this scope')).toBeUndefined();
    });

    it('explains that a library feature cannot be chosen, but leaves your own features alone', () => {
        expect(
            CargoBuildSystem.explainFailure(
                "error: the package 'output' does not contain this feature: rand/small_rng",
            ),
        ).toMatch(/its rlib is prebuilt/);
        // Their own feature, misspelled: cargo has already said the useful thing.
        expect(
            CargoBuildSystem.explainFailure("error: the package 'output' does not contain this feature: nosuch"),
        ).toBeUndefined();
    });

    it('drives the selected rustc and keeps cargo caches inside the sandbox', async () => {
        const env = makeRustEnv();
        const compiler = makeRustCompiler(env);
        const ctx = makeCargoContext(compiler, env, makeParsedRequest());
        const plan = await cargoBuildSystem.getBuildPlan(ctx);

        expect(plan.steps[0].execParams.env.RUSTC).toEqual(RUSTC_EXE);
        expect(plan.steps[0].execParams.env.CARGO_HOME).toEqual(path.join(ctx.dirPath, '.cargo-home'));
        expect(plan.steps[0].execParams.customCwd).toEqual(ctx.dirPath);
    });

    it('passes the user arguments to cargo, and compiler options through RUSTFLAGS', async () => {
        const env = makeRustEnv();
        const compiler = makeRustCompiler(env);
        const req = makeParsedRequest({buildSystemArgs: '--release'});
        req.options = ['-C', 'opt-level=3'];
        const plan = await cargoBuildSystem.getBuildPlan(makeCargoContext(compiler, env, req));

        expect(plan.steps[0].args).toContain('--release');
        expect(plan.steps[0].execParams.env.CARGO_ENCODED_RUSTFLAGS).toEqual('-C\x1fopt-level=3');
    });

    it('maps the sandbox paths cargo reports back to real ones', () => {
        // CE bind-mounts the project at /app, so this is what cargo actually reports during a real compilation.
        expect(CargoBuildSystem.toHostPath('/app/target/debug/output', '/tmp/ce-build')).toEqual(
            path.join('/tmp/ce-build', 'target/debug/output'),
        );
        // Run without a sandbox, cargo reports the real temp directory instead, which rebases to the same place.
        expect(
            CargoBuildSystem.toHostPath('/tmp/compiler-explorer-compilerXYZ/target/debug/output', '/tmp/ce-build'),
        ).toEqual(path.join('/tmp/ce-build', 'target/debug/output'));
        // A path under neither spelling of the root is not ours to rebase.
        expect(CargoBuildSystem.toHostPath('/tmp/elsewhere/target/debug/output', '/tmp/ce-build')).toEqual(
            '/tmp/elsewhere/target/debug/output',
        );
        // A directory that merely starts with the same letters is not the mount point.
        expect(CargoBuildSystem.toHostPath('/applications/thing', '/tmp/ce-build')).toEqual('/applications/thing');
    });

    it('copies what cargo built to the path the compilation was told to expect', async () => {
        const env = makeRustEnv();
        const compiler = makeRustCompiler(env);
        const ctx = makeCargoContext(compiler, env, makeParsedRequest());
        const artifact = cargoBuildSystem.getArtifactFilename(ctx);
        expect(artifact).toEqual(path.join('/tmp/ce-fake-cargo', 'output.s'));

        const copied: [string, string][] = [];
        const result = {
            buildsteps: [
                {
                    step: 'cargo',
                    stdout: [
                        {text: '{"reason":"compiler-artifact","executable":"/app/target/debug/mycrate"}'},
                        {text: '{"reason":"build-finished","success":true}'},
                    ],
                },
            ],
        } as any;

        await cargoBuildSystem.finaliseArtifact(ctx, result, artifact, async (from: string, to: string) => {
            copied.push([from, to]);
        });

        expect(copied).toEqual([[path.join(ctx.dirPath, 'target/debug/mycrate'), artifact]]);
        // The JSON records are for us, not the user, who reads cargo's stderr.
        expect(result.buildsteps[0].stdout).toEqual([]);
    });

    it('picks the artifact the user asked for when a project builds several', async () => {
        const env = makeRustEnv();
        const compiler = makeRustCompiler(env);
        const ctx = makeCargoContext(compiler, env, makeParsedRequest({customOutputFilename: 'helper'}));
        const artifact = cargoBuildSystem.getArtifactFilename(ctx);

        const copied: [string, string][] = [];
        const result = {
            buildsteps: [
                {
                    step: 'cargo',
                    stdout: [
                        {text: '{"reason":"compiler-artifact","executable":"/app/target/debug/main"}'},
                        {text: '{"reason":"compiler-artifact","executable":"/app/target/debug/helper"}'},
                    ],
                },
            ],
        } as any;

        await cargoBuildSystem.finaliseArtifact(ctx, result, artifact, async (from: string, to: string) => {
            copied.push([from, to]);
        });

        expect(copied).toEqual([[path.join(ctx.dirPath, 'target/debug/helper'), artifact]]);
    });

    it('will not copy the built binary to a path outside the project', () => {
        const env = makeRustEnv();
        const compiler = makeRustCompiler(env);
        const ctx = makeCargoContext(compiler, env, makeParsedRequest({customOutputFilename: '../../../../evil'}));

        // finaliseArtifact copies to whatever this returns, so the name has to be refused before it gets there.
        expect(() => cargoBuildSystem.getArtifactFilename(ctx)).toThrow(Error);
    });

    it('takes a bin named for the artifact without its extension, which is how cargo would spell it', async () => {
        const env = makeRustEnv();
        const compiler = makeRustCompiler(env);
        const ctx = makeCargoContext(compiler, env, makeParsedRequest());
        const artifact = cargoBuildSystem.getArtifactFilename(ctx);

        const copied: [string, string][] = [];
        const result = {
            buildsteps: [
                {
                    step: 'cargo',
                    stdout: [
                        // `output` is what the example's [[bin]] is called, and it is not the last one built.
                        {text: '{"reason":"compiler-artifact","executable":"/app/target/debug/output"}'},
                        {text: '{"reason":"compiler-artifact","executable":"/app/target/debug/helper"}'},
                    ],
                },
            ],
        } as any;

        await cargoBuildSystem.finaliseArtifact(ctx, result, artifact, async (from: string, to: string) => {
            copied.push([from, to]);
        });

        expect(copied).toEqual([[path.join(ctx.dirPath, 'target/debug/output'), artifact]]);
    });

    it('explains, rather than fails, when cargo built no executable', async () => {
        const env = makeRustEnv();
        const compiler = makeRustCompiler(env);
        const ctx = makeCargoContext(compiler, env, makeParsedRequest());
        const result = {
            buildsteps: [{step: 'cargo', stdout: [{text: '{"reason":"build-finished","success":true}'}]}],
        } as any;

        // A library-only crate builds fine; there is just nothing to disassemble.
        expect(await cargoBuildSystem.finaliseArtifact(ctx, result, cargoBuildSystem.getArtifactFilename(ctx))).toMatch(
            /did not build an executable/,
        );
    });

    it('keeps what cargo printed, dropping only the records we asked it for', async () => {
        const env = makeRustEnv();
        const compiler = makeRustCompiler(env);
        const ctx = makeCargoContext(compiler, env, makeParsedRequest());
        // `cargo build --help` prints to stdout and never builds; that output is the whole point of the request.
        const result = {
            buildsteps: [
                {
                    step: 'cargo',
                    stdout: [
                        {text: 'Compile a local package and all of its dependencies'},
                        {text: ''},
                        {text: 'Usage: cargo build [OPTIONS]'},
                        {text: '{"reason":"build-finished","success":true}'},
                    ],
                },
            ],
        } as any;

        await cargoBuildSystem.finaliseArtifact(ctx, result, cargoBuildSystem.getArtifactFilename(ctx));

        expect(result.buildsteps[0].stdout.map((l: any) => l.text)).toEqual([
            'Compile a local package and all of its dependencies',
            '',
            'Usage: cargo build [OPTIONS]',
        ]);
    });
});

describe('Maven build system', () => {
    const javaLanguages = {java: {id: 'java'}} as const;

    // The three things the driver reads, laid out as they are installed: a maven, a Kotlin installation holding the
    // jars, and the repository beside it holding a pom for what the installation answers for. A second Kotlin with
    // no repository of its own stands for one whose Maven artifacts were never installed.
    // realpathSync.native, not realpathSync: the JS one resolves symlinks but leaves a Windows 8.3 short name as it
    // found it, where the promise-based realpath a test reads a symlink back with returns the long form.
    const fixtures = fsSync.realpathSync.native(fsSync.mkdtempSync(path.join(os.tmpdir(), 'ce-maven-')));
    const fakeMavenHome = path.join(fixtures, 'maven');
    const fakeKotlinHome = path.join(fixtures, 'kotlin-jvm-2.1.21');
    const fakeKotlinRepository = `${fakeKotlinHome}-maven`;
    const fakeProjectDir = path.join(fixtures, 'project');
    fsSync.mkdirSync(fakeKotlinRepository, {recursive: true});
    // What infra records when it takes those jars back out, and so what the driver is to link in from the install.
    fsSync.writeFileSync(path.join(fakeKotlinRepository, '.ce-supplied-by-installation'), 'kotlin-compiler\n');
    fsSync.mkdirSync(path.join(fakeMavenHome, 'repository'), {recursive: true});
    fsSync.mkdirSync(path.join(fakeKotlinHome, 'lib'), {recursive: true});
    fsSync.writeFileSync(path.join(fakeKotlinHome, 'lib', 'kotlin-compiler.jar'), '');
    fsSync.mkdirSync(path.join(fixtures, 'kotlin-jvm-1.9.20', 'lib'), {recursive: true});
    fsSync.mkdirSync(fakeProjectDir, {recursive: true});
    // A JDK of the fixtures' own: the driver refuses a compiler it can find no java for, so one has to be here
    // rather than wherever this machine happens to have installed one.
    const fakeJavaHome = path.join(fixtures, 'jdk-21.0.0');
    fsSync.mkdirSync(path.join(fakeJavaHome, 'bin'), {recursive: true});
    fsSync.writeFileSync(path.join(fakeJavaHome, 'bin', 'java'), '');
    const JAVAC_EXE = path.join(fakeJavaHome, 'bin', 'javac');
    // The launcher the driver walks up from to find the repository primed when maven was installed.
    const MVN_EXE = path.join(fakeMavenHome, 'bin', 'mvn');

    function makeJavaEnv(): CompilationEnvironment {
        return makeCompilationEnvironment({
            languages: javaLanguages,
            props: {maven: MVN_EXE},
        });
    }

    function makeJavaCompiler(env: CompilationEnvironment, info: Partial<CompilerInfo> = {}): BaseCompiler {
        return new BaseCompiler(
            makeFakeCompilerInfo({exe: JAVAC_EXE, lang: 'java', ldPath: [], libPath: [], ...info}),
            env,
        );
    }

    function makeMavenContext(compiler: BaseCompiler, env: CompilationEnvironment, req: ParsedRequest): BuildContext {
        const dirPath = '/tmp/ce-fake-maven';
        return {
            compiler,
            env,
            dirPath,
            buildPath: mavenBuildSystem.getBuildPath(dirPath),
            key: compiler.getBuildProjectCacheKey(mavenBuildSystem, req, []),
            parsedRequest: req,
            files: [],
            libsAndOptions: {libraries: req.libraries, options: req.options},
            toolchainPath: undefined,
            buildSystemArgs: getBuildSystemArgs(req.backendOptions),
        };
    }

    it('builds in the project root, where maven makes its own target directory', () => {
        expect(mavenBuildSystem.descriptor.manifestFilename).toEqual('pom.xml');
        expect(mavenBuildSystem.getBuildPath('/tmp/project')).toEqual('/tmp/project');
    });

    it('takes JAVA_HOME from the selected compiler, whose exe lives in its bin', () => {
        const compiler = makeJavaCompiler(makeJavaEnv());
        expect(MavenBuildSystem.getJavaHome(compiler)).toEqual(fakeJavaHome);
    });

    it('refuses compilers that no JDK can be found for', async () => {
        const env = makeJavaEnv();
        expect(await mavenBuildSystem.getUnsupportedReason(makeJavaCompiler(env))).toBeUndefined();
        // Inside the fixtures rather than under /usr/bin, where a machine with a system java installed -- which is
        // most of them, CI included -- would have one for this to find and the driver would be right to allow it.
        const notAJdk = makeJavaCompiler(env, {exe: path.join(fixtures, 'not-a-jdk', 'bin', 'javac')});
        expect(await mavenBuildSystem.getUnsupportedReason(notAJdk)).toMatch(/No JDK could be found/);
    });

    it('asks a Kotlin compiler which JDK it belongs to, being no part of one itself', async () => {
        const env = makeJavaEnv();
        // What KotlinCompiler reads from compiler.<id>.java_home: kotlinc lives in its own kotlin-jvm-x.y.z, so
        // deriving a JDK from its path would find nothing to run maven with.
        const kotlinc = Object.assign(
            makeJavaCompiler(env, {exe: path.join(fakeKotlinHome, 'bin', 'kotlinc-jvm'), lang: 'kotlin'}),
            {javaHome: fakeJavaHome},
        );
        expect(MavenBuildSystem.getJavaHome(kotlinc)).toEqual(fakeJavaHome);
    });

    it('falls back to the JDK that runs the compiler output', async () => {
        const env = makeJavaEnv();
        // What JavaCompiler reads from compiler.<id>.runtime, for anything that does not declare a java_home.
        const compiler = Object.assign(makeJavaCompiler(env, {exe: '/somewhere/else/javac'}), {
            javaRuntime: path.join(fakeJavaHome, 'bin', 'java'),
        });
        expect(MavenBuildSystem.getJavaHome(compiler)).toEqual(fakeJavaHome);
    });

    it('builds Kotlin with the compiler that was selected, jars and all', async () => {
        const env = makeCompilationEnvironment({
            languages: {kotlin: {id: 'kotlin'}},
            props: {maven: path.join(fakeMavenHome, 'bin', 'mvn')},
        });
        const kotlinc = Object.assign(
            new BaseCompiler(
                makeFakeCompilerInfo({
                    exe: path.join(fakeKotlinHome, 'bin', 'kotlinc-jvm'),
                    lang: 'kotlin',
                    semver: '2.1.21',
                    ldPath: [],
                    libPath: [],
                }),
                env,
            ),
            {javaHome: fakeJavaHome},
        );
        const ctx = makeMavenContext(kotlinc, env, makeParsedRequest());
        ctx.dirPath = fakeProjectDir;
        const plan = await mavenBuildSystem.getBuildPlan(ctx);

        // The plugin resolves its compiler by version, so the selected one has to be named...
        expect(plan.steps[0].args).toContain('-Dkotlin.version=2.1.21');
        // ...and found: the installation's own jar, in a repository of this build's own, ahead of the one installed
        // beside that compiler, ahead of the shared one.
        expect(plan.steps[0].args).toContain(`-Dmaven.repo.local=${path.join(fakeProjectDir, '.m2')}`);
        expect(plan.steps[0].args).toContain(
            `-Dmaven.repo.local.tail=${fakeKotlinRepository},${path.join(fakeMavenHome, 'repository')}`,
        );
        const linked = path.join(
            fakeProjectDir,
            '.m2/org/jetbrains/kotlin/kotlin-compiler/2.1.21/kotlin-compiler-2.1.21.jar',
        );
        // Both sides resolved, so that what is compared is which file each names rather than how it is spelled.
        expect(await fs.realpath(linked)).toEqual(
            await fs.realpath(path.join(fakeKotlinHome, 'lib', 'kotlin-compiler.jar')),
        );
    });

    it('leaves a Java build reading the shared repository and nothing else', async () => {
        const env = makeJavaEnv();
        const plan = await mavenBuildSystem.getBuildPlan(
            makeMavenContext(makeJavaCompiler(env), env, makeParsedRequest()),
        );

        expect(plan.steps[0].args).toContain(`-Dmaven.repo.local.tail=${path.join(fakeMavenHome, 'repository')}`);
        expect(plan.steps[0].args.join(' ')).not.toContain('kotlin.version');
    });

    it('refuses a Kotlin whose maven artifacts were never installed, naming the ones that were', async () => {
        const env = makeCompilationEnvironment({
            languages: {kotlin: {id: 'kotlin'}},
            props: {maven: path.join(fakeMavenHome, 'bin', 'mvn')},
        });
        // 1.9.20 is installed beside 2.1.21, but without a repository of its own: infra installs those separately.
        const kotlinc = Object.assign(
            new BaseCompiler(
                makeFakeCompilerInfo({
                    exe: path.join(fixtures, 'kotlin-jvm-1.9.20', 'bin', 'kotlinc-jvm'),
                    lang: 'kotlin',
                    semver: '1.9.20',
                    ldPath: [],
                    libPath: [],
                }),
                env,
            ),
            {javaHome: fakeJavaHome},
        );

        const reason = await mavenBuildSystem.getUnsupportedReason(kotlinc);
        expect(reason).toMatch(/Maven artifacts for this Kotlin are not installed/);
        expect(reason).toMatch(/Maven is available for 2\.1\.21/);
    });

    it('says which Kotlin a project may ask for, maven bringing its own compiler rather than using the one picked', () => {
        const explanation = MavenBuildSystem.explainFailure(
            'Could not resolve dependencies: org.jetbrains.kotlin:kotlin-maven-plugin:jar:1.9.20 has not been downloaded',
            ['2.1.21'],
        );
        expect(explanation).toMatch(/installed here for 2\.1\.21\. Select one of those Kotlin compilers/);
    });

    it('blames the installation, not the project, when the selected Kotlin is not really there', () => {
        const explanation = MavenBuildSystem.explainFailure(
            'Could not resolve dependencies: org.jetbrains.kotlin:kotlin-compiler:jar:2.1.21 has not been downloaded',
            ['2.1.21'],
            '2.1.21',
            ['kotlin-compiler'],
        );
        expect(explanation).toMatch(/Kotlin 2\.1\.21 does not appear to be fully installed/);
        expect(explanation).toMatch(/for Compiler Explorer to fix, not your project/);
    });

    it('explains an unresolvable Kotlin nothing was bundled for as the ordinary offline failure', () => {
        const explanation = MavenBuildSystem.explainFailure(
            'Could not resolve dependencies: org.jetbrains.kotlin:kotlin-maven-plugin:jar:1.9.20 has not been downloaded',
            [],
        );
        expect(explanation).toMatch(/cannot download from Maven Central/);
    });

    it('asks javap for the bytecode rather than disassembling a binary', () => {
        const env = makeJavaEnv();
        const compiler = makeJavaCompiler(env);
        const req = makeParsedRequest();
        mavenBuildSystem.applyRequestDefaults(compiler, req);
        // Java reads binary as "run javap"; there is no native binary here at all.
        expect(req.filters.binary).toBe(true);
    });

    it('builds offline against the repository primed when maven was installed', async () => {
        const env = makeJavaEnv();
        const compiler = makeJavaCompiler(env);
        const plan = await mavenBuildSystem.getBuildPlan(makeMavenContext(compiler, env, makeParsedRequest()));

        expect(plan.steps[0].exe).toEqual(MVN_EXE);
        expect(plan.steps[0].args).toContain('-o');
        expect(plan.steps[0].args).toContain(`-Dmaven.repo.local.tail=${path.join(fakeMavenHome, 'repository')}`);
        expect(plan.steps[0].args).toContain('package');
        expect(plan.steps[0].execParams.env.JAVA_HOME).toEqual(fakeJavaHome);
    });

    it('keeps jansi out of the noexec temp directory', async () => {
        const env = makeJavaEnv();
        const compiler = makeJavaCompiler(env);
        const ctx = makeMavenContext(compiler, env, makeParsedRequest());
        const plan = await mavenBuildSystem.getBuildPlan(ctx);

        // Has to be an environment variable: jansi initialises before maven applies its own -D arguments. The path
        // is whatever the build will see -- unsandboxed here, so the real one rather than /app.
        expect(plan.steps[0].execParams.env.MAVEN_OPTS).toEqual(`-Djansi.tmpdir=${path.join(ctx.dirPath, '.jansi')}`);
    });

    it('keeps the environment the user asked for, alongside its own MAVEN_OPTS', async () => {
        const env = makeJavaEnv();
        const compiler = makeJavaCompiler(env);
        const req = makeParsedRequest({overrides: [envOverride({MAVEN_OPTS: '-Duser.language=fr'})]});
        const ctx = makeMavenContext(compiler, env, req);
        const plan = await mavenBuildSystem.getBuildPlan(ctx);

        // Ours first, so the user's wins on a duplicate: the JVM takes the last one.
        expect(plan.steps[0].execParams.env.MAVEN_OPTS).toEqual(
            `-Djansi.tmpdir=${path.join(ctx.dirPath, '.jansi')} -Duser.language=fr`,
        );
    });

    it('says the files are UTF-8, which they are, so maven stops warning about it', async () => {
        const env = makeJavaEnv();
        const compiler = makeJavaCompiler(env);
        const plan = await mavenBuildSystem.getBuildPlan(makeMavenContext(compiler, env, makeParsedRequest()));

        expect(plan.steps[0].args).toContain('-Dproject.build.sourceEncoding=UTF-8');
    });

    it('leaves the encoding alone when the project has set one', async () => {
        const env = makeJavaEnv();
        const compiler = makeJavaCompiler(env);
        const req = makeParsedRequest();
        // A -D would win over the pom, so a project that has chosen an encoding must keep it.
        req.source =
            '<project><properties><project.build.sourceEncoding>ISO-8859-1</project.build.sourceEncoding></properties></project>';
        const plan = await mavenBuildSystem.getBuildPlan(makeMavenContext(compiler, env, req));

        expect(plan.steps[0].args.join(' ')).not.toContain('sourceEncoding');
    });

    it('leaves the goals alone when the user names their own', async () => {
        const env = makeJavaEnv();
        const compiler = makeJavaCompiler(env);
        const req = makeParsedRequest({buildSystemArgs: 'javadoc:jar'});
        const plan = await mavenBuildSystem.getBuildPlan(makeMavenContext(compiler, env, req));

        expect(plan.steps[0].args).toContain('javadoc:jar');
        expect(plan.steps[0].args).not.toContain('package');
    });

    it('leaves the goals alone when the user names a lifecycle phase', async () => {
        const env = makeJavaEnv();
        const compiler = makeJavaCompiler(env);
        const req = makeParsedRequest({buildSystemArgs: 'compile'});
        const plan = await mavenBuildSystem.getBuildPlan(makeMavenContext(compiler, env, req));

        expect(plan.steps[0].args).not.toContain('package');
    });

    it("still names a goal when an argument is a flag of the user's own", async () => {
        // The value of a flag is not a goal, and maven has nothing to do if we take it for one.
        const env = makeJavaEnv();
        const compiler = makeJavaCompiler(env);
        for (const args of ['-P release', '-T 2', '-pl mod', '-f other/pom.xml']) {
            const req = makeParsedRequest({buildSystemArgs: args});
            const plan = await mavenBuildSystem.getBuildPlan(makeMavenContext(compiler, env, req));

            expect(plan.steps[0].args).toContain('package');
        }
    });

    it('says why there is nothing to run rather than naming a file the user never chose', async () => {
        const env = makeJavaEnv();
        const compiler = makeJavaCompiler(env);

        // No classes at all: the jar the rest of the compilation expects is beside the point, and saying it is
        // missing would send the user looking for the wrong thing.
        const nothing = makeMavenContext(compiler, env, makeParsedRequest());
        expect(await mavenBuildSystem.prepareExecution(nothing, 'output.jar')).toEqual({
            cannotRun: 'Nothing to run: maven compiled no classes.',
        });

        // Classes, but none of them a program.
        const dirPath = fsSync.mkdtempSync(path.join(os.tmpdir(), 'ce-maven-run-'));
        try {
            fsSync.mkdirSync(path.join(dirPath, 'target', 'classes'), {recursive: true});
            const noMain = {...makeMavenContext(compiler, env, makeParsedRequest()), dirPath};
            const plan = await mavenBuildSystem.prepareExecution(noMain, path.join(dirPath, 'output.jar'));

            expect('cannotRun' in plan && plan.cannotRun).toMatch(/main\(String\[\]\)/);
        } finally {
            fsSync.rmSync(dirPath, {recursive: true, force: true});
        }
    });

    it('explains that Maven Central is unreachable when resolution fails', () => {
        const offline = 'Cannot access central (https://repo.maven.apache.org/maven2) in offline mode';
        expect(MavenBuildSystem.explainFailure(offline)).toMatch(/cannot download from Maven Central/);
        expect(MavenBuildSystem.explainFailure('BUILD FAILURE: compilation error')).toBeUndefined();
    });
});

describe('Make build system', () => {
    function makeMakeEnv(): CompilationEnvironment {
        return makeCompilationEnvironment({
            languages: {'c++': {id: 'c++'}},
            props: {make: '/usr/bin/make'},
        });
    }

    function makeCppCompiler(env: CompilationEnvironment): BaseCompiler {
        return new BaseCompiler(makeFakeCompilerInfo({exe: '/usr/bin/g++', lang: 'c++', ldPath: [], libPath: []}), env);
    }

    function makeMakeContext(compiler: BaseCompiler, env: CompilationEnvironment, req: ParsedRequest): BuildContext {
        const dirPath = '/tmp/ce-fake-make';
        return {
            compiler,
            env,
            dirPath,
            buildPath: makeBuildSystem.getBuildPath(dirPath),
            key: compiler.getBuildProjectCacheKey(makeBuildSystem, req, []),
            parsedRequest: req,
            files: [],
            libsAndOptions: {libraries: req.libraries, options: req.options},
            toolchainPath: undefined,
            buildSystemArgs: getBuildSystemArgs(req.backendOptions),
        };
    }

    it('builds where the Makefile is, there being nowhere else it could', () => {
        expect(makeBuildSystem.descriptor.manifestFilename).toEqual('Makefile');
        expect(makeBuildSystem.getBuildPath('/tmp/project')).toEqual('/tmp/project');
    });

    it('is offered for anything, a Makefile saying for itself what to run', async () => {
        const env = makeMakeEnv();
        expect(await makeBuildSystem.getUnsupportedReason(makeCppCompiler(env))).toBeUndefined();
        expect(BuildSystems.make.compatibleLanguageIds).toEqual('all');
    });

    it('hands the Makefile the compiler that was selected, and the options chosen with it', async () => {
        const env = makeMakeEnv();
        const compiler = makeCppCompiler(env);
        const req = makeParsedRequest();
        req.options = ['-O2'];
        const plan = await makeBuildSystem.getBuildPlan(makeMakeContext(compiler, env, req));

        // What a recipe saying `$(CXX) $(CXXFLAGS)` picks up.
        expect(plan.steps[0].exe).toEqual('/usr/bin/make');
        expect(plan.steps[0].execParams.env.CXXFLAGS).toContain('-O2');
        expect(plan.steps[0].execParams.env.LDFLAGS).toBeDefined();
    });

    it('passes on the targets and arguments asked for, and invents none', async () => {
        const env = makeMakeEnv();
        const compiler = makeCppCompiler(env);
        const withArgs = makeParsedRequest({buildSystemArgs: '-j4 all'});
        const plan = await makeBuildSystem.getBuildPlan(makeMakeContext(compiler, env, withArgs));
        expect(plan.steps[0].args).toEqual(['-j4', 'all']);

        const bare = await makeBuildSystem.getBuildPlan(makeMakeContext(compiler, env, makeParsedRequest()));
        expect(bare.steps[0].args).toEqual([]);
    });

    it('names nvcc for a Makefile that asks for it, and only when it really is nvcc', async () => {
        const env = makeCompilationEnvironment({languages: {cuda: {id: 'cuda'}}, props: {make: '/usr/bin/make'}});
        const asCuda = (compilerType: string) =>
            new BaseCompiler(
                makeFakeCompilerInfo({
                    exe: '/opt/compiler-explorer/cuda/12.8.1/bin/nvcc',
                    lang: 'cuda',
                    compilerType,
                    ldPath: [],
                    libPath: [],
                }),
                env,
            );

        const nvcc = await makeBuildSystem.getBuildPlan(makeMakeContext(asCuda('nvcc'), env, makeParsedRequest()));
        expect(nvcc.steps[0].execParams.env.NVCC).toEqual('/opt/compiler-explorer/cuda/12.8.1/bin/nvcc');

        // clang compiles CUDA too, and it is not nvcc.
        const clang = await makeBuildSystem.getBuildPlan(
            makeMakeContext(asCuda('clang-cuda'), env, makeParsedRequest()),
        );
        expect(clang.steps[0].execParams.env.NVCC).toBeUndefined();
    });

    it('says so when the Makefile built nothing by the name we were told to look for', async () => {
        const env = makeMakeEnv();
        const compiler = makeCppCompiler(env);
        const ctx = makeMakeContext(compiler, env, makeParsedRequest());

        const nothingToInspect = await makeBuildSystem.finaliseArtifact(
            ctx,
            {code: 0, timedOut: false, stdout: [], stderr: []},
            path.join(ctx.dirPath, 'output'),
        );
        expect(nothingToInspect).toMatch(/Nothing was built at output/);
        expect(nothingToInspect).toMatch(/output file box/);
    });

    it('looks for what the project asked to inspect, and otherwise for something conventional', () => {
        const env = makeMakeEnv();
        const compiler = makeCppCompiler(env);
        const named = makeMakeContext(compiler, env, makeParsedRequest({customOutputFilename: 'demo'}));
        expect(makeBuildSystem.getArtifactFilename(named)).toEqual(path.join('/tmp/ce-fake-make', 'demo'));

        const unnamed = makeMakeContext(compiler, env, makeParsedRequest());
        expect(makeBuildSystem.getArtifactFilename(unnamed)).toEqual(path.join('/tmp/ce-fake-make', 'output.s'));
    });

    it('gives up on a build that waited too long, but never on one the cache already answered', async () => {
        const env = makeMakeEnv();
        const compiler = makeCppCompiler(env);

        class Fails extends MakeBuildSystem {
            override async writeProjectFiles(): Promise<{inputFilename: string}> {
                throw new Error('far enough');
            }
        }

        const enqueued = vi.spyOn(env, 'enqueue');

        // Nothing cached: a compiler has to run, so waiting too long for a slot is worth giving up on.
        await compiler.buildProject(new Fails(), [], makeParsedRequest(), BypassCache.None);
        expect(enqueued.mock.calls.at(-1)?.[1]).toEqual({abandonIfStale: true});

        // Cached: the result was there before the wait began, so there is nothing to be too late for.
        vi.spyOn(compiler, 'fetchExecutablePackage').mockResolvedValue(Buffer.from('not really a package'));
        await compiler.buildProject(new Fails(), [], makeParsedRequest(), BypassCache.None);
        expect(enqueued.mock.calls.at(-1)?.[1]).toEqual({abandonIfStale: false});

        vi.restoreAllMocks();
    });

    it('reports a build that could not be set up, instead of post-processing nothing', async () => {
        const env = makeMakeEnv();
        const compiler = makeCppCompiler(env);

        let dirPath = '';
        class FailsToWrite extends MakeBuildSystem {
            override async writeProjectFiles(ctx: BuildContext): Promise<{inputFilename: string}> {
                dirPath = ctx.dirPath;
                throw new Error('could not write the project');
            }
        }

        const result = await compiler.buildProject(
            new FailsToWrite(),
            [],
            makeParsedRequest(),
            BypassCache.Compilation,
        );

        // A project build's assembly is read from the nested result. Without one, post-processing threw a TypeError
        // over the top of the real error, and left the build directory behind because it threw before the cleanup.
        expect(result.result?.asm).toEqual([{text: '<could not write the project>'}]);
        expect(result.code).toEqual(-1);
        expect(dirPath).not.toEqual('');
        expect(fsSync.existsSync(dirPath)).toBe(false);
    });

    it('reports a missing launcher, which is thrown while working out what to run', async () => {
        // maven defaults to empty and asserts, so this has to arrive as a compilation result rather than a bare 400.
        const env = makeMakeEnv();
        const compiler = makeCppCompiler(env);

        let dirPath = '';
        class NoLauncher extends MakeBuildSystem {
            override async getBuildPlan(ctx: BuildContext): Promise<BuildPlan> {
                dirPath = ctx.dirPath;
                throw new Error('No make command found');
            }
        }

        const result = await compiler.buildProject(new NoLauncher(), [], makeParsedRequest(), BypassCache.Compilation);

        expect(result.result?.asm).toEqual([{text: '<No make command found>'}]);
        expect(fsSync.existsSync(dirPath)).toBe(false);
    });

    it('points at the output file box when the build named its artifact something else', async () => {
        const env = makeMakeEnv();
        const compiler = makeCppCompiler(env);
        const ctx = makeMakeContext(compiler, env, makeParsedRequest());

        const plan = await makeBuildSystem.prepareExecution(ctx, path.join(ctx.dirPath, 'output.s'));

        expect('cannotRun' in plan && plan.cannotRun).toMatch(/no output\.s/);
        expect('cannotRun' in plan && plan.cannotRun).toMatch(/output file box/);
    });

    it('refuses an output filename that leaves the project, whatever it is copied from', () => {
        const env = makeMakeEnv();
        const compiler = makeCppCompiler(env);
        // The second leaves the project via a sibling whose name starts with the project directory's.
        for (const outside of ['../../../../etc/passwd', '../ce-fake-make-evil/output']) {
            const ctx = makeMakeContext(compiler, env, makeParsedRequest({customOutputFilename: outside}));
            expect(() => makeBuildSystem.getArtifactFilename(ctx)).toThrow(Error);
        }
    });
});
