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

import {describe, expect, it} from 'vitest';

import {BaseCompiler} from '../lib/base-compiler.js';
import {CargoBuildSystem} from '../lib/build-systems/cargo.js';
import type {BuildContext} from '../lib/build-systems/index.js';
import {cargoBuildSystem, cmakeBuildSystem, getBuildSystem, getBuildSystemArgs} from '../lib/build-systems/index.js';
import {CompilationEnvironment} from '../lib/compilation-env.js';
import {ParsedRequest} from '../lib/handlers/compile.js';
import {getBuildSystemsForLanguage, isBuildSystemId} from '../shared/build-systems.js';
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

    it('does not resolve unknown build systems', () => {
        expect(isBuildSystemId('maven')).toBe(false);
        expect(getBuildSystem('maven')).toBeUndefined();
        expect(getBuildSystem('toString')).toBeUndefined();
    });

    it('offers each build system only for the languages it can build', () => {
        expect(getBuildSystemsForLanguage('c++').map(bs => bs.id)).toEqual(['cmake']);
        expect(getBuildSystemsForLanguage('rust').map(bs => bs.id)).toEqual(['cargo']);
        expect(getBuildSystemsForLanguage('haskell')).toEqual([]);
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
            libsAndOptions: {libraries: [], options: []},
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
        expect(CargoBuildSystem.getCargoPath(compiler)).toEqual('/opt/compiler-explorer/rust-1.91.0/bin/cargo');
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
        expect(plan.steps[0].exe).toEqual('/opt/compiler-explorer/rust-1.91.0/bin/cargo');
        expect(plan.steps[0].args).toEqual(['build', '--offline', '--message-format=json-render-diagnostics']);
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
        expect(plan.steps[0].execParams.env.RUSTFLAGS).toEqual('-C opt-level=3');
    });

    it('maps the sandbox paths cargo reports back to real ones', () => {
        // CE bind-mounts the project at /app, so this is what cargo actually reports during a real compilation.
        expect(CargoBuildSystem.toHostPath('/app/target/debug/output', '/tmp/ce-build')).toEqual(
            '/tmp/ce-build/target/debug/output',
        );
        // Run without a sandbox, cargo reports the real temp directory instead, which rebases to the same place.
        expect(
            CargoBuildSystem.toHostPath('/tmp/compiler-explorer-compilerXYZ/target/debug/output', '/tmp/ce-build'),
        ).toEqual('/tmp/ce-build/target/debug/output');
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
        expect(artifact).toEqual(path.join('/tmp/ce-fake-cargo', 'output'));

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
