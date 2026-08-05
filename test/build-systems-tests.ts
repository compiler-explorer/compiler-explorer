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
import type {BuildContext} from '../lib/build-systems/index.js';
import {cmakeBuildSystem, getBuildSystem, getBuildSystemArgs} from '../lib/build-systems/index.js';
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

    it('does not resolve unknown build systems', () => {
        expect(isBuildSystemId('cargo')).toBe(false);
        expect(getBuildSystem('cargo')).toBeUndefined();
        expect(getBuildSystem('toString')).toBeUndefined();
    });

    it('offers cmake only for the languages it can build', () => {
        expect(getBuildSystemsForLanguage('c++').map(bs => bs.id)).toEqual(['cmake']);
        expect(getBuildSystemsForLanguage('rust')).toEqual([]);
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

    it('refuses compilers that cannot produce binaries', () => {
        const env = makeEnv();
        expect(cmakeBuildSystem.getUnsupportedReason(makeCompiler(env))).toBeUndefined();
        expect(cmakeBuildSystem.getUnsupportedReason(makeCompiler(env, {supportsBinary: false}))).toEqual(
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
        expect(plan.steps[0].args).toEqual([
            '-DCMAKE_CXX_COMPILER_EXTERNAL_TOOLCHAIN=/usr',
            '-DCMAKE_BUILD_TYPE=Debug',
            '..',
        ]);
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

        expect(plan.steps[0].args).toEqual([
            '-DCMAKE_CXX_COMPILER_EXTERNAL_TOOLCHAIN=/usr',
            '-DSOMETHING=1',
            '-DUSER=2',
            '..',
        ]);
    });
});
