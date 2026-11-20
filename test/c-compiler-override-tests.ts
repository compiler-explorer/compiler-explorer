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

import {afterAll, beforeAll, describe, expect, it, vi} from 'vitest';

import {BaseCompiler} from '../lib/base-compiler.js';
import {CompilationEnvironment} from '../lib/compilation-env.js';
import {NexiumCompiler} from '../lib/compilers/nexium.js';
import {SailCompiler} from '../lib/compilers/sail.js';
import {ValaCompiler} from '../lib/compilers/vala.js';
import {getPossibleCCompilersFromCompilerInfo, getTargetFromCompilerCommand} from '../lib/toolchain-utils.js';
import {ExecutionOptions} from '../types/compilation/compilation.interfaces.js';
import {CompilerOverrideType, ConfiguredOverrides} from '../types/compilation/compiler-overrides.interfaces.js';
import {PreliminaryCompilerInfo} from '../types/compiler.interfaces.js';
import {LanguageKey} from '../types/languages.interfaces.js';
import {makeCompilationEnvironment, makeFakeCompilerInfo} from './utils.js';

const gcc = '/opt/compiler-explorer/gcc-14.2.0/bin/gcc';
const clang = '/opt/compiler-explorer/clang-19.1.0/bin/clang --gcc-toolchain=/opt/compiler-explorer/gcc-13.2.0';
const armClang =
    '/opt/compiler-explorer/clang-19.1.0/bin/clang -target aarch64-linux-gnu ' +
    '--gcc-toolchain=/opt/compiler-explorer/arm64/gcc-14.2.0 --sysroot=/opt/compiler-explorer/arm64/sysroot';

function cCompiler(props: Partial<PreliminaryCompilerInfo>): PreliminaryCompilerInfo {
    return {
        lang: 'c',
        name: props.id,
        exe: '/opt/compiler-explorer/gcc-14.2.0/bin/gcc',
        options: '',
        instructionSet: 'amd64',
        compilerCategories: ['gcc'],
        semver: '14.2.0',
        hidden: false,
        ...props,
    } as PreliminaryCompilerInfo;
}

describe('getTargetFromCompilerCommand', () => {
    it('reads the target in each of its spellings', () => {
        expect(getTargetFromCompilerCommand('clang -target aarch64-linux-gnu -O2')).toEqual('aarch64-linux-gnu');
        expect(getTargetFromCompilerCommand('clang --target aarch64-linux-gnu')).toEqual('aarch64-linux-gnu');
        expect(getTargetFromCompilerCommand('clang --target=riscv64-unknown-linux-gnu')).toEqual(
            'riscv64-unknown-linux-gnu',
        );
    });

    it('finds no target in a native command, or a -target without a value', () => {
        expect(getTargetFromCompilerCommand(clang)).toBeUndefined();
        expect(getTargetFromCompilerCommand('clang -target')).toBeUndefined();
        expect(getTargetFromCompilerCommand('')).toBeUndefined();
    });
});

describe('getPossibleCCompilersFromCompilerInfo', () => {
    const names = (compilers: PreliminaryCompilerInfo[]) =>
        getPossibleCCompilersFromCompilerInfo(compilers).map(option => option.name);

    it('offers x86-64 gcc and clang, with their configured options', () => {
        const options = getPossibleCCompilersFromCompilerInfo([
            cCompiler({id: 'gcc'}),
            cCompiler({
                id: 'clang',
                exe: '/opt/compiler-explorer/clang-19.1.0/bin/clang',
                options: '--gcc-toolchain=/opt/compiler-explorer/gcc-13.2.0',
                compilerCategories: ['clang'],
                semver: '19.1.0',
            }),
        ]);
        expect(options).toEqual([
            {name: 'gcc', value: gcc},
            {name: 'clang', value: clang},
        ]);
    });

    it('leaves out gcc older than 4.7, which has no -std=gnu11', () => {
        expect(names([cCompiler({id: 'old', semver: '4.6.4'}), cCompiler({id: 'new', semver: '4.7.1'})])).toEqual([
            'new',
        ]);
    });

    it('offers a cross clang only with a target and a sysroot, and no cross gcc', () => {
        const cross = {instructionSet: 'aarch64' as const};
        expect(
            names([
                cCompiler({id: 'arm-gcc', ...cross}),
                cCompiler({
                    id: 'arm-clang',
                    ...cross,
                    compilerCategories: ['clang'],
                    options: '-target aarch64-linux-gnu --sysroot=/sysroot',
                }),
                cCompiler({id: 'bare-clang', ...cross, compilerCategories: ['clang'], options: '-target mips-elf'}),
            ]),
        ).toEqual(['arm-clang']);
    });

    it('runs zig cc through its cc subcommand, whose instruction set is not known yet', () => {
        const options = getPossibleCCompilersFromCompilerInfo([
            cCompiler({
                id: 'zigcc',
                exe: '/opt/compiler-explorer/zig-0.14.1/zig',
                compilerType: 'zigcc',
                compilerCategories: [],
                instructionSet: null,
            }),
        ]);
        expect(options).toEqual([{name: 'zigcc', value: '/opt/compiler-explorer/zig-0.14.1/zig cc'}]);
    });

    it('leaves out other languages, other C compilers and those that cannot run here', () => {
        expect(
            names([
                cCompiler({id: 'cpp', lang: 'c++' as LanguageKey}),
                cCompiler({id: 'tcc', compilerCategories: []}),
                cCompiler({id: 'hidden', hidden: true}),
                cCompiler({id: 'remote', remote: {target: 'x', path: 'y', cmakePath: 'z', basePath: '/'}}),
                cCompiler({id: 'relative', exe: 'gcc'}),
            ]),
        ).toEqual([]);
    });

    it('offers a command once', () => {
        expect(names([cCompiler({id: 'first'}), cCompiler({id: 'alias'})])).toEqual(['first']);
    });
});

describe('C compiler override', () => {
    let ce: CompilationEnvironment;
    const choices = [
        {name: 'gcc', value: gcc},
        {name: 'clang', value: clang},
        {name: 'arm clang', value: armClang},
    ];

    beforeAll(() => {
        ce = makeCompilationEnvironment({
            languages: {
                nexium: {id: 'nexium' as LanguageKey},
                vala: {id: 'vala' as LanguageKey},
                sail: {id: 'sail' as LanguageKey},
            },
            props: {cCompiler: '/usr/bin/gcc'},
        });
        ce.setPossibleCCompilers(choices);
        // the other overrides ask the compiler itself, which these tests do not run
        vi.spyOn(BaseCompiler.prototype, 'getTargetsAsOverrideValues').mockResolvedValue([]);
        vi.spyOn(BaseCompiler.prototype, 'getPossibleStdversAsOverrideValues').mockResolvedValue([]);
    });

    afterAll(() => {
        vi.restoreAllMocks();
    });

    const chose = (value: string): ConfiguredOverrides => [{name: CompilerOverrideType.c_compiler, value}];
    const offered = (compiler: NexiumCompiler | ValaCompiler | SailCompiler) =>
        compiler.getInfo().possibleOverrides?.find(override => override.name === CompilerOverrideType.c_compiler);
    const make = <T>(Compiler: new (info: any, env: CompilationEnvironment) => T, lang: string): T =>
        new Compiler(makeFakeCompilerInfo({exe: '/dev/null', lang: lang as LanguageKey, possibleOverrides: []}), ce);

    describe('Nexium', () => {
        it('offers cross compilers, its llvm-objdump reading any architecture', async () => {
            const nexium = make(NexiumCompiler, 'nexium');
            await nexium.populatePossibleOverrides();
            expect(offered(nexium)).toMatchObject({type: 'options', values: choices});
        });

        it('passes the chosen command to nx with --cc', async () => {
            const nexium = make(NexiumCompiler, 'nexium');
            await nexium.populatePossibleOverrides();
            expect(nexium.changeOptionsBasedOnOverrides(['build'], chose(clang))).toEqual(['build', '--cc', clang]);
        });

        it('tells nx the target of a cross compiler, or nx asks it for -march=x86-64', async () => {
            const nexium = make(NexiumCompiler, 'nexium');
            await nexium.populatePossibleOverrides();
            expect(nexium.changeOptionsBasedOnOverrides(['build'], chose(armClang))).toEqual([
                'build',
                '--cc',
                armClang,
                '--target',
                'aarch64-linux-gnu',
            ]);
        });

        it('ignores a command it did not offer', async () => {
            const nexium = make(NexiumCompiler, 'nexium');
            await nexium.populatePossibleOverrides();
            expect(nexium.changeOptionsBasedOnOverrides(['build'], chose('/bin/sh -c id'))).toEqual(['build']);
        });
    });

    for (const [lang, Compiler] of [
        ['vala', ValaCompiler],
        ['sail', SailCompiler],
    ] as const) {
        describe(lang, () => {
            it('offers only native compilers, its objdump and libraries being the host’s', async () => {
                const compiler = make<ValaCompiler | SailCompiler>(Compiler, lang);
                await compiler.populatePossibleOverrides();
                expect(offered(compiler)).toMatchObject({values: [choices[0], choices[1]]});
            });

            it('hands the chosen command over as CC', async () => {
                const compiler = make<ValaCompiler | SailCompiler>(Compiler, lang);
                await compiler.populatePossibleOverrides();
                const execOptions: ExecutionOptions = {env: {}};
                compiler.applyOverridesToExecOptions(execOptions, chose(clang));
                expect(execOptions.env?.CC).toEqual(clang);
            });

            it('ignores a command it did not offer', async () => {
                const compiler = make<ValaCompiler | SailCompiler>(Compiler, lang);
                await compiler.populatePossibleOverrides();
                const execOptions: ExecutionOptions = {env: {}};
                compiler.applyOverridesToExecOptions(execOptions, chose(armClang));
                expect(execOptions.env?.CC).toBeUndefined();
            });
        });
    }
});
