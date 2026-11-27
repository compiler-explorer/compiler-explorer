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

import {readFileSync} from 'node:fs';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';

import {afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi} from 'vitest';

import {MachParser} from '../lib/compilers/argument-parsers.js';
import {MachCompiler} from '../lib/compilers/mach.js';
import {logger} from '../lib/logger.js';
import {LlvmObjdumper} from '../lib/objdumper/llvm.js';
import {AsmParser} from '../lib/parsers/asm-parser.js';
import {MACH_MODULE_SEP, MachAsmParser} from '../lib/parsers/asm-parser-mach.js';
import {toEditorColumns} from '../lib/parsers/mach-diagnostics.js';
import {MachIrParser} from '../lib/parsers/mach-ir.js';
import {parseProperties} from '../lib/properties.js';
import * as utils from '../lib/utils.js';
import {unwrap} from '../shared/assert.js';
import {CompilerOverrideType} from '../types/compilation/compiler-overrides.interfaces.js';
import {LanguageKey} from '../types/languages.interfaces.js';
import {makeCompilationEnvironment, makeFakeCompilerInfo, makeFakeParseFiltersAndOutputOptions} from './utils.js';

const languages = {
    mach: {id: 'mach' as LanguageKey, extensions: ['.mach']},
};

// `mach info targets` as mach 7.2.0 prints it.
const infoTargets = [
    'target                 env                                            abi                            code_model      linkage                                            note',
    'linux-x86_64           *none,gnu                                      *sysv64                        *small          *static gnu:*pie,dynamic,static',
    'linux-aarch64          *none,gnu                                      *aapcs64                       *small          *static gnu:*pie,dynamic,static',
    'linux-riscv64          *none,gnu                                      *lp64d,lp64,lp64f              *medany         *static gnu:*pie,dynamic,static',
    'linux-riscv32          *none,gnu                                      ilp32,ilp32f,ilp32d            *medany         *static gnu:*pie,dynamic,static                    needs --isa',
    'darwin-x86_64          *macos                                         *sysv64                        *small          macos:*pie,dynamic',
    'darwin-aarch64         *macos                                         *darwinpcs                     *small          macos:*pie',
    'windows-x86_64         *msvc                                          *win64                         *small          msvc:*pie,dynamic',
    'windows-aarch64        *msvc                                          *aapcs64win                    *small          msvc:*pie',
    'windows-arm64ec        *msvc                                          *arm64ec                       *small          msvc:*pie',
    'uefi-x86_64            *none                                          *win64                         *small          *pie',
    'uefi-aarch64           *none                                          *aapcs64win                    *small          *pie',
    'freestanding-x86_64    *none                                          *sysv64,win64                  *small          *static                                            needs --isa',
    'freestanding-aarch64   *none                                          *aapcs64,darwinpcs,aapcs64win  *small          *static                                            needs --isa',
    'freestanding-riscv64   *none                                          lp64,lp64f,lp64d               *medany,medlow  *static                                            needs --isa',
    'freestanding-riscv32   *none                                          ilp32,ilp32f,ilp32d            *medlow         *static                                            needs --isa',
    'freestanding-riscv32e  *none                                          ilp32e                         *medlow         *static                                            needs --isa',
    'freestanding-wasm32    *none                                          *basic                         -               *static',
    'wasi-wasm32            *preview1                                      *basic                         -               preview1:*static',
    'vulkan-spirv           *none,vulkan1.0,vulkan1.1,vulkan1.2,vulkan1.3  *shader                        -               - vulkan1.0:- vulkan1.1:- vulkan1.2:- vulkan1.3:-',
    '',
].join('\n');

const helpBuild = [
    'usage: mach build [options] <path|entry.mach> [inputs...]',
    '',
    'compile a project or one entry file',
    '',
    'options:',
    '  -h, --help            show help',
    '      --report <human|json|lines>',
    '                        report as human text, NDJSON records or tab-separated record lines',
    '                        (default: human)',
    '  -t, --target <pattern>...',
    "                        which targets: a name or glob of the manifest's targets in a project, a",
    "                        <sys>-<arch> in an entry build (default: the host's)",
    '  -v, --verbose...      show each phase as it closes, -vv each module too',
    "  -O, --[no-]optimize   optimize, or not with --no-optimize (default: the profile's)",
    "      --[no-]debug      emit debug information, or not with --no-debug (default: the profile's)",
    '      --emit <asm|ir|ir-opt|obj|exe>...',
    '                        a product to write, once each (default: exe)',
    '  -k, --kind <bin|static|shared|object|module>',
    '                        the product to make (default: bin)',
    "      --isa <selection> the instruction set (default: the target's)",
    '',
].join('\n');

/**
 * Reproduces what mach 7.2.0 with mach-std 10.1.1 answers for each probe build. Objects build for every tuple except
 * `windows-arm64ec`, where `std.runtime.windows` and `std.system.cpu.features` reject `arm64ec`, and any `needs --isa`
 * tuple invoked without `--isa`. A listing is refused on Wasm and SPIR-V, which have none, and fails on the
 * freestanding tuples with a Windows ABI.
 */
function fakeProbe(args: string[]) {
    const tIdx = args.indexOf('-t');
    const platform = tIdx !== -1 ? args[tIdx + 1] : '';
    const hasIsa = args.includes('--isa');
    const entry = args[1];

    if (platform === 'windows-arm64ec' && entry.startsWith('hosted/')) {
        return {
            code: 1,
            stdout: '',
            stderr: 'error[comptime.user_error] std.runtime.windows supports x86_64 and aarch64',
        };
    }
    if ((platform.startsWith('freestanding-') && !platform.endsWith('wasm32')) || platform === 'linux-riscv32') {
        if (!hasIsa) {
            return {
                code: 1,
                stdout: '',
                stderr: `error[target.isa_required] ${platform} sets no ISA floor; pass --isa`,
            };
        }
    }
    const emit = args[args.indexOf('--emit') + 1];
    if (emit === 'asm' && (platform.endsWith('-wasm32') || platform === 'vulkan-spirv')) {
        return {
            code: 1,
            stdout: '',
            stderr: "error[cli.flag_inapplicable] '--emit asm' does not apply to this target",
        };
    }
    const abi = args.includes('--abi') ? args[args.indexOf('--abi') + 1] : '';
    if (emit === 'asm' && platform.startsWith('freestanding-') && ['win64', 'aapcs64win'].includes(abi)) {
        return {
            code: 1,
            stdout: '',
            stderr: 'error[compiler.internal] masc cannot list the object it made',
        };
    }
    if (platform === 'vulkan-spirv' && !args.includes('module')) {
        return {
            code: 1,
            stdout: '',
            stderr: 'error[link.unsupported] vulkan-spirv requires -k module',
        };
    }
    return {code: 0, stdout: '', stderr: ''};
}

async function makeStd() {
    const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'ce-mach-std'));
    await fs.writeFile(path.join(dir, 'mach.toml'), '[project]\nid = "std"\n');
    return dir;
}

function start(compiler: MachCompiler) {
    return compiler.initialise(new Date(), {libs: {}} as any, true);
}

function fakeMach(compiler: MachCompiler) {
    vi.spyOn(compiler, 'exec').mockImplementation(async (_exe, args) => {
        if (args[0] === 'info' && args[1] === 'targets') return {code: 0, stdout: infoTargets, stderr: ''} as any;
        if (args[0] === 'help' && args[1] === 'build') return {code: 0, stdout: helpBuild, stderr: ''} as any;
        return fakeProbe(args) as any;
    });
}

function makeMach(stdPath: string) {
    return new MachCompiler(
        makeFakeCompilerInfo({
            id: 'mach',
            exe: '/opt/compiler-explorer/mach-7.2.0/mach',
            lang: 'mach',
            libsArr: [],
            objdumper: '/usr/bin/objdump',
            objdumperType: 'default',
            llvmObjdumper: '/usr/bin/llvm-objdump',
            supportsBinary: true,
            supportsBinaryObject: true,
        }),
        makeCompilationEnvironment({languages, props: {'compiler.mach.stdPath': stdPath}}),
    );
}

describe('Mach targets and entry build', () => {
    let compiler: MachCompiler;
    let stdPath: string;
    let startupExecArgs: string[][];

    beforeAll(async () => {
        stdPath = await makeStd();
        compiler = makeMach(stdPath);
        fakeMach(compiler);
        expect(await start(compiler)).toBe(compiler);
        startupExecArgs = (compiler.exec as any).mock.calls.map((call: any[]) => call[1]);
    });

    afterAll(async () => {
        await fs.rm(stdPath, {recursive: true, force: true});
    });

    it('offers every tuple that builds under its probe, disambiguated by ABI where needed', () => {
        expect(compiler.targets().map(t => t.name)).toEqual([
            'linux-x86_64',
            'linux-aarch64',
            'linux-riscv64-lp64d',
            'linux-riscv64-lp64',
            'linux-riscv64-lp64f',
            'linux-riscv32-ilp32',
            'linux-riscv32-ilp32f',
            'linux-riscv32-ilp32d',
            'darwin-x86_64',
            'darwin-aarch64',
            'windows-x86_64',
            'windows-aarch64',
            'uefi-x86_64',
            'uefi-aarch64',
            'freestanding-x86_64-sysv64',
            'freestanding-x86_64-win64',
            'freestanding-aarch64-aapcs64',
            'freestanding-aarch64-darwinpcs',
            'freestanding-aarch64-aapcs64win',
            'freestanding-riscv64-lp64',
            'freestanding-riscv64-lp64f',
            'freestanding-riscv64-lp64d',
            'freestanding-riscv32-ilp32',
            'freestanding-riscv32-ilp32f',
            'freestanding-riscv32-ilp32d',
            'freestanding-riscv32e',
            'freestanding-wasm32',
            'wasi-wasm32',
            'vulkan-spirv',
        ]);
    });

    it('reads the objects of a target the compiler writes no listing for', () => {
        expect(
            compiler
                .targets()
                .filter(t => !t.listing)
                .map(t => t.name),
        ).toEqual([
            'freestanding-x86_64-win64',
            'freestanding-aarch64-aapcs64win',
            'freestanding-wasm32',
            'wasi-wasm32',
            'vulkan-spirv',
        ]);
    });

    it('probes every tuple mach info targets lists, for its listing first and then for its objects', async () => {
        expect((await compiler.supportedTuples()).length).toEqual(30);
        const emits = startupExecArgs.filter(args => args[0] === 'build').map(args => args[args.indexOf('--emit') + 1]);
        expect(emits.filter(emit => emit === 'asm')).toHaveLength(30);
        // the five without a listing, and windows-arm64ec, which builds neither way
        expect(emits.filter(emit => emit === 'obj')).toHaveLength(6);
        expect(startupExecArgs.filter(args => args[0] !== 'build')).toEqual([['info', 'targets']]);
    });

    it('reuses cachedMachTargets on prediscovered starts', async () => {
        expect(compiler.getInfo().cachedMachTargets).toEqual(compiler.targets());

        const prediscovered = makeMach(stdPath);
        prediscovered.getInfo().cachedMachTargets = compiler.targets();
        const exec = vi.spyOn(prediscovered, 'exec');
        expect(await start(prediscovered)).toBe(prediscovered);
        expect(exec).not.toHaveBeenCalled();
        expect(prediscovered.targets()).toEqual(compiler.targets());
    });

    it('takes std from beside the executable by default, or from compiler.<id>.stdPath', () => {
        const env = makeCompilationEnvironment({languages});
        const installed = new MachCompiler(
            makeFakeCompilerInfo({id: 'mach720', exe: '/opt/compiler-explorer/mach-7.2.0/mach', lang: 'mach'}),
            env,
        );
        expect(installed.stdPath).toEqual(path.join('/opt/compiler-explorer/mach-7.2.0', 'std'));

        const devEnv = makeCompilationEnvironment({
            languages,
            props: {'compiler.machdev.stdPath': '/src/mach/dep/std'},
        });
        const dev = new MachCompiler(makeFakeCompilerInfo({id: 'machdev', exe: '/usr/bin/mach', lang: 'mach'}), devEnv);
        expect(dev.stdPath).toEqual('/src/mach/dep/std');
    });

    it('refuses to offer any target when the compiler has no std', async () => {
        const env = makeCompilationEnvironment({languages});
        const bare = new MachCompiler(
            makeFakeCompilerInfo({
                id: 'machbare',
                exe: path.join(os.tmpdir(), 'ce-no-such-mach', 'mach'),
                lang: 'mach',
            }),
            env,
        );
        const error = vi.spyOn(logger, 'error').mockImplementation(() => logger);
        expect(await start(bare)).toBeNull();
        expect(String((error.mock.calls[0] as unknown[])[1])).toContain('set compiler.machbare.stdPath');
        error.mockRestore();
        expect(() => bare.targets()).toThrow('targets read before initialise() probed them');
    });

    it('builds the entry directly with --source bindings, expanding target ABI, ISA floor, and module kind', () => {
        const root = path.join('/tmp', 'ce');
        const input = path.join(root, 'src', 'example.mach');

        expect(compiler.orderArguments(['--emit', 'asm'], input, [], [], [], [], ['-O'], [])).toEqual([
            'build',
            'src/example.mach',
            '--source',
            'example=src',
            '--source',
            `std=${stdPath}`,
            '-w',
            'out',
            '--report',
            'json',
            '--debug',
            '-t',
            expect.stringMatching(/^(linux|darwin|windows)-(x86_64|aarch64|riscv64)$/),
            '--emit',
            'asm',
            '-O',
        ]);

        // non-default ABI + needs --isa floor
        expect(
            compiler.orderArguments(
                ['--target', 'freestanding-aarch64-darwinpcs', '--emit', 'asm'],
                input,
                [],
                [],
                [],
                [],
                [],
                [],
            ),
        ).toContain('--abi');
        const fsArgs = compiler.orderArguments(
            ['--target', 'freestanding-aarch64-darwinpcs', '--emit', 'asm'],
            input,
            [],
            [],
            [],
            [],
            [],
            [],
        );
        expect(fsArgs.slice(fsArgs.indexOf('-t'))).toEqual([
            '-t',
            'freestanding-aarch64',
            '--abi',
            'darwinpcs',
            '--isa',
            'armv8-a+fp+simd',
            '--emit',
            'asm',
        ]);

        // user-supplied --isa and --no-debug override defaults cleanly
        const customIsa = compiler.orderArguments(
            ['--target', 'freestanding-x86_64-sysv64', '--emit', 'asm'],
            input,
            [],
            [],
            [],
            [],
            ['--isa', 'x86-64,avx2', '--no-debug'],
            [],
        );
        expect(customIsa).not.toContain('--debug');
        expect(customIsa.filter(arg => arg === '--isa')).toHaveLength(1);
        expect(customIsa).toContain('x86-64,avx2');

        // vulkan-spirv injects `-k module` unless user already passed `-k`
        const spvArgs = compiler.orderArguments(
            ['--target', 'vulkan-spirv', '--emit', 'obj'],
            input,
            [],
            [],
            [],
            [],
            [],
            [],
        );
        expect(spvArgs.slice(spvArgs.indexOf('-t'))).toEqual(['-t', 'vulkan-spirv', '-k', 'module', '--emit', 'obj']);
    });

    it('reads a listing on CPU targets and the objects on Wasm, SPIR-V and binaryObject', () => {
        const root = path.join('/tmp', 'ce');
        const listing = path.join(root, 'out', 'asm', 'example', 'example.s');
        const object = path.join(root, 'out', 'obj', 'example', 'example.o');

        // the filters as a request for `target` reaches the compiler with them, and the key they are part of
        const request = (
            target: string,
            asked: {binary?: boolean; binaryObject?: boolean} = {},
            options: string[] = [],
        ) => {
            const backendOptions = {overrides: [{name: CompilerOverrideType.arch, value: target}]};
            const filters = makeFakeParseFiltersAndOutputOptions(asked);
            compiler.fixFiltersForCompiler(filters, options, [], backendOptions);
            const key = {compiler: compiler.getInfo(), source: '', options, backendOptions, filters} as any;
            return {filters, output: compiler.getOutputFilename(root, 'output', key)};
        };

        const cpu = request('linux-x86_64');
        expect(cpu.filters.binaryObject).toBeFalsy();
        expect(cpu.output).toEqual(listing);
        expect(compiler.optionsForFilter(cpu.filters, cpu.output)).toEqual(['--emit', 'asm']);

        const cpuObject = request('linux-x86_64', {binaryObject: true});
        expect(cpuObject.output).toEqual(object);
        expect(compiler.optionsForFilter(cpuObject.filters, cpuObject.output)).toEqual(['--emit', 'obj']);

        for (const target of ['wasi-wasm32', 'freestanding-wasm32', 'vulkan-spirv']) {
            const unlisted = request(target);
            expect(unlisted.filters.binaryObject).toBe(true);
            expect(unlisted.output).toEqual(object);
            expect(compiler.optionsForFilter(unlisted.filters, unlisted.output)).toEqual(['--emit', 'obj']);
        }

        // a -t among the options is the target that gets built, whatever the picker says
        expect(request('wasi-wasm32', {}, ['-t', 'linux-aarch64']).output).toEqual(listing);
        expect(request('linux-x86_64', {}, ['--target', 'wasi-wasm32']).output).toEqual(object);

        const linked = request('linux-x86_64', {binary: true});
        expect(linked.filters.binaryObject).toBeFalsy();
        expect(linked.output).toEqual(path.join(root, 'out', 'bin', 'example'));

        const module = request('vulkan-spirv', {binary: true});
        expect(module.output).toEqual(path.join(root, 'out', 'bin', 'example.spv'));
        expect(compiler.optionsForFilter(module.filters, module.output)).toEqual([
            '--emit',
            'exe',
            '-o',
            'out/bin/example.spv',
        ]);
    });

    it('passes no gcc-style library flags to a linked build', () => {
        const root = path.join('/tmp', 'ce');
        const args = compiler.prepareArguments(
            [],
            makeFakeParseFiltersAndOutputOptions({binary: true, execute: true}),
            {},
            path.join(root, 'src', 'example.mach'),
            path.join(root, 'out', 'bin', 'example'),
            [],
            [{name: CompilerOverrideType.arch, value: 'linux-x86_64'}],
        );
        expect(args.some(arg => arg.startsWith('-Wl,'))).toBe(false);
        expect(args).not.toContain('-L./lib');
    });

    it('uses llvm-objdump when disassembling non-Linux x86_64 binaries', () => {
        const darwinObjdumper = (compiler as any).getObjdumperForResult({
            compilationOptions: ['build', 'src/example.mach', '-t', 'darwin-x86_64'],
            instructionSet: 'amd64',
        });
        expect(darwinObjdumper).toEqual({exe: '/usr/bin/llvm-objdump', cls: LlvmObjdumper});

        const linuxObjdumper = (compiler as any).getObjdumperForResult({
            compilationOptions: ['build', 'src/example.mach', '-t', 'linux-x86_64'],
            instructionSet: 'amd64',
        });
        expect(linuxObjdumper?.exe).toEqual('/usr/bin/objdump');
    });

    it('disassembles every object of the project, the entry first and std left out', async () => {
        const root = await fs.mkdtemp(path.join(os.tmpdir(), 'ce-mach-objects'));
        try {
            const entry = path.join(root, 'out', 'obj', 'example', 'example.o');
            for (const file of [
                entry,
                path.join(root, 'out', 'obj', 'example', 'util', 'fmt.o'),
                path.join(root, 'out', 'obj', 'example', 'other.o'),
                path.join(root, 'out', 'obj', 'std', 'print.o'),
            ]) {
                await fs.mkdir(path.dirname(file), {recursive: true});
                await fs.writeFile(file, '');
            }
            const filters = makeFakeParseFiltersAndOutputOptions({binaryObject: true});
            expect(await compiler.getObjdumpInputFilenames(entry, filters, root)).toEqual([
                entry,
                path.join(root, 'out', 'obj', 'example', 'other.o'),
                path.join(root, 'out', 'obj', 'example', 'util', 'fmt.o'),
            ]);

            // a linked binary is the one image there is to disassemble
            const linked = path.join(root, 'out', 'bin', 'example');
            expect(
                await compiler.getObjdumpInputFilenames(
                    linked,
                    makeFakeParseFiltersAndOutputOptions({binary: true}),
                    root,
                ),
            ).toEqual([linked]);
        } finally {
            await fs.rm(root, {recursive: true, force: true});
        }
    });

    it('asks for --emit ir-opt when the Mach IR pane wants it, and reads multi-module MIRL from out/ir-opt', async () => {
        const root = await fs.mkdtemp(path.join(os.tmpdir(), 'ce-mach-ir'));
        try {
            expect(compiler.optionsForBackend({}, '')).toEqual([]);
            expect(compiler.optionsForBackend({produceMachIr: true}, '')).toEqual(['--emit', 'ir-opt']);

            const irDir = path.join(root, 'out', 'ir-opt', 'example');
            await fs.mkdir(irDir, {recursive: true});
            await fs.writeFile(path.join(irDir, 'example.ir'), 'mirl "0.3.1";\npub fun example.example.main() {}\n');
            await fs.writeFile(path.join(irDir, 'helper.ir'), 'mirl "0.3.1";\npub fun example.helper.bump() {}\n');

            const input = path.join(root, 'src', 'example.mach');
            const lines = await compiler.processMachIrOutput(input, {code: 0} as any);
            expect(lines.map(l => l.text)).toEqual([
                'mirl "0.3.1";',
                'pub fun example.example.main() {}',
                '',
                'mirl "0.3.1";',
                'pub fun example.helper.bump() {}',
            ]);
        } finally {
            await fs.rm(root, {recursive: true, force: true});
        }
    });

    it('rejects execution on non-hosted targets while allowing hosted targets through', async () => {
        for (const nonHosted of ['freestanding-x86_64-sysv64', 'uefi-x86_64', 'wasi-wasm32', 'vulkan-spirv']) {
            const res = await compiler.doExecution(
                {
                    compiler: compiler.getInfo(),
                    source: '',
                    options: [],
                    backendOptions: {overrides: [{name: CompilerOverrideType.arch, value: nonHosted}]},
                    filters: makeFakeParseFiltersAndOutputOptions({execute: true}),
                    libraries: [],
                    files: [],
                } as any,
                {} as any,
                0 as any,
            );
            expect(res.didExecute).toBe(false);
            expect(res.stderr).toEqual([{text: `Execution is not supported for target ${nonHosted}`}]);
        }
    });

    it('parses mach help build flags including --[no-]optimize and multi-line descriptions', async () => {
        vi.spyOn(compiler, 'execCompilerCached').mockResolvedValue({
            code: 0,
            stdout: helpBuild,
            stderr: '',
        } as any);
        const parser = new MachParser(compiler);
        const options = await parser.getOptions('help build');
        expect(options['-O']).toBeDefined();
        expect(options['--optimize'].description).toContain('optimize, or not with --no-optimize');
        expect(options['--no-optimize'].description).toContain('optimize, or not with --no-optimize');
        expect(options['--emit <asm|ir|ir-opt|obj|exe>'].description).toEqual(
            'a product to write, once each (default: exe)',
        );
        expect(options['--report <human|json|lines>'].description).toEqual(
            'report as human text, NDJSON records or tab-separated record lines (default: human)',
        );
        expect(options['-v'].description).toEqual('show each phase as it closes, -vv each module too');
        expect(options['--verbose']).toEqual(options['-v']);
        expect(options['--isa <selection>'].description).toEqual("the instruction set (default: the target's)");
        expect(options['-t <pattern>']).toEqual(options['--target <pattern>']);
        expect(await parser.getPossibleTargets()).toContain('vulkan-spirv');
    });
});

describe('Mach multi-file projects', () => {
    let compiler: MachCompiler;
    let dirPath: string;
    let stdPath: string;

    beforeAll(async () => {
        stdPath = await makeStd();
        compiler = makeMach(stdPath);
        fakeMach(compiler);
        await start(compiler);
    });

    beforeEach(async () => {
        dirPath = await fs.mkdtemp(path.join(os.tmpdir(), 'ce-mach-layout'));
        vi.mocked(compiler.exec).mockClear();
    });

    afterEach(async () => {
        await fs.rm(dirPath, {recursive: true, force: true});
    });

    afterAll(async () => {
        await fs.rm(stdPath, {recursive: true, force: true});
    });

    it('roots every source at src/ and writes nothing else', async () => {
        const {inputFilename} = await (compiler as any).writeAllFiles(dirPath, 'entry', [
            {filename: 'util/fmt.mach', contents: 'fmt'},
            {filename: 'other.mach', contents: 'other'},
        ]);

        expect(inputFilename).toEqual(path.join(dirPath, 'src', 'example.mach'));
        expect(((await fs.readdir(dirPath, {recursive: true})) as string[]).sort()).toEqual([
            'src',
            path.join('src', 'example.mach'),
            path.join('src', 'other.mach'),
            path.join('src', 'util'),
            path.join('src', 'util', 'fmt.mach'),
        ]);
        expect(await fs.readFile(path.join(dirPath, 'src', 'util', 'fmt.mach'), 'utf8')).toEqual('fmt');
        expect(compiler.exec).not.toHaveBeenCalled();
    });

    it('refuses an extra file that would land outside src/', async () => {
        await expect(
            (compiler as any).writeAllFiles(dirPath, 'entry', [{filename: '../escape.mach', contents: 'no'}]),
        ).rejects.toThrow();
    });
});

/**
 * Verbatim captures from `mach build --report json` (schema 3) at mach 7.2.0 with mach-std 10.1.1.
 */
describe('Mach diagnostics', () => {
    const root = '/tmp/compiler-explorer-compiler-mach';
    const inputFilename = `${root}/src/example.mach`;
    let compiler: MachCompiler;

    beforeAll(() => {
        compiler = new MachCompiler(
            makeFakeCompilerInfo({id: 'mach', exe: '/opt/compiler-explorer/mach-7.2.0/mach', lang: 'mach'}),
            makeCompilationEnvironment({languages}),
        );
    });

    function parseText(stderr: string) {
        return compiler.processExecutionResult({code: 1, stdout: '', stderr} as any, inputFilename).stderr;
    }

    function capture(name: string) {
        return readFileSync(path.join(__dirname, 'mach', 'diagnostics', `${name}.ndjson`), 'utf8');
    }

    function parse(name: string) {
        return parseText(capture(name));
    }

    function marks(lines: ReturnType<typeof parse>) {
        return lines
            .filter(line => line.tag)
            .map(({tag}) => [
                tag!.file,
                tag!.line,
                tag!.column,
                tag!.endline,
                tag!.endcolumn,
                tag!.severity,
                tag!.text,
            ]);
    }

    function texts(lines: ReturnType<typeof parse>) {
        return lines.map(line => line.text);
    }

    it('renders an error as its headline and location, marks its span, and ends with the tally', () => {
        const lines = parse('error');
        expect(texts(lines)).toEqual([
            'error[name.unresolved]: unresolved identifier `bogus`',
            ' --> src/example.mach:2:9',
            '1 error / 0 warnings',
        ]);
        expect(marks(lines)).toEqual([
            ['example.mach', 2, 9, 2, 14, 3, 'error[name.unresolved]: unresolved identifier `bogus`'],
        ]);
    });

    it('marks a warning at warning severity, and keeps it apart from an error', () => {
        expect(marks(parse('warning'))).toEqual([
            ['example.mach', 1, 20, 1, 25, 2, 'warning[import.unused]: unused import `usize`'],
        ]);
        const both = parse('warning-and-error');
        expect(marks(both).map(([, line, , , , severity]) => [line, severity])).toEqual([
            [4, 3],
            [1, 2],
        ]);
        expect(texts(both).at(-1)).toEqual('1 error / 1 warning');
    });

    it('shows no tally for a build with nothing to report, and renders -v unit and phase records', () => {
        const lines = parse('verbose');
        expect(marks(lines)).toEqual([]);
        expect(texts(lines)[0]).toEqual('building example (linux-x86_64, debug)');
        expect(texts(lines)).toContain('  target: 13.7ms (1 target)');
        expect(texts(lines)).toContain('    lower: 0.3ms (1 object)');
        expect(texts(lines).at(-1)).toEqual('  other: 1.0ms');
    });

    it('folds help into the marker and offers the fix as a quick fix over the span it replaces', () => {
        const lines = parse('help-and-fix');
        expect(texts(lines)).toEqual([
            'error[name.unresolved]: unresolved identifier `helpr`',
            ' --> src/example.mach:6:9',
            '  = help: did you mean `helper`?',
            '  = fix: replace with `helper`',
            '1 error / 0 warnings',
        ]);
        expect(lines[0].tag).toEqual({
            file: 'example.mach',
            line: 6,
            column: 9,
            endline: 6,
            endcolumn: 14,
            severity: 3,
            text: 'error[name.unresolved]: unresolved identifier `helpr`\nhelp: did you mean `helper`?',
            fixes: [
                {
                    title: 'replace with `helper`',
                    edits: [{file: 'example.mach', line: 6, column: 9, endline: 6, endcolumn: 14, text: 'helper'}],
                },
            ],
        });
    });

    it('folds notes and help into single-edit and multi-edit cast quick fixes', () => {
        const [single] = parse('note-and-fix');
        expect(single.tag?.text).toEqual(
            'error[type.mismatch]: expected i64, found i32\n' +
                'note: implicit numeric conversion is not allowed\n' +
                'help: cast the value with `::i64`',
        );
        expect(single.tag?.fixes).toEqual([
            {
                title: 'cast the value with `::i64`',
                edits: [{file: 'example.mach', line: 2, column: 10, endline: 2, endcolumn: 10, text: '::i64'}],
            },
        ]);

        const [multi] = parse('two-edit-fix');
        expect(multi.tag?.text).toEqual(
            'error[type.mismatch]: expected i64, found i32\n' +
                'note: implicit numeric conversion is not allowed\n' +
                'help: parenthesize the value and cast it with `::i64`',
        );
        expect([multi.tag?.line, multi.tag?.column, multi.tag?.endline, multi.tag?.endcolumn]).toEqual([2, 5, 4, 11]);
        expect(multi.tag?.fixes).toEqual([
            {
                title: 'parenthesize the value and cast it with `::i64`',
                edits: [
                    {file: 'example.mach', line: 2, column: 9, endline: 2, endcolumn: 9, text: '('},
                    {file: 'example.mach', line: 4, column: 10, endline: 4, endcolumn: 10, text: ')::i64'},
                ],
            },
        ]);
    });

    it('marks a related site with its label', () => {
        const lines = parse('related');
        expect(texts(lines)).toContain(' --> src/example.mach:1:5: previous definition here');
        expect(marks(lines)).toEqual([
            ['example.mach', 2, 5, 2, 8, 3, 'error[name.duplicate]: `dup` is already bound in this scope'],
            ['example.mach', 1, 5, 1, 8, 1, 'previous definition here'],
        ]);
    });

    it('names a second file by the path the project tree knows it by', () => {
        expect(marks(parse('second-file'))).toEqual([
            ['util/fmt.mach', 2, 9, 2, 13, 3, 'error[name.unresolved]: unresolved identifier `nope`'],
        ]);
    });

    it('marks the user import while masking a related std site without placing an editor marker in std', () => {
        const lines = parse('std');
        expect(marks(lines)).toEqual([
            [
                'example.mach',
                1,
                12,
                1,
                21,
                3,
                'error[comptime.user_error]: std.print needs the files capability (std.system.capability.HAS_FILES)',
            ],
        ]);
        expect(texts(lines)).toContain(
            ' --> std/src/print.mach:35:5: the module this import reaches is refused by this module-scope `$error`',
        );
    });

    it.each(['link', 'unknown-target', 'flag-conflict'])('shows a %s failure with no location, unmarked', name => {
        const lines = parse(name);
        expect(marks(lines)).toEqual([]);
        expect(texts(lines)).toHaveLength(2);
        expect(texts(lines)[0]).toMatch(/^error\[[a-z_.]+]: /);
    });

    it('turns byte columns into the editor columns of a line with non-ASCII text before the span', () => {
        const source = utils.splitLines(readFileSync(path.join(__dirname, 'mach', 'diagnostics', 'utf8.mach'), 'utf8'));
        const [headline] = toEditorColumns(parse('utf8'), (file, line) =>
            file === 'example.mach' ? source[line - 1] : undefined,
        );
        expect([headline.tag?.column, headline.tag?.endcolumn]).toEqual([37, 41]);
        expect(source[1].slice(36, 40)).toEqual('nope');
    });

    it('remaps the edits of a quick fix with the marker', () => {
        const [headline] = toEditorColumns(parse('help-and-fix'), () => '    ret hélpr();');
        expect(headline.tag?.fixes?.[0].edits[0]).toMatchObject({column: 9, endcolumn: 13});
    });

    it('passes through non-JSON lines and unknown schemas untouched', () => {
        expect(parseText(`building in ${root}${path.sep}out`)).toEqual([{text: 'building in out'}]);
        const record = '{"schema":2,"record":"diagnostic"}';
        expect(parseText(record)).toEqual([{text: record}]);
    });

    it('offers no quick fix whose edits reach outside the marked editor', () => {
        const record = JSON.parse(capture('help-and-fix').split('\n')[0]);
        record.fixes[0].edits[0].file = 'src/other.mach';
        expect(parseText(JSON.stringify(record))[0].tag?.fixes).toBeUndefined();
    });
});

describe.skipIf(process.platform === 'win32')('Mach asm with an /app project root', () => {
    it('attributes an /app source line to the editor', () => {
        const objdump = readFileSync(path.join(__dirname, 'mach', 'app-objdump.asm'), 'utf8');
        const parsed = new AsmParser().process(
            objdump,
            makeFakeParseFiltersAndOutputOptions({binaryObject: true, directives: true}),
        );

        expect(parsed.asm.filter(line => line.source).map(line => line.source)).toContainEqual({
            file: null,
            line: 6,
            mainsource: true,
        });
        expect(parsed.asm.map(line => line.text)).toContain('main:');
    });

    it('strips the /app root when filenames are not masked', () => {
        const objdump = readFileSync(path.join(__dirname, 'mach', 'app-objdump.asm'), 'utf8');
        const parsed = new AsmParser().process(
            objdump,
            makeFakeParseFiltersAndOutputOptions({binaryObject: true, directives: true, dontMaskFilenames: true}),
        );

        expect(parsed.asm.filter(line => line.source).map(line => line.source)).toContainEqual({
            file: 'src/example.mach',
            line: 6,
            mainsource: true,
        });
    });
});

/**
 * Multi-module `--emit asm` `.s` outputs from a two-module project (`src/example.mach` and `src/util/fmt.mach`),
 * where each module defines its own `.file 0` and `.Ltmp*` labels.
 */
describe('Mach asm from two modules', () => {
    const entryAsm = readFileSync(path.join(__dirname, 'mach', 'two-module-entry.s'), 'utf8');
    const fmtAsm = readFileSync(path.join(__dirname, 'mach', 'two-module-fmt.s'), 'utf8');
    const joined = entryAsm + MACH_MODULE_SEP + fmtAsm;

    const parse = (filters: object) =>
        new MachAsmParser(undefined, 'src').process(
            joined,
            makeFakeParseFiltersAndOutputOptions({directives: true, labels: true, commentOnly: true, ...filters}),
        );
    const sources = (parsed: {asm: {source?: unknown}[]}) =>
        parsed.asm.filter(line => line.source).map(line => line.source);

    it('shows every function of every module, the entry first, and links cross-module calls', () => {
        const parsed = parse({});
        const labels = parsed.asm.map(line => line.text).filter(text => /^[A-Za-z_][\w.]*:$/.test(text));
        expect(labels).toEqual(['example.example.main:', 'example.util.fmt.twice:', 'example.util.fmt.unused:']);
        expect(parsed.labelDefinitions?.['example.util.fmt.twice']).toBeGreaterThan(1);

        const callLine = unwrap(parsed.asm.find(line => line.text.includes('call example.util.fmt.twice')));
        expect(callLine.labels?.map(l => l.name)).toContain('example.util.fmt.twice');
    });

    it('isolates .file 0 per module and names secondary files as the tree pane knows them', () => {
        const parsed = parse({dontMaskFilenames: true});
        expect(sources(parsed)).toContainEqual({file: 'example.mach', line: 6, column: 9, mainsource: true});
        expect(sources(parsed)).toContainEqual({file: 'util/fmt.mach', line: 2, column: 9, mainsource: false});
        expect(sources(parsed)).not.toContainEqual(expect.objectContaining({file: 'src/util/fmt.mach'}));
    });

    it("marks only the entry as the editor's when filenames are masked", () => {
        const parsed = parse({});
        expect(sources(parsed)).toContainEqual({file: null, line: 5, column: 5, mainsource: true});
        expect(sources(parsed)).toContainEqual({file: 'util/fmt.mach', line: 6, column: 9, mainsource: false});
    });

    it('keeps secondary user files under libraryCode while dropping non-src std functions', () => {
        const withStd =
            joined +
            MACH_MODULE_SEP +
            [
                '\t.intel_syntax noprefix',
                '\t.file 0 "/opt/compiler-explorer/mach-7.2.0/std/src/math.mach"',
                '\t.globl std.math.min',
                '\t.type std.math.min, @function',
                '\t.text',
                'std.math.min:',
                '\t.cfi_startproc',
                '\t.loc 0 14 9',
                '\tmov rax, rdi',
                '\tret',
                '\t.cfi_endproc',
            ].join('\n');

        const parsed = new MachAsmParser(undefined, 'src').process(
            withStd,
            makeFakeParseFiltersAndOutputOptions({
                directives: true,
                labels: true,
                commentOnly: true,
                libraryCode: true,
            }),
        );
        const labels = parsed.asm.map(line => line.text).filter(text => /^[A-Za-z_][\w.]*:$/.test(text));
        expect(labels).toContain('example.util.fmt.twice:');
        expect(labels).not.toContain('std.math.min:');
    });
});

describe('Mach SPIR-V asm', () => {
    const spvasm = readFileSync(path.join(__dirname, 'mach', 'shader.spvasm'), 'utf8');

    it('parses SPIR-V disassembly and normalises src/example.mach to the editor', () => {
        const parsed = new MachAsmParser(undefined, 'src').processSpirv(
            spvasm,
            makeFakeParseFiltersAndOutputOptions({directives: true, commentOnly: true}),
        );
        expect(parsed.languageId).toEqual('spirv');
        const retLine = unwrap(parsed.asm.find(line => line.text.includes('OpReturn')));
        expect(retLine.source).toEqual({file: null, line: 3, column: 9, mainsource: true});
    });
});

describe('Mach binary asm', () => {
    const objdump = readFileSync(path.join(__dirname, 'mach', 'hello-binary.asm'), 'utf8');

    it.each(['amazon', 'defaults'])('shows only the user functions under the %s properties', env => {
        const file = path.join(__dirname, '..', 'etc', 'config', `mach.${env}.properties`);
        const props = parseProperties(readFileSync(file, 'utf8'), file);
        const parsed = new AsmParser((key: string) => props[key]).process(
            objdump,
            makeFakeParseFiltersAndOutputOptions({binary: true, directives: true}),
        );
        expect(parsed.asm.map(line => line.text).filter(text => text.endsWith(':'))).toEqual(['main:']);
    });
});

/**
 * `--emit ir-opt` MIRL module captured from mach 7.2.0 with mach-std 10.1.1, where `example.helper.bump` is inlined
 * into `example.example.main` via `at l60 in s0`.
 */
describe('Mach MIRL output', () => {
    const mirl = readFileSync(path.join(__dirname, 'mach', 'ir-opt.ir'), 'utf8');
    const parsed = new MachIrParser('src/example.mach').process(mirl);

    function at(text: string) {
        return unwrap(parsed.find(line => line.text.includes(text)));
    }

    it('keeps every line of the module in order and masks the debug directory and entry file', () => {
        expect(parsed).toHaveLength(utils.splitLines(mirl).length);
        expect(parsed[0].text).toEqual('mirl "0.3.1";');
        expect(at('directory = ').text).toEqual('    directory = ".";');
        expect(at('file f0 = ').text).toEqual('    file f0 = "<source>";');
        expect(parsed.at(-1)!.text).toEqual('#[inline(always)] ext fun example.helper.bump(i64) i64;');
    });

    it('maps instructions in src/example.mach to the editor across nested decorator brackets', () => {
        expect(at('xor(_0, 6148914691236517205)').source).toEqual({
            file: null,
            line: 8,
            column: 9,
            mainsource: true,
        });
        expect(at('call.placed').source).toEqual({
            file: null,
            line: 12,
            column: 20,
            mainsource: true,
        });
    });

    it('keeps inlined instructions from another project file off the main editor', () => {
        const inlined = at('add(_5, 1) at l60 in s0;');
        expect(inlined.source).toEqual({file: 'src/helper.mach', line: 3, column: 9});
    });

    it('maps instructions carrying holds clauses and leaves structural lines unmapped', () => {
        expect(at('pub val example.example.LANES').source).toBeUndefined();
        expect(at('bind v1 = _0;').source).toBeUndefined();
        expect(at('_0(_2: ^i64):').source).toBeUndefined();
        expect(at('ext fun std.math.min').source).toBeUndefined();

        const withHolds = new MachIrParser('src/example.mach').process(
            [
                'debug {',
                '    file f0 = "src/example.mach";',
                '    loc l0 = f0:9:13;',
                '}',
                'pub fun example.f(_0: ptr) {',
                '    store(_0, 1) at l0 holds [v0][0, 8];',
                '}',
            ].join('\n'),
        );
        expect(withHolds[5].source).toEqual({file: null, line: 9, column: 13, mainsource: true});
    });
});
