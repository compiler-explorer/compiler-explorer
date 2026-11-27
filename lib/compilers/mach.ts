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

import type {ParsedAsmResultLine} from '../../types/asmresult/asmresult.interfaces.js';
import type {
    BypassCache,
    CacheKey,
    CompilationCacheKey,
    CompilationResult,
    ExecutionOptionsWithEnv,
    FiledataPair,
} from '../../types/compilation/compilation.interfaces.js';
import {CompilerOverrideType, type ConfiguredOverrides} from '../../types/compilation/compiler-overrides.interfaces.js';
import type {MachTarget, PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {
    BasicExecutionResult,
    ExecutableExecutionOptions,
    UnprocessedExecResult,
} from '../../types/execution/execution.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import type {InstructionSet} from '../../types/instructionsets.js';
import type {SelectedLibraryVersion} from '../../types/libraries/libraries.interfaces.js';
import type {ResultLine} from '../../types/resultline/resultline.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';
import {CompilationEnvironment} from '../compilation-env.js';
import {logger} from '../logger.js';
import {LlvmObjdumper} from '../objdumper/llvm.js';
import {ClientOptionsType} from '../options-handler.js';
import {MACH_MODULE_SEP, MACH_SPIRV_HEADER, MachAsmParser} from '../parsers/asm-parser-mach.js';
import {parseMachDiagnostics, toEditorColumns} from '../parsers/mach-diagnostics.js';
import {MachIrParser} from '../parsers/mach-ir.js';
import * as temp from '../temp.js';
import * as utils from '../utils.js';
import {MachParser} from './argument-parsers.js';

const sourceDir = 'src';
const workDir = 'out';

const hostedSystems = new Set(['linux', 'darwin', 'windows']);
const stdPrintSystems = new Set(['linux', 'darwin', 'windows', 'wasi']);

const hostedProbeSource = [
    'use std.runtime;',
    'use print: std.print;',
    '',
    '#[symbol("main")]',
    'fun main(argc: i64, argv: **u8) i64 {',
    '    print.println("probe");',
    '    ret 0;',
    '}',
    '',
].join('\n');

const bareProbeSource = [
    'use std.types.size.usize;',
    '',
    'pub fun probe(x: usize) usize {',
    '    ret x * 3;',
    '}',
    '',
].join('\n');

function defaultIsaFloor(arch: string, abi: string): string | undefined {
    switch (arch) {
        case 'x86_64':
            return 'x86-64';
        case 'aarch64':
            return 'armv8-a+fp+simd';
        case 'riscv64':
            return abi === 'lp64d' ? 'rv64gc' : abi === 'lp64f' ? 'rv64imafc' : 'rv64imac';
        case 'riscv32':
            return abi === 'ilp32d' ? 'rv32gc' : abi === 'ilp32f' ? 'rv32imafc' : 'rv32imac';
        case 'riscv32e':
            return 'rv32emc';
        default:
            return undefined;
    }
}

function instructionSetForArch(arch: string | undefined): InstructionSet | undefined {
    switch (arch) {
        case 'x86_64':
            return 'amd64';
        case 'aarch64':
        case 'arm64ec':
            return 'aarch64';
        case 'riscv64':
            return 'riscv64';
        case 'riscv32':
        case 'riscv32e':
            return 'riscv32';
        case 'wasm32':
            return 'wasm32';
        case 'spirv':
            return 'spirv';
        default:
            return undefined;
    }
}

function hostPlatform(): string {
    const sys = process.platform === 'darwin' ? 'darwin' : process.platform === 'win32' ? 'windows' : 'linux';
    const arch = process.arch === 'arm64' ? 'aarch64' : process.arch === 'riscv64' ? 'riscv64' : 'x86_64';
    return `${sys}-${arch}`;
}

function qualifyNames(rows: MachTarget[]): MachTarget[] {
    return rows.map(row => {
        const samePlatform = rows.filter(other => other.platform === row.platform);
        if (samePlatform.length === 1) return {...row, name: row.platform};
        return {...row, name: `${row.platform}-${row.abi}`};
    });
}

function hasFlag(args: string[], names: string[]): boolean {
    return args.some(arg => names.some(name => arg === name || arg.startsWith(`${name}=`)));
}

function extractTargetOption(args: string[]): {targetName?: string; rest: string[]} {
    let targetName: string | undefined;
    const rest: string[] = [];
    for (let i = 0; i < args.length; i++) {
        const arg = args[i];
        if ((arg === '--target' || arg === '-t') && i + 1 < args.length) {
            targetName = args[++i];
        } else if (arg.startsWith('--target=')) {
            targetName = arg.slice('--target='.length);
        } else if (arg.startsWith('-t=')) {
            targetName = arg.slice('-t='.length);
        } else {
            rest.push(arg);
        }
    }
    return {targetName, rest};
}

/**
 * Mach 7 builds single-file and multi-file trees directly through entry builds (`mach build src/example.mach`),
 * binding the user's source tree with `--source <id>=src` and the paired std checkout in place with
 * `--source std=<stdPath>`. No `mach.toml` is generated and `std` is never copied per compile.
 *
 *   src/example.mach            the user's entry module (and any extra files, rooted at src/)
 *   out/asm/example/*.s         masc assembly listings on CPU targets (`--emit asm`)
 *   out/ir-opt/example/*.ir     optimised/legalised MIRL text for the Mach IR pane (`--emit ir-opt`)
 *   out/obj/example/*.o         per-module objects (used on Wasm and unlinked SPIR-V)
 *   out/bin/example[.spv]       linked executable or finished SPIR-V module (`--emit exe`)
 */
export class MachCompiler extends BaseCompiler {
    static get key() {
        return 'mach';
    }

    readonly stdPath: string;
    private readonly spirvDisassembler: string;
    private readonly resultTarget = new WeakMap<CompilationResult, MachTarget>();
    private buildable?: MachTarget[];

    constructor(info: PreliminaryCompilerInfo, env: CompilationEnvironment) {
        super(info, env);
        this.stdPath =
            this.compilerProps<string>(`compiler.${this.compiler.id}.stdPath`) ??
            path.join(path.dirname(this.compiler.exe), 'std');
        this.spirvDisassembler = this.compilerProps<string>('disassemblerPath') || 'spirv-dis';
        this.compiler.supportsTarget = true;
        this.compiler.supportsMachIrView = true;
        this.asm = new MachAsmParser(this.compilerProps, sourceDir, this.compileFilename);
    }

    override async initialise(
        mtime: Date,
        clientOptions: ClientOptionsType,
        isPrediscovered = false,
    ): Promise<BaseCompiler | null> {
        const cached = this.compiler.cachedMachTargets;
        if (isPrediscovered && cached && cached.every(t => typeof t.platform === 'string')) {
            this.buildable = cached;
        } else if (!this.getRemote()) {
            try {
                this.buildable = await this.probeTargets();
                this.compiler.cachedMachTargets = this.buildable;
            } catch (e) {
                logger.error(`${this.compiler.id}: target probe failed:`, e);
                return null;
            }
        }
        return super.initialise(mtime, clientOptions, isPrediscovered);
    }

    override getArgumentParserClass() {
        return MachParser;
    }

    get projectId(): string {
        return path.parse(this.compileFilename).name;
    }

    projectRoot(inputFilename: string): string {
        return path.dirname(path.dirname(inputFilename));
    }

    /** Parses `mach info targets` into candidate (platform, abi, isa, kind) tuples. */
    async supportedTuples(): Promise<MachTarget[]> {
        const result = await this.exec(this.compiler.exe, ['info', 'targets'], this.getDefaultExecOptions());
        if (result.code !== 0) throw new Error(`mach info targets failed: ${result.stderr}`);

        const rows: MachTarget[] = [];
        for (const line of utils.splitLines(result.stdout)) {
            const trimmed = line.trim();
            if (!trimmed || trimmed.startsWith('target ')) continue;
            const cols = trimmed.split(/\s{2,}/);
            if (cols.length < 5 || !cols[0].includes('-')) continue;

            const [platform, , abiCol, , linkageCol, noteCol] = cols;
            const dash = platform.indexOf('-');
            const sys = platform.slice(0, dash);
            const arch = platform.slice(dash + 1);
            const needsIsa = noteCol?.includes('needs --isa') ?? false;
            const noExecutable = linkageCol.split(/\s+/).every(tok => tok === '-' || tok.endsWith(':-'));

            for (const rawAbi of abiCol.split(',')) {
                const defaultAbi = rawAbi.startsWith('*');
                const abi = defaultAbi ? rawAbi.slice(1) : rawAbi;
                if (!abi || abi === '-') continue;
                const isa = needsIsa ? defaultIsaFloor(arch, abi) : undefined;
                rows.push({
                    name: platform,
                    platform,
                    sys,
                    arch,
                    abi,
                    ...(defaultAbi && {defaultAbi: true}),
                    ...(isa && {isa}),
                    ...(noExecutable && {kind: 'module' as const}),
                });
            }
        }
        return rows;
    }

    targets(): MachTarget[] {
        if (!this.buildable) throw new Error(`${this.compiler.id}: targets read before initialise() probed them`);
        return this.buildable;
    }

    private findTarget(name: string | undefined): MachTarget | undefined {
        if (!this.buildable || this.buildable.length === 0) return undefined;
        if (name) {
            return (
                this.buildable.find(t => t.name === name) ??
                this.buildable.find(t => t.platform === name && t.defaultAbi) ??
                this.buildable.find(t => t.platform === name)
            );
        }
        const host = hostPlatform();
        return this.buildable.find(t => t.platform === host && t.defaultAbi) ?? this.buildable[0];
    }

    targetFlags(target: MachTarget, explicitArgs: string[] = []): string[] {
        const flags = ['-t', target.platform];
        if (!target.defaultAbi && !hasFlag(explicitArgs, ['--abi'])) {
            flags.push('--abi', target.abi);
        }
        if (target.isa && !hasFlag(explicitArgs, ['--isa'])) {
            flags.push('--isa', target.isa);
        }
        if (target.kind && !hasFlag(explicitArgs, ['-k', '--kind'])) {
            flags.push('-k', target.kind);
        }
        return flags;
    }

    private async probeTargets(): Promise<MachTarget[]> {
        if (!(await utils.fileExists(path.join(this.stdPath, 'mach.toml'))))
            throw new Error(
                `${this.compiler.id}: no mach std at ${this.stdPath}; install it there or set ` +
                    `compiler.${this.compiler.id}.stdPath (see docs/Mach.md)`,
            );
        const tuples = await this.supportedTuples();
        const dirPath = await this.newTempDir({hold: true});
        try {
            await fs.mkdir(path.join(dirPath, 'hosted'), {recursive: true});
            await fs.mkdir(path.join(dirPath, 'bare'), {recursive: true});
            await fs.writeFile(path.join(dirPath, 'hosted', 'probe.mach'), hostedProbeSource);
            await fs.writeFile(path.join(dirPath, 'bare', 'probe.mach'), bareProbeSource);

            const rows: MachTarget[] = [];
            for (const [index, target] of tuples.entries()) {
                if (await this.buildsUnderProbe(dirPath, index, target)) rows.push(target);
            }
            return qualifyNames(rows);
        } finally {
            temp.release(dirPath);
            await fs.rm(dirPath, {recursive: true, force: true});
        }
    }

    private async buildsUnderProbe(dirPath: string, index: number, target: MachTarget): Promise<boolean> {
        const sub = stdPrintSystems.has(target.sys) ? 'hosted' : 'bare';
        const args = [
            'build',
            `${sub}/probe.mach`,
            '--source',
            `probe=${sub}`,
            '--source',
            `std=${this.stdPath}`,
            '-w',
            `w${index}`,
            '--debug',
            '--emit',
            'obj',
            ...this.targetFlags(target),
        ];
        if (!target.kind && sub === 'bare') args.push('-k', 'object');
        const result = await this.exec(this.compiler.exe, args, {
            ...this.getDefaultExecOptions(),
            customCwd: dirPath,
        });
        return result.code === 0;
    }

    protected override async writeAllFiles(dirPath: string, source: string, files: FiledataPair[]) {
        if (!source) throw new Error(`File ${this.compileFilename} has no content or file is missing`);

        const srcDir = path.join(dirPath, sourceDir);
        await fs.mkdir(srcDir, {recursive: true});

        const inputFilename = path.join(srcDir, this.compileFilename);
        await fs.writeFile(inputFilename, source);
        if (files && files.length > 0) await this.writeMultipleFiles(files, srcDir);

        return {inputFilename};
    }

    private targetFromKey(key?: CacheKey | CompilationCacheKey): MachTarget | undefined {
        if (!this.isCacheKey(key)) return this.findTarget(undefined);
        const archOverride = key.backendOptions?.overrides?.find(
            (ov: ConfiguredOverrides[number]) => ov.name === CompilerOverrideType.arch && ov.value,
        );
        if (archOverride && 'value' in archOverride && archOverride.value) {
            return this.findTarget(archOverride.value);
        }
        const {targetName} = extractTargetOption(key.options ?? []);
        return this.findTarget(targetName);
    }

    private usesAsmText(target: MachTarget | undefined, binary?: boolean): boolean {
        if (binary) return false;
        if (!target) return true;
        return target.sys !== 'vulkan' && target.arch !== 'wasm32';
    }

    override getOutputFilename(dirPath: string, outputFilebase: string, key?: CacheKey | CompilationCacheKey) {
        const target = this.targetFromKey(key);
        const binary = this.isCacheKey(key) && Boolean(key.filters?.binary);
        if (binary) return this.getExecutableFilename(dirPath, outputFilebase, key);
        const binaryObject = this.isCacheKey(key) && Boolean(key.filters?.binaryObject);
        const id = this.projectId;
        if (this.usesAsmText(target, binaryObject)) {
            return path.join(dirPath, workDir, 'asm', id, `${id}.s`);
        }
        return path.join(dirPath, workDir, 'obj', id, `${id}.o`);
    }

    override getSharedLibraryPathsAsArguments(): string[] {
        return [];
    }

    override getExecutableFilename(dirPath: string, outputFilebase: string, key?: CacheKey | CompilationCacheKey) {
        const target = this.targetFromKey(key);
        const suffix = target?.sys === 'vulkan' ? '.spv' : '';
        return path.join(dirPath, workDir, 'bin', `${this.projectId}${suffix}`);
    }

    /** Collects `[entryFile, ...sortedSiblings]` with `ext` under `entryFile`'s module root (`out/<kind>/<id>/`). */
    private async collectModuleOutputs(entryFile: string, ext: string): Promise<string[]> {
        const moduleRoot = path.dirname(entryFile);
        let entries: string[];
        try {
            entries = (await fs.readdir(moduleRoot, {recursive: true})) as string[];
        } catch {
            return (await utils.fileExists(entryFile)) ? [entryFile] : [];
        }
        const others = entries
            .map(entry => path.join(moduleRoot, entry))
            .filter(file => file.endsWith(ext) && file !== entryFile)
            .sort();
        const hasEntry = await utils.fileExists(entryFile);
        return hasEntry ? [entryFile, ...others] : others;
    }

    override async getObjdumpInputFilenames(outputFilename: string, filters?: ParseFiltersAndOutputOptions) {
        if (filters?.binary) return [outputFilename];
        const files = await this.collectModuleOutputs(outputFilename, '.o');
        return files.length > 0 ? files : [outputFilename];
    }

    override prepareArguments(
        userOptions: string[],
        filters: ParseFiltersAndOutputOptions,
        backendOptions: Record<string, any>,
        inputFilename: string,
        outputFilename: string,
        libraries: SelectedLibraryVersion[],
        overrides: ConfiguredOverrides,
    ) {
        const archOverride = overrides.find(ov => ov.name === CompilerOverrideType.arch && ov.value);
        const {targetName: userTarget} = extractTargetOption(userOptions);
        const target = this.findTarget(
            (archOverride && 'value' in archOverride ? archOverride.value : undefined) ?? userTarget,
        );
        if (!filters.binary && target?.arch === 'wasm32') {
            filters.binaryObject = true;
        }
        return super.prepareArguments(
            userOptions,
            filters,
            backendOptions,
            inputFilename,
            outputFilename,
            libraries,
            overrides,
        );
    }

    override optionsForFilter(
        filters: ParseFiltersAndOutputOptions,
        outputFilename: string,
        userOptions: string[] = [],
    ) {
        const outRel = path.posix.join(workDir, 'bin', path.basename(outputFilename));
        if (filters.binary) {
            return ['--emit', 'exe', '-o', outRel];
        }
        if (outputFilename.endsWith('.o')) {
            return ['--emit', 'obj'];
        }
        return ['--emit', 'asm'];
    }

    override optionsForBackend(backendOptions: Record<string, any>, outputFilename: string) {
        const options = super.optionsForBackend(backendOptions, outputFilename);
        if (backendOptions.produceMachIr && this.compiler.supportsMachIrView) {
            options.push('--emit', 'ir-opt');
        }
        return options;
    }

    override orderArguments(
        options: string[],
        inputFilename: string,
        libIncludes: string[],
        libOptions: string[],
        libPaths: string[],
        libLinks: string[],
        userOptions: string[],
        staticLibLinks: string[],
    ) {
        const id = this.projectId;
        const entryRel = path.posix.join(sourceDir, this.compileFilename);
        const combined = [...options, ...userOptions];
        const {targetName, rest} = extractTargetOption(combined);
        const target = this.findTarget(targetName);

        const args = [
            'build',
            entryRel,
            '--source',
            `${id}=${sourceDir}`,
            '--source',
            `std=${this.stdPath}`,
            '-w',
            workDir,
            '--report',
            'json',
        ];
        if (!hasFlag(rest, ['--debug', '--no-debug'])) {
            args.push('--debug');
        }
        if (target) {
            args.push(...this.targetFlags(target, rest));
        } else if (targetName) {
            args.push('-t', targetName);
        }
        args.push(...libIncludes, ...libOptions, ...libPaths, ...libLinks, ...staticLibLinks, ...rest);
        return args;
    }

    override async runCompiler(
        compiler: string,
        options: string[],
        inputFilename: string,
        execOptions: ExecutionOptionsWithEnv,
        filters?: ParseFiltersAndOutputOptions,
    ): Promise<CompilationResult> {
        const projectRoot = this.projectRoot(inputFilename);
        execOptions.customCwd = projectRoot;

        const {targetName} = extractTargetOption(options);
        const target = this.findTarget(targetName);

        const result = await super.runCompiler(compiler, options, inputFilename, execOptions, filters);
        if (target) {
            this.resultTarget.set(result, target);
            result.instructionSet = instructionSetForArch(target.arch);
            if (target.sys === 'vulkan') result.languageId = 'spirv';
        }
        result.stderr = await this.inEditorColumns(result.stderr, path.join(projectRoot, sourceDir));
        return result;
    }

    protected override getObjdumperForResult(result: CompilationResult) {
        const target =
            this.resultTarget.get(result) ??
            (result.compilationOptions
                ? this.findTarget(extractTargetOption(result.compilationOptions).targetName)
                : undefined);
        if (this.compiler.llvmObjdumper && target && target.sys !== 'linux') {
            return {exe: this.compiler.llvmObjdumper, cls: LlvmObjdumper};
        }
        return super.getObjdumperForResult(result);
    }

    override async checkOutputFileAndDoPostProcess(
        asmResult: CompilationResult,
        outputFilename: string,
        filters: ParseFiltersAndOutputOptions,
        produceOptRemarks = false,
    ) {
        if (asmResult.languageId === 'spirv') {
            const inputs = filters.binary ? [outputFilename] : await this.collectModuleOutputs(outputFilename, '.o');
            const disassembled: string[] = [];
            for (const file of inputs) {
                if (!(await utils.fileExists(file))) continue;
                const spv = await this.exec(this.spirvDisassembler, [file, '--comment'], {
                    ...this.getDefaultExecOptions(),
                    customCwd: asmResult.dirPath,
                });
                if (spv.code === 0) {
                    disassembled.push(spv.stdout);
                } else {
                    logger.error('SPIR-V disassembly failed', spv);
                }
            }
            asmResult.asm =
                disassembled.length > 0
                    ? MACH_SPIRV_HEADER + disassembled.join(MACH_MODULE_SEP)
                    : '<No SPIR-V output generated>';
            return [asmResult, [], []] as [CompilationResult, any[], any[]];
        }

        if (!filters.binary && !filters.binaryObject && outputFilename.endsWith('.s')) {
            const asmFiles = await this.collectModuleOutputs(outputFilename, '.s');
            if (asmFiles.length > 0) {
                const chunks = await Promise.all(asmFiles.map(file => fs.readFile(file, 'utf8')));
                const joined = chunks.join(MACH_MODULE_SEP);
                asmResult.asmSize = Buffer.byteLength(joined, 'utf8');
                asmResult.asm = joined;
                return [asmResult, [], []] as [CompilationResult, any[], any[]];
            }
        }

        return super.checkOutputFileAndDoPostProcess(asmResult, outputFilename, filters, produceOptRemarks);
    }

    override getMachIrOutputFilename(inputFilename: string): string {
        const id = this.projectId;
        return path.join(this.projectRoot(inputFilename), workDir, 'ir-opt', id, `${id}.ir`);
    }

    override async processMachIrOutput(outpath: string, output: CompilationResult): Promise<ParsedAsmResultLine[]> {
        if (output.code !== 0) return [{text: 'Failed to run compiler to get Mach IR'}];
        const irFiles = await this.collectModuleOutputs(outpath, '.ir');
        if (irFiles.length === 0) return [{text: 'Internal error; unable to open output path'}];
        const modules = await Promise.all(irFiles.map(file => fs.readFile(file, 'utf8')));
        return new MachIrParser(path.posix.join(sourceDir, this.compileFilename)).processModules(modules);
    }

    private async inEditorColumns(lines: ResultLine[], srcDir: string): Promise<ResultLine[]> {
        const files = new Map<string, string[]>();
        for (const line of lines) {
            const file = line.tag?.file;
            if (file === undefined || files.has(file)) continue;
            const text = await fs.readFile(path.join(srcDir, file), 'utf8').catch(() => undefined);
            if (text !== undefined) files.set(file, utils.splitLines(text));
        }
        return toEditorColumns(lines, (file, line) => files.get(file)?.[line - 1]);
    }

    override async doExecution(
        key: CacheKey,
        executeParameters: ExecutableExecutionOptions,
        bypassCache: BypassCache,
    ): Promise<CompilationResult> {
        const target = this.targetFromKey(key);
        if (target && !hostedSystems.has(target.sys)) {
            return {
                code: -1,
                didExecute: false,
                buildResult: {
                    code: 0,
                    timedOut: false,
                    stdout: [],
                    stderr: [],
                    downloads: [],
                    executableFilename: '',
                    compilationOptions: [],
                },
                stderr: [{text: `Execution is not supported for target ${target.name}`}],
                stdout: [],
                execTime: 0,
                timedOut: false,
            };
        }
        return super.doExecution(key, executeParameters, bypassCache);
    }

    override async buildExecutable(
        compiler: string,
        options: string[],
        inputFilename: string,
        execOptions: ExecutionOptionsWithEnv,
    ) {
        execOptions.customCwd = this.projectRoot(inputFilename);
        return super.buildExecutable(compiler, options, inputFilename, execOptions);
    }

    override processExecutionResult(input: UnprocessedExecResult, inputFilename?: string): BasicExecutionResult {
        return {
            ...input,
            stdout: parseMachDiagnostics(input.stdout, inputFilename, this.stdPath),
            stderr: parseMachDiagnostics(input.stderr, inputFilename, this.stdPath),
        };
    }
}
