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

import type {
    BypassCache,
    CacheKey,
    CompilationCacheKey,
    CompilationResult,
    ExecutionOptionsWithEnv,
    FiledataPair,
} from '../../types/compilation/compilation.interfaces.js';
import {
    CompilerOverrideType,
    type ConfiguredOverrideGeneral,
    type ConfiguredOverrides,
} from '../../types/compilation/compiler-overrides.interfaces.js';
import type {MachTarget, PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {
    BasicExecutionResult,
    ExecutableExecutionOptions,
    UnprocessedExecResult,
} from '../../types/execution/execution.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import type {InstructionSet} from '../../types/instructionsets.js';
import type {ResultLine} from '../../types/resultline/resultline.interfaces.js';
import {unwrap} from '../assert.js';
import {BaseCompiler} from '../base-compiler.js';
import {CompilationEnvironment} from '../compilation-env.js';
import {logger} from '../logger.js';
import {LlvmObjdumper} from '../objdumper/llvm.js';
import {ClientOptionsType} from '../options-handler.js';
import {MACH_MODULE_SEP, MachAsmParser} from '../parsers/asm-parser-mach.js';
import {parseMachDiagnostics, toEditorColumns} from '../parsers/mach-diagnostics.js';
import {MachIrParser} from '../parsers/mach-ir.js';
import * as temp from '../temp.js';
import * as utils from '../utils.js';
import {MachParser} from './argument-parsers.js';

const sourceDir = 'src';
const workDir = 'out';

/** What `--emit` writes for the modules of the user's project, each kind under `out/<product>/<id>/`. */
type Product = 'asm' | 'obj' | 'ir-opt';

/** A tuple `mach info targets` lists, before the probe has settled whether it builds and what its asm view reads. */
type MachTuple = Omit<MachTarget, 'listing'>;

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

function instructionSetForArch(arch: string): InstructionSet | undefined {
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

/** The target a command line names, the last `-t` winning, and the command line without it. */
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
 * A compilation is an entry build (`mach build src/example.mach`), which needs no manifest: `--source <id>=src` binds
 * the user's source tree and `--source std=<stdPath>` the std release the compiler is paired with, read in place.
 *
 *   src/example.mach            the user's entry module (and any extra files, rooted at src/)
 *   out/asm/example/*.s         masc assembly listings (`--emit asm`), on a target that has them
 *   out/ir-opt/example/*.ir     optimised/legalised MIRL text for the Mach IR pane (`--emit ir-opt`)
 *   out/obj/example/*.o         per-module objects (`--emit obj`), the asm view of a target with no listing
 *   out/bin/example[.spv]       linked executable or finished SPIR-V module (`--emit exe`)
 *
 * std's own modules are written beside the project's, under `out/<product>/std/`, and no view reads them.
 */
export class MachCompiler extends BaseCompiler {
    static get key() {
        return 'mach';
    }

    readonly stdPath: string;
    private readonly spirvDisassembler: string;
    private readonly machAsm: MachAsmParser;
    private buildable?: MachTarget[];

    constructor(info: PreliminaryCompilerInfo, env: CompilationEnvironment) {
        super(info, env);
        this.stdPath =
            this.compilerProps<string>(`compiler.${this.compiler.id}.stdPath`) ??
            path.join(path.dirname(this.compiler.exe), 'std');
        this.spirvDisassembler = this.compilerProps<string>('disassemblerPath') || 'spirv-dis';
        this.compiler.supportsTarget = true;
        this.compiler.supportsMachIrView = true;
        this.machAsm = new MachAsmParser(this.compilerProps, sourceDir, this.compileFilename);
        this.asm = this.machAsm;
    }

    override async initialise(
        mtime: Date,
        clientOptions: ClientOptionsType,
        isPrediscovered = false,
    ): Promise<BaseCompiler | null> {
        if (isPrediscovered && this.compiler.cachedMachTargets) {
            this.buildable = this.compiler.cachedMachTargets;
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
    async supportedTuples(): Promise<MachTuple[]> {
        const result = await this.exec(this.compiler.exe, ['info', 'targets'], this.getDefaultExecOptions());
        if (result.code !== 0) throw new Error(`mach info targets failed: ${result.stderr}`);

        const rows: MachTuple[] = [];
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

    /** A target by the name it is offered under, or by its platform, which means that platform's default ABI. */
    private findTarget(name: string): MachTarget | undefined {
        const rows = this.targets();
        return (
            rows.find(t => t.name === name) ??
            rows.find(t => t.platform === name && t.defaultAbi) ??
            rows.find(t => t.platform === name)
        );
    }

    /**
     * The target a request builds for. A `-t` among the options wins over the target picker, as it does on the
     * command line `orderArguments` writes, and a request that names neither builds for the host.
     */
    private requestedTarget(options: string[], overrides: ConfiguredOverrides = []): MachTarget | undefined {
        const picked = overrides.find(
            (ov): ov is ConfiguredOverrideGeneral => ov.name === CompilerOverrideType.arch,
        )?.value;
        return this.findTarget(extractTargetOption(options).targetName || picked || hostPlatform());
    }

    targetFlags(target: MachTuple, explicitArgs: string[] = []): string[] {
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
            for (const [index, tuple] of tuples.entries()) {
                const target = await this.probe(dirPath, index, tuple);
                if (target) rows.push(target);
            }
            return qualifyNames(rows);
        } finally {
            temp.release(dirPath);
            await fs.rm(dirPath, {recursive: true, force: true});
        }
    }

    /**
     * A tuple is offered when the asm view of it builds: the listing where the compiler writes one, and otherwise the
     * objects, which is all there is to read on a target the compiler encodes without a text form.
     */
    private async probe(dirPath: string, index: number, tuple: MachTuple): Promise<MachTarget | undefined> {
        if (await this.buildsUnderProbe(dirPath, `w${index}-asm`, tuple, 'asm')) return {...tuple, listing: true};
        if (await this.buildsUnderProbe(dirPath, `w${index}-obj`, tuple, 'obj')) return {...tuple, listing: false};
        return undefined;
    }

    private async buildsUnderProbe(dirPath: string, work: string, tuple: MachTuple, emit: Product): Promise<boolean> {
        const sub = stdPrintSystems.has(tuple.sys) ? 'hosted' : 'bare';
        const args = [
            'build',
            `${sub}/probe.mach`,
            '--source',
            `probe=${sub}`,
            '--source',
            `std=${this.stdPath}`,
            '-w',
            work,
            '--debug',
            '--emit',
            emit,
            ...this.targetFlags(tuple),
        ];
        if (!tuple.kind && sub === 'bare') args.push('-k', 'object');
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

    override fixFiltersForCompiler(
        filters: ParseFiltersAndOutputOptions,
        options: string[],
        files: FiledataPair[],
        backendOptions: Record<string, any>,
    ) {
        super.fixFiltersForCompiler(filters, options, files, backendOptions);
        const target = this.requestedTarget(options, backendOptions.overrides);
        if (target && !target.listing && !filters.binary) filters.binaryObject = true;
    }

    override getOutputFilename(dirPath: string, outputFilebase: string, key?: CacheKey | CompilationCacheKey) {
        const filters = this.isCacheKey(key) ? key.filters : undefined;
        if (filters?.binary) return this.getExecutableFilename(dirPath, outputFilebase, key);
        const id = this.projectId;
        return filters?.binaryObject
            ? path.join(dirPath, workDir, 'obj', id, `${id}.o`)
            : path.join(dirPath, workDir, 'asm', id, `${id}.s`);
    }

    override getSharedLibraryPathsAsArguments(): string[] {
        return [];
    }

    override getExecutableFilename(dirPath: string, outputFilebase: string, key?: CacheKey | CompilationCacheKey) {
        const target = this.isCacheKey(key)
            ? this.requestedTarget(key.options, key.backendOptions.overrides)
            : undefined;
        const suffix = target?.sys === 'vulkan' ? '.spv' : '';
        return path.join(dirPath, workDir, 'bin', `${this.projectId}${suffix}`);
    }

    /** One product of the project's modules as the build in `dirPath` wrote it, the entry first and the rest by path. */
    private async moduleOutputs(dirPath: string, product: Product, ext: string): Promise<string[]> {
        const dir = path.join(dirPath, workDir, product, this.projectId);
        if (!(await utils.dirExists(dir))) return [];
        const entry = `${this.projectId}${ext}`;
        const names = (await fs.readdir(dir, {recursive: true})).filter(name => name.endsWith(ext));
        const others = names.filter(name => name !== entry).sort();
        return [...names.filter(name => name === entry), ...others].map(name => path.join(dir, name));
    }

    override async getObjdumpInputFilenames(
        outputFilename: string,
        filters?: ParseFiltersAndOutputOptions,
        dirPath?: string,
    ) {
        if (filters?.binary) return [outputFilename];
        const objects = await this.moduleOutputs(unwrap(dirPath), 'obj', '.o');
        return objects.length > 0 ? objects : [outputFilename];
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string) {
        if (filters.binary) {
            return ['--emit', 'exe', '-o', path.posix.join(workDir, 'bin', path.basename(outputFilename))];
        }
        return ['--emit', filters.binaryObject ? 'obj' : 'asm'];
    }

    override optionsForBackend(backendOptions: Record<string, any>, outputFilename: string) {
        const options = super.optionsForBackend(backendOptions, outputFilename);
        if (backendOptions.produceMachIr) options.push('--emit', 'ir-opt');
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
        const {targetName, rest} = extractTargetOption([...options, ...userOptions]);
        const target = this.findTarget(targetName || hostPlatform());

        const args = [
            'build',
            path.posix.join(sourceDir, this.compileFilename),
            '--source',
            `${this.projectId}=${sourceDir}`,
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
            // not one of the targets on offer: mach names what is wrong with it better than a refusal here would
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

        const result = await super.runCompiler(compiler, options, inputFilename, execOptions, filters);
        const target = this.requestedTarget(options);
        if (target) {
            result.instructionSet = instructionSetForArch(target.arch);
            if (target.sys === 'vulkan') result.languageId = 'spirv';
        }
        result.stderr = await this.inEditorColumns(result.stderr, path.join(projectRoot, sourceDir));
        return result;
    }

    protected override getObjdumperForResult(result: CompilationResult) {
        // the base class leaves the host's own architecture to GNU objdump, which reads no Mach-O or PE image
        const target = this.requestedTarget(result.compilationOptions ?? []);
        if (this.compiler.llvmObjdumper && target && target.sys !== 'linux') {
            return {exe: this.compiler.llvmObjdumper, cls: LlvmObjdumper};
        }
        return super.getObjdumperForResult(result);
    }

    private async disassembleSpirv(files: string[], dirPath: string): Promise<string> {
        const modules: string[] = [];
        for (const file of files) {
            if (!(await utils.fileExists(file))) continue;
            const spv = await this.exec(this.spirvDisassembler, [file, '--comment'], {
                ...this.getDefaultExecOptions(),
                customCwd: dirPath,
            });
            if (spv.code === 0) {
                modules.push(spv.stdout);
            } else {
                logger.error('SPIR-V disassembly failed', spv);
            }
        }
        return modules.length > 0 ? modules.join(MACH_MODULE_SEP) : '<No SPIR-V output generated>';
    }

    override async checkOutputFileAndDoPostProcess(
        asmResult: CompilationResult,
        outputFilename: string,
        filters: ParseFiltersAndOutputOptions,
        produceOptRemarks = false,
    ): Promise<[CompilationResult, any[], any[]]> {
        const dirPath = unwrap(asmResult.dirPath);

        if (asmResult.languageId === 'spirv') {
            const modules = filters.binary ? [outputFilename] : await this.moduleOutputs(dirPath, 'obj', '.o');
            asmResult.asm = await this.disassembleSpirv(modules, dirPath);
            return [asmResult, [], []];
        }

        if (!filters.binary && !filters.binaryObject) {
            const listings = await this.moduleOutputs(dirPath, 'asm', '.s');
            if (listings.length > 0) {
                const asm = (await Promise.all(listings.map(file => fs.readFile(file, 'utf8')))).join(MACH_MODULE_SEP);
                asmResult.asmSize = Buffer.byteLength(asm, 'utf8');
                asmResult.asm = asm;
                return [asmResult, [], []];
            }
        }

        return super.checkOutputFileAndDoPostProcess(asmResult, outputFilename, filters, produceOptRemarks);
    }

    override async processAsm(result, filters: ParseFiltersAndOutputOptions, options: string[]) {
        if (result.languageId === 'spirv') return this.machAsm.processSpirv(result.asm, filters);
        return super.processAsm(result, filters, options);
    }

    override async processMachIrOutput(inputFilename: string, output: CompilationResult) {
        if (output.code !== 0) return [{text: 'Failed to run compiler to get Mach IR'}];
        const irFiles = await this.moduleOutputs(this.projectRoot(inputFilename), 'ir-opt', '.ir');
        if (irFiles.length === 0) return [{text: 'Internal error; unable to open output path'}];
        const modules = await Promise.all(irFiles.map(file => fs.readFile(file, 'utf8')));
        return new MachIrParser(path.posix.join(sourceDir, this.compileFilename)).processModules(modules);
    }

    /**
     * mach counts columns in UTF-8 bytes and the editor in UTF-16 units, so every marker is remapped against the line
     * it marks.
     */
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
        const target = this.requestedTarget(key.options, key.backendOptions.overrides);
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
