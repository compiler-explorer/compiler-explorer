// Copyright (c) 2015, Compiler Explorer Authors
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

import path from 'path';

import fs from 'fs-extra';
import * as PromClient from 'prom-client';
import temp from 'temp';
import _ from 'underscore';

import {
    BufferOkFunc,
    BuildResult,
    BuildStep,
    BypassCache,
    CompilationCacheKey,
    CompilationInfo,
    CompilationResult,
    CustomInputForTool,
    ExecutionOptions,
    bypassCompilationCache,
    bypassExecutionCache,
} from '../types/compilation/compilation.interfaces.js';
import type {
    LLVMOptPipelineBackendOptions,
    LLVMOptPipelineOutput,
} from '../types/compilation/llvm-opt-pipeline-output.interfaces.js';
import type {CompilerInfo, ICompiler, PreliminaryCompilerInfo} from '../types/compiler.interfaces.js';
import type {
    BasicExecutionResult,
    ExecutableExecutionOptions,
    UnprocessedExecResult,
} from '../types/execution/execution.interfaces.js';
import type {CompilerOutputOptions, ParseFiltersAndOutputOptions} from '../types/features/filters.interfaces.js';
import type {Language} from '../types/languages.interfaces.js';
import type {Library, LibraryVersion, SelectedLibraryVersion} from '../types/libraries/libraries.interfaces.js';
import type {ResultLine} from '../types/resultline/resultline.interfaces.js';
import type {Artifact, ToolResult, ToolTypeKey} from '../types/tool.interfaces.js';

import {BuildEnvSetupBase, getBuildEnvTypeByKey} from './buildenvsetup/index.js';
import type {BuildEnvDownloadInfo} from './buildenvsetup/buildenv.interfaces.js';
import * as cfg from './cfg/cfg.js';
import {CompilationEnvironment} from './compilation-env.js';
import {CompilerArguments} from './compiler-arguments.js';
import {ClangCParser, ClangParser, GCCCParser, GCCParser, ICCParser} from './compilers/argument-parsers.js';
import {BaseDemangler, getDemanglerTypeByKey} from './demangler/index.js';
import {LLVMIRDemangler} from './demangler/llvm.js';
import * as exec from './exec.js';
import {getExternalParserByKey} from './external-parsers/index.js';
import {ExternalParserBase} from './external-parsers/base.js';
import {InstructionSets} from './instructionsets.js';
import {languages} from './languages.js';
import {LlvmAstParser} from './llvm-ast.js';
import {LlvmIrParser} from './llvm-ir.js';
import * as compilerOptInfo from './llvm-opt-transformer.js';
import * as StackUsageTransformer from './stack-usage-transformer.js';
import {logger} from './logger.js';
import {getObjdumperTypeByKey} from './objdumper/index.js';
import {Packager} from './packager.js';
import {AsmParser} from './parsers/asm-parser.js';
import type {IAsmParser} from './parsers/asm-parser.interfaces.js';
import {LlvmPassDumpParser} from './parsers/llvm-pass-dump-parser.js';
import type {PropertyGetter} from './properties.interfaces.js';
import {
    clang_style_sysroot_flag,
    getSpecificTargetBasedOnToolchainPath,
    getSysrootByToolchainPath,
    getToolchainFlagFromOptions,
    getToolchainPath,
    hasSysrootArg,
    hasToolchainArg,
    removeToolchainArg,
    replaceSysrootArg,
    replaceToolchainArg,
} from './toolchain-utils.js';
import type {ITool} from './tooling/base-tool.interface.js';
import * as utils from './utils.js';
import {unwrap} from './assert.js';
import {
    CompilerOverrideOption,
    CompilerOverrideOptions,
    CompilerOverrideType,
    ConfiguredOverrides,
} from '../types/compilation/compiler-overrides.interfaces.js';
import {LLVMIrBackendOptions} from '../types/compilation/ir.interfaces.js';
import {ParsedAsmResultLine} from '../types/asmresult/asmresult.interfaces.js';

const compilationTimeHistogram = new PromClient.Histogram({
    name: 'ce_base_compiler_compilation_duration_seconds',
    help: 'Time taken to compile code',
    buckets: [0.1, 0.5, 1, 5, 10, 20, 30],
});

const compilationQueueTimeHistogram = new PromClient.Histogram({
    name: 'ce_base_compiler_compilation_queue_seconds',
    help: 'Time requests spent in queue pending compilation',
    buckets: [0.1, 0.5, 1, 5, 10, 20, 30],
});

const executionTimeHistogram = new PromClient.Histogram({
    name: 'ce_base_compiler_execution_duration_seconds',
    help: 'Time taken to execute code',
    buckets: [0.1, 0.5, 1, 5, 10, 20, 30],
});

const executionQueueTimeHistogram = new PromClient.Histogram({
    name: 'ce_base_compiler_execution_queue_seconds',
    help: 'Time requests spent in the queue pending execution',
    buckets: [0.1, 0.5, 1, 5, 10, 20, 30],
});

export const c_default_target_description =
    'Change the target architecture of the compiler. ' +
    'Be aware that the architecture might not be fully supported by the compiler ' +
    'even though the option is available. ' +
    'The compiler might also require additional arguments to be fully functional.';

export const c_default_toolchain_description =
    'Change the default GCC toolchain for this compiler. ' +
    'This may or may not affect header usage (e.g. libstdc++ version) and linking to GCCs pre-built binaries.';

export const c_value_placeholder = '<value>';

export class BaseCompiler implements ICompiler {
    protected compiler: CompilerInfo; // TODO: Some missing types still present in Compiler type
    public lang: Language;
    protected compileFilename: string;
    protected env: CompilationEnvironment;
    protected compilerProps: PropertyGetter;
    protected alwaysResetLdPath: any;
    protected delayCleanupTemp: any;
    protected stubRe: RegExp;
    protected stubText: string;
    protected compilerWrapper: any;
    protected asm: IAsmParser;
    protected llvmIr: LlvmIrParser;
    protected llvmPassDumpParser: LlvmPassDumpParser;
    protected llvmAst: LlvmAstParser;
    protected toolchainPath: any;
    public possibleArguments: CompilerArguments;
    protected possibleTools: ITool[];
    protected demanglerClass: typeof BaseDemangler | null = null;
    protected objdumperClass: any;
    public outputFilebase: string;
    protected mtime: Date | null = null;
    protected cmakeBaseEnv: Record<string, string>;
    protected buildenvsetup: null | any;
    protected externalparser: null | ExternalParserBase;
    protected supportedLibraries?: Record<string, Library>;
    protected packager: Packager;
    private static objdumpAndParseCounter = new PromClient.Counter({
        name: 'ce_objdumpandparsetime_total',
        help: 'Time spent on objdump and parsing of objdumps',
        labelNames: [],
    });

    constructor(compilerInfo: PreliminaryCompilerInfo & {disabledFilters?: string[]}, env: CompilationEnvironment) {
        // Information about our compiler
        // By the end of construction / initialise() everything will be populated for CompilerInfo
        this.compiler = compilerInfo as CompilerInfo;
        this.lang = languages[compilerInfo.lang];
        if (!this.lang) {
            throw new Error(`Missing language info for ${compilerInfo.lang}`);
        }
        this.compileFilename = `example${this.lang.extensions[0]}`;
        this.env = env;
        // Partial application of compilerProps with the proper language id applied to it
        this.compilerProps = _.partial(this.env.compilerProps as any, this.lang.id);
        this.compiler.supportsIntel = !!this.compiler.intelAsm;

        this.alwaysResetLdPath = this.env.ceProps('alwaysResetLdPath');
        this.delayCleanupTemp = this.env.ceProps('delayCleanupTemp', false);
        this.stubRe = new RegExp(this.compilerProps('stubRe', ''));
        this.stubText = this.compilerProps('stubText', '');
        this.compilerWrapper = this.compilerProps('compiler-wrapper');

        if (!this.compiler.options) this.compiler.options = '';
        if (!this.compiler.optArg) this.compiler.optArg = '';
        if (!this.compiler.supportsOptOutput) this.compiler.supportsOptOutput = false;

        if (!compilerInfo.disabledFilters) this.compiler.disabledFilters = [];
        else if (typeof (this.compiler.disabledFilters as any) === 'string') {
            // When first loaded from props it may be a string so we split it here
            // I'd like a better way to do this that doesn't involve type hacks
            // TODO(jeremy-rifkin): branch may now be obsolete?
            this.compiler.disabledFilters = (this.compiler.disabledFilters as any).split(',');
        }

        this.asm = new AsmParser(this.compilerProps);
        const irDemangler = new LLVMIRDemangler(this.compiler.demangler, this);
        this.llvmIr = new LlvmIrParser(this.compilerProps, irDemangler);
        this.llvmPassDumpParser = new LlvmPassDumpParser(this.compilerProps);
        this.llvmAst = new LlvmAstParser(this.compilerProps);

        this.toolchainPath = getToolchainPath(this.compiler.exe, this.compiler.options);

        this.possibleArguments = new CompilerArguments(this.compiler.id);
        this.possibleTools = _.values(compilerInfo.tools) as ITool[];
        const demanglerExe = this.compiler.demangler;
        if (demanglerExe && this.compiler.demanglerType) {
            this.demanglerClass = getDemanglerTypeByKey(this.compiler.demanglerType);
        }
        const objdumperExe = this.compiler.objdumper;
        if (objdumperExe && this.compiler.objdumperType) {
            this.objdumperClass = getObjdumperTypeByKey(this.compiler.objdumperType);
        }

        this.outputFilebase = 'output';

        this.cmakeBaseEnv = {};

        this.buildenvsetup = null;
        if (!this.getRemote() && this.compiler.buildenvsetup && this.compiler.buildenvsetup.id) {
            const buildenvsetupclass = getBuildEnvTypeByKey(this.compiler.buildenvsetup.id);
            this.buildenvsetup = new buildenvsetupclass(this.compiler, this.env);
        }

        this.externalparser = null;
        if (!this.getRemote() && this.compiler.externalparser && this.compiler.externalparser.id) {
            const externalparserclass = getExternalParserByKey(this.compiler.externalparser.id);
            this.externalparser = new externalparserclass(this.compiler, this.env, this.exec);
        }

        if (!this.compiler.instructionSet) {
            const isets = new InstructionSets();
            if (this.buildenvsetup) {
                isets
                    .getCompilerInstructionSetHint(this.buildenvsetup.compilerArch, this.compiler.exe)
                    .then(res => (this.compiler.instructionSet = res))
                    .catch(() => {});
            } else {
                const temp = new BuildEnvSetupBase(this.compiler, this.env);
                isets
                    .getCompilerInstructionSetHint(temp.compilerArch, this.compiler.exe)
                    .then(res => (this.compiler.instructionSet = res))
                    .catch(() => {});
            }
        }

        this.packager = new Packager();
    }

    copyAndFilterLibraries(allLibraries, filter) {
        const filterLibAndVersion = _.map(filter, lib => {
            const match = lib.match(/([\w-]*)\.([\w-]*)/i);
            if (match) {
                return {
                    id: match[1],
                    version: match[2],
                };
            } else {
                return {
                    id: lib,
                    version: false,
                };
            }
        });

        const filterLibIds = new Set();
        _.each(filterLibAndVersion, lib => {
            filterLibIds.add(lib.id);
        });

        const copiedLibraries = {};
        _.each(allLibraries, (lib, libid) => {
            if (!filterLibIds.has(libid)) return;

            const libcopy = Object.assign({}, lib);
            libcopy.versions = _.omit(lib.versions, (version, versionid) => {
                for (const filter of filterLibAndVersion) {
                    if (filter.id === libid) {
                        if (!filter.version) return false;
                        if (filter.version === versionid) return false;
                    }
                }

                return true;
            });

            copiedLibraries[libid] = libcopy;
        });

        return copiedLibraries;
    }

    getSupportedLibraries(supportedLibrariesArr, allLibs) {
        if (supportedLibrariesArr.length > 0) {
            return this.copyAndFilterLibraries(allLibs, supportedLibrariesArr);
        }
        return allLibs;
    }

    async getCmakeBaseEnv() {
        if (!this.compiler.exe) return {};

        const env: Record<string, string> = {};

        if (this.lang.id === 'c++') {
            env.CXX = this.compiler.exe;

            if (this.compiler.exe.endsWith('clang++.exe')) {
                env.CC = this.compiler.exe.substring(0, this.compiler.exe.length - 6) + '.exe';
            } else if (this.compiler.exe.endsWith('g++.exe')) {
                env.CC = this.compiler.exe.substring(0, this.compiler.exe.length - 6) + 'cc.exe';
            } else if (this.compiler.exe.endsWith('clang++')) {
                env.CC = this.compiler.exe.substring(0, this.compiler.exe.length - 2);
            } else if (this.compiler.exe.endsWith('g++')) {
                env.CC = this.compiler.exe.substring(0, this.compiler.exe.length - 2) + 'cc';
            }
        } else if (this.lang.id === 'fortran') {
            env.FC = this.compiler.exe;
        } else if (this.lang.id === 'cuda') {
            env.CUDACXX = this.compiler.exe;
        } else {
            env.CC = this.compiler.exe;
        }

        // TODO(#5051): support changing of toolchainPath per compile
        if (this.toolchainPath) {
            if (process.platform === 'win32') {
                const ldPath = `${this.toolchainPath}/bin/ld.exe`;
                const arPath = `${this.toolchainPath}/bin/ar.exe`;
                const asPath = `${this.toolchainPath}/bin/as.exe`;

                if (await utils.fileExists(ldPath)) env.LD = ldPath;
                if (await utils.fileExists(arPath)) env.AR = arPath;
                if (await utils.fileExists(asPath)) env.AS = asPath;
            } else {
                const ldPath = `${this.toolchainPath}/bin/ld`;
                const arPath = `${this.toolchainPath}/bin/ar`;
                const asPath = `${this.toolchainPath}/bin/as`;

                if (await utils.fileExists(ldPath)) env.LD = ldPath;
                if (await utils.fileExists(arPath)) env.AR = arPath;
                if (await utils.fileExists(asPath)) env.AS = asPath;
            }
        }

        return env;
    }

    newTempDir(): Promise<string> {
        return new Promise((resolve, reject) => {
            temp.mkdir({prefix: 'compiler-explorer-compiler', dir: process.env.tmpDir}, (err, dirPath) => {
                if (err) reject(`Unable to open temp file: ${err}`);
                else resolve(dirPath);
            });
        });
    }

    optOutputRequested(options: string[]) {
        return options.includes('-fsave-optimization-record');
    }

    getRemote() {
        if (this.compiler.remote) return this.compiler.remote;
        return false;
    }

    async exec(filepath: string, args: string[], execOptions: ExecutionOptions) {
        // Here only so can be overridden by compiler implementations.
        return await exec.execute(filepath, args, execOptions);
    }

    protected getCompilerCacheKey(compiler, args, options): CompilationCacheKey {
        return {mtime: this.mtime, compiler, args, options};
    }

    protected async execCompilerCached(compiler, args, options) {
        if (this.mtime === null) {
            throw new Error('Attempt to access cached compiler before initialise() called');
        }
        if (!options) {
            options = this.getDefaultExecOptions();
            options.timeoutMs = 0;
            options.ldPath = this.getSharedLibraryPathsAsLdLibraryPaths([]);
        }

        // Take a (shallow) copy of the options before we add a random customCwd: The fact we have createAndUseTempDir
        // set is enough to make us different from an otherwise identical run without createAndUseTempDir. However, the
        // actual random path is unimportant for caching; and its presence prevents cache hits.
        const optionsForCache = {...options};
        if (options.createAndUseTempDir) {
            options.customCwd = await this.newTempDir();
        }

        const key = this.getCompilerCacheKey(compiler, args, optionsForCache);
        let result = await this.env.compilerCacheGet(key as any);
        if (!result) {
            result = await this.env.enqueue(async () => await exec.execute(compiler, args, options));
            if (result.okToCache) {
                this.env
                    .compilerCachePut(key as any, result, undefined)
                    .then(() => {
                        // Do nothing, but we don't await here.
                    })
                    .catch(e => {
                        logger.info('Uncaught exception caching compilation results', e);
                    });
            }
        }

        if (options.createAndUseTempDir) fs.remove(options.customCwd, () => {});

        return result;
    }

    getDefaultExecOptions(): ExecutionOptions {
        return {
            timeoutMs: this.env.ceProps('compileTimeoutMs', 7500),
            maxErrorOutput: this.env.ceProps('max-error-output', 5000),
            env: this.env.getEnv(this.compiler.needsMulti),
            wrapper: this.compilerWrapper,
        };
    }

    getCompilerResultLanguageId(): string | undefined {
        return undefined;
    }

    async runCompiler(
        compiler: string,
        options: string[],
        inputFilename: string,
        execOptions: ExecutionOptions,
    ): Promise<CompilationResult> {
        if (!execOptions) {
            execOptions = this.getDefaultExecOptions();
        }

        if (!execOptions.customCwd) {
            execOptions.customCwd = path.dirname(inputFilename);
        }

        const result = await this.exec(compiler, options, execOptions);
        return {
            ...this.transformToCompilationResult(result, inputFilename),
            languageId: this.getCompilerResultLanguageId(),
        };
    }

    async runCompilerRawOutput(compiler, options, inputFilename, execOptions) {
        if (!execOptions) {
            execOptions = this.getDefaultExecOptions();
        }

        if (!execOptions.customCwd) {
            execOptions.customCwd = path.dirname(inputFilename);
        }

        const result = await this.exec(compiler, options, execOptions);
        return {
            ...result,
            inputFilename,
        };
    }

    supportsObjdump() {
        return !!this.objdumperClass;
    }

    getObjdumpOutputFilename(defaultOutputFilename: string): string {
        return defaultOutputFilename;
    }

    postProcessObjdumpOutput(output) {
        return output;
    }

    async objdump(
        outputFilename,
        result: any,
        maxSize: number,
        intelAsm,
        demangle,
        staticReloc: boolean | undefined,
        dynamicReloc: boolean,
        filters: ParseFiltersAndOutputOptions,
    ) {
        outputFilename = this.getObjdumpOutputFilename(outputFilename);

        if (!(await utils.fileExists(outputFilename))) {
            result.asm = '<No output file ' + outputFilename + '>';
            return result;
        }

        const objdumper = new this.objdumperClass();
        const args = objdumper.getDefaultArgs(
            outputFilename,
            demangle,
            intelAsm,
            staticReloc,
            dynamicReloc,
            this.compiler.objdumperArgs,
        );

        if (this.externalparser) {
            const objResult = await this.externalparser.objdumpAndParseAssembly(result.dirPath, args, filters);
            if (objResult.parsingTime !== undefined) {
                objResult.objdumpTime = parseInt(result.execTime) - parseInt(result.parsingTime);
                delete objResult.execTime;
            }

            result = {...result, ...objResult};
        } else {
            const execOptions: ExecutionOptions = {
                maxOutput: maxSize,
                customCwd: (result.dirPath as string) || path.dirname(outputFilename),
            };
            const objResult = await this.exec(this.compiler.objdumper, args, execOptions);

            if (objResult.code === 0) {
                result.objdumpTime = objResult.execTime;
                result.asm = this.postProcessObjdumpOutput(objResult.stdout);
            } else {
                logger.error(`Error executing objdump ${this.compiler.objdumper}`, objResult);
                result.asm = `<No output: objdump returned ${objResult.code}>`;
            }
        }

        return result;
    }

    processExecutionResult(input: UnprocessedExecResult, inputFilename?: string): BasicExecutionResult {
        const start = performance.now();
        const stdout = utils.parseOutput(input.stdout, inputFilename);
        const stderr = utils.parseOutput(input.stderr, inputFilename);
        const end = performance.now();
        return {
            ...input,
            stdout,
            stderr,
            processExecutionResultTime: end - start,
        };
    }

    getEmptyExecutionResult(): BasicExecutionResult {
        return {
            code: -1,
            okToCache: false,
            filenameTransform: x => x,
            stdout: [],
            stderr: [],
            execTime: '',
            timedOut: false,
        };
    }

    transformToCompilationResult(input: UnprocessedExecResult, inputFilename): CompilationResult {
        const transformedInput = input.filenameTransform(inputFilename);

        return {
            inputFilename,
            languageId: input.languageId,
            ...this.processExecutionResult(input, transformedInput),
        };
    }

    async execBinary(
        executable,
        maxSize,
        executeParameters: ExecutableExecutionOptions,
        homeDir,
    ): Promise<BasicExecutionResult> {
        // We might want to save this in the compilation environment once execution is made available
        const timeoutMs = this.env.ceProps('binaryExecTimeoutMs', 2000);
        try {
            const execResult: UnprocessedExecResult = await exec.sandbox(executable, executeParameters.args, {
                maxOutput: maxSize,
                timeoutMs: timeoutMs,
                ldPath: _.union(this.compiler.ldPath, executeParameters.ldPath),
                input: executeParameters.stdin,
                env: executeParameters.env,
                customCwd: homeDir,
                appHome: homeDir,
            });

            return this.processExecutionResult(execResult);
        } catch (err: UnprocessedExecResult | any) {
            if (err.code && err.stderr) {
                return this.processExecutionResult(err);
            } else {
                return {
                    ...this.getEmptyExecutionResult(),
                    stdout: err.stdout ? utils.parseOutput(err.stdout) : [],
                    stderr: err.stderr ? utils.parseOutput(err.stderr) : [],
                    code: err.code === undefined ? -1 : err.code,
                };
            }
        }
    }

    protected filename(fn) {
        return fn;
    }

    getGccDumpFileName(outputFilename: string) {
        return outputFilename.replace(path.extname(outputFilename), '.dump');
    }

    getGccDumpOptions(gccDumpOptions, outputFilename: string) {
        const addOpts = ['-fdump-passes'];

        // Build dump options to append to the end of the -fdump command-line flag.
        // GCC accepts these options as a list of '-' separated names that may
        // appear in any order.
        let flags = '';
        if (gccDumpOptions.dumpFlags.address !== false) {
            flags += '-address';
        }
        if (gccDumpOptions.dumpFlags.slim !== false) {
            flags += '-slim';
        }
        if (gccDumpOptions.dumpFlags.raw !== false) {
            flags += '-raw';
        }
        if (gccDumpOptions.dumpFlags.details !== false) {
            flags += '-details';
        }
        if (gccDumpOptions.dumpFlags.stats !== false) {
            flags += '-stats';
        }
        if (gccDumpOptions.dumpFlags.blocks !== false) {
            flags += '-blocks';
        }
        if (gccDumpOptions.dumpFlags.vops !== false) {
            flags += '-vops';
        }
        if (gccDumpOptions.dumpFlags.lineno !== false) {
            flags += '-lineno';
        }
        if (gccDumpOptions.dumpFlags.uid !== false) {
            flags += '-uid';
        }
        if (gccDumpOptions.dumpFlags.all !== false) {
            flags += '-all';
        }

        // If we want to remove the passes that won't produce anything from the
        // drop down menu, we need to ask for all dump files and see what's
        // really created. This is currently only possible with regular GCC, not
        // for compilers that us libgccjit. The later can't easily move dump
        // files outside of the tempdir created on the fly.
        if (this.compiler.removeEmptyGccDump) {
            if (gccDumpOptions.treeDump !== false) {
                addOpts.push('-fdump-tree-all' + flags);
            }
            if (gccDumpOptions.rtlDump !== false) {
                addOpts.push('-fdump-rtl-all' + flags);
            }
            if (gccDumpOptions.ipaDump !== false) {
                addOpts.push('-fdump-ipa-all' + flags);
            }
        } else {
            // If not dumping everything, create a specific command like
            // -fdump-tree-fixup_cfg1-some-flags=somefilename
            if (gccDumpOptions.pass) {
                const dumpFile = this.getGccDumpFileName(outputFilename);
                const dumpCmd = gccDumpOptions.pass.command_prefix + flags + `=${dumpFile}`;
                addOpts.push(dumpCmd);
            }
        }
        return addOpts;
    }

    // Returns a list of additional options that may be required by some backend options.
    // Meant to be overloaded by compiler classes.
    // Default handles the GCC compiler with some debug dump enabled.
    optionsForBackend(backendOptions: Record<string, any>, outputFilename: string): string[] {
        let addOpts: string[] = [];

        if (backendOptions.produceGccDump && backendOptions.produceGccDump.opened && this.compiler.supportsGccDump) {
            addOpts = addOpts.concat(this.getGccDumpOptions(backendOptions.produceGccDump, outputFilename));
        }

        return addOpts;
    }

    protected optionsForFilter(
        filters: ParseFiltersAndOutputOptions,
        outputFilename: string,
        userOptions?: string[],
    ): string[] {
        let options = ['-g', '-o', this.filename(outputFilename)];
        if (this.compiler.intelAsm && filters.intel && !filters.binary && !filters.binaryObject) {
            options = options.concat(this.compiler.intelAsm.split(' '));
        }
        if (!filters.binary && !filters.binaryObject) options = options.concat('-S');
        else if (filters.binaryObject) options = options.concat('-c');

        return options;
    }

    findLibVersion(selectedLib: SelectedLibraryVersion): false | LibraryVersion {
        if (!this.supportedLibraries) return false;

        const foundLib = _.find(this.supportedLibraries, (o, libId) => libId === selectedLib.id);
        if (!foundLib) return false;

        const result: LibraryVersion | undefined = _.find(
            foundLib.versions,
            (o: LibraryVersion, versionId: string): boolean => {
                if (versionId === selectedLib.version) return true;
                return !!(o.alias && o.alias.includes(selectedLib.version));
            },
        );

        if (!result) return false;

        result.name = foundLib.name;
        return result;
    }

    findAutodetectStaticLibLink(linkname: string): SelectedLibraryVersion | false {
        const foundLib = _.findKey(this.supportedLibraries as Record<string, Library>, lib => {
            return (
                lib.versions.autodetect &&
                lib.versions.autodetect.staticliblink &&
                lib.versions.autodetect.staticliblink.includes(linkname)
            );
        });
        if (!foundLib) return false;

        return {
            id: foundLib,
            version: 'autodetect',
        };
    }

    getSortedStaticLibraries(libraries) {
        const dictionary = {};
        const links = _.uniq(
            _.flatten(
                _.map(libraries, selectedLib => {
                    const foundVersion = this.findLibVersion(selectedLib);
                    if (!foundVersion) return false;

                    return _.map(foundVersion.staticliblink, lib => {
                        if (lib) {
                            dictionary[lib] = foundVersion;
                            return [lib, foundVersion.dependencies];
                        } else {
                            return false;
                        }
                    });
                }),
            ),
        );

        const sortedlinks: string[] = [];

        _.each(links, libToInsertName => {
            const libToInsertObj = dictionary[libToInsertName];

            let idxToInsert = sortedlinks.length;
            for (const [idx, libCompareName] of sortedlinks.entries()) {
                const libCompareObj: LibraryVersion = dictionary[libCompareName];

                if (
                    libToInsertObj &&
                    libCompareObj &&
                    _.intersection(libToInsertObj.dependencies, libCompareObj.staticliblink).length > 0
                ) {
                    idxToInsert = idx;
                    break;
                } else if (libToInsertObj && libToInsertObj.dependencies.includes(libCompareName)) {
                    idxToInsert = idx;
                    break;
                } else if (libCompareObj && libCompareObj.dependencies.includes(libToInsertName)) {
                    continue;
                } else if (
                    libToInsertObj &&
                    libToInsertObj.staticliblink.includes(libToInsertName) &&
                    libToInsertObj.staticliblink.includes(libCompareName)
                ) {
                    if (
                        libToInsertObj.staticliblink.indexOf(libToInsertName) >
                        libToInsertObj.staticliblink.indexOf(libCompareName)
                    ) {
                        continue;
                    } else {
                        idxToInsert = idx;
                    }
                    break;
                } else if (
                    libCompareObj &&
                    libCompareObj.staticliblink.includes(libToInsertName) &&
                    libCompareObj.staticliblink.includes(libCompareName)
                ) {
                    if (
                        libCompareObj.staticliblink.indexOf(libToInsertName) >
                        libCompareObj.staticliblink.indexOf(libCompareName)
                    ) {
                        continue;
                    } else {
                        idxToInsert = idx;
                    }
                    break;
                }
            }

            if (idxToInsert < sortedlinks.length) {
                sortedlinks.splice(idxToInsert, 0, libToInsertName);
            } else {
                sortedlinks.push(libToInsertName);
            }
        });

        return sortedlinks;
    }

    getStaticLibraryLinks(libraries) {
        const linkFlag = this.compiler.linkFlag || '-l';

        return _.map(this.getSortedStaticLibraries(libraries), lib => {
            if (lib) {
                return linkFlag + lib;
            } else {
                return false;
            }
        }) as string[];
    }

    getSharedLibraryLinks(libraries: any[]): string[] {
        const linkFlag = this.compiler.linkFlag || '-l';

        return _.flatten(
            _.map(libraries, selectedLib => {
                const foundVersion = this.findLibVersion(selectedLib);
                if (!foundVersion) return false;

                return _.map(foundVersion.liblink, lib => {
                    if (lib) {
                        return linkFlag + lib;
                    } else {
                        return false;
                    }
                });
            }),
        ) as string[];
    }

    getSharedLibraryPaths(libraries) {
        return _.flatten(
            _.map(libraries, selectedLib => {
                const foundVersion = this.findLibVersion(selectedLib);
                if (!foundVersion) return false;

                const paths = [...foundVersion.libpath];
                if (!this.buildenvsetup.extractAllToRoot) {
                    paths.push(`./${selectedLib.id}/lib`);
                }
                return paths;
            }),
        ) as string[];
    }

    protected getSharedLibraryPathsAsArguments(libraries, libDownloadPath?: string, toolchainPath?: string) {
        const pathFlag = this.compiler.rpathFlag || '-Wl,-rpath,';
        const libPathFlag = this.compiler.libpathFlag || '-L';

        let toolchainLibraryPaths: string[] = [];
        if (toolchainPath) {
            toolchainLibraryPaths = [path.join(toolchainPath, '/lib64'), path.join(toolchainPath, '/lib32')];
        }

        if (!libDownloadPath) {
            libDownloadPath = './lib';
        }

        return _.union(
            [libPathFlag + libDownloadPath],
            [pathFlag + libDownloadPath],
            this.compiler.libPath.map(path => pathFlag + path),
            toolchainLibraryPaths.map(path => pathFlag + path),
            this.getSharedLibraryPaths(libraries).map(path => pathFlag + path),
            this.getSharedLibraryPaths(libraries).map(path => libPathFlag + path),
        ) as string[];
    }

    protected getSharedLibraryPathsAsLdLibraryPaths(libraries) {
        let paths = '';
        if (!this.alwaysResetLdPath) {
            paths = process.env.LD_LIBRARY_PATH || '';
        }
        return _.union(
            paths.split(path.delimiter).filter(p => !!p),
            this.compiler.ldPath,
            this.getSharedLibraryPaths(libraries),
        ) as string[];
    }

    getSharedLibraryPathsAsLdLibraryPathsForExecution(libraries) {
        let paths = '';
        if (!this.alwaysResetLdPath) {
            paths = process.env.LD_LIBRARY_PATH || '';
        }
        return _.union(
            paths.split(path.delimiter).filter(p => !!p),
            this.compiler.ldPath,
            this.compiler.libPath,
            this.getSharedLibraryPaths(libraries),
        );
    }

    getIncludeArguments(libraries: SelectedLibraryVersion[]): string[] {
        const includeFlag = this.compiler.includeFlag || '-I';
        return libraries.flatMap(selectedLib => {
            const foundVersion = this.findLibVersion(selectedLib);
            if (!foundVersion) return [];

            const paths = foundVersion.path.map(path => includeFlag + path);
            if (foundVersion.packagedheaders) {
                paths.push(includeFlag + `./${selectedLib.id}/include`);
            }
            return paths;
        });
    }

    getLibraryOptions(libraries: SelectedLibraryVersion[]): string[] {
        return libraries.flatMap(selectedLib => {
            const foundVersion = this.findLibVersion(selectedLib);
            if (!foundVersion) return [];
            return foundVersion.options;
        });
    }

    orderArguments(
        options: string[],
        inputFilename: string,
        libIncludes: string[],
        libOptions: string[],
        libPaths: string[],
        libLinks: string[],
        userOptions: string[],
        staticLibLinks: string[],
    ) {
        return options.concat(
            userOptions,
            [this.filename(inputFilename)],
            libIncludes,
            libOptions,
            libPaths,
            libLinks,
            staticLibLinks,
        );
    }

    getDefaultOrOverridenToolchainPath(overrides: ConfiguredOverrides): string {
        for (const override of overrides) {
            if (override.name !== CompilerOverrideType.env && override.value) {
                const possible = this.compiler.possibleOverrides?.find(ov => ov.name === override.name);
                if (possible && possible.name === CompilerOverrideType.toolchain) {
                    return override.value;
                }
            }
        }

        return this.toolchainPath;
    }

    getOverridenToolchainPath(overrides: ConfiguredOverrides): string | false {
        for (const override of overrides) {
            if (override.name !== CompilerOverrideType.env && override.value) {
                const possible = this.compiler.possibleOverrides?.find(ov => ov.name === override.name);
                if (possible && possible.name === CompilerOverrideType.toolchain) {
                    return override.value;
                }
            }
        }

        return false;
    }

    changeOptionsBasedOnOverrides(options: string[], overrides: ConfiguredOverrides): string[] {
        const overriddenToolchainPath = this.getOverridenToolchainPath(overrides);
        const sysrootPath: string | false =
            overriddenToolchainPath ?? getSysrootByToolchainPath(overriddenToolchainPath);
        const targetOverride = overrides.find(ov => ov.name === CompilerOverrideType.arch);
        const hasNeedForSysRoot =
            targetOverride && targetOverride.name !== CompilerOverrideType.env && !targetOverride.value.includes('x86');

        for (const override of overrides) {
            if (override.name !== CompilerOverrideType.env && override.value) {
                const possible = this.compiler.possibleOverrides?.find(ov => ov.name === override.name);
                if (!possible) continue;

                switch (possible.name) {
                    case CompilerOverrideType.toolchain: {
                        if (hasToolchainArg(options)) {
                            options = replaceToolchainArg(options, override.value);
                        } else {
                            for (const flag of possible.flags) {
                                options.push(flag.replace(c_value_placeholder, override.value));
                            }
                        }

                        if (sysrootPath) {
                            if (hasSysrootArg(options)) {
                                options = replaceSysrootArg(options, sysrootPath);
                            } else if (hasNeedForSysRoot) {
                                options.push(clang_style_sysroot_flag + sysrootPath);
                            }
                        }
                        break;
                    }
                    case CompilerOverrideType.arch: {
                        let betterTarget = override.value;
                        if (overriddenToolchainPath) {
                            betterTarget = getSpecificTargetBasedOnToolchainPath(
                                override.value,
                                overriddenToolchainPath,
                            );
                        }

                        for (const flag of possible.flags) {
                            options.push(flag.replace(c_value_placeholder, betterTarget));
                        }
                        break;
                    }
                    default: {
                        for (const flag of possible.flags) {
                            options.push(flag.replace(c_value_placeholder, override.value));
                        }
                        break;
                    }
                }
            }
        }

        return options;
    }

    prepareArguments(
        userOptions: string[],
        filters: ParseFiltersAndOutputOptions,
        backendOptions: Record<string, any>,
        inputFilename: string,
        outputFilename: string,
        libraries,
        overrides: ConfiguredOverrides,
    ) {
        let options = this.optionsForFilter(filters, outputFilename, userOptions);
        backendOptions = backendOptions || {};

        options = options.concat(this.optionsForBackend(backendOptions, outputFilename));

        if (this.compiler.options) {
            options = options.concat(utils.splitArguments(this.compiler.options));
        }

        if (this.compiler.supportsOptOutput && backendOptions.produceOptInfo) {
            options = options.concat(unwrap(this.compiler.optArg));
        }
        if (this.compiler.supportsStackUsageOutput && backendOptions.produceStackUsageInfo) {
            options = options.concat(unwrap(this.compiler.stackUsageArg));
        }

        const toolchainPath = this.getDefaultOrOverridenToolchainPath(backendOptions.overrides || []);

        const libIncludes = this.getIncludeArguments(libraries);
        const libOptions = this.getLibraryOptions(libraries);
        let libLinks: string[] = [];
        let libPaths: string[] = [];
        let staticLibLinks: string[] = [];

        if (filters.binary) {
            libLinks = this.getSharedLibraryLinks(libraries) || [];
            libPaths = this.getSharedLibraryPathsAsArguments(libraries, undefined, toolchainPath);
            staticLibLinks = this.getStaticLibraryLinks(libraries) || [];
        }

        userOptions = this.filterUserOptions(userOptions) || [];
        options = this.fixIncompatibleOptions(options, userOptions);
        options = this.changeOptionsBasedOnOverrides(options, overrides);

        return this.orderArguments(
            options,
            inputFilename,
            libIncludes,
            libOptions,
            libPaths,
            libLinks,
            userOptions,
            staticLibLinks,
        );
    }

    protected fixIncompatibleOptions(options: string[], userOptions: string[]): string[] {
        return options;
    }

    filterUserOptions(userOptions: string[]): string[] {
        return userOptions;
    }

    async generateAST(inputFilename, options) {
        // These options make Clang produce an AST dump
        const newOptions = _.filter(options, option => option !== '-fcolor-diagnostics').concat([
            '-Xclang',
            '-ast-dump',
            '-fsyntax-only',
        ]);

        const execOptions = this.getDefaultExecOptions();
        // A higher max output is needed for when the user includes headers
        execOptions.maxOutput = 1024 * 1024 * 1024;

        return this.llvmAst.processAst(
            await this.runCompiler(this.compiler.exe, newOptions, this.filename(inputFilename), execOptions),
        );
    }

    async generatePP(inputFilename, compilerOptions, rawPpOptions) {
        // -E to dump preprocessor output, remove -o so it is dumped to stdout
        compilerOptions = compilerOptions.concat(['-E']);
        if (compilerOptions.includes('-o')) {
            compilerOptions.splice(compilerOptions.indexOf('-o'), 2);
        }

        const ppOptions = _.extend(
            {
                'filter-headers': false,
                'clang-format': false,
            },
            rawPpOptions,
        );

        const execOptions = this.getDefaultExecOptions();
        // A higher max output is needed for when the user includes headers
        execOptions.maxOutput = 1024 * 1024 * 1024;
        const result = await this.runCompilerRawOutput(
            this.compiler.exe,
            compilerOptions,
            this.filename(inputFilename),
            execOptions,
        );
        let output = result.stdout;

        let numberOfLinesFiltered = 0;
        if (ppOptions['filter-headers']) {
            [numberOfLinesFiltered, output] = this.filterPP(output);
        }
        if (ppOptions['clang-format']) {
            output = await this.applyClangFormat(output);
        }

        return {
            numberOfLinesFiltered,
            output: output,
        };
    }

    filterPP(stdout): any[] {
        // Every compiler except Chibicc, as far as I've tested, outputs these line annotations
        // Compiler test: https://godbolt.org/z/K7Pncjs4o
        // Matching things like:
        // # 4 "/app/example.cpp"
        // # 11 "/usr/include/x86_64-linux-gnu/gnu/stubs.h" 2 3 4
        // #line 1816 "C:/WinSdk/Include/10.0.18362.0/ucrt\\corecrt.h"
        // # 13 "" 3
        // regex test cases: https://regex101.com/r/9dOsUI/1
        const lines = stdout.split('\n');
        const ppLineRe = /^\s*#\s*(?:line)?\s*\d+\s*"((?:\\"|[^"])*)"/i;
        let isInSourceRegion = true;
        let numberOfLinesFiltered = 0;
        const filteredLines: string[] = [];
        for (const line of lines) {
            const match = line.match(ppLineRe);
            if (match === null) {
                if (isInSourceRegion) {
                    filteredLines.push(line);
                } else {
                    numberOfLinesFiltered++;
                }
            } else {
                const path = match[1];
                if (
                    path.trim() === '' ||
                    path === '<source>' ||
                    path === '<stdin>' ||
                    path.endsWith('.c') ||
                    path.endsWith('.cpp')
                ) {
                    isInSourceRegion = true;
                } else {
                    isInSourceRegion = false;
                }
                numberOfLinesFiltered++;
            }
        }
        return [numberOfLinesFiltered, filteredLines.join('\n')];
    }

    async applyClangFormat(output): Promise<string> {
        // Currently hard-coding llvm style
        try {
            const [stdout, stderr] = await this.env.formatHandler.internalFormat('clangformat', 'LLVM', output);
            if (stderr) {
                return stdout + '\n/* clang-format stderr:\n' + stderr.trim() + '\n*/';
            }
            return stdout;
        } catch (err) {
            logger.error('Internal formatter error', {err});
            return '/* <Error while running clang-format> */\n\n' + output;
        }
    }

    async generateIR(
        inputFilename: string,
        options: string[],
        irOptions: LLVMIrBackendOptions,
        produceCfg: boolean,
        filters: ParseFiltersAndOutputOptions,
    ) {
        // These options make Clang produce an IR
        const newOptions = options
            .filter(option => option !== '-fcolor-diagnostics')
            .concat(unwrap(this.compiler.irArg));

        if (irOptions.noDiscardValueNames && this.compiler.llvmOptNoDiscardValueNamesArg) {
            newOptions.push(...this.compiler.llvmOptNoDiscardValueNamesArg);
        }

        const execOptions = this.getDefaultExecOptions();
        // A higher max output is needed for when the user includes headers
        execOptions.maxOutput = 1024 * 1024 * 1024;

        const output = await this.runCompiler(this.compiler.exe, newOptions, this.filename(inputFilename), execOptions);
        if (output.code !== 0) {
            return {
                asm: [{text: 'Failed to run compiler to get IR code'}],
            };
        }
        const ir = await this.processIrOutput(output, irOptions, filters);

        const result: {
            asm: ParsedAsmResultLine[];
            cfg?: Record<string, cfg.CFG>;
        } = {
            asm: ir.asm,
        };

        if (produceCfg) {
            result.cfg = cfg.generateStructure(
                this.compiler,
                ir.asm.map(line => ({text: line.text})),
                true,
            );
        }

        return result;
    }

    async processIrOutput(
        output,
        irOptions: LLVMIrBackendOptions,
        filters: ParseFiltersAndOutputOptions,
    ): Promise<{
        asm: ParsedAsmResultLine[];
        languageId: string;
    }> {
        const irPath = this.getIrOutputFilename(output.inputFilename, filters);
        if (await fs.pathExists(irPath)) {
            const output = await fs.readFile(irPath, 'utf8');
            return await this.llvmIr.process(output, irOptions);
        }
        return {
            asm: [{text: 'Internal error; unable to open output path'}],
            languageId: 'llvm-ir',
        };
    }

    async generateLLVMOptPipeline(
        inputFilename: string,
        options: string[],
        filters: ParseFiltersAndOutputOptions,
        llvmOptPipelineOptions: LLVMOptPipelineBackendOptions,
    ): Promise<LLVMOptPipelineOutput | undefined> {
        // These options make Clang produce the pass dumps
        const newOptions = options
            .filter(option => option !== '-fcolor-diagnostics')
            .concat(unwrap(this.compiler.llvmOptArg))
            .concat(llvmOptPipelineOptions.fullModule ? unwrap(this.compiler.llvmOptModuleScopeArg) : [])
            .concat(
                llvmOptPipelineOptions.noDiscardValueNames ? unwrap(this.compiler.llvmOptNoDiscardValueNamesArg) : [],
            )
            .concat(this.compiler.debugPatched ? ['-mllvm', '--debug-to-stdout'] : []);

        const execOptions = this.getDefaultExecOptions();
        execOptions.maxOutput = 1024 * 1024 * 1024;

        const compileStart = performance.now();
        const output = await this.runCompiler(this.compiler.exe, newOptions, this.filename(inputFilename), execOptions);
        const compileEnd = performance.now();

        if (output.timedOut) {
            return {
                error: 'Clang invocation timed out',
                results: {},
                clangTime: output.execTime || compileEnd - compileStart,
            };
        }

        if (output.code !== 0) {
            return;
        }

        try {
            const parseStart = performance.now();
            const llvmOptPipeline = await this.processLLVMOptPipeline(
                output,
                filters,
                llvmOptPipelineOptions,
                this.compiler.debugPatched,
            );
            const parseEnd = performance.now();

            if (llvmOptPipelineOptions.demangle) {
                // apply demangles after parsing, would otherwise greatly complicate the parsing of the passes
                // new this.demanglerClass(this.compiler.demangler, this);
                const demangler = new LLVMIRDemangler(this.compiler.demangler, this);
                // collect labels off the raw input
                if (this.compiler.debugPatched) {
                    demangler.collect({asm: output.stdout});
                } else {
                    demangler.collect({asm: output.stderr});
                }
                return {
                    results: await demangler.demangleLLVMPasses(llvmOptPipeline),
                    clangTime: compileEnd - compileStart,
                    parseTime: parseEnd - parseStart,
                };
            } else {
                return {
                    results: llvmOptPipeline,
                    clangTime: compileEnd - compileStart,
                    parseTime: parseEnd - parseStart,
                };
            }
        } catch (e: any) {
            return {
                error: e.toString(),
                results: {},
                clangTime: compileEnd - compileStart,
            };
        }
    }

    async processLLVMOptPipeline(
        output,
        filters: ParseFiltersAndOutputOptions,
        llvmOptPipelineOptions: LLVMOptPipelineBackendOptions,
        debugPatched?: boolean,
    ) {
        return this.llvmPassDumpParser.process(
            debugPatched ? output.stdout : output.stderr,
            filters,
            llvmOptPipelineOptions,
        );
    }

    getRustMacroExpansionOutputFilename(inputFilename) {
        return inputFilename.replace(path.extname(inputFilename), '.expanded.rs');
    }

    getRustHirOutputFilename(inputFilename) {
        return inputFilename.replace(path.extname(inputFilename), '.hir');
    }

    getRustMirOutputFilename(outputFilename) {
        return outputFilename.replace(path.extname(outputFilename), '.mir');
    }

    getHaskellCoreOutputFilename(inputFilename) {
        return inputFilename.replace(path.extname(inputFilename), '.dump-simpl');
    }

    getHaskellStgOutputFilename(inputFilename) {
        return inputFilename.replace(path.extname(inputFilename), '.dump-stg-final');
    }

    getHaskellCmmOutputFilename(inputFilename) {
        return inputFilename.replace(path.extname(inputFilename), '.dump-cmm');
    }

    // Currently called for getting macro expansion and HIR.
    // It returns the content of the output file created after using -Z unpretty=<unprettyOpt>.
    // The outputFriendlyName is a free form string used in case of error.
    async generateRustUnprettyOutput(inputFilename, options, unprettyOpt, outputFilename, outputFriendlyName) {
        const execOptions = this.getDefaultExecOptions();

        const rustcOptions = [...options];
        rustcOptions.splice(options.indexOf('-o', 2));
        rustcOptions.push(inputFilename, '-o', outputFilename, `-Zunpretty=${unprettyOpt}`);

        const output = await this.runCompiler(this.compiler.exe, rustcOptions, inputFilename, execOptions);
        if (output.code !== 0) {
            return [{text: `Failed to run compiler to get Rust ${outputFriendlyName}`}];
        }
        if (await utils.fileExists(outputFilename)) {
            const content = await fs.readFile(outputFilename, 'utf8');
            return content.split('\n').map(line => ({
                text: line,
            }));
        }
        return [{text: 'Internal error; unable to open output path'}];
    }

    async generateRustMacroExpansion(inputFilename, options) {
        const macroExpPath = this.getRustMacroExpansionOutputFilename(inputFilename);
        return this.generateRustUnprettyOutput(inputFilename, options, 'expanded', macroExpPath, 'Macro Expansion');
    }

    async generateRustHir(inputFilename, options) {
        const hirPath = this.getRustHirOutputFilename(inputFilename);
        return this.generateRustUnprettyOutput(inputFilename, options, 'hir-tree', hirPath, 'HIR');
    }

    async processRustMirOutput(outputFilename, output) {
        const mirPath = this.getRustMirOutputFilename(outputFilename);
        if (output.code !== 0) {
            return [{text: 'Failed to run compiler to get Rust MIR'}];
        }
        if (await utils.fileExists(mirPath)) {
            const content = await fs.readFile(mirPath, 'utf8');
            return content.split('\n').map(line => ({
                text: line,
            }));
        }
        return [{text: 'Internal error; unable to open output path'}];
    }

    async processHaskellExtraOutput(outpath, output) {
        if (output.code !== 0) {
            return [{text: 'Failed to run compiler to get Haskell Core'}];
        }
        if (await utils.fileExists(outpath)) {
            const content = await fs.readFile(outpath, 'utf8');
            // output file starts with
            //
            // ==================== <HEADER> ====================
            //
            // we want to drop this to make the output nicer
            return content
                .split('\n')
                .slice(3)
                .map(line => ({
                    text: line,
                }));
        }
        return [{text: 'Internal error; unable to open output path'}];
    }

    getIrOutputFilename(inputFilename: string, filters: ParseFiltersAndOutputOptions): string {
        // filters are passed because rust needs to know whether a binary is being produced or not
        return inputFilename.replace(path.extname(inputFilename), '.ll');
    }

    getOutputFilename(dirPath: string, outputFilebase: string, key?: any): string {
        let filename;
        if (key && key.backendOptions && key.backendOptions.customOutputFilename) {
            filename = key.backendOptions.customOutputFilename;
        } else {
            filename = `${outputFilebase}.s`;
        }

        if (dirPath) {
            return path.join(dirPath, filename);
        } else {
            return filename;
        }
    }

    getExecutableFilename(dirPath: string, outputFilebase: string, key?) {
        return this.getOutputFilename(dirPath, outputFilebase, key);
    }

    async processGnatDebugOutput(inputFilename, result) {
        const contentDebugExpanded: ResultLine[] = [];
        const contentDebugTree: ResultLine[] = [];
        const keep_stdout: ResultLine[] = [];

        // stdout layout:
        //
        // ----- start
        // everything here stays
        // ... in stdout
        // ... until :
        // Source recreated from tree... <-\
        // everything here is              |
        // ... sent in expanded            | this is optionnal
        // ... pane... until :           <-/
        // Tree created for ...          <-\
        // everything after is             | this is optionnal
        // ... sent in Tree pane         <-/
        // ----- EOF
        const startOfExpandedCode = /^Source recreated from tree/;
        const startOfTree = /^Tree created for/;

        let isInExpandedCode = false;
        let isInTree = false;

        for (const obj of Object.values(result.stdout) as ResultLine[]) {
            if (!isInExpandedCode && startOfExpandedCode.test(obj.text)) {
                isInExpandedCode = true;
                isInTree = false;
            } else if (!isInTree && startOfTree.test(obj.text)) {
                isInExpandedCode = false;
                isInTree = true;
            }

            if (isInExpandedCode) {
                contentDebugExpanded.push(obj);
            } else if (isInTree) {
                contentDebugTree.push(obj);
            } else {
                keep_stdout.push(obj);
            }
        }

        // Do not check compiler result before looking for expanded code. The
        // compiler may exit with an error after the emission. This dump is also
        // very usefull to debug error message.

        if (contentDebugExpanded.length === 0)
            if (result.code === 0) {
                contentDebugExpanded.push({
                    text: 'GNAT exited successfully but the expanded code is missing, something is wrong',
                });
            } else {
                contentDebugExpanded.push({text: 'GNAT exited with an error and did not create the expanded code'});
            }

        if (contentDebugTree.length === 0)
            if (result.code === 0) {
                contentDebugTree.push({text: 'GNAT exited successfully but the Tree is missing, something is wrong'});
            } else {
                contentDebugTree.push({text: 'GNAT exited with an error and did not create the Tree'});
            }

        return {
            stdout: keep_stdout,
            tree: contentDebugTree,
            expandedcode: contentDebugExpanded,
        };
    }

    /**
     * @returns {{filename_suffix: string, name: string, command_prefix: string}}
     * `filename_suffix`: dump file name suffix if GCC default dump name is used
     *
     * `name`: the name to be displayed in the UI
     *
     * `command_prefix`: command prefix to be used in case this dump is to be
     *  created using a targeted option (eg. -fdump-rtl-expand)
     */
    fromInternalGccDumpName(internalDumpName, selectedPasses) {
        if (!selectedPasses) selectedPasses = ['ipa', 'tree', 'rtl'];

        const internalNameRe = new RegExp('^\\s*(' + selectedPasses.join('|') + ')-([\\w_-]+).*ON$');
        const match = internalDumpName.match(internalNameRe);
        if (match)
            return {
                filename_suffix: `${match[1][0]}.${match[2]}`,
                name: match[2] + ' (' + match[1] + ')',
                command_prefix: `-fdump-${match[1]}-${match[2]}`,
            };
        else return null;
    }

    async checkOutputFileAndDoPostProcess(asmResult, outputFilename, filters: ParseFiltersAndOutputOptions) {
        try {
            const stat = await fs.stat(outputFilename);
            asmResult.asmSize = stat.size;
        } catch (e) {
            // Ignore errors
        }
        return await this.postProcess(asmResult, outputFilename, filters);
    }

    runToolsOfType(tools, type: ToolTypeKey, compilationInfo): Promise<ToolResult>[] {
        const tooling: Promise<ToolResult>[] = [];
        if (tools) {
            for (const tool of tools) {
                const matches = this.possibleTools.find(possibleTool => {
                    return possibleTool.id === tool.id && possibleTool.type === type;
                });

                if (matches) {
                    const toolPromise: Promise<ToolResult> = matches.runTool(
                        compilationInfo,
                        compilationInfo.inputFilename,
                        tool.args,
                        tool.stdin,
                        this.supportedLibraries,
                    );
                    tooling.push(toolPromise);
                }
            }
        }

        return tooling;
    }

    buildExecutable(compiler, options, inputFilename, execOptions) {
        // default implementation, but should be overridden by compilers
        return this.runCompiler(compiler, options, inputFilename, execOptions);
    }

    async getRequiredLibraryVersions(libraries): Promise<Record<string, LibraryVersion>> {
        const libraryDetails = {};
        _.each(libraries, selectedLib => {
            const foundVersion = this.findLibVersion(selectedLib);
            if (foundVersion) libraryDetails[selectedLib.id] = foundVersion;
        });
        return libraryDetails;
    }

    async setupBuildEnvironment(key: any, dirPath: string, binary: boolean): Promise<BuildEnvDownloadInfo[]> {
        if (this.buildenvsetup) {
            const libraryDetails = await this.getRequiredLibraryVersions(key.libraries);
            return this.buildenvsetup.setup(key, dirPath, libraryDetails, binary);
        } else {
            return [];
        }
    }

    async addArtifactToResult(
        result: CompilationResult,
        filepath: string,
        customType?: string,
        customTitle?: string,
        checkFunc?: BufferOkFunc,
    ) {
        const file_buffer = await fs.readFile(filepath);

        if (checkFunc && !checkFunc(file_buffer)) return;

        const artifact: Artifact = {
            content: file_buffer.toString('base64'),
            type: customType || 'application/octet-stream',
            name: path.basename(filepath),
            title: customTitle || path.basename(filepath),
        };

        if (!result.artifacts) result.artifacts = [];

        result.artifacts.push(artifact);
    }

    protected async writeMultipleFiles(files: any[], dirPath: string) {
        const filesToWrite: Promise<void>[] = [];

        for (const file of files) {
            if (!file.filename) throw new Error('One of more files do not have a filename');

            const fullpath = this.getExtraFilepath(dirPath, file.filename);
            filesToWrite.push(fs.outputFile(fullpath, file.contents));
        }

        return Promise.all(filesToWrite);
    }

    protected async writeAllFiles(
        dirPath: string,
        source: string,
        files: any[],
        filters: ParseFiltersAndOutputOptions,
    ) {
        if (!source) throw new Error(`File ${this.compileFilename} has no content or file is missing`);

        const inputFilename = path.join(dirPath, this.compileFilename);
        await fs.writeFile(inputFilename, source);

        if (files && files.length > 0) {
            await this.writeMultipleFiles(files, dirPath);
        }

        return {
            inputFilename,
        };
    }

    protected async writeAllFilesCMake(dirPath, source, files, filters: ParseFiltersAndOutputOptions) {
        if (!source) throw new Error('File CMakeLists.txt has no content or file is missing');

        const inputFilename = path.join(dirPath, 'CMakeLists.txt');
        await fs.writeFile(inputFilename, source);

        if (files && files.length > 0) {
            await this.writeMultipleFiles(files, dirPath);
        }

        return {
            inputFilename,
        };
    }

    async buildExecutableInFolder(key, dirPath: string): Promise<BuildResult> {
        const writeSummary = await this.writeAllFiles(dirPath, key.source, key.files, key.filters);
        const downloads = await this.setupBuildEnvironment(key, dirPath, true);

        const inputFilename = writeSummary.inputFilename;

        const outputFilename = this.getExecutableFilename(dirPath, this.outputFilebase, key);

        const buildFilters: ParseFiltersAndOutputOptions = Object.assign({}, key.filters);
        buildFilters.binaryObject = false;
        buildFilters.binary = true;
        buildFilters.execute = true;

        const overrides = this.sanitizeCompilerOverrides(key.backendOptions.overrides || []);

        const compilerArguments = _.compact(
            this.prepareArguments(
                key.options,
                buildFilters,
                key.backendOptions,
                inputFilename,
                outputFilename,
                key.libraries,
                overrides,
            ),
        );

        const execOptions = this.getDefaultExecOptions();
        execOptions.ldPath = this.getSharedLibraryPathsAsLdLibraryPaths(key.libraries);

        this.applyOverridesToExecOptions(execOptions, overrides);

        const result = await this.buildExecutable(key.compiler.exe, compilerArguments, inputFilename, execOptions);

        return {
            ...result,
            downloads,
            executableFilename: outputFilename,
            compilationOptions: compilerArguments,
        };
    }

    async getOrBuildExecutable(key, bypassCache: BypassCache) {
        const dirPath = await this.newTempDir();

        if (!bypassCompilationCache(bypassCache)) {
            const buildResults = await this.loadPackageWithExecutable(key, dirPath);
            if (buildResults) return buildResults;
        }

        let compilationResult;
        try {
            compilationResult = await this.buildExecutableInFolder(key, dirPath);
            if (compilationResult.code !== 0) {
                return compilationResult;
            }
        } catch (e) {
            return this.handleUserError(e, dirPath);
        }

        await this.storePackageWithExecutable(key, dirPath, compilationResult);

        if (!compilationResult.dirPath) {
            compilationResult.dirPath = dirPath;
        }

        return compilationResult;
    }

    async loadPackageWithExecutable(key, dirPath) {
        const compilationResultFilename = 'compilation-result.json';
        try {
            const startTime = process.hrtime.bigint();
            const outputFilename = await this.env.executableGet(key, dirPath);
            if (outputFilename) {
                logger.debug(`Using cached package ${outputFilename}`);
                await this.packager.unpack(outputFilename, dirPath);
                const buildResultsBuf = await fs.readFile(path.join(dirPath, compilationResultFilename));
                const buildResults = JSON.parse(buildResultsBuf.toString('utf8'));
                const endTime = process.hrtime.bigint();

                let inputFilename = path.join(dirPath, this.compileFilename);
                if (buildResults.inputFilename) {
                    inputFilename = path.join(dirPath, path.basename(buildResults.inputFilename));
                }

                return Object.assign({}, buildResults, {
                    code: 0,
                    inputFilename: inputFilename,
                    dirPath: dirPath,
                    executableFilename: this.getExecutableFilename(dirPath, this.outputFilebase, key),
                    packageDownloadAndUnzipTime: ((endTime - startTime) / BigInt(1000000)).toString(),
                });
            }
            logger.debug('Tried to get executable from cache, but got a cache miss');
        } catch (err) {
            logger.error('Tried to get executable from cache, but got an error:', {err});
        }
        return false;
    }

    async storePackageWithExecutable(key, dirPath, compilationResult) {
        const compilationResultFilename = 'compilation-result.json';

        const packDir = await this.newTempDir();
        const packagedFile = path.join(packDir, 'package.tgz');
        try {
            await fs.writeFile(path.join(dirPath, compilationResultFilename), JSON.stringify(compilationResult));
            await this.packager.package(dirPath, packagedFile);
            await this.env.executablePut(key, packagedFile);
        } catch (err) {
            logger.error('Caught an error trying to put to cache: ', {err});
        } finally {
            fs.remove(packDir);
        }
    }

    runExecutable(executable: string, executeParameters: ExecutableExecutionOptions, homeDir) {
        const maxExecOutputSize = this.env.ceProps('max-executable-output-size', 32 * 1024);
        // Hardcoded fix for #2339. Ideally I'd have a config option for this, but for now this is plenty good enough.
        executeParameters.env = {
            ASAN_OPTIONS: 'color=always',
            UBSAN_OPTIONS: 'color=always',
            MSAN_OPTIONS: 'color=always',
            LSAN_OPTIONS: 'color=always',
            ...executeParameters.env,
        };
        if (this.compiler.executionWrapper) {
            executeParameters.args = [...this.compiler.executionWrapperArgs, executable, ...executeParameters.args];
            executable = this.compiler.executionWrapper;
        }
        return this.execBinary(executable, maxExecOutputSize, executeParameters, homeDir);
    }

    protected fixExecuteParametersForInterpreting(executeParameters, outputFilename, key) {
        executeParameters.args.unshift(outputFilename);
    }

    async handleInterpreting(key, executeParameters): Promise<CompilationResult> {
        const source = key.source;
        const dirPath = await this.newTempDir();
        const outputFilename = this.getExecutableFilename(dirPath, this.outputFilebase);

        // cant use this.writeAllFiles here because outputFilename is used as the file to execute
        //  instead of inputFilename

        await fs.writeFile(outputFilename, source);
        if (key.files && key.files.length > 0) {
            await this.writeMultipleFiles(key.files, dirPath);
        }

        this.fixExecuteParametersForInterpreting(executeParameters, outputFilename, key);

        const result = await this.runExecutable(this.compiler.exe, executeParameters, dirPath);
        return {
            ...result,
            didExecute: true,
            buildResult: {
                code: 0,
                timedOut: false,
                stdout: [],
                stderr: [],
                downloads: [],
                executableFilename: outputFilename,
                compilationOptions: [],
            },
        };
    }

    async doExecution(key, executeParameters, bypassCache: BypassCache): Promise<CompilationResult> {
        if (this.compiler.interpreted) {
            return this.handleInterpreting(key, executeParameters);
        }
        const buildResult = await this.getOrBuildExecutable(key, bypassCache);
        if (buildResult.code !== 0) {
            return {
                code: -1,
                didExecute: false,
                buildResult,
                stderr: [{text: 'Build failed'}],
                stdout: [],
                timedOut: false,
            };
        }

        if (!(await utils.fileExists(buildResult.executableFilename))) {
            const verboseResult = {
                code: -1,
                didExecute: false,
                buildResult,
                stderr: [{text: 'Executable not found'}],
                stdout: [],
                timedOut: false,
            };

            verboseResult.buildResult.stderr.push({text: 'Compiler did not produce an executable'});

            return verboseResult;
        }

        if (!this.compiler.supportsExecute) {
            return {
                code: -1,
                didExecute: false,
                buildResult,
                stderr: [{text: 'Compiler does not support execution'}],
                stdout: [],
                timedOut: false,
            };
        }

        executeParameters.ldPath = this.getSharedLibraryPathsAsLdLibraryPathsForExecution(key.libraries);
        const result = await this.runExecutable(buildResult.executableFilename, executeParameters, buildResult.dirPath);
        return {
            ...result,
            didExecute: true,
            buildResult: buildResult,
        };
    }

    async handleExecution(key, executeParameters, bypassCache: BypassCache): Promise<CompilationResult> {
        // stringify now so shallow copying isn't a problem, I think the executeParameters get modified
        const execKey = JSON.stringify({key, executeParameters});
        if (!bypassExecutionCache(bypassCache)) {
            const cacheResult = await this.env.cacheGet(execKey as any);
            if (cacheResult) {
                return cacheResult;
            }
        }

        const result = await this.doExecution(key, executeParameters, bypassCache);
        if (!bypassExecutionCache(bypassCache)) {
            await this.env.cachePut(execKey, result, undefined);
        }
        return result;
    }

    getCacheKey(source, options, backendOptions, filters, tools, libraries, files) {
        return {compiler: this.compiler, source, options, backendOptions, filters, tools, libraries, files};
    }

    getCmakeCacheKey(key, files) {
        const cacheKey = Object.assign({}, key);
        cacheKey.compiler = this.compiler;
        cacheKey.files = files;
        cacheKey.api = 'cmake';

        if (cacheKey.filters) delete cacheKey.filters.execute;
        delete cacheKey.executionParameters;
        delete cacheKey.tools;

        return cacheKey;
    }

    getCompilationInfo(
        key: CompilationCacheKey,
        result: CompilationResult | CustomInputForTool,
        customBuildPath?: string,
    ): CompilationInfo {
        return {
            outputFilename: this.getOutputFilename(customBuildPath || result.dirPath || '', this.outputFilebase, key),
            executableFilename: this.getExecutableFilename(
                customBuildPath || result.dirPath || '',
                this.outputFilebase,
                key,
            ),
            asmParser: this.asm,
            ...key,
            ...result,
        };
    }

    tryAutodetectLibraries(libsAndOptions) {
        const linkFlag = this.compiler.linkFlag || '-l';

        const detectedLibs: SelectedLibraryVersion[] = [];
        const foundlibOptions: string[] = [];
        _.each(libsAndOptions.options, option => {
            if (option.indexOf(linkFlag) === 0) {
                const libVersion = this.findAutodetectStaticLibLink(option.substr(linkFlag.length).trim());
                if (libVersion) {
                    foundlibOptions.push(option);
                    detectedLibs.push(libVersion);
                }
            }
        });

        if (detectedLibs.length > 0) {
            libsAndOptions.options = _.filter(libsAndOptions.options, option => !foundlibOptions.includes(option));
            libsAndOptions.libraries = _.union(libsAndOptions.libraries, detectedLibs);

            return true;
        } else {
            return false;
        }
    }

    sanitizeCompilerOverrides(overrides: ConfiguredOverrides): ConfiguredOverrides {
        const allowedRegex = /^[A-Z_]{1,}[A-Z0-9_]*$/;
        for (const override of overrides) {
            if (override.name === CompilerOverrideType.env && override.values) {
                // lowercase names are allowed, but let's assume everyone means to use uppercase
                override.values.forEach(env => (env.name = env.name.trim().toUpperCase()));
                override.values = override.values.filter(
                    env => env.name !== 'LD_PRELOAD' && env.name.match(allowedRegex),
                );
            }
        }
        return overrides;
    }

    applyOverridesToExecOptions(execOptions: ExecutionOptions, overrides: ConfiguredOverrides): void {
        if (!execOptions.env) execOptions.env = {};

        for (const override of overrides) {
            if (override.name === CompilerOverrideType.env && override.values) {
                for (const env of override.values) {
                    execOptions.env[env.name] = env.value;
                }
            }
        }
    }

    async doCompilation(inputFilename, dirPath, key, options, filters, backendOptions, libraries, tools) {
        const inputFilenameSafe = this.filename(inputFilename);

        const outputFilename = this.getOutputFilename(dirPath, this.outputFilebase, key);

        const overrides = this.sanitizeCompilerOverrides(backendOptions.overrides || []);

        options = _.compact(
            this.prepareArguments(
                options,
                filters,
                backendOptions,
                inputFilename,
                outputFilename,
                libraries,
                overrides,
            ),
        );

        const execOptions = this.getDefaultExecOptions();
        execOptions.ldPath = this.getSharedLibraryPathsAsLdLibraryPaths([]);

        this.applyOverridesToExecOptions(execOptions, overrides);

        const makeAst = backendOptions.produceAst && this.compiler.supportsAstView;
        const makePp = backendOptions.producePp && this.compiler.supportsPpView;
        const makeGnatDebug = backendOptions.produceGnatDebug && this.compiler.supportsGnatDebugViews;
        const makeGnatDebugTree = backendOptions.produceGnatDebugTree && this.compiler.supportsGnatDebugViews;
        const makeIr = backendOptions.produceIr && this.compiler.supportsIrView;
        const makeLLVMOptPipeline = backendOptions.produceLLVMOptPipeline && this.compiler.supportsLLVMOptPipelineView;
        const makeRustMir = backendOptions.produceRustMir && this.compiler.supportsRustMirView;
        const makeRustMacroExp = backendOptions.produceRustMacroExp && this.compiler.supportsRustMacroExpView;
        const makeRustHir = backendOptions.produceRustHir && this.compiler.supportsRustHirView;
        const makeHaskellCore = backendOptions.produceHaskellCore && this.compiler.supportsHaskellCoreView;
        const makeHaskellStg = backendOptions.produceHaskellStg && this.compiler.supportsHaskellStgView;
        const makeHaskellCmm = backendOptions.produceHaskellCmm && this.compiler.supportsHaskellCmmView;
        const makeGccDump =
            backendOptions.produceGccDump && backendOptions.produceGccDump.opened && this.compiler.supportsGccDump;

        const downloads = await this.setupBuildEnvironment(key, dirPath, filters.binary || filters.binaryObject);
        const [
            asmResult,
            astResult,
            ppResult,
            irResult,
            llvmOptPipelineResult,
            rustHirResult,
            rustMacroExpResult,
            toolsResult,
        ] = await Promise.all([
            this.runCompiler(this.compiler.exe, options, inputFilenameSafe, execOptions),
            makeAst ? this.generateAST(inputFilename, options) : null,
            makePp ? this.generatePP(inputFilename, options, backendOptions.producePp) : null,
            makeIr
                ? this.generateIR(
                      inputFilename,
                      options,
                      backendOptions.produceIr,
                      backendOptions.produceCfg && backendOptions.produceCfg.ir,
                      filters,
                  )
                : null,
            makeLLVMOptPipeline
                ? this.generateLLVMOptPipeline(inputFilename, options, filters, backendOptions.produceLLVMOptPipeline)
                : null,
            makeRustHir ? this.generateRustHir(inputFilename, options) : null,
            makeRustMacroExp ? this.generateRustMacroExpansion(inputFilename, options) : null,
            Promise.all(
                this.runToolsOfType(
                    tools,
                    'independent',
                    this.getCompilationInfo(key, {
                        inputFilename,
                        dirPath,
                        outputFilename,
                    }),
                ),
            ),
        ]);

        // GNAT, GCC and rustc can produce their extra output files along
        // with the main compilation command.
        const gnatDebugResults =
            makeGnatDebug || makeGnatDebugTree ? await this.processGnatDebugOutput(inputFilenameSafe, asmResult) : '';

        const gccDumpResult = makeGccDump
            ? await this.processGccDumpOutput(
                  backendOptions.produceGccDump,
                  asmResult,
                  this.compiler.removeEmptyGccDump,
                  outputFilename,
              )
            : '';
        const rustMirResult = makeRustMir ? await this.processRustMirOutput(outputFilename, asmResult) : '';

        const haskellCoreResult = makeHaskellCore
            ? await this.processHaskellExtraOutput(this.getHaskellCoreOutputFilename(inputFilename), asmResult)
            : '';
        const haskellStgResult = makeHaskellStg
            ? await this.processHaskellExtraOutput(this.getHaskellStgOutputFilename(inputFilename), asmResult)
            : '';
        const haskellCmmResult = makeHaskellCmm
            ? await this.processHaskellExtraOutput(this.getHaskellCmmOutputFilename(inputFilename), asmResult)
            : '';

        asmResult.dirPath = dirPath;
        if (!asmResult.compilationOptions) asmResult.compilationOptions = options;
        asmResult.downloads = downloads;
        // Here before the check to ensure dump reports even on failure cases
        if (this.compiler.supportsGccDump && gccDumpResult) {
            asmResult.gccDumpOutput = gccDumpResult;
        }

        if (this.compiler.supportsGnatDebugViews && gnatDebugResults) {
            asmResult.stdout = gnatDebugResults.stdout;

            if (makeGnatDebug && gnatDebugResults.expandedcode.length > 0) {
                asmResult.hasGnatDebugOutput = true;
                asmResult.gnatDebugOutput = gnatDebugResults.expandedcode;
            }
            if (makeGnatDebugTree && gnatDebugResults.tree.length > 0) {
                asmResult.hasGnatDebugTreeOutput = true;
                asmResult.gnatDebugTreeOutput = gnatDebugResults.tree;
            }
        }

        // PP output populated here due to early return in the event of compilation failure
        if (ppResult) {
            asmResult.hasPpOutput = true;
            asmResult.ppOutput = ppResult;
        }

        asmResult.tools = toolsResult;
        if (this.compiler.supportsOptOutput && backendOptions.produceOptInfo) {
            const optPath = path.join(dirPath, `${this.outputFilebase}.opt.yaml`);
            if (await fs.pathExists(optPath)) {
                asmResult.hasOptOutput = true;
                asmResult.optPath = optPath;
            }
        }
        if (this.compiler.supportsStackUsageOutput && backendOptions.produceStackUsageInfo) {
            const suPath = path.join(dirPath, `${this.outputFilebase}.su`);
            if (await fs.pathExists(suPath)) {
                asmResult.hasStackUsageOutput = true;
                asmResult.stackUsagePath = suPath;
            }
        }

        if (astResult) {
            asmResult.hasAstOutput = true;
            asmResult.astOutput = astResult;
        }
        if (ppResult) {
            asmResult.hasPpOutput = true;
            asmResult.ppOutput = ppResult;
        }
        if (irResult) {
            asmResult.hasIrOutput = true;
            asmResult.irOutput = irResult;
        }
        if (llvmOptPipelineResult) {
            asmResult.hasLLVMOptPipelineOutput = true;
            asmResult.llvmOptPipelineOutput = llvmOptPipelineResult;
        }
        if (rustMirResult) {
            asmResult.hasRustMirOutput = true;
            asmResult.rustMirOutput = rustMirResult;
        }
        if (rustMacroExpResult) {
            asmResult.hasRustMacroExpOutput = true;
            asmResult.rustMacroExpOutput = rustMacroExpResult;
        }
        if (rustHirResult) {
            asmResult.hasRustHirOutput = true;
            asmResult.rustHirOutput = rustHirResult;
        }
        if (haskellCoreResult) {
            asmResult.hasHaskellCoreOutput = true;
            asmResult.haskellCoreOutput = haskellCoreResult;
        }
        if (haskellStgResult) {
            asmResult.hasHaskellStgOutput = true;
            asmResult.haskellStgOutput = haskellStgResult;
        }
        if (haskellCmmResult) {
            asmResult.hasHaskellCmmOutput = true;
            asmResult.haskellCmmOutput = haskellCmmResult;
        }
        if (asmResult.code !== 0) {
            return [{...asmResult, asm: '<Compilation failed>'}];
        }

        return this.checkOutputFileAndDoPostProcess(asmResult, outputFilename, filters);
    }

    doTempfolderCleanup(buildResult) {
        if (buildResult.dirPath && !this.delayCleanupTemp) {
            fs.remove(buildResult.dirPath);
        }
        buildResult.dirPath = undefined;
    }

    getCompilerEnvironmentVariables(compilerflags) {
        if (this.lang.id === 'c++') {
            return {...this.cmakeBaseEnv, CXXFLAGS: compilerflags};
        } else if (this.lang.id === 'fortran') {
            return {...this.cmakeBaseEnv, FFLAGS: compilerflags};
        } else if (this.lang.id === 'cuda') {
            return {...this.cmakeBaseEnv, CUDAFLAGS: compilerflags};
        } else {
            return {...this.cmakeBaseEnv, CFLAGS: compilerflags};
        }
    }

    async doBuildstep(command, args, execParams) {
        const result = await this.exec(command, args, execParams);
        return this.processExecutionResult(result);
    }

    handleUserError(error, dirPath) {
        return {
            dirPath,
            okToCache: false,
            code: -1,
            asm: [{text: `<${error.message}>`}],
            stdout: [],
            stderr: [{text: `<${error.message}>`}],
        };
    }

    async doBuildstepAndAddToResult(result, name, command, args, execParams): Promise<BuildStep> {
        const stepResult: BuildStep = {
            ...(await this.doBuildstep(command, args, execParams)),
            compilationOptions: args,
            step: name,
        };
        logger.debug(name);
        result.buildsteps.push(stepResult);
        return stepResult;
    }

    createCmakeExecParams(execParams: ExecutionOptions, dirPath: string, libsAndOptions, toolchainPath: string) {
        const cmakeExecParams = Object.assign({}, execParams);

        const libIncludes = this.getIncludeArguments(libsAndOptions.libraries);

        const options: string[] = [];
        if (this.compiler.options) {
            const compilerOptions = utils.splitArguments(this.compiler.options);
            options.push(...removeToolchainArg(compilerOptions));
        }
        options.push(...libsAndOptions.options);
        options.push(...libIncludes);

        _.extend(cmakeExecParams.env, this.getCompilerEnvironmentVariables(options.join(' ')));

        cmakeExecParams.ldPath = [dirPath];

        // todo: if we don't use nsjail, the path should not be /app but dirPath
        const libPaths = this.getSharedLibraryPathsAsArguments(libsAndOptions.libraries, '/app', toolchainPath);
        cmakeExecParams.env.LDFLAGS = libPaths.join(' ');

        return cmakeExecParams;
    }

    createLibsAndOptions(key) {
        const libsAndOptions = {libraries: key.libraries, options: key.options};
        if (this.tryAutodetectLibraries(libsAndOptions)) {
            key.libraries = libsAndOptions.libraries;
            key.options = libsAndOptions.options;
        }
        return libsAndOptions;
    }

    getExtraCMakeArgs(key): string[] {
        return [];
    }

    getCMakeExtToolchainParam(overrides: ConfiguredOverrides): string {
        const toolchainPath = this.getDefaultOrOverridenToolchainPath(overrides);
        if (toolchainPath) {
            return `-DCMAKE_CXX_COMPILER_EXTERNAL_TOOLCHAIN=${toolchainPath}`;
        }

        return '';
    }

    getUsedEnvironmentVariableFlags(makeExecParams) {
        if (this.lang.id === 'c++') {
            return utils.splitArguments(makeExecParams.env.CXXFLAGS);
        } else if (this.lang.id === 'fortran') {
            return utils.splitArguments(makeExecParams.env.FFLAGS);
        } else if (this.lang.id === 'cuda') {
            return utils.splitArguments(makeExecParams.env.CUDAFLAGS);
        } else {
            return utils.splitArguments(makeExecParams.env.CFLAGS);
        }
    }

    async cmake(files, key, bypassCache: BypassCache) {
        // key = {source, options, backendOptions, filters, bypassCache, tools, executionParameters, libraries};

        if (!this.compiler.supportsBinary) {
            const errorResult: CompilationResult = {
                code: -1,
                timedOut: false,
                didExecute: false,
                stderr: [],
                stdout: [],
            };

            errorResult.stderr.push({text: 'Compiler does not support compiling to binaries'});
            return errorResult;
        }

        _.defaults(key.filters, this.getDefaultFilters());
        key.filters.binary = true;
        key.filters.dontMaskFilenames = true;

        const libsAndOptions = this.createLibsAndOptions(key);

        const toolchainPath = this.getDefaultOrOverridenToolchainPath(key.backendOptions.overrides || []);

        const doExecute = key.filters.execute;
        const executeParameters: ExecutableExecutionOptions = {
            ldPath: this.getSharedLibraryPathsAsLdLibraryPaths(key.libraries),
            args: key.executionParameters.args || [],
            stdin: key.executionParameters.stdin || '',
            env: {},
        };

        const cacheKey = this.getCmakeCacheKey(key, files);

        const dirPath = await this.newTempDir();

        const outputFilename = this.getExecutableFilename(path.join(dirPath, 'build'), this.outputFilebase, key);

        let fullResult = !bypassExecutionCache(bypassCache)
            ? await this.loadPackageWithExecutable(cacheKey, dirPath)
            : null;
        if (fullResult) {
            fullResult.fetchedFromCache = true;

            delete fullResult.inputFilename;
            delete fullResult.executableFilename;
            delete fullResult.dirPath;
        } else {
            let writeSummary;
            try {
                writeSummary = await this.writeAllFilesCMake(dirPath, cacheKey.source, files, cacheKey.filters);
            } catch (e) {
                return this.handleUserError(e, dirPath);
            }

            const execParams = this.getDefaultExecOptions();
            execParams.appHome = dirPath;
            execParams.customCwd = path.join(dirPath, 'build');

            await fs.mkdir(execParams.customCwd);

            const makeExecParams = this.createCmakeExecParams(execParams, dirPath, libsAndOptions, toolchainPath);

            fullResult = {
                buildsteps: [],
                inputFilename: writeSummary.inputFilename,
            };

            fullResult.downloads = await this.setupBuildEnvironment(cacheKey, dirPath, true);

            const toolchainparam = this.getCMakeExtToolchainParam(key.backendOptions.overrides || []);

            const cmakeArgs = utils.splitArguments(key.backendOptions.cmakeArgs);
            const partArgs: string[] = [toolchainparam, ...this.getExtraCMakeArgs(key), ...cmakeArgs, '..'];
            let fullArgs: string[] = [];
            const useNinja = this.env.ceProps('useninja');
            if (useNinja) {
                fullArgs = ['-GNinja'].concat(partArgs);
            } else {
                fullArgs = partArgs;
            }

            const cmakeStepResult = await this.doBuildstepAndAddToResult(
                fullResult,
                'cmake',
                this.env.ceProps('cmake'),
                fullArgs,
                makeExecParams,
            );

            if (cmakeStepResult.code !== 0) {
                fullResult.result = {
                    dirPath,
                    okToCache: false,
                    code: cmakeStepResult.code,
                    asm: [{text: '<Build failed>'}],
                };
                fullResult.result.compilationOptions = this.getUsedEnvironmentVariableFlags(makeExecParams);
                return fullResult;
            }

            const makeStepResult = await this.doBuildstepAndAddToResult(
                fullResult,
                'build',
                this.env.ceProps('cmake'),
                ['--build', '.'],
                execParams,
            );

            if (makeStepResult.code !== 0) {
                fullResult.result = {
                    dirPath,
                    okToCache: false,
                    code: makeStepResult.code,
                    asm: [{text: '<Build failed>'}],
                };
                return fullResult;
            }

            fullResult.result = {
                dirPath,
                okToCache: true,
            };

            if (!key.backendOptions.skipAsm) {
                const [asmResult] = await this.checkOutputFileAndDoPostProcess(
                    fullResult.result,
                    outputFilename,
                    cacheKey.filters,
                );
                fullResult.result = asmResult;
            }

            fullResult.result.compilationOptions = this.getUsedEnvironmentVariableFlags(makeExecParams);

            fullResult.code = 0;
            _.each(fullResult.buildsteps, function (step) {
                fullResult.code += step.code;
            });

            await this.storePackageWithExecutable(cacheKey, dirPath, fullResult);
        }

        fullResult.result.dirPath = dirPath;

        if (this.compiler.supportsExecute && doExecute) {
            fullResult.execResult = await this.runExecutable(outputFilename, executeParameters, dirPath);
            fullResult.didExecute = true;
        }

        const optOutput = undefined;
        const stackUsageOutput = undefined;
        await this.afterCompilation(
            fullResult.result,
            false,
            cacheKey,
            [],
            key.tools,
            cacheKey.backendOptions,
            cacheKey.filters,
            libsAndOptions.options,
            optOutput,
            stackUsageOutput,
            bypassCache,
            path.join(dirPath, 'build'),
        );

        delete fullResult.result.dirPath;

        return fullResult;
    }

    protected getExtraFilepath(dirPath, filename) {
        // note: it's vitally important that the resulting path does not escape dirPath
        //       (filename is user input and thus unsafe)

        const joined = path.join(dirPath, filename);
        const normalized = path.normalize(joined);
        if (process.platform === 'win32') {
            if (!normalized.replace(/\\/g, '/').startsWith(dirPath.replace(/\\/g, '/'))) {
                throw new Error('Invalid filename');
            }
        } else {
            if (!normalized.startsWith(dirPath)) throw new Error('Invalid filename');
        }
        return normalized;
    }

    fixFiltersBeforeCacheKey(filters, options, files) {
        // Don't run binary for unsupported compilers, even if we're asked.
        if (filters.binary && !this.compiler.supportsBinary) {
            delete filters.binary;
        }

        // For C/C++ turn off filters when compiling with -E
        // This is done for all compilers. Not every compiler handles -E the same but they all use
        // it for preprocessor output.
        if ((this.compiler.lang === 'c++' || this.compiler.lang === 'c') && options.includes('-E')) {
            for (const key in filters) {
                filters[key] = false;
            }

            if (filters.binaryObject && !this.compiler.supportsBinaryObject) {
                delete filters.binaryObject;
            }
        }

        if (files && files.length > 0) {
            filters.dontMaskFilenames = true;
        }
    }

    async compile(
        source,
        options,
        backendOptions,
        filters,
        bypassCache: BypassCache,
        tools,
        executionParameters,
        libraries,
        files,
    ) {
        const optionsError = this.checkOptions(options);
        if (optionsError) throw optionsError;
        const sourceError = this.checkSource(source);
        if (sourceError) throw sourceError;

        const libsAndOptions = {libraries, options};
        if (this.tryAutodetectLibraries(libsAndOptions)) {
            libraries = libsAndOptions.libraries;
            options = libsAndOptions.options;
        }

        this.fixFiltersBeforeCacheKey(filters, options, files);

        const executeParameters = {
            args: executionParameters.args || [],
            stdin: executionParameters.stdin || '',
        };

        const key = this.getCacheKey(source, options, backendOptions, filters, tools, libraries, files);

        const doExecute = filters.execute;
        filters = Object.assign({}, filters);
        filters.execute = false;

        if (!bypassCompilationCache(bypassCache)) {
            const cacheRetreiveTimeStart = process.hrtime.bigint();
            // TODO: We should be able to eliminate this any cast. `key` should be cacheable (if it's not that's a big
            // problem) Because key coantains a CompilerInfo which contains a function member it can't be assigned to a
            // CacheableValue.
            const result = await this.env.cacheGet(key as any);
            if (result) {
                const cacheRetreiveTimeEnd = process.hrtime.bigint();
                result.retreivedFromCacheTime = (
                    (cacheRetreiveTimeEnd - cacheRetreiveTimeStart) /
                    BigInt(1000000)
                ).toString();
                result.retreivedFromCache = true;
                if (doExecute) {
                    const queueTime = performance.now();
                    result.execResult = await this.env.enqueue(async () => {
                        const start = performance.now();
                        executionQueueTimeHistogram.observe((start - queueTime) / 1000);
                        const res = await this.handleExecution(key, executeParameters, bypassCache);
                        executionTimeHistogram.observe((performance.now() - start) / 1000);
                        return res;
                    });

                    if (result.execResult && result.execResult.buildResult) {
                        this.doTempfolderCleanup(result.execResult.buildResult);
                    }
                }
                return result;
            }
        }
        const queueTime = performance.now();
        return this.env.enqueue(async () => {
            const start = performance.now();
            compilationQueueTimeHistogram.observe((start - queueTime) / 1000);
            const res = await (async () => {
                source = this.preProcess(source, filters);

                if (backendOptions.executorRequest) {
                    const execResult = await this.handleExecution(key, executeParameters, bypassCache);
                    if (execResult && execResult.buildResult) {
                        this.doTempfolderCleanup(execResult.buildResult);
                    }
                    return execResult;
                }

                const dirPath = await this.newTempDir();

                let writeSummary;
                try {
                    writeSummary = await this.writeAllFiles(dirPath, source, files, filters);
                } catch (e) {
                    return this.handleUserError(e, dirPath);
                }
                const inputFilename = writeSummary.inputFilename;

                const [result, optOutput, stackUsageOutput] = await this.doCompilation(
                    inputFilename,
                    dirPath,
                    key,
                    options,
                    filters,
                    backendOptions,
                    libraries,
                    tools,
                );

                return await this.afterCompilation(
                    result,
                    doExecute,
                    key,
                    executeParameters,
                    tools,
                    backendOptions,
                    filters,
                    options,
                    optOutput,
                    stackUsageOutput,
                    bypassCache,
                );
            })();
            compilationTimeHistogram.observe((performance.now() - start) / 1000);
            return res;
        });
    }

    async afterCompilation(
        result,
        doExecute,
        key,
        executeParameters,
        tools,
        backendOptions,
        filters,
        options,
        optOutput,
        stackUsageOutput,
        bypassCache: BypassCache,
        customBuildPath?,
    ) {
        // Start the execution as soon as we can, but only await it at the end.
        const execPromise =
            doExecute && result.code === 0 ? this.handleExecution(key, executeParameters, bypassCache) : null;

        if (result.hasOptOutput) {
            delete result.optPath;
            result.optOutput = optOutput;
        }

        if (result.hasStackUsageOutput) {
            delete result.stackUsagePath;
            result.stackUsageOutput = stackUsageOutput;
        }

        const compilationInfo = this.getCompilationInfo(key, result, customBuildPath);

        result.tools = _.union(
            result.tools,
            await Promise.all(this.runToolsOfType(tools, 'postcompilation', compilationInfo)),
        );

        result = await this.extractDeviceCode(result, filters, compilationInfo);

        this.doTempfolderCleanup(result);
        if (result.buildResult) {
            this.doTempfolderCleanup(result.buildResult);
        }

        if (backendOptions.skipAsm) {
            result.asm = [];
        } else {
            if (!result.externalParserUsed) {
                if (result.okToCache) {
                    const res = await this.processAsm(result, filters, options);
                    result.asm = res.asm;
                    result.labelDefinitions = res.labelDefinitions;
                    result.parsingTime = res.parsingTime;
                    result.filteredCount = res.filteredCount;
                    if (result.objdumpTime) {
                        const dumpAndParseTime = parseInt(result.objdumpTime) + parseInt(result.parsingTime);
                        BaseCompiler.objdumpAndParseCounter.inc(dumpAndParseTime);
                    }
                } else {
                    result.asm = [{text: result.asm}];
                }
            }
            // TODO rephrase this so we don't need to reassign
            result = filters.demangle ? await this.postProcessAsm(result, filters) : result;
            if (this.compiler.supportsCfg && backendOptions.produceCfg && backendOptions.produceCfg.asm) {
                const isLlvmIr = (options && options.includes('-emit-llvm')) || this.llvmIr.isLlvmIr(result.asm);
                result.cfg = cfg.generateStructure(this.compiler, result.asm, isLlvmIr);
            }
        }

        if (!backendOptions.skipPopArgs) result.popularArguments = this.possibleArguments.getPopularArguments(options);

        result = this.postCompilationPreCacheHook(result);

        if (result.okToCache) {
            await this.env.cachePut(key, result, undefined);
        }

        if (doExecute && result.code === 0) {
            result.execResult = await execPromise;

            if (result.execResult.buildResult) {
                this.doTempfolderCleanup(result.execResult.buildResult);
            }
        }
        return result;
    }

    postCompilationPreCacheHook(result: CompilationResult): CompilationResult {
        return result;
    }

    async processAsm(result, filters, options) {
        if ((options && options.includes('-emit-llvm')) || this.llvmIr.isLlvmIr(result.asm)) {
            return await this.llvmIr.processFromFilters(result.asm, filters);
        }

        return this.asm.process(result.asm, filters);
    }

    async postProcessAsm(result, filters?: ParseFiltersAndOutputOptions) {
        if (!result.okToCache || !this.demanglerClass || !result.asm) return result;
        const demangler = new this.demanglerClass(this.compiler.demangler, this, this.compiler.demanglerArgs);

        return await demangler.process(result);
    }

    async processOptOutput(optPath: string) {
        const output: compilerOptInfo.LLVMOptInfo[] = [];

        const optStream = fs
            .createReadStream(optPath, {encoding: 'utf8'})
            .pipe(new compilerOptInfo.LLVMOptTransformer());

        for await (const opt of optStream as AsyncIterable<compilerOptInfo.LLVMOptInfo>) {
            if (opt.DebugLoc && opt.DebugLoc.File && opt.DebugLoc.File.includes(this.compileFilename)) {
                output.push(opt);
            }
        }

        if (this.compiler.demangler) {
            const result = JSON.stringify(output, null, 4);
            const demangleResult: UnprocessedExecResult = await this.exec(
                this.compiler.demangler,
                [...this.compiler.demanglerArgs, '-n', '-p'],
                {input: result},
            );
            if (!demangleResult.truncated) {
                try {
                    return JSON.parse(demangleResult.stdout);
                } catch (exception) {
                    // swallow exception and return non-demangled output
                    logger.warn(`Caught exception ${exception} during opt demangle parsing`);
                }
            }
        }

        return output;
    }
    async processStackUsageOutput(suPath) {
        const output = StackUsageTransformer.parse(await fs.readFile(suPath, 'utf-8'));

        if (this.compiler.demangler) {
            const result = JSON.stringify(output, null, 4);
            try {
                const demangleResult = await this.exec(
                    this.compiler.demangler,
                    [...this.compiler.demanglerArgs, '-n', '-p'],
                    {input: result},
                );
                return JSON.parse(demangleResult.stdout);
            } catch (exception) {
                // swallow exception and return non-demangled output
                logger.warn(`Caught exception ${exception} during stack usage demangle parsing`);
            }
        }

        return output;
    }

    couldSupportASTDump(version: string) {
        const versionRegex = /version (\d+.\d+)/;
        const versionMatch = versionRegex.exec(version);

        if (versionMatch) {
            const versionNum = parseFloat(versionMatch[1]);
            return version.toLowerCase().includes('clang') && versionNum >= 3.3;
        }

        return false;
    }

    isCfgCompiler(compilerVersion: string) {
        return compilerVersion.includes('clang') || compilerVersion.match(/^([\w-]*-)?g((\+\+)|(cc)|(dc))/g) !== null;
    }

    async processGccDumpOutput(opts, result, removeEmptyPasses, outputFilename) {
        const rootDir = path.dirname(result.inputFilename);

        if (opts.treeDump === false && opts.rtlDump === false && opts.ipaDump === false) {
            return {
                all: [],
                selectedPass: null,
                currentPassOutput: 'Nothing selected for dump:\nselect at least one of Tree/IPA/RTL filter',
                syntaxHighlight: false,
            };
        }

        const output = {
            all: [] as any[],
            selectedPass: opts.pass,
            currentPassOutput: '<No pass selected>',
            syntaxHighlight: false,
        };
        const treeDumpsNotInPasses: any[] = [];

        const selectedPasses: string[] = [];

        if (opts.treeDump) {
            selectedPasses.push('tree');

            // Fake 2 lines as coming from -fdump-passes
            // This allows the insertion of 'gimple' and 'original'
            // tree dumps that are not really part of a tree pass.
            treeDumpsNotInPasses.push(
                [
                    {text: 'tree-original:  ON'},
                    {
                        filename_suffix: 't.original',
                        name: 'original (tree)',
                        command_prefix: '-fdump-tree-original',
                    },
                ],
                [
                    {text: 'tree-gimple:  ON'},
                    {
                        filename_suffix: 't.gimple',
                        name: 'gimple (tree)',
                        command_prefix: '-fdump-tree-gimple',
                    },
                ],
            );
        }

        if (opts.ipaDump) selectedPasses.push('ipa');
        if (opts.rtlDump) selectedPasses.push('rtl');

        // Defaults to a custom file derived from output file name. Works when
        // using the -fdump-tree-foo=FILE variant (!removeEmptyPasses).
        // Will be overriden later if not.
        let dumpFileName = this.getGccDumpFileName(outputFilename);
        let passFound = false;

        const filtered_stderr: any[] = [];
        const toRemoveFromStderr = /^\s*((ipa|tree|rtl)-)|(\*)([\w-]+).*(ON|OFF)$/;

        const dumpPassesLines = treeDumpsNotInPasses.concat(
            (Object.values(result.stderr) as ResultLine[]).map(x => [
                x,
                this.fromInternalGccDumpName(x.text, selectedPasses),
            ]),
        );

        for (const [obj, selectizeObject] of dumpPassesLines) {
            if (selectizeObject) {
                if (opts.pass && opts.pass.name === selectizeObject.name) passFound = true;

                if (removeEmptyPasses) {
                    const f = fs.readdirSync(rootDir).filter(fn => fn.endsWith(selectizeObject.filename_suffix));

                    // pass is enabled, but the dump hasn't produced anything:
                    // don't add it to the drop down menu
                    if (f.length === 0) continue;

                    if (opts.pass && opts.pass.name === selectizeObject.name) dumpFileName = path.join(rootDir, f[0]);
                }

                output.all.push(selectizeObject);
            }

            if (!toRemoveFromStderr.test(obj.text)) {
                filtered_stderr.push(obj);
            }
        }
        result.stderr = filtered_stderr;

        if (opts.pass && passFound) {
            output.currentPassOutput = '';

            if (dumpFileName && (await fs.pathExists(dumpFileName)))
                output.currentPassOutput = await fs.readFile(dumpFileName, 'utf8');
            // else leave the currentPassOutput empty. Can happen when some
            // UI options are changed and a now disabled pass is still
            // requested.

            if (/^\s*$/.test(output.currentPassOutput)) {
                output.currentPassOutput = `Pass '${opts.pass.name}' was requested
but nothing was dumped. Possible causes are:
 - pass is not valid in this (maybe you changed the compiler options);
 - pass is valid but did not emit anything (eg. it was not executed).`;
            } else {
                output.syntaxHighlight = true;
            }
        }
        return output;
    }

    // eslint-disable-next-line no-unused-vars
    async extractDeviceCode(result, filters, compilationInfo) {
        return result;
    }

    async execPostProcess(result, postProcesses, outputFilename, maxSize) {
        const postCommand = `cat "${outputFilename}" | ${postProcesses.join(' | ')}`;
        return this.handlePostProcessResult(result, await this.exec('bash', ['-c', postCommand], {maxOutput: maxSize}));
    }

    preProcess(source: string, filters: CompilerOutputOptions): string {
        if (filters.binary && !this.stubRe.test(source)) {
            source += `\n${this.stubText}\n`;
        }
        return source;
    }

    async postProcess(result, outputFilename: string, filters: ParseFiltersAndOutputOptions) {
        const postProcess = _.compact(this.compiler.postProcess);
        const maxSize = this.env.ceProps('max-asm-size', 64 * 1024 * 1024);
        const optPromise = result.hasOptOutput ? this.processOptOutput(unwrap(result.optPath)) : '';
        const stackUsagePromise = result.hasStackUsageOutput ? this.processStackUsageOutput(result.stackUsagePath) : '';
        const asmPromise =
            (filters.binary || filters.binaryObject) && this.supportsObjdump()
                ? this.objdump(
                      outputFilename,
                      result,
                      maxSize,
                      filters.intel,
                      filters.demangle,
                      filters.binaryObject,
                      false,
                      filters,
                  )
                : (async () => {
                      if (result.asmSize === undefined) {
                          result.asm = '<No output file>';
                          return result;
                      }
                      if (result.asmSize >= maxSize) {
                          result.asm =
                              '<No output: generated assembly was too large' +
                              ` (${result.asmSize} > ${maxSize} bytes)>`;
                          return result;
                      }
                      if (postProcess.length > 0) {
                          return await this.execPostProcess(result, postProcess, outputFilename, maxSize);
                      } else {
                          const contents = await fs.readFile(outputFilename);
                          result.asm = contents.toString();
                          return result;
                      }
                  })();
        return Promise.all([asmPromise, optPromise, stackUsagePromise]);
    }

    handlePostProcessResult(result, postResult): CompilationResult {
        result.asm = postResult.stdout;
        if (postResult.code !== 0) {
            result.asm = `<Error during post processing: ${postResult.code}>`;
            logger.error('Error during post-processing: ', result);
        }
        return result;
    }

    checkOptions(options: string[]) {
        const error = this.env.findBadOptions(options);
        if (error.length > 0) return `Bad options: ${error.join(', ')}`;
        return null;
    }

    // This check for arbitrary user-controlled preprocessor inclusions
    // can be circumvented in more than one way. The goal here is to respond
    // to simple attempts with a clear diagnostic; the service still needs to
    // assume that malicious actors can make the compiler open arbitrary files.
    checkSource(source) {
        const re = /^\s*#\s*i(nclude|mport)(_next)?\s+["<]((\.{1,2}|\/)[^">]*)[">]/;
        const failed: string[] = [];
        for (const [index, line] of utils.splitLines(source).entries()) {
            if (re.test(line)) {
                failed.push(`<stdin>:${index + 1}:1: no absolute or relative includes please`);
            }
        }
        if (failed.length > 0) return failed.join('\n');
        return null;
    }

    protected getArgumentParser(): any {
        const exe = this.compiler.exe.toLowerCase();
        const exeFilename = path.basename(exe);
        if (exeFilename.includes('icc')) {
            return ICCParser;
        } else if (exeFilename.includes('clang++') || exeFilename.includes('icpx')) {
            // check this first as "clang++" matches "g++"
            return ClangParser;
        } else if (exeFilename.includes('clang') || exeFilename.includes('icx')) {
            return ClangCParser;
        } else if (exeFilename.includes('gcc')) {
            return GCCCParser;
        } else if (exeFilename.includes('g++')) {
            return GCCParser;
        }
        //there is a lot of code around that makes this assumption.
        //probably not the best thing to do :D
        return GCCParser;
    }

    async getVersion() {
        logger.info(`Gathering ${this.compiler.id} version information on ${this.compiler.exe}...`);
        if (this.compiler.explicitVersion) {
            logger.debug(`${this.compiler.id} has forced version output: ${this.compiler.explicitVersion}`);
            return {stdout: [this.compiler.explicitVersion], stderr: [], code: 0};
        }
        const execOptions = this.getDefaultExecOptions();
        const versionFlag = this.compiler.versionFlag || ['--version'];
        execOptions.timeoutMs = 0; // No timeout for --version. A sort of workaround for slow EFS/NFS on the prod site
        execOptions.ldPath = this.getSharedLibraryPathsAsLdLibraryPaths([]);

        try {
            return await this.execCompilerCached(this.compiler.exe, versionFlag, execOptions);
        } catch (err) {
            logger.error(`Unable to get version for compiler '${this.compiler.exe}' - ${err}`);
            return null;
        }
    }

    initialiseLibraries(clientOptions) {
        this.supportedLibraries = this.getSupportedLibraries(this.compiler.libsArr, clientOptions.libs[this.lang.id]);
    }

    async getTargetsAsOverrideValues(): Promise<CompilerOverrideOption[]> {
        if (!this.buildenvsetup || !this.buildenvsetup.getCompilerArch()) {
            const parser = this.getArgumentParser();
            const targets = await parser.getPossibleTargets(this);

            return targets.map(target => {
                return {
                    name: target,
                    value: target,
                };
            });
        } else {
            return [];
        }
    }

    async getPossibleStdversAsOverrideValues(): Promise<CompilerOverrideOption[]> {
        const parser = this.getArgumentParser();
        return await parser.getPossibleStdvers(this);
    }

    async populatePossibleOverrides() {
        const targets = await this.getTargetsAsOverrideValues();
        if (targets.length > 0) {
            this.compiler.possibleOverrides?.push({
                name: CompilerOverrideType.arch,
                display_title: 'Target architecture',
                description: c_default_target_description,
                flags: this.getTargetFlags(),
                values: targets,
            });
        }

        const compilerOptions = utils.splitArguments(this.compiler.options);
        if (hasToolchainArg(compilerOptions)) {
            const possibleToolchains: CompilerOverrideOptions = await this.getPossibleToolchains();

            if (possibleToolchains.length > 0) {
                const flag = getToolchainFlagFromOptions(compilerOptions);
                this.compiler.possibleOverrides?.push({
                    name: CompilerOverrideType.toolchain,
                    display_title: 'Toolchain',
                    description: c_default_toolchain_description,
                    flags: [flag + '<value>'],
                    values: possibleToolchains,
                });
            }
        }

        const stdVersions = await this.getPossibleStdversAsOverrideValues();
        if (stdVersions.length > 0) {
            this.compiler.possibleOverrides?.push({
                name: CompilerOverrideType.stdver,
                display_title: 'Std version',
                description: this.getStdVerOverrideDescription(),
                flags: this.getStdverFlags(),
                values: stdVersions,
            });
        }
    }

    getStdVerOverrideDescription(): string {
        return 'Change the C/C++ standard version of the compiler.';
    }

    getStdverFlags(): string[] {
        return ['-std=<value>'];
    }

    getTargetFlags(): string[] {
        if (this.compiler.supportsMarch) return [`-march=${c_value_placeholder}`];
        if (this.compiler.supportsTargetIs) return [`--target=${c_value_placeholder}`];
        if (this.compiler.supportsTarget) return ['--target', c_value_placeholder];

        return [];
    }

    async getPossibleToolchains(): Promise<CompilerOverrideOptions> {
        return this.env.getPossibleToolchains();
    }

    async initialise(mtime: Date, clientOptions, isPrediscovered = false) {
        this.mtime = mtime;

        if (this.buildenvsetup) {
            await this.buildenvsetup.initialise(async (compiler, args, options) => {
                return this.execCompilerCached(compiler, args, options);
            });
        }

        if (this.getRemote()) return this;

        const compiler = this.compiler.exe;
        let version = this.compiler.version || '';
        if (!isPrediscovered) {
            const versionRe = new RegExp(this.compiler.versionRe || '.*', 'i');
            const result = await this.getVersion();
            if (!result) return null;
            if (result.code !== 0) {
                logger.warn(`Compiler '${compiler}' - non-zero result ${result.code}`);
            }
            const fullVersion = result.stdout + result.stderr;
            _.each(utils.splitLines(fullVersion), line => {
                if (version) return;
                const match = line.match(versionRe);
                if (match) version = match[0];
            });
            if (!version) {
                logger.error(`Unable to find compiler version for '${compiler}' with re ${versionRe}:`, result);
                return null;
            }
            logger.debug(`${compiler} is version '${version}'`);
            this.compiler.version = version;
            this.compiler.fullVersion = fullVersion;
            this.compiler.supportsCfg = this.isCfgCompiler(version);
            // all C/C++ compilers support -E
            this.compiler.supportsPpView = this.compiler.lang === 'c' || this.compiler.lang === 'c++';
            this.compiler.supportsAstView = this.couldSupportASTDump(version);
        }

        try {
            this.cmakeBaseEnv = await this.getCmakeBaseEnv();
        } catch (e) {
            logger.error(e);
        }

        this.initialiseLibraries(clientOptions);

        if (isPrediscovered) {
            logger.info(`${compiler} ${version} is ready`);
            if (this.compiler.cachedPossibleArguments) {
                this.possibleArguments.populateOptions(this.compiler.cachedPossibleArguments);
                delete this.compiler.cachedPossibleArguments;
            }
            return this;
        } else {
            const initResult = await this.getArgumentParser().parse(this);

            await this.populatePossibleOverrides();

            logger.info(`${compiler} ${version} is ready`);
            return initResult;
        }
    }

    getModificationTime(): number | undefined {
        return this.mtime ? this.mtime.getTime() : undefined;
    }

    getInfo() {
        return this.compiler;
    }

    getDefaultFilters(): ParseFiltersAndOutputOptions {
        return {
            binary: false,
            execute: false,
            demangle: true,
            intel: true,
            commentOnly: true,
            directives: true,
            labels: true,
            optOutput: false,
            libraryCode: false,
            trim: false,
            binaryObject: false,
            debugCalls: false,
        };
    }
}
