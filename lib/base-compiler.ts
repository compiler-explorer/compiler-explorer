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

import os from 'os';
import path from 'path';

import fs from 'fs-extra';
import * as PromClient from 'prom-client';
import temp from 'temp';
import _ from 'underscore';

import {splitArguments, unique} from '../shared/common-utils.js';
import {OptRemark} from '../static/panes/opt-view.interfaces.js';
import {PPOptions} from '../static/panes/pp-view.interfaces.js';
import {ParsedAsmResultLine} from '../types/asmresult/asmresult.interfaces.js';
import {ClangirBackendOptions} from '../types/compilation/clangir.interfaces.js';
import {
    ActiveTool,
    BuildResult,
    BuildStep,
    BypassCache,
    bypassCompilationCache,
    bypassExecutionCache,
    CacheKey,
    CmakeCacheKey,
    CompilationCacheKey,
    CompilationInfo,
    CompilationResult,
    ExecutionOptions,
    ExecutionOptionsWithEnv,
    ExecutionParams,
    FiledataPair,
    GccDumpOptions,
    LibsAndOptions,
} from '../types/compilation/compilation.interfaces.js';
import {
    CompilerOverrideOption,
    CompilerOverrideOptions,
    CompilerOverrideType,
    ConfiguredOverrides,
} from '../types/compilation/compiler-overrides.interfaces.js';
import {LLVMIrBackendOptions} from '../types/compilation/ir.interfaces.js';
import type {
    OptPipelineBackendOptions,
    OptPipelineOutput,
} from '../types/compilation/opt-pipeline-output.interfaces.js';
import type {CompilerInfo, PreliminaryCompilerInfo} from '../types/compiler.interfaces.js';
import {
    BasicExecutionResult,
    ExecutableExecutionOptions,
    RuntimeToolType,
    UnprocessedExecResult,
} from '../types/execution/execution.interfaces.js';
import type {CompilerOutputOptions, ParseFiltersAndOutputOptions} from '../types/features/filters.interfaces.js';
import {InstructionSet} from '../types/instructionsets.js';
import type {Language} from '../types/languages.interfaces.js';
import type {SelectedLibraryVersion} from '../types/libraries/libraries.interfaces.js';
import type {ResultLine} from '../types/resultline/resultline.interfaces.js';
import {type ToolResult, type ToolTypeKey} from '../types/tool.interfaces.js';

import {moveArtifactsIntoResult} from './artifact-utils.js';
import {assert, unwrap} from './assert.js';
import type {BuildEnvDownloadInfo} from './buildenvsetup/buildenv.interfaces.js';
import {BuildEnvSetupBase, getBuildEnvTypeByKey} from './buildenvsetup/index.js';
import * as cfg from './cfg/cfg.js';
import {CompilationEnvironment} from './compilation-env.js';
import {CompilerArguments} from './compiler-arguments.js';
import {
    BaseParser,
    ClangCParser,
    ClangirParser,
    ClangParser,
    GCCCParser,
    GCCParser,
    ICCParser,
} from './compilers/argument-parsers.js';
import {BaseDemangler, getDemanglerTypeByKey} from './demangler/index.js';
import {LLVMIRDemangler} from './demangler/llvm.js';
import * as exec from './exec.js';
import {BaseExecutionTriple} from './execution/base-execution-triple.js';
import {IExecutionEnvironment} from './execution/execution-env.interfaces.js';
import {RemoteExecutionQuery} from './execution/execution-query.js';
import {matchesCurrentHost} from './execution/execution-triple.js';
import {getExecutionEnvironmentByKey} from './execution/index.js';
import {RemoteExecutionEnvironment} from './execution/remote-execution-env.js';
import {ExternalParserBase} from './external-parsers/base.js';
import {getExternalParserByKey} from './external-parsers/index.js';
import {ParsedRequest} from './handlers/compile.js';
import {InstructionSets} from './instructionsets.js';
import {languages} from './languages.js';
import {LlvmAstParser} from './llvm-ast.js';
import {LlvmIrParser} from './llvm-ir.js';
import {processRawOptRemarks} from './llvm-opt-transformer.js';
import {logger} from './logger.js';
import {getObjdumperTypeByKey} from './objdumper/index.js';
import {ClientOptionsType, OptionsHandlerLibrary, VersionInfo} from './options-handler.js';
import {Packager} from './packager.js';
import type {IAsmParser} from './parsers/asm-parser.interfaces.js';
import {AsmParser} from './parsers/asm-parser.js';
import {LlvmPassDumpParser} from './parsers/llvm-pass-dump-parser.js';
import type {PropertyGetter} from './properties.interfaces.js';
import {propsFor} from './properties.js';
import {HeaptrackWrapper} from './runtime-tools/heaptrack-wrapper.js';
import {LibSegFaultHelper} from './runtime-tools/libsegfault-helper.js';
import {SentryCapture} from './sentry.js';
import * as StackUsage from './stack-usage-transformer.js';
import {
    clang_style_sysroot_flag,
    getSpecificTargetBasedOnToolchainPath,
    getSysrootByToolchainPath,
    getToolchainFlagFromOptions,
    getToolchainPath,
    getToolchainPathWithOptionsArr,
    hasSysrootArg,
    hasToolchainArg,
    removeToolchainArg,
    replaceSysrootArg,
    replaceToolchainArg,
} from './toolchain-utils.js';
import type {ITool} from './tooling/base-tool.interface.js';
import * as utils from './utils.js';

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

export interface SimpleOutputFilenameCompiler {
    getOutputFilename(dirPath: string): string;
}

function isOutputLikelyLllvmIr(compilerOptions) {
    return compilerOptions && (compilerOptions.includes('-emit-llvm') || compilerOptions.includes('-mlir-to-llvmir'));
}

export class BaseCompiler {
    public compiler: CompilerInfo;
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
    protected supportedLibraries?: Record<string, OptionsHandlerLibrary>;
    protected packager: Packager;
    protected executionType: string;
    protected sandboxType: string;
    protected defaultRpathFlag: string = '-Wl,-rpath,';
    private static objdumpAndParseCounter = new PromClient.Counter({
        name: 'ce_objdumpandparsetime_total',
        help: 'Time spent on objdump and parsing of objdumps',
        labelNames: [],
    });
    protected executionEnvironmentClass: any;

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
        this.compilerProps = this.env.getCompilerPropsForLanguage(this.lang.id);
        this.compiler.supportsIntel = !!this.compiler.intelAsm;

        this.alwaysResetLdPath = this.env.ceProps('alwaysResetLdPath');
        this.delayCleanupTemp = this.env.ceProps('delayCleanupTemp', false);
        this.stubRe = new RegExp(this.compilerProps('stubRe', ''));
        this.stubText = this.compilerProps('stubText', '');
        this.compilerWrapper = this.compilerProps('compiler-wrapper');

        const executionEnvironmentClassStr = this.compilerProps<string>('executionEnvironmentClass', 'local');
        this.executionEnvironmentClass = getExecutionEnvironmentByKey(executionEnvironmentClassStr);

        if (!this.compiler.options) this.compiler.options = '';
        if (!this.compiler.optArg) this.compiler.optArg = '';
        if (!this.compiler.supportsOptOutput) this.compiler.supportsOptOutput = false;
        if (!this.compiler.supportsVerboseAsm) this.compiler.supportsVerboseAsm = false;

        if (!compilerInfo.disabledFilters) this.compiler.disabledFilters = [];
        else if (typeof (this.compiler.disabledFilters as any) === 'string') {
            // When first loaded from props it may be a string so we split it here
            // I'd like a better way to do this that doesn't involve type hacks
            // TODO(jeremy-rifkin): branch may now be obsolete?
            this.compiler.disabledFilters = (this.compiler.disabledFilters as any).split(',');
        }

        const execProps = propsFor('execution');
        this.executionType = execProps('executionType', 'none');
        this.sandboxType = execProps('sandboxType', 'none');

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
                this.compiler.instructionSet = isets.getCompilerInstructionSetHint(
                    this.buildenvsetup.compilerArch,
                    this.compiler.exe,
                );
            } else {
                const temp = new BuildEnvSetupBase(this.compiler, this.env);
                this.compiler.instructionSet = isets.getCompilerInstructionSetHint(
                    temp.compilerArch,
                    this.compiler.exe,
                );
            }
        }

        this.packager = new Packager();
    }

    copyAndFilterLibraries(allLibraries: Record<string, OptionsHandlerLibrary>, filter: string[]) {
        const filterLibAndVersion = filter.map(lib => {
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

        const copiedLibraries: Record<string, any> = {};
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
            }) as Record<string, VersionInfo>;

            copiedLibraries[libid] = libcopy;
        });

        return copiedLibraries;
    }

    getSupportedLibraries(supportedLibrariesArr: string[], allLibs: Record<string, OptionsHandlerLibrary>) {
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

    async newTempDir(): Promise<string> {
        // `temp` caches the os tmp dir on import (which we may change), so here we ensure we use the current os.tmpdir
        // each time.
        return await temp.mkdir({prefix: utils.ce_temp_prefix, dir: os.tmpdir()});
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

    protected getCompilerCacheKey(
        compiler: string,
        args: string[],
        options: ExecutionOptionsWithEnv,
    ): CompilationCacheKey {
        return {mtime: this.mtime, compiler, args, options};
    }

    public async execCompilerCached(
        compiler: string,
        args: string[],
        options?: ExecutionOptionsWithEnv,
    ): Promise<UnprocessedExecResult> {
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
            result = await this.env.enqueue(async () => await this.exec(compiler, args, options));
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

        if (options.createAndUseTempDir) fs.remove(options.customCwd!, () => {});

        return result;
    }

    protected getExtraPaths(): string[] {
        const ninjaPath = this.env.ceProps('ninjaPath', '');
        if (this.compiler.extraPath && this.compiler.extraPath.length > 0) {
            return [ninjaPath, ...this.compiler.extraPath];
        }
        return [ninjaPath];
    }

    getDefaultExecOptions(): ExecutionOptionsWithEnv {
        const env = this.env.getEnv(this.compiler.needsMulti);
        if (!env.PATH) env.PATH = '';
        env.PATH = [...this.getExtraPaths(), env.PATH].filter(Boolean).join(path.delimiter);

        return {
            timeoutMs: this.env.ceProps('compileTimeoutMs', 7500),
            maxErrorOutput: this.env.ceProps('max-error-output', 5000),
            env,
            wrapper: this.compilerWrapper,
        };
    }

    getCompilerResultLanguageId(filters?: ParseFiltersAndOutputOptions): string | undefined {
        return undefined;
    }

    getTargetHintFromCompilerArgs(args: string[]): string | undefined {
        const allFlags = this.getAllPossibleTargetFlags();

        if (allFlags.length > 0) {
            for (const possibleFlag of allFlags) {
                // possible.flags contains either something like ['--target', '<value>'] or ['--target=<value>'], we want the flags without <value>
                const filteredFlags: string[] = [];
                let targetFlagOffset = -1;
                for (const [i, flag] of possibleFlag.entries()) {
                    if (flag.includes(c_value_placeholder)) {
                        filteredFlags.push(flag.replace(c_value_placeholder, ''));
                        targetFlagOffset = i;
                    } else {
                        filteredFlags.push(flag);
                    }
                }

                if (targetFlagOffset === -1) continue;

                // try to find matching flags in args
                let foundFlag = -1;
                for (const arg of args) {
                    if (arg.startsWith(filteredFlags[foundFlag + 1])) {
                        foundFlag = foundFlag + 1;
                    }

                    if (foundFlag === targetFlagOffset) {
                        if (arg.length > filteredFlags[foundFlag].length) {
                            return arg.substring(filteredFlags[foundFlag].length);
                        }
                        return arg;
                    }
                }
            }
        }

        return undefined;
    }

    getInstructionSetFromCompilerArgs(args: string[]): InstructionSet {
        try {
            const archHint = this.getTargetHintFromCompilerArgs(args);
            if (archHint) {
                const isets = new InstructionSets();
                return isets.getCompilerInstructionSetHint(archHint, this.compiler.exe);
            }
        } catch (e) {
            logger.debug('Unexpected error in getInstructionSetFromCompilerArgs(): ', e);
        }

        if (this.compiler.instructionSet) {
            return this.compiler.instructionSet;
        } else {
            return 'amd64';
        }
    }

    async runCompiler(
        compiler: string,
        options: string[],
        inputFilename: string,
        execOptions: ExecutionOptionsWithEnv,
        filters?: ParseFiltersAndOutputOptions,
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
            languageId: this.getCompilerResultLanguageId(filters),
            instructionSet: this.getInstructionSetFromCompilerArgs(options),
        };
    }

    async runCompilerRawOutput(
        compiler: string,
        options: string[],
        inputFilename: string,
        execOptions: ExecutionOptionsWithEnv,
    ) {
        if (!execOptions) {
            execOptions = this.getDefaultExecOptions();
        }

        if (!execOptions.customCwd) {
            execOptions.customCwd = path.dirname(inputFilename);
        }

        const result = await this.exec(compiler, options, execOptions);
        return {
            ...result,
            inputFilename: inputFilename,
        };
    }

    supportsObjdump() {
        return !!this.objdumperClass;
    }

    getObjdumpOutputFilename(defaultOutputFilename: string): string {
        return defaultOutputFilename;
    }

    postProcessObjdumpOutput(output: string) {
        return output;
    }

    async objdump(
        outputFilename: string,
        result: any,
        maxSize: number,
        intelAsm: boolean,
        demangle: boolean,
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

    transformToCompilationResult(input: UnprocessedExecResult, inputFilename: string): CompilationResult {
        const transformedInput = input.filenameTransform(inputFilename);

        return {
            inputFilename: inputFilename,
            languageId: input.languageId,
            ...this.processExecutionResult(input, transformedInput),
        };
    }

    protected filename(fn: string) {
        return fn;
    }

    getGccDumpFileName(outputFilename: string) {
        return utils.changeExtension(outputFilename, '.dump');
    }

    getGccDumpOptions(gccDumpOptions: Record<string, any>, outputFilename: string) {
        const addOpts = ['-fdump-passes'];

        // Build dump options to append to the end of the -fdump command-line flag.
        // GCC accepts these options as a list of '-' separated names that may
        // appear in any order.
        let flags = '';
        if (gccDumpOptions.dumpFlags.gimpleFe !== false) {
            flags += '-gimple';
        }
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
        if (this.compiler.supportsVerboseAsm) {
            options = options.concat(filters.commentOnly ? '-fno-verbose-asm' : '-fverbose-asm');
        }
        if (!filters.binary && !filters.binaryObject) options = options.concat('-S');
        else if (filters.binaryObject) options = options.concat('-c');

        return options;
    }

    findLibVersion(selectedLib: SelectedLibraryVersion): false | VersionInfo {
        if (!this.supportedLibraries) return false;

        const foundLib = _.find(this.supportedLibraries, (o, libId) => libId === selectedLib.id);
        if (!foundLib) return false;

        const result: VersionInfo | undefined = _.find(
            foundLib.versions,
            (o: VersionInfo, versionId: string): boolean => {
                if (versionId === selectedLib.version) return true;
                return !!(o.alias && o.alias.includes(selectedLib.version));
            },
        );

        if (!result) return false;

        result.name = foundLib.name;
        return result;
    }

    protected optionsForDemangler(filters?: ParseFiltersAndOutputOptions): string[] {
        return [...this.compiler.demanglerArgs];
    }

    findAutodetectStaticLibLink(linkname: string): SelectedLibraryVersion | false {
        const foundLib = _.findKey(this.supportedLibraries!, lib => {
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

    getSortedStaticLibraries(libraries: SelectedLibraryVersion[]) {
        const dictionary: Record<string, VersionInfo> = {};
        const links = unique(
            libraries
                .map(selectedLib => {
                    const foundVersion = this.findLibVersion(selectedLib);
                    if (!foundVersion) return false;

                    return foundVersion.staticliblink.map(lib => {
                        if (lib) {
                            dictionary[lib] = foundVersion;
                            return [lib, foundVersion.dependencies];
                        } else {
                            return false;
                        }
                    });
                })
                .flat(3),
        );

        const sortedlinks: string[] = [];

        for (const libToInsertName of links) {
            if (libToInsertName) {
                const libToInsertObj = dictionary[libToInsertName];

                let idxToInsert = sortedlinks.length;
                for (const [idx, libCompareName] of sortedlinks.entries()) {
                    const libCompareObj: VersionInfo = dictionary[libCompareName];

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
            }
        }

        return sortedlinks;
    }

    getStaticLibraryLinks(libraries: SelectedLibraryVersion[], libPaths: string[] = []): string[] {
        const linkFlag = this.compiler.linkFlag || '-l';

        return this.getSortedStaticLibraries(libraries)
            .filter(Boolean)
            .map(lib => linkFlag + lib);
    }

    getSharedLibraryLinks(libraries: SelectedLibraryVersion[]): string[] {
        const linkFlag = this.compiler.linkFlag || '-l';

        return libraries
            .flatMap(selectedLib => {
                const foundVersion = this.findLibVersion(selectedLib);
                if (!foundVersion) return false;

                return foundVersion.liblink.map(lib => {
                    if (lib) {
                        return linkFlag + lib;
                    } else {
                        return false;
                    }
                });
            })
            .filter(Boolean) as string[];
    }

    getSharedLibraryPaths(libraries: SelectedLibraryVersion[], dirPath?: string): string[] {
        return libraries.flatMap(selectedLib => {
            const foundVersion = this.findLibVersion(selectedLib);
            if (!foundVersion) return [];

            const paths = [...foundVersion.libpath];
            if (this.buildenvsetup && !this.buildenvsetup.extractAllToRoot && dirPath) {
                paths.push(path.join(dirPath, selectedLib.id, 'lib'));
            }
            return paths;
        });
    }

    protected getSharedLibraryPathsAsArguments(
        libraries: SelectedLibraryVersion[],
        libDownloadPath: string | undefined,
        toolchainPath: string | undefined,
        dirPath: string,
    ): string[] {
        const pathFlag = this.compiler.rpathFlag || this.defaultRpathFlag;
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
            this.getSharedLibraryPaths(libraries, dirPath).map(path => pathFlag + path),
            this.getSharedLibraryPaths(libraries, dirPath).map(path => libPathFlag + path),
        );
    }

    protected getSharedLibraryPathsAsLdLibraryPaths(libraries: SelectedLibraryVersion[], dirPath?: string): string[] {
        let paths = '';
        if (!this.alwaysResetLdPath) {
            paths = process.env.LD_LIBRARY_PATH || '';
        }
        return _.union(
            paths.split(path.delimiter).filter(p => !!p),
            this.compiler.ldPath,
            this.getSharedLibraryPaths(libraries, dirPath),
        );
    }

    getSharedLibraryPathsAsLdLibraryPathsForExecution(key: CacheKey, dirPath: string): string[] {
        let paths = '';
        if (!this.alwaysResetLdPath) {
            paths = process.env.LD_LIBRARY_PATH || '';
        }
        return _.union(
            paths.split(path.delimiter).filter(p => !!p),
            this.compiler.ldPath,
            this.getExtraLdPaths(key),
            this.compiler.libPath,
            this.getSharedLibraryPaths(key.libraries, dirPath),
        );
    }

    getExtraLdPaths(key: CacheKey): string[] {
        let toolchainPath: any;
        if (key.options) {
            toolchainPath = getToolchainPathWithOptionsArr(this.compiler.exe, key.options) || this.toolchainPath;
        }

        if (toolchainPath) {
            const sysrootPath = getSysrootByToolchainPath(toolchainPath);
            if (sysrootPath) {
                return [path.join(sysrootPath, 'lib')];
            }
        }

        return [];
    }

    getIncludeArguments(libraries: SelectedLibraryVersion[], dirPath: string): string[] {
        const includeFlag = this.compiler.includeFlag || '-I';
        return libraries.flatMap(selectedLib => {
            const foundVersion = this.findLibVersion(selectedLib);
            if (!foundVersion) return [];

            const paths = foundVersion.path.map(path => includeFlag + path);
            if (foundVersion.packagedheaders) {
                const includePath = path.join(dirPath, selectedLib.id, 'include');
                paths.push(includeFlag + includePath);
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
        libraries: SelectedLibraryVersion[],
        overrides: ConfiguredOverrides,
    ) {
        let options = this.optionsForFilter(filters, outputFilename, userOptions);
        backendOptions = backendOptions || {};

        options = options.concat(this.optionsForBackend(backendOptions, outputFilename));

        if (this.compiler.options) {
            options = options.concat(splitArguments(this.compiler.options));
        }

        if (this.compiler.supportsOptOutput && backendOptions.produceOptInfo) {
            options = options.concat(unwrap(this.compiler.optArg));
        }
        if (this.compiler.supportsStackUsageOutput && backendOptions.produceStackUsageInfo) {
            options = options.concat(unwrap(this.compiler.stackUsageArg));
        }

        const toolchainPath = this.getDefaultOrOverridenToolchainPath(backendOptions.overrides || []);

        const dirPath = path.dirname(inputFilename);

        const libIncludes = this.getIncludeArguments(libraries, dirPath);
        const libOptions = this.getLibraryOptions(libraries);
        const {libLinks, libPathsAsFlags, staticLibLinks} = this.getLibLinkInfo(
            filters,
            libraries,
            toolchainPath,
            dirPath,
        );

        userOptions = this.filterUserOptions(userOptions) || [];
        [options, overrides] = this.fixIncompatibleOptions(options, userOptions, overrides);
        options = this.changeOptionsBasedOnOverrides(options, overrides);

        return this.orderArguments(
            options,
            inputFilename,
            libIncludes,
            libOptions,
            libPathsAsFlags,
            libLinks,
            userOptions,
            staticLibLinks,
        );
    }

    protected getLibLinkInfo(
        filters: ParseFiltersAndOutputOptions,
        libraries: SelectedLibraryVersion[],
        toolchainPath: string,
        dirPath: string,
    ) {
        let libLinks: string[] = [];
        let libPathsAsFlags: string[] = [];
        let staticLibLinks: string[] = [];

        if (filters.binary) {
            libLinks = (this.getSharedLibraryLinks(libraries).filter(Boolean) as string[]) || [];
            libPathsAsFlags = this.getSharedLibraryPathsAsArguments(libraries, undefined, toolchainPath, dirPath);
            const libPaths = this.getSharedLibraryPaths(libraries, dirPath);
            staticLibLinks = (this.getStaticLibraryLinks(libraries, libPaths).filter(Boolean) as string[]) || [];
        }
        return {libLinks, libPathsAsFlags, staticLibLinks};
    }

    protected fixIncompatibleOptions(
        options: string[],
        userOptions: string[],
        overrides: ConfiguredOverrides,
    ): [string[], ConfiguredOverrides] {
        return [options, overrides];
    }

    filterUserOptions(userOptions: string[]): string[] {
        return userOptions;
    }

    async generateAST(inputFilename: string, options: string[]): Promise<ResultLine[]> {
        // These options make Clang produce an AST dump
        const newOptions = options
            .filter(option => option !== '-fcolor-diagnostics')
            .concat(['-Xclang', '-ast-dump', '-fsyntax-only']);

        const execOptions = this.getDefaultExecOptions();
        // A higher max output is needed for when the user includes headers
        execOptions.maxOutput = 1024 * 1024 * 1024;

        return this.llvmAst.processAst(
            await this.runCompiler(this.compiler.exe, newOptions, this.filename(inputFilename), execOptions),
        );
    }

    async generatePP(
        inputFilename: string,
        compilerOptions: string[],
        rawPpOptions: PPOptions,
    ): Promise<{numberOfLinesFiltered: number; output: string}> {
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

    filterPP(stdout: string): any[] {
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

    async applyClangFormat(output: string): Promise<string> {
        try {
            // Currently hard-coding llvm style
            const {stdout, stderr} = await this.env.formattingService.format('clangformat', output, {
                baseStyle: 'LLVM',
                tabWidth: 4,
                useSpaces: true,
            });
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
        const newOptions = options
            // `-E` /`-fsave-optimization-record` switches caused simultaneus writes into some output files,
            // see bugs #5854 / #6745
            .filter(option => !['-fcolor-diagnostics', '-E', '-fsave-optimization-record'].includes(option))
            .concat(unwrap(this.compiler.irArg)); // produce IR

        if (irOptions.noDiscardValueNames && this.compiler.optPipeline?.noDiscardValueNamesArg) {
            newOptions.push(...this.compiler.optPipeline.noDiscardValueNamesArg);
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

        if (result.asm.length > 0 && result.asm[result.asm.length - 1].text === '[truncated; too many lines]') {
            return result;
        }

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
        output: CompilationResult,
        irOptions: LLVMIrBackendOptions,
        filters: ParseFiltersAndOutputOptions,
    ): Promise<{
        asm: ParsedAsmResultLine[];
        languageId: string;
    }> {
        const irPath = this.getIrOutputFilename(output.inputFilename!, filters);
        if (await fs.pathExists(irPath)) {
            const output = await fs.readFile(irPath, 'utf8');
            return await this.llvmIr.process(output, irOptions);
        }
        return {
            asm: [{text: 'Internal error; unable to open output path'}],
            languageId: 'llvm-ir',
        };
    }

    getClangirOutputFilename(inputFilename: string) {
        return utils.changeExtension(inputFilename, '.cir');
    }

    async generateClangir(
        inputFilename: string,
        options: string[],
        clangirOptions: ClangirBackendOptions,
    ): Promise<ResultLine[]> {
        const outputFilename = this.getClangirOutputFilename(inputFilename);

        const newOptions = [...options];
        if (clangirOptions.flatCFG) {
            newOptions.push('-Xclang', '-emit-cir-flat');
        } else {
            newOptions.push('-Xclang', '-emit-cir');
        }

        // Replace `-o <name>.s` with `-o <name>.cir`
        newOptions.splice(options.indexOf('-o'), 2);
        newOptions.push('-o', outputFilename);

        const execOptions = this.getDefaultExecOptions();
        const output = await this.runCompiler(this.compiler.exe, newOptions, this.filename(inputFilename), execOptions);
        if (output.code !== 0) {
            return [{text: 'Failed to run compiler to get ClangIR code'}];
        }

        if (await utils.fileExists(outputFilename)) {
            const content = await fs.readFile(outputFilename, 'utf8');
            return content.split('\n').map(line => ({
                text: line,
            }));
        }
        return [{text: 'Internal error; unable to open output path'}];
    }

    async generateOptPipeline(
        inputFilename: string,
        options: string[],
        filters: ParseFiltersAndOutputOptions,
        optPipelineOptions: OptPipelineBackendOptions,
    ): Promise<OptPipelineOutput | undefined> {
        // These options make Clang produce the pass dumps
        const newOptions = options
            .filter(option => option !== '-fcolor-diagnostics')
            .concat(unwrap(this.compiler.optPipeline?.arg))
            .concat(optPipelineOptions.fullModule ? unwrap(this.compiler.optPipeline?.moduleScopeArg) : [])
            .concat(
                optPipelineOptions.noDiscardValueNames ? unwrap(this.compiler.optPipeline?.noDiscardValueNamesArg) : [],
            )
            .concat(this.compiler.debugPatched ? ['-mllvm', '--debug-to-stdout'] : []);

        const execOptions = this.getDefaultExecOptions();
        execOptions.maxOutput = 1024 * 1024 * 1024;

        const compileStart = performance.now();
        const output = await this.runCompiler(this.compiler.exe, newOptions, this.filename(inputFilename), execOptions);
        const compileEnd = performance.now();

        if (output.timedOut) {
            return {
                error: 'Invocation timed out',
                results: {},
                compileTime: output.execTime || compileEnd - compileStart,
            };
        }

        if (output.truncated) {
            return {
                error: 'Exceeded max output limit',
                results: {},
                compileTime: output.execTime || compileEnd - compileStart,
            };
        }

        try {
            const parseStart = performance.now();
            const optPipeline = await this.processOptPipeline(
                output,
                filters,
                optPipelineOptions,
                this.compiler.debugPatched,
            );
            const parseEnd = performance.now();

            if (optPipelineOptions.demangle) {
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
                    results: await demangler.demangleLLVMPasses(optPipeline),
                    compileTime: compileEnd - compileStart,
                    parseTime: parseEnd - parseStart,
                };
            } else {
                return {
                    results: optPipeline,
                    compileTime: compileEnd - compileStart,
                    parseTime: parseEnd - parseStart,
                };
            }
        } catch (e: any) {
            return {
                error: e.toString(),
                results: {},
                compileTime: compileEnd - compileStart,
            };
        }
    }

    async processOptPipeline(
        output: CompilationResult,
        filters: ParseFiltersAndOutputOptions,
        optPipelineOptions: OptPipelineBackendOptions,
        debugPatched?: boolean,
    ) {
        return this.llvmPassDumpParser.process(
            debugPatched ? output.stdout : output.stderr,
            filters,
            optPipelineOptions,
        );
    }

    getRustMacroExpansionOutputFilename(inputFilename: string) {
        return utils.changeExtension(inputFilename, '.expanded.rs');
    }

    getRustHirOutputFilename(inputFilename: string) {
        return utils.changeExtension(inputFilename, '.hir');
    }

    getRustMirOutputFilename(outputFilename: string) {
        return utils.changeExtension(outputFilename, '.mir');
    }

    getHaskellCoreOutputFilename(inputFilename: string) {
        return utils.changeExtension(inputFilename, '.dump-simpl');
    }

    getHaskellStgOutputFilename(inputFilename: string) {
        return utils.changeExtension(inputFilename, '.dump-stg-final');
    }

    getHaskellCmmOutputFilename(inputFilename: string) {
        return utils.changeExtension(inputFilename, '.dump-cmm');
    }

    // Currently called for getting macro expansion and HIR.
    // It returns the content of the output file created after using -Z unpretty=<unprettyOpt>.
    // The outputFriendlyName is a free form string used in case of error.
    async generateRustUnprettyOutput(
        inputFilename: string,
        options: string[],
        unprettyOpt: string,
        outputFilename: string,
        outputFriendlyName: string,
    ): Promise<ResultLine[]> {
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

    async generateRustMacroExpansion(inputFilename: string, options: string[]): Promise<ResultLine[]> {
        const macroExpPath = this.getRustMacroExpansionOutputFilename(inputFilename);
        return this.generateRustUnprettyOutput(inputFilename, options, 'expanded', macroExpPath, 'Macro Expansion');
    }

    async generateRustHir(inputFilename: string, options: string[]): Promise<ResultLine[]> {
        const hirPath = this.getRustHirOutputFilename(inputFilename);
        return this.generateRustUnprettyOutput(inputFilename, options, 'hir-tree', hirPath, 'HIR');
    }

    async processRustMirOutput(outputFilename: string, output: CompilationResult): Promise<ResultLine[]> {
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

    async processHaskellExtraOutput(outpath: string, output: CompilationResult): Promise<ResultLine[]> {
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

    getIrOutputFilename(inputFilename: string, filters?: ParseFiltersAndOutputOptions): string {
        // filters are passed because rust needs to know whether a binary is being produced or not
        return utils.changeExtension(inputFilename, '.ll');
    }

    isCacheKey(key: CacheKey | CompilationCacheKey | undefined): key is CacheKey {
        return key !== undefined && (key as CacheKey).backendOptions !== undefined;
    }

    getOutputFilename(dirPath: string, outputFilebase: string, key?: CacheKey | CompilationCacheKey): string {
        let filename: string;

        if (this.isCacheKey(key) && key.backendOptions.customOutputFilename) {
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

    getExecutableFilename(dirPath: string, outputFilebase: string, key?: CacheKey | CompilationCacheKey) {
        return this.getOutputFilename(dirPath, outputFilebase, key);
    }

    async processGnatDebugOutput(inputFilename: string, result: CompilationResult) {
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
    fromInternalGccDumpName(internalDumpName: string, selectedPasses: string[]) {
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

    async checkOutputFileAndDoPostProcess(
        asmResult: CompilationResult,
        outputFilename: string,
        filters: ParseFiltersAndOutputOptions,
    ) {
        try {
            const stat = await fs.stat(outputFilename);
            asmResult.asmSize = stat.size;
        } catch (e) {
            // Ignore errors
        }
        return await this.postProcess(asmResult, outputFilename, filters);
    }

    runToolsOfType(tools: ActiveTool[], type: ToolTypeKey, compilationInfo: CompilationInfo): Promise<ToolResult>[] {
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

    buildExecutable(compiler: string, options: string[], inputFilename: string, execOptions: ExecutionOptionsWithEnv) {
        // default implementation, but should be overridden by compilers
        return this.runCompiler(compiler, options, inputFilename, execOptions, {execute: true, binary: true});
    }

    protected maskPathsInArgumentsForUser(args: string[]): string[] {
        const maskedArgs: string[] = [];
        for (const arg of args) {
            maskedArgs.push(utils.maskRootdir(arg));
        }
        return maskedArgs;
    }

    async getRequiredLibraryVersions(libraries: SelectedLibraryVersion[]): Promise<Record<string, VersionInfo>> {
        const libraryDetails: Record<string, VersionInfo> = {};
        _.each(libraries, selectedLib => {
            const foundVersion = this.findLibVersion(selectedLib);
            if (foundVersion) libraryDetails[selectedLib.id] = foundVersion;
        });
        return libraryDetails;
    }

    async setupBuildEnvironment(key: CacheKey, dirPath: string, binary: boolean): Promise<BuildEnvDownloadInfo[]> {
        if (this.buildenvsetup) {
            const libraryDetails = await this.getRequiredLibraryVersions(key.libraries);
            return this.buildenvsetup.setup(key, dirPath, libraryDetails, binary);
        } else {
            return [];
        }
    }

    protected async writeMultipleFiles(files: FiledataPair[], dirPath: string) {
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
        files: FiledataPair[],
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

    protected async writeAllFilesCMake(
        dirPath: string,
        source: string,
        files: FiledataPair[],
        filters: ParseFiltersAndOutputOptions,
    ) {
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

    async buildExecutableInFolder(key: CacheKey, dirPath: string): Promise<BuildResult> {
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
        execOptions.ldPath = this.getSharedLibraryPathsAsLdLibraryPaths(key.libraries, dirPath);

        this.applyOverridesToExecOptions(execOptions, overrides);

        const result = await this.buildExecutable(key.compiler.exe, compilerArguments, inputFilename, execOptions);

        return await this.afterBuild(key, dirPath, {
            ...result,
            downloads,
            executableFilename: outputFilename,
            compilationOptions: compilerArguments,
        });
    }

    async afterBuild(key: CacheKey, dirPath: string, buildResult: BuildResult): Promise<BuildResult> {
        return buildResult;
    }

    async getOrBuildExecutable(
        key: CacheKey,
        bypassCache: BypassCache,
        executablePackageHash: string,
    ): Promise<BuildResult> {
        const dirPath = await this.newTempDir();

        if (!bypassCompilationCache(bypassCache)) {
            const buildResult = await this.loadPackageWithExecutable(key, executablePackageHash, dirPath);
            if (buildResult) return buildResult;
        }

        let buildResult: BuildResult;
        try {
            buildResult = await this.buildExecutableInFolder(key, dirPath);
            if (buildResult.code !== 0) {
                return buildResult;
            }
        } catch (e) {
            return this.handleUserBuildError(e, dirPath);
        }

        buildResult.preparedLdPaths = this.getSharedLibraryPathsAsLdLibraryPathsForExecution(key, dirPath);
        buildResult.defaultExecOptions = this.getDefaultExecOptions();

        await this.storePackageWithExecutable(executablePackageHash, dirPath, buildResult);

        if (!buildResult.dirPath) {
            buildResult.dirPath = dirPath;
        }

        return buildResult;
    }

    async loadPackageWithExecutable(key: CacheKey, executablePackageHash: string, dirPath: string) {
        const compilationResultFilename = 'compilation-result.json';
        try {
            const startTime = process.hrtime.bigint();
            const outputFilename = await this.env.executableGet(executablePackageHash, dirPath);
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
                    packageDownloadAndUnzipTime: utils.deltaTimeNanoToMili(startTime, endTime),
                });
            }
            logger.debug('Tried to get executable from cache, but got a cache miss');
        } catch (err) {
            logger.error('Tried to get executable from cache, but got an error:', {err});
        }
        return false;
    }

    async storePackageWithExecutable(
        executablePackageHash: string,
        dirPath: string,
        compilationResult: CompilationResult,
    ): Promise<void> {
        const compilationResultFilename = 'compilation-result.json';

        const packDir = await this.newTempDir();
        const packagedFile = path.join(packDir, 'package.tgz');
        try {
            // first remove tmpdir from executableFilename, this path will never be the same
            //  (it's kept in the original compilationResult to keep Tools from breaking that want the full path)
            // note: couldn't use structuredClone() here, not sure why not
            const clonedResult = JSON.parse(JSON.stringify(compilationResult));
            clonedResult.executableFilename = utils.maskRootdir(clonedResult.executableFilename);

            await fs.writeFile(path.join(dirPath, compilationResultFilename), JSON.stringify(clonedResult));
            await this.packager.package(dirPath, packagedFile);
            await this.env.executablePut(executablePackageHash, packagedFile);
        } catch (err) {
            logger.error('Caught an error trying to put to cache: ', {err});
        } finally {
            fs.remove(packDir);
        }
    }

    protected processExecutionResult(input: UnprocessedExecResult, inputFilename?: string): BasicExecutionResult {
        return utils.processExecutionResult(input, inputFilename);
    }

    async runExecutableRemotely(
        executablePackageHash: string,
        executeOptions: ExecutableExecutionOptions,
        execTriple: BaseExecutionTriple,
    ): Promise<BasicExecutionResult> {
        const env = new RemoteExecutionEnvironment(this.env, execTriple, executablePackageHash);
        return await env.execute(executeOptions);
    }

    async runExecutable(
        executable: string,
        executeParameters: ExecutableExecutionOptions,
        homeDir: string,
    ): Promise<BasicExecutionResult> {
        const execOptionsCopy: ExecutableExecutionOptions = JSON.parse(
            JSON.stringify(executeParameters),
        ) as ExecutableExecutionOptions;

        if (this.compiler.executionWrapper) {
            execOptionsCopy.args = [...this.compiler.executionWrapperArgs, executable, ...execOptionsCopy.args];
            executable = this.compiler.executionWrapper;
        }

        const execEnv: IExecutionEnvironment = new this.executionEnvironmentClass(this.env);
        return execEnv.execBinary(executable, execOptionsCopy, homeDir);
    }

    protected fixExecuteParametersForInterpreting(
        executeParameters: ExecutableExecutionOptions,
        outputFilename: string,
    ) {
        (executeParameters.args as string[]).unshift(outputFilename);
    }

    async handleInterpreting(key: CacheKey, executeParameters: ExecutableExecutionOptions): Promise<CompilationResult> {
        const source = key.source;
        const dirPath = await this.newTempDir();
        const outputFilename = this.getExecutableFilename(dirPath, this.outputFilebase);

        // cant use this.writeAllFiles here because outputFilename is used as the file to execute
        //  instead of inputFilename

        await fs.writeFile(outputFilename, source);
        if (key.files && key.files.length > 0) {
            await this.writeMultipleFiles(key.files, dirPath);
        }

        this.fixExecuteParametersForInterpreting(executeParameters, outputFilename);

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

    async doExecution(
        key: CacheKey,
        executeParameters: ExecutableExecutionOptions,
        bypassCache: BypassCache,
    ): Promise<CompilationResult> {
        if (this.compiler.interpreted) {
            return this.handleInterpreting(key, executeParameters);
        }

        const executablePackageHash = this.env.getExecutableHash(key);

        const buildResult = await this.getOrBuildExecutable(key, bypassCache, executablePackageHash);
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

        if (buildResult.preparedLdPaths) {
            executeParameters.ldPath = buildResult.preparedLdPaths;
        } else {
            executeParameters.ldPath = this.getSharedLibraryPathsAsLdLibraryPathsForExecution(
                key,
                buildResult.dirPath || '',
            );
        }

        const execTriple = await RemoteExecutionQuery.guessExecutionTripleForBuildresult(buildResult);
        if (!matchesCurrentHost(execTriple)) {
            if (await RemoteExecutionQuery.isPossible(execTriple)) {
                const result = await this.runExecutableRemotely(executablePackageHash, executeParameters, execTriple);
                return moveArtifactsIntoResult(buildResult, {
                    ...result,
                    didExecute: true,
                    buildResult: buildResult,
                });
            } else {
                return {
                    code: -1,
                    didExecute: false,
                    buildResult,
                    stderr: [{text: `No execution available for ${execTriple.toString()}`}],
                    stdout: [],
                    execTime: 0,
                    timedOut: false,
                };
            }
        }

        const result = await this.runExecutable(
            buildResult.executableFilename,
            executeParameters,
            unwrap<string>(buildResult.dirPath),
        );
        return moveArtifactsIntoResult(buildResult, {
            ...result,
            didExecute: true,
            buildResult: buildResult,
        });
    }

    async handleExecution(
        key: CacheKey,
        executeParameters: ExecutableExecutionOptions,
        bypassCache: BypassCache,
    ): Promise<CompilationResult> {
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

    getCacheKey(
        source: string,
        options: string[],
        backendOptions: Record<string, any>,
        filters: ParseFiltersAndOutputOptions,
        tools: ActiveTool[],
        libraries: SelectedLibraryVersion[],
        files: FiledataPair[],
    ): CacheKey {
        return {compiler: this.compiler, source, options, backendOptions, filters, tools, libraries, files};
    }

    getCmakeCacheKey(key: ParsedRequest, files: FiledataPair[]): CmakeCacheKey {
        const cacheKey: CmakeCacheKey = {
            source: key.source,
            options: key.options,
            backendOptions: key.backendOptions,
            filters: key.filters,
            libraries: key.libraries,

            compiler: this.compiler,
            files: files,
            api: 'cmake',
        };

        if (cacheKey.filters) delete cacheKey.filters.execute;
        return cacheKey;
    }

    getCompilationInfo(key: CacheKey, result: CompilationResult, customBuildPath?: string): CompilationInfo {
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
        } as any as CompilationInfo;
    }

    getCompilationInfoForTool(
        key: CacheKey,
        inputFilename: string,
        dirPath: string,
        outputFilename: string,
    ): CompilationInfo {
        return {
            executableFilename: this.getExecutableFilename(dirPath, this.outputFilebase, key),
            asmParser: this.asm,
            outputFilename: outputFilename,
            ...key,
            inputFilename: inputFilename,
            dirPath: dirPath,
        } as any as CompilationInfo;
    }

    tryAutodetectLibraries(libsAndOptions: LibsAndOptions): boolean {
        const linkFlag = this.compiler.linkFlag || '-l';

        const detectedLibs: SelectedLibraryVersion[] = [];
        const foundlibOptions: string[] = [];
        _.each(libsAndOptions.options, option => {
            if (option.indexOf(linkFlag) === 0) {
                const libVersion = this.findAutodetectStaticLibLink(option.substring(linkFlag.length).trim());
                if (libVersion) {
                    foundlibOptions.push(option);
                    detectedLibs.push(libVersion);
                }
            }
        });

        if (detectedLibs.length > 0) {
            libsAndOptions.options = libsAndOptions.options.filter(option => !foundlibOptions.includes(option));
            libsAndOptions.libraries = _.union(libsAndOptions.libraries, detectedLibs);

            return true;
        } else {
            return false;
        }
    }

    sanitizeCompilerOverrides(overrides: ConfiguredOverrides): ConfiguredOverrides {
        const allowedRegex = /^[A-Z_]+[\dA-Z_]*$/;
        for (const override of overrides) {
            if (override.name === CompilerOverrideType.env && override.values) {
                // lowercase names are allowed, but let's assume everyone means to use uppercase
                for (const env of override.values) env.name = env.name.trim().toUpperCase();
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

    async doCompilation(
        inputFilename: string,
        dirPath: string,
        key: CacheKey,
        options: string[],
        filters: ParseFiltersAndOutputOptions,
        backendOptions: Record<string, any>,
        libraries: SelectedLibraryVersion[],
        tools: ActiveTool[],
    ): Promise<[any, OptRemark[], StackUsage.StackUsageInfo[]]> {
        const inputFilenameSafe = this.filename(inputFilename);

        const outputFilename = this.getOutputFilename(dirPath, this.outputFilebase, key);

        const overrides = this.sanitizeCompilerOverrides(backendOptions.overrides || []);

        const downloads = await this.setupBuildEnvironment(key, dirPath, !!filters.binary || !!filters.binaryObject);

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
        execOptions.ldPath = this.getSharedLibraryPathsAsLdLibraryPaths([], dirPath);

        this.applyOverridesToExecOptions(execOptions, overrides);

        const makeAst = backendOptions.produceAst && this.compiler.supportsAstView;
        const makePp = backendOptions.producePp && this.compiler.supportsPpView;
        const makeGnatDebug = backendOptions.produceGnatDebug && this.compiler.supportsGnatDebugViews;
        const makeGnatDebugTree = backendOptions.produceGnatDebugTree && this.compiler.supportsGnatDebugViews;
        const makeIr = backendOptions.produceIr && this.compiler.supportsIrView;
        const makeClangir = backendOptions.produceClangir && this.compiler.supportsClangirView;
        const makeOptPipeline = backendOptions.produceOptPipeline && this.compiler.optPipeline;
        const makeRustMir = backendOptions.produceRustMir && this.compiler.supportsRustMirView;
        const makeRustMacroExp = backendOptions.produceRustMacroExp && this.compiler.supportsRustMacroExpView;
        const makeRustHir = backendOptions.produceRustHir && this.compiler.supportsRustHirView;
        const makeHaskellCore = backendOptions.produceHaskellCore && this.compiler.supportsHaskellCoreView;
        const makeHaskellStg = backendOptions.produceHaskellStg && this.compiler.supportsHaskellStgView;
        const makeHaskellCmm = backendOptions.produceHaskellCmm && this.compiler.supportsHaskellCmmView;
        const makeGccDump =
            backendOptions.produceGccDump && backendOptions.produceGccDump.opened && this.compiler.supportsGccDump;

        const [
            asmResult,
            astResult,
            ppResult,
            irResult,
            clangirResult,
            optPipelineResult,
            rustHirResult,
            rustMacroExpResult,
            toolsResult,
        ] = await Promise.all([
            this.runCompiler(this.compiler.exe, options, inputFilenameSafe, execOptions, filters),
            makeAst ? this.generateAST(inputFilename, options) : undefined,
            makePp ? this.generatePP(inputFilename, options, backendOptions.producePp) : undefined,
            makeIr
                ? this.generateIR(
                      inputFilename,
                      options,
                      backendOptions.produceIr,
                      backendOptions.produceCfg && backendOptions.produceCfg.ir,
                      filters,
                  )
                : undefined,
            makeClangir ? this.generateClangir(inputFilename, options, backendOptions.produceClangir) : undefined,
            makeOptPipeline
                ? this.generateOptPipeline(inputFilename, options, filters, backendOptions.produceOptPipeline)
                : undefined,
            makeRustHir ? this.generateRustHir(inputFilename, options) : undefined,
            makeRustMacroExp ? this.generateRustMacroExpansion(inputFilename, options) : undefined,
            Promise.all(
                this.runToolsOfType(
                    tools,
                    'independent',
                    this.getCompilationInfoForTool(key, inputFilename, dirPath, outputFilename),
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
                  !!this.compiler.removeEmptyGccDump,
                  outputFilename,
              )
            : null;
        const rustMirResult = makeRustMir ? await this.processRustMirOutput(outputFilename, asmResult) : undefined;

        const haskellCoreResult = makeHaskellCore
            ? await this.processHaskellExtraOutput(this.getHaskellCoreOutputFilename(inputFilename), asmResult)
            : undefined;
        const haskellStgResult = makeHaskellStg
            ? await this.processHaskellExtraOutput(this.getHaskellStgOutputFilename(inputFilename), asmResult)
            : undefined;
        const haskellCmmResult = makeHaskellCmm
            ? await this.processHaskellExtraOutput(this.getHaskellCmmOutputFilename(inputFilename), asmResult)
            : undefined;

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
                asmResult.gnatDebugOutput = gnatDebugResults.expandedcode;
            }
            if (makeGnatDebugTree && gnatDebugResults.tree.length > 0) {
                asmResult.gnatDebugTreeOutput = gnatDebugResults.tree;
            }
        }

        asmResult.tools = toolsResult;
        if (this.compiler.supportsOptOutput && backendOptions.produceOptInfo) {
            const optPath = path.join(dirPath, `${this.outputFilebase}.opt.yaml`);
            if (await fs.pathExists(optPath)) {
                asmResult.optPath = optPath;
            }
        }
        if (this.compiler.supportsStackUsageOutput && backendOptions.produceStackUsageInfo) {
            const suPath = path.join(dirPath, `${this.outputFilebase}.su`);
            if (await fs.pathExists(suPath)) {
                asmResult.stackUsagePath = suPath;
            }
        }

        asmResult.astOutput = astResult;

        asmResult.ppOutput = ppResult;

        asmResult.irOutput = irResult;
        asmResult.clangirOutput = clangirResult;
        asmResult.optPipelineOutput = optPipelineResult;

        asmResult.rustMirOutput = rustMirResult;
        asmResult.rustMacroExpOutput = rustMacroExpResult;
        asmResult.rustHirOutput = rustHirResult;

        asmResult.haskellCoreOutput = haskellCoreResult;
        asmResult.haskellStgOutput = haskellStgResult;
        asmResult.haskellCmmOutput = haskellCmmResult;

        if (asmResult.code !== 0) {
            return [{...asmResult, asm: '<Compilation failed>'}, [], []];
        }

        return this.checkOutputFileAndDoPostProcess(asmResult, outputFilename, filters);
    }

    doTempfolderCleanup(buildResult: BuildResult) {
        if (buildResult.dirPath && !this.delayCleanupTemp) {
            fs.remove(buildResult.dirPath);
        }
        buildResult.dirPath = undefined;
    }

    getCompilerEnvironmentVariables(compilerflags: string) {
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

    async doBuildstep(command: string, args: string[], execParams: ExecutionOptions) {
        const result = await this.exec(command, args, execParams);
        return this.processExecutionResult(result);
    }

    handleUserError(error: any, dirPath: string): CompilationResult {
        return {
            dirPath,
            okToCache: false,
            code: -1,
            timedOut: false,
            asm: [{text: `<${error.message}>`}],
            stdout: [],
            stderr: [{text: `<${error.message}>`}],
        };
    }

    handleUserBuildError(error: any, dirPath: string): BuildResult {
        return {
            dirPath,
            okToCache: false,
            code: -1,
            timedOut: false,
            asm: [{text: `<${error.message}>`}],
            stdout: [],
            stderr: [{text: `<${error.message}>`}],
            downloads: [],
            executableFilename: '',
            compilationOptions: [],
        };
    }

    async doBuildstepAndAddToResult(
        result: CompilationResult,
        name: string,
        command: string,
        args: string[],
        execParams: ExecutionOptions,
    ): Promise<BuildStep> {
        const stepResult: BuildStep = {
            ...(await this.doBuildstep(command, args, execParams)),
            compilationOptions: args,
            step: name,
        };
        logger.debug(name);
        assert(result.buildsteps);
        result.buildsteps.push(stepResult);
        return stepResult;
    }

    createCmakeExecParams(
        execParams: ExecutionOptionsWithEnv,
        dirPath: string,
        libsAndOptions: LibsAndOptions,
        toolchainPath: string,
    ): ExecutionOptionsWithEnv {
        const cmakeExecParams = Object.assign({}, execParams);

        const libIncludes = this.getIncludeArguments(libsAndOptions.libraries, dirPath);

        const options: string[] = [];
        if (this.compiler.options) {
            const compilerOptions = splitArguments(this.compiler.options);
            options.push(...removeToolchainArg(compilerOptions));
        }
        options.push(...libsAndOptions.options, ...libIncludes);

        _.extend(cmakeExecParams.env, this.getCompilerEnvironmentVariables(options.join(' ')));

        cmakeExecParams.ldPath = [dirPath];

        const libPaths = this.getSharedLibraryPathsAsArguments(
            libsAndOptions.libraries,
            dirPath,
            toolchainPath,
            dirPath,
        );
        cmakeExecParams.env.LDFLAGS = libPaths.join(' ');

        return cmakeExecParams;
    }

    createLibsAndOptions(key: ParsedRequest): LibsAndOptions {
        const libsAndOptions = {libraries: key.libraries, options: key.options};
        if (this.tryAutodetectLibraries(libsAndOptions)) {
            key.libraries = libsAndOptions.libraries;
            key.options = libsAndOptions.options;
        }
        return libsAndOptions;
    }

    getExtraCMakeArgs(key: ParsedRequest): string[] {
        return [];
    }

    getCMakeExtToolchainParam(overrides: ConfiguredOverrides): string {
        const toolchainPath = this.getDefaultOrOverridenToolchainPath(overrides);
        if (toolchainPath) {
            return `-DCMAKE_CXX_COMPILER_EXTERNAL_TOOLCHAIN=${toolchainPath}`;
        }

        return '';
    }

    getUsedEnvironmentVariableFlags(makeExecParams: ExecutionOptionsWithEnv) {
        if (this.lang.id === 'c++') {
            return splitArguments(makeExecParams.env.CXXFLAGS);
        } else if (this.lang.id === 'fortran') {
            return splitArguments(makeExecParams.env.FFLAGS);
        } else if (this.lang.id === 'cuda') {
            return splitArguments(makeExecParams.env.CUDAFLAGS);
        } else {
            return splitArguments(makeExecParams.env.CFLAGS);
        }
    }

    async cmake(
        files: FiledataPair[],
        parsedRequest: ParsedRequest,
        bypassCache: BypassCache,
    ): Promise<CompilationResult> {
        // key = {source, options, backendOptions, filters, bypassCache, tools, executeParameters, libraries};

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

        _.defaults(parsedRequest.filters, this.getDefaultFilters());
        parsedRequest.filters.binary = true;
        parsedRequest.filters.dontMaskFilenames = true;

        const libsAndOptions = this.createLibsAndOptions(parsedRequest);

        const toolchainPath = this.getDefaultOrOverridenToolchainPath(parsedRequest.backendOptions.overrides || []);

        const dirPath = await this.newTempDir();

        const doExecute = parsedRequest.filters.execute;

        // todo: executeOptions.env should be set??
        const executeOptions: ExecutableExecutionOptions = {
            args: parsedRequest.executeParameters.args || [],
            stdin: parsedRequest.executeParameters.stdin || '',
            ldPath: this.getSharedLibraryPathsAsLdLibraryPaths(parsedRequest.libraries, dirPath),
            runtimeTools: parsedRequest.executeParameters?.runtimeTools || [],
            env: {},
        };

        const cacheKey = this.getCmakeCacheKey(parsedRequest, files);
        const executablePackageHash = this.env.getExecutableHash(cacheKey);

        const outputFilename = this.getExecutableFilename(path.join(dirPath, 'build'), this.outputFilebase, cacheKey);

        let fullResult: CompilationResult = bypassExecutionCache(bypassCache)
            ? null
            : await this.loadPackageWithExecutable(cacheKey, executablePackageHash, dirPath);
        if (fullResult) {
            fullResult.retreivedFromCache = true;

            delete fullResult.inputFilename;
            delete fullResult.dirPath;
            fullResult.executableFilename = outputFilename;
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
                code: 0,
                timedOut: false,
                stdout: [],
                stderr: [],
                buildsteps: [],
                inputFilename: writeSummary.inputFilename,
                executableFilename: outputFilename,
            };

            fullResult.downloads = await this.setupBuildEnvironment(cacheKey, dirPath, true);

            const toolchainparam = this.getCMakeExtToolchainParam(parsedRequest.backendOptions.overrides || []);

            const cmakeArgs = splitArguments(parsedRequest.backendOptions.cmakeArgs);
            const partArgs: string[] = [
                toolchainparam,
                ...this.getExtraCMakeArgs(parsedRequest),
                ...cmakeArgs,
                '..',
            ].filter(Boolean); // filter out empty args
            const useNinja = this.env.ceProps('useninja');
            const fullArgs: string[] = useNinja ? ['-GNinja'].concat(partArgs) : partArgs;

            const cmd = this.env.ceProps('cmake') as string;
            assert(cmd, 'No cmake command found');

            const cmakeStepResult = await this.doBuildstepAndAddToResult(
                fullResult,
                'cmake',
                cmd,
                fullArgs,
                makeExecParams,
            );

            if (cmakeStepResult.code !== 0) {
                fullResult.result = {
                    dirPath,
                    timedOut: false,
                    stdout: [],
                    stderr: [],
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
                cmd,
                ['--build', '.'],
                execParams,
            );

            if (makeStepResult.code !== 0) {
                fullResult.result = {
                    dirPath,
                    timedOut: false,
                    stdout: [],
                    stderr: [],
                    okToCache: false,
                    code: makeStepResult.code,
                    asm: [{text: '<Build failed>'}],
                };
                return fullResult;
            }

            fullResult.result = {
                dirPath,
                code: 0,
                timedOut: false,
                stdout: [],
                stderr: [],
                okToCache: true,
                compilationOptions: this.getUsedEnvironmentVariableFlags(makeExecParams),
            };

            if (!parsedRequest.backendOptions.skipAsm) {
                const [asmResult] = await this.checkOutputFileAndDoPostProcess(
                    fullResult.result,
                    outputFilename,
                    cacheKey.filters,
                );
                fullResult.result = asmResult;
            }

            fullResult.code = 0;
            if (fullResult.buildsteps) {
                _.each(fullResult.buildsteps, function (step) {
                    fullResult.code += step.code;
                });
            }

            await this.storePackageWithExecutable(executablePackageHash, dirPath, fullResult);
        }

        if (fullResult.result) {
            fullResult.result.dirPath = dirPath;

            if (doExecute && fullResult.result.code === 0) {
                const execTriple = await RemoteExecutionQuery.guessExecutionTripleForBuildresult({
                    ...fullResult,
                    downloads: fullResult.downloads || [],
                    executableFilename: outputFilename,
                    compilationOptions: fullResult.compilationOptions || [],
                });

                if (matchesCurrentHost(execTriple)) {
                    fullResult.execResult = await this.runExecutable(outputFilename, executeOptions, dirPath);
                    fullResult.didExecute = true;
                } else {
                    if (await RemoteExecutionQuery.isPossible(execTriple)) {
                        fullResult.execResult = await this.runExecutableRemotely(
                            executablePackageHash,
                            executeOptions,
                            execTriple,
                        );
                        fullResult.didExecute = true;
                    } else {
                        fullResult.execResult = {
                            code: -1,
                            okToCache: false,
                            stdout: [],
                            stderr: [{text: `No execution available for ${execTriple.toString()}`}],
                            execTime: 0,
                            timedOut: false,
                        };
                    }
                }
            }
        }

        const optOutput = undefined;
        const stackUsageOutput = undefined;
        await this.afterCompilation(
            fullResult.result,
            false,
            cacheKey,
            executeOptions,
            parsedRequest.tools,
            cacheKey.backendOptions,
            cacheKey.filters,
            libsAndOptions.options,
            optOutput,
            stackUsageOutput,
            bypassCache,
            path.join(dirPath, 'build'),
        );

        if (fullResult.result) delete fullResult.result.dirPath;

        this.cleanupResult(fullResult);

        return fullResult;
    }

    protected getExtraFilepath(dirPath: string, filename: string) {
        // note: it's vitally important that the resulting path does not escape dirPath
        //       (filename is user input and thus unsafe)

        const joined = path.join(dirPath, filename);
        const normalized = path.normalize(joined);
        if (process.platform === 'win32') {
            if (!normalized.replaceAll('\\', '/').startsWith(dirPath.replaceAll('\\', '/'))) {
                throw new Error('Invalid filename');
            }
        } else {
            if (!normalized.startsWith(dirPath)) throw new Error('Invalid filename');
        }
        return normalized;
    }

    fixFiltersBeforeCacheKey(filters: ParseFiltersAndOutputOptions, options: string[], files: FiledataPair[]) {
        // Don't run binary for unsupported compilers, even if we're asked.
        if (filters.binary && !this.compiler.supportsBinary) {
            delete filters.binary;
        }

        // For C/C++ turn off filters when compiling with -E
        // This is done for all compilers. Not every compiler handles -E the same but they all use
        // it for preprocessor output.
        if ((this.compiler.lang === 'c++' || this.compiler.lang === 'c') && options.includes('-E')) {
            for (const key in filters) {
                (filters as any)[key] = false; // `any` cast is needed because filters can contain non-boolean fields
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
        source: string,
        options: string[],
        backendOptions: Record<string, any>,
        filters: ParseFiltersAndOutputOptions,
        bypassCache: BypassCache,
        tools: ActiveTool[],
        executeParameters: ExecutionParams,
        libraries: SelectedLibraryVersion[],
        files: FiledataPair[],
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

        const executeOptions: ExecutableExecutionOptions = {
            args: executeParameters.args || [],
            stdin: executeParameters.stdin || '',
            ldPath: [],
            env: {},
            runtimeTools: executeParameters.runtimeTools || [],
        };

        const key = this.getCacheKey(source, options, backendOptions, filters, tools, libraries, files);

        const doExecute = filters.execute;
        filters = Object.assign({}, filters);
        filters.execute = false;

        if (!bypassCompilationCache(bypassCache)) {
            const cacheRetrieveTimeStart = process.hrtime.bigint();
            // TODO: We should be able to eliminate this any cast. `key` should be cacheable (if it's not that's a big
            // problem) Because key contains a CompilerInfo which contains a function member it can't be assigned to a
            // CacheableValue.
            const result = await this.env.cacheGet(key as any);
            if (result) {
                const cacheRetrieveTimeEnd = process.hrtime.bigint();
                result.retreivedFromCacheTime = utils.deltaTimeNanoToMili(cacheRetrieveTimeStart, cacheRetrieveTimeEnd);
                result.retreivedFromCache = true;
                if (doExecute) {
                    const queueTime = performance.now();
                    result.execResult = await this.env.enqueue(
                        async () => {
                            const start = performance.now();
                            executionQueueTimeHistogram.observe((start - queueTime) / 1000);
                            const res = await this.handleExecution(key, executeOptions, bypassCache);
                            executionTimeHistogram.observe((performance.now() - start) / 1000);
                            return res;
                        },
                        {abandonIfStale: true},
                    );

                    if (result.execResult && result.execResult.buildResult) {
                        this.doTempfolderCleanup(result.execResult.buildResult);
                    }
                }
                return result;
            }
        }
        const queueTime = performance.now();
        return this.env.enqueue(
            async () => {
                const start = performance.now();
                compilationQueueTimeHistogram.observe((start - queueTime) / 1000);
                const res = await (async () => {
                    source = this.preProcess(source, filters);

                    if (backendOptions.executorRequest) {
                        const execResult = await this.handleExecution(key, executeOptions, bypassCache);
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
                        doExecute!,
                        key,
                        executeOptions,
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
            },
            {abandonIfStale: true},
        );
    }

    async afterCompilation(
        result,
        doExecute: boolean,
        key: CacheKey,
        executeOptions: ExecutableExecutionOptions,
        tools: ActiveTool[],
        backendOptions: Record<string, any>,
        filters: ParseFiltersAndOutputOptions,
        options: string[],
        optOutput: OptRemark[] | undefined,
        stackUsageOutput: StackUsage.StackUsageInfo[] | undefined,
        bypassCache: BypassCache,
        customBuildPath?: string,
    ) {
        // Start the execution as soon as we can, but only await it at the end.
        const execPromise =
            doExecute && result.code === 0 ? this.handleExecution(key, executeOptions, bypassCache) : null;

        if (result.optPath) {
            delete result.optPath;
        }
        result.optOutput = optOutput;

        if (result.stackUsagePath) {
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
                    if (res.languageId) result.languageId = res.languageId;
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
                const isLlvmIr =
                    this.compiler.instructionSet === 'llvm' ||
                    (options && isOutputLikelyLllvmIr(options)) ||
                    this.llvmIr.isLlvmIr(result.asm);
                result.cfg = cfg.generateStructure(this.compiler, result.asm, isLlvmIr);
            }
        }

        if (!backendOptions.skipPopArgs) result.popularArguments = this.possibleArguments.getPopularArguments(options);

        result = this.postCompilationPreCacheHook(result);

        if (this.compiler.license?.invasive) {
            result.asm = [
                {text: `# License: ${this.compiler.license.name}`},
                {text: `# ${this.compiler.license.preamble}`},
                {text: `# See ${this.compiler.license.link}`},
                ...result.asm,
            ];
        }

        if (result.okToCache) {
            await this.env.cachePut(key, result, undefined);
        }

        if (doExecute && result.code === 0) {
            result.execResult = await execPromise;

            if (result.execResult.buildResult) {
                this.doTempfolderCleanup(result.execResult.buildResult);
            }
        }

        this.cleanupResult(result);

        return result;
    }

    cleanupResult(result: CompilationResult) {
        if (result.compilationOptions) {
            result.compilationOptions = this.maskPathsInArgumentsForUser(result.compilationOptions);
        }

        if (result.inputFilename) {
            result.inputFilename = utils.maskRootdir(result.inputFilename);
        }
    }

    postCompilationPreCacheHook(result: CompilationResult): CompilationResult {
        return result;
    }

    async processAsm(result, filters: ParseFiltersAndOutputOptions, options: string[]) {
        if ((options && isOutputLikelyLllvmIr(options)) || this.llvmIr.isLlvmIr(result.asm)) {
            return await this.llvmIr.processFromFilters(result.asm, filters);
        }

        return this.asm.process(result.asm, filters);
    }

    async postProcessAsm(result, filters?: ParseFiltersAndOutputOptions) {
        if (!result.okToCache || !this.demanglerClass || !result.asm) return result;
        const demangler = new this.demanglerClass(this.compiler.demangler, this, this.optionsForDemangler(filters));

        return await demangler.process(result);
    }

    processGccOptInfo(stderr: ResultLine[], compileFileName: string): {remarks: OptRemark[]; newStdErr: ResultLine[]} {
        const remarks: OptRemark[] = [];
        const nonRemarkStderr: ResultLine[] = [];

        // example stderr lines:
        // <source>:3:20: optimized: loop vectorized using 8 byte vectors
        // <source>: 2: 6: note: vectorized 1 loops in function.
        // <source>:11:13: missed: statement clobbers memory: somefunc (&i);
        const remarkRegex = /^(.*?):\s*(\d+):\s*(\d+): (.*?): (.*)$/;

        const mapOptType = (type: string, line: ResultLine): 'Missed' | 'Passed' | 'Analysis' => {
            if (type === 'missed') return 'Missed';
            if (type === 'optimized') return 'Passed';
            if (type === 'note') return 'Analysis';

            // Did we miss any types?
            SentryCapture(line, `Unexpected opt type: ${type}`);
            return 'Analysis';
        };

        for (const line of stderr) {
            const match = line.text.match(remarkRegex);
            if (match) {
                const [_, file, lineNum, colNum, type, message] = match;
                if (!file || (file !== '<source>' && !file.includes(compileFileName))) continue;
                if (type === 'warning' || type === 'error') {
                    nonRemarkStderr.push(line);
                    continue;
                }

                // convert to llvm-emitted OptRemark format, just because it was here first
                remarks.push({
                    DebugLoc: {
                        File: file,
                        // Could use line.tag for these too:
                        Line: parseInt(lineNum, 10),
                        Column: parseInt(colNum, 10),
                    },
                    optType: mapOptType(type, line),
                    displayString: message,
                    // TODO: make these optional?
                    Function: '',
                    Pass: '',
                    Name: '',
                    Args: [],
                });
            } else {
                nonRemarkStderr.push(line);
            }
        }
        // We omit remark lines from stderr to avoid it causing red squigglies in the source pane
        return {remarks: remarks, newStdErr: nonRemarkStderr};
    }

    async processOptOutput(compilationRes: CompilationResult) {
        // The distinction between clang and gcc opt remarks is a bit ad-hoc. A cleaner way might have been
        // to override processOptOutput in ClangCompiler and GCCCompiler, but that would have required having
        // all llvm-based compilers inherit ClangCompiler and all gcc-based ones inherit GCCCompiler.
        // Might be a good idea to refactor this some day.

        let remarks: OptRemark[] = [];
        if (this.compiler.optArg && this.compiler.optArg === '-fopt-info-all') {
            // gcc-like
            ({remarks, newStdErr: compilationRes.stderr} = this.processGccOptInfo(
                compilationRes.stderr,
                this.compileFilename,
            ));
        } else if (compilationRes.optPath) {
            // clang-like
            const optRemarksRaw = await fs.readFile(compilationRes.optPath, 'utf8');
            remarks = processRawOptRemarks(optRemarksRaw, this.compileFilename);
        }

        if (remarks.length > 0 && this.compiler.demangler) {
            const result = JSON.stringify(remarks, null, 4);
            const demangleResult: UnprocessedExecResult = await this.exec(
                this.compiler.demangler,
                [...this.compiler.demanglerArgs, '-n'],
                {input: result},
            );
            if (demangleResult.stdout.length > 0 && !demangleResult.truncated) {
                try {
                    return JSON.parse(demangleResult.stdout) as OptRemark[];
                } catch (exception) {
                    // swallow exception and return non-demangled output
                    logger.warn(`Caught exception ${exception} during opt demangle parsing`);
                }
            }
        }
        return remarks;
    }

    async processStackUsageOutput(suPath: string): Promise<StackUsage.StackUsageInfo[]> {
        const output = StackUsage.parse(await fs.readFile(suPath, 'utf8'));

        if (this.compiler.demangler) {
            const result = JSON.stringify(output, null, 4);
            try {
                const demangleResult = await this.exec(
                    this.compiler.demangler,
                    [...this.compiler.demanglerArgs, '-n'],
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

    isCfgCompiler() {
        return (
            this.compiler.version.includes('clang') ||
            this.compiler.version.includes('icc (ICC)') ||
            ['amd64', 'arm32', 'aarch64', 'llvm'].includes(this.compiler.instructionSet ?? '') ||
            /^([\w-]*-)?g((\+\+)|(cc)|(dc))/.test(this.compiler.version) !== null
        );
    }

    async processGccDumpOutput(
        opts: GccDumpOptions,
        result,
        removeEmptyPasses: boolean | undefined,
        outputFilename: string,
    ) {
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
            selectedPass: opts.pass ?? null,
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
    async extractDeviceCode(
        result: CompilationResult,
        _filters: ParseFiltersAndOutputOptions,
        _compilationInfo: CompilationInfo,
    ) {
        return result;
    }

    async execPostProcess(result, postProcesses: string[], outputFilename: string, maxSize: number) {
        const postCommand = `cat "${outputFilename}" | ${postProcesses.join(' | ')}`;
        return this.handlePostProcessResult(result, await this.exec('bash', ['-c', postCommand], {maxOutput: maxSize}));
    }

    preProcess(source: string, filters: CompilerOutputOptions): string {
        if (filters.binary && !this.stubRe.test(source)) {
            source += `\n${this.stubText}\n`;
        }
        return source;
    }

    async postProcess(result: CompilationResult, outputFilename: string, filters: ParseFiltersAndOutputOptions) {
        const postProcess = _.compact(this.compiler.postProcess);
        const maxSize = this.env.ceProps('max-asm-size', 64 * 1024 * 1024);
        const optPromise = this.processOptOutput(result);
        const stackUsagePromise = result.stackUsagePath
            ? this.processStackUsageOutput(result.stackUsagePath)
            : ([] as StackUsage.StackUsageInfo[]);
        const asmPromise =
            (filters.binary || filters.binaryObject) && this.supportsObjdump()
                ? this.objdump(
                      outputFilename,
                      result,
                      maxSize,
                      !!filters.intel,
                      !!filters.demangle,
                      filters.binaryObject,
                      false,
                      filters,
                  )
                : (async () => {
                      if (result.validatorTool && result.code === 0) {
                          // A validator tool is unique because if successful, there will be no asm output
                          result.asm = '<Validator was successful>';
                          return result;
                      }
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

    handlePostProcessResult(result, postResult: UnprocessedExecResult): CompilationResult {
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
    checkSource(source: string) {
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

    protected getArgumentParserClass(): typeof BaseParser {
        const exe = this.compiler.exe.toLowerCase();
        const exeFilename = path.basename(exe);
        if (exeFilename.includes('icc')) {
            return ICCParser;
        } else if (exe.includes('clangir')) {
            return ClangirParser;
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
            return {stdout: this.compiler.explicitVersion, stderr: '', code: 0};
        }
        const execOptions = this.getDefaultExecOptions();
        const versionFlag = this.compiler.versionFlag || ['--version'];
        execOptions.timeoutMs = 0; // No timeout for --version. A sort of workaround for slow EFS/NFS on the prod site
        execOptions.ldPath = this.getSharedLibraryPathsAsLdLibraryPaths([]);

        try {
            const res = await this.execCompilerCached(this.compiler.exe, versionFlag, execOptions);
            return {stdout: res.stdout, stderr: res.stderr, code: res.code};
        } catch (err) {
            logger.error(`Unable to get version for compiler '${this.compiler.exe}' - ${err}`);
            return null;
        }
    }

    initialiseLibraries(clientOptions: ClientOptionsType) {
        this.supportedLibraries = this.getSupportedLibraries(
            this.compiler.libsArr,
            clientOptions.libs[this.lang.id] || [],
        );
    }

    async getTargetsAsOverrideValues(): Promise<CompilerOverrideOption[]> {
        if (!this.buildenvsetup || !this.buildenvsetup.getCompilerArch()) {
            const parserCls = this.getArgumentParserClass();
            const targets = await parserCls.getPossibleTargets(this);

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
        const parser = this.getArgumentParserClass();
        return await parser.getPossibleStdvers(this);
    }

    async populatePossibleRuntimeTools() {
        this.compiler.possibleRuntimeTools = [];

        if (HeaptrackWrapper.isSupported(this.env)) {
            this.compiler.possibleRuntimeTools.push({
                name: RuntimeToolType.heaptrack,
                description:
                    'Heaptrack gets loaded into your code and collects the heap allocations, ' +
                    "we'll display them in a flamegraph.",
                possibleOptions: [
                    {
                        name: 'graph',
                        possibleValues: ['yes'],
                    },
                    {
                        name: 'summary',
                        possibleValues: ['stderr'],
                    },
                    {
                        name: 'details',
                        possibleValues: ['stderr'],
                    },
                ],
            });
        }

        if (LibSegFaultHelper.isSupported(this.env)) {
            this.compiler.possibleRuntimeTools.push({
                name: RuntimeToolType.libsegfault,
                description: 'libSegFault will display tracing information in case of segfaults.',
                possibleOptions: [
                    {
                        name: 'enable',
                        possibleValues: ['yes'],
                    },
                    {
                        name: 'registers',
                        possibleValues: ['yes'],
                    },
                    {
                        name: 'memory',
                        possibleValues: ['yes'],
                    },
                ],
            });
        }
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

        const compilerOptions = splitArguments(this.compiler.options);
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

    getAllPossibleTargetFlags(): string[][] {
        const all: string[][] = [];
        if (this.compiler.supportsMarch) all.push([`-march=${c_value_placeholder}`]);
        if (this.compiler.supportsTargetIs) all.push([`--target=${c_value_placeholder}`]);
        if (this.compiler.supportsTarget) all.push(['--target', c_value_placeholder]);

        return all;
    }

    async getPossibleToolchains(): Promise<CompilerOverrideOptions> {
        return this.env.getPossibleToolchains();
    }

    async initialise(
        mtime: Date,
        clientOptions: ClientOptionsType,
        isPrediscovered = false,
    ): Promise<BaseCompiler | null> {
        this.mtime = mtime;

        if (this.buildenvsetup) {
            await this.buildenvsetup.initialise(
                async (compiler: string, args: string[], options: ExecutionOptionsWithEnv) => {
                    return this.execCompilerCached(compiler, args, options);
                },
            );
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
                if (version) return null;
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
            this.compiler.supportsCfg = this.isCfgCompiler();
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
            const initResult = await this.getArgumentParserClass().parse(this);
            this.possibleArguments.possibleArguments = {};

            await this.populatePossibleOverrides();
            await this.populatePossibleRuntimeTools();

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
