// Copyright (c) 2018, Compiler Explorer Authors
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

import fs from 'fs';
import path from 'path';

import _ from 'underscore';

import {OptRemark} from '../../static/panes/opt-view.interfaces.js';
import type {
    ActiveTool,
    BuildResult,
    BypassCache,
    CacheKey,
    CompilationInfo,
    CompilationResult,
    ExecutionOptionsWithEnv,
} from '../../types/compilation/compilation.interfaces.js';
import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {ExecutableExecutionOptions, UnprocessedExecResult} from '../../types/execution/execution.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {ArtifactType} from '../../types/tool.interfaces.js';
import {addArtifactToResult} from '../artifact-utils.js';
import {BaseCompiler} from '../base-compiler.js';
import {CompilationEnvironment} from '../compilation-env.js';
import {AmdgpuAsmParser} from '../parsers/asm-parser-amdgpu.js';
import {HexagonAsmParser} from '../parsers/asm-parser-hexagon.js';
import {SassAsmParser} from '../parsers/asm-parser-sass.js';
import {StackUsageInfo} from '../stack-usage-transformer.js';
import * as utils from '../utils.js';

const offloadRegexp = /^#\s+__CLANG_OFFLOAD_BUNDLE__(__START__|__END__)\s+(.*)$/gm;

export class ClangCompiler extends BaseCompiler {
    protected asanSymbolizerPath?: string;
    protected offloadBundlerPath?: string;
    protected llvmDisassemblerPath?: string;

    static get key() {
        return 'clang';
    }

    constructor(info: PreliminaryCompilerInfo, env: CompilationEnvironment) {
        // Prefer the demangler bundled with this clang version.
        // Still allows overriding from config (for bpf)
        if (!info.demangler || info.demangler.includes('llvm-cxxfilt')) {
            const demanglerPath = path.join(path.dirname(info.exe), 'llvm-cxxfilt');
            if (fs.existsSync(demanglerPath)) {
                info.demangler = demanglerPath;
            }
        }

        super(info, env);

        this.compiler.supportsDeviceAsmView = true;

        const asanSymbolizerPath = path.join(path.dirname(this.compiler.exe), 'llvm-symbolizer');
        if (fs.existsSync(asanSymbolizerPath)) {
            this.asanSymbolizerPath = asanSymbolizerPath;
        }

        const offloadBundlerPath = path.join(path.dirname(this.compiler.exe), 'clang-offload-bundler');
        if (fs.existsSync(offloadBundlerPath)) {
            this.offloadBundlerPath = offloadBundlerPath;
        }

        const llvmDisassemblerPath = path.join(path.dirname(this.compiler.exe), 'llvm-dis');
        if (fs.existsSync(llvmDisassemblerPath)) {
            this.llvmDisassemblerPath = llvmDisassemblerPath;
        } else {
            this.llvmDisassemblerPath = this.compilerProps<string | undefined>('llvmDisassembler');
        }
    }

    override isCfgCompiler() {
        return true;
    }

    async addTimeTraceToResult(result: CompilationResult, dirPath: string, outputFilename: string) {
        const timeTraceJson = utils.changeExtension(outputFilename, '.json');
        const alternativeFilename =
            outputFilename + '-' + utils.changeExtension(path.basename(result.inputFilename || 'example.cpp'), '.json');

        const mainFilepath = path.join(dirPath, timeTraceJson);
        const alternativeJsonFilepath = path.join(dirPath, alternativeFilename);

        let jsonFilepath = '';

        if (await utils.fileExists(mainFilepath)) {
            jsonFilepath = mainFilepath;
        } else if (await utils.fileExists(alternativeJsonFilepath)) {
            jsonFilepath = alternativeJsonFilepath;
        }

        if (jsonFilepath) {
            addArtifactToResult(result, jsonFilepath, ArtifactType.timetrace, 'Trace events JSON', (buffer: Buffer) => {
                return buffer.toString('utf8').startsWith('{"traceEvents":[');
            });
        }
    }

    override async afterBuild(key: CacheKey, dirPath: string, buildResult: BuildResult): Promise<BuildResult> {
        const compilationInfo = this.getCompilationInfo(key, buildResult, dirPath);

        const filename = path.basename(compilationInfo.outputFilename);
        await this.addTimeTraceToResult(buildResult, dirPath, filename);

        return super.afterBuild(key, dirPath, buildResult);
    }

    override runExecutable(executable: string, executeParameters: ExecutableExecutionOptions, homeDir: string) {
        if (this.asanSymbolizerPath) {
            executeParameters.env = {
                ASAN_SYMBOLIZER_PATH: this.asanSymbolizerPath,
                LSAN_SYMBOLIZER_PATH: this.asanSymbolizerPath,
                MSAN_SYMBOLIZER_PATH: this.asanSymbolizerPath,
                RTSAN_SYMBOLIZER_PATH: this.asanSymbolizerPath,
                TSAN_SYMBOLIZER_PATH: this.asanSymbolizerPath,
                UBSAN_SYMBOLIZER_PATH: this.asanSymbolizerPath,
                ...executeParameters.env,
            };
        }
        return super.runExecutable(executable, executeParameters, homeDir);
    }

    forceDwarf4UnlessOverridden(options: string[]) {
        const hasOverride = _.any(options, (option: string) => {
            return option.includes('-gdwarf-') || option.includes('-fdebug-default-version=');
        });

        if (!hasOverride) return ['-gdwarf-4'].concat(options);

        return options;
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string, userOptions?: string[]) {
        const options = super.optionsForFilter(filters, outputFilename);

        return this.forceDwarf4UnlessOverridden(options);
    }

    // Clang cross-compile with -stdlib=libc++ is currently (up to at least 18.1.0) broken:
    // https://github.com/llvm/llvm-project/issues/57104
    //
    // Below is a workaround discussed in CE issue #5293. If the llvm issue is ever resolved it would be best
    // to apply this only for clang versions up to the official resolution.
    // To smoke-test such future versions, check locally *without* this filterUserOptions overload whether
    // compiling `#include <string>` with flag `-stdlib=libc++` succeeds: https://godbolt.org/z/7dKrad7Wc

    override filterUserOptions(userOptions: string[]): string[] {
        if (
            this.lang.id === 'c++' &&
            !this.buildenvsetup?.compilerSupportsX86 && // cross-compilation
            _.any(userOptions, option => {
                return option === '-stdlib=libc++';
            })
        ) {
            const addedIncludePath =
                '-I' + path.join(path.dirname(this.compiler.exe), '../include/x86_64-unknown-linux-gnu/c++/v1/');
            if (
                !_.any(userOptions, option => {
                    return option === addedIncludePath;
                })
            )
                userOptions = userOptions.concat(addedIncludePath);
        }
        return userOptions;
    }

    override async afterCompilation(
        result: CompilationResult,
        doExecute: boolean,
        key: CacheKey,
        executeParameters: ExecutableExecutionOptions,
        tools: ActiveTool[],
        backendOptions: Record<string, any>,
        filters: ParseFiltersAndOutputOptions,
        options: string[],
        optOutput: OptRemark[] | undefined,
        stackUsageOutput: StackUsageInfo[] | undefined,
        bypassCache: BypassCache,
        customBuildPath?: string,
    ) {
        const compilationInfo = this.getCompilationInfo(key, result, customBuildPath);

        const dirPath = path.dirname(compilationInfo.outputFilename);
        const filename = path.basename(compilationInfo.outputFilename);
        await this.addTimeTraceToResult(result, dirPath, filename);

        return super.afterCompilation(
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
            customBuildPath,
        );
    }

    override async runCompiler(
        compiler: string,
        options: string[],
        inputFilename: string,
        execOptions: ExecutionOptionsWithEnv,
    ) {
        if (!execOptions) {
            execOptions = this.getDefaultExecOptions();
        }

        execOptions.customCwd = path.dirname(inputFilename);

        return await super.runCompiler(compiler, options, inputFilename, execOptions);
    }

    async splitDeviceCode(assembly: string) {
        // Check to see if there is any offload code in the assembly file.
        if (!offloadRegexp.test(assembly)) return null;

        offloadRegexp.lastIndex = 0;
        const matches = assembly.matchAll(offloadRegexp);
        let prevStart = 0;
        const devices: Record<string, string> = {};
        for (const match of matches) {
            const [full, startOrEnd, triple] = match;
            if (startOrEnd === '__START__') {
                prevStart = match.index + full.length + 1;
            } else {
                devices[triple] = assembly.substring(prevStart, match.index);
            }
        }
        return devices;
    }

    override async extractDeviceCode(result, filters: ParseFiltersAndOutputOptions, compilationInfo: CompilationInfo) {
        const split = await this.splitDeviceCode(result.asm);
        if (!split) return result;

        result.devices = {};
        for (const key of Object.keys(split)) {
            if (key.indexOf('host-') === 0) result.asm = split[key];
            else result.devices[key] = await this.processDeviceAssembly(key, split[key], filters, compilationInfo);
        }
        return result;
    }

    async extractBitcodeFromBundle(bundlefile: string, devicename: string): Promise<string> {
        const bcfile = path.join(path.dirname(bundlefile), devicename + '.bc');

        const env = this.getDefaultExecOptions();
        env.customCwd = path.dirname(bundlefile);

        if (this.offloadBundlerPath) {
            const unbundleResult: UnprocessedExecResult = await this.exec(
                this.offloadBundlerPath,
                ['-unbundle', '--type', 's', '--inputs', bundlefile, '--outputs', bcfile, '--targets', devicename],
                env,
            );
            if (unbundleResult.code !== 0) {
                return unbundleResult.stderr;
            }
        } else {
            return '<error: no offload bundler found to unbundle device code>';
        }

        if (this.llvmDisassemblerPath) {
            const llvmirFile = path.join(path.dirname(bundlefile), devicename + '.ll');

            const disassembleResult: UnprocessedExecResult = await this.exec(this.llvmDisassemblerPath, [bcfile], env);
            if (disassembleResult.code !== 0) {
                return disassembleResult.stderr;
            }

            return fs.readFileSync(llvmirFile, 'utf8');
        } else {
            return '<error: no llvm-dis found to disassemble bitcode>';
        }
    }

    async processDeviceAssembly(deviceName: string, deviceAsm: string, filters, compilationInfo: CompilationInfo) {
        if (deviceAsm.startsWith('BC')) {
            deviceAsm = await this.extractBitcodeFromBundle(compilationInfo.outputFilename, deviceName);
        }

        return this.llvmIr.isLlvmIr(deviceAsm)
            ? this.llvmIr.process(deviceAsm, filters)
            : this.asm.process(deviceAsm, filters);
    }
}

export class ClangCudaCompiler extends ClangCompiler {
    static override get key() {
        return 'clang-cuda';
    }

    constructor(info: PreliminaryCompilerInfo, env: CompilationEnvironment) {
        super(info, env);

        this.asm = new SassAsmParser();
    }

    override getCompilerResultLanguageId(filters?: ParseFiltersAndOutputOptions): string | undefined {
        return 'ptx';
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string) {
        return ['-o', this.filename(outputFilename), '-g1', filters.binary ? '-c' : '-S'];
    }

    override async objdump(outputFilename: string, result, maxSize: number) {
        // For nvdisasm.
        const args = [...this.compiler.objdumperArgs, outputFilename, '-c', '-g', '-hex'];
        const execOptions = {maxOutput: maxSize, customCwd: path.dirname(outputFilename)};

        const objResult = await this.exec(this.compiler.objdumper, args, execOptions);
        result.asm = objResult.stdout;
        if (objResult.code === 0) {
            result.objdumpTime = objResult.execTime;
        } else {
            result.asm = `<No output: nvdisasm returned ${objResult.code}>`;
        }
        return result;
    }
}

export class ClangHipCompiler extends ClangCompiler {
    static override get key() {
        return 'clang-hip';
    }

    constructor(info: PreliminaryCompilerInfo, env: CompilationEnvironment) {
        super(info, env);

        this.asm = new AmdgpuAsmParser();
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string) {
        return ['-o', this.filename(outputFilename), '-g1', '--no-gpu-bundle-output', filters.binary ? '-c' : '-S'];
    }
}

export class ClangIntelCompiler extends ClangCompiler {
    static override get key() {
        return 'clang-intel';
    }

    constructor(info: PreliminaryCompilerInfo, env: CompilationEnvironment) {
        super(info, env);

        if (!this.offloadBundlerPath) {
            // clang-offload-bundler is in a different folder in versions >= 2024.0.0
            const offloadBundlerPath = path.join(path.dirname(this.compiler.exe), '../bin-llvm/clang-offload-bundler');
            if (fs.existsSync(offloadBundlerPath)) {
                this.offloadBundlerPath = path.resolve(offloadBundlerPath);
            } else {
                const offloadBundlerPath = path.join(path.dirname(this.compiler.exe), 'compiler/clang-offload-bundler');
                if (fs.existsSync(offloadBundlerPath)) {
                    this.offloadBundlerPath = path.resolve(offloadBundlerPath);
                }
            }
        }
    }

    override getDefaultExecOptions(): ExecutionOptionsWithEnv {
        const opts = super.getDefaultExecOptions();
        opts.env.PATH = process.env.PATH + path.delimiter + path.dirname(this.compiler.exe);

        return opts;
    }

    override runExecutable(executable: string, executeParameters: ExecutableExecutionOptions, homeDir: string) {
        const base = path.dirname(this.compiler.exe);
        const ocl_pre2024 = path.resolve(`${base}/../lib/x64/libintelocl.so`);
        const ocl_2024 = path.resolve(`${base}/../lib/libintelocl.so`);
        executeParameters.env = {
            OCL_ICD_FILENAMES: `${ocl_2024}:${ocl_pre2024}`,
            ...executeParameters.env,
        };
        return super.runExecutable(executable, executeParameters, homeDir);
    }
}

export class ClangHexagonCompiler extends ClangCompiler {
    static override get key() {
        return 'clang-hexagon';
    }

    constructor(info: PreliminaryCompilerInfo, env: CompilationEnvironment) {
        super(info, env);

        this.asm = new HexagonAsmParser();
    }
}

export class ClangDxcCompiler extends ClangCompiler {
    static override get key() {
        return 'clang-dxc';
    }

    constructor(info: PreliminaryCompilerInfo, env: CompilationEnvironment) {
        super(info, env);

        this.compiler.supportsIntel = false;
        this.compiler.irArg = ['-Xclang', '-emit-llvm'];
        // dxc mode doesn't have -fsave-optimization-record or -fstack-usage
        this.compiler.supportsOptOutput = false;
        this.compiler.supportsStackUsageOutput = false;
    }

    override optionsForFilter(
        filters: ParseFiltersAndOutputOptions,
        outputFilename: string,
        userOptions?: string[],
    ): string[] {
        return ['--driver-mode=dxc', '-Zi', '-Qembed_debug', '-Fc', this.filename(outputFilename)];
    }
}
