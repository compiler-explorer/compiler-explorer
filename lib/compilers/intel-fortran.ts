// Copyright (c) 2024, Compiler Explorer Authors
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

import fs from 'node:fs';
import path from 'node:path';

import type {CompilationInfo, ExecutionOptionsWithEnv} from '../../types/compilation/compilation.interfaces.js';
import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {ExecutableExecutionOptions, UnprocessedExecResult} from '../../types/execution/execution.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {CompilationEnvironment} from '../compilation-env.js';
import {FortranCompiler} from './fortran.js';

// Regex to match offload bundle markers in Intel Fortran output
const offloadRegexp = /^#\s+__CLANG_OFFLOAD_BUNDLE__(__START__|__END__)\s+(.*)$/gm;

export class IntelFortranCompiler extends FortranCompiler {
    static override get key() {
        return 'intel-fortran';
    }

    protected offloadBundlerPath?: string;
    protected llvmDisassemblerPath?: string;

    constructor(info: PreliminaryCompilerInfo, env: CompilationEnvironment) {
        super(info, env);

        // Find llvm-dis for disassembling bitcode
        const llvmDisPath = this.findLlvmDisassembler();
        if (llvmDisPath) {
            this.llvmDisassemblerPath = llvmDisPath;
        }

        // Find clang-offload-bundler for extracting device code
        // Intel compilers >= 2024.0.0 have a different directory structure
        this.offloadBundlerPath = this.findOffloadBundler();
    }

    private findLlvmDisassembler(): string | undefined {
        const compilerDir = path.dirname(this.compiler.exe);

        // Try various possible locations for llvm-dis
        const possiblePaths = [
            path.join(compilerDir, 'llvm-dis'),
            path.join(compilerDir, '../bin-llvm/llvm-dis'),
            path.join(compilerDir, '../bin/llvm-dis'),
            path.join(compilerDir, 'compiler/llvm-dis'),
        ];

        for (const llvmDisPath of possiblePaths) {
            if (fs.existsSync(llvmDisPath)) {
                return path.resolve(llvmDisPath);
            }
        }

        return undefined;
    }

    private findOffloadBundler(): string | undefined {
        const compilerDir = path.dirname(this.compiler.exe);

        // Try Intel 2024+ structure first
        const newPathBundler = path.join(compilerDir, '../bin-llvm/clang-offload-bundler');
        if (fs.existsSync(newPathBundler)) {
            return path.resolve(newPathBundler);
        }

        // Try older Intel compiler structure
        const oldPathBundler = path.join(compilerDir, 'compiler/clang-offload-bundler');
        if (fs.existsSync(oldPathBundler)) {
            return path.resolve(oldPathBundler);
        }

        // Try same directory as compiler
        const sameDirBundler = path.join(compilerDir, 'clang-offload-bundler');
        if (fs.existsSync(sameDirBundler)) {
            return path.resolve(sameDirBundler);
        }

        return undefined;
    }

    override getDefaultExecOptions(): ExecutionOptionsWithEnv {
        const opts = super.getDefaultExecOptions();
        const compilerDir = path.dirname(this.compiler.exe);

        // Add compiler directory and LLVM tool directories to PATH
        const additionalPaths = [
            compilerDir,
            path.join(compilerDir, '../bin-llvm'),
            path.join(compilerDir, '../bin'),
            path.join(compilerDir, 'compiler'),
        ];

        // Find directories that exist and contain llvm-objcopy
        const pathsWithObjcopy = additionalPaths.filter(dir => {
            const objcopyPath = path.join(dir, 'llvm-objcopy');
            return fs.existsSync(dir) && fs.existsSync(objcopyPath);
        });

        // Add all potential tool directories to PATH
        const existingPaths = additionalPaths.filter(dir => fs.existsSync(dir));
        opts.env.PATH = [...pathsWithObjcopy, ...existingPaths, process.env.PATH].join(path.delimiter);

        return opts;
    }

    async splitDeviceCode(assembly: string): Promise<Record<string, string> | null> {
        // Check to see if there is any offload code in the assembly file
        if (!offloadRegexp.test(assembly)) return null;

        offloadRegexp.lastIndex = 0;
        const matches = assembly.matchAll(offloadRegexp);
        let prevStart = 0;
        const devices: Record<string, string> = {};

        for (const match of matches) {
            const [full, startOrEnd, triple] = match;
            if (startOrEnd === '__START__') {
                prevStart = (match.index ?? 0) + full.length + 1;
            } else {
                devices[triple] = assembly.substring(prevStart, match.index);
            }
        }

        return devices;
    }

    override async extractDeviceCode(
        result: any,
        filters: ParseFiltersAndOutputOptions,
        compilationInfo: CompilationInfo,
    ) {
        if (!result.asm) return result;

        const split = await this.splitDeviceCode(result.asm);
        if (!split) return result;

        result.devices = {};
        for (const key of Object.keys(split)) {
            if (key.indexOf('host-') === 0) {
                result.asm = split[key];
            } else {
                result.devices[key] = await this.processDeviceAssembly(key, split[key], filters, compilationInfo);
            }
        }
        return result;
    }

    async processDeviceAssembly(
        deviceName: string,
        deviceAsm: string,
        filters: ParseFiltersAndOutputOptions,
        compilationInfo: CompilationInfo,
    ) {
        // If the device assembly starts with 'BC', it's bitcode that needs to be extracted
        if (deviceAsm.startsWith('BC')) {
            deviceAsm = await this.extractBitcodeFromBundle(compilationInfo.outputFilename, deviceName);
        }

        // Process as LLVM IR if it looks like IR, otherwise process as assembly
        return this.llvmIr.isLlvmIr(deviceAsm)
            ? this.llvmIr.processFromFilters(deviceAsm, filters)
            : this.asm.process(deviceAsm, filters);
    }

    async extractBitcodeFromBundle(bundlefile: string, devicename: string): Promise<string> {
        if (!this.offloadBundlerPath) {
            return '<error: no offload bundler found to unbundle device code>';
        }

        const bcfile = path.join(path.dirname(bundlefile), devicename + '.bc');
        const env = this.getDefaultExecOptions();
        env.customCwd = path.dirname(bundlefile);

        // Use clang-offload-bundler to extract the bitcode
        const unbundleResult: UnprocessedExecResult = await this.exec(
            this.offloadBundlerPath,
            ['-unbundle', '--type', 's', '--inputs', bundlefile, '--outputs', bcfile, '--targets', devicename],
            env,
        );

        if (unbundleResult.code !== 0) {
            return unbundleResult.stderr;
        }

        // If we have llvm-dis, disassemble the bitcode to readable LLVM IR
        if (this.llvmDisassemblerPath) {
            const llvmirFile = path.join(path.dirname(bundlefile), devicename + '.ll');

            const disassembleResult: UnprocessedExecResult = await this.exec(
                this.llvmDisassemblerPath,
                [bcfile, '-o', llvmirFile],
                env,
            );

            if (disassembleResult.code !== 0) {
                return disassembleResult.stderr;
            }

            return await fs.promises.readFile(llvmirFile, 'utf8');
        }

        return '<error: no llvm-dis found to disassemble bitcode>';
    }

    override async runExecutable(executable: string, executeParameters: ExecutableExecutionOptions, homeDir: string) {
        // Set up Intel OpenCL runtime paths for GPU execution
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
