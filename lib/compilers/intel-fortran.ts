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

const offloadRegexp = /^#\s+__CLANG_OFFLOAD_BUNDLE__(__START__|__END__)\s+(.*)$/gm;

export class IntelFortranCompiler extends FortranCompiler {
    static override get key() {
        return 'intel-fortran';
    }

    protected offloadBundlerPath?: string;
    protected llvmDisassemblerPath?: string;

    constructor(info: PreliminaryCompilerInfo, env: CompilationEnvironment) {
        super(info, env);

        this.compiler.supportsDeviceAsmView = true;

        const llvmDisPath = this.findLlvmDisassembler();
        if (llvmDisPath) {
            this.llvmDisassemblerPath = llvmDisPath;
        }

        this.offloadBundlerPath = this.findOffloadBundler();
    }

    private findLlvmDisassembler(): string | undefined {
        const compilerDir = path.dirname(this.compiler.exe);

        const possiblePaths = [
            path.join(compilerDir, 'llvm-dis'),
            path.join(compilerDir, '../bin-llvm/llvm-dis'),
            path.join(compilerDir, '../bin/llvm-dis'),
            path.join(compilerDir, 'compiler/llvm-dis'),
            path.join(compilerDir, '../compiler/llvm-dis'),
            path.join(compilerDir, '../../bin/llvm-dis'),
            path.join(compilerDir, '../../bin-llvm/llvm-dis'),
            '/usr/bin/llvm-dis',
            '/usr/local/bin/llvm-dis',
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

        const newPathBundler = path.join(compilerDir, '../bin-llvm/clang-offload-bundler');
        if (fs.existsSync(newPathBundler)) {
            return path.resolve(newPathBundler);
        }

        const oldPathBundler = path.join(compilerDir, 'compiler/clang-offload-bundler');
        if (fs.existsSync(oldPathBundler)) {
            return path.resolve(oldPathBundler);
        }

        const sameDirBundler = path.join(compilerDir, 'clang-offload-bundler');
        if (fs.existsSync(sameDirBundler)) {
            return path.resolve(sameDirBundler);
        }

        return undefined;
    }

    override getDefaultExecOptions(): ExecutionOptionsWithEnv {
        const opts = super.getDefaultExecOptions();
        const compilerDir = path.dirname(this.compiler.exe);

        const additionalPaths = [
            compilerDir,
            path.join(compilerDir, '../bin-llvm'),
            path.join(compilerDir, '../bin'),
            path.join(compilerDir, 'compiler'),
        ];

        const pathsWithObjcopy = additionalPaths.filter(dir => {
            const objcopyPath = path.join(dir, 'llvm-objcopy');
            return fs.existsSync(dir) && fs.existsSync(objcopyPath);
        });

        const existingPaths = additionalPaths.filter(dir => fs.existsSync(dir));
        opts.env.PATH = [...pathsWithObjcopy, ...existingPaths, process.env.PATH].join(path.delimiter);

        return opts;
    }

    async splitDeviceCode(assembly: string): Promise<Record<string, string> | null> {
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
        let asmString: string;
        if (result.asm) {
            asmString = Array.isArray(result.asm)
                ? result.asm.map(line => line.text || line.toString()).join('\n')
                : result.asm.toString();
        } else {
            try {
                asmString = await fs.promises.readFile(compilationInfo.outputFilename, 'utf8');
            } catch (error) {
                return result;
            }
        }

        const split = await this.splitDeviceCode(asmString);
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
        if (deviceAsm.length <= 10) {
            deviceAsm = await this.extractBitcodeFromBundle(compilationInfo.outputFilename, deviceName);
        } else if (deviceAsm.startsWith('BC')) {
            deviceAsm = await this.extractBitcodeFromBundle(compilationInfo.outputFilename, deviceName);
        }
        const isLlvmIr = this.llvmIr.isLlvmIr(deviceAsm);

        const processed = isLlvmIr
            ? await this.llvmIr.processFromFilters(deviceAsm, filters)
            : this.asm.process(deviceAsm, filters);

        return {
            languageId: isLlvmIr ? 'llvm-ir' : 'asm',
            code: 0,
            stdout: [],
            stderr: [],
            ...processed,
        };
    }

    async extractBitcodeFromBundle(bundlefile: string, devicename: string): Promise<string> {
        if (!this.offloadBundlerPath) {
            return '<error: no offload bundler found to unbundle device code>';
        }

        const bcfile = path.join(path.dirname(bundlefile), devicename + '.bc');
        const env = this.getDefaultExecOptions();
        env.customCwd = path.dirname(bundlefile);

        const unbundleResult: UnprocessedExecResult = await this.exec(
            this.offloadBundlerPath,
            ['-unbundle', '--type', 's', '--inputs', bundlefile, '--outputs', bcfile, '--targets', devicename],
            env,
        );

        if (unbundleResult.code !== 0) {
            return `<error: clang-offload-bundler failed with code ${unbundleResult.code}: ${unbundleResult.stderr}>`;
        }

        if (!fs.existsSync(bcfile)) {
            return '<error: .bc file was not created by clang-offload-bundler>';
        }

        const llvmirFile = path.join(path.dirname(bundlefile), devicename + '.ll');

        const llvmDisPaths = this.llvmDisassemblerPath ? [this.llvmDisassemblerPath] : [];
        llvmDisPaths.push('llvm-dis');

        for (const llvmDisPath of llvmDisPaths) {
            try {
                const disassembleResult: UnprocessedExecResult = await this.exec(
                    llvmDisPath,
                    [bcfile, '-o', llvmirFile],
                    env,
                );

                if (disassembleResult.code === 0 && fs.existsSync(llvmirFile)) {
                    return await fs.promises.readFile(llvmirFile, 'utf8');
                }
            } catch (error) {
                // Continue to next llvm-dis path
            }
        }

        try {
            const bcContent = await fs.promises.readFile(bcfile);
            return `; Raw bitcode file (${bcContent.length} bytes) - llvm-dis not available\n; Device: ${devicename}\n; File: ${bcfile}`;
        } catch (error) {
            return `<error: unable to read bitcode file: ${error}>`;
        }
    }

    override async runExecutable(executable: string, executeParameters: ExecutableExecutionOptions, homeDir: string) {
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
