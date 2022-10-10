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

import {ExecutionOptions} from '../../types/compilation/compilation.interfaces';
import {UnprocessedExecResult} from '../../types/execution/execution.interfaces';
import {BaseCompiler} from '../base-compiler';
import {AmdgpuAsmParser} from '../parsers/asm-parser-amdgpu';
import {SassAsmParser} from '../parsers/asm-parser-sass';

const offloadRegexp = /^#\s+__CLANG_OFFLOAD_BUNDLE__(__START__|__END__)\s+(.*)$/gm;

export class ClangCompiler extends BaseCompiler {
    protected asanSymbolizerPath?: string;
    protected offloadBundlerPath?: string;
    protected llvmDisassemblerPath?: string;

    static get key() {
        return 'clang';
    }

    constructor(info, env) {
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
            this.llvmDisassemblerPath = this.compilerProps('llvmDisassembler');
        }
    }

    override runExecutable(executable, executeParameters, homeDir) {
        if (this.asanSymbolizerPath) {
            executeParameters.env = {
                ASAN_SYMBOLIZER_PATH: this.asanSymbolizerPath,
                ...executeParameters.env,
            };
        }
        return super.runExecutable(executable, executeParameters, homeDir);
    }

    forceDwarf4UnlessOverridden(options) {
        const hasOverride = _.any(options, option => {
            return option.includes('-gdwarf-') || option.includes('-fdebug-default-version=');
        });

        if (!hasOverride) return ['-gdwarf-4'].concat(options);

        return options;
    }

    override optionsForFilter(filters, outputFilename) {
        const options = super.optionsForFilter(filters, outputFilename);

        return this.forceDwarf4UnlessOverridden(options);
    }

    override runCompiler(compiler, options, inputFilename, execOptions) {
        if (!execOptions) {
            execOptions = this.getDefaultExecOptions();
        }

        execOptions.customCwd = path.dirname(inputFilename);

        return super.runCompiler(compiler, options, inputFilename, execOptions);
    }

    async splitDeviceCode(assembly) {
        // Check to see if there is any offload code in the assembly file.
        if (!offloadRegexp.test(assembly)) return null;

        offloadRegexp.lastIndex = 0;
        const matches = assembly.matchAll(offloadRegexp);
        let prevStart = 0;
        const devices = {};
        for (const match of matches) {
            const [full, startOrEnd, triple] = match;
            if (startOrEnd === '__START__') {
                prevStart = match.index + full.length + 1;
            } else {
                devices[triple] = assembly.substr(prevStart, match.index - prevStart);
            }
        }
        return devices;
    }

    override async extractDeviceCode(result, filters, compilationInfo) {
        const split = await this.splitDeviceCode(result.asm);
        if (!split) return result;

        const devices = (result.devices = {});
        for (const key of Object.keys(split)) {
            if (key.indexOf('host-') === 0) result.asm = split[key];
            else devices[key] = await this.processDeviceAssembly(key, split[key], filters, compilationInfo);
        }
        return result;
    }

    async extractBitcodeFromBundle(bundlefile, devicename): Promise<string> {
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

            return fs.readFileSync(llvmirFile, 'utf-8');
        } else {
            return '<error: no llvm-dis found to disassemble bitcode>';
        }
    }

    async processDeviceAssembly(deviceName, deviceAsm, filters, compilationInfo) {
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

    constructor(info, env) {
        super(info, env);

        this.asm = new SassAsmParser();
    }

    override getCompilerResultLanguageId() {
        return 'ptx';
    }

    override optionsForFilter(filters, outputFilename) {
        return ['-o', this.filename(outputFilename), '-g1', filters.binary ? '-c' : '-S'];
    }

    override async objdump(outputFilename, result, maxSize) {
        // For nvdisasm.
        const args = [outputFilename, '-c', '-g', '-hex'];
        const execOptions = {maxOutput: maxSize, customCwd: path.dirname(outputFilename)};

        const objResult = await this.exec(this.compiler.objdumper, args, execOptions);
        result.asm = objResult.stdout;
        if (objResult.code !== 0) {
            result.asm = `<No output: nvdisasm returned ${objResult.code}>`;
        } else {
            result.objdumpTime = objResult.execTime;
        }
        return result;
    }
}

export class ClangHipCompiler extends ClangCompiler {
    static override get key() {
        return 'clang-hip';
    }

    constructor(info, env) {
        super(info, env);

        this.asm = new AmdgpuAsmParser();
    }

    override optionsForFilter(filters, outputFilename) {
        return ['-o', this.filename(outputFilename), '-g1', '--no-gpu-bundle-output', filters.binary ? '-c' : '-S'];
    }
}

export class ClangIntelCompiler extends ClangCompiler {
    static override get key() {
        return 'clang-intel';
    }

    constructor(info, env) {
        super(info, env);

        if (!this.offloadBundlerPath) {
            const offloadBundlerPath = path.join(path.dirname(this.compiler.exe), '../bin-llvm/clang-offload-bundler');
            if (fs.existsSync(offloadBundlerPath)) {
                this.offloadBundlerPath = path.resolve(offloadBundlerPath);
            }
        }
    }

    override getDefaultExecOptions(): ExecutionOptions {
        const opts = super.getDefaultExecOptions();
        opts.env.PATH = process.env.PATH + path.delimiter + path.dirname(this.compiler.exe);

        return opts;
    }

    override runExecutable(executable, executeParameters, homeDir) {
        executeParameters.env = {
            OCL_ICD_FILENAMES: path.resolve(path.dirname(this.compiler.exe) + '/../lib/x64/libintelocl.so'),
            ...executeParameters.env,
        };
        return super.runExecutable(executable, executeParameters, homeDir);
    }
}
