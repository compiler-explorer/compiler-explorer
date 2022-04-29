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

import {BaseCompiler} from '../base-compiler';
import {AmdgpuAsmParser} from '../parsers/asm-parser-amdgpu';
import {SassAsmParser} from '../parsers/asm-parser-sass';

const offloadRegexp = /^#\s+__CLANG_OFFLOAD_BUNDLE__(__START__|__END__)\s+(.*)$/gm;

export class ClangCompiler extends BaseCompiler {
    static get key() {
        return 'clang';
    }

    constructor(info, env) {
        super(info, env);
        this.compiler.supportsDeviceAsmView = true;
        const asanSymbolizerPath = path.dirname(this.compiler.exe) + '/llvm-symbolizer';
        if (fs.existsSync(asanSymbolizerPath)) {
            this.asanSymbolizerPath = asanSymbolizerPath;
        }
    }

    runExecutable(executable, executeParameters, homeDir) {
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

    optionsForFilter(filters, outputFilename) {
        const options = super.optionsForFilter(filters, outputFilename);

        return this.forceDwarf4UnlessOverridden(options);
    }

    runCompiler(compiler, options, inputFilename, execOptions) {
        if (!execOptions) {
            execOptions = this.getDefaultExecOptions();
        }

        execOptions.customCwd = path.dirname(inputFilename);

        return super.runCompiler(compiler, options, inputFilename, execOptions);
    }

    splitDeviceCode(assembly) {
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

    extractDeviceCode(result, filters) {
        const split = this.splitDeviceCode(result.asm);
        if (!split) return result;

        const devices = (result.devices = {});
        for (const key of Object.keys(split)) {
            if (key.indexOf('host-') === 0) result.asm = split[key];
            else devices[key] = this.processDeviceAssembly(key, split[key], filters);
        }
        return result;
        // //result.asm = ...
        //
        // const extractor = this.compiler.deviceExtractor;
        // const extractor_result = await this.exec(extractor,
        //     [outputFilename, result.dirPath + '/devices'], {});
        // if (extractor_result.code !== 0) {
        //     result.devices = extractor_result.stderr;
        //     return result;
        // }
        // const extractor_info = JSON.parse(extractor_result.stdout);
        // result.asm = await fs.readFile(extractor_info.host, 'utf-8');
        // const devices = result.devices = {};
        // for (const device in extractor_info.devices) {
        //     const file = extractor_info.devices[device];
        //     devices[device] = await this.processDeviceAssembly(device, file, filters);
        // }
        // return result;
    }

    processDeviceAssembly(deviceName, deviceAsm, filters) {
        return this.llvmIr.isLlvmIr(deviceAsm)
            ? this.llvmIr.process(deviceAsm, filters)
            : this.asm.process(deviceAsm, filters);
    }
}

export class ClangCudaCompiler extends ClangCompiler {
    static get key() {
        return 'clang-cuda';
    }

    constructor(info, env) {
        super(info, env);

        this.asm = new SassAsmParser();
    }

    optionsForFilter(filters, outputFilename) {
        return ['-o', this.filename(outputFilename), '-g1', filters.binary ? '-c' : '-S'];
    }

    async objdump(outputFilename, result, maxSize) {
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
    static get key() {
        return 'clang-hip';
    }

    constructor(info, env) {
        super(info, env);

        this.asm = new AmdgpuAsmParser();
    }

    optionsForFilter(filters, outputFilename) {
        return ['-o', this.filename(outputFilename), '-g1', '--no-gpu-bundle-output', filters.binary ? '-c' : '-S'];
    }
}
