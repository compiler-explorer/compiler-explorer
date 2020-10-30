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

import path from 'path';

import fs from 'fs-extra';

import { BaseCompiler } from '../base-compiler';

export class ClangCompiler extends BaseCompiler {
    static get key() { return 'clang'; }

    constructor(info, env) {
        super(info, env);
        if (this.compiler.deviceExtractor)
            this.compiler.supportsDeviceAsmView = true;
    }

    runCompiler(compiler, options, inputFilename, execOptions) {
        if (!execOptions) {
            execOptions = this.getDefaultExecOptions();
        }

        execOptions.customCwd = path.dirname(inputFilename);

        return super.runCompiler(compiler, options, inputFilename, execOptions);
    }

    async extractDeviceCode(result, filters, outputFilename) {
        // Check to see if there is any offload code in the assembly file.
        const offload_regexp = /__CLANG_OFFLOAD_BUNDLE__/;
        if (!this.compiler.deviceExtractor || !offload_regexp.exec(result.asm))
            return result;

        const extractor = this.compiler.deviceExtractor;
        const extractor_result = await this.exec(extractor,
            [outputFilename, result.dirPath + '/devices'], {});
        if (extractor_result.code !== 0) {
            result.devices = extractor_result.stderr;
            return result;
        }
        const extractor_info = JSON.parse(extractor_result.stdout);
        result.asm = await fs.readFile(extractor_info.host, 'utf-8');
        const devices = result.devices = {};
        for (const device in extractor_info.devices) {
            const file = extractor_info.devices[device];
            devices[device] = await this.processDeviceAssembly(device, file, filters);
        }
        return result;
    }

    async processDeviceAssembly(deviceName, deviceFile, filters) {
        const deviceAsm = await fs.readFile(deviceFile, 'utf-8');
        return this.llvmIr.isLlvmIr(deviceAsm) ?
            this.llvmIr.process(deviceAsm, filters) :
            this.asm.process(deviceAsm, filters);
    }
}
