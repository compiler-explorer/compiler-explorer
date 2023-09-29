// Copyright (c) 2023, Simon Boehm
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

import type {CompilationInfo} from '../../types/compilation/compilation.interfaces.js';
import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';
import {asSafeVer} from '../utils.js';
import {SassAsmParser} from '../parsers/asm-parser-sass.js';

import Semver from 'semver';
import {unwrap} from '../assert.js';

import * as fs from 'fs/promises';
import Path from 'path';

import {BaseParser} from './argument-parsers.js';

export class TritonCompiler extends BaseCompiler {
    private readonly compileScriptPath: string;

    static get key() {
        return 'triton';
    }

    deviceAsmParser: SassAsmParser;

    constructor(compilerInfo: PreliminaryCompilerInfo, env) {
        super(compilerInfo, env);
        this.compiler.supportsDeviceAsmView = true;
        this.compileScriptPath = this.compilerProps<string>(`compiler.${this.compiler.id}.compileScript`);
        this.deviceAsmParser = new SassAsmParser(this.compilerProps);
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string) {
        return ['-I', this.compileScriptPath, '--out-path', outputFilename];
    }

    async nvdisasm(outputFilename: string, result: any, maxOutput: number) {
        // Read the content of output .c file
        const fileContent: string = await fs.readFile(outputFilename, 'utf-8');

        // Match the lines of cubin hex array called CUBIN_NAME
        const REGEX_HEX_LINE = /CUBIN_NAME\[\d+\]\s=\s\{(.*?)};/gs;
        const match = REGEX_HEX_LINE.exec(fileContent);

        if (!match) {
            result.asm = '<failed to extract cubin from Triton output>';
            return result;
        }

        // Strip down to just hexadecimal values and space
        const hexArrayStr = match[1].replace(/0x/g, '');

        // Convert hexArrayStr to buffer and then write to a temporary file
        const binaryData = Buffer.from(hexArrayStr.split(',').map(h => parseInt(h.trim(), 16)));
        const tempFilePath = Path.join(result.dirPath, 'temp.cubin');
        await fs.writeFile(tempFilePath, binaryData);

        // nvdisasm
        const {nvdisasm, semver} = this.compiler;
        const args = Semver.lt(asSafeVer(semver), '11.0.0', true)
            ? [tempFilePath, '-c', '-g']
            : [tempFilePath, '-c', '-g', '-hex'];

        const {code, execTime, stdout} = await this.exec(unwrap(nvdisasm), args, {
            maxOutput,
            customCwd: result.dirPath,
        });

        if (code === 0) {
            result.objdumpTime = execTime;
            result.asm = this.postProcessObjdumpOutput(stdout);
        } else {
            result.asm = `<No output: ${Path.basename(unwrap(nvdisasm))} returned ${code}>`;
        }
        return result;
    }

    override async extractDeviceCode(result, filters, compilationInfo: CompilationInfo) {
        const {dirPath} = result;
        const {demangle} = filters;
        const devices = {...result.devices};
        if (dirPath) {
            const files = await fs.readdir(dirPath);
            const maxSize = this.env.ceProps('max-asm-size', 64 * 1024 * 1024);
            await Promise.all(
                files
                    .filter(f => f.endsWith('.c'))
                    .map(async name => {
                        const type = 'SASS';
                        const {asm} = await this.nvdisasm(Path.join(dirPath, name), {dirPath}, maxSize);
                        Object.assign(devices, {
                            [type]: await this.postProcessAsm(
                                {
                                    okToCache: demangle,
                                    ...this.deviceAsmParser.process(asm, {...filters, binary: type === 'SASS'}),
                                },
                                {...filters, binary: type === 'SASS'},
                            ),
                        });
                    }),
            );
            result.devices = devices;
        }
        return result;
    }

    override getArgumentParser() {
        return BaseParser;
    }
}
