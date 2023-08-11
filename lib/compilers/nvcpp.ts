// Copyright (c) 2023, Compiler Explorer Authors
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

import * as fs from 'fs/promises';
import path from 'path';

import {CompilationInfo} from '../../types/compilation/compilation.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';
import {SassAsmParser} from '../parsers/asm-parser-sass.js';
import {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import {unwrap} from '../assert.js';

export class NvcppCompiler extends BaseCompiler {
    protected deviceAsmParser: SassAsmParser;
    protected cuobjdump: string | undefined;

    static get key() {
        return 'nvcpp';
    }

    constructor(info: PreliminaryCompilerInfo, env) {
        super(info, env);

        this.cuobjdump = this.compilerProps<string | undefined>(
            'compiler.' + this.compiler.id + '.cuobjdump',
            undefined,
        );

        this.deviceAsmParser = new SassAsmParser(this.compilerProps);

        this.compiler.supportsDeviceAsmView = true;
    }

    async nvdisasm(result, outputFilename: string, maxOutput: number) {
        const disasmResult = await this.exec(unwrap(this.compiler.nvdisasm), ['-c', '-g', '-hex', outputFilename], {
            maxOutput,
            customCwd: path.dirname(outputFilename),
        });

        const newResult = {asm: ''};

        if (disasmResult.code === 0) {
            newResult.asm = disasmResult.stdout;
        } else {
            newResult.asm = `<No output: ${path.basename(unwrap(this.compiler.nvdisasm))} returned ${
                disasmResult.code
            }>`;
        }
        return newResult;
    }

    async extractDeviceBinariesFromExecutable(result, compilationInfo: CompilationInfo) {
        if (this.cuobjdump) {
            const execOptions = this.getDefaultExecOptions();
            execOptions.customCwd = result.dirPath;

            if (!result.buildsteps) result.buildsteps = [];

            await this.exec(this.cuobjdump, ['-xelf', 'all', compilationInfo.executableFilename], execOptions);

            // couldn't test this, does this happen?
            await this.exec(this.cuobjdump, ['-xptx', 'all', compilationInfo.executableFilename], execOptions);

            return true;
        }

        return false;
    }

    override async extractDeviceCode(result, filters, compilationInfo: CompilationInfo) {
        const {dirPath} = result;
        const {demangle} = filters;
        const devices = {...result.devices};

        if (filters.binary) {
            await this.extractDeviceBinariesFromExecutable(result, compilationInfo);
        }

        if (dirPath) {
            const files = await fs.readdir(dirPath);
            const maxSize = this.env.ceProps('max-asm-size', 64 * 1024 * 1024);
            await Promise.all(
                files
                    .filter(f => f.endsWith('.ptx') || f.endsWith('.cubin'))
                    .map(async name => {
                        const type = name.endsWith('.ptx') ? 'PTX' : 'SASS';
                        const {asm} =
                            type === 'PTX'
                                ? {asm: await fs.readFile(path.join(dirPath, name), 'utf8')}
                                : await this.nvdisasm(result, path.join(dirPath, name), maxSize);
                        const archAndCode = name.split('.').slice(1, -1).join(', ') || '';
                        const nameAndArch = type + (archAndCode ? ` (${archAndCode.toLowerCase()})` : '');
                        Object.assign(devices, {
                            [nameAndArch]: await this.postProcessAsm(
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
}
