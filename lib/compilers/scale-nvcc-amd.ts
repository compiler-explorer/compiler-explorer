// Copyright (c) 2026, Compiler Explorer Authors
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

import * as fs from 'node:fs/promises';
import Path from 'node:path';

import _ from 'underscore';

import type {CompilationInfo, CompilationResult} from '../../types/compilation/compilation.interfaces.js';
import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {unwrap} from '../assert.js';
import {BaseCompiler} from '../base-compiler.js';
import {CompilationEnvironment} from '../compilation-env.js';
import {logger} from '../logger.js';
import {AmdgpuAsmParser} from '../parsers/asm-parser-amdgpu.js';
import {ClangParser} from './argument-parsers.js';

interface ExecResult {
    code: number;
    stdout: string;
    stderr: string;
}

export class ScaleNvccAMDCompiler extends BaseCompiler {
    static get key() {
        return 'scale-nvcc-amd';
    }

    amdgpuAsmParser: AmdgpuAsmParser;

    constructor(info: PreliminaryCompilerInfo, env: CompilationEnvironment) {
        super(info, env);
        this.compiler.supportsOptOutput = true;
        this.compiler.supportsDeviceAsmView = true;
        this.amdgpuAsmParser = new AmdgpuAsmParser();
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string) {
        const opts = ['-g', '-lineinfo', '--keep-device-functions'];
        if (!filters.execute) {
            opts.push('-keep', '-keep-dir', Path.dirname(outputFilename));
            if (!filters.binary) {
                opts.push('-Xcompiler=-S');
            }
        }
        return opts;
    }

    override getArgumentParserClass() {
        return ClangParser;
    }

    override optOutputRequested(options: string[]) {
        return (
            super.optOutputRequested(options) ||
            options.includes('--optimization-info') ||
            options.includes('-opt-info')
        );
    }

    //Helpers to get device cpdes (bc and .s)
    private static readonly scaleDeviceFileRe = /-cuda-amdgcn-amd-amdhsa-scale-(gfx[^./]+)\.s$/;
    private static readonly scaleDeviceBcFileRe = /-cuda-amdgcn-amd-amdhsa-scale-(gfx[^./]+)\.bc$/;

    // Temp workaround since -o -S does not work in scale-1.7.1
    // Side effect: it works only if one device file is present.
    // will be cleaned when scale's bug is fixed
    private async findHostAsmFile(dirPath: string): Promise<string | null> {
        try {
            const files = await fs.readdir(dirPath);
            const hostFiles = files.filter(f => f.endsWith('.s') && !ScaleNvccAMDCompiler.scaleDeviceFileRe.test(f));
            if (hostFiles.length !== 1) {
                return null;
            }

            return Path.join(dirPath, hostFiles[0]);
        } catch {
            return null;
        }
    }

    override async postProcess(result, outputFilename: string, filters: ParseFiltersAndOutputOptions) {
        // TEMP (scale support): outputFilename as originally computed may not
        // exist since we no longer pass `-o`. Try to recover the real file.
        if (!filters.binary && result.dirPath) {
            try {
                await fs.stat(outputFilename);
            } catch {
                const hostAsm = await this.findHostAsmFile(result.dirPath);
                if (hostAsm) {
                    try {
                        result.asmSize = (await fs.stat(hostAsm)).size;
                    } catch {
                        // leave asmSize as-is; base behaviour reports "no output" below
                    }
                    outputFilename = hostAsm;
                }
            }
        }

        const maxSize = this.env.ceProps('max-asm-size', 64 * 1024 * 1024);
        const optPromise = result.optPath ? this.processOptOutput(result.optPath) : Promise.resolve([]);
        const postProcess = _.compact(this.compiler.postProcess);
        const asmPromise = (
            filters.binary
                ? this.objdump(outputFilename, {}, maxSize, !!filters.intel, !!filters.demangle, false, false, filters)
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
                      }
                      const contents = await fs.readFile(outputFilename, {encoding: 'utf8'});
                      result.asm = contents.toString();
                      return result;
                  })()
        ).then(asm => {
            result.asm = typeof asm === 'string' ? asm : asm.asm;
            return result;
        });
        return Promise.all([asmPromise, optPromise, []]);
    }

    // ---- device-code extraction helpers -------------------------------------------------

    /** Pulls the captured arch (e.g. `gfx1100`) out of a scale per-target device filename. */
    private static archFromDeviceFileName(name: string, re: RegExp): string {
        return unwrap(name.match(re))[1];
    }

    private async runLlvmDis(bcPath: string, dirPath: string): Promise<ExecResult> {
        const result = await this.exec(unwrap(this.compiler.llvmDisassembler), [bcPath], {customCwd: dirPath});
        return result;
    }

    private static llvmIrOptions(demangle: boolean) {
        return {
            demangle,
            filterDebugInfo: false,
            filterIRMetadata: false,
            filterAttributes: false,
            filterComments: false,
        };
    }

    /**
     * Handles a single scale/nvcc device `.bc` (LLVM bitcode) file: runs
     * `llvm-dis` to get readable LLVM IR text
     */
    private async processBcDeviceFile(
        name: string,
        dirPath: string,
        filters: ParseFiltersAndOutputOptions,
        demangle: boolean,
        devices: Record<string, CompilationResult>,
    ): Promise<void> {
        const archAndCode = ScaleNvccAMDCompiler.archFromDeviceFileName(name, ScaleNvccAMDCompiler.scaleDeviceBcFileRe);

        const irNameAndArch = `Device LLVM IR (${archAndCode.toLowerCase()})`;

        if (!this.compiler.llvmDisassembler) {
            Object.assign(devices, {
                [irNameAndArch]: await this.postProcessAsm(
                    {
                        okToCache: false,
                        ...(await this.llvmIr.process(
                            '<error: no llvm-dis found to disassemble bitcode>',
                            ScaleNvccAMDCompiler.llvmIrOptions(demangle),
                        )),
                    },
                    {...filters, binary: false},
                ),
            });
            return;
        }

        const bcPath = Path.join(dirPath, name);
        const llPath = Path.join(dirPath, `${Path.basename(name, '.bc')}.ll`);

        let irText: string;
        try {
            const disResult = await this.runLlvmDis(bcPath, dirPath);
            irText =
                disResult.code === 0
                    ? await fs.readFile(llPath, 'utf8')
                    : `<llvm-dis failed with code ${disResult.code}>`;
        } catch (err) {
            irText = `<llvm-dis failed: ${err}>`;
        }

        Object.assign(devices, {
            [irNameAndArch]: await this.postProcessAsm(
                {
                    okToCache: demangle,
                    ...(await this.llvmIr.process(irText, ScaleNvccAMDCompiler.llvmIrOptions(demangle))),
                },
                {...filters, binary: false},
            ),
        });
    }

    //Get the .s an display it as AMDGPU code, get the .bc and extract llvmir with llvmdis
    override async extractDeviceCode(
        result: CompilationResult,
        filters: ParseFiltersAndOutputOptions,
        compilationInfo: CompilationInfo,
    ) {
        const {dirPath} = result;
        const {demangle} = filters;
        const devices: Record<string, CompilationResult> = {...result.devices};

        if (dirPath) {
            const files = await fs.readdir(dirPath);

            const asmDeviceFiles = files.filter(f => ScaleNvccAMDCompiler.scaleDeviceFileRe.test(f));
            const bcDeviceFiles = files.filter(f => ScaleNvccAMDCompiler.scaleDeviceBcFileRe.test(f));

            await Promise.all([
                ...asmDeviceFiles.map(async name => {
                    try {
                        const archAndCode = unwrap(name.match(ScaleNvccAMDCompiler.scaleDeviceFileRe))[1];
                        const asm = await fs.readFile(Path.join(dirPath, name), 'utf8');

                        const nameAndArch = `AMDGPU (${archAndCode.toLowerCase()})`;
                        Object.assign(devices, {
                            [nameAndArch]: await this.postProcessAsm(
                                {
                                    okToCache: demangle,
                                    ...this.amdgpuAsmParser.process(asm, {...filters, binary: false}),
                                },
                                {...filters, binary: false},
                            ),
                        });
                    } catch (err) {
                        logger.error('[extractDeviceCode]: exception running postProcessAsm for', name, err);
                        // Never let a single device-asm failure take down the whole result.
                    }
                }),
                ...bcDeviceFiles.map(async name => {
                    try {
                        await this.processBcDeviceFile(name, dirPath, filters, !!demangle, devices);
                    } catch (err) {
                        logger.error('[extractDeviceCode]: exception running processBcDeviceFile for', name, err);
                        // Never let a single device-IR failure take down the whole result.
                    }
                }),
            ]);

            result.devices = devices;
        }

        return result;
    }
}
