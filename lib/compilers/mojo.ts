// Copyright (c) 2025, Compiler Explorer Authors
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

import fs from 'node:fs/promises';
import path from 'node:path';

import semver from 'semver';

import type {CompilationInfo, CompilationResult} from '../../types/compilation/compilation.interfaces.js';
import type {LLVMIrBackendOptions} from '../../types/compilation/ir.interfaces.js';
import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import type {SelectedLibraryVersion} from '../../types/libraries/libraries.interfaces.js';
import {unwrap} from '../assert.js';
import {BaseCompiler} from '../base-compiler.js';
import type {CompilationEnvironment} from '../compilation-env.js';
import {getAmdGpuInstructionSetFromLabel} from '../instructionsets.js';
import type {ClientOptionsType} from '../options-handler.js';
import type {IAsmParser} from '../parsers/asm-parser.interfaces.js';
import {AmdgpuAsmParser} from '../parsers/asm-parser-amdgpu.js';
import {PTXAsmParser} from '../parsers/asm-parser-ptx.js';
import {SassAsmParser} from '../parsers/asm-parser-sass.js';
import {parseOutput} from '../utils.js';
import {MojoParser} from './argument-parsers.js';

export class MojoCompiler extends BaseCompiler {
    private deviceParsers: Record<string, IAsmParser>;
    private sassParser: SassAsmParser;

    static get key() {
        return 'mojo';
    }

    constructor(info: PreliminaryCompilerInfo, env: CompilationEnvironment) {
        super(info, env);
        this.compiler.supportsIrView = true;
        this.compiler.irArg = ['--emit', 'llvm'];
        this.compiler.minIrArgs = ['--emit=llvm'];
        this.deviceParsers = {
            '.ptx': new PTXAsmParser(this.compilerProps),
            '.amdgcn': new AmdgpuAsmParser(),
        };
        this.sassParser = new SassAsmParser(this.compilerProps);
    }

    override async initialise(mtime: Date, clientOptions: ClientOptionsType, isPrediscovered = false) {
        const compiler = await super.initialise(mtime, clientOptions, isPrediscovered);
        if (!compiler || this.getRemote()) return compiler;

        const version = semver.coerce(this.compiler.version);
        this.compiler.supportsDeviceAsmView = !!version && semver.gte(version, '0.26.2');
        return compiler;
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string) {
        return [
            'build',
            ...super.optionsForFilter(filters, outputFilename).map(option => {
                if (option === '-S') return '--emit=asm';
                // Full debug info changes generated code; line tables suffice for source mapping.
                if (option === '-g') return '-g1';
                return option;
            }),
        ];
    }

    override getSharedLibraryPathsAsArguments(
        libraries: SelectedLibraryVersion[],
        libDownloadPath: string | undefined,
        toolchainPath: string | undefined,
        dirPath: string,
    ) {
        const version = semver.coerce(this.compiler.version);
        // Older Mojo releases reject -Xlinker.
        if (!version || semver.lt(version, '0.26.1')) {
            return [];
        }

        return super.getSharedLibraryPathsAsArguments(libraries, libDownloadPath, toolchainPath, dirPath);
    }

    override async generateIR(
        inputFilename: string,
        options: string[],
        irOptions: LLVMIrBackendOptions,
        produceCfg: boolean,
        filters: ParseFiltersAndOutputOptions,
    ) {
        const newOptions = options.filter(
            (opt, idx, arr) =>
                opt !== '-o' &&
                opt !== '--emit' &&
                !opt.startsWith('--emit=') &&
                arr[idx - 1] !== '-o' &&
                arr[idx - 1] !== '--emit',
        );
        newOptions.push('-o', this.getIrOutputFilename(inputFilename));
        return await super.generateIR(inputFilename, newOptions, irOptions, produceCfg, filters);
    }

    override async postProcess(
        result: CompilationResult,
        outputFilename: string,
        filters: ParseFiltersAndOutputOptions,
        produceOptRemarks = false,
    ) {
        const processed = await super.postProcess(result, outputFilename, filters, produceOptRemarks);
        const outputType = result.compilationOptions
            ?.findLast((option, index, options) => option.startsWith('--emit=') || options[index - 1] === '--emit')
            ?.replace('--emit=', '');
        if (
            result.code === 0 &&
            result.asmSize !== undefined &&
            result.asmSize < this.env.ceProps('max-asm-size', 64 * 1024 * 1024) &&
            (outputType === 'air' || outputType === 'llvm-bitcode')
        ) {
            try {
                result.asm = await this.disassembleBitcode(outputFilename);
                result.languageId = 'llvm-ir';
                result.instructionSet = 'llvm';
            } catch (err) {
                result.asm = `<Unable to decode LLVM bitcode: ${err}>`;
                result.stderr.push({text: `Unable to decode LLVM bitcode: ${err}`});
            }
        }
        return processed;
    }

    override async extractDeviceCode(
        result: CompilationResult,
        filters: ParseFiltersAndOutputOptions,
        compilationInfo: CompilationInfo,
    ) {
        if (
            !this.compiler.supportsDeviceAsmView ||
            result.code !== 0 ||
            !result.dirPath ||
            filters.binary ||
            filters.binaryObject
        ) {
            return result;
        }

        const devices = {...result.devices};
        const outputFilenames = [compilationInfo.outputFilename];
        if (result.irOutput?.code === 0 && compilationInfo.inputFilename) {
            outputFilenames.push(this.getIrOutputFilename(compilationInfo.inputFilename));
        }
        const prefixesByDirectory = new Map<string, string[]>();
        for (const filename of outputFilenames) {
            const {dir, name} = path.parse(filename);
            const prefixes = prefixesByDirectory.get(dir) ?? [];
            prefixes.push(name + '_');
            prefixesByDirectory.set(dir, prefixes);
        }
        for (const [directory, prefixes] of prefixesByDirectory) {
            const files = await fs.readdir(directory, {withFileTypes: true});
            for (const file of files.sort((a, b) => a.name.localeCompare(b.name))) {
                const ext = path.extname(file.name);
                const parser = this.deviceParsers[ext];
                if (
                    !file.isFile() ||
                    !prefixes.some(prefix => file.name.startsWith(prefix)) ||
                    (!parser && ext !== '.ll' && ext !== '.air')
                ) {
                    continue;
                }

                const filename = path.join(directory, file.name);
                const deviceName = path.relative(result.dirPath, filename);
                try {
                    const data =
                        ext === '.air' ? await this.disassembleBitcode(filename) : await fs.readFile(filename, 'utf8');
                    const parsed = parser
                        ? await parser.process(data, filters)
                        : await this.llvmIr.processFromFilters(data, filters);
                    const instructionSet =
                        ext === '.ptx' ? 'ptx' : ext === '.amdgcn' ? getAmdGpuInstructionSetFromLabel(data) : 'llvm';
                    devices[deviceName] = {
                        code: 0,
                        timedOut: false,
                        stdout: [],
                        stderr: [],
                        ...parsed,
                        instructionSet,
                    };
                    if (ext === '.ptx' && this.compiler.ptxas && this.compiler.nvdisasm) {
                        devices[`SASS (${deviceName})`] = await this.generateSass(filename, data, filters);
                    }
                } catch (err) {
                    devices[deviceName] = {
                        code: -1,
                        timedOut: false,
                        stdout: [],
                        stderr: [{text: `Unable to process device output: ${err}`}],
                    };
                }
            }
        }
        result.devices = devices;
        return result;
    }

    private async disassembleBitcode(filename: string): Promise<string> {
        const tool = this.compiler.llvmDisassembler;
        if (!tool) throw new Error('No LLVM disassembler configured');
        const output = await this.exec(tool, [filename, '-o', '-'], {
            ...this.getDefaultExecOptions(),
            customCwd: path.dirname(filename),
            maxOutput: this.env.ceProps('max-asm-size', 64 * 1024 * 1024),
        });
        if (output.timedOut) throw new Error('LLVM disassembly timed out');
        if (output.truncated) throw new Error('LLVM disassembly exceeded the output size limit');
        if (output.code !== 0) throw new Error(`LLVM disassembly failed (${output.code}): ${output.stderr}`);
        return output.stdout;
    }

    private async generateSass(
        filename: string,
        ptx: string,
        filters: ParseFiltersAndOutputOptions,
    ): Promise<CompilationResult> {
        try {
            const target = ptx.match(/^\s*\.target\s+(sm_\d+[a-z]?)(?=\s|,|$)/m)?.[1];
            if (!target) throw new Error('No NVIDIA GPU target found in PTX');

            const cubin = filename.replace(/\.ptx$/, '.cubin');
            const execOptions = {
                ...this.getDefaultExecOptions(),
                customCwd: path.dirname(filename),
                maxOutput: this.env.ceProps('max-asm-size', 64 * 1024 * 1024),
            };
            let tool = unwrap(this.compiler.ptxas);
            let output = await this.exec(tool, ['-arch', target, filename, '-o', cubin], execOptions);
            if (output.code === 0 && !output.timedOut) {
                tool = unwrap(this.compiler.nvdisasm);
                output = await this.exec(tool, [cubin, '-c', '-g', '-hex'], execOptions);
            }
            const failed = output.code !== 0 || output.timedOut;
            return {
                code: failed ? output.code || -1 : 0,
                timedOut: output.timedOut,
                truncated: output.truncated,
                stdout: [],
                stderr: parseOutput(output.stderr),
                ...(failed
                    ? {asm: parseOutput(`<${path.basename(tool)} failed to generate SASS>\n${output.stderr}`)}
                    : this.sassParser.process(output.stdout, {...filters, binary: true})),
                instructionSet: 'sass',
            };
        } catch (err) {
            return {
                code: -1,
                timedOut: false,
                stdout: [],
                stderr: [{text: `Unable to generate SASS: ${err}`}],
                asm: [{text: `<Unable to generate SASS: ${err}>`}],
                instructionSet: 'sass',
            };
        }
    }

    override getArgumentParserClass() {
        return MojoParser;
    }

    override getDefaultExecOptions() {
        const execOptions = super.getDefaultExecOptions();
        execOptions.env.MODULAR_CRASH_REPORTING_ENABLED = 'false';
        return execOptions;
    }
}
