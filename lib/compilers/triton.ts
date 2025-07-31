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

import * as fs from 'node:fs/promises';
import Path from 'node:path';
import type {CompilationInfo, CompilationResult} from '../../types/compilation/compilation.interfaces.js';
import type {
    OptPipelineBackendOptions,
    OptPipelineOutput,
} from '../../types/compilation/opt-pipeline-output.interfaces.js';
import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';
import {CompilationEnvironment} from '../compilation-env.js';
import type {IAsmParser} from '../parsers/asm-parser.interfaces.js';
import {AmdgpuAsmParser} from '../parsers/asm-parser-amdgpu.js';
import {MlirAsmParser} from '../parsers/asm-parser-mlir.js';
import {PTXAsmParser} from '../parsers/asm-parser-ptx.js';
import {SassAsmParser} from '../parsers/asm-parser-sass.js';
import {MlirPassDumpParser} from '../parsers/mlir-pass-dump-parser.js';
import {parseOutput, resolvePathFromAppRoot} from '../utils.js';
import {BaseParser} from './argument-parsers.js';

export class TritonCompiler extends BaseCompiler {
    private compilerWrapperPath: string;

    static get key() {
        return 'triton';
    }

    parserMap: Record<string, IAsmParser>;
    mlirPassDumpParser: MlirPassDumpParser;

    constructor(compilerInfo: PreliminaryCompilerInfo, env: CompilationEnvironment) {
        super(compilerInfo, env);

        this.compilerWrapperPath =
            this.compilerProps('compilerWrapper', '') || resolvePathFromAppRoot('etc', 'scripts', 'triton_wrapper.py');

        // Enable the Opt Pipeline view
        this.compiler.optPipeline = {};
        // Used to parse the output of the opt pipeline
        this.mlirPassDumpParser = new MlirPassDumpParser(this.compilerProps);

        // Enable the Device Viewer
        this.compiler.supportsDeviceAsmView = true;
        // Define parsers for the different output files displayed in the Device Viewer
        const sassAsmParser = new SassAsmParser(this.compilerProps);
        const ptxAsmParser = new PTXAsmParser(this.compilerProps);
        const amdgpuAsmParser = new AmdgpuAsmParser();
        const mlirAsmParser = new MlirAsmParser();
        this.parserMap = {
            '.ttir': mlirAsmParser,
            '.ttgir': mlirAsmParser,
            '.ptx': ptxAsmParser,
            '.sass': sassAsmParser,
            '.source': mlirAsmParser,
            '.amdgcn': amdgpuAsmParser,
            '.llir': mlirAsmParser,
            '.json': sassAsmParser,
        };

        if (compilerInfo.group == 'triton_amd') {
            this.asm = amdgpuAsmParser;
        } else if (compilerInfo.group == 'triton_nvidia') {
            this.asm = ptxAsmParser;
        }
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string): string[] {
        // See etc/scripts/triton_wrapper.py for the options
        return ['-I', this.compilerWrapperPath, '--output_file', outputFilename];
    }

    override getArgumentParserClass() {
        return BaseParser;
    }

    override async extractDeviceCode(
        result: CompilationResult,
        filters: ParseFiltersAndOutputOptions,
        compilationInfo: CompilationInfo,
    ) {
        const devices = {...result.devices};

        const {dirPath} = result;
        if (!dirPath) {
            return result;
        }

        // Extract the device code from the output directory
        const files = await fs.readdir(dirPath);
        await Promise.all(
            files.map(async filename => {
                const ext = Path.extname(filename);
                const parser = this.parserMap[ext];
                if (!parser) {
                    return;
                }

                // Read the file
                const data = await fs.readFile(Path.join(dirPath, filename), 'utf8');

                // Parse the assembly with line numbers
                let device;
                if (ext === '.llir') {
                    device = await this.llvmIr.process(data, {
                        filterDebugInfo: false,
                        filterIRMetadata: false,
                        filterAttributes: false,
                        filterComments: false,
                        noDiscardValueNames: false,
                        demangle: false,
                    });
                } else {
                    device = await parser.process(data, filters);
                }

                Object.assign(devices, {[filename]: device});
            }),
        );
        result.devices = devices;
        return result;
    }

    override async generateOptPipeline(
        inputFilename: string,
        options: string[],
        filters: ParseFiltersAndOutputOptions,
        optPipelineOptions: OptPipelineBackendOptions,
    ): Promise<OptPipelineOutput | undefined> {
        // Call the script to generate the opt pipeline
        const execOptions = this.getDefaultExecOptions();
        const outputFilename = Path.join(Path.dirname(inputFilename), 'opt_pipeline.txt');
        const optOptions = [...options, '--opt_pipeline_file', outputFilename];

        const compileStart = performance.now();
        await this.runCompiler(this.compiler.exe, optOptions, inputFilename, execOptions);
        const compileEnd = performance.now();

        // Read the output file and parse it
        try {
            const rawText = await fs.readFile(outputFilename, {encoding: 'utf8'});
            const lines = parseOutput(rawText);

            const parseStart = performance.now();
            const llvmOptPipeline = await this.mlirPassDumpParser.process(lines, filters, optPipelineOptions);
            const parseEnd = performance.now();

            return {
                results: llvmOptPipeline,
                compileTime: compileEnd - compileStart,
                parseTime: parseEnd - parseStart,
            };
        } catch (e: any) {
            return {
                error: e.toString(),
                results: {},
                compileTime: compileEnd - compileStart,
            };
        }
    }

    override getDefaultFilters() {
        return {
            intel: false,
            commentOnly: false,
            directives: false,
            labels: false,
            optOutput: true,
            binary: false,
            execute: false,
            demangle: false,
            libraryCode: false,
            trim: false,
            binaryObject: false,
            debugCalls: false,
        };
    }
}
