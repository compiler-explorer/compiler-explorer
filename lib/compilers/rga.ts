// Copyright (c) 2022, Compiler Explorer Authors
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

import {readdir, readFile, rename, writeFile} from 'fs-extra';

import {CompilationResult, ExecutionOptions} from '../../types/compilation/compilation.interfaces';
import {ParseFilters} from '../../types/features/filters.interfaces';
import {BaseCompiler} from '../base-compiler';
import * as exec from '../exec';
import {logger} from '../logger';

interface ASICSelection {
    asic?: string;
    error?: string;
    printASICs?: boolean;
}

// RGA := Radeon GPU Analyzer
export class RGACompiler extends BaseCompiler {
    private dxcPath: string;

    static get key() {
        return 'rga';
    }

    constructor(info: any, env: any) {
        super(info, env);

        this.compiler.supportsIntel = false;
        this.dxcPath = this.compilerProps(`compiler.${this.compiler.id}.dxcPath`);
        logger.debug(`RGA compiler ${this.compiler.id} configured to use DXC at ${this.dxcPath}`);
    }

    override optionsForFilter(filters: ParseFilters, outputFilename: any, userOptions?: any): any[] {
        return [outputFilename];
    }

    extractASIC(dxcArgs: string[]): ASICSelection {
        // Scan dxc args for an `--asic` argument that should be stripped and passed later to RGA
        // Default to RDNA2
        let asic = 'gfx1030';
        let printASICs = true;
        for (let i = 0; i !== dxcArgs.length; ++i) {
            const arg = dxcArgs[i];
            if (arg === '--asic') {
                // NOTE: the last arguments are the input source file and -spirv, so check
                // if --asic immediately precedes that
                if (i === dxcArgs.length - 3) {
                    return {
                        error: '--asic flag supplied without subsequent ASIC!',
                    };
                }
                asic = dxcArgs[i + 1];
                // Do a quick sanity check to determine if a valid ASIC was supplied
                if (!asic.startsWith('gfx')) {
                    return {
                        error: `The argument immediately following --asic doesn't appear to be a valid ASIC.
Please supply an ASIC from the following options:`,
                        printASICs: true,
                    };
                }
                // Remove these two arguments from the dxcArgs list
                dxcArgs.splice(i, 2);

                // If the user supplied a specific ASIC, don't bother printing available ASIC options
                printASICs = false;
                break;
            }
        }

        return {
            asic,
            printASICs,
        };
    }

    execTime(startTime: bigint, endTime: bigint): string {
        return ((endTime - startTime) / BigInt(1000000)).toString();
    }

    override async runCompiler(
        compiler: string,
        options: string[],
        inputFilename: string,
        execOptions: ExecutionOptions,
    ): Promise<CompilationResult> {
        if (!execOptions) {
            execOptions = this.getDefaultExecOptions();
        }

        if (!execOptions.customCwd) {
            execOptions.customCwd = path.dirname(inputFilename);
        }

        const result = await this.execDXCandRGA(compiler, options, execOptions);
        return this.transformToCompilationResult(result, inputFilename);
    }

    async execDXCandRGA(filepath: string, args: string[], execOptions: ExecutionOptions): Promise<any> {
        // RGA is invoked in two steps. First, DXC is invoked to compile the SPIR-V output of the HLSL file.
        // Next, RGA is invoked to consume the SPIR-V output and produce the requested ISA.

        // Track the total time spent instead of relying on executeDirect's internal timing facility
        const startTime = process.hrtime.bigint();

        // The first argument is the target output file
        const outputFile = args[0];
        const outputDir = path.dirname(outputFile);
        const spvTemp = 'output.spv.txt';
        logger.debug(`Intermediate SPIR-V output: ${spvTemp}`);

        const dxcArgs = args.slice(1);
        if (!dxcArgs.includes('-spirv')) {
            dxcArgs.push('-spirv');
        }
        logger.debug(`DXC args: ${dxcArgs}`);

        const asicSelection = this.extractASIC(dxcArgs);
        if (asicSelection.error) {
            // Invalid user ASIC selected, bail out immediately
            const endTime = process.hrtime.bigint();

            // Synthesize a faux-execution result (see promise resolution code in executeDirect)
            return {
                code: -1,
                okToCache: true,
                filenameTransform: x => x,
                stdout: asicSelection.error,
                execTime: this.execTime(startTime, endTime),
            };
        }

        const dxcResult = await exec.execute(this.dxcPath, dxcArgs, execOptions);
        if (dxcResult.code !== 0) {
            // Failed to compile SPIR-V intermediate product. Exit immediately with DXC invocation result.
            const endTime = process.hrtime.bigint();
            dxcResult.execTime = this.execTime(startTime, endTime);
            return dxcResult;
        }

        try {
            await writeFile(path.join(outputDir, spvTemp), dxcResult.stdout);
        } catch (e) {
            const endTime = process.hrtime.bigint();
            return {
                code: -1,
                okToCache: true,
                filenameTransform: x => x,
                stdout: 'Failed to emit intermediate SPIR-V result.',
                execTime: this.execTime(startTime, endTime),
            };
        }

        let registerAnalysisFile = 'livereg.txt';
        const rgaArgs = [
            '-s',
            'vk-spv-txt-offline',
            '-c',
            asicSelection.asic || '',
            '--isa',
            outputFile,
            '--livereg',
            registerAnalysisFile,
            spvTemp,
        ];
        logger.debug(`RGA args: ${rgaArgs}`);

        const rgaResult = await exec.execute(filepath, rgaArgs, execOptions);
        if (rgaResult.code !== 0) {
            // Failed to compile AMD ISA
            const endTime = process.hrtime.bigint();
            rgaResult.execTime = this.execTime(startTime, endTime);
            return rgaResult;
        }

        // RGA doesn't emit the exact file we requested. It prepends the requested GPU
        // architecture and appends the shader type (with underscore separators). Here,
        // we rename the generated file to the output file Compiler Explorer expects.

        const files = await readdir(outputDir, {encoding: 'utf-8'});
        for (const file of files) {
            if (file.startsWith((asicSelection.asic as string) + '_output')) {
                await rename(path.join(outputDir, file), outputFile);

                registerAnalysisFile = path.join(outputDir, file.replace('output', 'livereg').replace('.s', '.txt'));
                // The register analysis file contains a legend, and register liveness data
                // for each line of disassembly. Interleave those lines into the final output
                // as assembly comments.
                const asm = await readFile(outputFile, 'utf-8');
                const asmLines = asm.split(/\r?\n/);
                const analysis = await readFile(registerAnalysisFile, 'utf-8');
                const analysisLines = analysis.split(/\r?\n/);

                // The first few lines of the register analysis are the legend. Emit those lines
                // as comments at the start of the output.
                let analysisOffset = analysisLines.indexOf('');
                analysisOffset += 3;
                const epilogueOffset = analysisLines.indexOf('', analysisOffset);
                const outputAsm = analysisLines.slice(epilogueOffset + 1).map(line => `; ${line}`);
                outputAsm.push(...analysisLines.slice(0, analysisOffset).map(line => `; ${line}`), '\n');

                let asmOffset = asmLines.indexOf('');
                outputAsm.push(...asmLines.slice(0, asmOffset));
                asmOffset += 1;

                // Perform the interleave
                for (let i = 0; i !== asmOffset + asmLines.length; ++i) {
                    if (i + analysisOffset >= epilogueOffset) {
                        outputAsm.push(...asmLines.slice(i));
                        break;
                    }

                    // Check if this line of assembly corresponds to a label. If so, emit the asm line
                    // and continue from the next line (the register analysis file operates on instructions,
                    // and labels are not considered instructions)
                    if (asmLines[i + asmOffset].startsWith('label')) {
                        outputAsm.push(asmLines[i + asmOffset]);
                        ++asmOffset;
                        --i;
                        continue;
                    }

                    outputAsm.push(`; ${analysisLines[i + analysisOffset]}`, asmLines[i + asmOffset]);
                }

                await writeFile(outputFile, outputAsm.join('\n'));

                if (asicSelection.printASICs) {
                    rgaResult.stdout += `ISA compiled with the default AMD ASIC (Radeon RX 6800 series RDNA2).
To override this, pass --asic [ASIC] to the options above (nonstandard DXC option),
where [ASIC] corresponds to one of the following options:`;

                    const asics = await exec.execute(filepath, ['-s', 'vk-spv-txt-offline', '-l'], execOptions);
                    rgaResult.stdout += '\n';
                    rgaResult.stdout += asics.stdout;
                }

                const endTime = process.hrtime.bigint();
                rgaResult.execTime = this.execTime(startTime, endTime);
                return rgaResult;
            }
        }

        // Arriving here means the expected ISA result wasn't emitted. Synthesize an error.
        const endTime = process.hrtime.bigint();
        rgaResult.execTime = this.execTime(startTime, endTime);
        rgaResult.stdout += `\nRGA didn't emit expected ISA output.`;
        return rgaResult;
    }
}
