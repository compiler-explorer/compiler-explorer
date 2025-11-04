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

import fs from 'node:fs';
import path from 'node:path';

import type {ParsedAsmResult, ParsedAsmResultLine} from '../../types/asmresult/asmresult.interfaces.js';
import type {CompilationResult} from '../../types/compilation/compilation.interfaces.js';
import type {LLVMIrBackendOptions} from '../../types/compilation/ir.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import type {Language} from '../../types/languages.interfaces.js';
import {assert} from '../assert.js';
import {BaseCompiler} from '../base-compiler.js';
import {maybeRemapJailedDir} from '../exec.js';
import {PolkaVMAsmParser} from '../parsers/asm-parser-polkavm.js';
import {ResolcRiscVAsmParser} from '../parsers/asm-parser-resolc-riscv.js';
import {changeExtension} from '../utils.js';
import {type BaseParser, ResolcParser} from './argument-parsers.js';

/**
 * The kind of input provided by the user.
 * This is determined by the language chosen as Resolc
 * supports both Solidity and Yul (Solidity IR).
 *
 * @note
 * The enum value must exactly match the {@link Language.id}.
 */
enum InputKind {
    Solidity = 'solidity',
    Yul = 'yul',
}

/**
 * The kind of output requested by the user.
 */
enum OutputKind {
    PolkaVM = 'pvm',
    RiscV = 'risc-v',
}

export class ResolcCompiler extends BaseCompiler {
    static get key() {
        return 'resolc';
    }

    private readonly pvmAsmParser: PolkaVMAsmParser;

    /**
     * @note
     * Needs to coincide with the [infrastructure configs](https://github.com/compiler-explorer/infra/blob/main/bin/yaml/solidity.yaml).
     */
    static get solcExe() {
        return '/opt/compiler-explorer/solc-0.8.30/solc';
    }

    constructor(...args: ConstructorParameters<typeof BaseCompiler>) {
        super(...args);

        this.asm = new ResolcRiscVAsmParser(this.compilerProps);
        this.pvmAsmParser = new PolkaVMAsmParser();

        // The arg producing LLVM IR (among other output) is already
        // included in optionsForFilter(), but irArg needs to be set.
        this.compiler.irArg = [];
        this.compiler.supportsIrView = true;
        this.compiler.supportsIrViewOptToggleOption = true;
        this.compiler.supportsYulView = this.inputIs(InputKind.Solidity);
    }

    override getSharedLibraryPathsAsArguments(): string[] {
        return [];
    }

    override getArgumentParserClass(): typeof BaseParser {
        return ResolcParser;
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions): string[] {
        filters.binaryObject = this.reinterpretBinaryObjectFilter(filters.binaryObject);
        filters.intel = false;

        const options = ['-g', '--solc', ResolcCompiler.solcExe, '--overwrite', '--debug-output-dir', 'artifacts'];
        if (this.inputIs(InputKind.Yul)) {
            options.push('--yul');
        }

        return options;
    }

    override isCfgCompiler(): boolean {
        return false;
    }

    override getOutputFilename(dirPath: string): string {
        return this.getOutputFilenameWithExtension(dirPath, '.pvmasm');
    }

    override getIrOutputFilename(
        inputFilename: string,
        _filters?: ParseFiltersAndOutputOptions,
        irOptions?: LLVMIrBackendOptions,
    ): string {
        const extension =
            irOptions?.showOptimized && this.compiler.supportsIrViewOptToggleOption
                ? '.optimized.ll'
                : '.unoptimized.ll';

        return this.getOutputFilenameWithExtension(path.dirname(inputFilename), extension);
    }

    override getObjdumpInputFilename(defaultOutputFilename: string): string {
        return changeExtension(defaultOutputFilename, '.o');
    }

    private getOutputFilenameWithExtension(dirPath: string, extension: string): string {
        const basenamePrefix = maybeRemapJailedDir(dirPath).split(path.sep).join('_');
        const contractName = this.inputIs(InputKind.Solidity)
            ? this.getSolidityContractName(dirPath)
            : this.getYulContractName(dirPath);
        const basename = `${basenamePrefix}_${this.compileFilename}.${contractName}${extension}`;

        return path.join(dirPath, 'artifacts', basename);
    }

    override async processAsm(
        result: CompilationResult,
        filters: ParseFiltersAndOutputOptions,
    ): Promise<ParsedAsmResult> {
        return this.outputIs(OutputKind.PolkaVM, filters)
            ? this.pvmAsmParser.process(result.asm as string, filters)
            : this.asm.process(result.asm as string, filters);
    }

    override async postProcessAsm(
        result: ParsedAsmResult,
        filters?: ParseFiltersAndOutputOptions,
    ): Promise<ParsedAsmResult> {
        result = await super.postProcessAsm(result, filters);
        result = this.removeOrphanedLabels(result, filters);
        this.maybeRemoveSourceMappings(result, filters);
        this.addOutputHeader(result, filters);

        return result;
    }

    /**
     * Remove orphaned labels.
     *
     * @example
     * Before:
     * ```
     * memset:
     * .LBB35_2:
     * .LBB35_3:
     * __entry:
     *      addi	sp, sp, -0x10
     * ```
     *
     * After:
     * ```
     * __entry:
     *      addi	sp, sp, -0x10
     * ```
     */
    private removeOrphanedLabels(result: ParsedAsmResult, filters?: ParseFiltersAndOutputOptions): ParsedAsmResult {
        // Orphaned RISC-V labels may be produced by the AsmParser when library code is skipped.
        if (!this.outputIs(OutputKind.RiscV, filters) || !filters?.libraryCode || !result.labelDefinitions) {
            return result;
        }

        const {asm, labelDefinitions} = result;
        result.asm = asm.filter((currentLine, index) => {
            const nextLine = asm[index + 1];
            const currentIsLabel = this.isLabel(currentLine, labelDefinitions);
            const nextIsLabel = nextLine && this.isLabel(nextLine, labelDefinitions);
            const isOrphaned = currentIsLabel && (nextIsLabel || !nextLine);
            return !isOrphaned;
        });

        return result;
    }

    /**
     * Current source mappings from RISC-V only map to the Yul line numbers. When
     * a Solidity source file is used, the mappings shown in CE are thus misleading.
     */
    private maybeRemoveSourceMappings(result: ParsedAsmResult, filters?: ParseFiltersAndOutputOptions): void {
        const inputIsSolidity = this.inputIs(InputKind.Solidity);
        const {asm, labelDefinitions} = result;

        if (this.outputIs(OutputKind.RiscV, filters)) {
            for (const line of asm) {
                if (inputIsSolidity) {
                    line.source = null;
                }
                if (!this.isLabel(line, labelDefinitions)) {
                    line.text = '       ' + line.text;
                }
            }
        }
    }

    /**
     * Whether the parsed asm result line represents a label.
     */
    private isLabel(line: ParsedAsmResultLine, labelDefinitions: ParsedAsmResult['labelDefinitions']): boolean {
        return line.text.endsWith(':') && !!labelDefinitions && line.text.slice(0, -1) in labelDefinitions;
    }

    /**
     * Whether the provided input kind matches the language used.
     */
    private inputIs(kind: InputKind): boolean {
        return this.lang.id === kind.valueOf();
    }

    /**
     * Whether the provided output kind matches the output requested.
     */
    private outputIs(kind: OutputKind, filters?: ParseFiltersAndOutputOptions): boolean {
        switch (kind) {
            case OutputKind.PolkaVM:
                return !filters?.binaryObject;
            case OutputKind.RiscV:
                return !!filters?.binaryObject;
            default:
                throw new Error('Unexpected output kind.');
        }
    }

    /**
     * Reinterpret the user-provided binary object filter to show the PolkaVM
     * assembly if selected, otherwise the RISC-V assembly.
     *
     * Users who select "Compile to binary object" should see the disassembled
     * PVM plob and not RISC-V. However, to see the RISC-V output, the binary
     * object filter needs to be reset to `true` in order to pass the binary
     * object (which will already exist after the first compilation) to the
     * objdumper during post-processing of the compilation result.
     */
    private reinterpretBinaryObjectFilter(binaryObjectFilter?: boolean): boolean {
        return !binaryObjectFilter;
    }

    /**
     * Get the Solidity contract name used in the compile file.
     *
     * @example
     * ```solidity
     * contract Square { ... } // Name = Square
     * ```
     */
    private getSolidityContractName(dirPath: string): string {
        const nameRe = /contract[\s\n]+(?<name>[\w$]+)[\s\n]*{/;

        return this.getContractName(dirPath, nameRe);
    }

    /**
     * Get the Yul contract name used in the compile file.
     *
     * @example
     * ```
     * object "Square" { ... } // Name = Square
     * ```
     */
    private getYulContractName(dirPath: string): string {
        const nameRe = /object[\s\n]+"(?<name>[\w$.]+)"[\s\n]*{/;

        return this.getContractName(dirPath, nameRe);
    }

    private getContractName(dirPath: string, nameRe: RegExp): string {
        const source = fs.readFileSync(path.join(dirPath, this.compileFilename), {encoding: 'utf8'});
        const match = source.match(nameRe);
        assert(match?.groups?.name, 'Expected to find a contract name in the source file.');

        return match.groups.name;
    }

    private addOutputHeader(result: ParsedAsmResult, filters?: ParseFiltersAndOutputOptions): void {
        const pvmHeader =
            '// PolkaVM Assembly:\n' +
            '// --------------------------\n' +
            '// To see the RISC-V assembly instead,\n' +
            '// disable "Compile to binary object".\n' +
            '// --------------------------';

        const riscvHeader =
            '; RISC-V (64 bits) Assembly:\n' +
            '; --------------------------\n' +
            '; To see the PolkaVM assembly instead,\n' +
            '; enable "Compile to binary object".\n' +
            '; --------------------------';

        const header = this.outputIs(OutputKind.PolkaVM, filters) ? pvmHeader : riscvHeader;
        result.asm.unshift(...header.split('\n').map(line => ({text: line})));
    }
}
