// Copyright (c) 2026, Compiler Explorer Authors
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright notice,
//       this list of conditions and the following disclaimer.
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

import type {ParsedAsmResult, ParsedAsmResultLine} from '../../types/asmresult/asmresult.interfaces.js';
import type {CompilationResult, ExecutionOptionsWithEnv} from '../../types/compilation/compilation.interfaces.js';
import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {BasicExecutionResult, ExecutableExecutionOptions} from '../../types/execution/execution.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';
import {CompilationEnvironment} from '../compilation-env.js';
import type {ClientOptionsType} from '../options-handler.js';
import * as utils from '../utils.js';

const packageName = 'compiler_explorer';
const sourceFilename = `${packageName}.gleam`;
const javascriptRunnerFilename = `${packageName}_runner.mjs`;
const beamListingArguments = [
    '-noshell',
    '-eval',
    '{ok, [Input]} = init:get_argument(input),' +
        "case compile:file(Input, ['S', binary, no_line_info, report]) of " +
        '{ok, _, Output} -> beam_listing:module(group_leader(), Output); ' +
        '{ok, _, Output, _} -> beam_listing:module(group_leader(), Output); ' +
        '_ -> halt(1) end, halt().',
];

type GleamTarget = 'erlang' | 'javascript';

type TargetConfiguration = {
    displayName: string;
    buildArguments: string[];
    executableBuildArguments: string[];
    buildDirectory: GleamTarget;
    executableFilename: string;
    resultLanguageId?: string;
};

const targetConfigurations: Record<GleamTarget, TargetConfiguration> = {
    erlang: {
        displayName: 'BEAM',
        buildArguments: ['build', '--no-print-progress'],
        executableBuildArguments: ['export', 'escript'],
        buildDirectory: 'erlang',
        executableFilename: packageName,
    },
    javascript: {
        displayName: 'JavaScript',
        buildArguments: ['build', '--target', 'javascript', '--no-print-progress'],
        executableBuildArguments: ['build', '--target', 'javascript', '--no-print-progress'],
        buildDirectory: 'javascript',
        executableFilename: javascriptRunnerFilename,
        resultLanguageId: 'typescript',
    },
};

export class GleamCompiler extends BaseCompiler {
    private readonly runtime: string;
    private readonly stdlib: string;
    private readonly target: GleamTarget;

    static get key() {
        return 'gleam';
    }

    static getDisplayName(version: string, target: GleamTarget): string {
        return `Gleam ${version} (${targetConfigurations[target].displayName})`;
    }

    constructor(info: PreliminaryCompilerInfo, env: CompilationEnvironment) {
        super(info, env);
        this.runtime = this.compilerProps<string>(`compiler.${this.compiler.id}.runtime`, 'erl');
        this.stdlib = this.compilerProps<string>(`compiler.${this.compiler.id}.stdlib`);
        const target = this.compilerProps<string>(`compiler.${this.compiler.id}.target`, 'erlang');
        if (target !== 'erlang' && target !== 'javascript') {
            throw new Error(`Gleam compiler ${this.compiler.id} has unsupported target "${target}"`);
        }
        this.target = target;
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions): string[] {
        filters.binary = this.target === 'erlang';
        return [];
    }

    override getCompilerResultLanguageId(): string | undefined {
        return this.targetConfiguration.resultLanguageId;
    }

    override filterUserOptions(_userOptions: string[]): string[] {
        return [];
    }

    override async initialise(mtime: Date, clientOptions: ClientOptionsType, isPrediscovered = false) {
        const compiler = await super.initialise(mtime, clientOptions, isPrediscovered);
        if (compiler && this.compiler.version) {
            this.compiler.name = GleamCompiler.getDisplayName(this.compiler.version, this.target);
        }
        return compiler;
    }

    override async runCompiler(
        compiler: string,
        _options: string[],
        inputFilename: string,
        execOptions: ExecutionOptionsWithEnv,
        _filters?: ParseFiltersAndOutputOptions,
    ): Promise<CompilationResult> {
        const projectDir = await this.prepareProject(inputFilename);
        const buildResult = await this.compileGleamProject(compiler, projectDir, execOptions);
        const result: CompilationResult = {
            ...this.transformToCompilationResult(buildResult, inputFilename),
            languageId: this.getCompilerResultLanguageId(),
            instructionSet: this.getInstructionSetFromCompilerArgs([]),
        };

        if (result.code === 0) {
            result.asm =
                this.target === 'erlang'
                    ? await this.disassembleProject(projectDir, execOptions)
                    : await this.getJavascriptOutput(projectDir);
        }

        return result;
    }

    override async buildExecutable(
        compiler: string,
        _options: string[],
        inputFilename: string,
        execOptions: ExecutionOptionsWithEnv,
    ): Promise<CompilationResult> {
        const projectDir = await this.prepareProject(inputFilename);
        const exportResult = await this.exec(compiler, this.targetConfiguration.executableBuildArguments, {
            ...this.withOtpToolchainPath(execOptions),
            customCwd: projectDir,
        });
        if (this.target === 'javascript' && exportResult.code === 0) {
            await fs.writeFile(path.join(projectDir, javascriptRunnerFilename), this.getJavascriptRunner());
        }
        return this.transformToCompilationResult(exportResult, inputFilename);
    }

    override getExecutableFilename(dirPath: string): string {
        return path.join(dirPath, this.targetConfiguration.executableFilename);
    }

    override async runExecutable(
        executable: string,
        executeParameters: ExecutableExecutionOptions,
        homeDir: string,
    ): Promise<BasicExecutionResult> {
        return await super.runExecutable(
            executable,
            this.target === 'erlang' ? this.withOtpLauncherPath(executeParameters) : executeParameters,
            homeDir,
        );
    }

    override async checkOutputFileAndDoPostProcess(asmResult: CompilationResult): Promise<[any, any[], any[]]> {
        return [asmResult, [], []];
    }

    override async processAsm(result: CompilationResult): Promise<ParsedAsmResult> {
        if (result.code !== 0) {
            return {asm: [{text: '<Compilation failed>', source: null}]};
        }
        return {asm: (result.asm as ParsedAsmResultLine[] | undefined) ?? []};
    }

    private async prepareProject(inputFilename: string): Promise<string> {
        if (!this.stdlib) {
            throw new Error(`Gleam compiler ${this.compiler.id} has no configured standard library path`);
        }

        const projectDir = path.dirname(inputFilename);
        const sourceDir = path.join(projectDir, 'src');
        await fs.mkdir(sourceDir, {recursive: true});
        await fs.copyFile(inputFilename, this.projectSourcePath(projectDir));
        await fs.writeFile(path.join(projectDir, 'gleam.toml'), this.getGleamToml());
        return projectDir;
    }

    private get targetConfiguration(): TargetConfiguration {
        return targetConfigurations[this.target];
    }

    private async compileGleamProject(compiler: string, projectDir: string, execOptions: ExecutionOptionsWithEnv) {
        return await this.exec(compiler, this.targetConfiguration.buildArguments, {
            ...this.withOtpToolchainPath(execOptions),
            customCwd: projectDir,
        });
    }

    private withOtpToolchainPath(execOptions: ExecutionOptionsWithEnv): ExecutionOptionsWithEnv {
        if (!path.isAbsolute(this.runtime)) return execOptions;

        return {
            ...execOptions,
            env: {
                ...execOptions.env,
                PATH: [path.dirname(this.runtime), execOptions.env.PATH].filter(Boolean).join(path.delimiter),
            },
        };
    }

    private withOtpLauncherPath(executeParameters: ExecutableExecutionOptions): ExecutableExecutionOptions {
        return {
            ...executeParameters,
            env: {
                ...executeParameters.env,
                PATH: [executeParameters.env.PATH, this.getDefaultExecOptions().env.PATH]
                    .filter(Boolean)
                    .join(path.delimiter),
            },
        };
    }

    private projectSourcePath(projectDir: string): string {
        return path.join(projectDir, 'src', sourceFilename);
    }

    private targetBuildPath(projectDir: string): string {
        return path.join(projectDir, 'build', 'dev', this.targetConfiguration.buildDirectory, packageName);
    }

    private getGleamToml(): string {
        return `name = "${packageName}"
version = "1.0.0"

[dependencies]
gleam_stdlib = { path = ${JSON.stringify(this.stdlib)} }
`;
    }

    private getJavascriptRunner(): string {
        return `import {main} from './build/dev/javascript/${packageName}/${packageName}.mjs';\n\nawait main();\n`;
    }

    private async getJavascriptOutput(projectDir: string): Promise<ParsedAsmResultLine[]> {
        const outputPath = path.join(this.targetBuildPath(projectDir), `${packageName}.mjs`);
        try {
            return utils.splitLines(await fs.readFile(outputPath, 'utf8')).map(text => ({text, source: null}));
        } catch (error) {
            return [{text: `<Unable to find generated Gleam JavaScript: ${(error as Error).message}>`, source: null}];
        }
    }

    private async disassembleProject(
        projectDir: string,
        execOptions: ExecutionOptionsWithEnv,
    ): Promise<ParsedAsmResultLine[]> {
        const artefactsPath = path.join(this.targetBuildPath(projectDir), '_gleam_artefacts');
        let sourceFiles: string[];
        try {
            sourceFiles = (await fs.readdir(artefactsPath))
                .filter(filename => filename.endsWith('.erl') && !filename.endsWith('@@main.erl'))
                .sort();
        } catch (error) {
            return [
                {text: `<Unable to find generated Gleam Erlang source: ${(error as Error).message}>`, source: null},
            ];
        }

        if (sourceFiles.length === 0) {
            return [{text: '<Gleam produced no Erlang modules>', source: null}];
        }

        const listings = await Promise.all(
            sourceFiles.map(sourceFile =>
                this.getBeamListing(path.join(artefactsPath, sourceFile), projectDir, execOptions),
            ),
        );

        return listings.flatMap((listing, index) =>
            listing.code === 0
                ? this.formatBeamListing(listing.stdout)
                : [{text: `<Unable to list ${sourceFiles[index]}: ${listing.stderr}>`, source: null}],
        );
    }

    private getBeamListing(sourcePath: string, projectDir: string, execOptions: ExecutionOptionsWithEnv) {
        return this.exec(this.runtime, [...beamListingArguments, '-input', sourcePath], {
            ...execOptions,
            customCwd: projectDir,
        });
    }

    private formatBeamListing(listing: string): ParsedAsmResultLine[] {
        return this.withoutModuleInfo(this.withoutMetadata(listing.split('\n'))).map(text => ({
            text: utils.maskRootdir(text),
            source: null,
        }));
    }

    private withoutMetadata(lines: string[]): string[] {
        const withoutMetadata: string[] = [];
        let braceDepth = 0;
        for (const line of lines) {
            if (braceDepth > 0) {
                braceDepth += this.getBraceDelta(line);
                continue;
            }
            // The BEAM compiler represents variable/type metadata as a brace-balanced `{'%', ...}` term.
            if (/^\s*\{'%',/.test(line)) {
                braceDepth = this.getBraceDelta(line);
                continue;
            }
            withoutMetadata.push(line);
        }
        return withoutMetadata;
    }

    private withoutModuleInfo(lines: string[]): string[] {
        const withoutModuleInfo: string[] = [];
        let moduleInfo = false;
        for (const line of lines) {
            // `module_info/0` and `/1` are implicit BEAM reflection functions, not Gleam code.
            if (/^\{function, module_info, [01], \d+\}\.$/.test(line)) {
                moduleInfo = true;
                continue;
            }
            if (moduleInfo && line.startsWith('{function,')) moduleInfo = false;
            if (!moduleInfo) withoutModuleInfo.push(line);
        }
        return withoutModuleInfo;
    }

    private getBraceDelta(text: string): number {
        return (text.match(/\{/g)?.length ?? 0) - (text.match(/\}/g)?.length ?? 0);
    }
}
