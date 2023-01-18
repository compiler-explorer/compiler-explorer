// Copyright (c) 2021, Compiler Explorer Authors
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
import _ from 'underscore';

import {CompilationResult, ExecutionOptions} from '../../types/compilation/compilation.interfaces';
import {
    BasicExecutionResult,
    ExecutableExecutionOptions,
    UnprocessedExecResult,
} from '../../types/execution/execution.interfaces';
import {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces';
import {BaseCompiler} from '../base-compiler';
import * as exec from '../exec';
import {DotNetAsmParser} from '../parsers/asm-parser-dotnet';
import * as utils from '../utils';

class DotNetCompiler extends BaseCompiler {
    private readonly sdkBaseDir: string;
    private readonly sdkVersion: string;
    private readonly targetFramework: string;
    private readonly buildConfig: string;
    private readonly clrBuildDir: string;
    private readonly langVersion: string;
    private readonly crossgen2Path: string;

    private crossgen2VersionString: string;

    constructor(compilerInfo, env) {
        super(compilerInfo, env);

        this.sdkBaseDir = path.join(path.dirname(compilerInfo.exe), 'sdk');
        this.sdkVersion = fs.readdirSync(this.sdkBaseDir)[0];

        const parts = this.sdkVersion.split('.');
        this.targetFramework = `net${parts[0]}.${parts[1]}`;

        this.buildConfig = this.compilerProps<string>(`compiler.${this.compiler.id}.buildConfig`);
        this.clrBuildDir = this.compilerProps<string>(`compiler.${this.compiler.id}.clrDir`);
        this.langVersion = this.compilerProps<string>(`compiler.${this.compiler.id}.langVersion`);

        this.crossgen2Path = path.join(this.clrBuildDir, 'crossgen2', 'crossgen2');
        this.asm = new DotNetAsmParser();
        this.crossgen2VersionString = '';
    }

    get compilerOptions() {
        return ['build', '-c', this.buildConfig, '-v', 'q', '--nologo', '--no-restore', '/clp:NoSummary'];
    }

    get configurableOptions() {
        return [
            '--targetos',
            '--targetarch',
            '--instruction-set',
            '--singlemethodtypename',
            '--singlemethodname',
            '--singlemethodindex',
            '--singlemethodgenericarg',
            '--codegenopt',
            '--codegen-options',
        ];
    }

    get configurableSwitches() {
        return [
            '-O',
            '--optimize',
            '--Od',
            '--optimize-disabled',
            '--Os',
            '--optimize-space',
            '--Ot',
            '--optimize-time',
        ];
    }

    async writeProjectfile(programDir: string, compileToBinary: boolean, sourceFile: string) {
        const projectFileContent = `<Project Sdk="Microsoft.NET.Sdk">
            <PropertyGroup>
                <TargetFramework>${this.targetFramework}</TargetFramework>
                <AllowUnsafeBlocks>true</AllowUnsafeBlocks>
                <AssemblyName>CompilerExplorer</AssemblyName>
                <LangVersion>${this.langVersion}</LangVersion>
                <EnableDefaultCompileItems>false</EnableDefaultCompileItems>
                <Nullable>enable</Nullable>
                <OutputType>${compileToBinary ? 'Exe' : 'Library'}</OutputType>
            </PropertyGroup>
            <ItemGroup>
                <Compile Include="${sourceFile}" />
            </ItemGroup>
        </Project>
        `;

        const projectFilePath = path.join(programDir, `CompilerExplorer${this.lang.extensions[0]}proj`);
        await fs.writeFile(projectFilePath, projectFileContent);
    }

    setCompilerExecOptions(execOptions: ExecutionOptions, programDir: string) {
        if (!execOptions) {
            execOptions = this.getDefaultExecOptions();
        }

        // See https://github.com/dotnet/runtime/issues/50391 - the .NET runtime tries to make a 2TB memfile if we have
        // this feature enabled (which is on by default on .NET 7) This blows out our nsjail sandbox limit, so for now
        // we disable it.
        execOptions.env.DOTNET_EnableWriteXorExecute = '0';
        // Disable any phone-home.
        execOptions.env.DOTNET_CLI_TELEMETRY_OPTOUT = 'true';
        // Some versions of .NET complain if they can't work out what the user's directory is. We force it to the output
        // directory here.
        execOptions.env.DOTNET_CLI_HOME = programDir;
        execOptions.env.DOTNET_ROOT = path.join(this.clrBuildDir, '.dotnet');
        // Place nuget packages in the output directory.
        execOptions.env.NUGET_PACKAGES = path.join(programDir, '.nuget');
        // Try to be less chatty
        execOptions.env.DOTNET_SKIP_FIRST_TIME_EXPERIENCE = 'true';
        execOptions.env.DOTNET_NOLOGO = 'true';

        execOptions.customCwd = programDir;
    }

    override async buildExecutable(compiler, options, inputFilename, execOptions) {
        const dirPath = path.dirname(inputFilename);
        const inputFilenameSafe = this.filename(inputFilename);
        const sourceFile = path.basename(inputFilenameSafe);
        await this.writeProjectfile(dirPath, true, sourceFile);
        return await this.buildToDll(compiler, options, inputFilename, execOptions);
    }

    override async doCompilation(inputFilename, dirPath, key, options, filters, backendOptions, libraries, tools) {
        const inputFilenameSafe = this.filename(inputFilename);
        const sourceFile = path.basename(inputFilenameSafe);
        await this.writeProjectfile(dirPath, filters.binary, sourceFile);
        return super.doCompilation(inputFilename, dirPath, key, options, filters, backendOptions, libraries, tools);
    }

    async buildToDll(
        compiler: string,
        options: string[],
        inputFilename: string,
        execOptions: ExecutionOptions,
    ): Promise<CompilationResult> {
        const programDir = path.dirname(inputFilename);
        const nugetConfigPath = path.join(programDir, 'nuget.config');
        const nugetConfigFileContent = `<?xml version="1.0" encoding="utf-8"?>
        <configuration>
            <packageSources>
                <clear />
            </packageSources>
        </configuration>
        `;

        await fs.writeFile(nugetConfigPath, nugetConfigFileContent);

        this.setCompilerExecOptions(execOptions, programDir);
        const restoreOptions = ['restore', '--configfile', nugetConfigPath, '-v', 'q', '--nologo', '/clp:NoSummary'];
        const restoreResult = await this.exec(compiler, restoreOptions, execOptions);
        if (restoreResult.code !== 0) {
            return this.transformToCompilationResult(restoreResult, inputFilename);
        }

        const compilerResult = await super.runCompiler(compiler, this.compilerOptions, inputFilename, execOptions);
        if (compilerResult.code === 0) {
            await fs.createFile(this.getOutputFilename(programDir, this.outputFilebase));
        }
        return compilerResult;
    }

    override async runCompiler(
        compiler: string,
        options: string[],
        inputFilename: string,
        execOptions: ExecutionOptions,
    ): Promise<CompilationResult> {
        const crossgen2Options: string[] = [];
        const configurableOptions = this.configurableOptions;
        const programDir = path.dirname(inputFilename);
        const programOutputPath = path.join(programDir, 'bin', this.buildConfig, this.targetFramework);
        const programDllPath = path.join(programOutputPath, 'CompilerExplorer.dll');

        for (const configurableOption of configurableOptions) {
            const optionIndex = options.indexOf(configurableOption);
            if (optionIndex === -1 || optionIndex === options.length - 1) {
                continue;
            }
            crossgen2Options.push(options[optionIndex], options[optionIndex + 1]);
        }

        const configurableSwitches = this.configurableSwitches;
        for (const configurableSwitch of configurableSwitches) {
            const switchIndex = options.indexOf(configurableSwitch);
            if (switchIndex === -1) {
                continue;
            }
            crossgen2Options.push(options[switchIndex]);
        }

        this.setCompilerExecOptions(execOptions, programDir);
        const compilerResult = await this.buildToDll(compiler, options, inputFilename, execOptions);
        if (compilerResult.code !== 0) {
            return compilerResult;
        }

        const crossgen2Result = await this.runCrossgen2(
            execOptions,
            this.clrBuildDir,
            programDllPath,
            crossgen2Options,
            this.getOutputFilename(programDir, this.outputFilebase),
        );

        if (crossgen2Result.code !== 0) {
            return crossgen2Result;
        }

        return compilerResult;
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions) {
        return this.compilerOptions;
    }

    override async execBinary(
        executable: string,
        maxSize: number | undefined,
        executeParameters: ExecutableExecutionOptions,
        homeDir: string | undefined,
    ): Promise<BasicExecutionResult> {
        const programDir = path.dirname(executable);
        const programOutputPath = path.join(programDir, 'bin', this.buildConfig, this.targetFramework);
        const programDllPath = path.join(programOutputPath, 'CompilerExplorer.dll');
        const execOptions = this.getDefaultExecOptions();
        execOptions.maxOutput = maxSize;
        execOptions.timeoutMs = this.env.ceProps('binaryExecTimeoutMs', 2000);
        execOptions.ldPath = _.union(this.compiler.ldPath, executeParameters.ldPath);
        execOptions.customCwd = homeDir;
        execOptions.appHome = homeDir;
        execOptions.env = executeParameters.env;
        execOptions.env.DOTNET_EnableWriteXorExecute = '0';
        execOptions.env.DOTNET_CLI_HOME = programDir;
        execOptions.env.CORE_ROOT = this.clrBuildDir;
        execOptions.input = executeParameters.stdin;
        const execArgs = ['-p', 'System.Runtime.TieredCompilation=false', programDllPath, ...executeParameters.args];
        const corerun = path.join(this.clrBuildDir, 'corerun');
        try {
            const execResult: UnprocessedExecResult = await exec.sandbox(corerun, execArgs, execOptions);
            return this.processExecutionResult(execResult);
        } catch (err: UnprocessedExecResult | any) {
            if (err.code && err.stderr) {
                return this.processExecutionResult(err);
            } else {
                return {
                    ...this.getEmptyExecutionResult(),
                    stdout: err.stdout ? utils.parseOutput(err.stdout) : [],
                    stderr: err.stderr ? utils.parseOutput(err.stderr) : [],
                    code: err.code === undefined ? -1 : err.code,
                };
            }
        }
    }

    async ensureCrossgen2Version(execOptions) {
        if (!this.crossgen2VersionString) {
            this.crossgen2VersionString = '// crossgen2 ';

            const versionFilePath = `${this.clrBuildDir}/version.txt`;
            const versionResult = await this.exec(this.crossgen2Path, ['--version'], execOptions);
            if (versionResult.code === 0) {
                this.crossgen2VersionString += versionResult.stdout;
            } else if (fs.existsSync(versionFilePath)) {
                const versionString = await fs.readFile(versionFilePath);
                this.crossgen2VersionString += versionString;
            } else {
                this.crossgen2VersionString += '<unknown version>';
            }
        }
    }

    async runCrossgen2(execOptions, bclPath, dllPath, options, outputPath) {
        await this.ensureCrossgen2Version(execOptions);

        const crossgen2Options = [
            '-r',
            path.join(bclPath, '/'),
            dllPath,
            '-o',
            'CompilerExplorer.r2r.dll',
            '--codegenopt',
            'NgenDisasm=*',
            '--codegenopt',
            'JitDisasm=*',
            '--codegenopt',
            'JitDiffableDasm=1',
            '--parallelism',
            '1',
            '--inputbubble',
            '--compilebubblegenerics',
        ].concat(options);

        const compilerExecResult = await this.exec(this.crossgen2Path, crossgen2Options, execOptions);
        const result = this.transformToCompilationResult(compilerExecResult, dllPath);

        await fs.writeFile(
            outputPath,
            `${this.crossgen2VersionString}\n\n${result.stdout.map(o => o.text).reduce((a, n) => `${a}\n${n}`, '')}`,
        );

        return result;
    }
}

export class CSharpCompiler extends DotNetCompiler {
    static get key() {
        return 'csharp';
    }
}

export class FSharpCompiler extends DotNetCompiler {
    static get key() {
        return 'fsharp';
    }
}

export class VBCompiler extends DotNetCompiler {
    static get key() {
        return 'vb';
    }
}
