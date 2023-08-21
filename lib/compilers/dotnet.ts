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

import type {CompilationResult, ExecutionOptions} from '../../types/compilation/compilation.interfaces.js';
import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {
    BasicExecutionResult,
    ExecutableExecutionOptions,
    UnprocessedExecResult,
} from '../../types/execution/execution.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';
import * as exec from '../exec.js';
import {DotNetAsmParser} from '../parsers/asm-parser-dotnet.js';
import * as utils from '../utils.js';

class DotNetCompiler extends BaseCompiler {
    private readonly sdkBaseDir: string;
    private readonly sdkVersion: string;
    private readonly targetFramework: string;
    private readonly buildConfig: string;
    private readonly clrBuildDir: string;
    private readonly langVersion: string;
    private readonly corerunPath: string;
    private readonly disassemblyLoaderPath: string;

    private versionString: string;

    constructor(compilerInfo: PreliminaryCompilerInfo, env) {
        super(compilerInfo, env);

        this.sdkBaseDir = path.join(path.dirname(compilerInfo.exe), 'sdk');
        this.sdkVersion = fs.readdirSync(this.sdkBaseDir)[0];

        const parts = this.sdkVersion.split('.');
        this.targetFramework = `net${parts[0]}.${parts[1]}`;

        this.buildConfig = this.compilerProps<string>(`compiler.${this.compiler.id}.buildConfig`);
        this.clrBuildDir = this.compilerProps<string>(`compiler.${this.compiler.id}.clrDir`);
        this.langVersion = this.compilerProps<string>(`compiler.${this.compiler.id}.langVersion`);

        this.corerunPath = path.join(this.clrBuildDir, 'corerun');
        this.asm = new DotNetAsmParser();
        this.versionString = '';
        this.disassemblyLoaderPath = path.join(this.clrBuildDir, 'DisassemblyLoader', 'DisassemblyLoader.dll');
    }

    get compilerOptions() {
        return ['build', '-c', this.buildConfig, '-v', 'q', '--nologo', '--no-restore', '/clp:NoSummary'];
    }

    async initializeAssemblyLoader() {}

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
                <Reference Include="DisassemblyLoader" HintPath="${this.disassemblyLoaderPath}" />
            </ItemGroup>
        </Project>
        `;

        const projectFilePath = path.join(programDir, `CompilerExplorer${this.lang.extensions[0]}proj`);
        await fs.writeFile(projectFilePath, projectFileContent);
    }

    setCompilerExecOptions(execOptions: ExecutionOptions & {env: Record<string, string>}, programDir: string) {
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
        return await this.buildToDll(compiler, inputFilename, execOptions);
    }

    override async doCompilation(inputFilename, dirPath, key, options, filters, backendOptions, libraries, tools) {
        const inputFilenameSafe = this.filename(inputFilename);
        const sourceFile = path.basename(inputFilenameSafe);
        await this.writeProjectfile(dirPath, filters.binary, sourceFile);
        return super.doCompilation(inputFilename, dirPath, key, options, filters, backendOptions, libraries, tools);
    }

    async buildToDll(
        compiler: string,
        inputFilename: string,
        execOptions: ExecutionOptions & {env: Record<string, string>},
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
        execOptions: ExecutionOptions & {env: Record<string, string>},
    ): Promise<CompilationResult> {
        const corerunOptions: string[] = [];
        const programDir = path.dirname(inputFilename);
        const programOutputPath = path.join(programDir, 'bin', this.buildConfig, this.targetFramework);
        const programDllPath = path.join(programOutputPath, 'CompilerExplorer.dll');
        const envVarFileContents = [
            'DOTNET_EnableWriteXorExecute=0',
            'DOTNET_JitDisasm=*',
            'DOTNET_JitDisasmAssemblies=CompilerExplorer',
            'DOTNET_TieredCompilation=0',
        ];

        while (options.length > 0) {
            const currentOption = options.shift();
            if (currentOption === '-e' || currentOption === '--env') {
                const envVar = options.shift();
                if (envVar) {
                    const [name] = envVar.split('=');
                    const normalizedName = name.trim().toUpperCase();
                    if (
                        normalizedName === 'DOTNET_JITDISASM' ||
                        normalizedName === 'DOTNET_JITDUMP' ||
                        normalizedName === 'DOTNET_JITDISASMASSEMBILES'
                    ) {
                        continue;
                    }
                    envVarFileContents.push(envVar);
                }
            } else if (currentOption === '-p' || currentOption === '--property') {
                const property = options.shift();
                if (property) {
                    corerunOptions.push('-p', property);
                }
            }
        }

        this.setCompilerExecOptions(execOptions, programDir);
        const compilerResult = await this.buildToDll(compiler, inputFilename, execOptions);
        if (compilerResult.code !== 0) {
            return compilerResult;
        }

        const envVarFilePath = path.join(programDir, '.env');
        await fs.writeFile(envVarFilePath, envVarFileContents.join('\n'));

        const corerunResult = await this.runCorerunForDisasm(
            execOptions,
            this.clrBuildDir,
            envVarFilePath,
            programDllPath,
            corerunOptions,
            this.getOutputFilename(programDir, this.outputFilebase),
        );

        if (corerunResult.code !== 0) {
            return corerunResult;
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
        try {
            const execResult: UnprocessedExecResult = await exec.sandbox(this.corerunPath, execArgs, execOptions);
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

    async checkRuntimeVersion(execOptions: ExecutionOptions) {
        if (!this.versionString) {
            this.versionString = '// dotnet runtime ';

            const versionFilePath = `${this.clrBuildDir}/version.txt`;
            if (fs.existsSync(versionFilePath)) {
                const versionString = await fs.readFile(versionFilePath);
                this.versionString += versionString;
            } else {
                this.versionString += '<unknown version>';
            }
        }
    }

    async runCorerunForDisasm(
        execOptions: ExecutionOptions,
        coreRoot: string,
        envPath: string,
        dllPath: string,
        options: string[],
        outputPath: string,
    ) {
        await this.checkRuntimeVersion(execOptions);

        const corerunOptions = ['--clr-path', coreRoot, '--env', envPath].concat([
            ...options,
            this.disassemblyLoaderPath,
            dllPath,
        ]);
        const compilerExecResult = await this.exec(this.corerunPath, corerunOptions, execOptions);
        const result = this.transformToCompilationResult(compilerExecResult, dllPath);

        await fs.writeFile(
            outputPath,
            `${this.versionString}\n\n${result.stdout.map(o => o.text).reduce((a, n) => `${a}\n${n}`, '')}`,
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
