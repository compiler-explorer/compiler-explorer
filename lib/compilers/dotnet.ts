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

import type {CompilationResult, ExecutionOptions} from '../../types/compilation/compilation.interfaces.js';
import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import {ExecutableExecutionOptions} from '../../types/execution/execution.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';
import {AssemblyName, DotnetExtraConfiguration} from '../execution/dotnet-execution-env.js';
import {IExecutionEnvironment} from '../execution/execution-env.interfaces.js';
import {DotNetAsmParser} from '../parsers/asm-parser-dotnet.js';

class DotNetCompiler extends BaseCompiler {
    private readonly sdkBaseDir: string;
    private readonly sdkVersion: string;
    private readonly targetFramework: string;
    private readonly buildConfig: string;
    private readonly clrBuildDir: string;
    private readonly langVersion: string;
    private readonly corerunPath: string;
    private readonly disassemblyLoaderPath: string;
    private readonly crossgen2Path: string;
    private readonly ilcPath: string;
    private readonly sdkMajorVersion: number;

    constructor(compilerInfo: PreliminaryCompilerInfo, env) {
        super(compilerInfo, env);

        this.sdkBaseDir = path.join(path.dirname(compilerInfo.exe), 'sdk');
        this.sdkVersion = fs.readdirSync(this.sdkBaseDir)[0];

        const parts = this.sdkVersion.split('.');
        this.targetFramework = `net${parts[0]}.${parts[1]}`;
        this.sdkMajorVersion = Number(parts[0]);

        this.buildConfig = this.compilerProps<string>(`compiler.${this.compiler.id}.buildConfig`);
        this.clrBuildDir = this.compilerProps<string>(`compiler.${this.compiler.id}.clrDir`);
        this.langVersion = this.compilerProps<string>(`compiler.${this.compiler.id}.langVersion`);

        this.corerunPath = path.join(this.clrBuildDir, 'corerun');
        this.crossgen2Path = path.join(this.clrBuildDir, 'crossgen2', 'crossgen2.dll');
        this.ilcPath = path.join(this.clrBuildDir, 'ilc-published', 'ilc');
        this.asm = new DotNetAsmParser();
        this.disassemblyLoaderPath = path.join(this.clrBuildDir, 'DisassemblyLoader', 'DisassemblyLoader.dll');
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
            '--maxgenericcycle',
            '--maxgenericcyclebreadth',
            '--max-vectort-bitwidth',
            '--runtimeopt',
            '--runtimeknob',
            '--feature',
            '--directpinvoke',
            '--root',
            '--conditionalroot',
            '--trim',
            '--jitpath',
            '--generateunmanagedentrypoints',
            '--guard',
            '--initassembly',
            '--reflectiondata',
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
            '--enable-generic-cycle-detection',
            '--inputbubble',
            '--compilebubblegenerics',
            '--aot',
            '--crossgen2',
            '--dehydrate',
            '--methodbodyfolding',
            '--stacktracedata',
            '--defaultrooting',
            '--preinitstatics',
            '--nopreinitstatics',
            '--scan',
            '--noscan',
            '--noinlinetls',
            '--completetypemetadata',
        ];
    }

    async writeProjectfile(programDir: string, compileToBinary: boolean, sourceFile: string) {
        const projectFileContent = `<Project Sdk="Microsoft.NET.Sdk">
            <PropertyGroup>
                <TargetFramework>${this.targetFramework}</TargetFramework>
                <AllowUnsafeBlocks>true</AllowUnsafeBlocks>
                <AssemblyName>${AssemblyName}</AssemblyName>
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

        const projectFilePath = path.join(programDir, `${AssemblyName}${this.lang.extensions[0]}proj`);
        await fs.writeFile(projectFilePath, projectFileContent);
    }

    setCompilerExecOptions(
        execOptions: ExecutionOptions & {env: Record<string, string>},
        programDir: string,
        skipNuget: boolean = false,
    ) {
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
        // Try to be less chatty
        execOptions.env.DOTNET_SKIP_FIRST_TIME_EXPERIENCE = 'true';
        execOptions.env.DOTNET_NOLOGO = 'true';

        execOptions.maxOutput = 1024 * 1024 * 1024;

        execOptions.customCwd = programDir;

        if (!skipNuget) {
            // Place nuget packages in the output directory.
            execOptions.env.NUGET_PACKAGES = path.join(programDir, '.nuget');
        }
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
        filters: ParseFiltersAndOutputOptions,
    ): Promise<CompilationResult> {
        const corerunArgs: string[] = [];
        // prettier-ignore
        const toolOptions: string[] = [
            '--codegenopt',
            this.sdkMajorVersion === 6 ? 'NgenDisasm=*' : 'JitDisasm=*',
            '--parallelism', '1',
        ];
        const toolSwitches: string[] = [];
        const programDir = path.dirname(inputFilename);
        const programOutputPath = path.join(programDir, 'bin', this.buildConfig, this.targetFramework);
        const programDllPath = path.join(programOutputPath, 'CompilerExplorer.dll');
        // prettier-ignore
        const envVarFileContents = [
            'DOTNET_EnableWriteXorExecute=0',
            'DOTNET_JitDisasm=*',
            'DOTNET_JitDisasmAssemblies=CompilerExplorer',
            'DOTNET_TieredCompilation=0',
        ];
        let overrideDiffable = false;
        let isAot = this.compiler.group === 'dotnetnativeaot';
        let isMono = this.compiler.group === 'dotnetmono';
        let isCrossgen2 = this.compiler.group === 'dotnetlegacy' && this.sdkMajorVersion === 6;

        while (options.length > 0) {
            const currentOption = options.shift();
            if (!currentOption) {
                continue;
            }

            if (currentOption === '-e' || currentOption === '--env') {
                const envVar = options.shift();
                if (envVar) {
                    const [name] = envVar.split('=');
                    const normalizedName = name.trim().toUpperCase();
                    if (normalizedName === 'DOTNET_JITDISASM' || normalizedName === 'DOTNET_JITDISASMASSEMBILES') {
                        continue;
                    }
                    if (normalizedName === 'DOTNET_JITDIFFABLEDASM' || normalizedName === 'DOTNET_JITDISASMDIFFABLE') {
                        overrideDiffable = true;
                    }
                    envVarFileContents.push(envVar);
                }
            } else if (currentOption === '-p' || currentOption === '--property') {
                const property = options.shift();
                if (property) {
                    corerunArgs.push('-p', property);
                }
            } else if (this.configurableSwitches.includes(currentOption)) {
                if (this.compiler.group === 'dotnetlegacy') {
                    if (currentOption === '--aot') {
                        isAot = true;
                    } else if (currentOption === '--crossgen2') {
                        isCrossgen2 = true;
                    } else if (currentOption === '--mono') {
                        isMono = true;
                    }
                } else {
                    toolSwitches.push(currentOption);
                }
            } else if (this.configurableOptions.includes(currentOption)) {
                const value = options.shift();
                if (value) {
                    toolOptions.push(currentOption, value);
                    const normalizedValue = value.trim().toUpperCase();
                    if (
                        (currentOption === '--codegenopt' || currentOption === '--codegen-options') &&
                        (normalizedValue.startsWith('JITDIFFABLEDASM=') ||
                            normalizedValue.startsWith('JITDISASMDIFFABLE='))
                    ) {
                        overrideDiffable = true;
                    }
                }
            }
        }

        if (this.compiler.group === 'dotnetcrossgen2') {
            isCrossgen2 = true;
        }

        if (!overrideDiffable) {
            if (this.sdkMajorVersion < 8) {
                toolOptions.push('--codegenopt', 'JitDiffableDasm=1');
                envVarFileContents.push('DOTNET_JitDiffableDasm=1');
            }
        }

        this.setCompilerExecOptions(execOptions, programDir);

        const compilerResult = await this.buildToDll(compiler, inputFilename, execOptions);
        if (compilerResult.code !== 0) {
            return compilerResult;
        }

        if (isCrossgen2) {
            const crossgen2Result = await this.runCrossgen2(
                compiler,
                execOptions,
                this.clrBuildDir,
                programDllPath,
                toolOptions,
                toolSwitches,
                this.getOutputFilename(programDir, this.outputFilebase),
            );

            if (crossgen2Result.code !== 0) {
                return crossgen2Result;
            }
        } else if (isAot) {
            const ilcResult = await this.runIlc(
                this.ilcPath,
                execOptions,
                programDllPath,
                toolOptions,
                toolSwitches,
                this.getOutputFilename(programDir, this.outputFilebase),
                filters.binary ?? false,
            );

            if (ilcResult.code !== 0) {
                return ilcResult;
            }
        } else {
            const envVarFilePath = path.join(programDir, '.env');
            await fs.writeFile(envVarFilePath, envVarFileContents.join('\n'));

            const corerunResult = await this.runCorerunForDisasm(
                execOptions,
                this.clrBuildDir,
                envVarFilePath,
                programDllPath,
                corerunArgs,
                this.getOutputFilename(programDir, this.outputFilebase),
                isMono,
            );

            if (corerunResult.code !== 0) {
                return corerunResult;
            }
        }

        return compilerResult;
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions) {
        return this.compilerOptions;
    }

    async getRuntimeVersion() {
        const versionFilePath = `${this.clrBuildDir}/version.txt`;
        if (fs.existsSync(versionFilePath)) {
            const versionString = await fs.readFile(versionFilePath);
            return versionString.toString();
        } else {
            return '<unknown version>';
        }
    }

    async runCorerunForDisasm(
        execOptions: ExecutionOptions,
        coreRoot: string,
        envPath: string,
        dllPath: string,
        options: string[],
        outputPath: string,
        isMono: boolean,
    ) {
        if (isMono) {
            coreRoot = path.join(coreRoot, 'mono');
        }

        const corerunOptions = ['--clr-path', coreRoot, '--env', envPath].concat([
            ...options,
            this.disassemblyLoaderPath,
            dllPath,
        ]);
        const compilerExecResult = await this.exec(this.corerunPath, corerunOptions, execOptions);
        const result = this.transformToCompilationResult(compilerExecResult, dllPath);

        await fs.writeFile(
            outputPath,
            `// ${isMono ? 'mono' : 'coreclr'} ${await this.getRuntimeVersion()}\n\n${result.stdout
                .map(o => o.text)
                .reduce((a, n) => `${a}\n${n}`, '')}`,
        );

        return result;
    }

    async runCrossgen2(
        compiler: string,
        execOptions: ExecutionOptions,
        bclPath: string,
        dllPath: string,
        toolOptions: string[],
        toolSwitches: string[],
        outputPath: string,
    ) {
        // prettier-ignore
        const crossgen2Options = [
            this.crossgen2Path,
            '-r', path.join(bclPath, '/'),
            '-r', this.disassemblyLoaderPath,
            dllPath,
            '-o', `${AssemblyName}.r2r.dll`,
        ].concat(toolOptions).concat(toolSwitches);

        const compilerExecResult = await this.exec(compiler, crossgen2Options, execOptions);
        const result = this.transformToCompilationResult(compilerExecResult, dllPath);

        await fs.writeFile(
            outputPath,
            `// crossgen2 ${await this.getRuntimeVersion()}\n\n${result.stdout
                .map(o => o.text)
                .reduce((a, n) => `${a}\n${n}`, '')}`,
        );

        return result;
    }

    async runIlc(
        compiler: string,
        execOptions: ExecutionOptions,
        dllPath: string,
        toolOptions: string[],
        toolSwitches: string[],
        outputPath: string,
        buildToBinary: boolean,
    ) {
        // prettier-ignore
        const ilcOptions = [
            dllPath,
            '-o', `${AssemblyName}.obj`,
            '-r', this.disassemblyLoaderPath,
            '-r', path.join(this.clrBuildDir, 'aotsdk', '*.dll'),
            '-r', path.join(this.clrBuildDir, '*.dll'),
            '--initassembly:System.Private.CoreLib',
            '--initassembly:System.Private.StackTraceMetadata',
            '--initassembly:System.Private.TypeLoader',
            '--initassembly:System.Private.Reflection.Execution',
            '--directpinvoke:libSystem.Native',
            '--directpinvoke:libSystem.Globalization.Native',
            '--directpinvoke:libSystem.IO.Compression.Native',
            '--directpinvoke:libSystem.Net.Security.Native',
            '--directpinvoke:libSystem.Security.Cryptography.Native.OpenSsl',
            '--resilient',
            '--singlewarn',
            '--scanreflection',
            '--nosinglewarnassembly:CompilerExplorer',
            '--generateunmanagedentrypoints:System.Private.CoreLib',
            '--notrimwarn',
            '--noaotwarn',
        ].concat(toolOptions).concat(toolSwitches);

        if (!buildToBinary) {
            ilcOptions.push('--nativelib');
        }

        const compilerExecResult = await this.exec(compiler, ilcOptions, execOptions);
        const result = this.transformToCompilationResult(compilerExecResult, dllPath);

        await fs.writeFile(
            outputPath,
            `// ilc ${await this.getRuntimeVersion()}\n\n${result.stdout
                .map(o => o.text)
                .reduce((a, n) => `${a}\n${n}`, '')}`,
        );

        return result;
    }

    override runExecutable(executable: string, executeParameters: ExecutableExecutionOptions, homeDir) {
        const execOptionsCopy: ExecutableExecutionOptions = JSON.parse(
            JSON.stringify(executeParameters),
        ) as ExecutableExecutionOptions;

        if (this.compiler.executionWrapper) {
            execOptionsCopy.args = [...this.compiler.executionWrapperArgs, executable, ...execOptionsCopy.args];
            executable = this.compiler.executionWrapper;
        }

        const isMono = this.compiler.group === 'dotnetmono';

        const extraConfiguration: DotnetExtraConfiguration = {
            buildConfig: this.buildConfig,
            clrBuildDir: isMono ? path.join(this.clrBuildDir, 'mono') : this.clrBuildDir,
            langVersion: this.langVersion,
            targetFramework: this.targetFramework,
            corerunPath: this.corerunPath,
        };

        const execEnv: IExecutionEnvironment = new this.executionEnvironmentClass(this.env);
        return execEnv.execBinary(executable, execOptionsCopy, homeDir, extraConfiguration);
    }
}

export class DotNetCoreClrCompiler extends DotNetCompiler {
    static get key() {
        return 'dotnetcoreclr';
    }
}

export class DotNetCrossgen2Compiler extends DotNetCompiler {
    static get key() {
        return 'dotnetcrossgen2';
    }
}

export class DotNetMonoCompiler extends DotNetCompiler {
    static get key() {
        return 'dotnetmono';
    }
}

export class DotNetNativeAotCompiler extends DotNetCompiler {
    static get key() {
        return 'dotnetnativeaot';
    }
}

export class DotNetLegacyCompiler extends DotNetCompiler {
    static get key() {
        return 'dotnetlegacy';
    }
}
