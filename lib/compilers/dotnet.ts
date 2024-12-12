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

import type {
    CompilationResult,
    ExecutionOptions,
    ExecutionOptionsWithEnv,
} from '../../types/compilation/compilation.interfaces.js';
import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import {ExecutableExecutionOptions} from '../../types/execution/execution.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {LanguageKey} from '../../types/languages.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';
import {CompilationEnvironment} from '../compilation-env.js';
import {AssemblyName, DotnetExtraConfiguration} from '../execution/dotnet-execution-env.js';
import {IExecutionEnvironment} from '../execution/execution-env.interfaces.js';
import {DotNetAsmParser} from '../parsers/asm-parser-dotnet.js';

class DotNetCompiler extends BaseCompiler {
    private readonly sdkBaseDir: string;
    private readonly buildConfig: string;
    private readonly clrBuildDir: string;
    private readonly langVersion: string;
    private readonly corerunPath: string;
    private readonly disassemblyLoaderPath: string;
    private readonly crossgen2Path: string;
    private readonly ilcPath: string;
    private readonly ildasmPath: string;
    private readonly toolsPath: string;

    constructor(compilerInfo: PreliminaryCompilerInfo, env: CompilationEnvironment) {
        super(compilerInfo, env);

        this.sdkBaseDir = path.join(path.dirname(compilerInfo.exe), 'sdk');

        this.buildConfig = this.compilerProps<string>(`compiler.${this.compiler.id}.buildConfig`);
        this.clrBuildDir = this.compilerProps<string>(`compiler.${this.compiler.id}.clrDir`);
        this.langVersion = this.compilerProps<string>(`compiler.${this.compiler.id}.langVersion`);

        this.corerunPath = path.join(this.clrBuildDir, 'corerun');
        this.crossgen2Path = path.join(this.clrBuildDir, 'crossgen2', 'crossgen2');
        this.ilcPath = path.join(this.clrBuildDir, 'ilc-published', 'ilc');
        this.ildasmPath = path.join(this.clrBuildDir, 'ildasm');
        this.toolsPath = path.join(this.clrBuildDir, 'dotnet-tools', '.store');

        this.asm = new DotNetAsmParser();
        this.disassemblyLoaderPath = path.join(this.clrBuildDir, 'DisassemblyLoader', 'DisassemblyLoader.dll');
    }

    async getCompilerInfo(lang: LanguageKey): Promise<DotNetCompilerInfo> {
        const sdkDirs = await fs.readdir(this.sdkBaseDir);
        const sdkVersion = sdkDirs[0];

        const parts = sdkVersion.split('.');
        const targetFramework = `net${parts[0]}.${parts[1]}`;
        const sdkMajorVersion = Number(parts[0]);

        switch (lang) {
            case 'csharp': {
                return {
                    targetFramework: targetFramework,
                    sdkVersion: sdkVersion,
                    sdkMajorVersion: sdkMajorVersion,
                    compilerPath: path.join(this.sdkBaseDir, sdkVersion, 'Roslyn', 'bincore', 'csc.dll'),
                };
            }
            case 'vb': {
                return {
                    targetFramework: targetFramework,
                    sdkVersion: sdkVersion,
                    sdkMajorVersion: sdkMajorVersion,
                    compilerPath: path.join(this.sdkBaseDir, sdkVersion, 'Roslyn', 'bincore', 'vbc.dll'),
                };
            }
            case 'fsharp': {
                return {
                    targetFramework: targetFramework,
                    sdkVersion: sdkVersion,
                    sdkMajorVersion: sdkMajorVersion,
                    compilerPath: path.join(this.sdkBaseDir, sdkVersion, 'FSharp', 'fsc.dll'),
                };
            }
            case 'il': {
                return {
                    targetFramework: targetFramework,
                    sdkVersion: sdkVersion,
                    sdkMajorVersion: sdkMajorVersion,
                    compilerPath: path.join(this.clrBuildDir, 'ilasm'),
                };
            }
            default: {
                throw new Error(`Unsupported language: ${lang}`);
            }
        }
    }

    setCompilerExecOptions(execOptions: ExecutionOptionsWithEnv, programDir: string) {
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
        execOptions.timeoutMs = 30000;

        execOptions.customCwd = programDir;
    }

    override async buildExecutable(
        compiler: string,
        options: string[],
        inputFilename: string,
        execOptions: ExecutionOptionsWithEnv,
    ) {
        const compilerInfo = await this.getCompilerInfo(this.lang.id);
        return await this.buildToDll(compiler, compilerInfo, inputFilename, execOptions, true);
    }

    async buildToDll(
        dotnetPath: string,
        compilerInfo: DotNetCompilerInfo,
        inputFilename: string,
        execOptions: ExecutionOptionsWithEnv,
        buildToBinary?: boolean,
    ): Promise<CompilationResult> {
        const programDir = path.dirname(inputFilename);
        const programOutputPath = path.join(programDir, 'bin', this.buildConfig, compilerInfo.targetFramework);
        await fs.mkdirs(programOutputPath);
        const outputFilename = path.join(programOutputPath, 'CompilerExplorer.dll');
        this.setCompilerExecOptions(execOptions, programDir);
        let compilerResult: CompilationResult;

        switch (this.lang.id) {
            case 'csharp': {
                compilerResult = await this.runCsc(
                    dotnetPath,
                    compilerInfo,
                    inputFilename,
                    outputFilename,
                    execOptions,
                    buildToBinary,
                );
                break;
            }
            case 'vb': {
                compilerResult = await this.runVbc(
                    dotnetPath,
                    compilerInfo,
                    inputFilename,
                    outputFilename,
                    execOptions,
                    buildToBinary,
                );
                break;
            }
            case 'fsharp': {
                compilerResult = await this.runFsc(
                    dotnetPath,
                    compilerInfo,
                    inputFilename,
                    outputFilename,
                    execOptions,
                    buildToBinary,
                );
                break;
            }
            case 'il': {
                compilerResult = await this.runIlAsm(
                    compilerInfo.compilerPath,
                    inputFilename,
                    outputFilename,
                    execOptions,
                    buildToBinary,
                );
                break;
            }
            default: {
                throw new Error(`Unsupported language: ${this.lang.id}`);
            }
        }

        if (compilerResult.code === 0) {
            await fs.createFile(this.getOutputFilename(programDir, this.outputFilebase));
        }
        return compilerResult;
    }

    getRefAssembliesAndAnalyzers(dotnetPath: string, compilerInfo: DotNetCompilerInfo, lang: LanguageKey) {
        const packDir = path.join(path.dirname(dotnetPath), 'packs', 'Microsoft.NETCore.App.Ref');
        const packVersion = fs.readdirSync(packDir)[0];
        const refDir = path.join(packDir, packVersion, 'ref', compilerInfo.targetFramework);
        const refAssemblies = fs
            .readdirSync(refDir)
            .filter(f => f.endsWith('.dll'))
            .map(f => path.join(refDir, f));
        const analyzers: string[] = [];
        const analyzersDir = path.join(
            this.sdkBaseDir,
            compilerInfo.sdkVersion,
            'Sdks',
            'Microsoft.NET.Sdk',
            'analyzers',
        );
        switch (lang) {
            case 'csharp': {
                const generatorsDir = path.join(packDir, packVersion, 'analyzers', 'dotnet', 'cs');
                if (fs.existsSync(generatorsDir)) {
                    analyzers.push(
                        ...fs
                            .readdirSync(generatorsDir)
                            .filter(f => f.endsWith('.dll'))
                            .map(f => path.join(generatorsDir, f)),
                    );
                }
                analyzers.push(
                    path.join(analyzersDir, 'Microsoft.CodeAnalysis.NetAnalyzers.dll'),
                    path.join(analyzersDir, 'Microsoft.CodeAnalysis.CSharp.NetAnalyzers.dll'),
                );
                break;
            }
            case 'vb': {
                analyzers.push(
                    path.join(analyzersDir, 'Microsoft.CodeAnalysis.NetAnalyzers.dll'),
                    path.join(analyzersDir, 'Microsoft.CodeAnalysis.VisualBasic.NetAnalyzers.dll'),
                );
                break;
            }
        }
        refAssemblies.push(this.disassemblyLoaderPath);
        return {refAssemblies, analyzers};
    }

    getPreprocessorDefines(sdkMajorVersion: number, lang: LanguageKey) {
        const defines = [
            'TRACE',
            'NET',
            `NET${sdkMajorVersion}_0`,
            'RELEASE',
            'NETCOREAPP',
            'NETCOREAPP1_0_OR_GREATER',
            'NETCOREAPP1_1_OR_GREATER',
            'NETCOREAPP2_0_OR_GREATER',
            'NETCOREAPP2_1_OR_GREATER',
            'NETCOREAPP2_2_OR_GREATER',
            'NETCOREAPP3_0_OR_GREATER',
            'NETCOREAPP3_1_OR_GREATER',
        ];

        for (let version = sdkMajorVersion; version >= 5; version--) {
            defines.push(`NET${version}_0_OR_GREATER`);
        }

        if (lang === 'vb') {
            const vbDefines = defines.map(d => `${d}=-1`);
            vbDefines.push('CONFIG="Release"', 'PLATFORM="AnyCPU"', '_MyType="Empty"');
            return vbDefines;
        }

        return defines;
    }

    async runCsc(
        dotnetPath: string,
        compilerInfo: DotNetCompilerInfo,
        inputFilename: string,
        outputFilename: string,
        execOptions: ExecutionOptionsWithEnv,
        buildToBinary?: boolean,
    ) {
        const {refAssemblies, analyzers} = this.getRefAssembliesAndAnalyzers(dotnetPath, compilerInfo, 'csharp');
        const defines = this.getPreprocessorDefines(compilerInfo.sdkMajorVersion, 'csharp');
        const options = [
            '-nologo',
            `-target:${buildToBinary ? 'exe' : 'library'}`,
            '-filealign:512',
            '-unsafe+',
            '-checked-',
            '-nowarn:1701,1702',
            '-fullpaths',
            '-nostdlib+',
            '-errorreport:prompt',
            '-warn:9',
            '-highentropyva+',
            '-nullable:enable',
            '-debug-',
            '-optimize+',
            '-warnaserror-',
            '-utf8output',
            '-deterministic+',
            `-langversion:${this.langVersion}`,
            '-warnaserror+:NU1605,SYSLIB0011',
        ];
        for (const analyzer of analyzers) {
            options.push(`-analyzer:${analyzer}`);
        }
        for (const refAssembly of refAssemblies) {
            options.push(`-reference:${refAssembly}`);
        }

        const assemblyInfo = `using System;
using System.Reflection;
[assembly: global::System.Runtime.Versioning.TargetFrameworkAttribute\
(".NETCoreApp,Version=v${compilerInfo.sdkMajorVersion}.0",\
 FrameworkDisplayName = ".NET ${compilerInfo.sdkMajorVersion}.0")]
[assembly: System.Reflection.AssemblyCompanyAttribute("CompilerExplorer")]
[assembly: System.Reflection.AssemblyConfigurationAttribute("Release")]
[assembly: System.Reflection.AssemblyFileVersionAttribute("1.0.0.0")]
[assembly: System.Reflection.AssemblyInformationalVersionAttribute("1.0.0")]
[assembly: System.Reflection.AssemblyProductAttribute("CompilerExplorer")]
[assembly: System.Reflection.AssemblyTitleAttribute("CompilerExplorer")]
[assembly: System.Reflection.AssemblyVersionAttribute("1.0.0.0")]
`;
        const programDir = path.dirname(inputFilename);
        const assemblyInfoPath = path.join(programDir, 'AssemblyInfo.cs');
        await fs.writeFile(assemblyInfoPath, assemblyInfo);

        options.push(`-define:${defines.join(';')}`, `-out:${outputFilename}`, inputFilename, assemblyInfoPath);
        return await super.runCompiler(dotnetPath, [compilerInfo.compilerPath, ...options], inputFilename, execOptions);
    }

    async runVbc(
        dotnetPath: string,
        compilerInfo: DotNetCompilerInfo,
        inputFilename: string,
        outputFilename: string,
        execOptions: ExecutionOptionsWithEnv,
        buildToBinary?: boolean,
    ) {
        const {refAssemblies, analyzers} = this.getRefAssembliesAndAnalyzers(dotnetPath, compilerInfo, 'vb');
        const defines = this.getPreprocessorDefines(compilerInfo.sdkMajorVersion, 'vb');
        const options = [
            '-nologo',
            `-target:${buildToBinary ? 'exe' : 'library'}`,
            '-filealign:512',
            `-imports:Microsoft.VisualBasic,System,System.Collections,System.Collections.Generic,\
System.Diagnostics,System.Linq,System.Xml.Linq,System.Threading.Tasks`,
            '-optioncompare:Binary',
            '-optionexplicit+',
            '-optionstrict:custom',
            '-nowarn:41999,42016,42017,42018,42019,42020,42021,42022,42032,42036',
            '-nosdkpath',
            '-optioninfer+',
            '-nostdlib',
            '-errorreport:prompt',
            '-rootnamespace:CompilerExplorer',
            '-highentropyva+',
            '-debug-',
            '-optimize+',
            '-warnaserror-',
            '-utf8output',
            '-deterministic+',
            `-langversion:${this.langVersion}`,
            '-warnaserror+:NU1605,SYSLIB0011',
        ];
        for (const analyzer of analyzers) {
            options.push(`-analyzer:${analyzer}`);
        }
        for (const refAssembly of refAssemblies) {
            options.push(`-reference:${refAssembly}`);
        }

        const vbRuntime = refAssemblies.find(f => f.includes('Microsoft.VisualBasic.dll'));
        if (vbRuntime) {
            options.push(`-vbruntime:${vbRuntime}`);
        }

        const assemblyInfo = `Option Strict Off
Option Explicit On

Imports System
Imports System.Reflection
<Assembly: Global.System.Runtime.Versioning.TargetFrameworkAttribute\
(".NETCoreApp,Version=v${compilerInfo.sdkMajorVersion}.0",\
 FrameworkDisplayName:=".NET ${compilerInfo.sdkMajorVersion}.0")>
<Assembly: System.Reflection.AssemblyCompanyAttribute("CompilerExplorer"),  _
 Assembly: System.Reflection.AssemblyConfigurationAttribute("Release"),  _
 Assembly: System.Reflection.AssemblyFileVersionAttribute("1.0.0.0"),  _
 Assembly: System.Reflection.AssemblyInformationalVersionAttribute("1.0.0"),  _
 Assembly: System.Reflection.AssemblyProductAttribute("CompilerExplorer"),  _
 Assembly: System.Reflection.AssemblyTitleAttribute("CompilerExplorer"),  _
 Assembly: System.Reflection.AssemblyVersionAttribute("1.0.0.0")>
`;
        const programDir = path.dirname(inputFilename);
        const assemblyInfoPath = path.join(programDir, 'AssemblyInfo.vb');
        await fs.writeFile(assemblyInfoPath, assemblyInfo);

        options.push(`-define:${defines.join(',')}`, `-out:${outputFilename}`, inputFilename, assemblyInfoPath);
        return await super.runCompiler(dotnetPath, [compilerInfo.compilerPath, ...options], inputFilename, execOptions);
    }

    async runFsc(
        dotnetPath: string,
        compilerInfo: DotNetCompilerInfo,
        inputFilename: string,
        outputFilename: string,
        execOptions: ExecutionOptionsWithEnv,
        buildToBinary?: boolean,
    ) {
        const {refAssemblies} = this.getRefAssembliesAndAnalyzers(dotnetPath, compilerInfo, 'fsharp');
        const defines = this.getPreprocessorDefines(compilerInfo.sdkMajorVersion, 'fsharp');
        const options = [
            '--nologo',
            `--target:${buildToBinary ? 'exe' : 'library'}`,
            `--langversion:${this.langVersion}`,
            '--noframework',
            '--optimize+',
            '--utf8output',
            '--warn:3',
            '--warnaserror:3239',
            '--fullpaths',
            '--flaterrors',
            '--highentropyva+',
            '--targetprofile:netcore',
            '--nocopyfsharpcore',
            '--deterministic+',
            '--simpleresolution',
        ];
        for (const refAssembly of refAssemblies) {
            options.push(`-r:${refAssembly}`);
        }

        const versionInfo = `namespace Microsoft.BuildSettings
[<System.Runtime.Versioning.TargetFrameworkAttribute\
(".NETCoreApp,Version=v${compilerInfo.sdkMajorVersion}.0",\
 FrameworkDisplayName=".NET ${compilerInfo.sdkMajorVersion}.0")>]
do ()
`;
        const assemblyInfo = `namespace FSharp

open System
open System.Reflection

[<assembly: System.Reflection.AssemblyCompanyAttribute("CompilerExplorer")>]
[<assembly: System.Reflection.AssemblyConfigurationAttribute("Release")>]
[<assembly: System.Reflection.AssemblyFileVersionAttribute("1.0.0.0")>]
[<assembly: System.Reflection.AssemblyInformationalVersionAttribute("1.0.0")>]
[<assembly: System.Reflection.AssemblyProductAttribute("CompilerExplorer")>]
[<assembly: System.Reflection.AssemblyTitleAttribute("CompilerExplorer")>]
[<assembly: System.Reflection.AssemblyVersionAttribute("1.0.0.0")>]
do()
`;
        const programDir = path.dirname(inputFilename);
        const versionInfoPath = path.join(programDir, 'VersionInfo.fs');
        const assemblyInfoPath = path.join(programDir, 'AssemblyInfo.fs');
        await fs.writeFile(versionInfoPath, versionInfo);
        await fs.writeFile(assemblyInfoPath, assemblyInfo);

        options.push(
            ...defines.map(d => `--define:${d}`),
            `-o:${outputFilename}`,
            `-r:${path.join(path.dirname(compilerInfo.compilerPath), 'FSharp.Core.dll')}`,
            versionInfoPath,
            assemblyInfoPath,
            inputFilename,
        );
        return await super.runCompiler(dotnetPath, [compilerInfo.compilerPath, ...options], inputFilename, execOptions);
    }

    async runIlAsm(
        ilasmPath: string,
        inputFilename: string,
        outputFilename: string,
        execOptions: ExecutionOptionsWithEnv,
        buildToBinary?: boolean,
    ) {
        const assemblyInfo = `.assembly extern DisassemblyLoader { }
.assembly CompilerExplorer
{
    .ver 1:0:0:0
}
.module CompilerExplorer.dll
#include "${path.basename(inputFilename)}"
`;

        const programDir = path.dirname(inputFilename);
        const assemblyInfoPath = path.join(programDir, 'AssemblyInfo.il');
        await fs.writeFile(assemblyInfoPath, assemblyInfo);
        const options = [
            '-nologo',
            '-quiet',
            '-optimize',
            buildToBinary ? '-exe' : '-dll',
            assemblyInfoPath,
            `-include:${programDir}`,
            `-output:${outputFilename}`,
        ];
        return await super.runCompiler(ilasmPath, options, inputFilename, execOptions);
    }

    override getSharedLibraryPathsAsArguments() {
        return [];
    }

    // eslint-disable-next-line max-statements
    override async runCompiler(
        compiler: string,
        options: string[],
        inputFilename: string,
        execOptions: ExecutionOptionsWithEnv,
        filters: ParseFiltersAndOutputOptions,
    ): Promise<CompilationResult> {
        const compilerInfo = await this.getCompilerInfo(this.lang.id);
        const programDir = path.dirname(inputFilename);
        const programOutputPath = path.join(programDir, 'bin', this.buildConfig, compilerInfo.targetFramework);
        const programDllPath = path.join(programOutputPath, 'CompilerExplorer.dll');
        const envVarFileContents = ['DOTNET_EnableWriteXorExecute=0'];
        const isIlDasm = this.compiler.group === 'dotnetildasm';
        const isIlSpy = this.compiler.group === 'dotnetilspy';
        const isCoreRun = this.compiler.group === 'dotnetcoreclr';
        const toolOptions: string[] = [];

        let overrideDiffable = false;
        let overrideDisasm = false;
        let overrideAssembly = false;
        let overrideTiered = false;
        let isAot = this.compiler.group === 'dotnetnativeaot';
        let isMono = this.compiler.group === 'dotnetmono';
        let isCrossgen2 =
            this.compiler.group === 'dotnetcrossgen2' ||
            (this.compiler.group === 'dotnetlegacy' && compilerInfo.sdkMajorVersion === 6);
        let codegenArch = 'x64';

        while (options.length > 0) {
            const currentOption = options.shift();
            if (!currentOption) {
                continue;
            }

            // options before the input filename are user options
            if (currentOption.includes(path.basename(inputFilename))) {
                break;
            }

            if ((isCoreRun || isMono) && (currentOption === '-e' || currentOption === '--env')) {
                const envVar = options.shift();
                if (envVar) {
                    const [name] = envVar.split('=');
                    const normalizedName = name.trim().toUpperCase();
                    if (normalizedName === 'DOTNET_TIEREDCOMPILATION') {
                        overrideTiered = true;
                    }
                    if (normalizedName === 'DOTNET_JITDISASM') {
                        overrideDisasm = true;
                    }
                    if (normalizedName === 'DOTNET_JITDISASMASSEMBILES') {
                        overrideAssembly = true;
                    }
                    if (normalizedName === 'DOTNET_JITDIFFABLEDASM' || normalizedName === 'DOTNET_JITDISASMDIFFABLE') {
                        overrideDiffable = true;
                    }
                    envVarFileContents.push(envVar);
                }
            } else {
                if (this.compiler.group === 'dotnetlegacy') {
                    if (currentOption === '--aot') {
                        isAot = true;
                    } else if (currentOption === '--crossgen2') {
                        isCrossgen2 = true;
                    } else if (currentOption === '--mono') {
                        isMono = true;
                    } else {
                        toolOptions.push(currentOption);
                    }
                } else {
                    toolOptions.push(currentOption);
                    if (currentOption === '--codegenopt' || currentOption === '--codegen-options') {
                        const value = options.shift();
                        if (value) {
                            toolOptions.push(value);
                            const normalizedValue = value.trim().toUpperCase();
                            if (normalizedValue.startsWith('TIEREDCOMPILATION=')) {
                                overrideTiered = true;
                            }
                            if (normalizedValue.startsWith('JITDISASM=')) {
                                overrideDisasm = true;
                            }
                            if (normalizedValue.startsWith('JITDISASMASSEMBILES=')) {
                                overrideAssembly = true;
                            }
                            if (
                                normalizedValue.startsWith('JITDIFFABLEDASM=') ||
                                normalizedValue.startsWith('JITDISASMDIFFABLE=')
                            ) {
                                overrideDiffable = true;
                            }
                        }
                    }
                    if (currentOption === '--targetarch') {
                        const value = options.shift();
                        if (value) {
                            toolOptions.push(value);
                            codegenArch = value.trim().toLowerCase();
                        }
                    }
                }
            }
        }

        const needCodegenOptions = isCrossgen2 || isAot;

        if (needCodegenOptions) {
            toolOptions.push('--parallelism', '1');
        }

        if (!isIlDasm && !isIlSpy) {
            if (!overrideDiffable && compilerInfo.sdkMajorVersion < 8) {
                if (needCodegenOptions) {
                    toolOptions.push('--codegenopt', 'JitDiffableDasm=1');
                }
                envVarFileContents.push('DOTNET_JitDiffableDasm=1');
            }

            if (!overrideDisasm) {
                if (needCodegenOptions) {
                    toolOptions.push(
                        '--codegenopt',
                        compilerInfo.sdkMajorVersion === 6 ? 'NgenDisasm=*' : 'JitDisasm=*',
                    );
                }
                envVarFileContents.push('DOTNET_JitDisasm=*');
            }

            if (!overrideAssembly) {
                if (needCodegenOptions && compilerInfo.sdkMajorVersion >= 9) {
                    toolOptions.push('--codegenopt', 'JitDisasmAssemblies=CompilerExplorer');
                }
                envVarFileContents.push('DOTNET_JitDisasmAssemblies=CompilerExplorer');
            }

            if (!overrideTiered) {
                envVarFileContents.push('DOTNET_TieredCompilation=0');
            }
        }

        this.setCompilerExecOptions(execOptions, programDir);

        const compilerResult = await this.buildToDll(
            compiler,
            compilerInfo,
            inputFilename,
            execOptions,
            filters.binary,
        );
        if (compilerResult.code !== 0) {
            return compilerResult;
        }

        if (isIlDasm) {
            const ilDasmResult = await this.runIlDasm(
                execOptions,
                programDllPath,
                toolOptions,
                this.getOutputFilename(programDir, this.outputFilebase),
            );

            if (ilDasmResult.code !== 0) {
                return ilDasmResult;
            }
        } else if (isIlSpy) {
            const ilSpyResult = await this.runIlSpy(
                execOptions,
                programDllPath,
                toolOptions,
                this.getOutputFilename(programDir, this.outputFilebase),
                compilerInfo.sdkMajorVersion <= 6,
            );

            if (ilSpyResult.code !== 0) {
                return ilSpyResult;
            }
        } else if (isCrossgen2) {
            const crossgen2Result = await this.runCrossgen2(
                compiler,
                compilerInfo.sdkMajorVersion,
                codegenArch,
                execOptions,
                this.clrBuildDir,
                programDllPath,
                toolOptions,
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
                this.getOutputFilename(programDir, this.outputFilebase),
                filters.binary ?? false,
            );

            if (ilcResult.code !== 0) {
                return ilcResult;
            }
        } else {
            const corerunResult = await this.runCorerunForDisasm(
                execOptions,
                this.clrBuildDir,
                envVarFileContents,
                programDllPath,
                toolOptions,
                this.getOutputFilename(programDir, this.outputFilebase),
                isMono,
                compilerInfo.sdkMajorVersion >= 7,
            );

            if (corerunResult.code !== 0) {
                return corerunResult;
            }
        }

        return compilerResult;
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions) {
        return [];
    }

    override postCompilationPreCacheHook(result: CompilationResult): CompilationResult {
        const isIlSpy = this.compiler.group === 'dotnetilspy';
        if (isIlSpy && result.code === 0) {
            result.languageId = 'csharp';
        }
        return result;
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
        execOptions: ExecutionOptionsWithEnv,
        coreRoot: string,
        envVars: string[],
        dllPath: string,
        options: string[],
        outputPath: string,
        isMono: boolean,
        useEnvFile: boolean,
    ) {
        if (isMono) {
            coreRoot = path.join(coreRoot, 'mono');
        }

        const corerunOptions = ['--clr-path', coreRoot].concat([...options, this.disassemblyLoaderPath, dllPath]);

        if (useEnvFile) {
            const envVarFilePath = path.join(path.dirname(outputPath), '.env');
            await fs.writeFile(envVarFilePath, envVars.join('\n'));
            corerunOptions.unshift('--env', envVarFilePath);
        } else {
            for (const env of envVars) {
                const delimiterIndex = env.indexOf('=');
                if (delimiterIndex !== -1) {
                    execOptions.env[env.substring(0, delimiterIndex)] = env.substring(delimiterIndex + 1);
                }
            }
        }

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

    async runIlSpy(
        execOptions: ExecutionOptions,
        dllPath: string,
        options: string[],
        outputPath: string,
        useDotNetHost: boolean,
    ) {
        const ilspyRoot = path.join(this.toolsPath, 'ilspycmd');
        const ilspyVersionDirs = await fs.readdir(ilspyRoot);
        const ilspyVersion = ilspyVersionDirs[0];
        const ilspyToolsDir = path.join(ilspyRoot, ilspyVersion, 'ilspycmd', ilspyVersion, 'tools');
        const targetFrameworkDirs = await fs.readdir(ilspyToolsDir);
        const targetFramework = targetFrameworkDirs[0];
        const ilspyPath = path.join(ilspyToolsDir, targetFramework, 'any', 'ilspycmd.dll');

        // prettier-ignore
        const ilspyOptions = [ilspyPath, dllPath, '--disable-updatecheck', '-r', this.clrBuildDir].concat(options);
        const compilerPath = useDotNetHost ? this.compiler.exe : this.corerunPath;
        const compilerExecResult = await this.exec(compilerPath, ilspyOptions, execOptions);
        const result = this.transformToCompilationResult(compilerExecResult, dllPath);

        await fs.writeFile(
            outputPath,
            `// ilspy ${await this.getRuntimeVersion()}\n\n${result.stdout
                .map(o => o.text)
                .reduce((a, n) => `${a}\n${n}`, '')}`,
        );

        return result;
    }

    async runIlDasm(execOptions: ExecutionOptions, dllPath: string, options: string[], outputPath: string) {
        // prettier-ignore
        const ildasmOptions = [dllPath, '-utf8'].concat(options);

        const compilerExecResult = await this.exec(this.ildasmPath, ildasmOptions, execOptions);
        const result = this.transformToCompilationResult(compilerExecResult, dllPath);

        await fs.writeFile(
            outputPath,
            `// ildasm ${await this.getRuntimeVersion()}\n\n${result.stdout
                .map(o => o.text)
                .reduce((a, n) => `${a}\n${n}`, '')}`,
        );

        return result;
    }

    async runCrossgen2(
        compiler: string,
        sdkMajorVersion: number,
        arch: string,
        execOptions: ExecutionOptions,
        bclPath: string,
        dllPath: string,
        options: string[],
        outputPath: string,
    ) {
        // prettier-ignore
        const crossgen2Options = [
            '-r', path.join(bclPath, '/'),
            '-r', this.disassemblyLoaderPath,
            dllPath,
            '-o', `${AssemblyName}.r2r.dll`,
        ].concat(options);

        const corelibPath = path.join(this.clrBuildDir, 'corelib', arch, 'System.Private.CoreLib.dll');
        if (await fs.exists(corelibPath)) {
            crossgen2Options.unshift('-r', corelibPath);
        }

        if (sdkMajorVersion >= 9) {
            crossgen2Options.push('--inputbubble', '--compilebubblegenerics');
        }

        if (await fs.exists(this.crossgen2Path)) {
            compiler = this.crossgen2Path;
        } else {
            crossgen2Options.unshift(this.crossgen2Path + '.dll');
        }

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
        options: string[],
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
        ].concat(options);

        if (!buildToBinary) {
            ilcOptions.push('--nativelib', '--root:CompilerExplorer');
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

    override async runExecutable(executable: string, executeParameters: ExecutableExecutionOptions, homeDir: string) {
        const execOptionsCopy: ExecutableExecutionOptions = JSON.parse(
            JSON.stringify(executeParameters),
        ) as ExecutableExecutionOptions;

        if (this.compiler.executionWrapper) {
            execOptionsCopy.args = [...this.compiler.executionWrapperArgs, executable, ...execOptionsCopy.args];
            executable = this.compiler.executionWrapper;
        }

        const isMono = this.compiler.group === 'dotnetmono';
        const compilerInfo = await this.getCompilerInfo(this.lang.id);
        const extraConfiguration: DotnetExtraConfiguration = {
            buildConfig: this.buildConfig,
            clrBuildDir: isMono ? path.join(this.clrBuildDir, 'mono') : this.clrBuildDir,
            langVersion: this.langVersion,
            targetFramework: compilerInfo.targetFramework,
            corerunPath: this.corerunPath,
        };

        const execEnv: IExecutionEnvironment = new this.executionEnvironmentClass(this.env);
        return await execEnv.execBinary(executable, execOptionsCopy, homeDir, extraConfiguration);
    }
}

interface DotNetCompilerInfo {
    targetFramework: string;
    sdkVersion: string;
    sdkMajorVersion: number;
    compilerPath: string;
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

export class DotNetIlDasmCompiler extends DotNetCompiler {
    static get key() {
        return 'dotnetildasm';
    }
}

export class DotNetIlSpyCompiler extends DotNetCompiler {
    static get key() {
        return 'dotnetilspy';
    }
}

export class DotNetLegacyCompiler extends DotNetCompiler {
    static get key() {
        return 'dotnetlegacy';
    }
}
