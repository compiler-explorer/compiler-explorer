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

/// <reference types="../base-compiler" />
import {BaseCompiler} from '../base-compiler';
import {DotNetAsmParser} from '../parsers/asm-parser-dotnet';

class DotNetCompiler extends BaseCompiler {
    private rID: string;
    private targetFramework: string;
    private buildConfig: string;
    private nugetPackagesPath: string;
    private clrBuildDir: string;
    private additionalSources: string;
    private langVersion: string;
    protected asm: DotNetAsmParser;

    constructor(compilerInfo, env) {
        super(compilerInfo, env);

        this.rID = this.compilerProps(`compiler.${this.compiler.id}.runtimeId`);
        this.targetFramework = this.compilerProps(`compiler.${this.compiler.id}.targetFramework`);
        this.buildConfig = this.compilerProps(`compiler.${this.compiler.id}.buildConfig`);
        this.nugetPackagesPath = this.compilerProps(`compiler.${this.compiler.id}.nugetPackages`);
        this.clrBuildDir = this.compilerProps(`compiler.${this.compiler.id}.clrDir`);
        this.additionalSources = this.compilerProps(`compiler.${this.compiler.id}.additionalSources`);
        this.langVersion = this.compilerProps(`compiler.${this.compiler.id}.langVersion`);
        this.asm = new DotNetAsmParser();
    }

    get compilerOptions() {
        return ['publish', '-c', this.buildConfig, '--self-contained', '--runtime', this.rID, '-v', 'q', '--nologo'];
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

    async runCompiler(compiler, options, inputFileName, execOptions) {
        if (!execOptions) {
            execOptions = this.getDefaultExecOptions();
        }

        const programDir = path.dirname(inputFileName);
        const sourceFile = path.basename(inputFileName);

        const projectFilePath = path.join(programDir, `CompilerExplorer${this.lang.extensions[0]}proj`);
        const crossgen2Path = path.join(this.clrBuildDir, 'crossgen2', 'crossgen2.dll');

        const programPublishPath = path.join(
            programDir,
            'bin',
            this.buildConfig,
            this.targetFramework,
            this.rID,
            'publish'
        );

        const programDllPath = path.join(programPublishPath, 'CompilerExplorer.dll');
        const projectFileContent = `<Project Sdk="Microsoft.NET.Sdk">
            <PropertyGroup>
                <TargetFramework>${this.targetFramework}</TargetFramework>
                <AllowUnsafeBlocks>true</AllowUnsafeBlocks>
                <AssemblyName>CompilerExplorer</AssemblyName>
                <LangVersion>${this.langVersion}</LangVersion>
                <EnableDefaultCompileItems>false</EnableDefaultCompileItems>
                <EnablePreviewFeatures>${this.langVersion === 'preview' ? 'true' : 'false'}</EnablePreviewFeatures>
                <RestoreAdditionalProjectSources>
                  https://api.nuget.org/v3/index.json;${this.additionalSources ? this.additionalSources : ''}
                </RestoreAdditionalProjectSources>
            </PropertyGroup>
            <ItemGroup>
                <Compile Include="${sourceFile}" />
            </ItemGroup>
         </Project>
        `;

        execOptions.env.DOTNET_CLI_TELEMETRY_OPTOUT = 'true';
        execOptions.env.DOTNET_SKIP_FIRST_TIME_EXPERIENCE = 'true';
        execOptions.env.NUGET_PACKAGES = this.nugetPackagesPath;
        execOptions.env.DOTNET_NOLOGO = 'true';

        execOptions.customCwd = programDir;
        await fs.writeFile(projectFilePath, projectFileContent);

        const crossgen2Options: string[] = [];
        const configurableOptions = this.configurableOptions;

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

        const compilerResult = await super.runCompiler(compiler, this.compilerOptions, inputFileName, execOptions);

        if (compilerResult.code !== 0) {
            return compilerResult;
        }

        const crossgen2Result = await this.runCrossgen2(
            compiler,
            execOptions,
            crossgen2Path,
            programPublishPath,
            programDllPath,
            crossgen2Options,
            this.getOutputFilename(programDir, this.outputFilebase)
        );

        if (crossgen2Result.code !== 0) {
            return crossgen2Result;
        }

        return compilerResult;
    }

    optionsForFilter() {
        return this.compilerOptions;
    }

    async runCrossgen2(compiler, execOptions, crossgen2Path, publishPath, dllPath, options, outputPath) {
        const crossgen2Options = [
            crossgen2Path,
            '-r',
            path.join(publishPath, '*'),
            dllPath,
            '-o',
            'CompilerExplorer.r2r.dll',
            '--codegenopt',
            'NgenDisasm=*',
            '--codegenopt',
            'JitDiffableDasm=1',
            '--parallelism',
            '1',
            '--inputbubble',
            '--compilebubblegenerics',
        ].concat(options);

        const result = await this.exec(compiler, crossgen2Options, execOptions);
        result.inputFilename = dllPath;
        const transformedInput = result.filenameTransform(dllPath);
        this.parseCompilationOutput(result, transformedInput);

        await fs.writeFile(
            outputPath,
            result.stdout.map(o => o.text).reduce((a, n) => `${a}\n${n}`)
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
