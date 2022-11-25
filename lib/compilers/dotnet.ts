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

import {CompilationResult, ExecutionOptions} from '../../types/compilation/compilation.interfaces';
import {BaseCompiler} from '../base-compiler';
import {DotNetAsmParser} from '../parsers/asm-parser-dotnet';

class DotNetCompiler extends BaseCompiler {
    private targetFramework: string;
    private buildConfig: string;
    private clrBuildDir: string;
    private langVersion: string;

    constructor(compilerInfo, env) {
        super(compilerInfo, env);

        this.targetFramework = this.compilerProps(`compiler.${this.compiler.id}.targetFramework`);
        this.buildConfig = this.compilerProps(`compiler.${this.compiler.id}.buildConfig`);
        this.clrBuildDir = this.compilerProps(`compiler.${this.compiler.id}.clrDir`);
        this.langVersion = this.compilerProps(`compiler.${this.compiler.id}.langVersion`);
        this.asm = new DotNetAsmParser();
    }

    get compilerOptions() {
        return ['build', '-c', this.buildConfig, '-v', 'q', '--nologo', '--no-restore'];
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

    override async runCompiler(
        compiler: string,
        options: string[],
        inputFilename: string,
        execOptions: ExecutionOptions,
    ): Promise<CompilationResult> {
        if (!execOptions) {
            execOptions = this.getDefaultExecOptions();
        }

        const programDir = path.dirname(inputFilename);
        const sourceFile = path.basename(inputFilename);

        const projectFilePath = path.join(programDir, `CompilerExplorer${this.lang.extensions[0]}proj`);
        const crossgen2Path = path.join(this.clrBuildDir, 'crossgen2', 'crossgen2.dll');
        const nugetConfigPath = path.join(programDir, 'nuget.config');

        const programOutputPath = path.join(programDir, 'bin', this.buildConfig, this.targetFramework);
        const programDllPath = path.join(programOutputPath, 'CompilerExplorer.dll');
        const projectFileContent = `<Project Sdk="Microsoft.NET.Sdk">
            <PropertyGroup>
                <TargetFramework>${this.targetFramework}</TargetFramework>
                <AllowUnsafeBlocks>true</AllowUnsafeBlocks>
                <AssemblyName>CompilerExplorer</AssemblyName>
                <LangVersion>${this.langVersion}</LangVersion>
                <EnableDefaultCompileItems>false</EnableDefaultCompileItems>
            </PropertyGroup>
            <ItemGroup>
                <Compile Include="${sourceFile}" />
            </ItemGroup>
         </Project>
        `;
        const nugetConfigFileContent = `<?xml version="1.0" encoding="utf-8"?>
        <configuration>
            <packageSources>
                <clear />
            </packageSources>
        </configuration>
        `;

        execOptions.env.DOTNET_CLI_TELEMETRY_OPTOUT = 'true';
        execOptions.env.DOTNET_SKIP_FIRST_TIME_EXPERIENCE = 'true';
        execOptions.env.NUGET_PACKAGES = path.join(programDir, '.nuget');
        execOptions.env.DOTNET_NOLOGO = 'true';

        execOptions.customCwd = programDir;
        await fs.writeFile(projectFilePath, projectFileContent);
        await fs.writeFile(nugetConfigPath, nugetConfigFileContent);

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

        const restoreOptions = ['restore', '--configfile', nugetConfigPath, '-v', 'q', '--nologo'];
        const restoreResult = await this.exec(compiler, restoreOptions, execOptions);

        const compilerResult = await super.runCompiler(compiler, this.compilerOptions, inputFilename, execOptions);

        if (compilerResult.code !== 0) {
            return compilerResult;
        }

        const crossgen2Result = await this.runCrossgen2(
            compiler,
            execOptions,
            crossgen2Path,
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

    override optionsForFilter() {
        return this.compilerOptions;
    }

    async runCrossgen2(compiler, execOptions, crossgen2Path, bclPath, dllPath, options, outputPath) {
        const crossgen2Options = [
            crossgen2Path,
            '-r',
            path.join(bclPath, '*'),
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

        const compilerExecResult = await this.exec(compiler, crossgen2Options, execOptions);
        const result = this.transformToCompilationResult(compilerExecResult, dllPath);

        await fs.writeFile(
            outputPath,
            result.stdout.map(o => o.text).reduce((a, n) => `${a}\n${n}`),
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
