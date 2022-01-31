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

import { BaseCompiler } from '../base-compiler';

class DotNetCompiler extends BaseCompiler {

    get rID() { return this.compilerProps(`compiler.${this.compiler.id}.runtimeId`); }
    get targetFramework() { return this.compilerProps(`compiler.${this.compiler.id}.targetFramework`); }
    get buildConfig() { return this.compilerProps(`compiler.${this.compiler.id}.buildConfig`); }

    get compilerOptions() {
        return ['publish', '-c', this.buildConfig, '--self-contained', '--runtime', this.rID, '-v', 'q', '--nologo'];
    }

    get configurableOptions() {
        return ['--targetos', '--targetarch', '--instruction-set', '--singlemethodtypename', '--singlemethodname',
                '--singlemethodindex', '--singlemethodgenericarg', '--codegenopt', '--codegen-options'];
    }

    get configurableSwitches() {
        return ['-O', '--optimize', '--Od', '--optimize-disabled', '--Os', '--optimize-space', '--Ot', 
                '--optimize-time'];
    }

    async runCompiler(compiler, options, inputFileName, execOptions) {
        if (!execOptions) {
            execOptions = this.getDefaultExecOptions();
        }

        const programDir = path.dirname(inputFileName);
        const sourceFile = path.basename(inputFileName);
        const clrBuildDir = this.compilerProps(`compiler.${this.compiler.id}.clrDir`);

        const projectFilePath = path.join(programDir, `CompilerExplorer${this.lang.extensions[0]}proj`);
        const crossgen2Path = path.join(clrBuildDir, 'crossgen2', 'crossgen2.dll');

        const programPublishPath = path.join(
            programDir,
            'bin',
            this.buildConfig,
            this.targetFramework,
            this.rID,
            'publish',
        );

        const programDllPath = path.join(programPublishPath, 'CompilerExplorer.dll');
        const additionalSources = this.compilerProps(`compiler.${this.compiler.id}.additionalSources`);
        const langVersion = this.compilerProps(`compiler.${this.compiler.id}.langVersion`);
        const projectFileContent =
            `<Project Sdk="Microsoft.NET.Sdk">
            <PropertyGroup>
                <TargetFramework>${this.targetFramework}</TargetFramework>
                <AllowUnsafeBlocks>true</AllowUnsafeBlocks>
                <Nullable>enable</Nullable>
                <AssemblyName>CompilerExplorer</AssemblyName>
                <LangVersion>${langVersion}</LangVersion>
                <EnableDefaultCompileItems>false</EnableDefaultCompileItems>
                <EnablePreviewFeatures>${langVersion === 'preview' ? 'true' : 'false'}</EnablePreviewFeatures>
                <RestoreAdditionalProjectSources>
                  https://api.nuget.org/v3/index.json;${additionalSources ? additionalSources : ''}
                </RestoreAdditionalProjectSources>
            </PropertyGroup>
            <ItemGroup>
                <Compile Include="${sourceFile}" />
            </ItemGroup>
         </Project>
        `;

        execOptions.customCwd = programDir;
        await fs.writeFile(projectFilePath, projectFileContent);

        let crossgen2Options = [];
        const configurableOptions = this.configurableOptions;

        for (const configurableOption of configurableOptions) {
            const optionIndex = options.indexOf(configurableOption);
            if (optionIndex === -1 || optionIndex === options.length - 1) {
                continue;
            }
            crossgen2Options = crossgen2Options.concat([options[optionIndex], options[optionIndex + 1]]);
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
            this.getOutputFilename(programDir, ''),
        );

        if (crossgen2Result.code !== 0) {
            return crossgen2Result;
        }

        return compilerResult;
    }

    optionsForFilter() {
        return this.compilerOptions;
    }

    getOutputFilename(dirPath) {
        return path.join(dirPath, `output.s`);
    }

    cleanAsm(stdout) {
        let cleanedAsm = '';

        for (const line of stdout) {
            if (line.text.startsWith('; Assembly listing for method')) {
                // ; Assembly listing for method ConsoleApplication.Program:Main(System.String[])
                //                               ^ This character is the 31st character in this string.
                // `substring` removes the first 30 characters from it and uses the rest as a label.
                cleanedAsm = cleanedAsm.concat(line.text.substring(30) + ':\n');
                continue;
            }

            if (line.text.startsWith('Emitting R2R PE file')) {
                continue;
            }

            if (line.text.startsWith(';') && !line.text.startsWith('; Emitting')) {
                continue;
            }

            cleanedAsm = cleanedAsm.concat(line.text + '\n');
        }

        return cleanedAsm;
    }

    async runCrossgen2(compiler, execOptions, crossgen2Path, publishPath, dllPath, options, outputPath) {
        const crossgen2Options = [
            crossgen2Path, '-r', path.join(publishPath, '*'), dllPath, '-o', 'CompilerExplorer.r2r.dll',
            '--codegenopt', 'NgenDisasm=*', '--codegenopt', 'JitDiffableDasm=1', '--parallelism', '1',
        ].concat(options);

        const result = await this.exec(compiler, crossgen2Options, execOptions);
        result.inputFilename = dllPath;
        const transformedInput = result.filenameTransform(dllPath);
        this.parseCompilationOutput(result, transformedInput);

        await fs.writeFile(
            outputPath,
            this.cleanAsm(result.stdout),
        );

        return result;
    }
}

export class CSharpCompiler extends DotNetCompiler {
    static get key() { return 'csharp'; }
}

export class FSharpCompiler extends DotNetCompiler {
    static get key() { return 'fsharp'; }
}

export class VBCompiler extends DotNetCompiler {
    static get key() { return 'vb'; }
}
