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

export class CSharpCompiler extends BaseCompiler {

    static get key() { return 'csharp'; }

    get rID() { return this.compilerProps('runtimeId'); }
    get targetFramework() { return this.compilerProps('targetFramework'); }
    get buildConfig() { return this.compilerProps('buildConfig'); }

    get compilerOptions() {
        return ['publish', '-c', this.buildConfig, '--self-contained', '--runtime', this.rID];
    }

    async runCompiler(compiler, options, inputFileName, execOptions) {
        if (!execOptions) {
            execOptions = this.getDefaultExecOptions();
        }

        const programDir = path.dirname(inputFileName);
        const clrBuildDir = this.compilerProps('clrDir');

        const projectFilePath = path.join(programDir, 'Program.csproj');
        const coreClrJitPath = path.join(clrBuildDir, 'libclrjit.so');
        const crossgenPath = path.join(clrBuildDir, 'crossgen');

        const programPublishPath = path.join(
            programDir,
            'bin',
            this.buildConfig,
            this.targetFramework,
            this.rID,
            'publish',
        );

        const programDllPath = path.join(programPublishPath, 'Program.dll');

        const projectFileContent =
        `<Project Sdk="Microsoft.NET.Sdk">
            <PropertyGroup>
         <TargetFramework>${this.targetFramework}</TargetFramework>
         <AllowUnsafeBlocks>true</AllowUnsafeBlocks>
         </PropertyGroup>
         </Project>
        `;

        execOptions.customCwd = programDir;
        await fs.writeFile(projectFilePath, projectFileContent);
        options = this.compilerOptions;

        const compilerResult = await super.runCompiler(
            compiler,
            options,
            inputFileName,
            execOptions,
        );

        if (compilerResult.code !== 0)
            return compilerResult;

        const crossgenResult = await this.runCrossgen(
            execOptions,
            crossgenPath,
            coreClrJitPath,
            programPublishPath,
            programDllPath,
            this.getOutputFilename(programDir, ''),
        );

        if (crossgenResult.code !== 0)
            return crossgenResult;

        return compilerResult;
    }

    optionsForFilter() {
        return this.compilerOptions;
    }

    getOutputFilename(dirPath) {
        return path.join(dirPath, `output.s`);
    }

    cleanAsm(asm) {
        var cleanedAsm = '';
        var asmLines = asm.split('\n');

        for (const line of asmLines) {
            if (line.startsWith('; Assembly listing for method')) {
                // ; Assembly listing for method ConsoleApplication.Program:Main(System.String[])
                //                               ^ This character is the 31st character in this string.
                // `substring` removes the first 30 characters from it and uses the rest as a label.
                cleanedAsm = cleanedAsm.concat(line.substring(30) + ':\n');
                continue;
            }

            if (line.startsWith('Microsoft (R)') ||
                line.startsWith('Copyright (c)') ||
                line.startsWith('Native image'))
            {
                continue;
            }

            // Removes the raw opcodes from crossgen's output.
            cleanedAsm = cleanedAsm.concat(line.replace(/ +[\dA-Z]+ +[^\dA-Za-z]/,' ') + '\n');
        }

        return cleanedAsm;
    }

    async runCrossgen(execOptions, crossgenPath, jitPath, publishPath, dllPath, outputPath) {
        execOptions.env['COMPlus_NgenDisasm'] = '*';
        execOptions.env['COMPlus_NgenDiffableDasm'] = '1';

        const crossgenOutput = await this.exec(
            crossgenPath,
            ['-JITPath', jitPath, '-p', publishPath, dllPath],
            execOptions,
        );

        await fs.writeFile(
            outputPath,
            this.cleanAsm(`${crossgenOutput.stdout}\n\n${crossgenOutput.stderr}`),
        );
        return crossgenOutput;
    }
}
