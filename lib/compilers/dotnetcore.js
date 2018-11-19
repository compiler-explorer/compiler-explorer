// Copyright (c) 2018, Compiler Explorer Team
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

const
    BaseCompiler = require('../base-compiler'),
    exec = require('../../lib/exec'),
    fs = require('fs-extra'),
    path = require('path'),
    AsmParser = require('../asm-parser.js');

const ProjectFileContent = 
`<Project Sdk="Microsoft.NET.Sdk">

<PropertyGroup>
  <TargetFramework>netcoreapp2.1</TargetFramework>
  <RuntimeIdentifier>win10-x64</RuntimeIdentifier>
</PropertyGroup>

</Project>
`;

class DotNetCoreCompiler extends BaseCompiler {

    constructor(info, env) {
        super(info, env);
        this.asm = new AsmParser(
            this.compilerProps,
            /^([.A-Za-z_$@][A-Za-z0-9$_@.]*):/i,
            /^\s*(((#|@|;|\/\/)((?!Assembly listing for method).)*)|(\/\*.*\*\/))$/,
            true);
    }

    optionsForFilter(filters, outputFilename) {
        filters.preProcessLines = (asmLines) => this.preProcessLines(asmLines);
        return [outputFilename];
    }

    preProcessLines(asmLines) {
        for (const [index, line] of asmLines.entries()) {
            // remove hex dump of each opcode that precedes the assembly
            const hexDump = /       ([A-Z]|[0-9])*( *)/;
            asmLines[index] = line.replace(hexDump, "        ");

            if (line.startsWith('Microsoft (R) CoreCLR Native Image Generator')) {
                asmLines.length = index;
                break;
            }
        }
        return asmLines;
    }

    runCompiler(compiler, options, inputFilename, execOptions) {
        if (!execOptions) {
            execOptions = this.getDefaultExecOptions();
        }

        execOptions.env = Object.assign({}, execOptions.env);
        execOptions.env['COMPlus_NgenDisasm'] = '*';
        execOptions.env['COMPlus_NgenDiffableDisasm'] = '1';

        for (const [env, to] of this.compiler.envVars) {
            execOptions.env[env] = to;
        }

        execOptions.customCwd = path.dirname(inputFilename);

        const projectFilePath = path.join(execOptions.customCwd, "Project.csproj");
        this.writeFile(projectFilePath, ProjectFileContent);

        const outputPath = this.filename(options[0]);

        options = ['publish', '-c', 'Release', '--self-contained'];

        const coreClrPath = 'C:\\GitHub\\coreclr\\bin\\Product\\Windows_NT.x64.Debug';
        const publishPath = path.join(execOptions.customCwd, 'bin', 'Release', 'netcoreapp2.1', 'win10-x64', 'publish');
        const crossgenPath = path.join(publishPath, 'crossgen');
        const dllPath = path.join(publishPath, 'Project.dll');
        return super.runCompiler(compiler, options, inputFilename, execOptions).
            then(compilerRet => fs.copy(coreClrPath, publishPath).
            then(() => exec.execute(crossgenPath, ['/Platform_Assemblies_Paths', publishPath, dllPath], execOptions)).
            then(ret => this.writeFile(outputPath, ret.stdout + '\n\n' + ret.stderr)).
            then(() => compilerRet)
        );
    }
}

module.exports = DotNetCoreCompiler;
