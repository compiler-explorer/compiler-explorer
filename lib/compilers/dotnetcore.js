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

// these could be put into DotNetCoreCompiler to allow for more configurations
const BuildConfiguration = 'Release';
const TargetFramework = 'netcoreapp2.1';
const RuntimeIdentifier = 'win10-x64';
const ProjectFileContent = 
`<Project Sdk="Microsoft.NET.Sdk">

<PropertyGroup>
  <TargetFramework>${TargetFramework}</TargetFramework>
  <RuntimeIdentifier>${RuntimeIdentifier}</RuntimeIdentifier>
  <AllowUnsafeBlocks>true</AllowUnsafeBlocks>
</PropertyGroup>

</Project>
`;

class DotNetCoreCompiler extends BaseCompiler {

    constructor(info, env) {
        super(info, env);
        this.asm = new AsmParser(this.compilerProps, {
            labelDef: /^([.A-Za-z_$@][A-Za-z0-9$_@.]*):/i,
            commentOnly: /^\s*(((#|@|;|\/\/)((?!Assembly listing for method).)*)|(\/\*.*\*\/))$/,
            removeBlankLines: true,
        });
    }

    optionsForFilter(filters, outputFilename) {
        filters.preProcessLines = (asmLines) => this.preProcessLines(asmLines);
        return [outputFilename];
    }

    preProcessLines(asmLines) {
        for (const [index, line] of asmLines.entries()) {
            // remove hex dump of each opcode that precedes the assembly
            const hexDump = /( *)([A-Z]|[0-9])*( *)/;
            asmLines[index] = line.replace(hexDump, (_, spaces) => spaces);

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
        for (const [env, to] of this.compiler.envVars) {
            execOptions.env[env] = to;
        }

        execOptions.customCwd = path.dirname(inputFilename);

        const projectFilePath = path.join(execOptions.customCwd, "Project.csproj");
        const outputPath = this.filename(options[0]);

        options = ['publish', '-c', BuildConfiguration, '--self-contained'];

        const coreClrPath = execOptions.env['CoreCLRDebugPath'];
        const publishPath = path.join(
            execOptions.customCwd, 'bin', BuildConfiguration,
            TargetFramework, RuntimeIdentifier, 'publish');
        const crossgenPath = path.join(publishPath, 'crossgen');
        const dllPath = path.join(publishPath, 'Project.dll');

        return this.writeFile(projectFilePath, ProjectFileContent).
            then(() => super.runCompiler(compiler, options, inputFilename, execOptions).
            // copy the debug CoreCLR executables (they need to be in the same directory as the DLL for some reason)
            then(compilerRet => fs.copy(coreClrPath, publishPath). 
            then(() => this.runCrossgen(execOptions, crossgenPath, publishPath, dllPath, outputPath)).
            then(() => compilerRet)));
    }

    runCrossgen(execOptions, crossgenPath, publishPath, dllPath, outputPath) {
        // run crossgen (managed to native)
        // set these environment variables to activate the disassembly output
        execOptions.env['COMPlus_NgenDisasm'] = '*';
        execOptions.env['COMPlus_NgenDiffableDasm'] = '1';
        return exec.execute(crossgenPath, ['/Platform_Assemblies_Paths', publishPath, dllPath], execOptions).
            then(crossgenRet => this.writeFile(outputPath, `${crossgenRet.stdout}\n\n${crossgenRet.stderr}`));
    }
}

module.exports = DotNetCoreCompiler;
