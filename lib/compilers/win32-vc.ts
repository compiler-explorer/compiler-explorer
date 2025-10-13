// Copyright (c) 2018, Microsoft Corporation
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

import path from 'node:path';

import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import {CompilationEnvironment} from '../compilation-env.js';
import {VcAsmParser} from '../parsers/asm-parser-vc.js';

import {VCParser} from './argument-parsers.js';
import {Win32Compiler} from './win32.js';

export class Win32VcCompiler extends Win32Compiler {
    static override get key() {
        return 'win32-vc';
    }

    constructor(info: PreliminaryCompilerInfo, env: CompilationEnvironment) {
        super(info, env);
        this.asm = new VcAsmParser(this.compilerProps);
    }

    override getArgumentParserClass() {
        return VCParser;
    }

    override getExtraCMakeArgs(key: any): string[] {
        const args: string[] = [];

        // Set CMAKE_C_COMPILER and CMAKE_CXX_COMPILER explicitly to prevent CMake from detecting MinGW
        const compilerDir = path.dirname(this.compiler.exe);
        const clExe = this.compiler.exe.replace(/\\/g, '/');

        // For MSVC, cl.exe handles both C and C++
        args.push(`-DCMAKE_C_COMPILER=${clExe}`);
        args.push(`-DCMAKE_CXX_COMPILER=${clExe}`);

        // Set CMAKE_LINKER, CMAKE_RC_COMPILER, and CMAKE_MT to prevent CMake from picking up MinGW tools
        const linkExe = path.join(compilerDir, 'link.exe').replace(/\\/g, '/');
        args.push(`-DCMAKE_LINKER=${linkExe}`);

        // Try to find rc.exe and mt.exe in Windows SDK if available
        if (this.compiler.includePath) {
            // includePath typically contains SDK include paths
            // SDK structure: Z:\compilers\windows-kits-10\bin\rc.exe
            const includePathMatch = this.compiler.includePath.match(/([^;]+windows-kits-[^;]+)/i);
            if (includePathMatch) {
                const sdkPath = includePathMatch[1].split(/[/\\]include/i)[0];
                const rcExe = path.join(sdkPath, 'bin', 'rc.exe').replace(/\\/g, '/');
                const mtExe = path.join(sdkPath, 'bin', 'mt.exe').replace(/\\/g, '/');
                args.push(`-DCMAKE_RC_COMPILER=${rcExe}`);
                args.push(`-DCMAKE_MT=${mtExe}`);
            }
        }

        return args;
    }
}
