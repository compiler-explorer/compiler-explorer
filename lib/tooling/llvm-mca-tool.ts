// Copyright (c) 2023, Compiler Explorer Authors
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

import fs from 'fs-extra';

// import {CompilationInfo} from '../../types/compilation/compilation.interfaces.js';
import {CompilationInfo} from '../../types/compilation/compilation.interfaces.js';
import {InstructionSets} from '../instructionsets.js';

import {BaseTool} from './base-tool.js';

export class LLVMMcaTool extends BaseTool {
    static get key() {
        return 'llvm-mca-tool';
    }

    rewriteAsm(asm: string) {
        return asm
            .replaceAll(/.hword\s/gim, '.short ')
            .replaceAll(/offset flat:/gim, '')
            .replaceAll(/ptr\s%fs/gim, 'PTR fs')
            .replaceAll(/^\s*\.(fnstart|eabi_attribute|fpu).*/gim, ''); // https://github.com/compiler-explorer/compiler-explorer/issues/1270
    }

    writeAsmFile(data: string, destination: string) {
        return fs.writeFile(destination, this.rewriteAsm(data));
    }

    override async runTool(compilationInfo: CompilationInfo, _inputFilepath?: string, args?: string[]) {
        const isets = new InstructionSets();
        let target = isets.getInstructionSetTarget(compilationInfo.compiler.instructionSet);
        const prependArgs: string[] = [];

        // Check if compiler target is overridden with --target=<foo> or -target <foo>.
        let argIsTarget = false;
        for (const arg of compilationInfo.options) {
            if (arg.startsWith('--target=')) target = arg.replace(/^--target=/, '');
            if (argIsTarget) {
                target = arg;
                argIsTarget = false;
            }
            if (arg === '-target') argIsTarget = true;
        }

        if (target) prependArgs.push('-mtriple=' + target);

        let haveCPU = false;
        let haveM32 = false;
        for (const arg of compilationInfo.options) {
            if (arg.startsWith('-march') && (target?.startsWith('x86_64') || target?.startsWith('i386'))) {
                prependArgs.push(arg.replace(/^-march=/, '-mcpu='));
                haveCPU = true;
            }
            if (arg === '-m32') haveM32 = true;
            if (arg.startsWith('-mcpu')) {
                prependArgs.push(arg);
                haveCPU = true;
            }
        }

        if (compilationInfo.filters.binary || compilationInfo.filters.binaryObject) {
            return this.createErrorResponse('<cannot run analysis on binary>');
        }

        // If no other CPU specified, default to mcpu=generic to
        // override llvm-mca's default of -mcpu=native, otherwise MCA's
        // CPU analysis varies according to the machine MCA happens to
        // run on which can change over time.
        if (!haveCPU) {
            prependArgs.push('-mcpu=generic');
        }

        if (target?.startsWith('x86_64') && haveM32) {
            prependArgs.push('-march=x86'); // (i.e. i686).
        }

        // Prepend the detected arguments, so that user specified come
        // later (and therefore can be used to override the detected
        // ones above).
        const newArgs: string[] = prependArgs.concat(args || []);

        const rewrittenOutputFilename = compilationInfo.outputFilename + '.mca';
        await this.writeAsmFile(compilationInfo.asm as string, rewrittenOutputFilename);
        return super.runTool(compilationInfo, rewrittenOutputFilename, newArgs);
    }
}
