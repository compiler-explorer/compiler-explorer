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

import type {CompilerInfo} from '../../types/compiler.interfaces.js';

import {getParserByKey, Node, Edge, AssemblyLine} from './cfg-parsers/index.js';
import {getInstructionSetByKey} from './instruction-sets/index.js';

// TODO(jeremy-rifkin):
// I've done some work to split out the compiler / instruction set logic
// We'll want to do some work to fill in information for instruction sets and other compilers
// A good comparison https://godbolt.org/z/8EvqoWhYo
// MSVC especially is a little weird, LLVM is also a much different structure than normal asm

function isLLVMBased({compilerType, version}: CompilerInfo) {
    return (
        version.includes('clang') ||
        version.includes('LLVM') ||
        version.includes('rustc') ||
        compilerType === 'swift' ||
        compilerType === 'zig' ||
        compilerType === 'ispc'
    );
}

export type CFG = {
    nodes: Node[];
    edges: Edge[];
};

export function generateStructure(compilerInfo: CompilerInfo, asmArr: AssemblyLine[], isLlvmIr: boolean) {
    // figure out what we're working with
    const isa = isLlvmIr ? 'llvm' : compilerInfo.instructionSet;
    const compilerGroup = isLlvmIr ? 'llvm' : isLLVMBased(compilerInfo) ? 'clang' : compilerInfo.group;
    const instructionSet = new (getInstructionSetByKey(isa ?? 'base'))();
    const parser = new (getParserByKey(compilerGroup))(instructionSet);

    const code = parser.filterData(asmArr);
    const functions = parser.splitToFunctions(code);
    const result: Record<string, CFG> = {};
    for (const fn of functions) {
        result[parser.getFnName(code, fn)] = parser.generateFunctionCfg(code, fn);
    }

    return result;
}
