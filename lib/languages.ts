// Copyright (c) 2017, Compiler Explorer Authors
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
import _ from 'underscore';

import {Language, LanguageKey} from '../types/languages.interfaces';

type DefKeys = 'name' | 'monaco' | 'extensions' | 'alias' | 'previewFilter' | 'formatter' | 'logoUrl' | 'logoUrlDark';
type LanguageDefinition = Pick<Language, DefKeys>;

const definitions: Record<LanguageKey, LanguageDefinition> = {
    jakt: {
        name: 'Jakt',
        monaco: 'jakt',
        extensions: ['.jakt'],
        alias: [],
        logoUrl: '',
        logoUrlDark: null,
        formatter: null,
        previewFilter: null,
    },
    'c++': {
        name: 'C++',
        monaco: 'cppp',
        extensions: ['.cpp', '.cxx', '.h', '.hpp', '.hxx', '.c', '.cc', '.ixx'],
        alias: ['gcc', 'cpp'],
        logoUrl: 'c++.svg',
        logoUrlDark: null,
        formatter: 'clangformat',
        previewFilter: /^\s*#include/,
    },
    ada: {
        name: 'Ada',
        monaco: 'ada',
        extensions: ['.adb', '.ads'],
        alias: [],
        logoUrl: 'ada.svg',
        logoUrlDark: 'ada-dark.svg',
        formatter: null,
        previewFilter: null,
    },
    analysis: {
        name: 'Analysis',
        monaco: 'asm',
        extensions: ['.asm'], // maybe add more? Change to a unique one?
        alias: ['tool', 'tools'],
        logoUrl: 'analysis.png', // TODO: Find a better alternative
        logoUrlDark: null,
        formatter: null,
        previewFilter: null,
    },
    assembly: {
        name: 'Assembly',
        monaco: 'asm',
        extensions: ['.asm', '.6502'],
        alias: ['asm'],
        logoUrl: 'assembly.png', // TODO: Find a better alternative
        logoUrlDark: null,
        formatter: null,
        previewFilter: null,
    },
    c: {
        name: 'C',
        monaco: 'nc',
        extensions: ['.c', '.h'],
        alias: [],
        logoUrl: 'c.svg',
        logoUrlDark: null,
        formatter: 'clangformat',
        previewFilter: /^\s*#include/,
    },
    carbon: {
        name: 'Carbon',
        monaco: 'carbon',
        extensions: ['.carbon'],
        alias: [],
        logoUrl: null,
        logoUrlDark: null,
        formatter: null,
        previewFilter: null,
    },
    circle: {
        name: 'C++ (Circle)',
        monaco: 'cppcircle',
        extensions: ['.cpp', '.cxx', '.h', '.hpp', '.hxx', '.c'],
        alias: [],
        previewFilter: /^\s*#include/,
        logoUrl: 'c++.svg', // TODO: Find a better alternative
        logoUrlDark: null,
        formatter: null,
    },
    circt: {
        name: 'CIRCT',
        monaco: 'mlir',
        extensions: ['.mlir'],
        alias: [],
        logoUrl: 'circt.svg',
        formatter: null,
        logoUrlDark: null,
        previewFilter: null,
    },
    clean: {
        name: 'Clean',
        monaco: 'clean',
        extensions: ['.icl'],
        alias: [],
        logoUrl: 'clean.svg', // TODO: Find a better alternative
        logoUrlDark: null,
        formatter: null,
        previewFilter: null,
    },
    cmake: {
        name: 'CMake',
        monaco: 'cmake',
        extensions: ['.txt'],
        alias: [],
        logoUrl: 'cmake.svg',
        logoUrlDark: null,
        formatter: null,
        previewFilter: null,
    },
    cpp_for_opencl: {
        name: 'C++ for OpenCL',
        monaco: 'cpp-for-opencl',
        extensions: ['.clcpp', '.cl', '.ocl'],
        alias: [],
        logoUrl: 'opencl.svg', // TODO: Find a better alternative
        logoUrlDark: 'opencl-dark.svg',
        formatter: null,
        previewFilter: null,
    },
    mlir: {
        name: 'MLIR',
        monaco: 'mlir',
        extensions: ['.mlir'],
        alias: [],
        logoUrl: 'mlir.svg',
        formatter: null,
        logoUrlDark: null,
        previewFilter: null,
    },
    cppx: {
        name: 'Cppx',
        monaco: 'cppp',
        extensions: ['.cpp', '.cxx', '.h', '.hpp', '.hxx', '.c'],
        alias: [],
        logoUrl: 'c++.svg',
        logoUrlDark: null,
        formatter: null,
        previewFilter: /^\s*#include/,
    },
    cppx_blue: {
        name: 'Cppx-Blue',
        monaco: 'cppx-blue',
        extensions: ['.blue', '.cpp', '.cxx', '.h', '.hpp', '.hxx', '.c'],
        alias: [],
        logoUrl: 'c++.svg', // TODO: Find a better alternative
        logoUrlDark: null,
        formatter: null,
        previewFilter: null,
    },
    cppx_gold: {
        name: 'Cppx-Gold',
        monaco: 'cppx-gold',
        extensions: ['.usyntax', '.cpp', '.cxx', '.h', '.hpp', '.hxx', '.c'],
        alias: [],
        logoUrl: 'c++.svg', // TODO: Find a better alternative
        logoUrlDark: null,
        formatter: null,
        previewFilter: null,
    },
    crystal: {
        name: 'Crystal',
        monaco: 'crystal',
        extensions: ['.cr'],
        alias: [],
        logoUrl: 'crystal.svg',
        logoUrlDark: 'crystal-dark.svg',
        formatter: null,
        previewFilter: null,
    },
    csharp: {
        name: 'C#',
        monaco: 'csharp',
        extensions: ['.cs'],
        alias: [],
        logoUrl: 'dotnet.svg',
        logoUrlDark: null,
        formatter: null,
        previewFilter: null,
    },
    cuda: {
        name: 'CUDA C++',
        monaco: 'cuda',
        extensions: ['.cu'],
        alias: ['nvcc'],
        logoUrl: 'cuda.svg',
        logoUrlDark: 'cuda-dark.svg',
        formatter: null,
        previewFilter: null,
    },
    d: {
        name: 'D',
        monaco: 'd',
        extensions: ['.d'],
        alias: [],
        logoUrl: 'd.svg',
        logoUrlDark: null,
        formatter: null,
        previewFilter: null,
    },
    dart: {
        name: 'Dart',
        monaco: 'dart',
        extensions: ['.dart'],
        alias: [],
        logoUrl: 'dart.svg',
        logoUrlDark: null,
        formatter: 'dartformat',
        previewFilter: null,
    },
    erlang: {
        name: 'Erlang',
        monaco: 'erlang',
        extensions: ['.erl', '.hrl'],
        alias: [],
        logoUrl: 'erlang.svg',
        logoUrlDark: null,
        formatter: null,
        previewFilter: null,
    },
    fortran: {
        name: 'Fortran',
        monaco: 'fortran',
        extensions: ['.f90', '.F90', '.f95', '.F95', '.f'],
        alias: [],
        logoUrl: 'fortran.svg',
        logoUrlDark: null,
        formatter: null,
        previewFilter: null,
    },
    fsharp: {
        name: 'F#',
        monaco: 'fsharp',
        extensions: ['.fs'],
        alias: [],
        logoUrl: 'fsharp.svg',
        logoUrlDark: null,
        formatter: null,
        previewFilter: null,
    },
    go: {
        name: 'Go',
        monaco: 'go',
        extensions: ['.go'],
        alias: [],
        logoUrl: 'go.svg',
        logoUrlDark: null,
        formatter: null,
        previewFilter: null,
    },
    haskell: {
        name: 'Haskell',
        monaco: 'haskell',
        extensions: ['.hs', '.haskell'],
        alias: [],
        logoUrl: 'haskell.png',
        logoUrlDark: null,
        formatter: null,
        previewFilter: null,
    },
    hlsl: {
        name: 'HLSL',
        monaco: 'hlsl',
        extensions: ['.hlsl', '.hlsli'],
        alias: [],
        logoUrl: null,
        logoUrlDark: null,
        formatter: null,
        previewFilter: null,
    },
    ispc: {
        name: 'ispc',
        monaco: 'ispc',
        extensions: ['.ispc'],
        alias: [],
        logoUrl: 'ispc.png',
        logoUrlDark: null,
        formatter: null,
        previewFilter: null,
    },
    java: {
        name: 'Java',
        monaco: 'java',
        extensions: ['.java'],
        alias: [],
        logoUrl: 'java.svg',
        logoUrlDark: null,
        formatter: null,
        previewFilter: null,
    },
    kotlin: {
        name: 'Kotlin',
        monaco: 'kotlin',
        extensions: ['.kt'],
        alias: [],
        logoUrl: 'kotlin.png',
        logoUrlDark: null,
        formatter: null,
        previewFilter: null,
    },
    llvm: {
        name: 'LLVM IR',
        monaco: 'llvm-ir',
        extensions: ['.ll'],
        alias: [],
        logoUrl: 'llvm.png',
        logoUrlDark: null,
        formatter: null,
        previewFilter: null,
    },
    nim: {
        name: 'Nim',
        monaco: 'nim',
        extensions: ['.nim'],
        alias: [],
        logoUrl: 'nim.svg',
        logoUrlDark: null,
        formatter: null,
        previewFilter: null,
    },
    ocaml: {
        name: 'OCaml',
        monaco: 'ocaml',
        extensions: ['.ml', '.mli'],
        alias: [],
        logoUrl: 'ocaml.svg',
        logoUrlDark: null,
        formatter: null,
        previewFilter: null,
    },
    openclc: {
        name: 'OpenCL C',
        monaco: 'openclc',
        extensions: ['.cl', '.ocl'],
        alias: [],
        logoUrl: 'opencl.svg',
        logoUrlDark: 'opencl-dark.svg',
        formatter: null,
        previewFilter: null,
    },
    pascal: {
        name: 'Pascal',
        monaco: 'pascal',
        extensions: ['.pas', '.dpr'],
        alias: [],
        logoUrl: 'pascal.svg', // TODO: Find a better alternative
        logoUrlDark: 'pascal-dark.svg',
        formatter: null,
        previewFilter: null,
    },
    pony: {
        name: 'Pony',
        monaco: 'pony',
        extensions: ['.pony'],
        alias: [],
        logoUrl: 'pony.svg',
        logoUrlDark: null,
        formatter: null,
        previewFilter: null,
    },
    python: {
        name: 'Python',
        monaco: 'python',
        extensions: ['.py'],
        alias: [],
        logoUrl: 'python.svg',
        logoUrlDark: null,
        formatter: null,
        previewFilter: null,
    },
    ruby: {
        name: 'Ruby',
        monaco: 'ruby',
        extensions: ['.rb'],
        alias: [],
        logoUrl: 'ruby.svg',
        logoUrlDark: null,
        formatter: null,
        previewFilter: null,
    },
    rust: {
        name: 'Rust',
        monaco: 'rust',
        extensions: ['.rs'],
        alias: [],
        logoUrl: 'rust.svg',
        logoUrlDark: 'rust-dark.svg',
        formatter: 'rustfmt',
        previewFilter: null,
    },
    scala: {
        name: 'Scala',
        monaco: 'scala',
        extensions: ['.scala'],
        alias: [],
        logoUrl: 'scala.png',
        logoUrlDark: null,
        formatter: null,
        previewFilter: null,
    },
    solidity: {
        name: 'Solidity',
        monaco: 'sol',
        extensions: ['.sol'],
        alias: [],
        logoUrl: 'solidity.svg',
        logoUrlDark: null,
        formatter: null,
        previewFilter: null,
    },
    swift: {
        name: 'Swift',
        monaco: 'swift',
        extensions: ['.swift'],
        alias: [],
        logoUrl: 'swift.svg',
        logoUrlDark: null,
        formatter: null,
        previewFilter: null,
    },
    toit: {
        name: 'Toit',
        monaco: 'toit',
        extensions: ['.toit'],
        alias: [],
        logoUrl: 'toit.svg',
        logoUrlDark: null,
        formatter: null,
        previewFilter: null,
    },
    typescript: {
        name: 'TypeScript Native',
        monaco: 'typescript',
        extensions: ['.ts', '.d.ts'],
        alias: [],
        logoUrl: 'ts.svg',
        logoUrlDark: null,
        formatter: null,
        previewFilter: null,
    },
    vb: {
        name: 'Visual Basic',
        monaco: 'vb',
        extensions: ['.vb'],
        alias: [],
        logoUrl: 'dotnet.svg',
        logoUrlDark: null,
        formatter: null,
        previewFilter: null,
    },
    zig: {
        name: 'Zig',
        monaco: 'zig',
        extensions: ['.zig'],
        alias: [],
        logoUrl: 'zig.svg',
        logoUrlDark: null,
        formatter: null,
        previewFilter: null,
    },
};

export const languages: Record<LanguageKey, Language> = _.mapObject(definitions, (lang, key) => {
    let example: string;
    try {
        example = fs.readFileSync(path.join('examples', key, 'default' + lang.extensions[0]), 'utf8');
    } catch (error) {
        example = 'Oops, something went wrong and we could not get the default code for this language.';
    }

    const def: Language = {
        ...lang,
        id: key as LanguageKey,
        supportsExecute: false,
        example,
    };
    return def;
});
