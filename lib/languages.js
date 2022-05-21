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

/***
 * TODO: Use the types/languages once available
 * Language object type
 *
 * @typedef {Object} CELanguage
 * @property {string} id - Id of language. Added programmatically based on CELanguages key
 * @property {string} name - UI display name of the language
 * @property {string} monaco - Monaco Editor language ID (Selects which language Monaco will use to highlight the code)
 * @property {string[]} extensions - Usual extensions associated with the language. First one is used as file input etx
 * @property {string[]} alias - Different ways in which we can also refer to this language
 * @property {string} [formatter] - Format API name to use (See https://godbolt.org/api/formats)
 * @property {boolean} supportsExecute - Whether there's at least 1 compiler in this language that supportsExecute
 */

/***
 * Currently supported languages on Compiler Explorer
 *
 * @typedef {Object.<string, CELanguage>} CELanguages
 */

/***
 * Current supported languages
 * @type {CELanguages}
 */
export const languages = {
    'c++': {
        name: 'C++',
        monaco: 'cppp',
        extensions: ['.cpp', '.cxx', '.h', '.hpp', '.hxx', '.c', '.cc', '.ixx'],
        alias: ['gcc', 'cpp'],
        previewFilter: /^\s*#include/,
        formatter: 'clangformat',
        logoUrl: 'c++.svg',
    },
    llvm: {
        name: 'LLVM IR',
        monaco: 'llvm-ir',
        extensions: ['.ll'],
        alias: [],
        logoUrl: 'llvm.png',
    },
    cppx: {
        name: 'Cppx',
        monaco: 'cppp',
        extensions: ['.cpp', '.cxx', '.h', '.hpp', '.hxx', '.c'],
        alias: [],
        previewFilter: /^\s*#include/,
        logoUrl: 'c++.svg',
    },
    cppx_gold: {
        name: 'Cppx-Gold',
        monaco: 'cppx-gold',
        extensions: ['.usyntax', '.cpp', '.cxx', '.h', '.hpp', '.hxx', '.c'],
        alias: [],
        logoUrl: 'c++.svg', // TODO: Find a better alternative
    },
    cppx_blue: {
        name: 'Cppx-Blue',
        monaco: 'cppx-blue',
        extensions: ['.blue', '.cpp', '.cxx', '.h', '.hpp', '.hxx', '.c'],
        alias: [],
        logoUrl: 'c++.svg', // TODO: Find a better alternative
    },
    c: {
        name: 'C',
        monaco: 'nc',
        extensions: ['.c', '.h'],
        alias: [],
        previewFilter: /^\s*#include/,
        logoUrl: 'c.svg',
    },
    openclc: {
        name: 'OpenCL C',
        monaco: 'openclc',
        extensions: ['.cl', '.ocl'],
        alias: [],
        logoUrl: 'opencl.svg',
        logoUrlDark: 'opencl-dark.svg',
    },
    cpp_for_opencl: {
        name: 'C++ for OpenCL',
        monaco: 'cpp-for-opencl',
        extensions: ['.clcpp', '.cl', '.ocl'],
        alias: [],
        logoUrl: 'opencl.svg', // TODO: Find a better alternative
        logoUrlDark: 'opencl-dark.svg',
    },
    rust: {
        name: 'Rust',
        monaco: 'rust',
        extensions: ['.rs'],
        alias: [],
        formatter: 'rustfmt',
        logoUrl: 'rust.svg',
        logoUrlDark: 'rust-dark.svg',
    },
    d: {
        name: 'D',
        monaco: 'd',
        extensions: ['.d'],
        alias: [],
        logoUrl: 'd.svg',
    },
    erlang: {
        name: 'Erlang',
        monaco: 'erlang',
        extensions: ['.erl', '.hrl'],
        alias: [],
        logoUrl: 'erlang.svg',
    },
    go: {
        name: 'Go',
        monaco: 'go',
        extensions: ['.go'],
        alias: [],
        logoUrl: 'go.svg',
    },
    ispc: {
        name: 'ispc',
        monaco: 'ispc',
        extensions: ['.ispc'],
        alias: [],
        logoUrl: 'ispc.png',
    },
    haskell: {
        name: 'Haskell',
        monaco: 'haskell',
        extensions: ['.hs', '.haskell'],
        alias: [],
        logoUrl: 'haskell.png',
    },
    java: {
        name: 'Java',
        monaco: 'java',
        extensions: ['.java'],
        alias: [],
        logoUrl: 'java.svg',
    },
    kotlin: {
        name: 'Kotlin',
        monaco: 'kotlin',
        extensions: ['.kt'],
        alias: [],
        logoUrl: 'kotlin.png',
    },
    scala: {
        name: 'Scala',
        monaco: 'scala',
        extensions: ['.scala'],
        alias: [],
        logoUrl: 'scala.png',
    },
    ocaml: {
        name: 'OCaml',
        monaco: 'ocaml',
        extensions: ['.ml', '.mli'],
        alias: [],
        logoUrl: 'ocaml.svg',
    },
    python: {
        name: 'Python',
        monaco: 'python',
        extensions: ['.py'],
        alias: [],
        logoUrl: 'python.svg',
    },
    swift: {
        name: 'Swift',
        monaco: 'swift',
        extensions: ['.swift'],
        alias: [],
        logoUrl: 'swift.svg',
    },
    pascal: {
        name: 'Pascal',
        monaco: 'pascal',
        extensions: ['.pas', '.dpr'],
        alias: [],
        logoUrl: 'pascal.svg', // TODO: Find a better alternative
        logoUrlDark: 'pascal-dark.svg',
    },
    fortran: {
        id: 'fortran',
        name: 'Fortran',
        monaco: 'fortran',
        extensions: ['.f90', '.F90', '.f95', '.F95', '.f'],
        alias: [],
        logoUrl: 'fortran.svg',
    },
    assembly: {
        name: 'Assembly',
        monaco: 'asm',
        extensions: ['.asm', '.6502'],
        alias: ['asm'],
        logoUrl: 'assembly.png', // TODO: Find a better alternative
    },
    analysis: {
        name: 'Analysis',
        monaco: 'asm',
        extensions: ['.asm'], // maybe add more? Change to a unique one?
        alias: ['tool', 'tools'],
        logoUrl: 'analysis.png', // TODO: Find a better alternative
    },
    cuda: {
        name: 'CUDA C++',
        monaco: 'cuda',
        extensions: ['.cu'],
        alias: ['nvcc'],
        monacoDisassembly: 'ptx',
        logoUrl: 'cuda.svg',
        logoUrlDark: 'cuda-dark.svg',
    },
    zig: {
        name: 'Zig',
        monaco: 'zig',
        extensions: ['.zig'],
        alias: [],
        logoUrl: 'zig.svg',
    },
    clean: {
        name: 'Clean',
        monaco: 'clean',
        extensions: ['.icl'],
        alias: [],
        logoUrl: 'clean.svg', // TODO: Find a better alternative
    },
    ada: {
        name: 'Ada',
        monaco: 'ada',
        extensions: ['.adb', '.ads'],
        alias: [],
        logoUrl: 'ada.svg',
        logoUrlDark: 'ada-dark.svg',
    },
    nim: {
        name: 'Nim',
        monaco: 'nim',
        extensions: ['.nim'],
        alias: [],
        logoUrl: 'nim.svg',
    },
    crystal: {
        name: 'Crystal',
        monaco: 'crystal',
        extensions: ['.cr'],
        alias: [],
        logoUrl: 'crystal.svg',
        logoUrlDark: 'crystal-dark.svg',
    },
    circle: {
        name: 'C++ (Circle)',
        monaco: 'cppcircle',
        extensions: ['.cpp', '.cxx', '.h', '.hpp', '.hxx', '.c'],
        alias: [],
        previewFilter: /^\s*#include/,
        logoUrl: 'c++.svg', // TODO: Find a better alternative
    },
    ruby: {
        name: 'Ruby',
        monaco: 'ruby',
        extensions: ['.rb'],
        alias: [],
        monacoDisassembly: 'asmruby',
        logoUrl: 'ruby.svg',
    },
    cmake: {
        name: 'CMake',
        monaco: 'cmake',
        extensions: ['.txt'],
        alias: [],
        logoUrl: 'cmake.svg',
    },
    csharp: {
        name: 'C#',
        monaco: 'csharp',
        extensions: ['.cs'],
        alias: [],
        logoUrl: 'dotnet.svg',
    },
    fsharp: {
        name: 'F#',
        monaco: 'fsharp',
        extensions: ['.fs'],
        alias: [],
        logoUrl: 'fsharp.svg',
    },
    vb: {
        name: 'Visual Basic',
        monaco: 'vb',
        extensions: ['.vb'],
        alias: [],
        logoUrl: 'dotnet.svg',
    },
    dart: {
        name: 'Dart',
        monaco: 'dart',
        extensions: ['.dart'],
        alias: [],
        formatter: 'dartformat',
        logoUrl: 'dart.svg',
    },
    typescript: {
        name: 'TypeScript',
        monaco: 'typescript',
        extensions: ['.ts', '.d.ts'],
        alias: [],
        logoUrl: 'ts.svg',
    },
    solidity: {
        name: 'Solidity',
        monaco: 'sol',
        extensions: ['.sol'],
        alias: [],
        logoUrl: 'solidity.svg',
    },
};

_.each(languages, (lang, key) => {
    lang.id = key;
    lang.supportsExecute = false;
    try {
        lang.example = fs.readFileSync(path.join('examples', lang.id, 'default' + lang.extensions[0]), 'utf8');
    } catch (error) {
        lang.example = 'Oops, something went wrong and we could not get the default code for this language.';
    }
});
