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
 * Language object type
 *
 * @typedef {Object} CELanguage
 * @property {string} id - Id of language. Added programmatically based on CELanguages key
 * @property {string} name - UI display name of the language
 * @property {string} monaco - Monaco Editor language ID (Selects which language Monaco will use to highlight the code)
 * @property {string[]} extensions - Usual extensions associated with the language. First one is used as file input etx
 * @property {string[]} alias - Different ways in which we can also refer to this language
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
        extensions: ['.cpp', '.cxx', '.h', '.hpp', '.hxx', '.c'],
        alias: ['gcc', 'cpp'],
        previewFilter: /^\s*#include/,
    },
    llvm: {
        name: 'LLVM IR',
        monaco: 'llvm-ir',
        extensions: ['.ll'],
        alias: [],
    },
    cppx: {
        name: 'Cppx',
        monaco: 'cppp',
        extensions: ['.cpp', '.cxx', '.h', '.hpp', '.hxx', '.c'],
        alias: [],
        previewFilter: /^\s*#include/,
    },
    cppx_gold: {
        name: 'Cppx-Gold',
        monaco: 'cppx-gold',
        extensions: ['.usyntax', '.cpp', '.cxx', '.h', '.hpp', '.hxx', '.c'],
        alias: [],
    },
    cppx_blue: {
        name: 'Cppx-Blue',
        monaco: 'cppx-blue',
        extensions: ['.blue', '.cpp', '.cxx', '.h', '.hpp', '.hxx', '.c'],
        alias: [],
    },
    c: {
        name: 'C',
        monaco: 'nc',
        extensions: ['.c', '.h'],
        alias: [],
        previewFilter: /^\s*#include/,
    },
    openclc: {
        name: 'OpenCL C',
        monaco: 'openclc',
        extensions: ['.cl', '.ocl'],
        alias: [],
    },
    cpp_for_opencl: {
        name: 'C++ for OpenCL',
        monaco: 'cpp-for-opencl',
        extensions: ['.clcpp', '.cl', '.ocl'],
        alias: [],
    },
    rust: {
        name: 'Rust',
        monaco: 'rust',
        extensions: ['.rs'],
        alias: [],
    },
    d: {
        name: 'D',
        monaco: 'd',
        extensions: ['.d'],
        alias: [],
    },
    go: {
        name: 'Go',
        monaco: 'go',
        extensions: ['.go'],
        alias: [],
    },
    ispc: {
        name: 'ispc',
        monaco: 'ispc',
        extensions: ['.ispc'],
        alias: [],
    },
    haskell: {
        name: 'Haskell',
        monaco: 'haskell',
        extensions: ['.hs', '.haskell'],
        alias: [],
    },
    java: {
        name: 'Java',
        monaco: 'java',
        extensions: ['.java'],
        alias: [],
    },
    kotlin: {
        name: 'Kotlin',
        monaco: 'kotlin',
        extensions: ['.kt'],
        alias: [],
    },
    scala: {
        name: 'Scala',
        monaco: 'scala',
        extensions: ['.scala'],
        alias: [],
    },
    ocaml: {
        name: 'OCaml',
        monaco: 'ocaml',
        extensions: ['.ml', '.mli'],
        alias: [],
    },
    python: {
        name: 'Python',
        monaco: 'python',
        extensions: ['.py'],
        alias: [],
    },
    swift: {
        name: 'Swift',
        monaco: 'swift',
        extensions: ['.swift'],
        alias: [],
    },
    pascal: {
        name: 'Pascal',
        monaco: 'pascal',
        extensions: ['.pas'],
        alias: [],
    },
    fortran: {
        id: 'fortran',
        name: 'Fortran',
        monaco: 'fortran',
        extensions: ['.f90', '.F90', '.f95', '.F95'],
        alias: [],
    },
    assembly: {
        name: 'Assembly',
        monaco: 'asm',
        extensions: ['.asm'],
        alias: ['asm'],
    },
    analysis: {
        name: 'Analysis',
        monaco: 'asm',
        extensions: ['.asm'],  // maybe add more? Change to a unique one?
        alias: ['tool', 'tools'],
    },
    cuda: {
        name: 'CUDA',
        monaco: 'cuda',
        extensions: ['.cu'],
        alias: ['nvcc'],
        monacoDisassembly: 'ptx',
    },
    zig: {
        name: 'Zig',
        monaco: 'zig',
        extensions: ['.zig'],
        alias: [],
    },
    clean: {
        name: 'Clean',
        monaco: 'clean',
        extensions: ['.icl'],
        alias: [],
    },
    ada: {
        name: 'Ada',
        monaco: 'ada',
        extensions: ['.adb', '.ads'],
        alias: [],
    },
    nim: {
        name: 'Nim',
        monaco: 'nim',
        extensions: ['.nim'],
        alias: [],
    },
    crystal: {
        name: 'Crystal',
        monaco: 'crystal',
        extensions: ['.cr'],
        alias: [],
    },
    circle: {
        name: 'C++ (Circle)',
        monaco: 'cppcircle',
        extensions: ['.cpp', '.cxx', '.h', '.hpp', '.hxx', '.c'],
        alias: [],
        previewFilter: /^\s*#include/,
    },
};

_.each(languages, (lang, key) => {
    lang.id = key;
    try {
        lang.example = fs.readFileSync(path.join('examples', lang.id, 'default' + lang.extensions[0]), 'utf8');
    } catch (error) {
        lang.example = 'Oops, something went wrong and we could not get the default code for this language.';
    }
});
