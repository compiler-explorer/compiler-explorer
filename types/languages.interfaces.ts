// Copyright (c) 2022, Compiler Explorer Authors
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

export type LanguageKey =
    | 'mlir'
    | 'jakt'
    | 'c++'
    | 'llvm'
    | 'cppx'
    | 'cppx_gold'
    | 'cppx_blue'
    | 'c'
    | 'openclc'
    | 'cpp_for_opencl'
    | 'rust'
    | 'd'
    | 'erlang'
    | 'go'
    | 'ispc'
    | 'haskell'
    | 'hlsl'
    | 'java'
    | 'kotlin'
    | 'scala'
    | 'ocaml'
    | 'pony'
    | 'python'
    | 'swift'
    | 'pascal'
    | 'fortran'
    | 'assembly'
    | 'analysis'
    | 'cuda'
    | 'zig'
    | 'clean'
    | 'ada'
    | 'nim'
    | 'crystal'
    | 'circle'
    | 'circt'
    | 'ruby'
    | 'cmake'
    | 'csharp'
    | 'fsharp'
    | 'vb'
    | 'dart'
    | 'typescript'
    | 'solidity'
    | 'toit'
    | 'carbon';

export interface Language {
    /** Id of language. Added programmatically based on CELanguages key */
    id: LanguageKey;
    /** UI display name of the language */
    name: string;
    /** Monaco Editor language ID (Selects which language Monaco will use to highlight the code) */
    monaco: string;
    /** Usual extensions associated with the language. First one is used as file input extension */
    extensions: [string, ...string[]];
    /** Different ways in which we can also refer to this language */
    alias: string[];
    /** Format API name to use (See https://godbolt.org/api/formats) */
    formatter: string | null;
    /** Whether there's at least 1 compiler in this language that supportsExecute */
    supportsExecute: boolean | null;
    /** Path in /views/resources/logos to the logo of the language */
    logoUrl: string | null;
    /** Path in /views/resources/logos to the logo of the language for dark mode use */
    logoUrlDark: string | null;
    /** Example code to show in the language's editor */
    example: string;
    previewFilter: RegExp | null;
}
