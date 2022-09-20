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

import {Language} from '../types/languages.interfaces';
import {CompilerInfo} from '../types/compiler.interfaces';

export type LibraryVersion = {
    alias: string[];
    hidden: boolean;
    libId: string;
    used: boolean;
    version?: string;
};

export type Library = {
    dependencies: string[];
    description?: string;
    examples?: string[];
    name?: string;
    url?: string;
    versions: Record<string, LibraryVersion>;
};

export type LanguageLibs = Record<string, Library>;

export type Libs = Record<string, LanguageLibs>;

export type LibsPerRemote = Record<string, LanguageLibs>;

export type Options = {
    libs: Libs;
    remoteLibs: LibsPerRemote;
    languages: Record<string, Language>;
    compilers: CompilerInfo[];
    defaultCompiler: Record<string, string>;
    defaultLibs: Record<string, string | null>;
    defaultFontScale: number;
    sentryDsn?: string;
    release?: string;
    sentryEnvironment?: string;
};
