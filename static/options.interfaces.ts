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

import {Language, LanguageKey} from '../types/languages.interfaces.js';
import {CompilerInfo} from '../types/compiler.interfaces.js';
import {Tool} from '../types/tool.interfaces.js';

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

// TODO: Is this the same as OptionsType in lib/options-handler.ts?
export type Options = {
    libs: Libs;
    remoteLibs: LibsPerRemote;
    languages: Partial<Record<LanguageKey, Language>>;
    compilers: CompilerInfo[];
    defaultCompiler: Record<LanguageKey, string>;
    defaultLibs: Record<LanguageKey, string | null>;
    defaultFontScale: number;
    sentryDsn?: string;
    release?: string;
    sentryEnvironment?: string;
    compileOptions: Record<LanguageKey, string>;
    tools: Record<LanguageKey, Record<string, Tool>>;
    slides?: any[];
    cookieDomainRe: string;
    motdUrl: string;
    pageloadUrl: string;
    mobileViewer: boolean;
    readOnly: boolean;
    policies: {
        cookies: {
            enabled: boolean;
            key: string;
        };
        privacy: {
            enabled: boolean;
            key: string;
        };
    };
    supportsExecute: boolean;
    supportsLibraryCodeFilter: boolean;
    cvCompilerCountMax: number;
};
