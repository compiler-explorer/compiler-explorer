// Copyright (c) 2026, Compiler Explorer Authors
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

import type {LanguageKey} from '../types/languages.interfaces.js';

/**
 * Identifies a build system. This value appears in the API route, in the compilation cache key (as `api`) and in the
 * stats, so existing ids must never change.
 */
export type BuildSystemId = 'cmake' | 'cargo' | 'maven' | 'make';

/** What a tree is built with, or `'none'` for a plain multi-file project compiled directly. */
export type TreeBuildSystem = BuildSystemId | 'none';

/**
 * What both the frontend and the backend need to know about a build system. The backend additionally has a driver per
 * build system (`lib/build-systems/`) that knows how to actually run it.
 */
export type BuildSystemDescriptor = {
    id: BuildSystemId;
    /** Shown in the UI, e.g. in the tree pane's build system picker */
    name: string;
    /** The file a project's main source is written to */
    manifestFilename: string;
    /** The language the manifest is highlighted as */
    manifestLanguageId: LanguageKey;
    /** The languages this build system can build, or 'all' for one that does not care what it is driving */
    compatibleLanguageIds: LanguageKey[] | 'all';
    /** Prefilled into the build system arguments input */
    defaultArgs: string;
    /** Placeholder for the build system arguments input */
    argsPlaceholder: string;
};

export const BuildSystems: Record<BuildSystemId, BuildSystemDescriptor> = {
    cmake: {
        id: 'cmake',
        name: 'CMake',
        manifestFilename: 'CMakeLists.txt',
        manifestLanguageId: 'cmake',
        compatibleLanguageIds: ['c++', 'c', 'fortran', 'cuda', 'assembly'],
        defaultArgs: '-DCMAKE_BUILD_TYPE=Debug',
        argsPlaceholder: 'CMake arguments...',
    },
    cargo: {
        id: 'cargo',
        name: 'Cargo',
        manifestFilename: 'Cargo.toml',
        manifestLanguageId: 'cargo',
        compatibleLanguageIds: ['rust'],
        defaultArgs: '',
        argsPlaceholder: 'Cargo arguments...',
    },
    make: {
        id: 'make',
        name: 'Make',
        manifestFilename: 'Makefile',
        manifestLanguageId: 'makefile',
        // A Makefile says what to run itself, so there is no language this cannot be pointed at.
        compatibleLanguageIds: 'all',
        defaultArgs: '',
        argsPlaceholder: 'Make arguments and targets...',
    },
    maven: {
        id: 'maven',
        name: 'Maven',
        manifestFilename: 'pom.xml',
        manifestLanguageId: 'maven',
        compatibleLanguageIds: ['java', 'kotlin'],
        defaultArgs: '',
        argsPlaceholder: 'Maven arguments...',
    },
};

export function isBuildSystemId(id: string): id is BuildSystemId {
    return Object.prototype.hasOwnProperty.call(BuildSystems, id);
}

export function getBuildSystemDescriptor(id: string): BuildSystemDescriptor | undefined {
    return isBuildSystemId(id) ? BuildSystems[id] : undefined;
}

/** The build systems that can build the given language. */
export function getBuildSystemsForLanguage(langId: LanguageKey): BuildSystemDescriptor[] {
    return Object.values(BuildSystems).filter(
        buildSystem =>
            buildSystem.compatibleLanguageIds === 'all' || buildSystem.compatibleLanguageIds.includes(langId),
    );
}
