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
    /**
     * What the build's artifact is called when `customOutputFilename` does not say. `output.s` is the name a plain
     * compilation gives its output, so every build system that produces a binary keeps it, and a project only has to
     * know one name. Maven is the exception: what it builds is a jar, and calling one `output.s` helps nobody.
     *
     * Also the placeholder for the tree pane's output file input. CMake derives the same name from the compiler
     * instead of reading this, because a compiler is free to override it -- the Windows ones want a `.exe`.
     */
    defaultArtifactName: string;
};

export const BuildSystems: Record<BuildSystemId, BuildSystemDescriptor> = {
    cmake: {
        id: 'cmake',
        name: 'CMake',
        manifestFilename: 'CMakeLists.txt',
        manifestLanguageId: 'cmake',
        compatibleLanguageIds: ['c++', 'c', 'fortran', 'cuda', 'assembly', 'rust'],
        defaultArgs: '-DCMAKE_BUILD_TYPE=Debug',
        argsPlaceholder: 'CMake arguments...',
        defaultArtifactName: 'output.s',
    },
    cargo: {
        id: 'cargo',
        name: 'Cargo',
        manifestFilename: 'Cargo.toml',
        manifestLanguageId: 'cargo',
        compatibleLanguageIds: ['rust'],
        defaultArgs: '',
        argsPlaceholder: 'Cargo arguments...',
        defaultArtifactName: 'output.s',
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
        defaultArtifactName: 'output.s',
    },
    maven: {
        id: 'maven',
        name: 'Maven',
        manifestFilename: 'pom.xml',
        manifestLanguageId: 'maven',
        compatibleLanguageIds: ['java', 'kotlin'],
        defaultArgs: '',
        argsPlaceholder: 'Maven arguments...',
        defaultArtifactName: 'output.jar',
    },
};

export function isBuildSystemId(id: string): id is BuildSystemId {
    return Object.prototype.hasOwnProperty.call(BuildSystems, id);
}

export function getBuildSystemDescriptor(id: string): BuildSystemDescriptor | undefined {
    return isBuildSystemId(id) ? BuildSystems[id] : undefined;
}

/**
 * The build system this file is the manifest of, if any. A manifest is recognised by its name rather than by its
 * extension, because `Makefile` has none for an extension to be read from.
 */
export function getBuildSystemByManifestFilename(filename: string): BuildSystemDescriptor | undefined {
    const basename = filename.split(/[/\\]/).pop();
    return Object.values(BuildSystems).find(buildSystem => buildSystem.manifestFilename === basename);
}

/** The build system whose manifest is written in this language, if any. */
export function getBuildSystemByManifestLanguageId(langId: string): BuildSystemDescriptor | undefined {
    return Object.values(BuildSystems).find(buildSystem => buildSystem.manifestLanguageId === langId);
}

/**
 * Whether this is the language a build system's manifest is written in. Those are not languages anything is compiled
 * as, so they are not offered for an editor until one is needed.
 */
export function isManifestLanguageId(langId: string): boolean {
    return getBuildSystemByManifestLanguageId(langId) !== undefined;
}

/** The build systems that can build the given language. */
export function getBuildSystemsForLanguage(langId: LanguageKey): BuildSystemDescriptor[] {
    return Object.values(BuildSystems).filter(
        buildSystem =>
            buildSystem.compatibleLanguageIds === 'all' || buildSystem.compatibleLanguageIds.includes(langId),
    );
}
