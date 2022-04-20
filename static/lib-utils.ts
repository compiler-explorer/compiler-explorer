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

import {options} from './options';
import {LanguageLibs, Library, Libs} from './options.interfaces';

const LIB_MATCH_RE = /([\w-]*)\.([\w-]*)/i;

function getRemoteId(language: string, remoteUrl: string): string {
    const url: URL = new URL(remoteUrl);
    return url.host.replace(/\./g, '_') + '_' + language;
}

function getRemoteLibraries(language: string, remoteUrl: string): LanguageLibs {
    const remoteId = getRemoteId(language, remoteUrl);
    return options.remoteLibs[remoteId];
}

function copyAndFilterLibraries(allLibraries: LanguageLibs, filter: string[]) {
    const filterLibAndVersion = filter.map(lib => {
        const match = lib.match(LIB_MATCH_RE);
        return {
            id: match ? match[1] : lib,
            version: match ? match[2] : false,
        };
    });

    const filterLibIds = new Set(filterLibAndVersion.map(lib => lib.id));

    const copiedLibraries: Record<string, Library> = {};
    for (const libid in allLibraries) {
        if (!filterLibIds.has(libid)) continue;
        const lib = {...allLibraries[libid]};
        for (const versionid in lib.versions) {
            for (const filter of filterLibAndVersion) {
                if (!(!filter.version || filter.version === versionid)) {
                    delete filterLibAndVersion[versionid];
                }
            }
        }
        copiedLibraries[libid] = lib;
    }

    return copiedLibraries;
}

export function getSupportedLibraries(
    supportedLibrariesArr: string[] | undefined,
    langId: string,
    remote
): LanguageLibs {
    if (!remote) {
        const allLibs = options.libs[langId];
        if (supportedLibrariesArr && supportedLibrariesArr.length > 0) {
            return copyAndFilterLibraries(allLibs, supportedLibrariesArr);
        }
        return allLibs;
    } else {
        const allRemotes = getRemoteLibraries(langId, remote.target);
        const allLibs = allRemotes;
        if (supportedLibrariesArr && supportedLibrariesArr.length > 0) {
            return copyAndFilterLibraries(allLibs, supportedLibrariesArr);
        }
        return allLibs;
    }
}
