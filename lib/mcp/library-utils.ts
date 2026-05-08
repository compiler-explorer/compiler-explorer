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

// Library entries from ApiHandler.getLibrariesAsArray have shape:
//   {id: 'boost', versions: [{id: '188', version: '1.88.0', ...}, ...]}
// MCP callers send `{id: 'boost', version: '188'}` OR `{id: 'boost', version: '1.88.0'}`
// — both should work. This helper normalises the human form to the version id.
type LibraryVersion = {id: string; version: string};
type Library = {id: string; versions: LibraryVersion[]};

export type NormaliseSuccess = {ok: true; version: string};
export type NormaliseFailure = {
    ok: false;
    reason: 'unknown-library' | 'unknown-version';
    available?: Array<{id: string; version: string}>;
};
export type NormaliseResult = NormaliseSuccess | NormaliseFailure;

/**
 * Resolve a user-supplied library version to the canonical version id CE uses
 * internally. Accepts either form:
 *   - The id directly (e.g. "188") — returned as-is.
 *   - The human version (e.g. "1.88.0") — looked up and converted to its id.
 * Anything else returns a structured failure with the available versions so the
 * caller can produce a useful error message.
 */
export function normaliseLibraryVersion(libraries: Library[], libId: string, userVersion: string): NormaliseResult {
    const lib = libraries.find(l => l.id === libId);
    if (!lib) return {ok: false, reason: 'unknown-library'};
    if (lib.versions.some(v => v.id === userVersion)) return {ok: true, version: userVersion};
    const match = lib.versions.find(v => v.version === userVersion);
    if (match) return {ok: true, version: match.id};
    return {
        ok: false,
        reason: 'unknown-version',
        available: lib.versions.map(v => ({id: v.id, version: v.version})),
    };
}
