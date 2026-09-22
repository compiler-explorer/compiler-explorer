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

import {describe, expect, it} from 'vitest';

import {cargoBuildSystem, cmakeBuildSystem} from '../lib/build-systems/index.js';
import {
    getRequestedBuildSystem,
    isJsonContentType,
    type RemoteCompilationRequest,
} from '../lib/compilation/sqs-compilation-queue.js';
import {CompileHandler} from '../lib/handlers/compile.js';

function makeMessage(fields: Partial<RemoteCompilationRequest>): RemoteCompilationRequest {
    return fields as RemoteCompilationRequest;
}

describe('Which build system a queued request asked for', () => {
    it('resolves one it knows, however the request spells it', () => {
        expect(getRequestedBuildSystem(makeMessage({buildSystem: 'cargo'}))).toBe(cargoBuildSystem);
        // The CMake-only boolean, still sent by producers deployed apart from this.
        expect(getRequestedBuildSystem(makeMessage({isCMake: true}))).toBe(cmakeBuildSystem);
    });

    it('is undefined only when nothing was asked for', () => {
        expect(getRequestedBuildSystem(makeMessage({}))).toBeUndefined();
        expect(getRequestedBuildSystem(makeMessage({isCMake: false}))).toBeUndefined();
    });

    it('refuses a build system it does not know rather than reading it as none', () => {
        // Undefined would send the project's manifest through a plain compilation, which answers with syntax errors
        // about the manifest and counts as an ordinary compile. A producer can be ahead of a worker across a deploy.
        expect(() => getRequestedBuildSystem(makeMessage({buildSystem: 'gradle'}))).toThrow(
            /Unknown build system 'gradle'/,
        );
        expect(() => getRequestedBuildSystem(makeMessage({buildSystem: 'toString'}))).toThrow(/Unknown build system/);
    });
});

describe('Whether a queued request recorded a JSON content-type', () => {
    it('accepts the type however the caller spelled it', () => {
        expect(isJsonContentType('application/json')).toBe(true);
        // Producers record the caller's header verbatim, and plenty of HTTP clients append a charset.
        expect(isJsonContentType('application/json; charset=utf-8')).toBe(true);
        expect(isJsonContentType('application/json;charset=UTF-8')).toBe(true);
        expect(isJsonContentType('  APPLICATION/JSON  ')).toBe(true);
        expect(isJsonContentType(['application/json; charset=utf-8'])).toBe(true);
    });

    it('rejects anything else, including no header at all', () => {
        expect(isJsonContentType('text/plain')).toBe(false);
        expect(isJsonContentType('application/x-www-form-urlencoded')).toBe(false);
        expect(isJsonContentType('application/jsonish')).toBe(false);
        expect(isJsonContentType(undefined)).toBe(false);
        expect(isJsonContentType('')).toBe(false);
    });
});

describe('Parsing a queued request whose content-type carries a charset', () => {
    // Reading it as text loses everything the caller asked for except the source, and the compilation still succeeds:
    // the user gets the right code built with default flags and no libraries, and nothing logs a complaint.
    const compiler = {getDefaultFilters: () => ({intel: true, demangle: true})} as any;

    function parseAsWorkerDoes(contentType: string) {
        const msg = makeMessage({
            headers: {'content-type': contentType},
            queryStringParameters: {},
            source: 'int main(){}',
            options: {
                userArguments: '-O3 -march=native',
                filters: {intel: false, binary: true},
                libraries: [{id: 'fmt', version: '901'}],
            },
        });
        const isJson = isJsonContentType(msg.headers['content-type']);
        return CompileHandler.parseRequestReusable(
            isJson,
            msg.queryStringParameters,
            isJson ? msg : msg.source,
            compiler,
        );
    }

    it('keeps what the caller asked for, exactly as the bare type does', () => {
        const bare = parseAsWorkerDoes('application/json');
        const withCharset = parseAsWorkerDoes('application/json; charset=utf-8');

        expect(withCharset.options).toEqual(['-O3', '-march=native']);
        expect(withCharset.libraries).toEqual([{id: 'fmt', version: '901'}]);
        expect(withCharset.filters).toMatchObject({intel: false, binary: true});
        expect(withCharset).toEqual(bare);
    });
});
