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

import {afterEach, describe, expect, it, vi} from 'vitest';

import {getRemoteBrowserUrl} from '../../shared/remote-utils.js';
import {Remote} from '../../types/compiler.interfaces.js';
import {getSupportedLibraries} from '../lib-utils.js';

function makeRemote(basePath: string, browserUrl?: string): Remote {
    return {
        target: 'https://alb.example.com:443',
        path: `${basePath}/api/compiler/x/compile`,
        cmakePath: `${basePath}/api/compiler/x/cmake`,
        basePath,
        browserUrl,
    };
}

describe('getRemoteBrowserUrl', () => {
    it('uses the root-relative path for same-origin remotes', () => {
        expect(getRemoteBrowserUrl(makeRemote('/gpu', '/gpu'))).toEqual('/gpu');
    });

    it('uses the absolute URL for remotes on a different host', () => {
        expect(getRemoteBrowserUrl(makeRemote('/gpu', 'https://other.example.com:443/gpu'))).toEqual(
            'https://other.example.com:443/gpu',
        );
    });

    it('falls back to target + basePath for remotes discovered without browserUrl', () => {
        expect(getRemoteBrowserUrl(makeRemote('/winprod'))).toEqual('https://alb.example.com:443/winprod');
    });
});

describe('getSupportedLibraries for remote compilers', () => {
    const libsResponse = [{id: 'fmt', name: 'fmt', versions: [{id: 'trunk', version: 'trunk'}]}];

    const mockFetch = () => {
        const fetchMock = vi.fn(
            async () =>
                new Response(JSON.stringify(libsResponse), {
                    status: 200,
                    headers: {'content-type': 'application/json'},
                }),
        );
        vi.stubGlobal('fetch', fetchMock);
        return fetchMock;
    };

    const fetchedUrl = (fetchMock: ReturnType<typeof mockFetch>) => (fetchMock.mock.calls[0] as unknown[])[0] as string;

    afterEach(() => {
        vi.unstubAllGlobals();
    });

    it('fetches a same-origin remote from its path on this origin, not from the site root path', async () => {
        // The test DOM hosts CE under the /test/ sub-path; the remote's path is root-relative regardless.
        expect(window.httpRoot).toEqual('/test/');
        const fetchMock = mockFetch();
        const libs = await getSupportedLibraries(undefined, 'c++', makeRemote('/gpu', '/gpu'));
        expect(fetchMock).toHaveBeenCalledTimes(1);
        expect(fetchedUrl(fetchMock)).toMatch(/^\/gpu\/api\/libraries\/c%2B%2B\?fields=/);
        expect(Object.keys(libs)).toEqual(['fmt']);
    });

    it('fetches a different-host remote from its absolute URL', async () => {
        const fetchMock = mockFetch();
        await getSupportedLibraries(undefined, 'c++', makeRemote('/win', 'https://other.example.com:443/win'));
        expect(fetchedUrl(fetchMock)).toMatch(/^https:\/\/other\.example\.com:443\/win\/api\/libraries\/c%2B%2B\?/);
    });

    it('caches per remote URL and language', async () => {
        const fetchMock = mockFetch();
        const remote = makeRemote('/cached', '/cached');
        await getSupportedLibraries(undefined, 'c', remote);
        await getSupportedLibraries(undefined, 'c', remote);
        await getSupportedLibraries(undefined, 'c++', remote);
        expect(fetchMock).toHaveBeenCalledTimes(2);
    });
});
