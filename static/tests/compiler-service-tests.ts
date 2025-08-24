// Copyright (c) 2025, Compiler Explorer Authors
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

import {beforeEach, describe, expect, it, vi} from 'vitest';
import createFetchMock from 'vitest-fetch-mock';

import {CompilerService} from '../compiler-service.js';

const fetch = createFetchMock(vi);
fetch.enableMocks();

// Mock console.debug to avoid noise in tests
vi.spyOn(console, 'debug').mockImplementation(() => {});
vi.spyOn(console, 'log').mockImplementation(() => {});

// Mock dependencies
vi.mock('../options.js', () => ({
    options: {
        httpRoot: '/test/',
        compilers: [{id: 'test-compiler', lang: 'c++', name: 'Test Compiler'}],
        doCache: false,
        defaultCompiler: {'c++': 'test-compiler'},
    },
}));

vi.mock('../hub.js', () => ({
    Hub: class MockHub {
        on = vi.fn();
    },
}));

describe('CompilerService HTTP Headers', () => {
    let compilerService: CompilerService;

    const expectCorrectHeaders = (callIndex = 0) => {
        const [url, options] = fetch.mock.calls[callIndex] as [string, RequestInit];
        expect(options?.method).toBe('POST');
        const headers = options?.headers as Record<string, string> | undefined;
        expect(headers?.['Content-Type']).toBe('application/json');
        expect(headers?.['Accept']).toBe('application/json');
        expect(options?.body).toBeDefined();
        return {url, options, headers};
    };

    beforeEach(() => {
        fetch.resetMocks();
        vi.clearAllMocks();

        const mockEventHub = {
            on: vi.fn(),
        };

        compilerService = new CompilerService(mockEventHub as any);
    });

    describe('compilation request headers', () => {
        it('should send Accept: application/json header for compilation requests', async () => {
            const mockResponse = {
                code: 0,
                stdout: [],
                stderr: [],
                asm: [],
                okToCache: true,
            };

            fetch.mockResponseOnce(JSON.stringify(mockResponse), {
                status: 200,
                headers: {'Content-Type': 'application/json'},
            });

            const request = {
                source: 'int main() { return 0; }',
                compiler: 'test-compiler',
                options: {userArguments: ''},
                lang: 'c++',
                files: [],
                bypassCache: 0,
                allowStoreCodeDebug: true,
            };

            await compilerService.submit(request);

            // Verify the request was made with correct headers
            expect(fetch).toHaveBeenCalledTimes(1);
            const {url} = expectCorrectHeaders();
            expect(url).toContain('api/compiler/test-compiler/compile');
        });

        it('should send Accept: application/json header for CMake requests', async () => {
            const mockResponse = {
                code: 0,
                stdout: [],
                stderr: [],
                okToCache: true,
            };

            fetch.mockResponseOnce(JSON.stringify(mockResponse), {
                status: 200,
                headers: {'Content-Type': 'application/json'},
            });

            const request = {
                source: 'cmake_minimum_required(VERSION 3.0)',
                compiler: 'test-compiler',
                options: {userArguments: ''},
                lang: 'c++',
                files: [],
                bypassCache: 0,
                allowStoreCodeDebug: true,
            };

            await compilerService.submitCMake(request);

            // Verify the request was made with correct headers
            expect(fetch).toHaveBeenCalledTimes(1);
            const {url} = expectCorrectHeaders();
            expect(url).toContain('api/compiler/test-compiler/cmake');
        });

        it('should send Accept: application/json header for popular arguments requests', async () => {
            const mockResponse = {
                popularArguments: ['-O2', '-std=c++17'],
            };

            fetch.mockResponseOnce(JSON.stringify(mockResponse), {
                status: 200,
                headers: {'Content-Type': 'application/json'},
            });

            await compilerService.requestPopularArguments('test-compiler', '-O2');

            // Verify the request was made with correct headers
            expect(fetch).toHaveBeenCalledTimes(1);
            const {url} = expectCorrectHeaders();
            expect(url).toContain('api/popularArguments/test-compiler');
        });
    });
});
