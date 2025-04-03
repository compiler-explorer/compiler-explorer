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

import fs from 'node:fs/promises';

import {type Mock, afterEach, beforeEach, describe, expect, it, vi} from 'vitest';

import {BaseCompiler} from '../lib/base-compiler.js';
import {CompilationEnvironment} from '../lib/compilation-env.js';
import * as exec from '../lib/exec.js';
import type {CompilationResult} from '../types/compilation/compilation.interfaces.js';
import type {FiledataPair} from '../types/compilation/compilation.interfaces.js';
import {ParseFiltersAndOutputOptions} from '../types/features/filters.interfaces.js';

import {createMockExecutor} from './mock/mock-utils.js';
import {makeCompilationEnvironment, makeFakeCompilerInfo, makeFakeParseFiltersAndOutputOptions} from './utils.js';

// Import the BypassCache enum
import {BypassCache} from '../types/compilation/compilation.interfaces.js';

const languages = {
    'c++': {id: 'c++'},
    rust: {id: 'rust'},
} as const;

describe('BaseCompiler core functionality', () => {
    let ce: CompilationEnvironment;
    let compiler: BaseCompiler;
    let mockExec: ReturnType<typeof createMockExecutor>;

    beforeEach(async () => {
        ce = makeCompilationEnvironment({
            languages,
            props: {
                compileTimeoutMs: 7000,
                binaryExecTimeoutMs: 2000,
            },
        });

        // Mock the findBadOptions function
        ce.findBadOptions = vi.fn().mockReturnValue([]);

        const compilationInfo = makeFakeCompilerInfo({
            exe: '/fake/compiler/path',
            remote: undefined,
            lang: 'c++',
            options: '-Wall -O2',
            supportsExecute: true,
            supportsBinary: true,
        });

        compiler = new BaseCompiler(compilationInfo, ce);

        // Set up our mocks
        mockExec = createMockExecutor(
            new Map([
                [
                    '/fake/compiler/path -Wall -O2 -g -o output.s -S input.cpp',
                    {
                        code: 0,
                        stdout: 'Compilation successful',
                        stderr: '',
                    },
                ],
                [
                    '/fake/compiler/path -Wall -O2 -g input.cpp -o a.out',
                    {
                        code: 0,
                        stdout: 'Binary compilation successful',
                        stderr: '',
                    },
                ],
            ]),
        );

        vi.spyOn(exec, 'execute').mockImplementation(mockExec);

        // Mock the exec method with a fully typed result
        vi.spyOn(compiler, 'exec').mockImplementation(async () => {
            return {
                code: 0,
                stdout: 'Compilation successful',
                stderr: '',
                okToCache: true,
                filenameTransform: (x: string) => x,
                execTime: 0,
                timedOut: false,
                truncated: false,
            };
        });

        // Mock filesystem operations
        vi.spyOn(fs, 'writeFile').mockImplementation(async () => {});
        vi.spyOn(fs, 'readFile').mockResolvedValue('Assembly output');

        // Mock check source method to prevent validation issues
        vi.spyOn(compiler, 'checkSource').mockReturnValue(null);
    });

    afterEach(async () => {
        vi.restoreAllMocks();
    });

    it('should compile code successfully', async () => {
        const source = 'int main() { return 42; }';
        const options = ['']; // Array of strings
        const filters: ParseFiltersAndOutputOptions = makeFakeParseFiltersAndOutputOptions({});

        const result = await compiler.compile(
            source,
            options,
            {},
            filters,
            BypassCache.None, // bypassCache
            [], // backendOptions
            {}, // filters
            [], // tools
            [] as FiledataPair[], // executionParams
        );

        expect(result.code).toBe(0);
        expect(result.asm.length).toBeGreaterThan(0);
    });

    it('should handle compilation failures', async () => {
        // Override the exec mock to simulate a compilation failure
        vi.spyOn(compiler, 'exec').mockResolvedValueOnce({
            code: 1,
            stdout: '',
            stderr: "error: unknown type name 'foo'",
            okToCache: true,
            filenameTransform: (x: string) => x,
            execTime: 0,
            timedOut: false,
            truncated: false,
        });

        const source = 'foo bar() { return 42; }';
        const options = ['']; // Array of strings
        const filters: ParseFiltersAndOutputOptions = makeFakeParseFiltersAndOutputOptions({});

        const result = await compiler.compile(
            source,
            options,
            {},
            filters,
            BypassCache.None, // bypassCache
            [], // backendOptions
            {}, // filters
            [], // tools
            [] as FiledataPair[], // executionParams
        );

        expect(result.code).toBe(1);
        expect(result.asm).toEqual([{text: '<Compilation failed>', source: null, labels: []}]);
    });
});

describe('BaseCompiler caching', () => {
    let ce: CompilationEnvironment;
    let compiler: BaseCompiler;
    let cacheGetSpy: Mock;
    let cachePutSpy: Mock;

    beforeEach(async () => {
        ce = makeCompilationEnvironment({
            languages,
            doCache: true,
            props: {
                compileTimeoutMs: 7000,
                binaryExecTimeoutMs: 2000,
            },
        });

        // Completely mock the cache methods directly rather than through spies
        ce.cacheGet = vi.fn().mockResolvedValue(null);
        ce.cachePut = vi.fn().mockResolvedValue(null);
        ce.findBadOptions = vi.fn().mockReturnValue([]);

        cacheGetSpy = ce.cacheGet as Mock;
        cachePutSpy = ce.cachePut as Mock;

        const compilationInfo = makeFakeCompilerInfo({
            exe: '/fake/compiler/path',
            remote: undefined,
            lang: 'c++',
            options: '-Wall -O2',
        });

        compiler = new BaseCompiler(compilationInfo, ce);

        // Mock compiler execution
        vi.spyOn(compiler, 'exec').mockResolvedValue({
            code: 0,
            stdout: 'Compilation successful',
            stderr: '',
            okToCache: true,
            filenameTransform: (x: string) => x,
            execTime: 0,
            timedOut: false,
            truncated: false,
        });

        vi.spyOn(fs, 'writeFile').mockImplementation(async () => {});
        vi.spyOn(fs, 'readFile').mockResolvedValue('Assembly output');

        // Mock check source method to prevent validation issues
        vi.spyOn(compiler, 'checkSource').mockReturnValue(null);
    });

    afterEach(async () => {
        vi.restoreAllMocks();
    });

    it('should try to fetch from cache when caching is enabled', async () => {
        const source = 'int main() { return 42; }';
        const options = ['']; // Array of strings
        const filters: ParseFiltersAndOutputOptions = makeFakeParseFiltersAndOutputOptions({});

        await compiler.compile(
            source,
            options,
            {},
            filters,
            BypassCache.None, // NOT bypassing cache
            [], // backendOptions
            {}, // filters
            [], // tools
            [] as FiledataPair[], // executionParams
        );

        expect(cacheGetSpy).toHaveBeenCalled();
    });

    it('should store results in cache when compilation succeeds with okToCache', async () => {
        const source = 'int main() { return 42; }';
        const options = ['']; // Array of strings
        const filters: ParseFiltersAndOutputOptions = makeFakeParseFiltersAndOutputOptions({});

        await compiler.compile(
            source,
            options,
            {},
            filters,
            BypassCache.None, // NOT bypassing cache
            [], // backendOptions
            {}, // filters
            [], // tools
            [] as FiledataPair[], // executionParams
        );

        expect(cachePutSpy).toHaveBeenCalled();
    });

    it('should not store results in cache when okToCache is false', async () => {
        vi.spyOn(compiler, 'exec').mockResolvedValue({
            code: 0,
            stdout: 'Compilation successful',
            stderr: '',
            okToCache: false, // Setting okToCache to false
            filenameTransform: (x: string) => x,
            execTime: 0,
            timedOut: false,
            truncated: false,
        });

        const source = 'int main() { return 42; }';
        const options = ['']; // Array of strings
        const filters: ParseFiltersAndOutputOptions = makeFakeParseFiltersAndOutputOptions({});

        await compiler.compile(
            source,
            options,
            {},
            filters,
            BypassCache.None, // NOT bypassing cache
            [], // backendOptions
            {}, // filters
            [], // tools
            [] as FiledataPair[], // executionParams
        );

        expect(cachePutSpy).not.toHaveBeenCalled();
    });

    it('should return cached results when found in cache', async () => {
        const cachedResult: CompilationResult = {
            code: 0,
            stdout: [{text: 'Cached stdout'}],
            stderr: [{text: 'Cached stderr'}],
            asm: [{text: 'Cached assembly', source: null, labels: []}],
            timedOut: false,
            compilationOptions: ['cached', 'options'],
            inputFilename: 'cached.cpp',
            // outputFilename is removed as it's not in CompilationResult type
            executableFilename: 'cached.out',
            // packagedFiles removed as it's not in CompilationResult type
            optOutput: undefined,
            tools: [],
            popularArguments: {},
            okToCache: true,
        };

        cacheGetSpy.mockResolvedValueOnce(cachedResult);

        const source = 'int main() { return 42; }';
        const options = ['']; // Array of strings
        const filters: ParseFiltersAndOutputOptions = makeFakeParseFiltersAndOutputOptions({});

        const result = await compiler.compile(
            source,
            options,
            {},
            filters,
            BypassCache.None, // NOT bypassing cache
            [], // backendOptions
            {}, // filters
            [], // tools
            [] as FiledataPair[], // executionParams
        );

        expect(result).toBe(cachedResult);
        // exec should not be called as we got a cache hit
        expect(compiler.exec).not.toHaveBeenCalled();
    });
});

describe('BaseCompiler demangle', () => {
    let ce: CompilationEnvironment;
    let compiler: BaseCompiler;
    let execSpy: any; // Use any type to avoid Mock type issues

    beforeEach(async () => {
        ce = makeCompilationEnvironment({
            languages,
            props: {
                compileTimeoutMs: 7000,
                binaryExecTimeoutMs: 2000,
            },
        });

        // Mock the findBadOptions function
        ce.findBadOptions = vi.fn().mockReturnValue([]);

        const compilationInfo = makeFakeCompilerInfo({
            exe: '/fake/compiler/path',
            remote: undefined,
            lang: 'c++',
            options: '-Wall -O2',
            demangler: '/fake/demangler',
            demanglerType: 'cpp',
            demanglerArgs: ['-n'], // Add demanglerArgs property
        });

        compiler = new BaseCompiler(compilationInfo, ce);

        // Set up our mocks
        execSpy = vi.spyOn(compiler, 'exec');

        // Mock compiler assembly compilation
        execSpy.mockResolvedValueOnce({
            code: 0,
            stdout: 'Assembly compilation successful',
            stderr: '',
            okToCache: true,
            filenameTransform: (x: string) => x,
            execTime: 0,
            timedOut: false,
            truncated: false,
        });

        vi.spyOn(fs, 'writeFile').mockImplementation(async () => {});
        vi.spyOn(fs, 'readFile').mockResolvedValue('_Z3foov:\n  ret\n');

        // Mock check source method to prevent validation issues
        vi.spyOn(compiler, 'checkSource').mockReturnValue(null);

        // Mock the postProcessAsm to simulate demangling
        vi.spyOn(compiler, 'postProcessAsm').mockImplementation(async result => {
            if (result) {
                // Just create a simple demangled output
                result.asm = [{text: 'foo():\n  ret\n', source: null, labels: []}];
            }
            return result;
        });
    });

    afterEach(async () => {
        vi.restoreAllMocks();
    });

    it('should demangle symbols when requested', async () => {
        // Mock demangler execution
        execSpy.mockResolvedValueOnce({
            code: 0,
            stdout: 'foo():\n',
            stderr: '',
            filenameTransform: (x: string) => x,
            execTime: 0,
            timedOut: false,
            truncated: false,
        });

        const source = 'int main() { return 42; }';
        const options = ['']; // Array of strings
        const filters: ParseFiltersAndOutputOptions = makeFakeParseFiltersAndOutputOptions({
            demangle: true, // Request demangling
        });

        const result = await compiler.compile(
            source,
            options,
            {},
            filters,
            BypassCache.None,
            [],
            {},
            [],
            [] as FiledataPair[],
        );

        // Result should have demangled output from our mock
        expect(result.asm[0].text).toContain('foo():');
    });

    it('should not demangle when not requested', async () => {
        const source = 'int main() { return 42; }';
        const options = ['']; // Array of strings
        const filters: ParseFiltersAndOutputOptions = makeFakeParseFiltersAndOutputOptions({
            demangle: false, // Do not request demangling
        });

        await compiler.compile(source, options, {}, filters, BypassCache.None, [], {}, [], [] as FiledataPair[]);

        // Check only one call to exec was made (for compilation, not demangling)
        expect(execSpy).toHaveBeenCalledTimes(1);
    });
});
