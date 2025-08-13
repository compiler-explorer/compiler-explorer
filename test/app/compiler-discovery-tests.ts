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
import process from 'node:process';
import {afterEach, beforeEach, describe, expect, it, vi} from 'vitest';
import {
    discoverCompilers,
    findAndValidateCompilers,
    handleDiscoveryOnlyMode,
    loadPrediscoveredCompilers,
} from '../../lib/app/compiler-discovery.js';
import {AppArguments} from '../../lib/app.interfaces.js';
import {CompilerFinder} from '../../lib/compiler-finder.js';
import {logger} from '../../lib/logger.js';
import {LanguageKey} from '../../types/languages.interfaces.js';

vi.mock('node:fs/promises');
vi.mock('../../lib/logger.js');
vi.mock('../../lib/compiler-finder.js');

describe('compiler-discovery module', () => {
    const mockCompilers = [
        {
            id: 'gcc',
            name: 'GCC',
            lang: 'c++',
        },
    ];

    beforeEach(() => {
        vi.spyOn(logger, 'info').mockImplementation(() => logger);
        vi.spyOn(logger, 'warn').mockImplementation(() => logger);
        vi.spyOn(logger, 'debug').mockImplementation(() => logger);
        vi.spyOn(process, 'exit').mockImplementation(() => undefined as never);
    });

    afterEach(() => {
        vi.restoreAllMocks();
    });

    it('should load prediscovered compilers', async () => {
        vi.mocked(fs.readFile).mockResolvedValue(JSON.stringify(mockCompilers));

        const mockCompilerFinder = {
            loadPrediscovered: vi.fn().mockResolvedValue(mockCompilers),
        } as unknown as CompilerFinder;

        const result = await loadPrediscoveredCompilers('/test/path.json', mockCompilerFinder);

        expect(result).toEqual(mockCompilers);
        expect(fs.readFile).toHaveBeenCalledWith('/test/path.json', 'utf8');
        expect(mockCompilerFinder.loadPrediscovered).toHaveBeenCalledWith(mockCompilers);
    });

    it('should throw an error if no compilers are loaded from prediscovered file', async () => {
        vi.mocked(fs.readFile).mockResolvedValue(JSON.stringify(mockCompilers));

        const mockCompilerFinder = {
            loadPrediscovered: vi.fn().mockResolvedValue([]),
        } as unknown as CompilerFinder;

        await expect(loadPrediscoveredCompilers('/test/path.json', mockCompilerFinder)).rejects.toThrow(
            'Unexpected failure, no compilers found!',
        );
    });

    it('should find and validate compilers', async () => {
        const mockFindResults = {
            compilers: mockCompilers,
            foundClash: false,
        };

        const mockCompilerFinder = {
            find: vi.fn().mockResolvedValue(mockFindResults),
        } as unknown as CompilerFinder;

        const mockAppArgs = {
            ensureNoCompilerClash: false,
        } as AppArguments;

        const result = await findAndValidateCompilers(mockAppArgs, mockCompilerFinder, false);

        expect(result).toEqual(mockFindResults);
        expect(mockCompilerFinder.find).toHaveBeenCalled();
    });

    it('should throw an error if no compilers are found', async () => {
        const mockFindResults = {
            compilers: [],
            foundClash: false,
        };

        const mockCompilerFinder = {
            find: vi.fn().mockResolvedValue(mockFindResults),
        } as unknown as CompilerFinder;

        const mockAppArgs = {
            ensureNoCompilerClash: false,
        } as AppArguments;

        await expect(findAndValidateCompilers(mockAppArgs, mockCompilerFinder, false)).rejects.toThrow(
            'Unexpected failure, no compilers found!',
        );
    });

    it('should throw an error if compiler clash found and ensureNoCompilerClash is true', async () => {
        const mockFindResults = {
            compilers: mockCompilers,
            foundClash: true,
        };

        const mockCompilerFinder = {
            find: vi.fn().mockResolvedValue(mockFindResults),
        } as unknown as CompilerFinder;

        const mockAppArgs = {
            ensureNoCompilerClash: true,
        } as AppArguments;

        await expect(findAndValidateCompilers(mockAppArgs, mockCompilerFinder, false)).rejects.toThrow(
            'Clashing compilers in the current environment found!',
        );
    });

    it('should handle discovery-only mode', async () => {
        const mockCompilerInstance = {
            possibleArguments: {
                possibleArguments: ['arg1', 'arg2'],
            },
        };

        const mockCompilerFinder = {
            compileHandler: {
                findCompiler: vi.fn().mockReturnValue(mockCompilerInstance),
            },
        } as unknown as CompilerFinder;

        const compilers = [
            {
                id: 'gcc1',
                lang: 'c++' as unknown as LanguageKey,
                buildenvsetup: {id: '', props: vi.fn()},
                externalparser: {id: ''},
            },
            {
                id: 'gcc2',
                lang: 'c++' as unknown as LanguageKey,
                buildenvsetup: {id: 'setup1', props: vi.fn()},
                externalparser: {id: 'parser1'},
            },
        ];

        await handleDiscoveryOnlyMode('/save/path.json', compilers, mockCompilerFinder);

        // Check that buildenvsetup and externalparser are removed when id is empty
        const expectedCompilers = [
            {
                id: 'gcc1',
                lang: 'c++',
                cachedPossibleArguments: ['arg1', 'arg2'],
            },
            {
                id: 'gcc2',
                lang: 'c++',
                buildenvsetup: {id: 'setup1'},
                externalparser: {id: 'parser1'},
                cachedPossibleArguments: ['arg1', 'arg2'],
            },
        ];

        expect(fs.writeFile).toHaveBeenCalledWith('/save/path.json', JSON.stringify(expectedCompilers));
        expect(process.exit).toHaveBeenCalledWith(0);
    });

    it('should discover compilers from prediscovered file', async () => {
        vi.mocked(fs.readFile).mockResolvedValue(JSON.stringify(mockCompilers));

        const mockCompilerFinder = {
            loadPrediscovered: vi.fn().mockResolvedValue(mockCompilers),
        } as unknown as CompilerFinder;

        const mockAppArgs = {
            prediscovered: '/test/prediscovered.json',
        } as AppArguments;

        const result = await discoverCompilers(mockAppArgs, mockCompilerFinder, false);

        expect(result).toEqual(mockCompilers);
        expect(fs.readFile).toHaveBeenCalledWith('/test/prediscovered.json', 'utf8');
    });

    it('should discover compilers using compiler finder', async () => {
        const mockFindResults = {
            compilers: mockCompilers,
            foundClash: false,
        };

        const mockCompilerFinder = {
            find: vi.fn().mockResolvedValue(mockFindResults),
        } as unknown as CompilerFinder;

        const mockAppArgs = {} as AppArguments;

        const result = await discoverCompilers(mockAppArgs, mockCompilerFinder, false);

        expect(result).toEqual(mockCompilers);
        expect(mockCompilerFinder.find).toHaveBeenCalled();
    });
});
