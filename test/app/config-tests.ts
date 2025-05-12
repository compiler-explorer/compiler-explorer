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

import {afterEach, beforeEach, describe, expect, it, vi} from 'vitest';
import {createPropertyHierarchy, filterLanguages} from '../../lib/app/config.js';
import {logger} from '../../lib/logger.js';
import type {Language, LanguageKey} from '../../types/languages.interfaces.js';

describe('Config Module', () => {
    describe('createPropertyHierarchy', () => {
        let originalPlatform: string;
        let platformMock: string;

        beforeEach(() => {
            vi.spyOn(logger, 'info').mockImplementation(() => logger as any);
            originalPlatform = process.platform;
            Object.defineProperty(process, 'platform', {
                get: () => platformMock,
            });
        });

        afterEach(() => {
            vi.restoreAllMocks();
            platformMock = originalPlatform;
        });

        it('should create a property hierarchy with local props', () => {
            platformMock = 'linux';
            const env = ['beta', 'prod'];
            const useLocalProps = true;

            const result = createPropertyHierarchy(env, useLocalProps);

            expect(result).toContain('defaults');
            expect(result).toContain('beta');
            expect(result).toContain('prod');
            expect(result).toContain('beta.linux');
            expect(result).toContain('prod.linux');
            expect(result).toContain('linux');
            expect(result).toContain('local');
            expect(logger.info).toHaveBeenCalled();
        });

        it('should create a property hierarchy without local props', () => {
            platformMock = 'win32';
            const env = ['dev'];
            const useLocalProps = false;

            const result = createPropertyHierarchy(env, useLocalProps);

            expect(result).toContain('defaults');
            expect(result).toContain('dev');
            expect(result).toContain('dev.win32');
            expect(result).toContain('win32');
            expect(result).not.toContain('local');
        });
    });

    describe('filterLanguages', () => {
        const createMockLanguage = (id: LanguageKey, name: string, alias: string[]): Language => ({
            id,
            name,
            alias,
            extensions: [`.${id}`],
            monaco: id,
            formatter: null,
            supportsExecute: null,
            logoUrl: null,
            logoUrlDark: null,
            example: '',
            previewFilter: null,
            monacoDisassembly: null,
        });

        // Start with an empty record of the right type
        const mockLanguages: Record<LanguageKey, Language> = {} as Record<LanguageKey, Language>;

        beforeEach(() => {
            // Reset and recreate test languages before each test
            Object.keys(mockLanguages).forEach(key => delete mockLanguages[key as LanguageKey]);

            mockLanguages['c++'] = createMockLanguage('c++', 'C++', ['cpp']);
            mockLanguages.c = createMockLanguage('c', 'C', ['c99', 'c11']);
            mockLanguages.rust = createMockLanguage('rust', 'Rust', []);
            mockLanguages.go = createMockLanguage('go', 'Go', ['golang']);
            mockLanguages.cmake = createMockLanguage('cmake', 'CMake', []);
        });

        it('should return all languages when no filter specified', () => {
            const result = filterLanguages(undefined, mockLanguages);
            expect(result).toEqual(mockLanguages);
        });

        it('should filter languages by id', () => {
            const result = filterLanguages(['cpp', 'rust'], mockLanguages);
            expect(Object.keys(result)).toHaveLength(3); // c++, rust, and always cmake
            expect(result['c++']).toEqual(mockLanguages['c++']);
            expect(result.rust).toEqual(mockLanguages.rust);
            expect(result.cmake).toEqual(mockLanguages.cmake);
            expect(result.c).toBeUndefined();
        });

        it('should filter languages by name', () => {
            const result = filterLanguages(['C++', 'C'], mockLanguages);
            expect(Object.keys(result)).toHaveLength(3); // c++, c, and always cmake
            expect(result['c++']).toEqual(mockLanguages['c++']);
            expect(result.c).toEqual(mockLanguages.c);
            expect(result.cmake).toEqual(mockLanguages.cmake);
        });

        it('should filter languages by alias', () => {
            const result = filterLanguages(['c99', 'golang'], mockLanguages);
            expect(Object.keys(result)).toHaveLength(3); // c, go, and always cmake
            expect(result.c).toEqual(mockLanguages.c);
            expect(result.go).toEqual(mockLanguages.go);
            expect(result.cmake).toEqual(mockLanguages.cmake);
        });

        it('should always include cmake language', () => {
            const result = filterLanguages(['non-existent'], mockLanguages);
            expect(Object.keys(result)).toHaveLength(1);
            expect(result.cmake).toEqual(mockLanguages.cmake);
        });
    });

    // We'll skip testing setupEventLoopLagMonitoring for now due to mocking complexities
    // These tests would need to be rewritten with proper mocking for PromClient
});
