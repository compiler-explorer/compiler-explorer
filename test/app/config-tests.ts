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

import os from 'node:os';
import path from 'node:path';
import process from 'node:process';

// Test helper functions
function createMockAppArgs(overrides: Partial<AppArguments> = {}): AppArguments {
    return {
        port: 10240,
        hostname: 'localhost',
        env: ['prod'],
        gitReleaseName: '',
        releaseBuildNumber: '',
        rootDir: '/test/root',
        wantedLanguages: undefined,
        doCache: true,
        fetchCompilersFromRemote: false,
        ensureNoCompilerClash: undefined,
        prediscovered: undefined,
        discoveryOnly: undefined,
        staticPath: undefined,
        metricsPort: undefined,
        useLocalProps: true,
        propDebug: false,
        tmpDir: undefined,
        isWsl: false,
        devMode: false,
        loggingOptions: {
            debug: false,
            suppressConsoleLog: false,
            paperTrailIdentifier: 'prod',
        },
        ...overrides,
    };
}

import {afterEach, beforeEach, describe, expect, it, vi} from 'vitest';
import {
    createPropertyHierarchy,
    filterLanguages,
    loadConfiguration,
    measureEventLoopLag,
    setupEventLoopLagMonitoring,
} from '../../lib/app/config.js';
import type {AppArguments} from '../../lib/app.interfaces.js';
import * as logger from '../../lib/logger.js';
import type {CompilerProps} from '../../lib/properties.js';
import * as props from '../../lib/properties.js';
import type {Language, LanguageKey} from '../../types/languages.interfaces.js';

// Mock modules
vi.mock('node:os', async () => {
    const actual = await vi.importActual('node:os');
    return {
        ...actual,
        hostname: vi.fn(() => 'test-hostname'),
    };
});

vi.mock('../../lib/logger.js', async () => {
    const actual = await vi.importActual('../../lib/logger.js');
    return {
        ...actual,
        logger: {
            info: vi.fn(),
            error: vi.fn(),
            warn: vi.fn(),
        },
    };
});

vi.mock('../../lib/properties.js', async () => {
    const actual = await vi.importActual('../../lib/properties.js');
    return {
        ...actual,
        initialize: vi.fn(),
        propsFor: vi.fn(),
        setDebug: vi.fn(),
        CompilerProps: vi.fn().mockImplementation(() => ({
            ceProps: vi.fn(),
        })),
    };
});

// Mock PromClient.Gauge class
class MockGauge {
    set = vi.fn();
}

vi.mock('prom-client', () => {
    return {
        default: {
            Gauge: vi.fn().mockImplementation(() => new MockGauge()),
        },
    };
});

describe('Config Module', () => {
    describe('measureEventLoopLag', () => {
        it('should return a Promise resolving to a number', () => {
            // Just verify the function returns a Promise that resolves to a number
            // We don't test actual timing as that's environment-dependent
            return expect(measureEventLoopLag(1)).resolves.toBeTypeOf('number');
        });
    });

    describe('createPropertyHierarchy', () => {
        let originalPlatform: string;
        let platformMock: string;
        let hostnameBackup: typeof os.hostname;

        beforeEach(() => {
            vi.spyOn(logger.logger, 'info').mockImplementation(() => logger.logger);
            vi.spyOn(logger.logger, 'error').mockImplementation(() => logger.logger);
            originalPlatform = process.platform;
            platformMock = 'linux';
            Object.defineProperty(process, 'platform', {
                get: () => platformMock,
                configurable: true,
            });

            // Ensure hostname is properly mocked
            hostnameBackup = os.hostname;
            os.hostname = vi.fn().mockReturnValue('test-hostname');
        });

        afterEach(() => {
            vi.restoreAllMocks();
            Object.defineProperty(process, 'platform', {
                value: originalPlatform,
                configurable: true,
            });

            // Restore hostname
            os.hostname = hostnameBackup;
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
            expect(result).toContain('test-hostname');
            expect(result).toContain('local');
            expect(logger.logger.info).toHaveBeenCalled();
            expect(os.hostname).toHaveBeenCalled();
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
            expect(result).toContain('test-hostname');
            expect(result).not.toContain('local');
            expect(os.hostname).toHaveBeenCalled();
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
            logoFilename: null,
            logoFilenameDark: null,
            example: '',
            previewFilter: null,
            monacoDisassembly: null,
        });

        // Start with an empty record of the right type
        const mockLanguages: Record<LanguageKey, Language> = {} as Record<LanguageKey, Language>;

        beforeEach(() => {
            // Reset and recreate test languages before each test
            Object.keys(mockLanguages).forEach(key => {
                delete mockLanguages[key as LanguageKey];
            });

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
            const result = filterLanguages(['c++', 'rust'], mockLanguages);
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

    describe('setupEventLoopLagMonitoring', () => {
        let setImmediateSpy: any;
        let measureEventLoopLagSpy: any;

        beforeEach(() => {
            setImmediateSpy = vi.spyOn(global, 'setImmediate');
            measureEventLoopLagSpy = vi.spyOn({measureEventLoopLag}, 'measureEventLoopLag');
            measureEventLoopLagSpy.mockResolvedValue(50);
        });

        afterEach(() => {
            vi.restoreAllMocks();
        });

        it('should not set up monitoring if interval is 0', () => {
            const mockCeProps = vi.fn().mockImplementation((key: string, defaultValue: any) => {
                if (key === 'eventLoopMeasureIntervalMs') return 0;
                return defaultValue;
            });

            setupEventLoopLagMonitoring(mockCeProps);
            expect(setImmediateSpy).not.toHaveBeenCalled();
        });

        it('should set up monitoring if interval is greater than 0', () => {
            const mockCeProps = vi.fn().mockImplementation((key: string, defaultValue: any) => {
                if (key === 'eventLoopMeasureIntervalMs') return 100;
                if (key === 'eventLoopLagThresholdWarn') return 50;
                if (key === 'eventLoopLagThresholdErr') return 100;
                return defaultValue;
            });

            setupEventLoopLagMonitoring(mockCeProps);
            expect(setImmediateSpy).toHaveBeenCalled();
        });
    });

    describe('loadConfiguration', () => {
        let mockCeProps: any;
        let mockCompilerProps: CompilerProps;

        beforeEach(() => {
            // Mock needed dependencies
            mockCeProps = {
                staticMaxAgeSecs: 3600,
                maxUploadSize: '10mb',
                extraBodyClass: 'test-class',
                storageSolution: 'local',
                httpRoot: '/ce',
                staticUrl: undefined,
                restrictToLanguages: undefined,
            };

            // Set up props mocks
            vi.spyOn(props, 'initialize').mockImplementation(() => {});
            vi.spyOn(props, 'propsFor').mockImplementation(() => (key: string, defaultValue?: any) => {
                if (key in mockCeProps) return mockCeProps[key];
                return defaultValue;
            });

            mockCompilerProps = {
                ceProps: vi.fn().mockImplementation((key: string, defaultValue: any) => {
                    if (key === 'storageSolution') return 'local';
                    return defaultValue;
                }),
            } as any;

            vi.spyOn(props, 'CompilerProps').mockImplementation(() => mockCompilerProps);
        });

        afterEach(() => {
            vi.restoreAllMocks();
        });

        it('should load configuration and return expected properties', () => {
            const appArgs = createMockAppArgs({
                useLocalProps: true,
                propDebug: false,
            });

            const result = loadConfiguration(appArgs);

            // Verify initialization happened correctly
            expect(props.initialize).toHaveBeenCalledWith(path.normalize('/test/root/config'), expect.any(Array));
            expect(props.propsFor).toHaveBeenCalledWith('compiler-explorer');
            expect(props.CompilerProps).toHaveBeenCalled();

            // Verify expected result properties
            expect(result).toHaveProperty('ceProps');
            expect(result).toHaveProperty('compilerProps');
            expect(result).toHaveProperty('languages');
            expect(result).toHaveProperty('staticMaxAgeSecs', 3600);
            expect(result).toHaveProperty('maxUploadSize', '10mb');
            expect(result).toHaveProperty('extraBodyClass', 'test-class');
            expect(result).toHaveProperty('storageSolution', 'local');
            expect(result).toHaveProperty('httpRoot', '/ce/');
            expect(result).toHaveProperty('staticRoot', '/ce/');
        });

        it('should enable property debugging when propDebug is true', () => {
            const appArgs = createMockAppArgs({
                useLocalProps: true,
                propDebug: true,
            });

            loadConfiguration(appArgs);

            expect(props.setDebug).toHaveBeenCalledWith(true);
        });

        it('should set wantedLanguages from restrictToLanguages property', () => {
            mockCeProps.restrictToLanguages = 'c++,rust';
            const appArgs = createMockAppArgs({
                useLocalProps: true,
                propDebug: false,
            });

            loadConfiguration(appArgs);

            expect(appArgs.wantedLanguages).toEqual(['c++', 'rust']);
        });

        it('should handle extraBodyClass for dev mode', () => {
            // Set up app args with dev mode enabled
            const appArgs = createMockAppArgs({
                devMode: true,
            });

            // Don't set extraBodyClass in mockCeProps, so it will use the default
            delete mockCeProps.extraBodyClass;

            // Load configuration with dev mode enabled
            const result = loadConfiguration(appArgs);

            // In dev mode, extraBodyClass should be 'dev'
            expect(result.extraBodyClass).toBe('dev');
        });

        it('should handle staticUrl when provided', () => {
            mockCeProps.staticUrl = 'https://static.example.com';
            const appArgs = createMockAppArgs({
                useLocalProps: true,
                propDebug: false,
            });

            const result = loadConfiguration(appArgs);

            expect(result.staticUrl).toBe('https://static.example.com');
            expect(result.staticRoot).toBe('https://static.example.com/');
        });

        it('should handle staticUrl with trailing slash correctly', () => {
            // This tests the production scenario where staticUrl already has trailing slash
            mockCeProps.staticUrl = 'https://static.ce-cdn.net/';
            const appArgs = createMockAppArgs({
                useLocalProps: true,
                propDebug: false,
            });

            const result = loadConfiguration(appArgs);

            expect(result.staticUrl).toBe('https://static.ce-cdn.net/');
            // urljoin normalizes trailing slashes correctly
            expect(result.staticRoot).toBe('https://static.ce-cdn.net/');
        });
    });
});
