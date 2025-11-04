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
import express from 'express';
import {afterEach, beforeEach, describe, expect, it, vi} from 'vitest';
import {initialiseApplication} from '../../lib/app/main.js';
import * as server from '../../lib/app/server.js';
import {AppArguments} from '../../lib/app.interfaces.js';
import * as aws from '../../lib/aws.js';
import {CompilationQueue} from '../../lib/compilation-queue.js';
import * as exec from '../../lib/exec.js';
import {RemoteExecutionQuery} from '../../lib/execution/execution-query.js';
import * as execTriple from '../../lib/execution/execution-triple.js';
import * as execQueue from '../../lib/execution/sqs-execution-queue.js';
import {logger} from '../../lib/logger.js';
import {setupMetricsServer} from '../../lib/metrics-server.js';
import * as sentry from '../../lib/sentry.js';
import * as sponsors from '../../lib/sponsors.js';
import {getStorageTypeByKey} from '../../lib/storage/index.js';

// We need to mock all these modules to avoid actual API calls
// Vitest 4.0 requires constructor mocks to be actual classes/functions, not arrow functions.
// We define classes directly here because .mockImplementation() gets auto-converted to arrow
// functions by Biome, which breaks constructor calls. Do not refactor to arrow functions!
vi.mock('../../lib/aws.js');
vi.mock('../../lib/exec.js');
vi.mock('../../lib/execution/execution-query.js');
vi.mock('../../lib/execution/execution-triple.js');
vi.mock('../../lib/execution/sqs-execution-queue.js');
vi.mock('../../lib/compilation-env.js', () => ({
    CompilationEnvironment: class {
        setCompilerFinder = vi.fn();
    },
}));
vi.mock('../../lib/compilation-queue.js');
// Shared mock implementations for CompilerFinder
const mockCompilerFinderFind = vi.fn();
const mockCompilerFinderLoadPrediscovered = vi.fn();
const mockCompilerFinderCompileHandler = {findCompiler: vi.fn()};

vi.mock('../../lib/compiler-finder.js', () => ({
    CompilerFinder: class {
        find = mockCompilerFinderFind;
        loadPrediscovered = mockCompilerFinderLoadPrediscovered;
        compileHandler = mockCompilerFinderCompileHandler;
    },
}));
vi.mock('../../lib/formatting-service.js', () => ({
    FormattingService: class {
        initialize = vi.fn().mockResolvedValue(undefined);
    },
}));
vi.mock('../../lib/handlers/compile.js', () => ({
    CompileHandler: class {
        findCompiler = vi.fn().mockReturnValue({
            possibleArguments: {possibleArguments: []},
        });
        handle = vi.fn();
    },
}));
vi.mock('../../lib/handlers/noscript.js', () => ({
    NoScriptHandler: class {
        initializeRoutes = vi.fn();
        createRouter = vi.fn().mockReturnValue({});
    },
}));
vi.mock('../../lib/handlers/route-api.js', () => ({
    RouteAPI: class {
        apiHandler = {
            setCompilers: vi.fn(),
            setLanguages: vi.fn(),
            setOptions: vi.fn(),
        };
        initializeRoutes = vi.fn();
    },
}));
vi.mock('../../lib/metrics-server.js');
vi.mock('../../lib/options-handler.js', () => ({
    ClientOptionsHandler: class {
        setCompilers = vi.fn().mockResolvedValue(undefined);
        get = vi.fn();
        getHash = vi.fn();
        getJSON = vi.fn();
    },
}));
vi.mock('../../lib/sentry.js');
vi.mock('../../lib/sponsors.js');
vi.mock('../../lib/storage/index.js');
vi.mock('../../lib/app/server.js');
vi.mock('node:fs/promises');

// Mock the routes-setup module with a simplified implementation that doesn't use the controllers
vi.mock('../../lib/app/routes-setup.js', () => ({
    setupRoutesAndApi: vi.fn().mockImplementation(() => ({
        apiHandler: {
            setCompilers: vi.fn(),
            setLanguages: vi.fn(),
            setOptions: vi.fn(),
        },
        initializeRoutes: vi.fn(),
    })),
}));

// Also mock the compiler-changes module to avoid the apiHandler issue
vi.mock('../../lib/app/compiler-changes.js', () => ({
    setupCompilerChangeHandling: vi.fn().mockResolvedValue(undefined),
}));

describe('Main module', () => {
    const mockAppArgs: AppArguments = {
        rootDir: '/test/root',
        env: ['test'],
        port: 10240,
        gitReleaseName: 'test-release',
        releaseBuildNumber: '123',
        wantedLanguages: ['c++'],
        doCache: true,
        fetchCompilersFromRemote: true,
        ensureNoCompilerClash: false,
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
            paperTrailIdentifier: 'test',
        },
    };

    const mockConfig = {
        ceProps: vi.fn((_key: string, defaultValue?: any) => defaultValue),
        compilerProps: {
            ceProps: vi.fn((_key: string, defaultValue?: any) => defaultValue),
        },
        languages: {},
        staticMaxAgeSecs: 60,
        maxUploadSize: '1mb',
        extraBodyClass: '',
        storageSolution: 'local',
        httpRoot: '/',
        staticRoot: '/static',
        staticUrl: undefined,
    };

    const mockCompilers = [
        {
            id: 'gcc',
            name: 'GCC',
            lang: 'c++',
        },
    ];

    // Setup mocks
    beforeEach(() => {
        vi.spyOn(logger, 'info').mockImplementation(() => logger);
        vi.spyOn(logger, 'warn').mockImplementation(() => logger);
        vi.spyOn(logger, 'debug').mockImplementation(() => logger);

        vi.mocked(aws.initConfig).mockResolvedValue();
        vi.mocked(aws.getConfig).mockReturnValue('sentinel');
        vi.mocked(sentry.SetupSentry).mockReturnValue();
        vi.mocked(exec.startWineInit).mockReturnValue();
        vi.mocked(RemoteExecutionQuery.initRemoteExecutionArchs).mockReturnValue();

        // Classes are mocked with proper constructors in vi.mock above
        // Configure the default behavior for CompilerFinder methods
        mockCompilerFinderFind.mockResolvedValue({compilers: mockCompilers, foundClash: false});
        mockCompilerFinderLoadPrediscovered.mockResolvedValue(mockCompilers);
        mockCompilerFinderCompileHandler.findCompiler.mockReturnValue({possibleArguments: {possibleArguments: []}});

        const mockCompilationQueue = {
            queue: vi.fn(),
        };
        vi.mocked(CompilationQueue.fromProps).mockReturnValue(mockCompilationQueue as unknown as CompilationQueue);

        const mockStorageType = vi.fn();
        vi.mocked(getStorageTypeByKey).mockReturnValue(
            mockStorageType as unknown as ReturnType<typeof getStorageTypeByKey>,
        );

        vi.mocked(mockConfig.ceProps).mockImplementation((key: string, defaultValue?: any) => {
            if (key === 'execqueue.is_worker') return false;
            if (key === 'healthCheckFilePath') return null;
            if (key === 'sentrySlowRequestMs') return 0;
            if (key === 'rescanCompilerSecs') return 0;
            return defaultValue;
        });

        vi.mocked(sponsors.loadSponsorsFromString).mockResolvedValue({
            getLevels: vi.fn().mockReturnValue([]),
            pickTopIcons: vi.fn().mockReturnValue([]),
            getAllTopIcons: vi.fn().mockReturnValue([]),
        } as unknown as ReturnType<typeof sponsors.loadSponsorsFromString>);
        vi.mocked(fs.readFile).mockResolvedValue('sponsors: []');

        const mockRouter = {
            use: vi.fn().mockReturnThis(),
        };

        const mockWebServerResult = {
            webServer: {} as express.Express,
            router: mockRouter as unknown as express.Router,
            renderConfig: vi.fn(),
            renderGoldenLayout: vi.fn(),
            pugRequireHandler: vi.fn(),
        };
        vi.mocked(server.setupWebServer).mockResolvedValue(mockWebServerResult);
        vi.mocked(server.startListening).mockImplementation(() => {});
        // NoScriptHandler, RouteAPI, and ClientOptionsHandler are already properly mocked with classes
    });

    afterEach(() => {
        vi.restoreAllMocks();
    });

    it('should initialize and return a web server', async () => {
        const result = await initialiseApplication({
            appArgs: mockAppArgs,
            config: mockConfig as any,
            distPath: '/test/dist',
            awsProps: vi.fn() as any,
        });

        // Verify the application was properly initialized
        expect(result).toHaveProperty('webServer');
        expect(aws.initConfig).toHaveBeenCalled();
        expect(sentry.SetupSentry).toHaveBeenCalled();
        expect(exec.startWineInit).toHaveBeenCalled();
        expect(RemoteExecutionQuery.initRemoteExecutionArchs).toHaveBeenCalled();
        expect(CompilationQueue.fromProps).toHaveBeenCalled();
        expect(server.setupWebServer).toHaveBeenCalled();
        expect(server.startListening).toHaveBeenCalled();
    });

    it('should load prediscovered compilers if provided', async () => {
        vi.mocked(fs.readFile).mockImplementation(async path => {
            if (typeof path === 'string' && path.includes('prediscovered')) {
                return JSON.stringify(mockCompilers);
            }
            return 'sponsors: []';
        });

        const result = await initialiseApplication({
            appArgs: {
                ...mockAppArgs,
                prediscovered: '/path/to/prediscovered.json',
            },
            config: mockConfig as any,
            distPath: '/test/dist',
            awsProps: vi.fn() as any,
        });

        expect(result).toHaveProperty('webServer');
        // The actual prediscovered compiler loading is tested in compiler-discovery-tests.ts
    });

    it('should set up metrics server if configured', async () => {
        await initialiseApplication({
            appArgs: {
                ...mockAppArgs,
                metricsPort: 9000,
            },
            config: mockConfig as any,
            distPath: '/test/dist',
            awsProps: vi.fn() as any,
        });

        expect(setupMetricsServer).toHaveBeenCalledWith(9000, undefined);
    });

    it('should initialize execution worker if configured', async () => {
        vi.mocked(mockConfig.ceProps).mockImplementation((key: string, defaultValue?: any) => {
            if (key === 'execqueue.is_worker') return true;
            return defaultValue;
        });

        await initialiseApplication({
            appArgs: mockAppArgs,
            config: mockConfig as any,
            distPath: '/test/dist',
            awsProps: vi.fn() as any,
        });

        expect(execTriple.initHostSpecialties).toHaveBeenCalled();
        expect(execQueue.startExecutionWorkerThread).toHaveBeenCalled();
    });

    it('should throw an error if no compilers are found', async () => {
        mockCompilerFinderFind.mockResolvedValue({compilers: [], foundClash: false});

        await expect(
            initialiseApplication({
                appArgs: mockAppArgs,
                config: mockConfig as any,
                distPath: '/test/dist',
                awsProps: vi.fn() as any,
            }),
        ).rejects.toThrow('Unexpected failure, no compilers found!');
    });

    it('should throw an error if there are compiler clashes and ensureNoCompilerClash is set', async () => {
        mockCompilerFinderFind.mockResolvedValue({compilers: mockCompilers, foundClash: true});

        await expect(
            initialiseApplication({
                appArgs: {...mockAppArgs, ensureNoCompilerClash: true},
                config: mockConfig as any,
                distPath: '/test/dist',
                awsProps: vi.fn() as any,
            }),
        ).rejects.toThrow('Clashing compilers in the current environment found!');
    });
});
