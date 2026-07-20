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

import path from 'node:path';
import process from 'node:process';

// Test helper functions
function createMockAppArgs(overrides: Partial<AppArguments> = {}): AppArguments {
    return {
        port: 10240,
        hostname: 'localhost',
        env: ['test'],
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
            paperTrailIdentifier: 'test',
        },
        ...overrides,
    };
}

import express, {type Request, type Response} from 'express';
import {afterEach, beforeEach, describe, expect, it, vi} from 'vitest';

import type {ServerDependencies, ServerOptions} from '../../lib/app/server.interfaces.js';
import {setupWebServer} from '../../lib/app/server.js'; // TODO
import type {AppArguments} from '../../lib/app.interfaces.js';
import type {ClientOptionsSource} from '../../lib/options-handler.interfaces.js';
import * as utils from '../../lib/utils.js';

describe('Server Module', () => {
    // Reset mocks between tests
    beforeEach(() => {
        vi.resetAllMocks();
        // Clear gauge cache before each test
    });

    afterEach(() => {
        vi.restoreAllMocks();
    });

    describe('setupWebServer', () => {
        // Create reusable mocks for the dependencies
        let mockAppArgs: AppArguments;
        let mockOptions: ServerOptions;
        let mockDependencies: ServerDependencies;

        beforeEach(() => {
            // Mock process.env for production mode
            process.env.NODE_ENV = 'PROD';

            // Setup mock dependencies
            mockAppArgs = createMockAppArgs({
                gitReleaseName: 'test-release',
                releaseBuildNumber: '123',
            });

            mockOptions = {
                staticPath: './static',
                staticMaxAgeSecs: 42,
                staticRoot: '/static',
                httpRoot: '',
                sentrySlowRequestMs: 500,
                manifestPath: '/mocked/dist', // Use absolute path for testing
                extraBodyClass: '',
                maxUploadSize: '1mb',
            };

            const mockClientOptionsHandler: ClientOptionsSource = {
                get: vi.fn().mockReturnValue({}),
                getHash: vi.fn().mockReturnValue('hash123'),
                getJSON: vi.fn().mockReturnValue('{}'),
            };

            mockDependencies = {
                ceProps: vi.fn((key, defaultValue) => {
                    if (key === 'bodyParserLimit') return defaultValue;
                    if (key === 'allowedShortUrlHostRe') return '.*';
                    return '';
                }),
                awsProps: vi.fn().mockReturnValue(''),
                healthcheckController: {
                    createRouter: vi.fn().mockReturnValue(express.Router()),
                },
                sponsorConfig: {
                    getLevels: vi.fn().mockReturnValue([]),
                    pickTopIcons: vi.fn().mockReturnValue([]),
                    getAllTopIcons: vi.fn().mockReturnValue([]),
                },
                clientOptionsHandler: mockClientOptionsHandler,
                storageSolution: 'mock-storage',
            };

            // Setup utils mock
            vi.spyOn(utils, 'resolvePathFromAppRoot').mockReturnValue('/mocked/path');
        });

        it('should create a web server instance', async () => {
            const {webServer} = await setupWebServer(mockAppArgs, mockOptions, mockDependencies);
            // Just check it's a function, since express returns a function
            expect(typeof webServer).toBe('function');
        });

        it('should create a renderConfig function', async () => {
            const {renderConfig} = await setupWebServer(mockAppArgs, mockOptions, mockDependencies);
            const config = renderConfig({foo: 'bar'});

            expect(config).toHaveProperty('foo', 'bar');
            expect(config).toHaveProperty('httpRoot', '');
            expect(config).toHaveProperty('staticRoot', '/static');
            expect(config).toHaveProperty('storageSolution', 'mock-storage');
            expect(config).toHaveProperty('optionsHash', 'hash123');
        });

        it('should set extraBodyClass based on embedded status', async () => {
            const {renderConfig} = await setupWebServer(mockAppArgs, mockOptions, mockDependencies);

            // When embedded is true
            const embeddedConfig = renderConfig({embedded: true});
            expect(embeddedConfig).toHaveProperty('extraBodyClass', 'embedded');

            // When embedded is false the configured extraBodyClass is passed through
            const normalConfig = renderConfig({embedded: false});
            expect(normalConfig).toHaveProperty('extraBodyClass', '');
        });

        it('should include mobile viewer slides when needed', async () => {
            const {renderConfig} = await setupWebServer(mockAppArgs, mockOptions, mockDependencies);

            // Test with mobile viewer and config
            const dummyConfig = {content: [{type: 'component', componentName: 'test'}]};
            const mobileConfig = renderConfig({mobileViewer: true, config: dummyConfig});

            expect(mobileConfig).toHaveProperty('slides');
            expect(Array.isArray(mobileConfig.slides)).toBe(true);
        });

        it('should create a renderGoldenLayout function', async () => {
            const {renderGoldenLayout} = await setupWebServer(mockAppArgs, mockOptions, mockDependencies);

            const mockRequest = {
                query: {},
                params: {id: 'test-id'},
                header: vi.fn().mockReturnValue(null),
            } as unknown as Request;

            const mockResponse = {
                render: vi.fn(),
            } as unknown as Response;

            renderGoldenLayout({} as Record<string, unknown>, {} as Record<string, unknown>, mockRequest, mockResponse);

            expect(mockResponse.render).toHaveBeenCalledWith('index', expect.any(Object));
        });

        describe('branding validation in the CDN deploy layout', () => {
            // Regression for the gh-8755 revert: on AWS the static assets ship in the CDN
            // bundle, not on the node's disk — only manifest.json ships with the app. A branded
            // env (e.g. staging) must boot by validating against the manifest, never staticPath.
            let manifestDir: string;

            const writeManifest = async (manifest: Record<string, string>) => {
                const fs = await import('node:fs/promises');
                const os = await import('node:os');
                manifestDir = await fs.mkdtemp(path.join(os.tmpdir(), 'ce-manifest-test-'));
                await fs.writeFile(path.join(manifestDir, 'manifest.json'), JSON.stringify(manifest));
                mockOptions.manifestPath = manifestDir;
                mockOptions.staticPath = '/nonexistent/.deploy/static';
                mockOptions.staticUrl = 'https://static.example.com/';
                mockOptions.extraBodyClass = 'staging';
            };

            afterEach(async () => {
                const fs = await import('node:fs/promises');
                await fs.rm(manifestDir, {recursive: true, force: true});
            });

            it('boots when the manifest lists the branding assets, despite no local static files', async () => {
                await writeManifest({
                    'favicon-staging.ico': 'favicon-staging.ico',
                    'site-logo-staging.svg': 'site-logo-staging.svg',
                });
                await expect(setupWebServer(mockAppArgs, mockOptions, mockDependencies)).resolves.toBeDefined();
            });

            it('fails startup when the manifest is missing the branding assets', async () => {
                await writeManifest({'main.js': 'main.abc123.js'});
                await expect(setupWebServer(mockAppArgs, mockOptions, mockDependencies)).rejects.toThrow(
                    /Missing branding assets for extraBodyClass='staging' in the static manifest/,
                );
            });
        });
    });
});
