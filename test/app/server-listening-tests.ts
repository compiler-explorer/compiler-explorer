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

function createMockWebServer(): express.Express {
    return {
        listen: vi.fn(),
        all: vi.fn(),
    } as unknown as express.Express;
}

import express from 'express';
import systemdSocket from 'systemd-socket';
import {afterEach, beforeEach, describe, expect, it, vi} from 'vitest';
import {startListening} from '../../lib/app/server.js'; // TODO
import type {AppArguments} from '../../lib/app.interfaces.js';
import * as logger from '../../lib/logger.js';

// Mock systemd-socket
vi.mock('systemd-socket', () => {
    return {
        default: vi.fn(() => null),
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
        makeLogStream: vi.fn(() => ({write: vi.fn()})),
    };
});

describe('Server Listening', () => {
    // Reset mocks between tests
    beforeEach(() => {
        vi.resetAllMocks();
    });

    afterEach(() => {
        vi.restoreAllMocks();
    });

    describe('startListening', () => {
        it('should start the web server listening on the specified port', () => {
            const mockWebServer = createMockWebServer();
            const mockAppArgs = createMockAppArgs();

            // Reset systemd socket mock
            vi.mocked(systemdSocket).mockReturnValue(null);

            startListening(mockWebServer, mockAppArgs);

            expect(mockWebServer.listen).toHaveBeenCalledWith(10240, 'localhost');
            expect(logger.logger.info).toHaveBeenCalledWith(
                expect.stringContaining('Listening on http://localhost:10240/'),
            );
        });

        it('should use systemd socket when available', () => {
            const mockWebServer = createMockWebServer();
            const mockAppArgs = createMockAppArgs();

            // Set systemd socket mock to return data
            const socketData = {fd: 123};
            vi.mocked(systemdSocket).mockReturnValue(socketData);

            startListening(mockWebServer, mockAppArgs);

            expect(mockWebServer.listen).toHaveBeenCalledWith(socketData);
            expect(logger.logger.info).toHaveBeenCalledWith(expect.stringContaining('Listening on systemd socket'));
        });

        it('should setup idle timeout when using systemd socket and IDLE_TIMEOUT is set', () => {
            const mockWebServer = createMockWebServer();
            const mockAppArgs = createMockAppArgs({hostname: ''});

            // Set systemd socket mock to return data
            vi.mocked(systemdSocket).mockReturnValue({fd: 123});

            // Set up env for idle timeout
            const originalEnv = process.env.IDLE_TIMEOUT;
            process.env.IDLE_TIMEOUT = '5';

            startListening(mockWebServer, mockAppArgs);

            expect(mockWebServer.all).toHaveBeenCalledWith('*', expect.any(Function));
            expect(logger.logger.info).toHaveBeenCalledWith(expect.stringContaining('IDLE_TIMEOUT: 5'));

            // Restore env
            process.env.IDLE_TIMEOUT = originalEnv;
        });
    });
});
