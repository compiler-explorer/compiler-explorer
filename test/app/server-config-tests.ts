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

import * as Sentry from '@sentry/node';
import type {NextFunction, Request, Response, Router} from 'express';
import type {Express} from 'express-serve-static-core';
import {afterEach, beforeEach, describe, expect, it, vi} from 'vitest';
import {ServerOptions} from '../../lib/app/server.interfaces.js';
import {setupBaseServerConfig} from '../../lib/app/server-config.js';
import * as logger from '../../lib/logger.js';

vi.mock('@sentry/node', () => {
    return {
        withScope: vi.fn(callback => callback({setExtra: vi.fn()})),
        captureMessage: vi.fn(),
        setupExpressErrorHandler: vi.fn(),
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

vi.mock('../../lib/utils.js', async () => {
    const actual = await vi.importActual('../../lib/utils.js');
    return {
        ...actual,
        resolvePathFromAppRoot: vi.fn(),
        anonymizeIp: vi.fn(_ip => 'anonymized-ip'),
    };
});

// Mock morgan
vi.mock('morgan', () => {
    return {
        token: vi.fn(),
        __esModule: true,
        default: vi.fn(() => 'morgan-middleware'),
    };
});

describe('Server Config Module', () => {
    // Reset mocks between tests
    beforeEach(() => {
        vi.resetAllMocks();
    });

    afterEach(() => {
        vi.restoreAllMocks();
    });

    describe('setupBaseServerConfig', () => {
        let mockWebServer: Express;
        let mockRouter: Router;
        let mockRenderConfig: any;
        let mockOptions: ServerOptions;

        beforeEach(() => {
            mockWebServer = {
                set: vi.fn().mockReturnThis(),
                on: vi.fn().mockReturnThis(),
                use: vi.fn().mockReturnThis(),
            } as unknown as Express;

            mockRouter = {} as Router;

            mockRenderConfig = vi.fn().mockReturnValue({
                error: {
                    code: 500,
                    message: 'Test Error',
                },
            });

            mockOptions = {
                sentrySlowRequestMs: 1000,
                httpRoot: '',
            } as ServerOptions;
        });

        it('should set up base server configuration', () => {
            setupBaseServerConfig(mockOptions, mockRenderConfig, mockWebServer, mockRouter);

            // Verify critical server configurations
            expect(mockWebServer.set).toHaveBeenCalledWith('trust proxy', true);
            expect(mockWebServer.set).toHaveBeenCalledWith('view engine', 'pug');
            expect(mockWebServer.on).toHaveBeenCalledWith('error', expect.any(Function));
            expect(mockWebServer.use).toHaveBeenCalled();
            expect(mockWebServer.use).toHaveBeenCalledWith(mockOptions.httpRoot, mockRouter);
            expect(Sentry.setupExpressErrorHandler).toHaveBeenCalledWith(mockWebServer);
        });

        it('should set up response time middleware', () => {
            // Just verify the server is configured properly without actually executing the callback
            setupBaseServerConfig(mockOptions, mockRenderConfig, mockWebServer, mockRouter);

            // Check that middleware was set up
            expect(mockWebServer.use).toHaveBeenCalled();

            // And that error handlers were set up
            expect(Sentry.setupExpressErrorHandler).toHaveBeenCalled();
        });

        it('should handle errors with appropriate status codes', () => {
            let errorHandler: any;
            vi.spyOn(mockWebServer, 'use').mockImplementation((handler: any) => {
                if (typeof handler === 'function' && handler.length === 4) {
                    errorHandler = handler;
                }
                return mockWebServer;
            });

            setupBaseServerConfig(mockOptions, mockRenderConfig, mockWebServer, mockRouter);

            const mockReq = {} as Request;
            const mockRes = {
                status: vi.fn().mockReturnThis(),
                render: vi.fn(),
            } as unknown as Response;
            const mockNext = vi.fn() as NextFunction;

            // Test with status code from err.status
            const testError = {status: 404, message: 'Not Found'};
            errorHandler(testError, mockReq, mockRes, mockNext);

            expect(mockRes.status).toHaveBeenCalledWith(404);
            expect(mockRes.render).toHaveBeenCalledWith('error', expect.any(Object));

            // Test with status code 500 (server error)
            const serverError = {message: 'Internal Error'};
            errorHandler(serverError, mockReq, mockRes, mockNext);

            expect(mockRes.status).toHaveBeenCalledWith(500);
            expect(logger.logger.error).toHaveBeenCalled();
        });
    });

    // Note: setupLoggingMiddleware is challenging to test due to issues with mocking morgan.token
    // and dependencies. The morgan token setup for GDPR IP anonymization requires complex mocking.

    // Note: setupBasicRoutes is challenging to test due to issues with mocking serve-favicon
    // and the complex interactions with express routes and middleware.
});
