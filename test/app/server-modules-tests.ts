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

import type {Request, Response} from 'express';
import {afterEach, beforeEach, describe, expect, it, vi} from 'vitest';

import {createDefaultPugRequireHandler, getFaviconFilename} from '../../lib/app/static-assets.js';
import {OldGoogleUrlHandler, isMobileViewer} from '../../lib/app/url-handlers.js';

// Mock modules
vi.mock('../../lib/logger.js');
vi.mock('node:fs/promises');
vi.mock('@sentry/node');
vi.mock('morgan');
vi.mock('serve-favicon');
vi.mock('systemd-socket', () => ({
    default: vi.fn(),
}));

describe('Server Modules', () => {
    describe('isMobileViewer', () => {
        it('should return true when CloudFront header is "true"', () => {
            const req = {
                header: vi.fn().mockReturnValue('true'),
            } as unknown as Request;

            expect(isMobileViewer(req)).toBe(true);
            expect(req.header).toHaveBeenCalledWith('CloudFront-Is-Mobile-Viewer');
        });

        it('should return false when CloudFront header is missing or not "true"', () => {
            const req = {
                header: vi.fn().mockReturnValue('false'),
            } as unknown as Request;

            expect(isMobileViewer(req)).toBe(false);
            expect(req.header).toHaveBeenCalledWith('CloudFront-Is-Mobile-Viewer');
        });
    });

    describe('getFaviconFilename', () => {
        it('should return dev favicon in dev mode', () => {
            expect(getFaviconFilename(true, ['prod'])).toBe('favicon-dev.ico');
            expect(getFaviconFilename(true)).toBe('favicon-dev.ico');
        });

        it('should return beta favicon for beta environment', () => {
            expect(getFaviconFilename(false, ['beta'])).toBe('favicon-beta.ico');
        });

        it('should return staging favicon for staging environment', () => {
            expect(getFaviconFilename(false, ['staging'])).toBe('favicon-staging.ico');
        });

        it('should return default favicon otherwise', () => {
            expect(getFaviconFilename(false, ['prod'])).toBe('favicon.ico');
            expect(getFaviconFilename(false)).toBe('favicon.ico');
        });
    });

    describe('createDefaultPugRequireHandler', () => {
        it('should handle paths with manifest', () => {
            const manifest = {
                'file1.js': 'file1.hash123.js',
            };
            const handler = createDefaultPugRequireHandler('/static', manifest);

            expect(handler('file1.js')).toBe('/static/file1.hash123.js');
            expect(handler('file2.js')).toBe(''); // Not in manifest
        });

        it('should handle paths without manifest', () => {
            const handler = createDefaultPugRequireHandler('/static');

            expect(handler('file1.js')).toBe('/static/file1.js');
        });
    });

    describe('OldGoogleUrlHandler', () => {
        let handler: OldGoogleUrlHandler;
        let mockCeProps: any;
        let mockRequest: Partial<Request>;
        let mockResponse: Partial<Response>;
        let mockNext: any;

        beforeEach(() => {
            mockCeProps = vi.fn().mockImplementation(key => {
                if (key === 'allowedShortUrlHostRe') return '.*';
                return '';
            });
            handler = new OldGoogleUrlHandler(mockCeProps);

            // Mock ShortLinkResolver (private property)
            Object.defineProperty(handler, 'googleShortUrlResolver', {
                value: {
                    resolve: vi.fn(),
                },
            });

            mockRequest = {
                params: {id: 'abcdef'},
            };
            mockResponse = {
                writeHead: vi.fn(),
                end: vi.fn(),
            };
            mockNext = vi.fn();
        });

        afterEach(() => {
            vi.restoreAllMocks();
        });

        it('should redirect valid URLs', async () => {
            (handler['googleShortUrlResolver'].resolve as any).mockResolvedValue({
                longUrl: 'https://example.com/path',
            });

            await handler.handle(mockRequest as Request, mockResponse as Response, mockNext);

            expect(handler['googleShortUrlResolver'].resolve).toHaveBeenCalledWith('https://goo.gl/abcdef');
            expect(mockResponse.writeHead).toHaveBeenCalledWith(301, {
                Location: 'https://example.com/path',
                'Cache-Control': 'public',
            });
            expect(mockResponse.end).toHaveBeenCalled();
        });

        it('should reject URLs that do not match allowed hosts', async () => {
            (handler['googleShortUrlResolver'].resolve as any).mockResolvedValue({
                longUrl: 'https://example.com/path',
            });
            mockCeProps.mockImplementation(key => {
                if (key === 'allowedShortUrlHostRe') return 'allowed\\.com';
                return '';
            });

            await handler.handle(mockRequest as Request, mockResponse as Response, mockNext);

            expect(mockNext).toHaveBeenCalledWith({
                statusCode: 404,
                message: 'ID "abcdef" could not be found',
            });
        });

        it('should handle errors from URL resolver', async () => {
            (handler['googleShortUrlResolver'].resolve as any).mockRejectedValue(new Error('Not found'));

            await handler.handle(mockRequest as Request, mockResponse as Response, mockNext);

            expect(mockNext).toHaveBeenCalledWith({
                statusCode: 404,
                message: 'ID "abcdef" could not be found',
            });
        });
    });

    // Note: The startListening function tests would be better
    // tested after refactoring the module to be more testable.
    // The current implementation relies on direct module imports
    // which makes mocking difficult.
});
