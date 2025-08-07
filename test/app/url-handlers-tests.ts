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
import {isMobileViewer, LegacyGoogleUrlHandler} from '../../lib/app/url-handlers.js';

describe('Url Handlers', () => {
    describe('isMobileViewer', () => {
        it('should return true if CloudFront-Is-Mobile-Viewer header is "true"', () => {
            const mockRequest = {
                header: vi.fn().mockImplementation(name => {
                    if (name === 'CloudFront-Is-Mobile-Viewer') return 'true';
                    return undefined;
                }),
            } as unknown as Request;

            expect(isMobileViewer(mockRequest)).toBe(true);
            expect(mockRequest.header).toHaveBeenCalledWith('CloudFront-Is-Mobile-Viewer');
        });

        it('should return false if CloudFront-Is-Mobile-Viewer header is not "true"', () => {
            const mockRequest = {
                header: vi.fn().mockImplementation(name => {
                    if (name === 'CloudFront-Is-Mobile-Viewer') return 'false';
                    return undefined;
                }),
            } as unknown as Request;

            expect(isMobileViewer(mockRequest)).toBe(false);
            expect(mockRequest.header).toHaveBeenCalledWith('CloudFront-Is-Mobile-Viewer');
        });

        it('should return false if CloudFront-Is-Mobile-Viewer header is missing', () => {
            const mockRequest = {
                header: vi.fn().mockReturnValue(undefined),
            } as unknown as Request;

            expect(isMobileViewer(mockRequest)).toBe(false);
            expect(mockRequest.header).toHaveBeenCalledWith('CloudFront-Is-Mobile-Viewer');
        });
    });
    describe('LegacyGoogleUrlHandler', () => {
        let handler: LegacyGoogleUrlHandler;
        let mockCeProps: any;
        let mockAwsProps: any;
        let mockRequest: Partial<Request>;
        let mockResponse: Partial<Response>;
        let mockNext: any;

        beforeEach(() => {
            mockCeProps = vi.fn().mockImplementation(key => {
                if (key === 'allowedShortUrlHostRe') return '.*';
                return '';
            });
            mockAwsProps = vi.fn().mockReturnValue('');
            handler = new LegacyGoogleUrlHandler(mockCeProps, mockAwsProps);

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
});
