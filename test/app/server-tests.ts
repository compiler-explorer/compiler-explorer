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

import type {Request} from 'express';
import {describe, expect, it, vi} from 'vitest';
import {isMobileViewer} from '../../lib/app/server.js';

describe('Server Module', () => {
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

    // Note: We don't test setupWebServer and startListening directly due to their complexity
    // and many dependencies. These would typically be covered by integration tests.
});
