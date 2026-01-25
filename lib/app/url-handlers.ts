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

import url from 'node:url';
import type {NextFunction, Request, Response} from 'express';

import {unwrapString} from '../assert.js';
import {logger} from '../logger.js';
import {PropertyGetter} from '../properties.interfaces.js';
import {ShortLinkResolver} from '../shortener/google.js';

/**
 * Detects if the request is from a mobile viewer
 * @param req - Express request object
 * @returns true if the request is from a mobile viewer
 */
export function isMobileViewer(req: Request): boolean {
    return req.header('CloudFront-Is-Mobile-Viewer') === 'true';
}

/**
 * Handles legacy Google URL shortener redirects
 */
export class LegacyGoogleUrlHandler {
    private readonly googleShortUrlResolver: ShortLinkResolver;

    /**
     * Create a new handler for legacy Google URL shortcuts
     * @param ceProps - Compiler Explorer properties
     * @param awsProps - AWS properties
     */
    constructor(
        private readonly ceProps: PropertyGetter,
        awsProps: PropertyGetter,
    ) {
        this.googleShortUrlResolver = new ShortLinkResolver(awsProps);
    }

    /**
     * Handle a request for a legacy Google short URL
     * @param req - Express request object
     * @param res - Express response object
     * @param next - Express next function
     */
    async handle(req: Request, res: Response, next: NextFunction) {
        const id = unwrapString(req.params.id);
        const googleUrl = `https://goo.gl/${encodeURIComponent(id)}`;

        try {
            const resultObj = await this.googleShortUrlResolver.resolve(googleUrl);
            const parsed = new url.URL(resultObj.longUrl);
            const allowedRe = new RegExp(this.ceProps<string>('allowedShortUrlHostRe'));

            if (parsed.host.match(allowedRe) === null) {
                logger.warn(`Denied access to short URL ${id} - linked to ${resultObj.longUrl}`);
                return next({
                    statusCode: 404,
                    message: `ID "${id}" could not be found`,
                });
            }

            res.writeHead(301, {
                Location: resultObj.longUrl,
                'Cache-Control': 'public',
            });
            res.end();
        } catch (err: unknown) {
            logger.error(`Failed to expand ${googleUrl} - ${err}`);
            next({
                statusCode: 404,
                message: `ID "${id}" could not be found`,
            });
        }
    }
}
