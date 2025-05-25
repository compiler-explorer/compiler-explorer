// Copyright (c) 2017, Compiler Explorer Authors
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

import {DatabaseSync} from 'node:sqlite';

import {logger} from '../logger.js';

// This will stop working in August 2025.
// https://developers.googleblog.com/en/google-url-shortener-links-will-no-longer-be-available/
export class ShortLinkResolver {
    private readonly database?: DatabaseSync;

    constructor(sqlitePath?: string) {
        if (sqlitePath) {
            this.database = new DatabaseSync(sqlitePath, {readOnly: true});
        }
    }

    async resolve(url: string): Promise<{longUrl: string}> {
        // Extract fragment from URL (e.g., "abcd" from "https://goo.gl/abcd")
        const fragment = url.split('/').pop()?.split('?')[0];

        if (this.database && fragment) {
            try {
                const query = this.database.prepare('SELECT expanded_url FROM google_links WHERE fragment = ?');
                const result = query.get(fragment) as {expanded_url: string | null} | undefined;

                if (result) {
                    if (result.expanded_url === null) {
                        throw new Error('404: Not Found');
                    }
                    return {
                        longUrl: result.expanded_url,
                    };
                }
            } catch (err) {
                // If database lookup fails, fall back to Google URL shortener
                logger.error('Database lookup failed:', err);
            }
        }

        // In August 2025, the Google URL shortener will stop working. At that point, use this code:
        //     throw new Error('404: Not Found');

        // Fall back to Google URL shortener
        const settings: RequestInit = {
            method: 'HEAD',
            redirect: 'manual',
        };

        const res = await fetch(url + '?si=1', settings);
        if (res.status !== 302) {
            throw new Error(`Got response ${res.status}`);
        }
        const targetLocation = res.headers.get('Location');
        if (!targetLocation) {
            throw new Error(`Missing location url in ${targetLocation}`);
        }
        return {
            longUrl: targetLocation,
        };
    }
}
