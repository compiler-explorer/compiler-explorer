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

import nock from 'nock';
import {afterAll, describe, expect, it} from 'vitest';

import * as google from '../lib/shortener/google.js';

const googlDomain = 'https://goo.gl';
const shortUrl = '/short';

describe('Google short URL resolver tests', () => {
    afterAll(() => {
        nock.cleanAll();
    });

    const resolver = new google.ShortLinkResolver();

    it('Resolves simple URLs', async () => {
        nock(googlDomain).head(shortUrl).reply(302, {}, {location: 'http://long.url/'});

        await expect(resolver.resolve(googlDomain + shortUrl)).resolves.toEqual({longUrl: 'http://long.url/'});
    });

    it('Handles missing long urls', async () => {
        nock(googlDomain).head(shortUrl).reply(404);

        await expect(resolver.resolve(googlDomain + shortUrl)).rejects.toThrow('Got response 404');
    });

    it('Handles missing location header', async () => {
        nock(googlDomain).head(shortUrl).reply(302);

        await expect(resolver.resolve(googlDomain + shortUrl)).rejects.toThrow('Missing location url in undefined');
    });

    it('Handles failed requests', async () => {
        nock(googlDomain).head(shortUrl).replyWithError('Something went wrong');

        await expect(resolver.resolve(googlDomain + shortUrl)).rejects.toThrow('Something went wrong');
    });
});
