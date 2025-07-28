// Copyright (c) 2024, Compiler Explorer Authors
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

import zlib from 'node:zlib';
import express from 'express';
import request from 'supertest';
import {beforeAll, describe, expect, it} from 'vitest';
import {GoldenLayoutRootStruct} from '../../lib/clientstate-normalizer.js';
import {HandlerConfig, ShortLinkMetaData} from '../../lib/handlers/handler.interfaces.js';
import {extractJsonFromBufferAndInflateIfRequired, RouteAPI} from '../../lib/handlers/route-api.js';

function possibleCompression(buffer: Buffer): boolean {
    // code used in extractJsonFromBufferAndInflateIfRequired
    // required here to check criticality of test cases
    const firstByte = buffer.at(0); // for uncompressed data this is probably '{'
    return firstByte !== undefined && (firstByte & 0x0f) === 0x8; // https://datatracker.ietf.org/doc/html/rfc1950, https://datatracker.ietf.org/doc/html/rfc1950, for '{' this yields 11
}

describe('extractJsonFromBufferAndInflateIfRequired test cases', () => {
    it('check that data extraction works (good case, no compression)', () => {
        const buffer = Buffer.from('{"a":"test","b":1}');
        expect(possibleCompression(buffer)).toBeFalsy();
        const data = extractJsonFromBufferAndInflateIfRequired(buffer);
        expect(data.a).toBe('test');
        expect(data.b).toBe(1);
    });
    it('check that data extraction works (crirical case - first char indicates possible compression, no compression)', () => {
        const buffer = Buffer.from('810');
        expect(possibleCompression(buffer)).toBeTruthy();
        const data = extractJsonFromBufferAndInflateIfRequired(buffer);
        expect(data).toBe(810);
    });
    it('check that data extraction works (good case, with compression)', () => {
        const text = '{"a":"test test test test test test test test test test test test test","b":1}';
        const buffer = zlib.deflateSync(Buffer.from(text), {level: 9});
        expect(buffer.length).lessThan(text.length);
        expect(possibleCompression(buffer)).toBeTruthy();
        const data = extractJsonFromBufferAndInflateIfRequired(buffer);
        expect(data.a).toBe('test test test test test test test test test test test test test');
        expect(data.b).toBe(1);
    });
    it('check that data extraction fails (bad case)', () => {
        const buffer = Buffer.from('no json');
        expect(() => extractJsonFromBufferAndInflateIfRequired(buffer)).toThrow();
    });
    it('check that data extraction fails for empty buffer', () => {
        const buffer = Buffer.from('');
        expect(() => extractJsonFromBufferAndInflateIfRequired(buffer)).toThrow('Invalid JSON in client state');
    });
    it('check that data extraction fails for whitespace-only buffer', () => {
        const buffer = Buffer.from('   \n\t  ');
        expect(() => extractJsonFromBufferAndInflateIfRequired(buffer)).toThrow('Invalid JSON in client state');
    });
    it('check that data extraction fails for truncated JSON', () => {
        const buffer = Buffer.from('{"sessions":[{"id":1,"languag');
        expect(() => extractJsonFromBufferAndInflateIfRequired(buffer)).toThrow('Invalid JSON in client state');
    });
    it('check that data extraction fails for invalid JSON characters', () => {
        const buffer = Buffer.from('{"invalid": json}');
        expect(() => extractJsonFromBufferAndInflateIfRequired(buffer)).toThrow('Invalid JSON in client state');
    });
    it('check that data extraction fails for gzipped malformed JSON', () => {
        const malformedJson = '{"sessions":[{"id":1,"languag';
        const buffer = zlib.deflateSync(Buffer.from(malformedJson));
        expect(possibleCompression(buffer)).toBeTruthy();
        expect(() => extractJsonFromBufferAndInflateIfRequired(buffer)).toThrow('Invalid JSON in client state');
    });
    it('check that data extraction fails for gzipped empty data', () => {
        const emptyJson = '';
        const buffer = zlib.deflateSync(Buffer.from(emptyJson));
        expect(possibleCompression(buffer)).toBeTruthy();
        expect(() => extractJsonFromBufferAndInflateIfRequired(buffer)).toThrow('Invalid JSON in client state');
    });
    it('check that data extraction handles corrupt gzip data', () => {
        // Create data that looks like gzip (first byte & 0x0f === 0x8) but isn't valid gzip
        const corruptGzipData = Buffer.from([0x18, 0x01, 0x02, 0x03]); // First byte makes it look like gzip
        expect(possibleCompression(corruptGzipData)).toBeTruthy();
        expect(() => extractJsonFromBufferAndInflateIfRequired(corruptGzipData)).toThrow();
    });
});

describe('clientStateHandler', () => {
    let app: express.Express;

    beforeAll(() => {
        app = express();
        const apiHandler = new RouteAPI(app, {
            renderGoldenLayout: (
                config: GoldenLayoutRootStruct,
                metadata: ShortLinkMetaData,
                req: express.Request,
                res: express.Response,
            ) => {
                res.send('This is ok');
            },
        } as unknown as HandlerConfig);
        apiHandler.initializeRoutes();
    });

    it('should return 200 for /clientstate', async () => {
        // A valid document is a base64 encoded JSON object with a sessions array.
        const document = Buffer.from(JSON.stringify({sessions: []}), 'utf-8').toString('base64');
        const response = await request(app).get(`/clientstate/${document}`);
        expect(response.status).toBe(200);
    });
    it('should return 200 for /clientstate even if the base64 contains `/`', async () => {
        let document = '';
        for (let i = 0; i < 1000; i++) {
            document = Buffer.from(
                JSON.stringify({
                    sessions: [],
                    magic: String.fromCharCode(i, i + 1234),
                }),
                'utf-8',
            ).toString('base64');
            if (document.includes('/')) {
                break;
            }
        }
        expect(document).toContain('/');
        const response = await request(app).get(`/clientstate/${document}`);
        expect(response.status).toBe(200);
    });
    it('should return 400 for invalid base64 in /clientstate', async () => {
        const invalidBase64 = 'INVALID_BASE64!!!';
        const response = await request(app).get(`/clientstate/${invalidBase64}`);
        expect(response.status).toBe(400);
    });
    it('should return 400 for empty data in /clientstate', async () => {
        const emptyData = Buffer.from('').toString('base64');
        const response = await request(app).get(`/clientstate/${emptyData}`);
        expect(response.status).toBe(400);
    });
    it('should return 400 for malformed JSON in /clientstate', async () => {
        const malformedJson = Buffer.from('{"invalid": json}').toString('base64');
        const response = await request(app).get(`/clientstate/${malformedJson}`);
        expect(response.status).toBe(400);
    });
    it('should return 400 for truncated JSON in /clientstate', async () => {
        const truncatedJson = Buffer.from('{"sessions":[{"id":1,"languag').toString('base64');
        const response = await request(app).get(`/clientstate/${truncatedJson}`);
        expect(response.status).toBe(400);
    });
    it('should return 400 for binary data in /clientstate', async () => {
        const binaryData = Buffer.from([0x00, 0x01, 0x02, 0xff, 0xfe, 0xfd]).toString('base64');
        const response = await request(app).get(`/clientstate/${binaryData}`);
        expect(response.status).toBe(400);
    });
    it('should return 400 for gzipped malformed JSON in /clientstate', async () => {
        const malformedJson = '{"sessions":[{"id":1,"languag';
        const gzippedData = zlib.deflateSync(Buffer.from(malformedJson)).toString('base64');
        const response = await request(app).get(`/clientstate/${gzippedData}`);
        expect(response.status).toBe(400);
    });
    it('should return 400 for gzipped empty data in /clientstate', async () => {
        const emptyData = zlib.deflateSync(Buffer.from('')).toString('base64');
        const response = await request(app).get(`/clientstate/${emptyData}`);
        expect(response.status).toBe(400);
    });
    it('should return 400 for corrupt gzip data in /clientstate', async () => {
        const corruptGzipData = Buffer.from([0x18, 0x01, 0x02, 0x03]).toString('base64');
        const response = await request(app).get(`/clientstate/${corruptGzipData}`);
        expect(response.status).toBe(400);
    });
});
