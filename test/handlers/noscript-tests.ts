// Copyright (c) 2026, Compiler Explorer Authors
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

import express from 'express';
import request from 'supertest';
import {beforeAll, describe, expect, it} from 'vitest';
import {NoScriptHandler} from '../../lib/handlers/noscript.js';
import type {ClientOptionsHandler} from '../../lib/options-handler.js';
import type {StorageBase} from '../../lib/storage/index.js';

describe('NoScriptHandler clientstate', () => {
    let app: express.Express;

    beforeAll(() => {
        app = express();

        // Mock the render method before initializing routes
        app.use((req, res, next) => {
            res.render = (view: string, options?: object) => {
                res.json({view, options});
            };
            next();
        });

        const mockClientOptionsHandler = {
            get: () => ({
                languages: {
                    'c++': {example: '// Example C++ code'},
                },
                defaultCompiler: {
                    'c++': 'g121',
                },
            }),
        } as unknown as ClientOptionsHandler;

        const mockStorageHandler = {} as StorageBase;

        const mockRenderConfig = (options: Record<string, any>, urlOptions?: Record<string, any>) => ({
            ...options,
            ...urlOptions,
        });

        const handler = new NoScriptHandler(app, mockClientOptionsHandler, mockRenderConfig, mockStorageHandler, 'c++');
        handler.initializeRoutes();
    });

    it('should return 200 for valid clientstate', async () => {
        const clientstate = {
            sessions: [
                {
                    id: 1,
                    language: 'c++',
                    source: 'int main() { return 0; }',
                    compilers: [{id: 'g121', options: ''}],
                },
            ],
        };
        const encoded = Buffer.from(JSON.stringify(clientstate)).toString('base64');
        const response = await request(app).get(`/noscript/clientstate/${encoded}`);
        expect(response.status).toBe(200);
    });

    it('should return 200 for clientstate with slashes in base64', async () => {
        let encoded = '';
        for (let i = 0; i < 1000; i++) {
            const clientstate = {
                sessions: [
                    {
                        id: 1,
                        language: 'c++',
                        source: `int main() { return ${i}; } // ${String.fromCharCode(i, i + 1234)}`,
                        compilers: [{id: 'g121', options: ''}],
                    },
                ],
            };
            encoded = Buffer.from(JSON.stringify(clientstate)).toString('base64');
            if (encoded.includes('/')) {
                break;
            }
        }
        expect(encoded).toContain('/');
        const response = await request(app).get(`/noscript/clientstate/${encoded}`);
        expect(response.status).toBe(200);
    });

    it('should return 400 for invalid base64', async () => {
        const response = await request(app).get('/noscript/clientstate/INVALID_BASE64!!!');
        expect(response.status).toBe(400);
    });

    it('should return 400 for malformed JSON', async () => {
        const malformedJson = Buffer.from('{"invalid": json}').toString('base64');
        const response = await request(app).get(`/noscript/clientstate/${malformedJson}`);
        expect(response.status).toBe(400);
    });

    it('should return 400 for truncated JSON', async () => {
        const truncatedJson = Buffer.from('{"sessions":[{"id":1,"languag').toString('base64');
        const response = await request(app).get(`/noscript/clientstate/${truncatedJson}`);
        expect(response.status).toBe(400);
    });
});
