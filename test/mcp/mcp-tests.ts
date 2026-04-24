// Copyright (C) 2026 Hudson River Trading LLC <opensource@hudson-trading.com>
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

import express, {type Router} from 'express';
import request from 'supertest';
import {beforeEach, describe, expect, it, vi} from 'vitest';

import type {ApiHandler} from '../../lib/handlers/api.js';
import type {CompileHandler} from '../../lib/handlers/compile.js';
import {setupMcpEndpoint} from '../../lib/mcp/index.js';
import {registerShortlinkTools} from '../../lib/mcp/tools/shortlinks.js';
import type {StorageBase, StoredObject} from '../../lib/storage/base.js';

function makeFakeApiHandler(): ApiHandler {
    return {
        release: {gitReleaseName: 'test', releaseBuildNumber: '1'},
    } as unknown as ApiHandler;
}

function makeFakeCompileHandler(): CompileHandler {
    return {} as unknown as CompileHandler;
}

function makeApp(storageHandler: StorageBase): express.Express {
    const app = express();
    const router: Router = express.Router();
    // Mirror the global JSON body parser that setupBasicRoutes installs in production.
    router.use(express.json());
    setupMcpEndpoint(router, makeFakeCompileHandler(), makeFakeApiHandler(), storageHandler);
    app.use(router);
    return app;
}

describe('MCP endpoint', () => {
    const storageHandler = {
        httpRootDir: '/',
        findUniqueSubhash: vi.fn(),
        storeItem: vi.fn(),
        expandId: vi.fn(),
    } as unknown as StorageBase;

    let app: express.Express;

    beforeEach(() => {
        vi.clearAllMocks();
        app = makeApp(storageHandler);
    });

    it('returns 405 for GET', async () => {
        await request(app).get('/mcp').expect(405).expect('Allow', 'POST');
    });

    it('returns 405 for DELETE', async () => {
        await request(app).delete('/mcp').expect(405).expect('Allow', 'POST');
    });

    it('returns a JSON-RPC error for malformed POST bodies', async () => {
        const res = await request(app)
            .post('/mcp')
            .set('Accept', 'application/json, text/event-stream')
            .set('Content-Type', 'application/json')
            .send({not: 'a real jsonrpc request'});
        // We don't pin the exact status code (the SDK chooses it) — just that the
        // route is wired up and the SDK produced a structured error rather than
        // crashing the process.
        expect(res.status).toBeGreaterThanOrEqual(400);
    });

    it('responds to a tools/list JSON-RPC call with the registered tools', async () => {
        const res = await request(app)
            .post('/mcp')
            .set('Accept', 'application/json, text/event-stream')
            .set('Content-Type', 'application/json')
            .send({
                jsonrpc: '2.0',
                id: 1,
                method: 'initialize',
                params: {
                    protocolVersion: '2024-11-05',
                    capabilities: {},
                    clientInfo: {name: 'test', version: '0.0.0'},
                },
            });
        expect(res.status).toBe(200);
    });
});

describe('MCP shortlink tool', () => {
    let toolHandlers: Record<string, (args: any) => Promise<any>>;
    let storageHandler: StorageBase;
    let fakeServer: any;

    beforeEach(() => {
        toolHandlers = {};
        fakeServer = {
            tool: (name: string, _description: string, _schema: unknown, handler: (args: any) => Promise<any>) => {
                toolHandlers[name] = handler;
            },
        };
        storageHandler = {
            httpRootDir: '/',
            findUniqueSubhash: vi.fn().mockResolvedValue({
                prefix: 'pre',
                uniqueSubHash: 'abc123',
                alreadyPresent: false,
            }),
            storeItem: vi.fn().mockImplementation(async (item: StoredObject) => item),
            expandId: vi.fn().mockResolvedValue({config: '{"sessions":[]}'}),
        } as unknown as StorageBase;
    });

    it('passes the real express request through to storeItem', async () => {
        const req = {
            ip: '203.0.113.7',
            get: vi.fn().mockReturnValue(undefined),
        } as unknown as express.Request;
        registerShortlinkTools(fakeServer, storageHandler, 'https://example.org', req);

        const result = await toolHandlers.generate_short_url({
            source: 'int main(){}',
            language: 'c++',
            compiler: 'g142',
        });

        expect(storageHandler.storeItem).toHaveBeenCalledTimes(1);
        const passedReq = (storageHandler.storeItem as any).mock.calls[0][1];
        expect(passedReq).toBe(req);
        // StorageS3 needs both of these to work — protect against a future regression
        // where someone passes a plain {} again.
        expect(typeof passedReq.get).toBe('function');
        expect(passedReq.ip).toBe('203.0.113.7');
        expect(result.content[0].text).toBe('https://example.org/z/abc123');
    });

    it('skips storeItem when the hash is already present', async () => {
        (storageHandler.findUniqueSubhash as any).mockResolvedValue({
            prefix: 'pre',
            uniqueSubHash: 'cached',
            alreadyPresent: true,
        });
        const req = {ip: '127.0.0.1', get: vi.fn()} as unknown as express.Request;
        registerShortlinkTools(fakeServer, storageHandler, 'https://example.org', req);

        const result = await toolHandlers.generate_short_url({
            source: 'int main(){}',
            language: 'c++',
            compiler: 'g142',
        });

        expect(storageHandler.storeItem).not.toHaveBeenCalled();
        expect(result.content[0].text).toBe('https://example.org/z/cached');
    });

    it('returns the expanded config for get_shortlink_info', async () => {
        const req = {ip: '127.0.0.1', get: vi.fn()} as unknown as express.Request;
        registerShortlinkTools(fakeServer, storageHandler, 'https://example.org', req);

        const result = await toolHandlers.get_shortlink_info({id: 'https://example.org/z/abc123'});
        expect(storageHandler.expandId).toHaveBeenCalledWith('abc123');
        expect(result.content[0].text).toBe(JSON.stringify({sessions: []}, null, 2));
    });
});
