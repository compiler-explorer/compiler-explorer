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
import {registerAsmDocsTool} from '../../lib/mcp/tools/asm-docs.js';
import {registerCompileTool} from '../../lib/mcp/tools/compile.js';
import {registerCompilersTool} from '../../lib/mcp/tools/compilers.js';
import {registerLanguagesTool} from '../../lib/mcp/tools/languages.js';
import {registerLibrariesTool} from '../../lib/mcp/tools/libraries.js';
import {registerShortlinkTools} from '../../lib/mcp/tools/shortlinks.js';
import type {StorageBase, StoredObject} from '../../lib/storage/base.js';
import type {CompilerInfo} from '../../types/compiler.interfaces.js';

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
        await request(app).get('/mcp').expect(405).expect('Allow', 'POST, OPTIONS');
    });

    it('returns 405 for DELETE', async () => {
        await request(app).delete('/mcp').expect(405).expect('Allow', 'POST, OPTIONS');
    });

    it('OPTIONS preflight advertises Allow-Methods (browser clients need this for cross-origin POST)', async () => {
        const res = await request(app).options('/mcp');
        expect(res.status).toBe(200);
        // Without these, a browser making a cross-origin POST with Content-Type:
        // application/json fails preflight before the request is even sent.
        expect(res.headers['access-control-allow-methods']).toMatch(/POST/);
        expect(res.headers['access-control-allow-origin']).toBe('*');
        expect(res.headers['access-control-allow-headers']).toMatch(/Content-Type/);
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

    it('accepts the initialize handshake', async () => {
        const res = await postJsonRpc(app, {
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
        const body = parseMcpResponse(res);
        expect(body.result.serverInfo.name).toBe('compiler-explorer');
    });

    it('lists every registered tool when asked for tools/list', async () => {
        // Initialize first; the SDK requires it even in stateless mode.
        await postJsonRpc(app, {
            jsonrpc: '2.0',
            id: 1,
            method: 'initialize',
            params: {
                protocolVersion: '2024-11-05',
                capabilities: {},
                clientInfo: {name: 'test', version: '0.0.0'},
            },
        });
        const res = await postJsonRpc(app, {jsonrpc: '2.0', id: 2, method: 'tools/list'});
        expect(res.status).toBe(200);
        const body = parseMcpResponse(res);
        const toolNames = body.result.tools.map((t: {name: string}) => t.name).sort();
        expect(toolNames).toEqual([
            'compile',
            'generate_short_url',
            'get_shortlink_info',
            'list_compilers',
            'list_languages',
            'list_libraries',
            'lookup_asm_instruction',
        ]);
    });
});

function postJsonRpc(app: express.Express, payload: object) {
    return request(app)
        .post('/mcp')
        .set('Accept', 'application/json, text/event-stream')
        .set('Content-Type', 'application/json')
        .send(payload);
}

// The streamable-HTTP transport returns either a plain JSON body or an
// `event: message\ndata: {...}` SSE frame depending on the negotiated content
// type. Normalise both shapes for assertions.
function parseMcpResponse(res: {headers: Record<string, string>; body: any; text: string}): any {
    const contentType = res.headers['content-type'] || '';
    if (contentType.includes('application/json')) return res.body;
    const dataLine = res.text.split('\n').find(l => l.startsWith('data:'));
    if (!dataLine) throw new Error(`No data: line in SSE response: ${res.text}`);
    return JSON.parse(dataLine.slice('data:'.length).trim());
}

describe('MCP shortlink tool', () => {
    let toolHandlers: Record<string, (args: any) => Promise<any>>;
    let storageHandler: StorageBase;
    let fakeServer: any;

    beforeEach(() => {
        toolHandlers = {};
        fakeServer = {
            tool: (name: string, ..._rest: unknown[]) => {
                toolHandlers[name] = _rest[_rest.length - 1] as (args: any) => Promise<any>;
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

    const fakeApiHandler = {
        getLibrariesAsArray: () => [],
    } as unknown as ApiHandler;

    it('passes the real express request through to storeItem', async () => {
        const req = {
            ip: '203.0.113.7',
            get: vi.fn().mockReturnValue(undefined),
        } as unknown as express.Request;
        registerShortlinkTools(fakeServer, storageHandler, fakeApiHandler, 'https://example.org', req);

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
        expect(JSON.parse(result.content[0].text)).toEqual({url: 'https://example.org/z/abc123'});
    });

    it('skips storeItem when the hash is already present', async () => {
        (storageHandler.findUniqueSubhash as any).mockResolvedValue({
            prefix: 'pre',
            uniqueSubHash: 'cached',
            alreadyPresent: true,
        });
        const req = {ip: '127.0.0.1', get: vi.fn()} as unknown as express.Request;
        registerShortlinkTools(fakeServer, storageHandler, fakeApiHandler, 'https://example.org', req);

        const result = await toolHandlers.generate_short_url({
            source: 'int main(){}',
            language: 'c++',
            compiler: 'g142',
        });

        expect(storageHandler.storeItem).not.toHaveBeenCalled();
        expect(JSON.parse(result.content[0].text)).toEqual({url: 'https://example.org/z/cached'});
    });

    it('normalises library version to id when generating a short url', async () => {
        const req = {ip: '127.0.0.1', get: vi.fn()} as unknown as express.Request;
        const apiWithLibs = {
            getLibrariesAsArray: () => [{id: 'boost', versions: [{id: '188', version: '1.88.0'}]}],
        } as unknown as ApiHandler;
        registerShortlinkTools(fakeServer, storageHandler, apiWithLibs, 'https://example.org', req);

        // Pass the human form — should be normalised to the id before being saved.
        await toolHandlers.generate_short_url({
            source: 'int main(){}',
            language: 'c++',
            compiler: 'g161',
            libraries: [{id: 'boost', version: '1.88.0'}],
        });

        const stored = (storageHandler.storeItem as any).mock.calls[0][0];
        const config = JSON.parse(stored.config);
        expect(config.sessions[0].compilers[0].libs).toEqual([{id: 'boost', ver: '188'}]);
    });

    it('returns the expanded config for get_shortlink_info in compile-ready shape', async () => {
        const req = {ip: '127.0.0.1', get: vi.fn()} as unknown as express.Request;
        // Saved config uses the legacy shape: compilers[].id / .libs[].ver
        const handler = {
            ...storageHandler,
            expandId: vi.fn().mockResolvedValue({
                config: JSON.stringify({
                    sessions: [
                        {
                            language: 'c++',
                            source: 'int main(){}',
                            compilers: [{id: 'g161', options: '-O2', libs: [{id: 'boost', ver: '188'}]}],
                        },
                    ],
                }),
            }),
        } as unknown as StorageBase;
        registerShortlinkTools(fakeServer, handler, fakeApiHandler, 'https://example.org', req);

        const result = await toolHandlers.get_shortlink_info({id: 'https://example.org/z/abc123'});
        const parsed = JSON.parse(result.content[0].text);
        // Renamed for compile-tool symmetry: id → compiler, libs[].ver → libraries[].version.
        expect(parsed.sessions[0].compilers[0]).toEqual({
            compiler: 'g161',
            options: '-O2',
            libraries: [{id: 'boost', version: '188'}],
        });
    });
});

function makeFakeServer(): {fakeServer: any; toolHandlers: Record<string, (args: any) => Promise<any>>} {
    const toolHandlers: Record<string, (args: any) => Promise<any>> = {};
    const fakeServer = {
        // The MCP SDK's server.tool overload accepts either (name, description, schema, handler)
        // or (name, description, handler). Last argument is always the handler.
        tool: (name: string, ..._rest: unknown[]) => {
            toolHandlers[name] = _rest[_rest.length - 1] as (args: any) => Promise<any>;
        },
    };
    return {fakeServer, toolHandlers};
}

function makeCompiler(id: string, lang = 'c++', extra: Partial<CompilerInfo> = {}): CompilerInfo {
    return {
        id,
        name: id.toUpperCase(),
        lang,
        compilerType: '',
        semver: '',
        instructionSet: 'amd64',
        releaseTrack: 'stable',
        ...extra,
    } as unknown as CompilerInfo;
}

describe('MCP list_compilers tool', () => {
    it('returns full per-entry detail when under the cap', async () => {
        const {fakeServer, toolHandlers} = makeFakeServer();
        const apiHandler = {compilers: [makeCompiler('g142'), makeCompiler('clang20')]} as unknown as ApiHandler;
        registerCompilersTool(fakeServer, apiHandler);

        const result = await toolHandlers.list_compilers({});
        const parsed = JSON.parse(result.content[0].text);
        expect(parsed.total).toBe(2);
        expect(parsed.leanMode).toBeUndefined();
        expect(parsed.compilers[0]).toEqual({
            id: 'g142',
            name: 'G142',
            lang: 'c++',
            compilerType: '',
            semver: '',
            instructionSet: 'amd64',
            releaseTrack: 'stable',
            supportsExecute: false,
            supportsBinary: false,
        });
    });

    it('filters by language before applying match', async () => {
        const {fakeServer, toolHandlers} = makeFakeServer();
        const apiHandler = {
            compilers: [makeCompiler('g142', 'c++'), makeCompiler('rustc', 'rust'), makeCompiler('clang20', 'c++')],
        } as unknown as ApiHandler;
        registerCompilersTool(fakeServer, apiHandler);

        const result = await toolHandlers.list_compilers({language: 'c++', match: 'CLANG'});
        const parsed = JSON.parse(result.content[0].text);
        expect(parsed.compilers).toEqual([
            {
                id: 'clang20',
                name: 'CLANG20',
                lang: 'c++',
                compilerType: '',
                semver: '',
                instructionSet: 'amd64',
                releaseTrack: 'stable',
                supportsExecute: false,
                supportsBinary: false,
            },
        ]);
    });

    it('degrades to lean mode when the filtered set exceeds the cap', async () => {
        const {fakeServer, toolHandlers} = makeFakeServer();
        const apiHandler = {
            compilers: Array.from({length: 80}, (_, i) => makeCompiler(`g${i}`)),
        } as unknown as ApiHandler;
        registerCompilersTool(fakeServer, apiHandler);

        const result = await toolHandlers.list_compilers({});
        const parsed = JSON.parse(result.content[0].text);
        // Lean mode returns ALL matches with id+name only.
        expect(parsed.compilers).toHaveLength(80);
        expect(parsed.total).toBe(80);
        expect(parsed.leanMode).toBe(true);
        expect(parsed.compilers[0]).toEqual({id: 'g0', name: 'G0'});
        expect(parsed.compilers[0].compilerType).toBeUndefined();
        expect(parsed.hint).toMatch(/Refine your filter/);
        expect(parsed.hint).toMatch(/exact id/);
    });

    it('keeps full shape when maxResults is raised above the filtered count', async () => {
        const {fakeServer, toolHandlers} = makeFakeServer();
        const apiHandler = {
            compilers: Array.from({length: 80}, (_, i) => makeCompiler(`g${i}`)),
        } as unknown as ApiHandler;
        registerCompilersTool(fakeServer, apiHandler);

        const result = await toolHandlers.list_compilers({maxResults: 200});
        const parsed = JSON.parse(result.content[0].text);
        expect(parsed.compilers).toHaveLength(80);
        expect(parsed.leanMode).toBeUndefined();
        expect(parsed.compilers[0].instructionSet).toBe('amd64');
    });

    it('returns the catalog index when lean: true is set explicitly', async () => {
        const {fakeServer, toolHandlers} = makeFakeServer();
        const apiHandler = {
            compilers: [makeCompiler('g142'), makeCompiler('clang20')],
        } as unknown as ApiHandler;
        registerCompilersTool(fakeServer, apiHandler);

        const result = await toolHandlers.list_compilers({lean: true});
        const parsed = JSON.parse(result.content[0].text);
        expect(parsed.compilers).toEqual([
            {id: 'g142', name: 'G142'},
            {id: 'clang20', name: 'CLANG20'},
        ]);
        expect(parsed.leanMode).toBe(true);
        expect(parsed.hint).toBeUndefined();
    });

    it('filters by instructionSet before applying match', async () => {
        const {fakeServer, toolHandlers} = makeFakeServer();
        const apiHandler = {
            compilers: [
                makeCompiler('g142_amd64', 'c++', {instructionSet: 'amd64'}),
                makeCompiler('g142_arm', 'c++', {instructionSet: 'aarch64'}),
                makeCompiler('rustc', 'rust', {instructionSet: 'amd64'}),
            ],
        } as unknown as ApiHandler;
        registerCompilersTool(fakeServer, apiHandler);

        const result = await toolHandlers.list_compilers({language: 'c++', instructionSet: 'amd64'});
        const parsed = JSON.parse(result.content[0].text);
        expect(parsed.compilers.map((c: any) => c.id)).toEqual(['g142_amd64']);
    });

    it('instructionSet + latestPerMajor cleanly answers "newest x86-64 GCC"', async () => {
        const {fakeServer, toolHandlers} = makeFakeServer();
        const apiHandler = {
            compilers: [
                // amd64 entries — should appear
                makeCompiler('g142', 'c++', {
                    isSemVer: true,
                    semver: '14.2',
                    instructionSet: 'amd64',
                    releaseTrack: 'stable',
                }),
                makeCompiler('g161', 'c++', {
                    isSemVer: true,
                    semver: '16.1',
                    instructionSet: 'amd64',
                    releaseTrack: 'stable',
                }),
                // arm64 entries — should be filtered out
                makeCompiler('garm161', 'c++', {
                    isSemVer: true,
                    semver: '16.1',
                    instructionSet: 'aarch64',
                    releaseTrack: 'stable',
                }),
            ],
        } as unknown as ApiHandler;
        registerCompilersTool(fakeServer, apiHandler);

        const result = await toolHandlers.list_compilers({
            language: 'c++',
            instructionSet: 'amd64',
            latestPerMajor: true,
        });
        const parsed = JSON.parse(result.content[0].text);
        expect(parsed.compilers.map((c: any) => c.id).sort()).toEqual(['g142', 'g161']);
    });

    it('latestPerMajor: stable compilers grouped by (instructionSet, major), newest per major wins', async () => {
        const {fakeServer, toolHandlers} = makeFakeServer();
        const apiHandler = {
            compilers: [
                makeCompiler('g141', 'c++', {
                    isSemVer: true,
                    semver: '14.1',
                    instructionSet: 'amd64',
                    releaseTrack: 'stable',
                }),
                makeCompiler('g142', 'c++', {
                    isSemVer: true,
                    semver: '14.2',
                    instructionSet: 'amd64',
                    releaseTrack: 'stable',
                }),
                makeCompiler('g152', 'c++', {
                    isSemVer: true,
                    semver: '15.2',
                    instructionSet: 'amd64',
                    releaseTrack: 'stable',
                }),
                makeCompiler('g161', 'c++', {
                    isSemVer: true,
                    semver: '16.1',
                    instructionSet: 'amd64',
                    releaseTrack: 'stable',
                }),
                makeCompiler('garm142', 'c++', {
                    isSemVer: true,
                    semver: '14.2',
                    instructionSet: 'aarch64',
                    releaseTrack: 'stable',
                }),
            ],
        } as unknown as ApiHandler;
        registerCompilersTool(fakeServer, apiHandler);

        const result = await toolHandlers.list_compilers({latestPerMajor: true});
        const parsed = JSON.parse(result.content[0].text);
        const ids = parsed.compilers.map((c: any) => c.id).sort();
        // amd64: 14.2 (beats 14.1), 15.2, 16.1; aarch64: 14.2 (only entry).
        expect(ids).toEqual(['g142', 'g152', 'g161', 'garm142'].sort());
    });

    it('latestPerMajor: every nightly track is kept (no collapsing)', async () => {
        const {fakeServer, toolHandlers} = makeFakeServer();
        const apiHandler = {
            compilers: [
                makeCompiler('gsnapshot', 'c++', {
                    isSemVer: true,
                    semver: '(trunk)',
                    instructionSet: 'amd64',
                    releaseTrack: 'nightly',
                }),
                makeCompiler('nightly', 'rust', {
                    isSemVer: true,
                    semver: 'nightly',
                    instructionSet: 'amd64',
                    releaseTrack: 'nightly',
                }),
                makeCompiler('rustccggcc-master', 'rust', {
                    isSemVer: false,
                    semver: '',
                    instructionSet: 'amd64',
                    releaseTrack: 'nightly',
                }),
            ],
        } as unknown as ApiHandler;
        registerCompilersTool(fakeServer, apiHandler);

        const result = await toolHandlers.list_compilers({latestPerMajor: true});
        const parsed = JSON.parse(result.content[0].text);
        expect(parsed.compilers.map((c: any) => c.id).sort()).toEqual(
            ['gsnapshot', 'nightly', 'rustccggcc-master'].sort(),
        );
    });

    it('latestPerMajor: prerelease tracks (rust beta, dxc preview) are first-class', async () => {
        const {fakeServer, toolHandlers} = makeFakeServer();
        const apiHandler = {
            compilers: [
                makeCompiler('r1950', 'rust', {
                    isSemVer: true,
                    semver: '1.95.0',
                    instructionSet: 'amd64',
                    releaseTrack: 'stable',
                }),
                makeCompiler('beta', 'rust', {
                    isSemVer: true,
                    semver: 'beta',
                    instructionSet: 'amd64',
                    releaseTrack: 'prerelease',
                }),
                makeCompiler('dxc_preview', 'hlsl', {
                    isSemVer: true,
                    semver: '1.8.2306-preview',
                    instructionSet: 'amd64',
                    releaseTrack: 'prerelease',
                }),
            ],
        } as unknown as ApiHandler;
        registerCompilersTool(fakeServer, apiHandler);

        const result = await toolHandlers.list_compilers({latestPerMajor: true});
        const parsed = JSON.parse(result.content[0].text);
        expect(parsed.compilers.map((c: any) => c.id).sort()).toEqual(['beta', 'dxc_preview', 'r1950'].sort());
    });

    it('latestPerMajor: experimental compilers skipped by default with a hint', async () => {
        const {fakeServer, toolHandlers} = makeFakeServer();
        const apiHandler = {
            compilers: [
                makeCompiler('g142', 'c++', {
                    isSemVer: true,
                    semver: '14.2',
                    instructionSet: 'amd64',
                    releaseTrack: 'stable',
                }),
                makeCompiler('gcontracts-trunk', 'c++', {
                    isSemVer: true,
                    semver: '(contracts)',
                    instructionSet: 'amd64',
                    releaseTrack: 'experimental',
                }),
                makeCompiler('gmodules-trunk', 'c++', {
                    isSemVer: true,
                    semver: '(modules)',
                    instructionSet: 'amd64',
                    releaseTrack: 'experimental',
                }),
            ],
        } as unknown as ApiHandler;
        registerCompilersTool(fakeServer, apiHandler);

        const result = await toolHandlers.list_compilers({latestPerMajor: true});
        const parsed = JSON.parse(result.content[0].text);
        expect(parsed.compilers.map((c: any) => c.id)).toEqual(['g142']);
        expect(parsed.droppedExperimental).toBe(2);
        expect(parsed.latestHint).toMatch(/experimental/);
        expect(parsed.latestHint).toMatch(/includeExperimental/);
    });

    it('list_compilers full shape exposes supportsExecute / supportsBinary', async () => {
        const {fakeServer, toolHandlers} = makeFakeServer();
        const apiHandler = {
            compilers: [makeCompiler('g142', 'c++', {supportsExecute: true, supportsBinary: true})],
        } as unknown as ApiHandler;
        registerCompilersTool(fakeServer, apiHandler);

        const result = await toolHandlers.list_compilers({});
        const parsed = JSON.parse(result.content[0].text);
        expect(parsed.compilers[0].supportsExecute).toBe(true);
        expect(parsed.compilers[0].supportsBinary).toBe(true);
    });

    it('latestPerMajor: includeExperimental: true brings the experimental forks back', async () => {
        const {fakeServer, toolHandlers} = makeFakeServer();
        const apiHandler = {
            compilers: [
                makeCompiler('g142', 'c++', {
                    isSemVer: true,
                    semver: '14.2',
                    instructionSet: 'amd64',
                    releaseTrack: 'stable',
                }),
                makeCompiler('gcontracts-trunk', 'c++', {
                    isSemVer: true,
                    semver: '(contracts)',
                    instructionSet: 'amd64',
                    releaseTrack: 'experimental',
                }),
            ],
        } as unknown as ApiHandler;
        registerCompilersTool(fakeServer, apiHandler);

        const result = await toolHandlers.list_compilers({latestPerMajor: true, includeExperimental: true});
        const parsed = JSON.parse(result.content[0].text);
        expect(parsed.compilers.map((c: any) => c.id).sort()).toEqual(['g142', 'gcontracts-trunk'].sort());
        expect(parsed.droppedExperimental).toBeUndefined();
        expect(parsed.latestHint).toBeUndefined();
    });
});

describe('MCP list_libraries tool', () => {
    function makeApiHandlerWithLibs(libs: {id: string; name: string}[]): ApiHandler {
        return {getLibrariesAsArray: () => libs} as unknown as ApiHandler;
    }

    it('filters libraries by match substring', async () => {
        const {fakeServer, toolHandlers} = makeFakeServer();
        const apiHandler = makeApiHandlerWithLibs([
            {id: 'fmt', name: '{fmt}'},
            {id: 'boost', name: 'Boost'},
            {id: 'eigen', name: 'Eigen'},
        ]);
        registerLibrariesTool(fakeServer, apiHandler);

        const result = await toolHandlers.list_libraries({language: 'c++', match: 'BOOST'});
        const parsed = JSON.parse(result.content[0].text);
        expect(parsed.libraries.map((l: any) => l.id)).toEqual(['boost']);
        expect(parsed.total).toBe(1);
    });

    it('degrades to lean mode above the cap', async () => {
        const {fakeServer, toolHandlers} = makeFakeServer();
        const libs = Array.from({length: 50}, (_, i) => ({id: `lib${i}`, name: `Library ${i}`, versions: ['1.0']}));
        const apiHandler = makeApiHandlerWithLibs(libs as any);
        registerLibrariesTool(fakeServer, apiHandler);

        const result = await toolHandlers.list_libraries({language: 'c++'});
        const parsed = JSON.parse(result.content[0].text);
        expect(parsed.libraries).toHaveLength(50);
        expect(parsed.total).toBe(50);
        expect(parsed.leanMode).toBe(true);
        expect(parsed.libraries[0]).toEqual({id: 'lib0', name: 'Library 0'});
        expect(parsed.libraries[0].versions).toBeUndefined();
        expect(parsed.hint).toMatch(/exact id/);
    });

    it('returns the catalog index when lean: true is set explicitly', async () => {
        const {fakeServer, toolHandlers} = makeFakeServer();
        const libs = Array.from({length: 5}, (_, i) => ({id: `lib${i}`, name: `Library ${i}`, versions: ['1.0']}));
        const apiHandler = makeApiHandlerWithLibs(libs as any);
        registerLibrariesTool(fakeServer, apiHandler);

        const result = await toolHandlers.list_libraries({language: 'c++', lean: true});
        const parsed = JSON.parse(result.content[0].text);
        expect(parsed.libraries).toHaveLength(5);
        expect(parsed.leanMode).toBe(true);
        expect(parsed.libraries[0]).toEqual({id: 'lib0', name: 'Library 0'});
        expect(parsed.libraries[0].versions).toBeUndefined();
        expect(parsed.hint).toBeUndefined();
    });

    it('returns a structured isError response when getLibrariesAsArray throws', async () => {
        const {fakeServer, toolHandlers} = makeFakeServer();
        const apiHandler = {
            getLibrariesAsArray: () => {
                throw new Error('options not loaded yet');
            },
        } as unknown as ApiHandler;
        registerLibrariesTool(fakeServer, apiHandler);

        const result = await toolHandlers.list_libraries({language: 'c++'});
        expect(result.isError).toBe(true);
        expect(result.content[0].text).toMatch(/not available/i);
    });
});

describe('MCP compile tool', () => {
    type ExecResult = {code: number; stdout: string[]; stderr: string[]; didExecute: boolean};

    type BuildResultShape = {code: number; stdout: string[]; stderr: string[]};

    // Most compile tests don't exercise the apiHandler paths (default-compiler resolution
    // or library-version normalisation), so a stub that returns no languages / no
    // libraries is enough. Tests that DO exercise those paths build their own.
    const fakeApiForCompile = {
        getDefaultCompilerFor: () => undefined,
        getLibrariesAsArray: () => [],
    } as unknown as ApiHandler;

    function makeCompileHandler(
        asm: string[],
        stdout: string[] = [],
        stderr: string[] = [],
        execResult?: ExecResult,
        topLevelCode = 0,
        buildResult?: BuildResultShape,
    ): CompileHandler {
        const fakeBaseCompiler = {
            getDefaultFilters: () => ({}),
            compile: vi.fn().mockResolvedValue({
                code: topLevelCode,
                asm: asm.map(text => ({text})),
                stdout: stdout.map(text => ({text})),
                stderr: stderr.map(text => ({text})),
                execResult: execResult && {
                    code: execResult.code,
                    stdout: execResult.stdout.map(text => ({text})),
                    stderr: execResult.stderr.map(text => ({text})),
                    didExecute: execResult.didExecute,
                },
                buildResult: buildResult && {
                    code: buildResult.code,
                    stdout: buildResult.stdout.map(text => ({text})),
                    stderr: buildResult.stderr.map(text => ({text})),
                },
            }),
        };
        return {
            findCompiler: () => fakeBaseCompiler,
        } as unknown as CompileHandler;
    }

    it('returns full output below caps without truncation markers', async () => {
        const {fakeServer, toolHandlers} = makeFakeServer();
        const compileHandler = makeCompileHandler(['mov rax, 1', 'ret'], ['hi'], []);
        registerCompileTool(fakeServer, compileHandler, fakeApiForCompile);

        const result = await toolHandlers.compile({source: 'x', language: 'c++', compiler: 'g142'});
        const parsed = JSON.parse(result.content[0].text);
        expect(parsed.asm).toBe('mov rax, 1\nret');
        expect(parsed.asmTruncated).toBeUndefined();
        expect(parsed.asmTotalLines).toBeUndefined();
        expect(parsed.hint).toBeUndefined();
    });

    it('truncates assembly above maxAsmLines and reports total', async () => {
        const {fakeServer, toolHandlers} = makeFakeServer();
        const lines = Array.from({length: 1200}, (_, i) => `line${i}`);
        const compileHandler = makeCompileHandler(lines);
        registerCompileTool(fakeServer, compileHandler, fakeApiForCompile);

        const result = await toolHandlers.compile({source: 'x', language: 'c++', compiler: 'g142'});
        const parsed = JSON.parse(result.content[0].text);
        expect(parsed.asm.split('\n')).toHaveLength(500);
        expect(parsed.asmTruncated).toBe(true);
        expect(parsed.asmTotalLines).toBe(1200);
        expect(parsed.hint).toMatch(/maxAsmLines/);
    });

    it('honours an explicit maxAsmLines override', async () => {
        const {fakeServer, toolHandlers} = makeFakeServer();
        const lines = Array.from({length: 1200}, (_, i) => `line${i}`);
        const compileHandler = makeCompileHandler(lines);
        registerCompileTool(fakeServer, compileHandler, fakeApiForCompile);

        const result = await toolHandlers.compile({
            source: 'x',
            language: 'c++',
            compiler: 'g142',
            maxAsmLines: 2000,
        });
        const parsed = JSON.parse(result.content[0].text);
        expect(parsed.asm.split('\n')).toHaveLength(1200);
        expect(parsed.asmTruncated).toBeUndefined();
    });

    it('surfaces buildResult.stdout/stderr so execute-mode build failures are not silent', async () => {
        const {fakeServer, toolHandlers} = makeFakeServer();
        // Mirror the shape base-compiler returns when execute=true and the build fails:
        // top-level stderr is the generic "Build failed" wrapper, and the actual diagnostic
        // is on buildResult.
        const compileHandler = makeCompileHandler([], [], ['Build failed'], undefined, -1, {
            code: 1,
            stdout: [],
            stderr: ["error: 'foo' was not declared in this scope"],
        });
        registerCompileTool(fakeServer, compileHandler, fakeApiForCompile);

        const result = await toolHandlers.compile({
            source: 'x',
            language: 'c++',
            compiler: 'g142',
            execute: true,
        });
        const parsed = JSON.parse(result.content[0].text);
        expect(parsed.code).toBe(-1);
        expect(parsed.stderr).toBe('Build failed');
        expect(parsed.buildResult.code).toBe(1);
        expect(parsed.buildResult.stderr).toMatch(/not declared in this scope/);
    });

    it('truncates buildResult streams and sets the hint', async () => {
        const {fakeServer, toolHandlers} = makeFakeServer();
        const longStderr = Array.from({length: 250}, (_, i) => `err${i}`);
        const compileHandler = makeCompileHandler([], [], ['Build failed'], undefined, -1, {
            code: 1,
            stdout: [],
            stderr: longStderr,
        });
        registerCompileTool(fakeServer, compileHandler, fakeApiForCompile);

        const result = await toolHandlers.compile({
            source: 'x',
            language: 'c++',
            compiler: 'g142',
            execute: true,
        });
        const parsed = JSON.parse(result.content[0].text);
        expect(parsed.buildResult.stderrTruncated).toBe(true);
        expect(parsed.buildResult.stderrTotalLines).toBe(250);
        expect(parsed.hint).toMatch(/maxStderrLines/);
    });

    it('sets the hint when only execResult output is truncated', async () => {
        const {fakeServer, toolHandlers} = makeFakeServer();
        const execStdout = Array.from({length: 250}, (_, i) => `out${i}`);
        const compileHandler = makeCompileHandler([], [], [], {
            code: 0,
            stdout: execStdout,
            stderr: [],
            didExecute: true,
        });
        registerCompileTool(fakeServer, compileHandler, fakeApiForCompile);

        const result = await toolHandlers.compile({source: 'x', language: 'c++', compiler: 'g142', execute: true});
        const parsed = JSON.parse(result.content[0].text);
        expect(parsed.execResult.stdoutTruncated).toBe(true);
        expect(parsed.execResult.stdoutTotalLines).toBe(250);
        expect(parsed.hint).toMatch(/maxStdoutLines/);
    });

    it('resolves the language default compiler when none is given', async () => {
        const {fakeServer, toolHandlers} = makeFakeServer();
        const compileHandler = makeCompileHandler(['ret']);
        const findCompilerSpy = vi.spyOn(compileHandler, 'findCompiler');
        const apiHandler = {
            getDefaultCompilerFor: () => 'g161',
            getLibrariesAsArray: () => [],
        } as unknown as ApiHandler;
        registerCompileTool(fakeServer, compileHandler, apiHandler);

        const result = await toolHandlers.compile({source: 'x', language: 'c++'});
        expect(result.isError).toBeUndefined();
        expect(findCompilerSpy).toHaveBeenCalledWith('c++', 'g161');
    });

    it('errors clearly when no compiler is given and no default exists', async () => {
        const {fakeServer, toolHandlers} = makeFakeServer();
        const compileHandler = makeCompileHandler(['ret']);
        const apiHandler = {
            getDefaultCompilerFor: () => undefined,
            getLibrariesAsArray: () => [],
        } as unknown as ApiHandler;
        registerCompileTool(fakeServer, compileHandler, apiHandler);

        const result = await toolHandlers.compile({source: 'x', language: 'wat'});
        expect(result.isError).toBe(true);
        expect(result.content[0].text).toMatch(/No compiler specified.*language "wat"/);
    });

    it('normalises a human library version (1.88.0) to its id (188) and reaches baseCompiler.compile', async () => {
        const {fakeServer, toolHandlers} = makeFakeServer();
        const compileHandler = makeCompileHandler(['ret']);
        // findCompiler returns the same fakeBaseCompiler each call; spy on its compile().
        const baseCompiler = compileHandler.findCompiler('c++' as any, 'g161');
        const compileSpy = vi.spyOn(baseCompiler!, 'compile');
        const apiHandler = {
            getDefaultCompilerFor: () => 'g161',
            getLibrariesAsArray: () => [{id: 'boost', versions: [{id: '188', version: '1.88.0'}]}],
        } as unknown as ApiHandler;
        registerCompileTool(fakeServer, compileHandler, apiHandler);

        const result = await toolHandlers.compile({
            source: 'x',
            language: 'c++',
            libraries: [{id: 'boost', version: '1.88.0'}],
        });
        expect(result.isError).toBeUndefined();
        // The 8th positional arg of baseCompiler.compile(...) is parsed libraries;
        // human "1.88.0" should have been normalised to id "188" before reaching it.
        const passedLibraries = compileSpy.mock.calls[0][7];
        expect(passedLibraries).toEqual([{id: 'boost', version: '188'}]);
    });

    it('errors clearly on an unknown library version (neither id nor human form)', async () => {
        const {fakeServer, toolHandlers} = makeFakeServer();
        const compileHandler = makeCompileHandler(['ret']);
        const apiHandler = {
            getDefaultCompilerFor: () => 'g161',
            getLibrariesAsArray: () => [
                {
                    id: 'boost',
                    versions: [
                        {id: '188', version: '1.88.0'},
                        {id: '189', version: '1.89.0'},
                    ],
                },
            ],
        } as unknown as ApiHandler;
        registerCompileTool(fakeServer, compileHandler, apiHandler);

        const result = await toolHandlers.compile({
            source: 'x',
            language: 'c++',
            libraries: [{id: 'boost', version: 'banana'}],
        });
        expect(result.isError).toBe(true);
        expect(result.content[0].text).toMatch(/Version "banana" not found/);
        expect(result.content[0].text).toMatch(/188 \(1\.88\.0\)/);
    });
});

describe('MCP list_languages tool', () => {
    it('includes a compilerCount per language', async () => {
        const {fakeServer, toolHandlers} = makeFakeServer();
        const apiHandler = {
            getAvailableLanguages: () => [
                {id: 'c++', name: 'C++', extensions: ['.cpp'], defaultCompiler: 'g161'},
                {id: 'rust', name: 'Rust', extensions: ['.rs'], defaultCompiler: 'r1950'},
                {id: 'cobol', name: 'COBOL', extensions: ['.cob'], defaultCompiler: ''},
            ],
            compilers: [
                makeCompiler('g161', 'c++'),
                makeCompiler('g152', 'c++'),
                makeCompiler('r1950', 'rust'),
                // No COBOL compilers — should report 0 not undefined.
            ],
        } as unknown as ApiHandler;
        registerLanguagesTool(fakeServer, apiHandler);

        const result = await toolHandlers.list_languages({});
        const parsed = JSON.parse(result.content[0].text);
        const byId = Object.fromEntries(parsed.map((l: any) => [l.id, l.compilerCount]));
        expect(byId).toEqual({'c++': 2, rust: 1, cobol: 0});
    });
});

describe('MCP lookup_asm_instruction tool', () => {
    it('returns documentation for a known opcode on a known instruction set', async () => {
        const {fakeServer, toolHandlers} = makeFakeServer();
        registerAsmDocsTool(fakeServer);

        // MOV on amd64 is in the Intel docs that ship with CE.
        const result = await toolHandlers.lookup_asm_instruction({instruction_set: 'amd64', opcode: 'MOV'});
        expect(result.isError).toBeUndefined();
        const parsed = JSON.parse(result.content[0].text);
        expect(parsed).toHaveProperty('html');
        expect(parsed).toHaveProperty('tooltip');
    });

    it('lowercase opcode is upper-cased before lookup', async () => {
        const {fakeServer, toolHandlers} = makeFakeServer();
        registerAsmDocsTool(fakeServer);

        const result = await toolHandlers.lookup_asm_instruction({instruction_set: 'amd64', opcode: 'mov'});
        expect(result.isError).toBeUndefined();
    });

    it('returns isError for an unknown opcode on a valid instruction set', async () => {
        const {fakeServer, toolHandlers} = makeFakeServer();
        registerAsmDocsTool(fakeServer);

        const result = await toolHandlers.lookup_asm_instruction({
            instruction_set: 'amd64',
            opcode: 'NOTAREALOPCODE',
        });
        expect(result.isError).toBe(true);
        expect(result.content[0].text).toMatch(/No documentation found/);
    });
});
