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
import {registerCompileTool} from '../../lib/mcp/tools/compile.js';
import {registerCompilersTool} from '../../lib/mcp/tools/compilers.js';
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

function makeFakeServer(): {fakeServer: any; toolHandlers: Record<string, (args: any) => Promise<any>>} {
    const toolHandlers: Record<string, (args: any) => Promise<any>> = {};
    const fakeServer = {
        tool: (name: string, _description: string, _schema: unknown, handler: (args: any) => Promise<any>) => {
            toolHandlers[name] = handler;
        },
    };
    return {fakeServer, toolHandlers};
}

function makeCompiler(id: string, lang = 'c++'): CompilerInfo {
    return {
        id,
        name: id.toUpperCase(),
        lang,
        compilerType: '',
        semver: '',
        instructionSet: 'amd64',
    } as unknown as CompilerInfo;
}

describe('MCP list_compilers tool', () => {
    it('returns all compilers under the cap with total count', async () => {
        const {fakeServer, toolHandlers} = makeFakeServer();
        const apiHandler = {compilers: [makeCompiler('g142'), makeCompiler('clang20')]} as unknown as ApiHandler;
        registerCompilersTool(fakeServer, apiHandler);

        const result = await toolHandlers.list_compilers({});
        const parsed = JSON.parse(result.content[0].text);
        expect(parsed.total).toBe(2);
        expect(parsed.truncated).toBeUndefined();
        expect(parsed.compilers.map((c: any) => c.id)).toEqual(['g142', 'clang20']);
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
            {id: 'clang20', name: 'CLANG20', lang: 'c++', compilerType: '', semver: '', instructionSet: 'amd64'},
        ]);
    });

    it('truncates results above the cap and includes a hint', async () => {
        const {fakeServer, toolHandlers} = makeFakeServer();
        const apiHandler = {
            compilers: Array.from({length: 250}, (_, i) => makeCompiler(`g${i}`)),
        } as unknown as ApiHandler;
        registerCompilersTool(fakeServer, apiHandler);

        const result = await toolHandlers.list_compilers({});
        const parsed = JSON.parse(result.content[0].text);
        expect(parsed.compilers).toHaveLength(100);
        expect(parsed.total).toBe(250);
        expect(parsed.truncated).toBe(true);
        expect(parsed.hint).toMatch(/match/);
    });

    it('honours an explicit maxResults override', async () => {
        const {fakeServer, toolHandlers} = makeFakeServer();
        const apiHandler = {
            compilers: Array.from({length: 250}, (_, i) => makeCompiler(`g${i}`)),
        } as unknown as ApiHandler;
        registerCompilersTool(fakeServer, apiHandler);

        const result = await toolHandlers.list_compilers({maxResults: 300});
        const parsed = JSON.parse(result.content[0].text);
        expect(parsed.compilers).toHaveLength(250);
        expect(parsed.truncated).toBeUndefined();
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

    it('truncates above the cap', async () => {
        const {fakeServer, toolHandlers} = makeFakeServer();
        const libs = Array.from({length: 150}, (_, i) => ({id: `lib${i}`, name: `Library ${i}`}));
        const apiHandler = makeApiHandlerWithLibs(libs);
        registerLibrariesTool(fakeServer, apiHandler);

        const result = await toolHandlers.list_libraries({language: 'c++'});
        const parsed = JSON.parse(result.content[0].text);
        expect(parsed.libraries).toHaveLength(100);
        expect(parsed.total).toBe(150);
        expect(parsed.truncated).toBe(true);
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

    function makeCompileHandler(
        asm: string[],
        stdout: string[] = [],
        stderr: string[] = [],
        execResult?: ExecResult,
    ): CompileHandler {
        const fakeBaseCompiler = {
            getDefaultFilters: () => ({}),
            compile: vi.fn().mockResolvedValue({
                code: 0,
                asm: asm.map(text => ({text})),
                stdout: stdout.map(text => ({text})),
                stderr: stderr.map(text => ({text})),
                execResult: execResult && {
                    code: execResult.code,
                    stdout: execResult.stdout.map(text => ({text})),
                    stderr: execResult.stderr.map(text => ({text})),
                    didExecute: execResult.didExecute,
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
        registerCompileTool(fakeServer, compileHandler);

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
        registerCompileTool(fakeServer, compileHandler);

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
        registerCompileTool(fakeServer, compileHandler);

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

    it('sets the hint when only execResult output is truncated', async () => {
        const {fakeServer, toolHandlers} = makeFakeServer();
        const execStdout = Array.from({length: 250}, (_, i) => `out${i}`);
        const compileHandler = makeCompileHandler([], [], [], {
            code: 0,
            stdout: execStdout,
            stderr: [],
            didExecute: true,
        });
        registerCompileTool(fakeServer, compileHandler);

        const result = await toolHandlers.compile({source: 'x', language: 'c++', compiler: 'g142', execute: true});
        const parsed = JSON.parse(result.content[0].text);
        expect(parsed.execResult.stdoutTruncated).toBe(true);
        expect(parsed.execResult.stdoutTotalLines).toBe(250);
        expect(parsed.hint).toMatch(/maxStdoutLines/);
    });
});
