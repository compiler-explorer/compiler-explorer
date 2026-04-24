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

import {McpServer} from '@modelcontextprotocol/sdk/server/mcp.js';
import {StreamableHTTPServerTransport} from '@modelcontextprotocol/sdk/server/streamableHttp.js';
import express, {type Router} from 'express';

import type {ApiHandler} from '../handlers/api.js';
import type {CompileHandler} from '../handlers/compile.js';
import {logger} from '../logger.js';
import type {StorageBase} from '../storage/index.js';
import {registerAsmDocsTool} from './tools/asm-docs.js';
import {registerCompileTool} from './tools/compile.js';
import {registerCompilersTool} from './tools/compilers.js';
import {registerLanguagesTool} from './tools/languages.js';
import {registerLibrariesTool} from './tools/libraries.js';
import {registerShortlinkTools} from './tools/shortlinks.js';

function createMcpServer(
    compileHandler: CompileHandler,
    apiHandler: ApiHandler,
    storageHandler: StorageBase,
    baseUrl: string,
): McpServer {
    const server = new McpServer(
        {
            name: 'compiler-explorer',
            version: apiHandler.release.releaseBuildNumber || apiHandler.release.gitReleaseName || 'dev',
        },
        {
            capabilities: {
                tools: {},
            },
        },
    );

    registerCompileTool(server, compileHandler);
    registerCompilersTool(server, apiHandler);
    registerLanguagesTool(server, apiHandler);
    registerLibrariesTool(server, apiHandler);
    registerAsmDocsTool(server);
    registerShortlinkTools(server, storageHandler, baseUrl);

    return server;
}

export function setupMcpEndpoint(
    router: Router,
    compileHandler: CompileHandler,
    apiHandler: ApiHandler,
    storageHandler: StorageBase,
): void {
    router.post('/mcp', express.json(), async (req, res) => {
        try {
            const baseUrl = `${req.protocol}://${req.get('host')}`;
            const server = createMcpServer(compileHandler, apiHandler, storageHandler, baseUrl);
            const transport = new StreamableHTTPServerTransport({
                sessionIdGenerator: undefined,
            });
            await server.connect(transport);
            await transport.handleRequest(req, res, req.body);
        } catch (e) {
            logger.error('MCP request error:', e);
            if (!res.headersSent) {
                res.status(500).json({
                    jsonrpc: '2.0',
                    error: {code: -32603, message: 'Internal server error'},
                    id: null,
                });
            }
        }
    });

    router.get('/mcp', (_req, res) => {
        res.status(405).set('Allow', 'POST').send('Method Not Allowed');
    });

    router.delete('/mcp', (_req, res) => {
        res.status(405).set('Allow', 'POST').send('Method Not Allowed');
    });

    logger.info('MCP endpoint registered at /mcp');
}
