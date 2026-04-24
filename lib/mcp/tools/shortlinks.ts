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

import type {McpServer} from '@modelcontextprotocol/sdk/server/mcp.js';
import type express from 'express';
import {z} from 'zod';

import {ClientStateNormalizer} from '../../clientstate-normalizer.js';
import {logger} from '../../logger.js';
import type {StorageBase} from '../../storage/base.js';
import {getSafeHash} from '../../storage/base.js';

export function registerShortlinkTools(
    server: McpServer,
    storageHandler: StorageBase,
    baseUrl: string,
    req: express.Request,
): void {
    server.tool(
        'generate_short_url',
        'Create a Compiler Explorer short URL for sharing code with compiler settings',
        {
            source: z.string().describe('Source code'),
            language: z.string().describe('Language ID (e.g. "c++", "c", "rust")'),
            compiler: z.string().describe('Compiler ID (e.g. "g142", "clang_trunk")'),
            options: z.string().optional().describe('Compiler flags (e.g. "-O2 -std=c++20")'),
            libraries: z
                .array(
                    z.object({
                        id: z.string(),
                        version: z.string(),
                    }),
                )
                .optional()
                .describe('Libraries to include'),
        },
        async ({source, language, compiler, options, libraries}) => {
            try {
                const config = {
                    sessions: [
                        {
                            id: 1,
                            language,
                            source,
                            compilers: [
                                {
                                    id: compiler,
                                    options: options || '',
                                    libs: libraries || [],
                                },
                            ],
                        },
                    ],
                };

                const {config: configStr, configHash} = getSafeHash(config);
                const result = await storageHandler.findUniqueSubhash(configHash);
                if (!result.alreadyPresent) {
                    await storageHandler.storeItem(
                        {
                            prefix: result.prefix,
                            uniqueSubHash: result.uniqueSubHash,
                            fullHash: configHash,
                            config: configStr,
                        },
                        req,
                    );
                }
                const url = `${baseUrl}${storageHandler.httpRootDir}z/${result.uniqueSubHash}`;
                return {content: [{type: 'text', text: url}]};
            } catch (e) {
                return {
                    content: [{type: 'text', text: `Failed to create short URL: ${(e as Error).message}`}],
                    isError: true,
                };
            }
        },
    );

    server.tool(
        'get_shortlink_info',
        'Retrieve source code and compiler configuration from a Compiler Explorer short URL',
        {
            id: z
                .string()
                .describe('Short link ID or full URL (e.g. "G38YP7eW4" or "https://godbolt.org/z/G38YP7eW4")'),
        },
        async ({id}) => {
            // Extract ID from URL if a full URL was provided
            const match = id.match(/\/z\/([^/]+)$/);
            const shortId = match ? match[1] : id;

            try {
                const result = await storageHandler.expandId(shortId);
                const config = JSON.parse(result.config);

                // Normalise old golden-layout format to modern sessions format
                if (config.content) {
                    const normalizer = new ClientStateNormalizer();
                    normalizer.fromGoldenLayout(config);
                    return {content: [{type: 'text', text: JSON.stringify(normalizer.normalized, null, 2)}]};
                }

                return {content: [{type: 'text', text: JSON.stringify(config, null, 2)}]};
            } catch (e) {
                logger.warn(`MCP shortlink expand failed for ${shortId}:`, e);
                return {
                    content: [{type: 'text', text: `Short link "${shortId}" not found`}],
                    isError: true,
                };
            }
        },
    );
}
