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

import type {LanguageKey} from '../../../types/languages.interfaces.js';
import {ClientStateNormalizer} from '../../clientstate-normalizer.js';
import type {ApiHandler} from '../../handlers/api.js';
import {logger} from '../../logger.js';
import type {StorageBase} from '../../storage/base.js';
import {getSafeHash} from '../../storage/base.js';
import {extractShortId} from '../../url-utils.js';
import {normaliseRequestLibraries} from '../library-utils.js';

// CE shortlinks save the canonical "config" shape used by the web app, where each
// compiler entry uses {id, options, libs:[{id, ver}]}. The MCP `compile` tool accepts
// {compiler, options, libraries:[{id, version}]}. Translate between the two so a
// shortlink round-trip is friction-free for an LLM caller.
type CompileReadyCompiler = {
    compiler: string;
    options: string;
    libraries: Array<{id: string; version: string}>;
};
type SavedCompiler = {id?: string; options?: string; libs?: Array<{id?: string; ver?: string}>};
type SavedSession = {language?: string; source?: string; compilers?: SavedCompiler[]};

function toCompileReady(savedCompiler: SavedCompiler): CompileReadyCompiler {
    return {
        compiler: savedCompiler.id ?? '',
        options: savedCompiler.options ?? '',
        libraries: (savedCompiler.libs ?? [])
            .filter(l => l.id !== undefined && l.ver !== undefined)
            .map(l => ({id: l.id as string, version: l.ver as string})),
    };
}

export function registerShortlinkTools(
    server: McpServer,
    storageHandler: StorageBase,
    apiHandler: ApiHandler,
    baseUrl: string,
    req: express.Request,
): void {
    server.tool(
        'generate_short_url',
        'Create a Compiler Explorer short URL for sharing code with compiler settings',
        {
            source: z.string().describe('Source code'),
            language: z.string().describe('Language ID (e.g. "c++", "c", "rust")'),
            compiler: z.string().describe('Compiler ID from list_compilers (e.g. "g161", "clang_trunk")'),
            options: z.string().optional().describe('Compiler flags (e.g. "-O2 -std=c++20")'),
            libraries: z
                .array(
                    z.object({
                        id: z.string().describe('Library ID from list_libraries.'),
                        version: z.string().describe('Version id ("188") OR human form ("1.88.0").'),
                    }),
                )
                .optional()
                .describe('Libraries to include.'),
        },
        {
            // Persists a shortlink record to S3, hence not readOnly. It's additive only
            // (never deletes / overwrites prior shortlinks) and dedupes by config hash,
            // so repeat calls with the same input return the same URL — idempotent.
            title: 'Generate Compiler Explorer short URL',
            readOnlyHint: false,
            destructiveHint: false,
            idempotentHint: true,
            openWorldHint: false,
        },
        async ({source, language, compiler, options, libraries}) => {
            try {
                // Normalise library versions before saving so the resulting shortlink
                // always carries the canonical version id regardless of input form.
                let storedLibs: Array<{id: string; ver: string}> = [];
                if (libraries && libraries.length > 0) {
                    let knownLibraries: ReturnType<ApiHandler['getLibrariesAsArray']>;
                    try {
                        knownLibraries = apiHandler.getLibrariesAsArray(language as LanguageKey);
                    } catch {
                        knownLibraries = [];
                    }
                    const result = normaliseRequestLibraries(knownLibraries, language, libraries);
                    if (!result.ok) {
                        return {content: [{type: 'text', text: result.errorText}], isError: true};
                    }
                    storedLibs = result.libraries.map(l => ({id: l.id, ver: l.version}));
                }
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
                                    libs: storedLibs,
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
                return {content: [{type: 'text', text: JSON.stringify({url}, null, 2)}]};
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
        'Retrieve source and compiler config from a CE short URL. Compiler entries are returned in the ' +
            "`compile` tool's shape ({compiler, options, libraries:[{id, version}]}). Multi-pane shortlinks " +
            '(executors, conformance views, CMake trees, per-compiler filters/tools) are flattened — only ' +
            'the basic compile inputs survive.',
        {
            id: z
                .string()
                .describe('Short link ID or full URL (e.g. "G38YP7eW4" or "https://godbolt.org/z/G38YP7eW4")'),
        },
        {
            title: 'Get Compiler Explorer short URL info',
            readOnlyHint: true,
            openWorldHint: false,
        },
        async ({id}) => {
            // Extract the short id whether a full URL or a bare id was provided.
            const shortId = extractShortId(id);

            try {
                const result = await storageHandler.expandId(shortId);
                const config = JSON.parse(result.config);

                // Normalise old golden-layout format to modern sessions format.
                let sessions: SavedSession[];
                if (config.content) {
                    const normalizer = new ClientStateNormalizer();
                    normalizer.fromGoldenLayout(config);
                    sessions = (normalizer.normalized as {sessions?: SavedSession[]}).sessions ?? [];
                } else {
                    sessions = (config as {sessions?: SavedSession[]}).sessions ?? [];
                }

                // Translate the saved-config compiler shape ({id, options, libs:[{id, ver}]})
                // into the same shape `compile` accepts ({compiler, options, libraries:[{id, version}]}).
                const friendly = {
                    sessions: sessions.map(s => ({
                        language: s.language,
                        source: s.source,
                        compilers: (s.compilers ?? []).map(toCompileReady),
                    })),
                };
                return {content: [{type: 'text', text: JSON.stringify(friendly, null, 2)}]};
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
