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
import {z} from 'zod';

import type {ApiHandler} from '../../handlers/api.js';
import {applyCap, applyMatch} from '../utils.js';

const DEFAULT_MAX_RESULTS = 25;

export function registerCompilersTool(server: McpServer, apiHandler: ApiHandler): void {
    server.tool(
        'list_compilers',
        'List available compilers, optionally filtered by language',
        {
            language: z.string().optional().describe('Language ID to filter by (e.g. "c++", "rust", "python")'),
            match: z
                .string()
                .optional()
                .describe(
                    'Case-insensitive substring filter on compiler id and name. Be specific: broad terms ' +
                        '(e.g. "gcc", "clang") match hundreds of compilers and trigger lean mode (id+name only). ' +
                        'Prefer "gcc 13", "clang 17", "arm64 gcc", or an exact id like "g132".',
                ),
            maxResults: z
                .number()
                .int()
                .positive()
                .optional()
                .describe(
                    `Maximum entries to return in the full response shape (default ${DEFAULT_MAX_RESULTS}). ` +
                        'When the filtered count exceeds this, the response degrades to lean mode (id+name only) ' +
                        'and returns ALL matches; raise this if you need full per-entry detail across many results.',
                ),
        },
        async ({language, match, maxResults}) => {
            const filtered = applyMatch(
                language ? apiHandler.compilers.filter(c => c.lang === language) : apiHandler.compilers,
                match,
                c => [c.id, c.name],
            );
            const {items, ...meta} = applyCap(
                filtered,
                maxResults ?? DEFAULT_MAX_RESULTS,
                c => ({
                    id: c.id,
                    name: c.name,
                    lang: c.lang,
                    compilerType: c.compilerType,
                    semver: c.semver,
                    instructionSet: c.instructionSet,
                }),
                'compilers',
            );
            return {content: [{type: 'text', text: JSON.stringify({compilers: items, ...meta}, null, 2)}]};
        },
    );
}
