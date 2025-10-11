// Copyright (c) 2021, Compiler Explorer Authors
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

import _ from 'underscore';

import {
    AssemblyDocumentationRequest,
    AssemblyDocumentationResponse,
} from '../../types/features/assembly-documentation.interfaces.js';
import {FetchResult, safeFetch} from '../http-utils.js';
import {FormattingRequest, FormattingResponse} from './formatting.interfaces.js';

/** CE API request helper using safeFetch for better error handling */
const request = async <R>(uri: string, options?: RequestInit): Promise<FetchResult<R>> =>
    safeFetch<R>(
        `${window.location.origin}${window.httpRoot}api${uri}`,
        {
            ...options,
            parseAs: 'json',
            credentials: 'include',
            headers: {
                ...options?.headers,
            },
        },
        `CE API ${uri}`,
    );

/** GET /api/asm/:arch/:instruction */
export const getAssemblyDocumentation = async (options: AssemblyDocumentationRequest) =>
    await request<AssemblyDocumentationResponse>(`/asm/${options.instructionSet}/${options.opcode}`);

/** POST /api/format/:formatter */
export const getFormattedCode = async (options: FormattingRequest) =>
    await request<FormattingResponse>(`/format/${options.formatterId}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(_.pick(options, 'source', 'base', 'tabWidth', 'useSpaces')),
    });
