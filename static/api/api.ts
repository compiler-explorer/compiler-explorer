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
    AssemblyDocumentationResponse,
    AssemblyDocumentationRequest,
} from '../../types/features/assembly-documentation.interfaces';
import {FormattingRequest, FormattingResponse} from './formatting.interfaces';

/** Type wrapper allowing .json() to resolve to a concrete type */
interface TypedResponse<T> extends Response {
    json(): Promise<T>;
}

/** Lightweight fetch() wrapper for CE API urls */
const request = async <R>(uri: string, options?: RequestInit): Promise<TypedResponse<R>> =>
    fetch(`${window.location.origin}${window.httpRoot}api${uri}`, {
        ...options,
        credentials: 'include',
        headers: {
            ...options?.headers,
            Accept: 'application/json',
        },
    });

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

interface CompilerRepositorySearchRequest {
    compilerId: string;
    query: string;
    limit?: number;
    offset?: number;
}

interface CompilerRepositorySearchResponse {
    results: string[];
    total: number;
    limit: number;
    offset: number;
}

export async function searchCompilerRepository(
    options: CompilerRepositorySearchRequest
): Promise<TypedResponse<CompilerRepositorySearchResponse>> {
    const searchParams = new URLSearchParams({q: options.query});
    if (typeof options.limit === 'number') {
        searchParams.set('limit', options.limit.toString());
    }
    if (typeof options.offset === 'number') {
        searchParams.set('offset', options.offset.toString());
    }
    return await request<CompilerRepositorySearchResponse>(
        `/compilerRepository/${options.compilerId}/search?${searchParams}`
    );
}
