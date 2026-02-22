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

/**
 * BackendApi — a single seam between the CE frontend and its HTTP backend.
 *
 * All CE-specific backend requests go through this interface. The default
 * implementation (HttpBackendApi) is module-private and uses native fetch.
 *
 * All modules call getBackendApi() to obtain the current instance. No
 * explicit initialisation is required for normal operation. A future PR will
 * add an override mechanism so tests can substitute a fake implementation.
 */

import _ from 'underscore';

import {CompilationResult} from '../../types/compilation/compilation.interfaces.js';
import {PossibleArguments} from '../../types/compiler-arguments.interfaces.js';
import {
    AssemblyDocumentationRequest,
    AssemblyDocumentationResponse,
} from '../../types/features/assembly-documentation.interfaces.js';
import {FormattingRequest, FormattingResponse} from './formatting.interfaces.js';

export interface BackendApi {
    /** POST /api/compiler/:id/compile */
    compile(compilerId: string, request: Record<string, unknown>): Promise<CompilationResult>;

    /** POST /api/compiler/:id/cmake */
    compileCMake(compilerId: string, request: Record<string, unknown>): Promise<CompilationResult>;

    /** POST /api/popularArguments/:id */
    popularArguments(compilerId: string, usedOptions: string): Promise<PossibleArguments>;

    /** POST /api/format/:formatterId */
    formatCode(req: FormattingRequest): Promise<FormattingResponse>;

    /** GET /api/asm/:instructionSet/:opcode */
    getAssemblyDocumentation(req: AssemblyDocumentationRequest): Promise<AssemblyDocumentationResponse>;

    /** POST /api/shortener */
    shortenUrl(body: Record<string, unknown>): Promise<{url: string}>;
}

class HttpBackendApi implements BackendApi {
    constructor(private readonly baseUrl: string) {}

    private async postJson<T>(url: string, body: unknown): Promise<T> {
        const response = await fetch(url, {
            method: 'POST',
            headers: {Accept: 'application/json', 'Content-Type': 'application/json'},
            body: JSON.stringify(body),
        });
        if (!response.ok) throw new Error(`Request failed: ${response.status} ${response.statusText}`);
        return response.json() as Promise<T>;
    }

    async compile(compilerId: string, request: Record<string, unknown>): Promise<CompilationResult> {
        return this.postJson(`${this.baseUrl}api/compiler/${encodeURIComponent(compilerId)}/compile`, request);
    }

    async compileCMake(compilerId: string, request: Record<string, unknown>): Promise<CompilationResult> {
        return this.postJson(`${this.baseUrl}api/compiler/${encodeURIComponent(compilerId)}/cmake`, request);
    }

    async popularArguments(compilerId: string, usedOptions: string): Promise<PossibleArguments> {
        return this.postJson(`${this.baseUrl}api/popularArguments/${compilerId}`, {usedOptions, presplit: false});
    }

    async formatCode(req: FormattingRequest): Promise<FormattingResponse> {
        return this.postJson(
            `${this.baseUrl}api/format/${req.formatterId}`,
            _.pick(req, 'source', 'base', 'tabWidth', 'useSpaces'),
        );
    }

    async getAssemblyDocumentation(req: AssemblyDocumentationRequest): Promise<AssemblyDocumentationResponse> {
        const response = await fetch(`${this.baseUrl}api/asm/${req.instructionSet}/${req.opcode}`, {
            credentials: 'omit',
            headers: {Accept: 'application/json'},
        });
        if (!response.ok) {
            let errMessage = `HTTP ${response.status}`;
            try {
                const errBody = (await response.json()) as {error?: string};
                if (errBody.error) errMessage = errBody.error;
            } catch {
                // Non-JSON error body; use the HTTP status message above.
            }
            throw new Error(errMessage);
        }
        return response.json() as Promise<AssemblyDocumentationResponse>;
    }

    async shortenUrl(body: Record<string, unknown>): Promise<{url: string}> {
        return this.postJson(`${this.baseUrl}api/shortener`, body);
    }
}

const _api: BackendApi = new HttpBackendApi(window.location.origin + window.httpRoot);

/** Obtain the backend API instance. */
export function getBackendApi(): BackendApi {
    return _api;
}
