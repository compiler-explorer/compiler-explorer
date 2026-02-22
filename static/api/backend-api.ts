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
 * BackendApi — a single injectable seam between the CE frontend and its
 * HTTP backend.
 *
 * All CE-specific backend requests go through this interface. The real
 * implementation (HttpBackendApi) is used by default and is module-private.
 *
 * All modules call getBackendApi() to obtain the current instance. No
 * explicit initialisation is required for normal operation.
 */

import $ from 'jquery';
import _ from 'underscore';

import {CompilationResult} from '../../types/compilation/compilation.interfaces.js';
import {PossibleArguments} from '../../types/compiler-arguments.interfaces.js';
import {
    AssemblyDocumentationRequest,
    AssemblyDocumentationResponse,
} from '../../types/features/assembly-documentation.interfaces.js';
import {FormattingRequest, FormattingResponse} from './formatting.interfaces.js';

import jqXHR = JQuery.jqXHR;
import ErrorTextStatus = JQuery.Ajax.ErrorTextStatus;

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

    async compile(compilerId: string, request: Record<string, unknown>): Promise<CompilationResult> {
        return new Promise((resolve, reject) => {
            $.ajax({
                type: 'POST',
                url: `${this.baseUrl}api/compiler/${encodeURIComponent(compilerId)}/compile`,
                dataType: 'json',
                contentType: 'application/json',
                data: JSON.stringify(request),
                success: resolve,
                error: (jqXhr, textStatus, errorThrown) => {
                    reject(HttpBackendApi.makeError(request, jqXhr, textStatus, errorThrown));
                },
            });
        });
    }

    async compileCMake(compilerId: string, request: Record<string, unknown>): Promise<CompilationResult> {
        return new Promise((resolve, reject) => {
            $.ajax({
                type: 'POST',
                url: `${this.baseUrl}api/compiler/${encodeURIComponent(compilerId)}/cmake`,
                dataType: 'json',
                contentType: 'application/json',
                data: JSON.stringify(request),
                success: resolve,
                error: (jqXhr, textStatus, errorThrown) => {
                    reject(HttpBackendApi.makeError(request, jqXhr, textStatus, errorThrown));
                },
            });
        });
    }

    async popularArguments(compilerId: string, usedOptions: string): Promise<PossibleArguments> {
        return new Promise((resolve, reject) => {
            $.ajax({
                type: 'POST',
                url: `${this.baseUrl}api/popularArguments/${compilerId}`,
                dataType: 'json',
                contentType: 'application/json',
                data: JSON.stringify({usedOptions, presplit: false}),
                success: resolve,
                error: (jqXhr, textStatus, errorThrown) => {
                    reject(HttpBackendApi.makeError(compilerId, jqXhr, textStatus, errorThrown));
                },
            });
        });
    }

    async formatCode(req: FormattingRequest): Promise<FormattingResponse> {
        const response = await fetch(`${this.baseUrl}api/format/${req.formatterId}`, {
            method: 'POST',
            credentials: 'include',
            headers: {
                Accept: 'application/json',
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(_.pick(req, 'source', 'base', 'tabWidth', 'useSpaces')),
        });
        if (!response.ok) throw new Error(`Formatting request failed: ${response.status} ${response.statusText}`);
        return response.json() as Promise<FormattingResponse>;
    }

    async getAssemblyDocumentation(req: AssemblyDocumentationRequest): Promise<AssemblyDocumentationResponse> {
        const response = await fetch(`${this.baseUrl}api/asm/${req.instructionSet}/${req.opcode}`, {
            credentials: 'omit',
            headers: {Accept: 'application/json'},
        });
        if (!response.ok) {
            // Try to extract a structured error message; fall back to HTTP status if the
            // response body is not JSON (e.g. a proxy returning an HTML error page).
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
        return new Promise((resolve, reject) => {
            $.ajax({
                type: 'POST',
                url: `${this.baseUrl}api/shortener`,
                dataType: 'json',
                contentType: 'application/json',
                data: JSON.stringify(body),
                success: resolve,
                error: (jqXhr, textStatus, errorThrown) => {
                    reject(HttpBackendApi.makeError(body, jqXhr, textStatus, errorThrown));
                },
                cache: true,
            });
        });
    }

    private static makeError(request: unknown, xhr: jqXHR, textStatus: ErrorTextStatus, errorThrown: string): Error {
        let message = errorThrown;
        if (!message) {
            switch (textStatus) {
                case 'timeout':
                    message = 'Request timed out';
                    break;
                case 'abort':
                    message = 'Request was aborted';
                    break;
                case 'error':
                    switch (xhr.status) {
                        case 500:
                            message = 'Request failed: internal server error';
                            break;
                        case 504:
                            message = 'Request failed: gateway timeout';
                            break;
                        default:
                            message = 'Request failed: HTTP error code ' + xhr.status;
                            break;
                    }
                    break;
                default:
                    message = 'Error sending request';
                    break;
            }
        }
        const err = new Error(message);
        (err as any).request = request;
        return err;
    }
}

const _api: BackendApi = new HttpBackendApi(window.location.origin + window.httpRoot);

/** Obtain the backend API instance. */
export function getBackendApi(): BackendApi {
    return _api;
}
