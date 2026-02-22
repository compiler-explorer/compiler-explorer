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
 * implementation (HttpBackendApi) makes the actual fetch/$.ajax calls.
 * Tests supply a different implementation to avoid any network traffic.
 *
 * Initialisation: Hub calls setBackendApi(new HttpBackendApi()) at startup.
 * All other modules call getBackendApi() to obtain the current instance.
 */

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
    getAssemblyDoc(req: AssemblyDocumentationRequest): Promise<AssemblyDocumentationResponse>;

    /** POST /api/shortener */
    shortenUrl(body: Record<string, unknown>): Promise<{url: string}>;
}

export class HttpBackendApi implements BackendApi {
    private readonly baseUrl: string;

    constructor() {
        this.baseUrl = window.location.origin + window.httpRoot;
    }

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
        if (!response.ok) throw new Error(`Formatting request failed: HTTP ${response.status}`);
        return response.json() as Promise<FormattingResponse>;
    }

    async getAssemblyDoc(req: AssemblyDocumentationRequest): Promise<AssemblyDocumentationResponse> {
        const response = await fetch(`${this.baseUrl}api/asm/${req.instructionSet}/${req.opcode}`, {
            credentials: 'omit',
            headers: {Accept: 'application/json'},
        });
        const body = (await response.json()) as AssemblyDocumentationResponse & {error?: string};
        if (!response.ok) throw new Error(body.error ?? `HTTP ${response.status}`);
        return body;
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
                error: err => reject(new Error(err.statusText || 'URL shortening failed')),
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

let _api: BackendApi | undefined;

/**
 * Register the backend API implementation. Called once by Hub at startup
 * before any panes are created.
 */
export function setBackendApi(api: BackendApi): void {
    _api = api;
}

/**
 * Obtain the current backend API. Available after Hub calls setBackendApi().
 */
export function getBackendApi(): BackendApi {
    if (!_api) throw new Error('BackendApi not initialised — call setBackendApi() first');
    return _api;
}
