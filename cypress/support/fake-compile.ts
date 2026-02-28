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
 * Cypress intercept-based fake compiler for E2E tests.
 *
 * All compilation is handled client-side via cy.intercept — no custom
 * compiler class on the server. The server just needs valid compiler entries
 * in its config (using /bin/true as a no-op binary).
 *
 * API:
 *   setupFakeCompiler()           — call in beforeEach before cy.visit()
 *   stubCompileResponse(data)     — override compile responses (queued)
 *   stubExecutorResponse(data)    — override executor responses (queued)
 *
 * stubCompileResponse/stubExecutorResponse use cy.intercept internally,
 * so they queue properly in the Cypress command chain. Each call replaces
 * the previous intercept handler.
 */

const FAKE_CAPABILITIES = {
    supportsGccDump: true,
    supportsOptOutput: true,
    supportsPpView: true,
    supportsCfg: true,
    supportsExecute: true,
};

function echoSourceAsAsm(source: string, options: string[], filters: Record<string, boolean>) {
    const lines: Array<Record<string, any>> = [];
    const displayOptions = options.filter(o => !o.startsWith('--fake-'));
    if (displayOptions.length > 0) {
        lines.push({text: `; Options: ${displayOptions.join(' ')}`, source: null, labels: []});
    }
    const activeFilters = Object.entries(filters)
        .filter(([, v]) => v)
        .map(([k]) => k);
    if (activeFilters.length > 0) {
        lines.push({text: `; Filters: ${activeFilters.join(' ')}`, source: null, labels: []});
    }
    for (const [i, line] of source.split('\n').entries()) {
        lines.push({
            text: line,
            source: line.trim() ? {file: null, line: i + 1, mainsource: true} : null,
            labels: [],
        });
    }
    return lines;
}

function buildEchoResponse(reqBody: Record<string, any>): Record<string, any> {
    const source = reqBody.source || '';
    const userArgs = reqBody.options?.userArguments || '';
    const options = userArgs ? userArgs.split(/\s+/).filter(Boolean) : [];
    const filters = reqBody.options?.filters || {};
    const backendOptions = reqBody.options?.compilerOptions || {};

    if (backendOptions.executorRequest) {
        return {
            code: 0,
            didExecute: true,
            timedOut: false,
            stdout: [],
            stderr: [],
            inputFilename: 'example.cpp',
            compilationOptions: options,
            downloads: [],
            tools: [],
            asm: [],
            languageId: 'c++',
            execResult: {
                didExecute: true,
                code: 0,
                stdout: [],
                stderr: [],
                timedOut: false,
                buildResult: {
                    code: 0,
                    timedOut: false,
                    stdout: [],
                    stderr: [],
                    inputFilename: 'example.cpp',
                    compilationOptions: [],
                    downloads: [],
                    executableFilename: '',
                    tools: [],
                    asm: [],
                    languageId: 'c++',
                },
            },
        };
    }

    return {
        code: 0,
        timedOut: false,
        okToCache: true,
        stdout: [],
        stderr: [],
        inputFilename: 'example.cpp',
        compilationOptions: options,
        downloads: [],
        tools: [],
        asm: echoSourceAsAsm(source, options, filters),
        languageId: 'c++',
    };
}

function deepMerge(target: Record<string, any>, source: Record<string, any>) {
    for (const key of Object.keys(source)) {
        if (
            source[key] &&
            typeof source[key] === 'object' &&
            !Array.isArray(source[key]) &&
            target[key] &&
            typeof target[key] === 'object' &&
            !Array.isArray(target[key])
        ) {
            deepMerge(target[key], source[key]);
        } else {
            target[key] = source[key];
        }
    }
}

/**
 * Set up the fake compiler intercepts. Call BEFORE cy.visit().
 *
 * Default: echoes source as asm, with options and filters shown as comments.
 */
export function setupFakeCompiler() {
    // Patch compiler capabilities in the client options
    cy.intercept('GET', '**/client-options.js*', req => {
        req.continue(res => {
            const body = String(res.body);
            try {
                const jsonStart = body.indexOf('{');
                const jsonEnd = body.lastIndexOf('}');
                if (jsonStart >= 0 && jsonEnd > jsonStart) {
                    const opts = JSON.parse(body.substring(jsonStart, jsonEnd + 1));
                    if (opts.compilers) {
                        for (const compiler of opts.compilers) {
                            Object.assign(compiler, FAKE_CAPABILITIES);
                        }
                    }
                    res.body = `window.compilerExplorerOptions = ${JSON.stringify(opts)};`;
                }
            } catch {
                // If parsing fails, let it through unmodified
            }
        });
    }).as('clientOptions');

    // Default compile handler — echo source as asm
    cy.intercept('POST', '/api/compiler/*/compile', req => {
        req.reply(buildEchoResponse(req.body));
    }).as('compile');
}

/**
 * Override compile responses. This is a Cypress command (uses cy.intercept)
 * so it queues properly in the command chain. Overrides are merged with the
 * default echo response. Executor requests still use the default echo.
 */
export function stubCompileResponse(overrides: Record<string, any>) {
    cy.intercept('POST', '/api/compiler/*/compile', req => {
        const backendOptions = req.body.options?.compilerOptions || {};
        if (backendOptions.executorRequest) {
            req.reply(buildEchoResponse(req.body));
        } else {
            const base = buildEchoResponse(req.body);
            deepMerge(base, overrides);
            req.reply(base);
        }
    }).as('compile');
}

/**
 * Override executor responses. Overrides are merged into execResult.
 * Normal compile requests still use the default echo.
 */
export function stubExecutorResponse(overrides: Record<string, any>) {
    cy.intercept('POST', '/api/compiler/*/compile', req => {
        const backendOptions = req.body.options?.compilerOptions || {};
        if (backendOptions.executorRequest) {
            const base = buildEchoResponse(req.body);
            deepMerge(base.execResult, overrides);
            req.reply(base);
        } else {
            req.reply(buildEchoResponse(req.body));
        }
    }).as('compile');
}
