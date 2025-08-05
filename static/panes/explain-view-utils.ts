// Copyright (c) 2025, Compiler Explorer Authors
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

import {marked} from 'marked';
import {capitaliseFirst} from '../../shared/common-utils.js';
import {CompilationResult} from '../../types/compilation/compilation.interfaces.js';
import {CompilerInfo} from '../../types/compiler.interfaces.js';
import {AvailableOptions, ClaudeExplainResponse, ExplainRequest} from './explain-view.interfaces.js';

// Anything to do with the explain view that doesn't need any direct UI access, so we can test it easily.
// Includes validation, request building, caching, formatting, and other pure functions.

export interface ExplainContext {
    lastResult: CompilationResult | null;
    compiler: CompilerInfo | null;
    selectedAudience: string;
    selectedExplanation: string;
    explainApiEndpoint: string;
    consentGiven: boolean;
    availableOptions: AvailableOptions | null;
}

export enum ValidationErrorCode {
    MISSING_REQUIRED_DATA = 'MISSING_REQUIRED_DATA',
    OPTIONS_NOT_AVAILABLE = 'OPTIONS_NOT_AVAILABLE',
    API_ENDPOINT_NOT_CONFIGURED = 'API_ENDPOINT_NOT_CONFIGURED',
    NO_AI_DIRECTIVE_FOUND = 'NO_AI_DIRECTIVE_FOUND',
}

export type ValidationResult = {success: true} | {success: false; errorCode: ValidationErrorCode; message: string};

/**
 * Validates that all preconditions are met for fetching an explanation.
 * Returns a result object indicating success or failure with error message.
 */
export function validateExplainPreconditions(context: ExplainContext): ValidationResult {
    if (!context.lastResult || !context.consentGiven || !context.compiler) {
        return {
            success: false,
            errorCode: ValidationErrorCode.MISSING_REQUIRED_DATA,
            message: 'Missing required data: compilation result, consent, or compiler info',
        };
    }

    if (context.availableOptions === null) {
        return {
            success: false,
            errorCode: ValidationErrorCode.OPTIONS_NOT_AVAILABLE,
            message: 'Explain options not available',
        };
    }

    if (!context.explainApiEndpoint) {
        return {
            success: false,
            errorCode: ValidationErrorCode.API_ENDPOINT_NOT_CONFIGURED,
            message: 'Claude Explain API endpoint not configured',
        };
    }

    if (context.lastResult.source && checkForNoAiDirective(context.lastResult.source)) {
        return {
            success: false,
            errorCode: ValidationErrorCode.NO_AI_DIRECTIVE_FOUND,
            message: 'no-ai directive found in source code',
        };
    }

    return {success: true};
}

/**
 * Builds the request payload for the explain API.
 * Handles defaults for optional fields and constructs a complete ExplainRequest.
 */
export function buildExplainRequest(context: ExplainContext, bypassCache: boolean): ExplainRequest {
    if (!context.compiler || !context.lastResult) {
        throw new Error('Missing compiler or compilation result');
    }

    return {
        language: context.compiler.lang,
        compiler: context.compiler.name,
        code: context.lastResult.source ?? '',
        compilationOptions: context.lastResult.compilationOptions ?? [],
        instructionSet: context.lastResult.instructionSet ?? 'amd64',
        asm: Array.isArray(context.lastResult.asm) ? context.lastResult.asm : [],
        audience: context.selectedAudience,
        explanation: context.selectedExplanation,
        ...(bypassCache && {bypassCache: true}),
    };
}

/**
 * Checks if the source code contains a no-ai directive (case-insensitive).
 * Returns true if the directive is found, false otherwise.
 */
export function checkForNoAiDirective(sourceCode: string): boolean {
    return /no-ai/i.test(sourceCode);
}

/**
 * Generates a consistent cache key from the request payload.
 * Uses JSON serialization of normalized payload fields.
 */
export function generateCacheKey(payload: ExplainRequest): string {
    return JSON.stringify({
        language: payload.language,
        compiler: payload.compiler,
        code: payload.code,
        compilationOptions: payload.compilationOptions ?? [],
        instructionSet: payload.instructionSet,
        asm: payload.asm,
        audience: payload.audience,
        explanation: payload.explanation,
    });
}

/**
 * Formats markdown text to HTML using marked with consistent options.
 * Returns the HTML string ready for display.
 */
export function formatMarkdown(markdown: string): string {
    const markedOptions = {
        gfm: true, // GitHub Flavored Markdown
        breaks: true, // Convert line breaks to <br>
    };

    // marked.parse() is synchronous and returns a string, but TypeScript types suggest it could be Promise<string>
    // The cast is safe because we're using the default synchronous implementation
    return marked.parse(markdown, markedOptions) as string;
}

/**
 * Formats statistics text from Claude API response data.
 * Returns an array of formatted stats strings.
 */
export function formatStatsText(
    data: ClaudeExplainResponse,
    clientCacheHit: boolean,
    serverCacheHit: boolean,
): string[] {
    if (!data.usage) return [];

    const stats: string[] = [clientCacheHit ? 'Cached (client)' : serverCacheHit ? 'Cached (server)' : 'Fresh'];

    if (data.model) {
        stats.push(`Model: ${data.model}`);
    }
    if (data.usage.totalTokens) {
        stats.push(`Tokens: ${data.usage.totalTokens}`);
    }
    if (data.cost?.totalCost !== undefined) {
        stats.push(`Cost: $${data.cost.totalCost.toFixed(6)}`);
    }

    return stats;
}

/**
 * Creates HTML content for popover tooltips from an array of options.
 * Each option becomes a formatted div with bold value and description.
 */
export function createPopoverContent(optionsList: Array<{value: string; description: string}>): string {
    return optionsList
        .map(
            option =>
                `<div class='mb-2'><strong>${capitaliseFirst(option.value)}:</strong> ${option.description}</div>`,
        )
        .join('');
}

/**
 * Formats an error for display, handling both Error objects and other types.
 * Returns a user-friendly error message string.
 */
export function formatErrorMessage(error: unknown): string {
    let errorMessage: string;

    if (error instanceof Error) {
        errorMessage = error.message;
    } else if (typeof error === 'string') {
        errorMessage = error;
    } else if (typeof error === 'object' && error !== null) {
        // Try to extract useful information from object errors
        const errorObj = error as Record<string, unknown>;
        if ('message' in errorObj && typeof errorObj.message === 'string') {
            errorMessage = errorObj.message;
        } else if ('error' in errorObj && typeof errorObj.error === 'string') {
            errorMessage = errorObj.error;
        } else {
            // Fall back to JSON.stringify for better debugging
            try {
                errorMessage = JSON.stringify(error);
            } catch {
                errorMessage = String(error);
            }
        }
    } else {
        errorMessage = String(error);
    }

    return `Error: ${errorMessage}`;
}
