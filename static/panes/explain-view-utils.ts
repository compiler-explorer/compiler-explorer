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

import {CompilationResult} from '../../types/compilation/compilation.interfaces.js';
import {CompilerInfo} from '../../types/compiler.interfaces.js';
import {AvailableOptions, ExplainRequest} from './explain-view.interfaces.js';

// Anything to do with the explain view that doesn't need any direct UI access, so we can test it easily.

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
