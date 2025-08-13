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

import {beforeEach, describe, expect, it} from 'vitest';

import {CompilationResult} from '../../../types/compilation/compilation.interfaces.js';
import {CompilerInfo} from '../../../types/compiler.interfaces.js';
import {ClaudeExplainResponse, ExplainRequest} from '../../panes/explain-view.interfaces.js';
import {
    buildExplainRequest,
    checkForNoAiDirective,
    createPopoverContent,
    ExplainContext,
    formatErrorMessage,
    formatMarkdown,
    formatStatsText,
    generateCacheKey,
    ValidationErrorCode,
    validateExplainPreconditions,
} from '../../panes/explain-view-utils.js';

// Test utilities for creating fake objects for testing
function createFakeCompilerInfo(overrides: Partial<CompilerInfo> = {}): CompilerInfo {
    return {
        id: 'gcc',
        exe: '/usr/bin/gcc',
        name: 'GCC 12.2.0',
        version: '12.2.0',
        fullVersion: 'gcc (GCC) 12.2.0',
        baseName: 'gcc',
        alias: ['gcc'],
        options: '-O2',
        versionFlag: ['--version'],
        lang: 'c++',
        group: 'cpp',
        groupName: 'C++',
        compilerType: 'gcc',
        semver: '12.2.0',
        libsArr: [],
        unwantedLibsArr: [],
        tools: {},
        supportedLibraries: {},
        includeFlag: '-I',
        notification: '',
        instructionSet: 'amd64',
        supportsAsmDocs: false,
        supportsLibraryCodeFilter: false,
        supportsOptOutput: false,
        supportedOpts: [],
        nvccProps: undefined,
        ...overrides,
    } as CompilerInfo;
}

function createFakeCompilationResult(overrides: Partial<CompilationResult> = {}): CompilationResult {
    return {
        code: 0,
        timedOut: false,
        okToCache: true,
        source: 'int main() { return 0; }',
        compilationOptions: ['-O2', '-g'],
        instructionSet: 'amd64',
        asm: [
            {text: 'main:', opcodes: [], address: 0x1000},
            {text: '  ret', opcodes: ['c3'], address: 0x1001},
        ],
        stdout: [],
        stderr: [],
        ...overrides,
    };
}

function createFakeAvailableOptions() {
    return {
        audience: [
            {value: 'beginner', description: 'New to programming'},
            {value: 'intermediate', description: 'Some programming experience'},
            {value: 'expert', description: 'Experienced programmer'},
        ],
        explanation: [
            {value: 'assembly', description: 'Explain the assembly code'},
            {value: 'optimization', description: 'Explain compiler optimizations'},
        ],
    };
}

describe('ExplainView Utils - Pure Functions', () => {
    let testContext: ExplainContext;
    let fakeCompiler: CompilerInfo;
    let fakeResult: CompilationResult;

    beforeEach(() => {
        fakeCompiler = createFakeCompilerInfo();
        fakeResult = createFakeCompilationResult();
        testContext = {
            lastResult: fakeResult,
            compiler: fakeCompiler,
            selectedAudience: 'beginner',
            selectedExplanation: 'assembly',
            explainApiEndpoint: 'https://api.example.com/explain',
            consentGiven: true,
            availableOptions: createFakeAvailableOptions(),
        };
    });

    // Helper function to test validation failures
    function expectValidationFailure(expectedErrorCode: ValidationErrorCode, expectedMessage: string) {
        const result = validateExplainPreconditions(testContext);
        expect(result.success).toBe(false);
        if (!result.success) {
            expect(result.errorCode).toBe(expectedErrorCode);
            expect(result.message).toBe(expectedMessage);
        }
    }

    describe('validateExplainPreconditions()', () => {
        it('should pass with valid context', () => {
            const result = validateExplainPreconditions(testContext);
            expect(result.success).toBe(true);
        });

        it('should return error when lastResult is missing', () => {
            testContext.lastResult = null;
            expectValidationFailure(
                ValidationErrorCode.MISSING_REQUIRED_DATA,
                'Missing required data: compilation result, consent, or compiler info',
            );
        });

        it('should return error when consent is not given', () => {
            testContext.consentGiven = false;
            expectValidationFailure(
                ValidationErrorCode.MISSING_REQUIRED_DATA,
                'Missing required data: compilation result, consent, or compiler info',
            );
        });

        it('should return error when compiler is missing', () => {
            testContext.compiler = null;
            expectValidationFailure(
                ValidationErrorCode.MISSING_REQUIRED_DATA,
                'Missing required data: compilation result, consent, or compiler info',
            );
        });

        it('should return error when options are not available', () => {
            testContext.availableOptions = null;
            expectValidationFailure(ValidationErrorCode.OPTIONS_NOT_AVAILABLE, 'Explain options not available');
        });

        it('should return error when API endpoint is not configured', () => {
            testContext.explainApiEndpoint = '';
            expectValidationFailure(
                ValidationErrorCode.API_ENDPOINT_NOT_CONFIGURED,
                'Claude Explain API endpoint not configured',
            );
        });

        it('should return error when no-ai directive is found', () => {
            testContext.lastResult!.source = 'int main() { /* no-ai */ return 0; }';
            expectValidationFailure(ValidationErrorCode.NO_AI_DIRECTIVE_FOUND, 'no-ai directive found in source code');
        });
    });

    describe('buildExplainRequest()', () => {
        it('should build complete payload with all fields', () => {
            const request = buildExplainRequest(testContext, false);

            expect(request).toEqual({
                language: 'c++',
                compiler: 'GCC 12.2.0',
                code: 'int main() { return 0; }',
                compilationOptions: ['-O2', '-g'],
                instructionSet: 'amd64',
                asm: [
                    {text: 'main:', opcodes: [], address: 0x1000},
                    {text: '  ret', opcodes: ['c3'], address: 0x1001},
                ],
                audience: 'beginner',
                explanation: 'assembly',
            });
        });

        it('should handle missing optional fields with defaults', () => {
            testContext.lastResult = createFakeCompilationResult({
                source: undefined,
                compilationOptions: undefined,
                instructionSet: undefined,
                asm: undefined,
            });

            const request = buildExplainRequest(testContext, false);

            expect(request).toEqual({
                language: 'c++',
                compiler: 'GCC 12.2.0',
                code: '',
                compilationOptions: [],
                instructionSet: 'amd64',
                asm: [],
                audience: 'beginner',
                explanation: 'assembly',
            });
        });

        it('should include bypassCache flag when true', () => {
            const request = buildExplainRequest(testContext, true);
            expect(request.bypassCache).toBe(true);
        });

        it('should handle non-array asm field', () => {
            testContext.lastResult!.asm = 'some string content';
            const request = buildExplainRequest(testContext, false);
            expect(request.asm).toEqual([]);
        });

        it('should throw when compiler is missing', () => {
            testContext.compiler = null;
            expect(() => buildExplainRequest(testContext, false)).toThrow('Missing compiler or compilation result');
        });

        it('should throw when lastResult is missing', () => {
            testContext.lastResult = null;
            expect(() => buildExplainRequest(testContext, false)).toThrow('Missing compiler or compilation result');
        });
    });

    describe('checkForNoAiDirective()', () => {
        it('should return false for normal code', () => {
            const result = checkForNoAiDirective('int main() { return 0; }');
            expect(result).toBe(false);
        });

        it('should detect case-insensitive no-ai directive', () => {
            expect(checkForNoAiDirective('// NO-AI directive')).toBe(true);
            expect(checkForNoAiDirective('// no-ai directive')).toBe(true);
            expect(checkForNoAiDirective('// No-Ai directive')).toBe(true);
        });

        it('should detect no-ai in various contexts', () => {
            expect(checkForNoAiDirective('/* no-ai explanation not wanted */')).toBe(true);
            expect(checkForNoAiDirective('int main() { /* no-ai */ return 0; }')).toBe(true);
            expect(checkForNoAiDirective('# no-ai Python comment')).toBe(true);
        });

        it('should handle edge cases', () => {
            expect(checkForNoAiDirective('')).toBe(false);
            expect(checkForNoAiDirective('   ')).toBe(false);
            expect(checkForNoAiDirective('no ai (without hyphen)')).toBe(false);
        });
    });

    describe('generateCacheKey()', () => {
        let testPayload: ExplainRequest;

        beforeEach(() => {
            testPayload = {
                language: 'c++',
                compiler: 'GCC 12.2.0',
                code: 'int main() { return 0; }',
                compilationOptions: ['-O2'],
                instructionSet: 'amd64',
                asm: [{text: 'main:', opcodes: [], address: 0x1000}],
                audience: 'beginner',
                explanation: 'assembly',
            };
        });

        it('should generate consistent keys for same input', () => {
            const key1 = generateCacheKey(testPayload);
            const key2 = generateCacheKey(testPayload);
            expect(key1).toBe(key2);
        });

        it('should generate different keys for different inputs', () => {
            const key1 = generateCacheKey(testPayload);

            const modifiedPayload = {...testPayload, audience: 'expert'};
            const key2 = generateCacheKey(modifiedPayload);

            expect(key1).not.toBe(key2);
        });

        it('should include all relevant fields in cache key', () => {
            const originalKey = generateCacheKey(testPayload);

            // Test that changing each field changes the key
            const fieldsToTest = ['language', 'compiler', 'code', 'instructionSet', 'audience', 'explanation'] as const;

            fieldsToTest.forEach(field => {
                const modifiedPayload = {...testPayload};
                (modifiedPayload as any)[field] = `modified_${field}`;
                const modifiedKey = generateCacheKey(modifiedPayload);
                expect(modifiedKey).not.toBe(originalKey);
            });
        });

        it('should handle empty compilation options', () => {
            const payloadWithEmptyOptions = {...testPayload, compilationOptions: []};
            expect(() => generateCacheKey(payloadWithEmptyOptions)).not.toThrow();
        });
    });

    describe('formatMarkdown()', () => {
        it('should convert basic markdown to HTML', () => {
            const markdown = '# Hello\n\nThis is **bold** text.';
            const html = formatMarkdown(markdown);

            expect(html).toContain('<h1>Hello</h1>');
            expect(html).toContain('<strong>bold</strong>');
        });

        it('should handle GitHub flavored markdown', () => {
            const markdown = '```cpp\nint main() {}\n```';
            const html = formatMarkdown(markdown);

            expect(html).toContain('<code class="language-cpp">');
            expect(html).toContain('int main() {}');
        });

        it('should convert line breaks to <br> tags', () => {
            const markdown = 'Line 1\nLine 2';
            const html = formatMarkdown(markdown);

            expect(html).toContain('<br>');
        });

        it('should handle empty input', () => {
            expect(formatMarkdown('')).toBe('');
        });
    });

    describe('formatStatsText()', () => {
        let fakeResponse: ClaudeExplainResponse;

        beforeEach(() => {
            fakeResponse = {
                status: 'success',
                explanation: 'Test explanation',
                cached: false,
                usage: {
                    inputTokens: 100,
                    outputTokens: 50,
                    totalTokens: 150,
                },
                model: 'claude-3-sonnet',
                cost: {
                    inputCost: 0.001,
                    outputCost: 0.002,
                    totalCost: 0.003,
                },
            };
        });

        it('should format complete stats with client cache hit', () => {
            const stats = formatStatsText(fakeResponse, true, false);

            expect(stats).toEqual(['Cached (client)', 'Model: claude-3-sonnet', 'Tokens: 150', 'Cost: $0.003000']);
        });

        it('should format complete stats with server cache hit', () => {
            const stats = formatStatsText(fakeResponse, false, true);

            expect(stats).toEqual(['Cached (server)', 'Model: claude-3-sonnet', 'Tokens: 150', 'Cost: $0.003000']);
        });

        it('should format complete stats with fresh response', () => {
            const stats = formatStatsText(fakeResponse, false, false);

            expect(stats).toEqual(['Fresh', 'Model: claude-3-sonnet', 'Tokens: 150', 'Cost: $0.003000']);
        });

        it('should handle missing optional fields', () => {
            const minimalResponse: ClaudeExplainResponse = {
                status: 'success',
                explanation: 'Test explanation',
                cached: false,
                usage: {
                    inputTokens: 100,
                    outputTokens: 50,
                    totalTokens: 150,
                },
            };

            const stats = formatStatsText(minimalResponse, false, false);

            expect(stats).toEqual(['Fresh', 'Tokens: 150']);
        });

        it('should return empty array when usage is missing', () => {
            const noUsageResponse: ClaudeExplainResponse = {
                status: 'success',
                explanation: 'Test explanation',
                cached: false,
                model: 'claude-3-sonnet',
            };
            const stats = formatStatsText(noUsageResponse, false, false);

            expect(stats).toEqual([]);
        });

        it('should handle zero cost correctly', () => {
            const zeroCostResponse: ClaudeExplainResponse = {
                status: 'success',
                explanation: 'Test explanation',
                cached: false,
                usage: {
                    inputTokens: 100,
                    outputTokens: 50,
                    totalTokens: 150,
                },
                cost: {
                    inputCost: 0,
                    outputCost: 0,
                    totalCost: 0,
                },
            };

            const stats = formatStatsText(zeroCostResponse, false, false);

            expect(stats).toEqual(['Fresh', 'Tokens: 150', 'Cost: $0.000000']);
        });
    });

    describe('createPopoverContent()', () => {
        it('should create HTML content for options list', () => {
            const options = [
                {value: 'beginner', description: 'New to programming'},
                {value: 'expert', description: 'Experienced programmer'},
            ];

            const html = createPopoverContent(options);

            expect(html).toContain("<div class='mb-2'><strong>Beginner:</strong> New to programming</div>");
            expect(html).toContain("<div class='mb-2'><strong>Expert:</strong> Experienced programmer</div>");
        });

        it('should handle empty options list', () => {
            const html = createPopoverContent([]);
            expect(html).toBe('');
        });

        it('should capitalize first letter of option values', () => {
            const options = [{value: 'assembly', description: 'Explain assembly code'}];
            const html = createPopoverContent(options);

            expect(html).toContain('<strong>Assembly:</strong>');
        });

        it('should handle special characters in descriptions', () => {
            const options = [{value: 'test', description: 'Description with "quotes" & symbols'}];
            const html = createPopoverContent(options);

            expect(html).toContain('Description with "quotes" & symbols');
        });
    });

    describe('formatErrorMessage()', () => {
        it('should format Error object with message', () => {
            const error = new Error('Something went wrong');
            const formatted = formatErrorMessage(error);

            expect(formatted).toBe('Error: Something went wrong');
        });

        it('should format string error', () => {
            const error = 'Network timeout';
            const formatted = formatErrorMessage(error);

            expect(formatted).toBe('Error: Network timeout');
        });

        it('should format number error', () => {
            const error = 404;
            const formatted = formatErrorMessage(error);

            expect(formatted).toBe('Error: 404');
        });

        it('should format null/undefined errors', () => {
            expect(formatErrorMessage(null)).toBe('Error: null');
            expect(formatErrorMessage(undefined)).toBe('Error: undefined');
        });

        it('should format object error with message property', () => {
            const error = {code: 500, message: 'Internal server error'};
            const formatted = formatErrorMessage(error);

            expect(formatted).toBe('Error: Internal server error');
        });

        it('should format object error with error property', () => {
            const error = {status: 'failed', error: 'Network timeout'};
            const formatted = formatErrorMessage(error);

            expect(formatted).toBe('Error: Network timeout');
        });

        it('should format generic object error as JSON', () => {
            const error = {code: 500, details: 'Something went wrong'};
            const formatted = formatErrorMessage(error);

            expect(formatted).toBe('Error: {"code":500,"details":"Something went wrong"}');
        });
    });
});
