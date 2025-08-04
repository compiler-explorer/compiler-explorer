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
import {ExplainRequest} from '../../panes/explain-view.interfaces.js';
import {
    buildExplainRequest,
    checkForNoAiDirective,
    ExplainContext,
    generateCacheKey,
    ValidationErrorCode,
    validateExplainPreconditions,
} from '../../panes/explain-view-utils.js';

// Test utilities for creating mock objects - NO MOCKING NEEDED!
function createMockCompilerInfo(overrides: Partial<CompilerInfo> = {}): CompilerInfo {
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

function createMockCompilationResult(overrides: Partial<CompilationResult> = {}): CompilationResult {
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

function createMockAvailableOptions() {
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
    let mockContext: ExplainContext;
    let mockCompiler: CompilerInfo;
    let mockResult: CompilationResult;

    beforeEach(() => {
        mockCompiler = createMockCompilerInfo();
        mockResult = createMockCompilationResult();
        mockContext = {
            lastResult: mockResult,
            compiler: mockCompiler,
            selectedAudience: 'beginner',
            selectedExplanation: 'assembly',
            explainApiEndpoint: 'https://api.example.com/explain',
            consentGiven: true,
            availableOptions: createMockAvailableOptions(),
        };
    });

    // Helper function to test validation failures
    function expectValidationFailure(expectedErrorCode: ValidationErrorCode, expectedMessage: string) {
        const result = validateExplainPreconditions(mockContext);
        expect(result.success).toBe(false);
        if (!result.success) {
            expect(result.errorCode).toBe(expectedErrorCode);
            expect(result.message).toBe(expectedMessage);
        }
    }

    describe('validateExplainPreconditions()', () => {
        it('should pass with valid context', () => {
            const result = validateExplainPreconditions(mockContext);
            expect(result.success).toBe(true);
        });

        it('should return error when lastResult is missing', () => {
            mockContext.lastResult = null;
            expectValidationFailure(
                ValidationErrorCode.MISSING_REQUIRED_DATA,
                'Missing required data: compilation result, consent, or compiler info',
            );
        });

        it('should return error when consent is not given', () => {
            mockContext.consentGiven = false;
            expectValidationFailure(
                ValidationErrorCode.MISSING_REQUIRED_DATA,
                'Missing required data: compilation result, consent, or compiler info',
            );
        });

        it('should return error when compiler is missing', () => {
            mockContext.compiler = null;
            expectValidationFailure(
                ValidationErrorCode.MISSING_REQUIRED_DATA,
                'Missing required data: compilation result, consent, or compiler info',
            );
        });

        it('should return error when options are not available', () => {
            mockContext.availableOptions = null;
            expectValidationFailure(ValidationErrorCode.OPTIONS_NOT_AVAILABLE, 'Explain options not available');
        });

        it('should return error when API endpoint is not configured', () => {
            mockContext.explainApiEndpoint = '';
            expectValidationFailure(
                ValidationErrorCode.API_ENDPOINT_NOT_CONFIGURED,
                'Claude Explain API endpoint not configured',
            );
        });

        it('should return error when no-ai directive is found', () => {
            mockContext.lastResult!.source = 'int main() { /* no-ai */ return 0; }';
            expectValidationFailure(ValidationErrorCode.NO_AI_DIRECTIVE_FOUND, 'no-ai directive found in source code');
        });
    });

    describe('buildExplainRequest()', () => {
        it('should build complete payload with all fields', () => {
            const request = buildExplainRequest(mockContext, false);

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
            mockContext.lastResult = createMockCompilationResult({
                source: undefined,
                compilationOptions: undefined,
                instructionSet: undefined,
                asm: undefined,
            });

            const request = buildExplainRequest(mockContext, false);

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
            const request = buildExplainRequest(mockContext, true);
            expect(request.bypassCache).toBe(true);
        });

        it('should handle non-array asm field', () => {
            mockContext.lastResult!.asm = 'some string content';
            const request = buildExplainRequest(mockContext, false);
            expect(request.asm).toEqual([]);
        });

        it('should throw when compiler is missing', () => {
            mockContext.compiler = null;
            expect(() => buildExplainRequest(mockContext, false)).toThrow('Missing compiler or compilation result');
        });

        it('should throw when lastResult is missing', () => {
            mockContext.lastResult = null;
            expect(() => buildExplainRequest(mockContext, false)).toThrow('Missing compiler or compilation result');
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
});
