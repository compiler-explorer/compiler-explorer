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

import {describe, expect, it} from 'vitest';

import {SourceHandlerContext, SourceLineHandler} from '../lib/parsers/source-line-handler.js';

describe('SourceLineHandler tests', () => {
    const handler = new SourceLineHandler();
    const context: SourceHandlerContext = {
        files: {
            1: '/path/to/source.cpp',
            2: '/path/to/header.h',
        },
        dontMaskFilenames: false,
    };

    describe('handleSourceTag', () => {
        it('should parse basic .loc directive', () => {
            const result = handler.handleSourceTag('\t.loc\t1 23 0', context);
            expect(result).toEqual({
                file: '/path/to/source.cpp',
                line: 23,
            });
        });

        it('should parse .loc directive with column', () => {
            const result = handler.handleSourceTag('\t.loc\t1 23 5', context);
            expect(result).toEqual({
                file: '/path/to/source.cpp',
                line: 23,
                column: 5,
            });
        });

        it('should return null for non-matching lines', () => {
            const result = handler.handleSourceTag('mov rax, rbx', context);
            expect(result).toBeNull();
        });

        it('should handle dontMaskFilenames flag', () => {
            const contextWithMasking: SourceHandlerContext = {
                ...context,
                dontMaskFilenames: true,
            };
            const result = handler.handleSourceTag('\t.loc\t1 23 0', contextWithMasking);
            expect(result).toEqual({
                file: '/path/to/source.cpp',
                line: 23,
                mainsource: false,
            });
        });
    });

    describe('handleD2Tag', () => {
        it('should parse .d2line directive', () => {
            const result = handler.handleD2Tag('\t.d2line 42');
            expect(result).toEqual({
                file: null,
                line: 42,
            });
        });

        it('should return null for non-matching lines', () => {
            const result = handler.handleD2Tag('mov rax, rbx');
            expect(result).toBeNull();
        });
    });

    describe('handleCVTag', () => {
        it('should parse .cv_loc directive', () => {
            const result = handler.handleCVTag('\t.cv_loc 1 2 42 5', context);
            expect(result).toEqual({
                file: '/path/to/header.h',
                line: 42,
                column: 5,
            });
        });

        it('should parse .cv_loc directive without column', () => {
            const result = handler.handleCVTag('\t.cv_loc 1 1 23 0', context);
            expect(result).toEqual({
                file: '/path/to/source.cpp',
                line: 23,
            });
        });

        it('should return null for non-matching lines', () => {
            const result = handler.handleCVTag('mov rax, rbx', context);
            expect(result).toBeNull();
        });
    });

    describe('handle6502Debug', () => {
        it('should parse .dbg line directive', () => {
            const result = handler.handle6502Debug('\t.dbg line, "test.asm", 42', context);
            expect(result).toEqual({
                file: 'test.asm',
                line: 42,
            });
        });

        it('should return null for .dbg line end directive', () => {
            const result = handler.handle6502Debug('\t.dbg line end', context);
            expect(result).toBeNull();
        });

        it('should return null for non-matching lines', () => {
            const result = handler.handle6502Debug('mov rax, rbx', context);
            expect(result).toBeNull();
        });
    });

    describe('handleStabs', () => {
        it('should handle stab type 68', () => {
            const result = handler.handleStabs('\t.stabn 68,0,42,.');
            expect(result).toEqual({
                file: null,
                line: 42,
            });
        });

        it('should handle stab type 132', () => {
            const result = handler.handleStabs('\t.stabn 132,0,42,.');
            expect(result).toBeNull();
        });

        it('should return undefined for non-matching lines', () => {
            const result = handler.handleStabs('mov rax, rbx');
            expect(result).toBeUndefined();
        });
    });

    describe('processSourceLine', () => {
        it('should process source tag', () => {
            const result = handler.processSourceLine('\t.loc\t1 23 0', context);
            expect(result.source).toEqual({
                file: '/path/to/source.cpp',
                line: 23,
            });
            expect(result.resetPrevLabel).toBe(false);
        });

        it('should return undefined for non-source lines', () => {
            const result = handler.processSourceLine('mov rax, rbx', context);
            expect(result.source).toBeUndefined();
            expect(result.resetPrevLabel).toBe(false);
        });
    });
});
