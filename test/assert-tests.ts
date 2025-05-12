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

import path from 'node:path';
import {beforeEach, describe, expect, it} from 'vitest';

import {assert, check_path, removeFileProtocol, setBaseDirectory, unwrap, unwrapString} from '../lib/assert.js';

describe('Assert module', () => {
    // Reset the base directory for each test
    beforeEach(() => {
        setBaseDirectory(new URL('file:///test/path/'));
    });

    describe('setBaseDirectory', () => {
        it('should set the base directory from a URL', () => {
            const testUrl = new URL('file:///test/path/');
            setBaseDirectory(testUrl);

            // We can verify the base directory was set by testing check_path
            // which uses the base directory internally
            const result = check_path('/test/path/', '/test/path/subdir/file.js');
            expect(result).toEqual('subdir/file.js');
        });
    });

    describe('removeFileProtocol', () => {
        it('should remove file:// prefix', () => {
            expect(removeFileProtocol('file:///path/to/file')).toEqual('/path/to/file');
        });

        it('should not modify paths without file:// prefix', () => {
            expect(removeFileProtocol('/path/to/file')).toEqual('/path/to/file');
        });
    });

    describe('check_path', () => {
        it('should return relative path for valid subdirectories', () => {
            // This test now uses normalized paths with forward slashes for cross-platform compatibility
            expect(check_path('/root', '/root/sub/file.js')).toEqual('sub/file.js');
        });

        it('should return false for paths outside the parent directory', () => {
            expect(check_path('/root/sub', '/root/file.js')).toBe(false);
            expect(check_path('/root', '/other/file.js')).toBe(false);
        });

        it('should return false for absolute paths', () => {
            expect(check_path('/root', '/root')).toBe(false);
        });
    });

    describe('get_diagnostic function components', () => {
        // Since get_diagnostic relies on stack traces which are hard to test directly,
        // we test each component separately to verify the logic is correct

        describe('stack trace handling', () => {
            it('should correctly process file paths with removeFileProtocol', () => {
                const pathWithProtocol = 'file:///path/to/file.js';
                expect(removeFileProtocol(pathWithProtocol)).toEqual('/path/to/file.js');

                const pathWithoutProtocol = '/path/to/file.js';
                expect(removeFileProtocol(pathWithoutProtocol)).toEqual('/path/to/file.js');
            });

            it('should properly determine if a path is within the base directory using check_path', () => {
                // Valid paths inside base directory - these paths use forward slashes regardless of platform
                // for cross-platform test compatibility
                const testRoot = '/base/dir';
                const testFile = path.join(testRoot, 'file.js');
                const testSubdirFile = path.join(testRoot, 'subdir', 'file.js');

                expect(check_path(testRoot, testFile)).toEqual('file.js');
                expect(check_path(testRoot, testSubdirFile)).toEqual('subdir/file.js');

                // Invalid paths
                expect(check_path('/base/dir', '/other/path/file.js')).toBe(false);
                expect(check_path('/base/dir', '/base')).toBe(false);
            });

            it('should handle the combination of file protocol removal and path checking', () => {
                const baseDir = '/base/dir';
                const filePathWithProtocol = 'file:///base/dir/file.js';

                // First remove protocol (as done in get_diagnostic)
                const filePath = removeFileProtocol(filePathWithProtocol);

                // Then check if path is within base directory (as done in get_diagnostic)
                const relativePath = check_path(baseDir, filePath);

                // The relative path should be 'file.js' because it's directly in the base dir
                expect(relativePath).toEqual('file.js');
            });
        });

        describe('file content extraction', () => {
            it('should correctly extract a specific line from file content', () => {
                // This mimics the line extraction logic in get_diagnostic
                const fileContent = 'line1\nline2\nline3\nline4\nline5';
                const lines = fileContent.split('\n');

                expect(lines[0].trim()).toEqual('line1');
                expect(lines[2].trim()).toEqual('line3');
                expect(lines[4].trim()).toEqual('line5');
            });
        });
    });

    describe('assert', () => {
        it('should not throw for truthy values', () => {
            expect(() => assert(true)).not.toThrow();
            expect(() => assert(1)).not.toThrow();
            expect(() => assert('string')).not.toThrow();
            expect(() => assert({})).not.toThrow();
        });

        it('should throw for falsy values', () => {
            expect(() => assert(false)).toThrow('Assertion failed');
            expect(() => assert(0)).toThrow('Assertion failed');
            expect(() => assert('')).toThrow('Assertion failed');
            expect(() => assert(null)).toThrow('Assertion failed');
            expect(() => assert(undefined)).toThrow('Assertion failed');
        });

        it('should include custom message in error', () => {
            expect(() => assert(false, 'Custom message')).toThrow(/Custom message/);
        });

        it('should include extra info in error message', () => {
            const extra = {a: 1, b: 2};
            expect(() => assert(false, 'Message', extra)).toThrow(JSON.stringify([extra]));
        });

        it('should include the assertion message', () => {
            try {
                assert(false, 'Test assertion');
                // Should not reach here
                expect(true).toBe(false);
            } catch (e: any) {
                expect(e.message).toContain('Assertion failed: Test assertion');
            }
        });
    });

    describe('unwrap', () => {
        it('should return the value for non-null/undefined values', () => {
            expect(unwrap(5)).toEqual(5);
            expect(unwrap('test')).toEqual('test');
            expect(unwrap(false)).toEqual(false);
            expect(unwrap(0)).toEqual(0);

            const obj = {test: true};
            expect(unwrap(obj)).toBe(obj);
        });

        it('should throw for null values', () => {
            expect(() => unwrap(null)).toThrow('Unwrap failed');
        });

        it('should throw for undefined values', () => {
            expect(() => unwrap(undefined)).toThrow('Unwrap failed');
        });

        it('should include custom message in error', () => {
            expect(() => unwrap(null, 'Custom unwrap message')).toThrow(/Custom unwrap message/);
        });

        it('should include extra info in error message', () => {
            const extra = {reason: 'test'};
            expect(() => unwrap(null, 'Message', extra)).toThrow(JSON.stringify([extra]));
        });
    });

    describe('unwrapString', () => {
        it('should return string values', () => {
            expect(unwrapString('test')).toEqual('test');
            expect(unwrapString('')).toEqual('');
        });

        it('should throw for non-string values', () => {
            expect(() => unwrapString(123)).toThrow('String unwrap failed');
            expect(() => unwrapString(null)).toThrow('String unwrap failed');
            expect(() => unwrapString(undefined)).toThrow('String unwrap failed');
            expect(() => unwrapString({})).toThrow('String unwrap failed');
            expect(() => unwrapString([])).toThrow('String unwrap failed');
        });

        it('should include custom message in error', () => {
            expect(() => unwrapString(123, 'Expected string value')).toThrow(/Expected string value/);
        });

        it('should include extra info in error message', () => {
            const extra = {value: 123, type: 'number'};
            expect(() => unwrapString(123, 'Message', extra)).toThrow(JSON.stringify([extra]));
        });
    });
});
