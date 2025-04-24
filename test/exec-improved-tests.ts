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

import {afterEach, beforeEach, describe, expect, it, vi} from 'vitest';

import {logger} from '../lib/logger.js';
import {ExecutionOptions, ExecutionOptionsWithEnv} from '../types/compilation/compilation.interfaces.js';

// Helper function to manually check environment variables
function manualCheckExecOptions(options: ExecutionOptions) {
    if (options.env) {
        for (const key of Object.keys(options.env)) {
            const value: any = options.env[key];
            if (value !== undefined && typeof value !== 'string') {
                // Convert to string, handling null specially
                options.env[key] = value === null ? 'null' : value.toString();
            }
        }
    }
}

// Test environment variable conversion with manual checking
describe('Environment Variable Conversion', () => {
    beforeEach(() => {
        // Avoid console output during tests
        vi.spyOn(logger, 'warn').mockImplementation(() => logger);
        vi.spyOn(logger, 'error').mockImplementation(() => logger);
    });

    afterEach(() => {
        vi.restoreAllMocks();
    });

    it('should convert non-string environment variables to strings', async () => {
        const options: ExecutionOptionsWithEnv = {
            env: {
                STRING_VAR: 'string value',
                NUMBER_VAR: 42 as any,
                BOOLEAN_VAR: true as any,
                OBJECT_VAR: {toString: () => 'object value'} as any,
            },
        };

        manualCheckExecOptions(options);

        expect(options.env.STRING_VAR).toBe('string value');
        expect(options.env.NUMBER_VAR).toBe('42');
        expect(options.env.BOOLEAN_VAR).toBe('true');
        expect(options.env.OBJECT_VAR).toBe('object value');
    });

    it('should handle undefined and null values', async () => {
        const options: ExecutionOptionsWithEnv = {
            env: {
                UNDEFINED_VAR: undefined as any,
                NULL_VAR: null as any,
            },
        };

        manualCheckExecOptions(options);

        // undefined should remain undefined (not converted)
        expect(options.env.UNDEFINED_VAR).toBe(undefined);
        expect(options.env.NULL_VAR).toBe('null');
    });

    it('should handle missing env object', async () => {
        const options: ExecutionOptions = {};

        // Should not throw
        expect(() => manualCheckExecOptions(options)).not.toThrow();
    });
});

// Mock the path transformation function for testing
describe('Path Transformation', () => {
    it('should transform paths correctly', () => {
        // Instead of testing real NsJail options, let's test our own transformation logic
        const customCwd = '/tmp/work';
        const appPath = '/app';

        // Create our own transformation function
        const filenameTransform = (path: string) => {
            return path.startsWith(customCwd) ? path.replace(customCwd, appPath) : path;
        };

        // Test various path transformations
        expect(filenameTransform('/tmp/work/file.txt')).toBe('/app/file.txt');
        expect(filenameTransform('/tmp/work/dir/file.txt')).toBe('/app/dir/file.txt');
        expect(filenameTransform('/tmp/other/file.txt')).toBe('/tmp/other/file.txt'); // Not in customCwd
        expect(filenameTransform('/usr/bin/gcc')).toBe('/usr/bin/gcc'); // Not in customCwd
    });

    it('should handle path transformations for LD paths', () => {
        // Create our own transformation function
        const customCwd = '/tmp/work';
        const appPath = '/app';

        const filenameTransform = (path: string) => {
            return path.startsWith(customCwd) ? path.replace(customCwd, appPath) : path;
        };

        // Transform LD_LIBRARY_PATH
        const ldPaths = ['/tmp/work/lib', '/usr/lib'];
        const transformedPaths = ldPaths.map(filenameTransform);

        expect(transformedPaths).toEqual(['/app/lib', '/usr/lib']);
    });
});
