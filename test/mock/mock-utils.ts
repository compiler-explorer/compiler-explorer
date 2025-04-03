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

import fs from 'node:fs/promises';
import path from 'node:path';
import {type Mock, vi} from 'vitest';

import {UnprocessedExecResult} from '../../types/execution/execution.interfaces.js';

/**
 * Creates a mock executor function for testing base-compiler.exec calls
 *
 * @param execResults Map of command:result pairs to return for different commands
 * @param defaultResult Default result to return if command not found in map
 * @returns Mock function that returns appropriate results based on command
 */
export function createMockExecutor(
    execResults: Map<string, Partial<UnprocessedExecResult>>,
    defaultResult: Partial<UnprocessedExecResult> = {
        code: 0,
        stdout: '',
        stderr: '',
        timedOut: false,
        okToCache: true,
    },
): Mock {
    return vi.fn().mockImplementation(async (command: string, args: string[], options: any) => {
        // Basic command matching - could be made more sophisticated
        const commandKey = [command, ...args].join(' ');
        const result = execResults.get(commandKey) || defaultResult;

        // Ensure we always have a filenameTransform
        if (!result.filenameTransform) {
            result.filenameTransform = x => x;
        }

        return result;
    });
}

/**
 * Creates a mock fs.writeFile function that actually writes to a temp directory
 */
export function createMockFileWriter(tempDir: string): Mock {
    return vi.fn().mockImplementation(async (filePath: string, content: string) => {
        const filename = path.basename(filePath);
        const targetPath = path.join(tempDir, filename);
        await fs.writeFile(targetPath, content);
        return targetPath;
    });
}

/**
 * Creates a mock fs.readFile function that returns predetermined content
 */
export function createMockFileReader(fileContents: Map<string, string>): Mock {
    return vi.fn().mockImplementation(async (filePath: string) => {
        const filename = path.basename(filePath);
        if (fileContents.has(filename)) {
            return fileContents.get(filename);
        }
        if (fileContents.has(filePath)) {
            return fileContents.get(filePath);
        }
        throw new Error(`File not found: ${filePath}`);
    });
}

/**
 * Creates mock compilation environment methods for testing caching behavior
 */
export function createMockCacheMethods() {
    const cacheGet = vi.fn();
    const cachePut = vi.fn();

    return {
        cacheGet,
        cachePut,
    };
}

/**
 * Creates a set of commonly needed mocks for compiler testing
 */
export function setupCompilerTestMocks(tempDir: string) {
    const mockExec = createMockExecutor(new Map());
    const mockFs = {
        writeFile: createMockFileWriter(tempDir),
        readFile: createMockFileReader(new Map()),
        stat: vi.fn().mockResolvedValue({isFile: () => true}),
    };
    const mockCache = createMockCacheMethods();

    return {
        mockExec,
        mockFs,
        mockCache,
    };
}
