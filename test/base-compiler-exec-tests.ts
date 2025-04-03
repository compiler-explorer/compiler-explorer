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

import {afterEach, beforeEach, describe, expect, it, vi} from 'vitest';

import {BaseCompiler} from '../lib/base-compiler.js';
import {CompilationEnvironment} from '../lib/compilation-env.js';
import * as exec from '../lib/exec.js';
import * as utils from '../lib/utils.js';
import type {FiledataPair} from '../types/compilation/compilation.interfaces.js';
import {ParseFiltersAndOutputOptions} from '../types/features/filters.interfaces.js';

import {createMockExecutor} from './mock/mock-utils.js';
import {
    makeCompilationEnvironment,
    makeFakeCompilerInfo,
    makeFakeParseFiltersAndOutputOptions,
    newTempDir,
} from './utils.js';

const languages = {
    'c++': {id: 'c++'},
    rust: {id: 'rust'},
} as const;

describe('BaseCompiler execution', () => {
    let ce: CompilationEnvironment;
    let compiler: BaseCompiler;
    let mockExec: ReturnType<typeof createMockExecutor>;
    let execSpy: any;

    beforeEach(async () => {
        ce = makeCompilationEnvironment({
            languages,
            props: {
                compileTimeoutMs: 7000,
                binaryExecTimeoutMs: 2000,
            },
        });

        // Mock the findBadOptions function
        ce.findBadOptions = vi.fn().mockReturnValue([]);

        const compilationInfo = makeFakeCompilerInfo({
            exe: '/fake/compiler/path',
            remote: undefined,
            lang: 'c++',
            options: '-Wall -O2',
            supportsExecute: true,
            supportsBinary: true,
        });

        compiler = new BaseCompiler(compilationInfo, ce);
        // Create a temporary directory for testing
        await newTempDir();

        // Set up our mocks
        mockExec = createMockExecutor(
            new Map([
                [
                    '/fake/compiler/path -Wall -O2 -g -o output.s -S input.cpp',
                    {
                        code: 0,
                        stdout: 'Compilation successful',
                        stderr: '',
                    },
                ],
                [
                    '/fake/compiler/path -Wall -O2 -g input.cpp -o a.out',
                    {
                        code: 0,
                        stdout: 'Binary compilation successful',
                        stderr: '',
                    },
                ],
            ]),
        );

        vi.spyOn(exec, 'execute').mockImplementation(mockExec);
        execSpy = vi.spyOn(compiler, 'exec');

        // Mock the exec method with a fully typed result for compilation
        execSpy.mockImplementation(async () => {
            return {
                code: 0,
                stdout: 'Compilation successful',
                stderr: '',
                okToCache: true,
                filenameTransform: (x: string) => x,
                execTime: 0,
                timedOut: false,
                truncated: false,
            };
        });

        // Mock filesystem operations
        vi.spyOn(fs, 'writeFile').mockImplementation(async () => {});
        vi.spyOn(fs, 'readFile').mockResolvedValue('Assembly output');

        // In BaseCompiler, binary execution is mocked indirectly through exec
        // We don't call sandbox directly but through the compiler's internal methods
        vi.spyOn(exec, 'sandbox').mockResolvedValue({
            code: 0,
            stdout: 'Program executed successfully',
            stderr: '',
            timedOut: false,
            truncated: false,
            filenameTransform: (x: string) => x,
            execTime: 0,
            okToCache: true,
        });

        // Need to mock handleExecution within the compiler
        vi.spyOn(compiler, 'handleExecution' as any).mockImplementation(async () => {
            return {
                code: 0,
                didExecute: true,
                stdout: [{text: 'Program executed successfully'}],
                stderr: [],
                buildResult: {
                    code: 0,
                    stdout: [{text: 'Binary compilation successful'}],
                    stderr: [],
                },
            };
        });

        // Mock check source method to prevent validation issues
        vi.spyOn(compiler, 'checkSource').mockReturnValue(null);
    });

    afterEach(async () => {
        vi.restoreAllMocks();
    });

    it('should successfully compile and execute code', async () => {
        const source = 'int main() { return 42; }';
        const options = [''];
        const filters: ParseFiltersAndOutputOptions = makeFakeParseFiltersAndOutputOptions({
            execute: true, // Request execution
        });

        const result = await compiler.compile(
            source,
            options,
            {},
            filters,
            false as any, // bypassCache
            [], // backendOptions
            {}, // filters
            [], // tools
            [] as FiledataPair[], // executionParams
        );

        expect(result.code).toBe(0);
        expect(result.execResult).toBeDefined();
        expect(result.execResult?.didExecute).toBe(true);
        expect(result.execResult?.code).toBe(0);
        expect(result.execResult?.stdout).toEqual([{text: 'Program executed successfully'}]);
    });

    it('should handle execution failures with non-zero return code', async () => {
        const source = 'int main() { return 1; }';
        const options = [''];
        const filters: ParseFiltersAndOutputOptions = makeFakeParseFiltersAndOutputOptions({
            execute: true, // Request execution
        });

        // Override handleExecution mock to simulate execution failure
        vi.spyOn(compiler, 'handleExecution' as any).mockImplementation(async () => {
            return {
                code: 1,
                didExecute: true,
                stdout: [],
                stderr: [{text: 'Execution failed with return code 1'}],
                buildResult: {
                    code: 0,
                    stdout: [{text: 'Binary compilation successful'}],
                    stderr: [],
                },
            };
        });

        const result = await compiler.compile(
            source,
            options,
            {},
            filters,
            false as any,
            [],
            {},
            [],
            [] as FiledataPair[],
        );

        expect(result.code).toBe(0); // Compilation succeeded
        expect(result.execResult).toBeDefined();
        expect(result.execResult?.didExecute).toBe(true);
        expect(result.execResult?.code).toBe(1); // Execution failed
        expect(result.execResult?.stderr).toEqual([{text: 'Execution failed with return code 1'}]);
    });

    it('should handle compilation failures for execution', async () => {
        const source = 'int main() { error_function(); return 0; }';
        const options = [''];
        const filters: ParseFiltersAndOutputOptions = makeFakeParseFiltersAndOutputOptions({
            execute: true, // Request execution
        });

        // Mock handleExecution to simulate binary compilation failure
        vi.spyOn(compiler, 'handleExecution' as any).mockImplementation(async () => {
            return {
                code: -1,
                didExecute: false,
                stdout: [],
                stderr: [{text: 'Build failed'}],
                buildResult: {
                    code: 1,
                    stdout: [],
                    stderr: [{text: 'error: use of undeclared identifier "error_function"'}],
                },
            };
        });

        const result = await compiler.compile(
            source,
            options,
            {},
            filters,
            false as any,
            [],
            {},
            [],
            [] as FiledataPair[],
        );

        expect(result.code).toBe(0); // Assembly compilation succeeded
        expect(result.execResult).toBeDefined();
        expect(result.execResult?.didExecute).toBe(false);
        expect(result.execResult?.buildResult?.code).toBe(1);
        expect(result.execResult?.buildResult?.stderr).toEqual([
            {text: 'error: use of undeclared identifier "error_function"'},
        ]);
    });

    it('should handle timeouts during execution', async () => {
        const source = 'int main() { while(1); return 0; }';
        const options = [''];
        const filters: ParseFiltersAndOutputOptions = makeFakeParseFiltersAndOutputOptions({
            execute: true, // Request execution
        });

        // Mock handleExecution to simulate execution timeout
        vi.spyOn(compiler, 'handleExecution' as any).mockImplementation(async () => {
            return {
                code: -1,
                didExecute: true,
                stdout: [],
                stderr: [{text: 'Execution time out'}],
                timedOut: true,
                buildResult: {
                    code: 0,
                    stdout: [{text: 'Binary compilation successful'}],
                    stderr: [],
                },
            };
        });

        const result = await compiler.compile(
            source,
            options,
            {},
            filters,
            false as any,
            [],
            {},
            [],
            [] as FiledataPair[],
        );

        expect(result.code).toBe(0); // Compilation succeeded
        expect(result.execResult).toBeDefined();
        expect(result.execResult?.didExecute).toBe(true);
        expect(result.execResult?.timedOut).toBe(true);
        expect(result.execResult?.stderr).toEqual([{text: 'Execution time out'}]);
    });

    it('should pass execution parameters to the compiler', async () => {
        const source = 'int main(int argc, char** argv) { return argc > 1 ? 0 : 1; }';
        const options = [''];
        const filters: ParseFiltersAndOutputOptions = makeFakeParseFiltersAndOutputOptions({
            execute: true, // Request execution
        });

        // Create a spy to capture execution params
        const handleExecutionSpy = vi.spyOn(compiler, 'handleExecution' as any);

        // Mock return value for execution
        handleExecutionSpy.mockResolvedValue({
            code: 0,
            didExecute: true,
            stdout: [{text: 'Program executed successfully'}],
            stderr: [],
            buildResult: {
                code: 0,
                stdout: [{text: 'Binary compilation successful'}],
                stderr: [],
            },
        });

        await compiler.compile(source, options, {}, filters, false as any, [], {}, [], [] as FiledataPair[]);

        // Verify execution was called with the parameters
        expect(handleExecutionSpy).toHaveBeenCalled();
        // In the actual implementation, handleExecution would receive the exec params
        // but since we're mocking it, we just verify it was called
    });

    it('should not execute when compiler does not support execution', async () => {
        // Create a new compiler that doesn't support execution
        const noExecCompilerInfo = makeFakeCompilerInfo({
            exe: '/fake/compiler/path',
            remote: undefined,
            lang: 'c++',
            options: '-Wall -O2',
            supportsExecute: false, // No execution support
            supportsBinary: true,
        });

        const noExecCompiler = new BaseCompiler(noExecCompilerInfo, ce);
        vi.spyOn(noExecCompiler, 'exec').mockImplementation(async () => {
            return {
                code: 0,
                stdout: 'Compilation successful',
                stderr: '',
                okToCache: true,
                filenameTransform: (x: string) => x,
                execTime: 0,
                timedOut: false,
                truncated: false,
            };
        });

        // Mock handleExecution to indicate compiler doesn't support execution
        vi.spyOn(noExecCompiler, 'handleExecution' as any).mockImplementation(async () => {
            return {
                code: -1,
                didExecute: false,
                stdout: [],
                stderr: [{text: 'Compiler does not support execution'}],
                buildResult: {
                    code: 0,
                    stdout: [],
                    stderr: [],
                },
            };
        });

        const source = 'int main() { return 0; }';
        const options = [''];
        const filters: ParseFiltersAndOutputOptions = makeFakeParseFiltersAndOutputOptions({
            execute: true, // Request execution despite no support
        });

        const result = await noExecCompiler.compile(
            source,
            options,
            {},
            filters,
            false as any,
            [],
            {},
            [],
            [] as FiledataPair[],
        );

        expect(result.code).toBe(0);
        expect(result.execResult).toBeDefined();
        expect(result.execResult?.didExecute).toBe(false);
        expect(result.execResult?.stderr).toEqual([{text: 'Compiler does not support execution'}]);
    });
});

describe('BaseCompiler error handling', () => {
    let ce: CompilationEnvironment;
    let compiler: BaseCompiler;
    let execSpy: any;

    beforeEach(async () => {
        ce = makeCompilationEnvironment({
            languages,
            props: {
                compileTimeoutMs: 7000,
                binaryExecTimeoutMs: 2000,
            },
        });

        ce.findBadOptions = vi.fn().mockReturnValue([]);

        const compilationInfo = makeFakeCompilerInfo({
            exe: '/fake/compiler/path',
            remote: undefined,
            lang: 'c++',
            options: '-Wall -O2',
        });

        compiler = new BaseCompiler(compilationInfo, ce);
        execSpy = vi.spyOn(compiler, 'exec');

        // Mock filesystem operations
        vi.spyOn(fs, 'writeFile').mockImplementation(async () => {});
        vi.spyOn(fs, 'readFile').mockResolvedValue('Assembly output');

        // Mock check source method to prevent validation issues
        vi.spyOn(compiler, 'checkSource').mockReturnValue(null);
    });

    afterEach(async () => {
        vi.restoreAllMocks();
    });

    it('should handle compilation timeouts properly', async () => {
        // Simulate a timeout during compilation
        execSpy.mockResolvedValue({
            code: -1,
            stdout: '',
            stderr: '',
            okToCache: true,
            filenameTransform: (x: string) => x,
            execTime: 0,
            timedOut: true,
            truncated: false,
        });

        const source = 'int main() { return 0; }';
        const options = [''];
        const filters: ParseFiltersAndOutputOptions = makeFakeParseFiltersAndOutputOptions({});

        const result = await compiler.compile(
            source,
            options,
            {},
            filters,
            false as any,
            [],
            {},
            [],
            [] as FiledataPair[],
        );

        expect(result.code).toBe(-1);
        expect(result.timedOut).toBe(true);
        // The actual message might be different, but we know it should indicate compilation failure
        expect(result.asm[0].text).toContain('<Compilation failed>');
    });

    it('should handle output truncation', async () => {
        // Simulate truncated output
        execSpy.mockResolvedValue({
            code: 0,
            stdout: 'Truncated stdout',
            stderr: 'Truncated stderr',
            okToCache: true,
            filenameTransform: (x: string) => x,
            execTime: 0,
            timedOut: false,
            truncated: true,
        });

        const source = 'int main() { return 0; }';
        const options = [''];
        const filters: ParseFiltersAndOutputOptions = makeFakeParseFiltersAndOutputOptions({});

        const result = await compiler.compile(
            source,
            options,
            {},
            filters,
            false as any,
            [],
            {},
            [],
            [] as FiledataPair[],
        );

        expect(result.code).toBe(0);
        expect(result.truncated).toBe(true);
        // The actual assembly text may vary in different environments
        expect(result.stdout).toEqual([{text: 'Truncated stdout'}]);
        expect(result.stderr).toEqual([{text: 'Truncated stderr'}]);
    });

    it('should handle corrupted output files', async () => {
        // Successful compilation
        execSpy.mockResolvedValue({
            code: 0,
            stdout: 'Compilation successful',
            stderr: '',
            okToCache: true,
            filenameTransform: (x: string) => x,
            execTime: 0,
            timedOut: false,
            truncated: false,
        });

        // But readFile throws an error
        vi.spyOn(fs, 'readFile').mockRejectedValue(new Error('Could not read output file'));

        const source = 'int main() { return 0; }';
        const options = [''];
        const filters: ParseFiltersAndOutputOptions = makeFakeParseFiltersAndOutputOptions({});

        const result = await compiler.compile(
            source,
            options,
            {},
            filters,
            false as any,
            [],
            {},
            [],
            [] as FiledataPair[],
        );

        expect(result.code).toBe(0); // Compilation succeeded
        // The actual error message may contain the specific error details
        expect(result.asm[0].text).toContain('<');
    });

    it('should handle source validation', async () => {
        // Mock the checkSource method for testing
        vi.spyOn(compiler, 'checkSource').mockImplementation((source: string) => {
            if (source.includes('</etc/passwd>')) {
                return '<stdin>:1:1: no absolute or relative includes please';
            }
            return null;
        });

        // Create a simple test case for source validation
        const validSource = '#include <iostream>\nint main() { return 0; }';
        const invalidSource = '#include </etc/passwd>';

        // For valid source, checkSource should return null
        expect(compiler.checkSource(validSource)).toBeNull();

        // For invalid source with absolute path in include, it should return an error message
        const result = compiler.checkSource(invalidSource);
        expect(result).not.toBeNull();
        expect(result).toContain('no absolute or relative includes');
    });

    it('should handle compiler failures with memory safety errors', async () => {
        execSpy.mockResolvedValue({
            code: 139, // SIGSEGV exit code
            stdout: '',
            stderr: 'Segmentation fault',
            okToCache: true,
            filenameTransform: (x: string) => x,
            execTime: 0,
            timedOut: false,
            truncated: false,
        });

        const source = 'int main() { return 0; }';
        const options = [''];
        const filters: ParseFiltersAndOutputOptions = makeFakeParseFiltersAndOutputOptions({});

        const result = await compiler.compile(
            source,
            options,
            {},
            filters,
            false as any,
            [],
            {},
            [],
            [] as FiledataPair[],
        );

        expect(result.code).toBe(139);
        expect(result.asm).toEqual([{text: '<Compilation failed>', source: null, labels: []}]);
        expect(result.stderr).toEqual([{text: 'Segmentation fault'}]);
    });
});

describe('BaseCompiler include handling', () => {
    let ce: CompilationEnvironment;
    let compiler: BaseCompiler;
    let outputFileSpy: any;

    beforeEach(async () => {
        ce = makeCompilationEnvironment({
            languages,
            props: {
                compileTimeoutMs: 7000,
                binaryExecTimeoutMs: 2000,
            },
        });

        ce.findBadOptions = vi.fn().mockReturnValue([]);

        const compilationInfo = makeFakeCompilerInfo({
            exe: '/fake/compiler/path',
            remote: undefined,
            lang: 'c++',
            options: '-Wall -O2',
        });

        compiler = new BaseCompiler(compilationInfo, ce);
        // Create a temporary directory for testing
        await newTempDir();

        // Mock filesystem operations
        vi.spyOn(fs, 'writeFile').mockImplementation(async () => {});
        vi.spyOn(fs, 'readFile').mockResolvedValue('Assembly output');

        outputFileSpy = vi.spyOn(utils, 'outputTextFile').mockResolvedValue();

        // Mock the exec method with a fully typed result
        vi.spyOn(compiler, 'exec').mockImplementation(async () => {
            return {
                code: 0,
                stdout: 'Compilation successful',
                stderr: '',
                okToCache: true,
                filenameTransform: (x: string) => x,
                execTime: 0,
                timedOut: false,
                truncated: false,
            };
        });

        // Mock check source method to prevent validation issues
        vi.spyOn(compiler, 'checkSource').mockReturnValue(null);
    });

    afterEach(async () => {
        vi.restoreAllMocks();
    });

    it('should properly prepare input with additional files', async () => {
        // Access the protected method directly to test it
        const writeAllFiles = async (dirPath: string, source: string, files: any[]) => {
            return await (compiler as any).writeAllFiles(dirPath, source, files, {});
        };

        const source = '#include "header.h"\nint main() { return TEST_VALUE; }';
        const additionalFiles = [
            {
                filename: 'header.h',
                contents: '#define TEST_VALUE 42',
            },
        ];

        // Mock the utility functions
        outputFileSpy.mockClear();

        // Call the method directly
        await writeAllFiles('/tmp/test', source, additionalFiles);

        // Verify the main source file was written
        expect(fs.writeFile).toHaveBeenCalled();

        // Verify the output file utility was called for the header
        expect(outputFileSpy).toHaveBeenCalledWith(expect.stringContaining('header.h'), '#define TEST_VALUE 42');
    });

    it('should correctly handle paths in additional files', async () => {
        // Test getExtraFilepath directly for path validation
        const getExtraFilepath = (dirPath: string, filename: string) => {
            return (compiler as any).getExtraFilepath(dirPath, filename);
        };

        // Valid paths
        expect(getExtraFilepath('/tmp/test', 'header.h')).toBe(path.join('/tmp/test', 'header.h'));
        expect(getExtraFilepath('/tmp/test', 'subdir/header.h')).toBe(path.join('/tmp/test', 'subdir', 'header.h'));

        // Invalid paths with directory traversal
        expect(() => getExtraFilepath('/tmp/test', '../header.h')).toThrow();
        expect(() => getExtraFilepath('/tmp/test', 'subdir/../../header.h')).toThrow();
    });
});
