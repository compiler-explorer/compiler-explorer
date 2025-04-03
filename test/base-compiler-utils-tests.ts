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
import * as utils from '../lib/utils.js';
import {makeCompilationEnvironment, makeFakeCompilerInfo, newTempDir} from './utils.js';

describe('BaseCompiler utilities', () => {
    let ce: CompilationEnvironment;
    let compiler: BaseCompiler;
    let tempDir: string;

    beforeEach(async () => {
        ce = makeCompilationEnvironment({
            props: {
                compileTimeoutMs: 7000,
                binaryExecTimeoutMs: 2000,
            },
        });

        ce.findBadOptions = vi.fn().mockReturnValue([]);

        tempDir = await newTempDir();

        const compilationInfo = makeFakeCompilerInfo({
            exe: '/fake/compiler/path',
            remote: undefined,
            lang: 'c++',
            options: '-Wall -O2',
        });

        compiler = new BaseCompiler(compilationInfo, ce);
    });

    afterEach(async () => {
        vi.restoreAllMocks();
    });

    describe('getExtraFilepath', () => {
        it('should join paths correctly', () => {
            // Skip this test on Windows as paths are handled differently
            if (process.platform === 'win32') {
                return;
            }

            // Access the protected method using a type assertion
            const extraFilepath = (compiler as any).getExtraFilepath('/tmp/base/dir', 'file.txt');

            expect(extraFilepath).toBe('/tmp/base/dir/file.txt');
        });

        it('should normalize paths', () => {
            // Skip this test on Windows as paths are handled differently
            if (process.platform === 'win32') {
                return;
            }

            // Access the protected method using a type assertion
            const extraFilepath = (compiler as any).getExtraFilepath('/tmp/base/dir', './subdir/../file.txt');

            expect(extraFilepath).toBe('/tmp/base/dir/file.txt');
        });

        it('should reject paths with parent directory traversal', () => {
            // Access the protected method using a type assertion
            expect(() => (compiler as any).getExtraFilepath('/tmp/base/dir', '../file.txt')).toThrow(Error);
            expect(() => (compiler as any).getExtraFilepath('/tmp/base/dir', 'subdir/../../file.txt')).toThrow(Error);
        });
    });

    describe('optionsForFilter', () => {
        // Using TypeScript's "any" assertion to access protected methods
        const getOptions = (filter: any, output: string, userOptions?: string[]) => {
            return (compiler as any).optionsForFilter(filter, output, userOptions);
        };

        it('should handle binary output option', () => {
            const options = getOptions({binary: true}, 'output.out', []);

            // Should include -o output.out but not -S
            expect(options).toContain('-o');
            expect(options).toContain('output.out');
            expect(options).not.toContain('-S');
        });

        it('should handle assembly output option', () => {
            const options = getOptions({binary: false}, 'output.s', []);

            // Should include -o output.s and -S
            expect(options).toContain('-o');
            expect(options).toContain('output.s');
            expect(options).toContain('-S');
        });

        it('should handle debug info option', () => {
            const options = getOptions({binary: false}, 'output.s', []);

            // Should include -g
            expect(options).toContain('-g');
        });

        it('should accept user options but not include them directly', () => {
            // The optionsForFilter method doesn't directly add userOptions to the return value
            // That's handled later in prepareArguments and orderArguments methods
            const options = getOptions({binary: false}, 'output.s', ['-O3', '-march=native']);

            // Should not contain user options yet - they're processed later in the compile chain
            expect(options).not.toContain('-O3');
            expect(options).not.toContain('-march=native');

            // Basic options should still be present
            expect(options).toContain('-S');
            expect(options).toContain('-o');
            expect(options).toContain('output.s');
        });
    });

    describe('writeAllFiles', () => {
        it('should write source file to directory', async () => {
            const fsWriteSpy = vi.spyOn(fs, 'writeFile').mockResolvedValue();

            const source = 'int main() { return 0; }';

            // Access the protected method using a type assertion
            await (compiler as any).writeAllFiles(tempDir, source, [], {});

            // Should have written the source file
            // Access the protected compileFilename using a type assertion
            expect(fsWriteSpy).toHaveBeenCalledWith(path.join(tempDir, (compiler as any).compileFilename), source);
        });

        it('should write multiple files', async () => {
            const fsWriteSpy = vi.spyOn(fs, 'writeFile').mockResolvedValue();
            const outputFileSpy = vi.spyOn(utils, 'outputTextFile').mockResolvedValue();

            const source = 'int main() { return 0; }';
            const extraFiles = [
                {filename: 'header.h', contents: '#pragma once'},
                {filename: 'impl.cpp', contents: '#include "header.h"'},
            ];

            // Access the protected method using a type assertion
            await (compiler as any).writeAllFiles(tempDir, source, extraFiles, {});

            // Should have written the source file
            // Access the protected compileFilename using a type assertion
            expect(fsWriteSpy).toHaveBeenCalledWith(path.join(tempDir, (compiler as any).compileFilename), source);

            // Should have called outputTextFile for each extra file
            expect(outputFileSpy).toHaveBeenCalledTimes(2);

            if (process.platform === 'win32') {
                // On Windows, just check that both calls were made with the correct contents
                // and don't check the exact paths (since they'll use backslashes)
                expect(outputFileSpy.mock.calls[0][1]).toBe('#pragma once');
                expect(outputFileSpy.mock.calls[1][1]).toBe('#include "header.h"');

                // Verify that the first call has header.h in the path
                expect(outputFileSpy.mock.calls[0][0]).toContain('header.h');

                // Verify that the second call has impl.cpp in the path
                expect(outputFileSpy.mock.calls[1][0]).toContain('impl.cpp');
            } else {
                // On non-Windows, check the exact paths with forward slashes
                expect(outputFileSpy).toHaveBeenCalledWith(`${tempDir}/header.h`, '#pragma once');
                expect(outputFileSpy).toHaveBeenCalledWith(`${tempDir}/impl.cpp`, '#include "header.h"');
            }
        });
    });

    describe('findBadOptions', () => {
        // We'll temporarily override the findBadOptions mock to test the real method
        beforeEach(() => {
            vi.restoreAllMocks();
        });

        it('should check options against compiler environment', () => {
            const ceSpy = vi.spyOn(ce, 'findBadOptions').mockReturnValue([]);

            const result = compiler.checkOptions(['-Wall', '-O2']);

            expect(ceSpy).toHaveBeenCalledWith(['-Wall', '-O2']);
            expect(result).toBeNull();
        });

        it('should return error message for bad options', () => {
            vi.spyOn(ce, 'findBadOptions').mockReturnValue(['-badoption']);

            const result = compiler.checkOptions(['-Wall', '-badoption']);

            expect(result).toBe('Bad options: -badoption');
        });
    });
});
