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

import child_process from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {MockInstance, afterEach, beforeEach, describe, expect, it, vi} from 'vitest';
import {
    convertOptionsToAppArguments,
    detectWsl,
    getGitReleaseName,
    getReleaseBuildNumber,
    parseCommandLine,
    parsePortNumberForOptions,
    setupTempDir,
} from '../../lib/app/cli.js';

describe('CLI Module', () => {
    describe('parsePortNumberForOptions', () => {
        it('should parse valid numbers', () => {
            expect(parsePortNumberForOptions('123')).toEqual(123);
            expect(parsePortNumberForOptions('0')).toEqual(0);
            expect(parsePortNumberForOptions('65535')).toEqual(65535);
        });

        it('should throw on negative numbers', () => {
            expect(() => parsePortNumberForOptions('-1')).toThrow();
        });

        it('should throw on invalid numbers', () => {
            expect(() => parsePortNumberForOptions('65536')).toThrow();
        });

        it('should throw on invalid number - abc', () => {
            expect(() => parsePortNumberForOptions('abc')).toThrow();
        });

        it('should throw on invalid number - 123abc', () => {
            expect(() => parsePortNumberForOptions('123abc')).toThrow();
        });

        it('should throw on empty string', () => {
            expect(() => parsePortNumberForOptions('')).toThrow();
        });
    });

    describe('getGitReleaseName', () => {
        // Create a temporary directory for each test
        let tempDir: string;
        let spyOnExecSync: MockInstance;

        beforeEach(() => {
            tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'ce-git-test-'));
            spyOnExecSync = vi.spyOn(child_process, 'execSync');
        });

        afterEach(() => {
            // Clean up the temporary directory
            fs.rmSync(tempDir, {recursive: true, force: true});
            vi.restoreAllMocks();
        });

        it('should read from git_hash in dist mode', () => {
            // Create the git_hash file with a known hash
            const expectedHash = 'abcdef123456';
            fs.writeFileSync(path.join(tempDir, 'git_hash'), expectedHash + '\n');

            const result = getGitReleaseName(tempDir, true);

            expect(result).toEqual(expectedHash);
            // Ensure git command was not called
            expect(spyOnExecSync).not.toHaveBeenCalled();
        });

        it('should use git command if not in dist mode but in git repo', () => {
            // We need to ensure a fake .git directory exists in the current directory
            // since that's what the function checks for
            const prevDir = process.cwd();
            try {
                process.chdir(tempDir);
                const gitDir = path.join(tempDir, '.git');
                fs.mkdirSync(gitDir, {recursive: true});

                // Create a mock implementation for execSync
                const expectedHash = 'abcdef123456';
                spyOnExecSync.mockReturnValue(Buffer.from(expectedHash + '\n'));

                // Run the test
                const result = getGitReleaseName(tempDir, false);

                // Verify expectations
                expect(spyOnExecSync).toHaveBeenCalledWith('git rev-parse HEAD');
                expect(result).toEqual(expectedHash);
            } finally {
                process.chdir(prevDir);
            }
        });

        it('should return a placeholder message if no git info available', () => {
            // No git_hash file and no .git directory
            const prevDir = process.cwd();
            try {
                process.chdir(tempDir);
                const result = getGitReleaseName(tempDir, false);
                expect(result).toEqual('<no git hash found>');
            } finally {
                process.chdir(prevDir);
            }
        });
    });

    describe('getReleaseBuildNumber', () => {
        // Create a temporary directory for each test
        let tempDir: string;

        beforeEach(() => {
            tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'ce-release-test-'));
        });

        afterEach(() => {
            // Clean up the temporary directory
            fs.rmSync(tempDir, {recursive: true, force: true});
        });

        it('should read from release_build in dist mode', () => {
            // Create the release_build file with a known build number
            const expectedBuild = '12345';
            fs.writeFileSync(path.join(tempDir, 'release_build'), expectedBuild + '\n');

            const result = getReleaseBuildNumber(tempDir, true);

            expect(result).toEqual(expectedBuild);
        });

        it('should return empty string if no release build info available', () => {
            // No release_build file
            const result = getReleaseBuildNumber(tempDir, false);

            expect(result).toEqual('');
        });
    });

    describe('detectWsl', () => {
        const originalPlatform = process.platform;
        let platform: string;

        beforeEach(() => {
            vi.spyOn(child_process, 'execSync');
            Object.defineProperty(process, 'platform', {
                get: () => platform,
            });
        });

        afterEach(() => {
            vi.restoreAllMocks();
            platform = originalPlatform;
        });

        it('should detect WSL on Linux with Microsoft in uname', () => {
            platform = 'linux';
            vi.mocked(child_process.execSync).mockReturnValue(
                Buffer.from('Linux hostname 5.10.16.3-microsoft-standard-WSL2'),
            );

            expect(detectWsl()).toBe(true);
            expect(child_process.execSync).toHaveBeenCalledWith('uname -a');
        });

        it('should return false on Linux without Microsoft in uname', () => {
            platform = 'linux';
            vi.mocked(child_process.execSync).mockReturnValue(Buffer.from('Linux hostname 5.10.0-generic'));

            expect(detectWsl()).toBe(false);
            expect(child_process.execSync).toHaveBeenCalledWith('uname -a');
        });

        it('should return false on non-Linux platforms', () => {
            platform = 'win32';

            expect(detectWsl()).toBe(false);
            expect(child_process.execSync).not.toHaveBeenCalled();
        });

        it('should handle errors while detecting', () => {
            platform = 'linux';
            vi.mocked(child_process.execSync).mockImplementation(() => {
                throw new Error('Command failed');
            });

            expect(detectWsl()).toBe(false);
        });
    });

    describe('setupTempDir', () => {
        let originalEnv: NodeJS.ProcessEnv;

        beforeEach(() => {
            originalEnv = {...process.env};
            vi.spyOn(child_process, 'execSync');
        });

        afterEach(() => {
            process.env = originalEnv;
            vi.restoreAllMocks();
        });

        it('should set TMP env var with tmpDir option on non-WSL', () => {
            // Skip on Windows as it has different environment variable defaults
            if (process.platform === 'win32') return;

            const options = {
                tmpDir: '/custom/tmp',
            } as any;

            setupTempDir(options, false);

            expect(process.env.TMP).toEqual('/custom/tmp');
            expect(process.env.TEMP).toBeUndefined();
        });

        it('should set TEMP env var with tmpDir option on WSL', () => {
            // Skip on Windows as it has different environment variable defaults
            if (process.platform === 'win32') return;

            const options = {
                tmpDir: '/custom/tmp',
            } as any;

            setupTempDir(options, true);

            expect(process.env.TEMP).toEqual('/custom/tmp');
            expect(process.env.TMP).toEqual(originalEnv.TMP);
        });

        it('should try to use Windows TEMP on WSL without tmpDir option', () => {
            // Skip on Windows due to path separator differences (we ironically test this from linux)
            if (process.platform === 'win32') return;

            const options = {} as any;
            vi.mocked(child_process.execSync).mockReturnValue(Buffer.from('C:\\Users\\user\\AppData\\Local\\Temp\n'));

            setupTempDir(options, true);

            expect(process.env.TEMP).toEqual('/mnt/c/Users/user/AppData/Local/Temp');
            expect(child_process.execSync).toHaveBeenCalledWith('cmd.exe /c echo %TEMP%');
        });
    });

    describe('convertOptionsToAppArguments', () => {
        it('should convert command-line options to AppArguments', () => {
            const options = {
                rootDir: './etc',
                env: ['dev'],
                host: 'localhost',
                port: 10240,
                language: ['cpp'],
                cache: true,
                remoteFetch: true,
                ensureNoIdClash: true,
                suppressConsoleLog: false,
                extraField: 'should be ignored',
            } as any;

            const gitReleaseName = 'abc123';
            const releaseBuildNumber = '456';

            const result = convertOptionsToAppArguments(options, gitReleaseName, releaseBuildNumber);

            expect(result).toEqual({
                rootDir: './etc',
                env: ['dev'],
                hostname: 'localhost',
                port: 10240,
                gitReleaseName: 'abc123',
                releaseBuildNumber: '456',
                wantedLanguages: ['cpp'],
                doCache: true,
                fetchCompilersFromRemote: true,
                ensureNoCompilerClash: true,
                suppressConsoleLog: false,
            });
        });
    });

    describe('parseCommandLine', () => {
        // Integration tests for command-line parsing
        it('should parse basic command-line args', () => {
            const argv = ['node', 'app.js', '--port', '1234', '--debug'];

            const result = parseCommandLine(argv);

            expect(result.port).toEqual(1234);
            expect(result.debug).toBe(true);
            expect(result.env).toEqual(['dev']); // Default value
        });

        it('should parse array options', () => {
            const argv = ['node', 'app.js', '--env', 'prod', 'beta', '--language', 'cpp', 'rust'];

            const result = parseCommandLine(argv);

            expect(result.env).toEqual(['prod', 'beta']);
            expect(result.language).toEqual(['cpp', 'rust']);
        });

        it('should handle negated options', () => {
            const argv = ['node', 'app.js', '--no-cache', '--no-local'];

            const result = parseCommandLine(argv);

            expect(result.cache).toBe(false);
            expect(result.local).toBe(false);
        });

        it('should handle long option names with dashes', () => {
            const argv = ['node', 'app.js', '--root-dir', '/custom/path', '--metrics-port', '9000'];

            const result = parseCommandLine(argv);

            expect(result.rootDir).toEqual('/custom/path');
            expect(result.metricsPort).toEqual(9000);
        });
    });
});
