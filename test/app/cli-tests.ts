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
import path from 'node:path';
import {afterEach, beforeEach, describe, expect, it, vi} from 'vitest';
import {
    convertOptionsToAppArguments,
    detectWsl,
    getGitReleaseName,
    getReleaseBuildNumber,
    setupTempDir,
} from '../../lib/app/cli.js';

describe('CLI Module', () => {
    describe('getGitReleaseName', () => {
        beforeEach(() => {
            vi.mock('node:fs', () => ({
                existsSync: vi.fn(),
                readFileSync: vi.fn(),
            }));
            vi.spyOn(child_process, 'execSync');
        });

        afterEach(() => {
            vi.restoreAllMocks();
            vi.resetModules();
        });

        it('should read from git_hash in dist mode', async () => {
            const fs = await import('node:fs');
            const distPath = '/path/to/dist';
            const gitHashPath = path.join(distPath, 'git_hash');
            const expectedHash = 'abcdef123456';

            vi.mocked(fs.existsSync).mockReturnValue(true);
            vi.mocked(fs.readFileSync).mockReturnValue(Buffer.from(expectedHash + '\n'));

            const result = getGitReleaseName(distPath, true);

            expect(fs.existsSync).toHaveBeenCalledWith(gitHashPath);
            expect(fs.readFileSync).toHaveBeenCalledWith(gitHashPath);
            expect(result).toEqual(expectedHash);
        });

        it('should use git command if not in dist mode but in git repo', async () => {
            const fs = await import('node:fs');
            const distPath = '/path/to/dist';
            const expectedHash = 'abcdef123456';

            vi.mocked(fs.existsSync).mockImplementation(filePath => {
                const pathStr = filePath.toString();
                if (pathStr === path.join(distPath, 'git_hash')) return false;
                if (pathStr === '.git/') return true;
                return false;
            });
            vi.mocked(child_process.execSync).mockReturnValue(Buffer.from(expectedHash + '\n'));

            const result = getGitReleaseName(distPath, false);

            expect(fs.existsSync).toHaveBeenCalledWith('.git/');
            expect(child_process.execSync).toHaveBeenCalledWith('git rev-parse HEAD');
            expect(result).toEqual(expectedHash);
        });

        it('should return empty string if no git info available', async () => {
            const fs = await import('node:fs');
            const distPath = '/path/to/dist';

            vi.mocked(fs.existsSync).mockReturnValue(false);

            const result = getGitReleaseName(distPath, false);

            expect(result).toEqual('');
        });
    });

    describe('getReleaseBuildNumber', () => {
        beforeEach(() => {
            vi.mock('node:fs', () => ({
                existsSync: vi.fn(),
                readFileSync: vi.fn(),
            }));
        });

        afterEach(() => {
            vi.restoreAllMocks();
            vi.resetModules();
        });

        it('should read from release_build in dist mode', async () => {
            const fs = await import('node:fs');
            const distPath = '/path/to/dist';
            const releaseBuildPath = path.join(distPath, 'release_build');
            const expectedBuild = '12345';

            vi.mocked(fs.existsSync).mockReturnValue(true);
            vi.mocked(fs.readFileSync).mockReturnValue(Buffer.from(expectedBuild + '\n'));

            const result = getReleaseBuildNumber(distPath, true);

            expect(fs.existsSync).toHaveBeenCalledWith(releaseBuildPath);
            expect(fs.readFileSync).toHaveBeenCalledWith(releaseBuildPath);
            expect(result).toEqual(expectedBuild);
        });

        it('should return empty string if no release build info available', async () => {
            const fs = await import('node:fs');
            const distPath = '/path/to/dist';

            vi.mocked(fs.existsSync).mockReturnValue(false);

            const result = getReleaseBuildNumber(distPath, false);

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
            const options = {
                tmpDir: '/custom/tmp',
            } as any;

            setupTempDir(options, false);

            expect(process.env.TMP).toEqual('/custom/tmp');
            expect(process.env.TEMP).toBeUndefined();
        });

        it('should set TEMP env var with tmpDir option on WSL', () => {
            const options = {
                tmpDir: '/custom/tmp',
            } as any;

            setupTempDir(options, true);

            expect(process.env.TEMP).toEqual('/custom/tmp');
            expect(process.env.TMP).toEqual(originalEnv.TMP);
        });

        it('should try to use Windows TEMP on WSL without tmpDir option', () => {
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
});
