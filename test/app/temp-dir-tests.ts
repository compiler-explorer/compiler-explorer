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
import os from 'node:os';
import process from 'node:process';

import {afterEach, beforeEach, describe, expect, it, vi} from 'vitest';

import {setupTempDir} from '../../lib/app/temp-dir.js';

const TEMP_VARS = ['TMPDIR', 'TMP', 'TEMP'] as const;

describe('setupTempDir', () => {
    let savedVars: Record<string, string | undefined>;

    beforeEach(() => {
        // Save and clear the individual variables rather than reassigning process.env
        // wholesale: a replaced process.env is a plain object whose writes no longer reach
        // the real environment, while os.tmpdir() reads the real environment.
        savedVars = Object.fromEntries(TEMP_VARS.map(v => [v, process.env[v]]));
        for (const v of TEMP_VARS) delete process.env[v];
        vi.spyOn(child_process, 'execSync');
    });

    afterEach(() => {
        for (const v of TEMP_VARS) {
            if (savedVars[v] === undefined) delete process.env[v];
            else process.env[v] = savedVars[v];
        }
        vi.restoreAllMocks();
    });

    it('should set all temp env vars with tmpDir option', () => {
        // Skip on Windows as it has different environment variable defaults
        if (process.platform === 'win32') return;

        setupTempDir('/custom/tmp', false);

        expect(process.env.TMPDIR).toEqual('/custom/tmp');
        expect(process.env.TMP).toEqual('/custom/tmp');
        expect(process.env.TEMP).toEqual('/custom/tmp');
    });

    it('should make os.tmpdir() return the configured dir even when TMPDIR was inherited', () => {
        // Skip on Windows, where os.tmpdir() does not consult TMPDIR
        if (process.platform === 'win32') return;

        process.env.TMPDIR = '/inherited/elsewhere';
        setupTempDir('/custom/tmp', false);

        expect(os.tmpdir()).toEqual('/custom/tmp');
    });

    it('should try to use Windows TEMP on WSL without tmpDir option', () => {
        // Skip on Windows due to path separator differences (we ironically test this from linux)
        if (process.platform === 'win32') return;

        vi.mocked(child_process.execSync).mockReturnValue(Buffer.from('C:\\Users\\user\\AppData\\Local\\Temp\n'));

        setupTempDir(undefined, true);

        expect(process.env.TEMP).toEqual('/mnt/c/Users/user/AppData/Local/Temp');
        expect(os.tmpdir()).toEqual('/mnt/c/Users/user/AppData/Local/Temp');
        expect(child_process.execSync).toHaveBeenCalledWith('cmd.exe /c echo %TEMP%');
    });
});
