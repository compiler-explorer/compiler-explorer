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
import process from 'node:process';

import {logger} from '../logger.js';

/**
 * Set up temporary directory, especially for WSL environments
 * @param tmpDir - Optional path to use as temporary directory
 * @param isWsl - Whether running under Windows Subsystem for Linux
 */
export function setupTempDir(tmpDir: string | undefined, isWsl: boolean): void {
    // If a tempDir is supplied, use it
    if (tmpDir) {
        if (isWsl) {
            process.env.TEMP = tmpDir; // for Windows
        } else {
            process.env.TMP = tmpDir; // for Linux
        }
    }
    // If running under WSL without explicit tmpDir, try to use Windows %TEMP%
    else if (isWsl) {
        try {
            const windowsTemp = child_process.execSync('cmd.exe /c echo %TEMP%').toString().replaceAll('\\', '/');
            const driveLetter = windowsTemp.substring(0, 1).toLowerCase();
            const directoryPath = windowsTemp.substring(2).trim();
            process.env.TEMP = path.join('/mnt', driveLetter, directoryPath);
        } catch {
            logger.warn('Unable to invoke cmd.exe to get windows %TEMP% path.');
        }
    }
    logger.info(`Using temporary dir: ${process.env.TEMP || process.env.TMP}`);
}
