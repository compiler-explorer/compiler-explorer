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

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {logger} from './logger.js';
import * as utils from './utils.js';

const pendingRemoval: string[] = [];
export type Stats = {
    numCreated: number;
    numActive: number;
    numRemoved: number;
    numAlreadyGone: number;
};

const stats = {
    numCreated: 0,
    numRemoved: 0,
    numAlreadyGone: 0,
};

/**
 * Get the current stats for temporary directories.
 */
export function getStats(): Stats {
    return {
        ...stats,
        numActive: pendingRemoval.length,
    };
}

// Reset stats, for tests only.
export function resetStats() {
    stats.numCreated = 0;
    stats.numRemoved = 0;
    stats.numAlreadyGone = 0;
}

/**
 * Create a temporary directory, always in the operating systems' temporary directory.
 * @param prefix a prefix for the directory name
 */
export async function mkdir(prefix: string) {
    const result = await fs.promises.mkdtemp(path.join(os.tmpdir(), prefix));
    ++stats.numCreated;
    pendingRemoval.push(result);
    return result;
}

/**
 * Synchronously create a temporary directory, always in the operating systems' temporary directory.
 * @param prefix a prefix for the directory name
 */
export function mkdirSync(prefix: string) {
    const result = fs.mkdtempSync(path.join(os.tmpdir(), prefix));
    ++stats.numCreated;
    pendingRemoval.push(result);
    return result;
}

/**
 * Remove all temporary directories created by this module.
 */
export async function cleanup() {
    // "Atomically" take a copy of the things to remove and set it to an empty array.
    const toRemove = pendingRemoval.splice(0, pendingRemoval.length);
    let numRemoved = 0;
    let numAlreadyGone = 0;
    for (const dir of toRemove) {
        if (!(await utils.dirExists(dir))) {
            ++stats.numAlreadyGone;
            ++numAlreadyGone;
            continue;
        }
        try {
            await fs.promises.rm(dir, {recursive: true, force: true});
            ++numRemoved;
            ++stats.numRemoved;
        } catch (e) {
            logger.error(`Failed to remove ${dir}: ${e}`);
        }
    }
    logger.debug(`Removed ${numRemoved} (${numAlreadyGone} already gone) of ${toRemove.length} temporary directories`);
}

process.on('exit', async () => {
    await cleanup();
});
