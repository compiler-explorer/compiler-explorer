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

/** Directories exempt from cleanup while something is using them, each with the time its exemption lapses. */
const heldUntil = new Map<string, number>();

/**
 * How long a hold lasts before cleanup reclaims the directory regardless. Reaching it means a release was missed, so
 * the directory is removed and the miss logged: leaking one until the process exits is the worse outcome. Comfortably
 * longer than `compilationEnvTimeoutMs`, which is what bounds how long legitimate work can hold one.
 */
let maxHoldMs = 10 * 60 * 1000;

/** For the app to align the ceiling with its compilation timeout, and for tests to make holds lapse at once. */
export function setMaxHoldMs(ms: number) {
    maxHoldMs = ms;
}

export type Stats = {
    numCreated: number;
    numActive: number;
    numHeld: number;
    numRemoved: number;
    numAlreadyGone: number;
    numHoldsExpired: number;
};

const stats = {
    numCreated: 0,
    numRemoved: 0,
    numAlreadyGone: 0,
    numHoldsExpired: 0,
};

/**
 * Get the current stats for temporary directories.
 */
export function getStats(): Stats {
    return {
        ...stats,
        numActive: pendingRemoval.length,
        numHeld: heldUntil.size,
    };
}

// Reset stats and any holds, for tests only.
export function resetStats() {
    stats.numCreated = 0;
    stats.numRemoved = 0;
    stats.numAlreadyGone = 0;
    stats.numHoldsExpired = 0;
    heldUntil.clear();
}

/**
 * Exempt a directory from cleanup for as long as something is using it, returning the release. Cleanup otherwise
 * removes every directory this module created the moment it runs, whatever is still reading from or writing to one.
 *
 * A directory being made for something that will use it wants `mkdir(prefix, {hold: true})` rather than this: holding
 * it after the fact leaves a gap in which cleanup can run.
 */
export function hold(dir: string): () => void {
    heldUntil.set(dir, Date.now() + maxHoldMs);
    return () => release(dir);
}

/** Give a directory back to cleanup. Harmless if it was never held or has already been removed. */
export function release(dir: string) {
    heldUntil.delete(dir);
}

/** Whether cleanup must leave this directory alone, lapsing a hold that has outlived its welcome. */
function isHeld(dir: string, now: number): boolean {
    const until = heldUntil.get(dir);
    if (until === undefined) return false;
    if (until > now) return true;

    heldUntil.delete(dir);
    ++stats.numHoldsExpired;
    logger.warn(`Hold on ${dir} was never released and has lapsed after ${maxHoldMs}ms; removing it anyway`);
    return false;
}

/**
 * The directory under which this module creates temporary directories (unless callers pass
 * an absolute prefix). The --tmp-dir command line option is not read directly: at startup
 * setupTempDir() exports it as $TMPDIR/$TMP/$TEMP, which os.tmpdir() consults.
 * See lib/app/temp-dir.ts.
 */
export function getTempRoot(): string {
    return os.tmpdir();
}

/**
 * Create a temporary directory. If the prefix is an absolute path, use it directly;
 * otherwise create the directory in the operating system's temporary directory.
 * @param prefix a prefix for the directory name, or an absolute path prefix
 */
export async function mkdir(prefix: string, options?: {hold?: boolean}) {
    const baseDir = path.isAbsolute(prefix) ? prefix : path.join(getTempRoot(), prefix);
    const result = await fs.promises.mkdtemp(baseDir);
    ++stats.numCreated;
    pendingRemoval.push(result);
    // Held here and not by the caller: returning from this is an await boundary, and a cleanup running in it would
    // take the directory before the caller's next statement could hold it.
    if (options?.hold) hold(result);
    return result;
}

/**
 * Synchronously create a temporary directory. If the prefix is an absolute path, use it directly;
 * otherwise create the directory in the operating system's temporary directory.
 * @param prefix a prefix for the directory name, or an absolute path prefix
 */
export function mkdirSync(prefix: string) {
    const baseDir = path.isAbsolute(prefix) ? prefix : path.join(getTempRoot(), prefix);
    const result = fs.mkdtempSync(baseDir);
    ++stats.numCreated;
    pendingRemoval.push(result);
    return result;
}

/**
 * Remove every temporary directory created by this module, except those held by something still using them. A held
 * directory stays tracked, so a later cleanup removes it once released.
 */
export async function cleanup() {
    // "Atomically" take a copy of the things to remove and set it to an empty array.
    const taken = pendingRemoval.splice(0, pendingRemoval.length);
    const toRemove: string[] = [];
    const now = Date.now();
    for (const dir of taken) {
        if (isHeld(dir, now)) pendingRemoval.push(dir);
        else toRemove.push(dir);
    }

    let numRemoved = 0;
    let numAlreadyGone = 0;
    for (const dir of toRemove) {
        // Rechecked because every await below is a chance for hold() to run: one taken as free may since have been
        // claimed by a request that arrived mid-cleanup.
        if (isHeld(dir, Date.now())) {
            pendingRemoval.push(dir);
            continue;
        }
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

/**
 * Synchronously remove all temporary directories created by this module; for use at process
 * exit, where asynchronous work never runs. Holds are ignored: nothing is still using a
 * directory once the process is on its way out, and this is the last chance to remove one.
 */
export function cleanupSync() {
    heldUntil.clear();
    const toRemove = pendingRemoval.splice(0, pendingRemoval.length);
    for (const dir of toRemove) {
        try {
            if (!fs.existsSync(dir)) {
                ++stats.numAlreadyGone;
                continue;
            }
            fs.rmSync(dir, {recursive: true, force: true});
            ++stats.numRemoved;
        } catch {
            // Best effort only: we may be partway through exiting.
        }
    }
}

process.on('exit', () => {
    cleanupSync();
});
