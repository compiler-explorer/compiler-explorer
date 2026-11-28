// Copyright (c) 2026, Compiler Explorer Authors
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

// Code run inside a jail can leave anything it likes in the directory bound into it, and the server reads that
// directory once the jail exits. Most of what it could leave is harmless to read: a directory fails with EISDIR, a
// socket with ENXIO, and the jail cannot make device nodes. A FIFO is the exception, since opening one for reading
// blocks until something writes to it, and nothing ever will: the read never finishes and takes a libuv thread with
// it. Rather than have every reader of every output remember to check, the directory is swept here, once, after the
// jailed runs in it have finished and before any of their callers sees a result.
//
// Several jails run in the same directory at once (the compile, the AST/IR/opt-pipeline views and the independent
// tools are started together), so a sweep when one of them exits proves nothing while another is still running and
// can still make a FIFO. Each run's result is therefore held until no jailed run is left in its directory and that
// directory has been swept since the last of them started.
//
// Symbolic links are left alone: later jailed steps rely on them (library SONAME links, build-tool links into the
// tree), and the server's own reads are kept from following them by mounting the temp root nosymfollow.

import type {Dir, Dirent} from 'node:fs';
import fs from 'node:fs/promises';
import path from 'node:path';

import {logger} from './logger.js';
import * as temp from './temp.js';

export type SweepLimits = {
    /** Entries (of any type) the walk will look at before giving up. */
    maxEntries: number;
    /** Directory nesting below the root the walk will descend to before giving up. */
    maxDepth: number;
};

// Walking 100,000 entries took about a tenth of a second in testing (warm cache), and a request directory normally
// holds a handful. A tree past these limits is failed rather than walked.
export const defaultSweepLimits: SweepLimits = {maxEntries: 100_000, maxDepth: 64};

export class RequestDirSweepError extends Error {
    constructor(message: string) {
        super(message);
        this.name = 'RequestDirSweepError';
    }
}

/** Paths the server put in a request directory itself and needs to keep across a sweep. */
const exempt = new Set<string>();

/**
 * Keep `filePath` across sweeps until the returned function is called. For something the server itself places in a
 * request directory on purpose and still needs once a jailed run there has exited, such as a FIFO it is reading the
 * run's trace through.
 */
export function exemptFromSweep(filePath: string): () => void {
    const resolved = path.resolve(filePath);
    exempt.add(resolved);
    return () => exempt.delete(resolved);
}

function isRemovable(entry: Dirent) {
    return entry.isFIFO() || entry.isSocket() || entry.isBlockDevice() || entry.isCharacterDevice();
}

function errorCode(e: unknown): string | undefined {
    return (e as NodeJS.ErrnoException)?.code;
}

/**
 * Run `op`, and if it is refused for want of permission, give the owner full rights to `dir` (and to `parent`, whose
 * search permission reaching `dir` needs) and try once more. Code in the jail runs as the same user as the server
 * outside it, so it can make a directory the server can't read or write to, but cannot make one the server can't take
 * back.
 */
async function withAccessTo<T>(dir: string, parent: string | undefined, op: () => Promise<T>): Promise<T> {
    try {
        return await op();
    } catch (e) {
        if (errorCode(e) !== 'EACCES' && errorCode(e) !== 'EPERM') throw e;
    }
    if (parent) await fs.chmod(parent, 0o700);
    await fs.chmod(dir, 0o700);
    return await op();
}

/**
 * Remove every FIFO, socket and device node below `root`, without opening, or following a link to, anything. Throws
 * `RequestDirSweepError` if the tree is bigger than `limits` allow or an entry can't be dealt with. A missing root is
 * not an error: there is nothing in it to read.
 */
export async function sweepNonRegularEntries(root: string, limits: SweepLimits = defaultSweepLimits) {
    let seen = 0;
    const pending: {dir: string; parent: string | undefined; depth: number}[] = [
        {dir: root, parent: undefined, depth: 0},
    ];
    while (pending.length > 0) {
        const {dir, parent, depth} = pending.pop()!;
        let handle: Dir;
        try {
            handle = await withAccessTo(dir, parent, () => fs.opendir(dir));
        } catch (e) {
            // Gone already: the jail is no longer running, so this is only the directory being removed under us.
            if (errorCode(e) === 'ENOENT') continue;
            throw new RequestDirSweepError(`Could not read ${dir}: ${e}`);
        }
        try {
            // Read as a stream, so a directory with millions of entries costs no more than the limit before failing.
            // Leaving the loop, by any route, closes the handle.
            for await (const entry of handle) {
                if (++seen > limits.maxEntries) {
                    throw new RequestDirSweepError(`More than ${limits.maxEntries} entries in ${root}`);
                }
                const entryPath = path.join(dir, entry.name);
                // A Dirent describes the entry itself, so a symlink to a directory is never descended into.
                if (entry.isDirectory()) {
                    if (depth + 1 > limits.maxDepth) {
                        throw new RequestDirSweepError(`Directories in ${root} nest deeper than ${limits.maxDepth}`);
                    }
                    pending.push({dir: entryPath, parent: dir, depth: depth + 1});
                } else if (isRemovable(entry) && !exempt.has(entryPath)) {
                    try {
                        await withAccessTo(dir, parent, () => fs.unlink(entryPath));
                    } catch (e) {
                        if (errorCode(e) === 'ENOENT') continue;
                        throw new RequestDirSweepError(`Could not remove ${entryPath}: ${e}`);
                    }
                }
            }
        } catch (e) {
            if (e instanceof RequestDirSweepError) throw e;
            throw new RequestDirSweepError(`Could not read ${dir}: ${e}`);
        }
    }
}

/**
 * The per-request temporary directory that `boundDir` lies in, or undefined if it isn't in one. Request directories
 * are made directly under the temp root, and everything bound into a jail for a request lies inside one of them.
 * Anything else (the temp root itself, a compiler's own install directory) is never swept.
 */
export function requestDirFor(boundDir: string, tempRoot: string = temp.getTempRoot()): string | undefined {
    const relative = path.relative(path.resolve(tempRoot), path.resolve(boundDir));
    if (relative === '' || relative.startsWith('..') || path.isAbsolute(relative)) return undefined;
    return path.join(path.resolve(tempRoot), relative.split(path.sep)[0]);
}

type Gate = {
    /** Runs admitted to this directory that have not exited, including any waiting for a sweep to finish to start. */
    running: number;
    /** The sweep under way, if any. Runs arriving during one start once it is done, so it never removes their files. */
    sweeping: Promise<void> | undefined;
    /** Callers whose run has exited, waiting for the directory to be quiet and swept. */
    waiting: ((error: Error | undefined) => void)[];
};

const gates = new Map<string, Gate>();

/** For tests. */
export function numActiveGates() {
    return gates.size;
}

async function sweepThenRelease(dir: string, gate: Gate, limits: SweepLimits) {
    let error: Error | undefined;
    try {
        await sweepNonRegularEntries(dir, limits);
    } catch (e) {
        error = e instanceof Error ? e : new Error(String(e));
    }
    gate.sweeping = undefined;

    // Runs that arrived during the walk start now; when the last of them exits the directory is swept again and
    // everyone, including those already waiting, is released then.
    if (gate.running > 0) return;

    gates.delete(dir);
    if (error) logger.warn(`Sweep of ${dir} failed: ${error.message}`);
    for (const release of gate.waiting) release(error);
}

/**
 * Run `run`, which writes to `boundDir`, and return what it returns once no other gated run is left in the same request
 * directory and that directory holds no FIFOs, sockets or device nodes. If the directory can't be swept, every run
 * waiting on it fails with a `RequestDirSweepError` instead of returning, so nothing goes on to read from it. A
 * `boundDir` outside the temp root's request directories is run with no sweep.
 *
 * `run` must finish in bounded time, since every other caller in the directory is held until it does, and must not
 * itself await another `runThenSweep` in the same directory: that one would wait for `run` to exit, and so would never
 * return.
 */
export async function runThenSweep<T>(
    boundDir: string | undefined,
    run: () => Promise<T>,
    limits: SweepLimits = defaultSweepLimits,
): Promise<T> {
    const dir = boundDir ? requestDirFor(boundDir) : undefined;
    if (!dir) return await run();

    let gate = gates.get(dir);
    if (!gate) {
        gate = {running: 0, sweeping: undefined, waiting: []};
        gates.set(dir, gate);
    }
    ++gate.running;
    // No sweep can start while this run is counted, so once this one is done the directory is ours until we exit.
    if (gate.sweeping) await gate.sweeping;

    let outcome: {ok: true; value: T} | {ok: false; error: unknown};
    try {
        outcome = {ok: true, value: await run()};
    } catch (error) {
        outcome = {ok: false, error};
    }

    --gate.running;
    const swept = new Promise<Error | undefined>(resolve => gate.waiting.push(resolve));
    if (gate.running === 0 && !gate.sweeping) gate.sweeping = sweepThenRelease(dir, gate, limits);
    const sweepError = await swept;

    // Checked first, so that a caller which turns a failed run into an ordinary result still sees this one.
    if (sweepError) throw new RequestDirSweepError(`Could not clean up after the run: ${sweepError.message}`);
    if (!outcome.ok) throw outcome.error;
    return outcome.value;
}
