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
import os from 'node:os';
import path from 'node:path';

import {afterEach, describe, expect, it} from 'vitest';

import * as temp from '../lib/temp.js';
import * as utils from '../lib/utils.js';

describe('Creates and tracks temporary directories', () => {
    const osTemp = os.tmpdir();
    afterEach(async () => {
        await temp.cleanup();
        temp.resetStats();
    });

    it('creates directories under $TMPDIR', async () => {
        expect(temp.getStats()).toEqual({
            numCreated: 0,
            numActive: 0,
            numHeld: 0,
            numRemoved: 0,
            numAlreadyGone: 0,
            numHoldsExpired: 0,
        });
        const newTemp = await temp.mkdir('prefix');
        expect(newTemp).toContain(osTemp);
        expect(await utils.dirExists(newTemp)).toBe(true);
        expect(temp.getStats()).toEqual({
            numCreated: 1,
            numActive: 1,
            numHeld: 0,
            numRemoved: 0,
            numAlreadyGone: 0,
            numHoldsExpired: 0,
        });
    });
    it('creates directories with prefix', async () => {
        const newTemp = await temp.mkdir('prefix');
        expect(newTemp).toContain('prefix');
    });
    it('creates uniquely-named directories', async () => {
        const temp1 = await temp.mkdir('prefix');
        const temp2 = await temp.mkdir('prefix');
        const temp3 = await temp.mkdir('prefix');
        expect(temp1).not.toEqual(temp2);
        expect(temp1).not.toEqual(temp3);
        expect(temp2).not.toEqual(temp3);
        expect(temp.getStats()).toEqual({
            numCreated: 3,
            numActive: 3,
            numHeld: 0,
            numRemoved: 0,
            numAlreadyGone: 0,
            numHoldsExpired: 0,
        });
    });
    it('cleans up directories even if not empty', async () => {
        const newTemp1 = await temp.mkdir('prefix');
        await utils.ensureFileExists(path.join(newTemp1, 'some', 'dirs', 'under', 'file'));
        const newTemp2 = await temp.mkdir('prefix');
        const newTemp3 = await temp.mkdir('prefix');
        expect(temp.getStats()).toEqual({
            numCreated: 3,
            numActive: 3,
            numHeld: 0,
            numRemoved: 0,
            numAlreadyGone: 0,
            numHoldsExpired: 0,
        });
        await temp.cleanup();
        expect(temp.getStats()).toEqual({
            numCreated: 3,
            numActive: 0,
            numHeld: 0,
            numRemoved: 3,
            numAlreadyGone: 0,
            numHoldsExpired: 0,
        });
        expect(await utils.dirExists(newTemp1)).toBe(false);
        expect(await utils.dirExists(newTemp2)).toBe(false);
        expect(await utils.dirExists(newTemp3)).toBe(false);
    });
    it('counts already-cleaned-up directiories', async () => {
        const newTemp = await temp.mkdir('prefix');
        await fs.rm(newTemp, {recursive: true});
        expect(temp.getStats()).toEqual({
            numCreated: 1,
            numActive: 1,
            numHeld: 0,
            numRemoved: 0,
            numAlreadyGone: 0,
            numHoldsExpired: 0,
        });
        await temp.cleanup();
        expect(temp.getStats()).toEqual({
            numCreated: 1,
            numActive: 0,
            numHeld: 0,
            numRemoved: 0,
            numAlreadyGone: 1,
            numHoldsExpired: 0,
        });
    });

    it('uses absolute paths directly when provided', async () => {
        const customBase = await fs.mkdtemp(path.join(osTemp, 'custom-base-'));
        try {
            const absolutePrefix = path.join(customBase, 'myprefix');
            const newTemp = await temp.mkdir(absolutePrefix);
            expect(newTemp).toContain(customBase);
            expect(newTemp).toContain('myprefix');
            expect(newTemp).not.toContain(path.join(osTemp, customBase));
            expect(await utils.dirExists(newTemp)).toBe(true);
        } finally {
            await fs.rm(customBase, {recursive: true, force: true});
        }
    });

    it('uses absolute paths directly for mkdirSync', async () => {
        const customBase = await fs.mkdtemp(path.join(osTemp, 'custom-base-'));
        try {
            const absolutePrefix = path.join(customBase, 'syncprefix');
            const newTemp = temp.mkdirSync(absolutePrefix);
            expect(newTemp).toContain(customBase);
            expect(newTemp).toContain('syncprefix');
            expect(await utils.dirExists(newTemp)).toBe(true);
        } finally {
            await fs.rm(customBase, {recursive: true, force: true});
        }
    });
});

describe('cleanupSync', () => {
    it('synchronously removes pending directories', async () => {
        const dir = await temp.mkdir('ce-temp-test');
        expect(await utils.dirExists(dir)).toBe(true);
        temp.cleanupSync();
        expect(await utils.dirExists(dir)).toBe(false);
        expect(temp.getStats().numActive).toEqual(0);
    });

    it('removes a held directory, nothing being able to use one as the process exits', async () => {
        const dir = await temp.mkdir('ce-temp-test');
        temp.hold(dir);
        temp.cleanupSync();
        expect(await utils.dirExists(dir)).toBe(false);
        expect(temp.getStats().numHeld).toEqual(0);
    });
});

describe('Holds directories in use back from cleanup', () => {
    afterEach(async () => {
        temp.setMaxHoldMs(10 * 60 * 1000);
        await temp.cleanup();
        temp.resetStats();
    });

    it('keeps a held directory, and removes it once released', async () => {
        const held = await temp.mkdir('ce-held');
        const free = await temp.mkdir('ce-free');
        const release = temp.hold(held);

        await temp.cleanup();
        expect(await utils.dirExists(held)).toBe(true);
        expect(await utils.dirExists(free)).toBe(false);
        // Still tracked, so the cleanup after the release finds it.
        expect(temp.getStats()).toMatchObject({numActive: 1, numHeld: 1});

        release();
        await temp.cleanup();
        expect(await utils.dirExists(held)).toBe(false);
        expect(temp.getStats()).toMatchObject({numActive: 0, numHeld: 0});
    });

    it('releases by directory as well as through the returned function', async () => {
        const dir = await temp.mkdir('ce-held');
        temp.hold(dir);
        temp.release(dir);

        await temp.cleanup();
        expect(await utils.dirExists(dir)).toBe(false);
    });

    it('removes a directory whose hold was never released, once the hold lapses', async () => {
        // What guarantees the directory goes away even when the code holding it never gives it back.
        temp.setMaxHoldMs(0);
        const leaked = await temp.mkdir('ce-leaked');
        temp.hold(leaked);

        await temp.cleanup();
        expect(await utils.dirExists(leaked)).toBe(false);
        expect(temp.getStats()).toMatchObject({numActive: 0, numHeld: 0, numHoldsExpired: 1});
    });

    it('does not lapse a hold that is still within its time', async () => {
        temp.setMaxHoldMs(60_000);
        const dir = await temp.mkdir('ce-held');
        temp.hold(dir);

        await temp.cleanup();
        expect(await utils.dirExists(dir)).toBe(true);
        expect(temp.getStats().numHoldsExpired).toEqual(0);
    });
});
