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
        expect(temp.getStats()).toEqual({numCreated: 0, numActive: 0, numRemoved: 0, numAlreadyGone: 0});
        const newTemp = await temp.mkdir('prefix');
        expect(newTemp).toContain(osTemp);
        expect(await utils.dirExists(newTemp)).toBe(true);
        expect(temp.getStats()).toEqual({numCreated: 1, numActive: 1, numRemoved: 0, numAlreadyGone: 0});
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
        expect(temp.getStats()).toEqual({numCreated: 3, numActive: 3, numRemoved: 0, numAlreadyGone: 0});
    });
    it('cleans up directories even if not empty', async () => {
        const newTemp1 = await temp.mkdir('prefix');
        await utils.ensureFileExists(path.join(newTemp1, 'some', 'dirs', 'under', 'file'));
        const newTemp2 = await temp.mkdir('prefix');
        const newTemp3 = await temp.mkdir('prefix');
        expect(temp.getStats()).toEqual({numCreated: 3, numActive: 3, numRemoved: 0, numAlreadyGone: 0});
        await temp.cleanup();
        expect(temp.getStats()).toEqual({numCreated: 3, numActive: 0, numRemoved: 3, numAlreadyGone: 0});
        expect(await utils.dirExists(newTemp1)).toBe(false);
        expect(await utils.dirExists(newTemp2)).toBe(false);
        expect(await utils.dirExists(newTemp3)).toBe(false);
    });
    it('counts already-cleaned-up directiories', async () => {
        const newTemp = await temp.mkdir('prefix');
        await fs.rm(newTemp, {recursive: true});
        expect(temp.getStats()).toEqual({numCreated: 1, numActive: 1, numRemoved: 0, numAlreadyGone: 0});
        await temp.cleanup();
        expect(temp.getStats()).toEqual({numCreated: 1, numActive: 0, numRemoved: 0, numAlreadyGone: 1});
    });
});
