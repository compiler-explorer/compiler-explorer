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

import {describe, expect, it} from 'vitest';

import {CompilationQueue} from '../lib/compilation-queue.js';

describe('CompilationQueue', () => {
    it('runs an enqueued job and returns its result', async () => {
        const queue = new CompilationQueue(1, 1000, 1000);
        await expect(queue.enqueue(async () => 42)).resolves.toEqual(42);
        expect(queue.status().busy).toBe(false);
    });

    it('does not deadlock when a job enqueues another job', async () => {
        const queue = new CompilationQueue(1, 1000, 1000);
        const result = await queue.enqueue(() => {
            return queue.enqueue(async () => 42);
        });
        expect(result).toEqual(42);
    });

    it('times out a job that never settles, freeing the queue', async () => {
        // A job whose promise never settles (e.g. a download severed mid-stream, see #8811)
        // must not occupy a queue slot forever: the queue timeout has to reject it so the
        // queue can go back to being non-busy (which gates temp dir cleanup).
        const queue = new CompilationQueue(1, 100, 1000);
        const wedged = queue.enqueue(() => new Promise(() => {}));
        await expect(wedged).rejects.toThrow(/timed out/i);

        await expect(queue.enqueue(async () => 'still works')).resolves.toEqual('still works');
        expect(queue.status().busy).toBe(false);
        // Generous test budget: the 100ms queue timeout can fire very late when vitest workers
        // saturate the machine (e.g. during pre-commit runs).
    }, 15_000);
});
