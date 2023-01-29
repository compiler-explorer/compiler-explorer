// Copyright (c) 2020, Compiler Explorer Authors
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

import {executionAsyncId} from 'async_hooks';

import {default as Queue} from 'p-queue';
import PromClient from 'prom-client';

import {assert} from './assert';

// globals as essentially the compilation queue is a singleton, and if we make them members of the queue, tests fail as
// when we create a second queue, the previous counters are still registered.
const queueEnqueued = new PromClient.Counter({
    name: 'ce_compilation_queue_enqueued_total',
    help: 'Total number of jobs enqueued',
});
const queueDequeued = new PromClient.Counter({
    name: 'ce_compilation_queue_dequeued_total',
    help: 'Total number of jobs dequeued',
});
const queueCompleted = new PromClient.Counter({
    name: 'ce_compilation_queue_completed_total',
    help: 'Total number of jobs completed',
});

export type Job<TaskResultType> = () => PromiseLike<TaskResultType>;

export class CompilationQueue {
    private readonly _running: Set<number> = new Set();
    // Track both when we queue jobs and start running them
    private readonly _outstanding: Record<number, number> = {};
    private readonly _queue: Queue;

    constructor(concurrency: number, timeout: number) {
        this._queue = new Queue({
            concurrency,
            timeout,
            throwOnTimeout: true,
        });
    }

    static fromProps(ceProps) {
        return new CompilationQueue(ceProps('maxConcurrentCompiles', 1), ceProps('compilationEnvTimeoutMs'));
    }

    markOutstanding(id: number) {
        assert(!(id in this._outstanding));
        this._outstanding[id] = Date.now();
    }

    resolveOutstanding(id: number) {
        assert(id in this._outstanding);
        delete this._outstanding[id];
    }

    enqueue<Result>(job: Job<Result>): PromiseLike<Result> {
        const enqueueAsyncId = executionAsyncId();
        // If we're asked to enqueue a job when we're already in a async queued job context, just run it.
        // This prevents a deadlock.
        if (this._running.has(enqueueAsyncId)) return job();
        queueEnqueued.inc();
        this.markOutstanding(enqueueAsyncId);
        return this._queue.add(() => {
            queueDequeued.inc();
            this.resolveOutstanding(enqueueAsyncId);
            const jobAsyncId = executionAsyncId();
            if (this._running.has(jobAsyncId)) throw new Error('somehow we entered the context twice');
            try {
                this._running.add(jobAsyncId);
                this.markOutstanding(jobAsyncId);
                return job();
            } finally {
                this._running.delete(jobAsyncId);
                queueCompleted.inc();
                this.resolveOutstanding(jobAsyncId);
            }
        });
    }

    status(): {busy: boolean; pending: number; size: number} {
        const pending = this._queue.pending;
        const size = this._queue.size;
        return {
            busy: pending > 0 || size > 0,
            pending,
            size,
        };
    }

    // Find the longest amount of time anything currently running or waiting to run is running/waiting
    longestOutstanding() {
        const timestamps = Object.values(this._outstanding);
        if (timestamps.length === 0) {
            return 0;
        } else {
            const now = Date.now();
            return Math.max(...timestamps.map(timestamp => now - timestamp));
        }
    }
}
