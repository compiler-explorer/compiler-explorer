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

import {afterEach, beforeEach, describe, expect, it, vi} from 'vitest';

import {PersistentEventsSender} from '../lib/execution/events-websocket.js';

// A real socket is not needed: what is under test is which branch the ack timer takes, which
// depends only on the pending map, the retry count and the deadline.
function makeSender(): any {
    const sender = Object.create(PersistentEventsSender.prototype);
    sender.requireAcknowledgments = true;
    sender.isConnected = true;
    sender.maxRetries = 3;
    sender.ackTimeoutMs = 3000;
    sender.requestDeadlineMs = 60_000;
    sender.pendingAcks = new Map();
    sender.ws = {readyState: 1, send: vi.fn()};
    return sender;
}

function sendAwaitingAck(sender: any, guid: string, expiresAtMs?: number) {
    const rejected: Error[] = [];
    sender.setupAckTimeout(guid, {guid}, vi.fn(), (e: Error) => rejected.push(e), expiresAtMs);
    return rejected;
}

describe('Waiting for an acknowledgement that may never come', () => {
    beforeEach(() => vi.useFakeTimers());
    afterEach(() => vi.useRealTimers());

    // The subscriber is often a router that is reconnecting, and its subscriptions come back
    // with it - so an unacknowledged result is normally worth resending.
    it('keeps retrying while the requester could still be listening', () => {
        const sender = makeSender();
        const rejected = sendAwaitingAck(sender, 'guid', Date.now() + 60_000);

        vi.advanceTimersByTime(3000);

        expect(sender.ws.send).toHaveBeenCalled();
        expect(sender.getPendingAckCount()).toBe(1);
        expect(rejected).toHaveLength(0);
    });

    // Past the deadline there is nobody left to reconnect for, and the worker pulls no new work
    // while a result is outstanding - so further retries cost throughput and can never pay off.
    it('gives up once the requester has stopped waiting', () => {
        const sender = makeSender();
        const rejected = sendAwaitingAck(sender, 'guid', Date.now() + 1000);

        vi.advanceTimersByTime(3000);

        expect(sender.getPendingAckCount()).toBe(0);
        expect(rejected).toHaveLength(1);
        expect(rejected[0].message).toMatch(/deadline passed/);
        expect(sender.isReadyForNewMessages()).toBe(true);
    });

    it('still exhausts its retries when no deadline is known', () => {
        const sender = makeSender();
        const rejected = sendAwaitingAck(sender, 'guid', undefined);

        vi.advanceTimersByTime(3000);
        expect(sender.getPendingAckCount()).toBe(1);
        expect(rejected).toHaveLength(0);
    });

    // send() is where the queue's SentTimestamp becomes a deadline, so it is worth going
    // through it rather than only exercising the timer directly.
    it('derives the deadline from when the request was queued', async () => {
        const sender = makeSender();
        const expired = sender.send('old-guid', {code: 0} as any, Date.now() - 90_000);
        const outcome = expect(expired).rejects.toThrow(/deadline passed/);

        vi.advanceTimersByTime(3000);
        await outcome;
        expect(sender.getPendingAckCount()).toBe(0);
    });

    it('keeps its full budget for a request queued a moment ago', () => {
        const sender = makeSender();
        const rejected: Error[] = [];
        sender.send('fresh-guid', {code: 0} as any, Date.now()).catch((e: Error) => rejected.push(e));

        vi.advanceTimersByTime(3000);

        expect(sender.getPendingAckCount()).toBe(1);
        expect(rejected).toHaveLength(0);
    });

    it('an acknowledgement that does arrive is still honoured', () => {
        const sender = makeSender();
        const rejected = sendAwaitingAck(sender, 'guid', Date.now() + 1000);

        sender.handleAcknowledgment('guid');

        vi.advanceTimersByTime(10_000);
        expect(sender.getPendingAckCount()).toBe(0);
        expect(rejected).toHaveLength(0);
    });
});
