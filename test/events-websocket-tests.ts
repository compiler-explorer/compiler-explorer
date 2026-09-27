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

import {EventsWsWaiter, PersistentEventsSender} from '../lib/execution/events-websocket.js';

// The heartbeat only touches the socket and two timers, so a stand-in socket suffices.
function makeSender() {
    const sent: string[] = [];
    const sender = Object.create(PersistentEventsSender.prototype) as any;
    sender.heartbeatIntervalMs = 30_000;
    sender.pongTimeoutMs = 10_000;
    sender.lastActivityTime = Date.now();
    sender.pendingAcks = new Map();
    sender.isConnected = true;
    sender.ws = {
        readyState: 1,
        send: (data: string) => sent.push(data),
        terminate: vi.fn(),
        ping: vi.fn(),
    };
    return {sender, sent};
}

function makeSenderWithPending(guid: string, retryCount = 0) {
    const sent: string[] = [];
    const rejected: Error[] = [];
    const sender = Object.create(PersistentEventsSender.prototype) as any;
    sender.maxRetries = 3;
    sender.ackTimeoutMs = 3000;
    sender.stableConnectionMs = 30_000;
    sender.lastOpenedAt = 0;
    sender.reconnectAttempts = 0;
    sender.reconnectDelay = 1000;
    sender.ws = {readyState: 1, send: (d: string) => sent.push(d), terminate: vi.fn()};
    sender.pendingAcks = new Map([
        [
            guid,
            {
                timeout: setTimeout(() => {}, 60_000),
                retryCount,
                resolve: vi.fn(),
                reject: (e: Error) => rejected.push(e),
                messageData: {guid},
            },
        ],
    ]);
    return {sender, sent, rejected};
}

describe('Retrying an acknowledgement across reconnections', () => {
    beforeEach(() => vi.useFakeTimers());
    afterEach(() => vi.useRealTimers());

    // Reconnections arriving faster than ackTimeoutMs replace the timer before it fires, so
    // without spending budget here the same result is resent forever and pendingAcks never
    // empties - which keeps the worker from ever pulling new work again.
    it('spends a retry per reconnection', () => {
        const {sender} = makeSenderWithPending('guid');
        sender.retryPendingAcknowledgments();
        expect(sender.pendingAcks.get('guid').retryCount).toBe(1);
        sender.retryPendingAcknowledgments();
        expect(sender.pendingAcks.get('guid').retryCount).toBe(2);
    });

    it('gives up once the budget is spent, so the worker can take work again', () => {
        const {sender, rejected} = makeSenderWithPending('guid', 3);
        sender.retryPendingAcknowledgments();
        expect(sender.pendingAcks.size).toBe(0);
        expect(rejected).toHaveLength(1);
    });
});

describe('Reconnection budget', () => {
    beforeEach(() => vi.useFakeTimers());
    afterEach(() => vi.useRealTimers());

    it('is not reset by a connection that closed straight away', () => {
        const {sender} = makeSenderWithPending('guid');
        const first = Date.now();
        sender.noteConnectionOpened(first);
        sender.reconnectAttempts = 2;

        // Opened again a second later: that connection proved nothing.
        sender.noteConnectionOpened(first + 1000);

        expect(sender.reconnectAttempts).toBe(2);
    });

    it('is reset by a connection that lasted', () => {
        const {sender} = makeSenderWithPending('guid');
        const first = Date.now();
        sender.noteConnectionOpened(first);
        sender.reconnectAttempts = 2;

        sender.noteConnectionOpened(first + 30_000);

        expect(sender.reconnectAttempts).toBe(0);
    });
});

describe('Keeping the events websocket honest', () => {
    beforeEach(() => vi.useFakeTimers());
    afterEach(() => vi.useRealTimers());

    // API Gateway answers protocol pings at its edge, so ws.ping() proves nothing about
    // the Lambda backend.
    it('sends an application-level ping, not a protocol frame', () => {
        const {sender, sent} = makeSender();
        sender.sendHeartbeat();
        expect(sent).toEqual(['ping']);
        expect(sender.ws.ping).not.toHaveBeenCalled();
    });

    it('terminates a socket that answers nothing, so the reconnect path runs', () => {
        const {sender} = makeSender();
        sender.sendHeartbeat();
        expect(sender.ws.terminate).not.toHaveBeenCalled();

        vi.advanceTimersByTime(10_000);

        expect(sender.ws.terminate).toHaveBeenCalled();
    });

    it('leaves a socket alone when a pong comes back', () => {
        const {sender} = makeSender();
        sender.sendHeartbeat();
        sender.recordActivity();

        vi.advanceTimersByTime(30_000);

        expect(sender.ws.terminate).not.toHaveBeenCalled();
    });

    // Waiting specifically for a pong would terminate a connection that is plainly working.
    it('counts a delivered result as proof of life', () => {
        const {sender} = makeSender();
        sender.sendHeartbeat();
        expect(sender.pongTimer).toBeDefined();

        sender.recordActivity();

        expect(sender.pongTimer).toBeUndefined();
    });

    it('does not stack a second pong timer while one is outstanding', () => {
        const {sender, sent} = makeSender();
        sender.sendHeartbeat();
        const first = sender.pongTimer;
        sender.sendHeartbeat();
        expect(sender.pongTimer).toBe(first);
        expect(sent).toEqual(['ping']);
    });
});

describe('Labelling a result with the right guid', () => {
    // A remote execution's result arrives carrying the guid it was relayed under, and is cached
    // that way. Spreading it over the outgoing frame's guid sent later compiles out under that
    // execution's guid, so no router recognised them and each waited out its full deadline.
    it('keeps the guid of the request being answered when the result carries one of its own', async () => {
        const {sender, sent} = makeSender();
        sender.requireAcknowledgments = false;
        sender.messageQueue = [];

        await sender.send('the-real-guid', {code: 0, guid: 'a-stale-execution-guid'} as any);

        expect(JSON.parse(sent[0]).guid).toEqual('the-real-guid');
    });

    it('does not return the transport guid as part of an execution result', () => {
        const waiter = Object.create(EventsWsWaiter.prototype) as any;
        const handlers: Record<string, (m: any) => void> = {};
        waiter.timeout = 10_000;
        waiter.ws = {on: (event: string, fn: (m: any) => void) => (handlers[event] = fn)};

        const result = waiter.data();
        handlers.message(Buffer.from(JSON.stringify({guid: 'an-execution-guid', code: 0})));

        return expect(result).resolves.toEqual({code: 0});
    });
});
