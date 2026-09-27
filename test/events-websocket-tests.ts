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
