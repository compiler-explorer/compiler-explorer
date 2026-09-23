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

import {describe, expect, it, vi} from 'vitest';

import {PersistentEventsSender} from '../lib/execution/events-websocket.js';

// isReadyForNewMessages() and the nack handling both only touch the connection flags and the
// pending-ack map, so a real socket is not needed to exercise the interaction between them.
function makeSender(): any {
    const sender = Object.create(PersistentEventsSender.prototype) as any;
    sender.requireAcknowledgments = true;
    sender.isConnected = true;
    sender.ws = {readyState: 1};
    sender.pendingAcks = new Map();
    return sender;
}

function addPending(sender: any, guid: string) {
    const rejected: Error[] = [];
    sender.pendingAcks.set(guid, {
        timeout: setTimeout(() => {}, 60_000),
        retryCount: 0,
        resolve: vi.fn(),
        reject: (e: Error) => rejected.push(e),
        messageData: {},
    });
    return rejected;
}

describe('A result nobody is waiting for', () => {
    // A worker pulls no new work while any result is unacknowledged, and a result whose
    // requester has timed out is never acknowledged - so it used to hold the worker for the
    // full retry budget, three retries at three seconds, on a queue that is deep precisely
    // because requests are timing out. The events server now says so, and the worker stops.
    it('is abandoned as soon as the events server says there are no listeners', () => {
        const sender = makeSender();
        const rejected = addPending(sender, 'lonely-guid');
        expect(sender.isReadyForNewMessages()).toBe(false);

        sender.handleNoListeners('lonely-guid', 'no-listeners');

        expect(sender.getPendingAckCount()).toBe(0);
        expect(sender.isReadyForNewMessages()).toBe(true);
        expect(rejected).toHaveLength(1);
    });

    it('does not disturb results that are still outstanding', () => {
        const sender = makeSender();
        addPending(sender, 'lonely-guid');
        addPending(sender, 'still-waiting');

        sender.handleNoListeners('lonely-guid', 'no-listeners');

        expect(sender.getPendingAckCount()).toBe(1);
        // The gate is unchanged: one result still in flight is still a reason to hold off.
        expect(sender.isReadyForNewMessages()).toBe(false);
    });

    it('ignores a nack for something it is not tracking', () => {
        const sender = makeSender();
        expect(() => sender.handleNoListeners('never-heard-of-it', 'no-listeners')).not.toThrow();
        expect(sender.getPendingAckCount()).toBe(0);
    });
});
