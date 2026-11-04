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

import {PersistentEventsSender} from '../../lib/execution/events-websocket.js';
import {CompilationResult} from '../../types/compilation/compilation.interfaces.js';

vi.mock('ws', async () => {
    const {EventEmitter} = await import('node:events');

    class FakeWebSocket extends EventEmitter {
        static readonly OPEN = 1;
        static readonly CLOSED = 3;
        static instances: FakeWebSocket[] = [];

        readyState: number = FakeWebSocket.OPEN;
        sent: string[] = [];

        constructor(public url: string) {
            super();
            FakeWebSocket.instances.push(this);
        }

        send(data: string) {
            this.sent.push(data);
        }

        ping() {}

        close() {}
    }

    return {WebSocket: FakeWebSocket};
});

const {WebSocket: FakeWebSocket} = (await import('ws')) as any;

const props = (key: string, defaultValue?: any) =>
    key === 'execqueue.events_url' ? 'ws://events.example/api' : defaultValue;

const emptyResult = {code: 0, stdout: [], stderr: [], timedOut: false} as unknown as CompilationResult;

function latestSocket() {
    return FakeWebSocket.instances[FakeWebSocket.instances.length - 1];
}

describe('PersistentEventsSender permanent failure', () => {
    beforeEach(() => {
        FakeWebSocket.instances = [];
        vi.useFakeTimers();
    });

    afterEach(() => {
        vi.useRealTimers();
    });

    // Burn through every reconnect attempt without ever reaching a successful 'open', which is what
    // resets the attempt counter.
    function failPermanently() {
        for (let attempt = 0; attempt < 5; attempt++) {
            latestSocket().emit('close');
            vi.advanceTimersByTime(60000);
        }
        latestSocket().emit('close');
    }

    it('rejects in-flight acknowledgements so the sender does not stay busy forever', async () => {
        const sender = new PersistentEventsSender(props);
        latestSocket().emit('open');

        const sent = sender.send('some-guid', emptyResult);
        expect(sender.getPendingAckCount()).toEqual(1);

        failPermanently();

        await expect(sent).rejects.toThrow('WebSocket connection failed permanently');
        expect(sender.hasFailedPermanently()).toBe(true);
        expect(sender.getPendingAckCount()).toEqual(0);
    });

    it('reports itself unready once it has failed permanently', async () => {
        const sender = new PersistentEventsSender(props);
        latestSocket().emit('open');
        expect(sender.isReadyForNewMessages()).toBe(true);

        const sent = sender.send('some-guid', emptyResult);
        failPermanently();
        await expect(sent).rejects.toThrow();

        expect(sender.isReadyForNewMessages()).toBe(false);
    });

    it('rejects queued messages that never made it onto the wire', async () => {
        const sender = new PersistentEventsSender(props);

        // Never opened, so the message sits in the queue rather than in pendingAcks.
        const sent = sender.send('some-guid', emptyResult);
        failPermanently();

        await expect(sent).rejects.toThrow('WebSocket connection failed permanently');
    });
});
