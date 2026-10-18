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

import {Graceful} from '../lib/node-graceful.js';

// Graceful is all static state, and registering through Graceful.on() would install real process signal
// handlers, so these tests set the listeners directly and deliver signals through the private handler.
const graceful = Graceful as any;

function deliver(signal: string) {
    graceful.onDeadlyEvent(signal);
}

describe('Graceful receiving a second deadly signal while exiting', () => {
    let exit: ReturnType<typeof vi.spyOn>;
    const savedTimeout = Graceful.timeout;

    beforeEach(() => {
        vi.useFakeTimers();
        exit = vi.spyOn(process, 'exit').mockImplementation((() => {}) as any);
        Graceful.timeout = 0;
        graceful.isExiting = false;
        graceful.exitSignal = undefined;
        // A listener that never finishes, standing in for a worker still draining.
        graceful.listeners = [() => new Promise(() => {})];
    });

    afterEach(() => {
        graceful.listeners = [];
        graceful.isExiting = false;
        Graceful.timeout = savedTimeout;
        vi.useRealTimers();
        vi.restoreAllMocks();
    });

    it('keeps waiting for its listeners when the same signal arrives again straight away', () => {
        deliver('SIGTERM');
        vi.advanceTimersByTime(50);
        deliver('SIGTERM');

        expect(exit).not.toHaveBeenCalled();
    });

    it('exits at once when the same signal is sent again later', () => {
        deliver('SIGTERM');
        vi.advanceTimersByTime(Graceful.duplicateSignalWindowMs);
        deliver('SIGTERM');

        expect(exit).toHaveBeenCalledWith(1);
    });

    it('exits at once on a different signal, however soon it follows', () => {
        deliver('SIGTERM');
        deliver('SIGINT');

        expect(exit).toHaveBeenCalledWith(1);
    });
});
