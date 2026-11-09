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

import {afterEach, describe, expect, it} from 'vitest';
import LokiTransport from 'winston-loki';

import {initialiseLogging, logger} from '../lib/logger.js';
import {Graceful} from '../lib/node-graceful.js';

describe('Logging to loki', () => {
    afterEach(() => {
        for (const transport of logger.transports.filter(t => t instanceof LokiTransport)) {
            (transport as LokiTransport).close?.();
            logger.remove(transport);
        }
        Graceful.clear();
    });

    it('leaves SIGTERM to Graceful instead of exiting from a handler of its own', () => {
        const before = new Set(process.listeners('SIGTERM'));

        initialiseLogging({
            debug: false,
            suppressConsoleLog: false,
            paperTrailIdentifier: '',
            loki: 'http://127.0.0.1:9',
        });

        const added = process.listeners('SIGTERM').filter(listener => !before.has(listener));
        expect(Graceful.handlesSignal('SIGTERM')).toBe(true);
        expect(added).toEqual([(Graceful as any).signalsListeners.SIGTERM]);
    });
});
