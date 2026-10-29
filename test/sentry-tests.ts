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

import type {ErrorEvent} from '@sentry/browser';
import {describe, expect, it} from 'vitest';

import {isWorkerErrorNoise} from '../static/sentry-filters.js';

function makeEvent(value: string | undefined, mechanismType: string | undefined): ErrorEvent {
    return {
        type: undefined,
        exception: {values: [{value, mechanism: mechanismType ? {type: mechanismType, handled: false} : undefined}]},
    };
}

describe('isWorkerErrorNoise', () => {
    const value = 'Event `Event` (type=error) captured as exception';

    it('matches the messageless worker error from instrumented timers', () => {
        expect(isWorkerErrorNoise(makeEvent(value, 'auto.browser.browserapierrors.setTimeout'))).toBe(true);
    });

    it('keeps the same value from other mechanisms', () => {
        expect(isWorkerErrorNoise(makeEvent(value, 'auto.browser.global_handlers.onerror'))).toBe(false);
        expect(isWorkerErrorNoise(makeEvent(value, undefined))).toBe(false);
    });

    it('keeps other errors and events without exceptions', () => {
        expect(isWorkerErrorNoise(makeEvent('boom', 'auto.browser.browserapierrors.setTimeout'))).toBe(false);
        expect(isWorkerErrorNoise({type: undefined})).toBe(false);
    });
});
