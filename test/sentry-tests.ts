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

import {defaultStackParser, type ErrorEvent, type StackFrame} from '@sentry/browser';
import {describe, expect, it} from 'vitest';

import {isWorkerErrorNoise, restoreRethrownFrames} from '../static/sentry-filters.js';

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

describe('restoreRethrownFrames', () => {
    const vendor = 'https://static.ce-cdn.net/vendor.ba852ea1f516decc3050.js';
    const rethrowSite: StackFrame[] = [
        {filename: vendor, function: 'sentryWrapped', lineno: 2, colno: 81131, in_app: true},
        {filename: vendor, function: 'apply', lineno: 2, colno: 1349960, in_app: true},
    ];

    function makeRethrow(
        value: string,
        frames: StackFrame[],
        mechanismType = 'auto.browser.browserapierrors.setTimeout',
    ) {
        return {
            type: undefined,
            exception: {
                values: [{value, mechanism: {type: mechanismType, handled: false}, stacktrace: {frames: [...frames]}}],
            },
        } satisfies ErrorEvent;
    }

    function functionsOf(event: ErrorEvent) {
        return event.exception?.values?.[0].stacktrace?.frames?.map(frame => frame.function);
    }

    it('appends the original frames to a Firefox rethrow', () => {
        const event = makeRethrow(
            'nexium: @brackets token returned but no bracket defined as: .{\n\n' +
                `me@${vendor}:2:4171275\n` +
                `_myTokenize@${vendor}:2:4182709\n` +
                '43951/t.prototype.onLanguageChange@https://static.ce-cdn.net/main.ac366949bbd05dabc1f4.js:1:621400\n',
            rethrowSite,
        );
        restoreRethrownFrames(event);
        expect(functionsOf(event)).toEqual([
            'sentryWrapped',
            'apply',
            '43951/t.prototype.onLanguageChange',
            '_myTokenize',
            'me',
        ]);
        expect(event.exception.values[0].stacktrace.frames.at(-1)).toMatchObject({
            filename: vendor,
            lineno: 2,
            colno: 4171275,
        });
    });

    it('appends the original frames to a Safari rethrow', () => {
        const event = makeRethrow(
            "null is not an object (evaluating 'e.children')\n\n" +
                `readHorizontalRanges@${vendor}:2:1940799\n` +
                'dispatchEvent@[native code]\n' +
                `onTouchEnd@${vendor}:2:1135483\n`,
            rethrowSite,
        );
        restoreRethrownFrames(event);
        expect(functionsOf(event)?.slice(2)).toEqual(['onTouchEnd', 'dispatchEvent', 'readHorizontalRanges']);
    });

    it('leaves a Chrome rethrow alone, as its frames already include the original stack', () => {
        const value =
            't.isCheapToTokenize is not a function\n\n' +
            'TypeError: t.isCheapToTokenize is not a function\n' +
            `    at e.getEditOperations (${vendor}:7:6159887)\n` +
            `    at mo._getEditOperationsFromCommand (${vendor}:2:2182835)\n`;
        const event = makeRethrow(value, [...rethrowSite, ...defaultStackParser(value)]);
        const before = functionsOf(event);
        restoreRethrownFrames(event);
        expect(functionsOf(event)).toEqual(before);
    });

    it('leaves errors from other mechanisms alone', () => {
        const event = makeRethrow(
            `boom\n\nme@${vendor}:2:4171275\n`,
            rethrowSite,
            'auto.browser.global_handlers.onerror',
        );
        restoreRethrownFrames(event);
        expect(functionsOf(event)).toEqual(['sentryWrapped', 'apply']);
    });

    it('leaves messages without an embedded stack alone', () => {
        for (const value of ['boom', 'boom\n\nsome more detail']) {
            const event = makeRethrow(value, rethrowSite);
            restoreRethrownFrames(event);
            expect(functionsOf(event)).toEqual(['sentryWrapped', 'apply']);
        }
    });
});
