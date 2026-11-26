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

import * as Sentry from '@sentry/browser';

const WORKER_ERROR_VALUE = 'Event `Event` (type=error) captured as exception';
const TIMER_MECHANISM = 'auto.browser.browserapierrors.setTimeout';

// Scanner browsers (e.g. old Chromium on Windows from cloud data centres) fail to start our web workers. The
// resulting error carries no message or stack, so there is nothing to act on.
export function isWorkerErrorNoise(event: Sentry.ErrorEvent): boolean {
    const exception = event.exception?.values?.[0];
    return (
        exception?.value === WORKER_ERROR_VALUE &&
        exception.mechanism?.type?.startsWith('auto.browser.browserapierrors.') === true
    );
}

// Monaco reports an internal error by rethrowing it from a timer as `new Error(message + '\n\n' + stack)`. Chrome
// starts `stack` with the message, so the original frames are parsed along with the rethrow site. Firefox and Safari
// don't, leaving only the rethrow site, so every Monaco error from those browsers lands in one Sentry issue.
export function restoreRethrownFrames(event: Sentry.ErrorEvent): void {
    const exception = event.exception?.values?.[0];
    if (exception?.mechanism?.type !== TIMER_MECHANISM || !exception.value) return;
    const stackStart = exception.value.indexOf('\n\n');
    if (stackStart < 0) return;
    const original = Sentry.defaultStackParser(exception.value.slice(stackStart + 2));
    const throwSite = original[original.length - 1];
    if (!throwSite) return;
    const frames = exception.stacktrace?.frames ?? [];
    const alreadyParsed = frames.some(
        frame =>
            frame.filename === throwSite.filename &&
            frame.lineno === throwSite.lineno &&
            frame.colno === throwSite.colno,
    );
    if (alreadyParsed) return;
    exception.stacktrace = {...exception.stacktrace, frames: [...frames, ...original]};
}
