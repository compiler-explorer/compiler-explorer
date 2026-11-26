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

import {performance} from 'node:perf_hooks';

import {logger} from './logger.js';

export class StartupPhases {
    private readonly phases = new Map<string, number>();
    private lastMarkMs = 0;
    private finished = false;

    /**
     * Attribute the time since the previous mark (or process start, for the first) to `phase`.
     * Once startup has finished this is a no-op, so later compiler rescans don't report as startup.
     */
    mark(phase: string): void {
        if (this.finished) return;
        const now = performance.now();
        const elapsedMs = now - this.lastMarkMs;
        this.lastMarkMs = now;
        this.phases.set(phase, (this.phases.get(phase) ?? 0) + elapsedMs);
        logger.info(`Startup phase '${phase}': ${Math.round(elapsedMs)}ms`);
    }

    /** Stop recording and return the elapsed milliseconds of each phase. */
    finish(): ReadonlyMap<string, number> {
        this.finished = true;
        return this.phases;
    }
}

export const startupPhases = new StartupPhases();
