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

import {afterEach, describe, expect, it, vi} from 'vitest';

import {logger} from '../lib/logger.js';
import {StartupPhases} from '../lib/startup-phases.js';

describe('StartupPhases', () => {
    afterEach(() => {
        vi.restoreAllMocks();
    });

    it('attributes the time since the previous mark to each phase', () => {
        const now = vi.spyOn(performance, 'now');
        const info = vi.spyOn(logger, 'info').mockImplementation(() => logger);
        const phases = new StartupPhases();

        now.mockReturnValue(1000);
        phases.mark('config');
        now.mockReturnValue(1250);
        phases.mark('create-compilers');
        now.mockReturnValue(1300);
        phases.mark('create-compilers');

        expect([...phases.finish()]).toEqual([
            ['config', 1000],
            ['create-compilers', 300],
        ]);
        expect(info).toHaveBeenCalledWith("Startup phase 'config': 1000ms");
        expect(info).toHaveBeenCalledWith("Startup phase 'create-compilers': 250ms");
    });

    it('ignores marks after startup has finished', () => {
        vi.spyOn(performance, 'now').mockReturnValue(10);
        const phases = new StartupPhases();
        phases.mark('config');
        phases.finish();
        phases.mark('possible-arguments');

        expect([...phases.finish()]).toEqual([['config', 10]]);
    });
});
