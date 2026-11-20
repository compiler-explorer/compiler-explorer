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

import {afterEach, describe, expect, it, vi} from 'vitest';

import * as exec from '../lib/exec.js';
import {getFormatterTypeByKey} from '../lib/formatters/index.js';
import {RocFmtFormatter} from '../lib/formatters/rocfmt.js';
import type {UnprocessedExecResult} from '../types/execution/execution.interfaces.js';

function makeExecResult(code = 0, stdout = '', stderr = ''): UnprocessedExecResult {
    return {
        code,
        okToCache: true,
        filenameTransform: filename => filename,
        stdout,
        stderr,
        execTime: 0,
        timedOut: false,
        truncated: false,
    };
}

afterEach(() => vi.restoreAllMocks());

describe('Roc formatter', () => {
    it('is registered as the rocfmt formatter type and invokes roc fmt --stdin', async () => {
        expect(getFormatterTypeByKey('rocfmt')).toBe(RocFmtFormatter);

        const formatter = new RocFmtFormatter({
            name: 'rocfmt',
            exe: '/opt/compiler-explorer/roc/roc',
            version: 'nightly',
            styles: [],
            styleFiles: {},
            type: 'rocfmt',
        });
        expect(formatter.isValidStyle('__DefaultStyle')).toBe(true);

        const execSpy = vi
            .spyOn(exec, 'execute')
            .mockResolvedValue(makeExecResult(0, 'square = |num| num * num\n', ''));

        const source = 'square =   |num|   num*num\n';
        const result = await formatter.format(source, {
            useSpaces: false,
            tabWidth: 4,
            baseStyle: '__DefaultStyle',
        });

        expect(execSpy).toHaveBeenCalledWith('/opt/compiler-explorer/roc/roc', ['fmt', '--stdin'], {input: source});
        expect(result.code).toBe(0);
        expect(result.stdout).toBe('square = |num| num * num\n');
    });
});
