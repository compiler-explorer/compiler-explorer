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

import express from 'express';
import request from 'supertest';
import {beforeEach, describe, expect, it, vi} from 'vitest';

import {CompilationQueue} from '../lib/compilation-queue.js';
import {MachFmtFormatter} from '../lib/formatters/machfmt.js';
import {FormattingService} from '../lib/formatting-service.js';
import {FormattingController} from '../lib/handlers/api/formatting-controller.js';
import {fakeProps} from '../lib/properties.js';
import {ExecutionOptions} from '../types/compilation/compilation.interfaces.js';
import {UnprocessedExecResult} from '../types/execution/execution.interfaces.js';

vi.mock('../lib/exec.js', () => ({execute: vi.fn()}));

import * as exec from '../lib/exec.js';

const machExe = '/opt/compiler-explorer/mach-7.3.1/mach';

const unformatted = 'fun main()i32{ret 0;}';
const canonical = 'fun main() i32 {\n    ret 0;\n}\n';
const malformed = 'fun broken(';
const diagnostics = [
    'error[syntax.expected_name] expected parameter name',
    '  @ <stdin>:1:12',
    '1 | fun broken(',
    '  |            ^',
    '',
    'failed <stdin>',
    '  errors  1',
    '',
].join('\n');

function execResult(code: number, stdout: string, stderr: string): UnprocessedExecResult {
    return {
        code,
        okToCache: true,
        filenameTransform: x => x,
        stdout,
        stderr,
        execTime: 0,
        timedOut: false,
        truncated: false,
    };
}

/**
 * Stands in for the mach binary: `--version` reports the semver, `fmt -` writes the buffer's canonical layout to
 * stdout, and a buffer that does not parse writes nothing to stdout, reports on stderr and exits non-zero.
 */
function fakeMach(exe: string, args: string[], options: ExecutionOptions): Promise<UnprocessedExecResult> {
    if (args[0] === '--version') return Promise.resolve(execResult(0, '7.3.1\n', ''));
    if (options.input === malformed) return Promise.resolve(execResult(1, '', diagnostics));
    return Promise.resolve(execResult(0, canonical, ''));
}

function machProps() {
    return fakeProps({
        formatters: 'machfmt',
        'formatter.machfmt.name': 'machfmt',
        'formatter.machfmt.exe': machExe,
        'formatter.machfmt.type': 'machfmt',
        'formatter.machfmt.version': '--version',
        'formatter.machfmt.versionRe': '^\\d+\\.\\d+\\.\\d+',
        'formatter.machfmt.styles': '',
    });
}

const formatOptions = {useSpaces: true, tabWidth: 4, baseStyle: '__DefaultStyle'};

describe('Mach formatter', () => {
    beforeEach(() => {
        vi.mocked(exec.execute).mockReset();
        vi.mocked(exec.execute).mockImplementation(fakeMach);
    });

    it('feeds the buffer through mach fmt on stdin', async () => {
        const formatter = new MachFmtFormatter({
            name: 'machfmt',
            exe: machExe,
            styles: [],
            styleFiles: {},
            type: 'machfmt',
            version: '7.3.1',
        });
        const result = await formatter.format(unformatted, formatOptions);
        expect(vi.mocked(exec.execute)).toHaveBeenCalledWith(machExe, ['fmt', '-'], {input: unformatted});
        expect(result.code).toBe(0);
        expect(result.stdout).toBe(canonical);
    });

    it('accepts every style, because mach fmt has one canonical layout', () => {
        const formatter = new MachFmtFormatter({
            name: 'machfmt',
            exe: machExe,
            styles: [],
            styleFiles: {},
            type: 'machfmt',
            version: '7.3.1',
        });
        expect(formatter.isValidStyle('__DefaultStyle')).toBe(true);
        expect(formatter.isValidStyle('Google')).toBe(true);
    });

    it('discovers the formatter and its version from mach --version', async () => {
        const service = new FormattingService();
        await service.initialize(machProps());
        const formatter = service.getFormatterById('machfmt');
        expect(formatter).toBeInstanceOf(MachFmtFormatter);
        expect(vi.mocked(exec.execute)).toHaveBeenCalledWith(machExe, ['--version'], {});
        expect(formatter?.formatterInfo.version).toBe('7.3.1');
    });
});

describe('Mach formatting over the API', () => {
    let app: express.Express;

    beforeEach(async () => {
        vi.mocked(exec.execute).mockReset();
        vi.mocked(exec.execute).mockImplementation(fakeMach);
        const service = new FormattingService();
        await service.initialize(machProps());
        app = express();
        app.use(express.json());
        app.use(new FormattingController(service, new CompilationQueue(1, 10_000, 10_000)).createRouter());
    });

    it('answers with the canonical layout', async () => {
        await request(app)
            .post('/api/format/machfmt')
            .send({base: '__DefaultStyle', source: unformatted})
            .set('Accept', 'application/json')
            .set('Content-Type', 'application/json')
            .expect('Content-Type', /json/)
            .expect(200, {exit: 0, answer: canonical});
    });

    it('reports the diagnostics and no source when the buffer does not parse', async () => {
        const res = await request(app)
            .post('/api/format/machfmt')
            .send({base: '__DefaultStyle', source: malformed})
            .set('Accept', 'application/json')
            .set('Content-Type', 'application/json')
            .expect('Content-Type', /json/)
            .expect(200);
        expect(res.body.exit).not.toBe(0);
        expect(res.body.answer).toBe(diagnostics);
        expect(res.body.answer).not.toContain('fun broken() {');
    });
});
