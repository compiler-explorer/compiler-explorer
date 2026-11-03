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

import childProcess from 'node:child_process';
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

import {afterEach, beforeEach, describe, expect, it, vi} from 'vitest';

import {GoooCompiler} from '../lib/compilers/gooo.js';
import {languages} from '../lib/languages.js';
import {BypassCache} from '../types/compilation/compilation.interfaces.js';
import type {UnprocessedExecResult} from '../types/execution/execution.interfaces.js';
import {makeCompilationEnvironment, makeFakeCompilerInfo} from './utils.js';

const fixturePath = fileURLToPath(new URL('./fixtures/gooo/', import.meta.url));
const valid = await fs.readFile(path.join(fixturePath, 'valid.json'), 'utf8');
const changed = await fs.readFile(path.join(fixturePath, 'changed.json'), 'utf8');
const repeat = await fs.readFile(path.join(fixturePath, 'repeat.json'), 'utf8');
const failClosed = await fs.readFile(path.join(fixturePath, 'fail-closed.json'), 'utf8');
const validInput = await fs.readFile(path.join(fixturePath, 'valid.gooo'));
const changedInput = await fs.readFile(path.join(fixturePath, 'changed.gooo'));
const buildIdentity = JSON.parse(await fs.readFile(path.join(fixturePath, 'build-identity-darwin.json'), 'utf8'));
const inputFilename = path.join(path.sep, 'fake', 'gooo-test', 'example.gooo');
const executable = path.join(path.sep, 'fake', 'gooo-test', 'gooo');

function execution(stdout = valid, code = 0): UnprocessedExecResult {
    return {
        code,
        stdout,
        stderr: '',
        execTime: 0,
        timedOut: false,
        truncated: false,
        okToCache: true,
        filenameTransform: filename => filename,
    };
}

function compiler() {
    const metadata = vi.spyOn(childProcess, 'execSync').mockReturnValue(Buffer.from(''));
    try {
        return new GoooCompiler(
            makeFakeCompilerInfo({exe: executable, id: 'gooo_body_83928fa', lang: 'gooo'}),
            makeCompilationEnvironment({languages: {gooo: {id: 'gooo'}}}),
        );
    } finally {
        metadata.mockRestore();
    }
}

function options() {
    return {
        env: {PATH: '/ce/sandbox/helpers', KEEP: 'retained', GOOO_LAYA_URL: 'unwanted', GOOO_LAYA_API_KEY: 'unwanted'},
        timeoutMs: 7500,
        wrapper: 'sandbox-wrapper',
    };
}

function mutate(change: (value: any) => void): string {
    const value = JSON.parse(valid);
    change(value);
    return JSON.stringify(value);
}

describe('Gooo activity to Go source', () => {
    afterEach(() => vi.restoreAllMocks());

    it('registers the scoped language and proven default example', () => {
        expect(languages.gooo.monaco).toBe('gooo');
        expect(languages.gooo.monacoDisassembly).toBe('go');
        expect(languages.gooo.extensions).toEqual(['.gooo']);
        expect(languages.gooo.example).toBe(validInput.toString());
        const c = compiler();
        expect(GoooCompiler.key).toBe('gooo-body');
        expect(c.getArgumentParserClass().name).toBe('BaseParser');
        expect(c.optionsForFilter()).toEqual([]);
        expect(c.getInfo()).toMatchObject({
            supportsExecute: false,
            supportsBinary: false,
            supportsBinaryObject: false,
            supportsCfg: false,
            instructionSet: null,
        });
        expect(c.isCfgCompiler()).toBe(false);
    });

    it('preserves source lines and ignores assembly filters without inventing mappings', async () => {
        const source = JSON.parse(valid).source;
        const result = await compiler().processAsm({asm: source});
        expect(result).toEqual({asm: source.split('\n').map((text: string) => ({text})), languageId: 'go'});
        expect(result.asm.some(line => line.text.startsWith('//gooo:'))).toBe(true);
    });

    it('refuses direct execution requests without compiling or running generated Go', async () => {
        const c = compiler();
        const exec = vi.spyOn(c, 'exec');
        const result = await c.compile(
            validInput.toString(),
            ['--activity', 'Double'],
            {executorRequest: true},
            {},
            BypassCache.Compilation,
            [],
            {},
            [],
            [],
        );
        expect(result).toMatchObject({code: 1, didExecute: false, asm: []});
        expect(exec).not.toHaveBeenCalled();
        const filters = {execute: true, binary: true, binaryObject: true, commentOnly: true};
        c.fixFiltersBeforeCacheKey(filters);
        expect(filters).toEqual({commentOnly: true});
    });

    describe('body-codegen subprocess contract', () => {
        beforeEach(() => {
            vi.spyOn(fs, 'readFile').mockResolvedValue(validInput);
            vi.spyOn(fs, 'rm').mockResolvedValue(undefined);
            vi.spyOn(fs, 'writeFile').mockResolvedValue(undefined);
            vi.spyOn(fs, 'rename').mockResolvedValue(undefined);
        });

        it('uses only a fixed argv and preserves the sandbox environment without provider routing', async () => {
            const c = compiler();
            const exec = vi.spyOn(c, 'exec').mockResolvedValue(execution());
            const supplied = options();
            const args = c.prepareArguments(['--activity', 'Double']);
            expect(args).toEqual(['--activity', 'Double']);
            const result = await c.runCompiler(executable, args, inputFilename, supplied);
            expect(result).toMatchObject({code: 0, languageId: 'go', stdout: []});
            expect(result.instructionSet).toBeUndefined();
            expect(exec).toHaveBeenCalledExactlyOnceWith(
                executable,
                ['body-codegen', '--json', '--activity', 'Double', inputFilename],
                {
                    ...supplied,
                    customCwd: path.dirname(inputFilename),
                    maxOutput: 1024 * 1024,
                    env: {...supplied.env, GOOO_LAYA_URL: '', GOOO_LAYA_API_KEY: ''},
                },
            );
            expect(supplied.env.GOOO_LAYA_URL).toBe('unwanted');
            expect(supplied.env.PATH).toBe('/ce/sandbox/helpers');
            const output = c.getOutputFilename(path.dirname(inputFilename), c.outputFilebase);
            expect(fs.writeFile).toHaveBeenCalledExactlyOnceWith(`${output}.validated`, JSON.parse(valid).source, {
                flag: 'wx',
            });
            expect(fs.rename).toHaveBeenCalledExactlyOnceWith(`${output}.validated`, output);
        });

        it('uses the actual valid, repeat and changed fixtures without mixing requests on one instance', async () => {
            expect(repeat).toBe(valid);
            expect(JSON.parse(changed).source).not.toBe(JSON.parse(valid).source);
            const c = compiler();
            vi.spyOn(c, 'exec')
                .mockResolvedValueOnce(execution(valid))
                .mockResolvedValueOnce(execution(repeat))
                .mockResolvedValueOnce(execution(changed));
            vi.mocked(fs.readFile)
                .mockResolvedValueOnce(validInput)
                .mockResolvedValueOnce(validInput)
                .mockResolvedValueOnce(changedInput);
            for (let i = 0; i < 3; ++i) {
                expect((await c.runCompiler(executable, ['--activity', 'Double'], inputFilename, options())).code).toBe(
                    0,
                );
            }
            expect(vi.mocked(fs.writeFile).mock.calls.map(call => call[1])).toEqual([
                JSON.parse(valid).source,
                JSON.parse(valid).source,
                JSON.parse(changed).source,
            ]);
        });

        it.each(
            [
                [],
                ['Double'],
                ['--activity'],
                ['--activity', 'Double', '--json'],
                ['--activity', 'Double', '--activity', 'Other'],
                ['--activity=Double'],
                ['--activity', '../Double'],
                ['--activity', 'Double;echo'],
                ['body-codegen', '--activity', 'Double'],
                ['-o', '/tmp/out'],
                ['--activity', 'Double', '--', '/tmp/input.gooo'],
            ].map(args => ({args})),
        )('refuses unsupported option tokens $args', async ({args}) => {
            const c = compiler();
            const exec = vi.spyOn(c, 'exec');
            expect((await c.runCompiler(executable, args, inputFilename, options())).code).toBe(1);
            expect(exec).not.toHaveBeenCalled();
            expect(fs.writeFile).not.toHaveBeenCalled();
        });

        const malformed = [
            'not json',
            '',
            'null',
            '[]',
            '{}',
            JSON.stringify({source: 'partial'}),
            mutate(value => (value.report.schema = 'gooo/body-codegen-report/v4')),
            mutate(value => (value.report.decision = 'FAIL_CLOSED')),
            mutate(value => (value.report.activity = 'Other')),
            mutate(value => (value.report.compiler_source_sha = '0'.repeat(40))),
            mutate(value => (value.report.typecheck_passed = false)),
            mutate(value => (value.report.typecheck_passed = 'true')),
            mutate(value => (value.report.deterministic_replay = false)),
            mutate(value => (value.report.deterministic_replay = 1)),
            mutate(value => (value.report.repository_writes = 1)),
            mutate(value => (value.report.repository_writes = false)),
            mutate(value => (value.report.source_digest = 'sha256:' + '0'.repeat(64))),
            mutate(value => (value.report.generated_digest = 'sha256:' + '0'.repeat(64))),
            mutate(value => (value.report.replay_digest = 'sha256:' + '0'.repeat(64))),
            mutate(value => (value.report.plan_sha256 = 'unknown')),
            mutate(value => (value.report.program_digest = true)),
            mutate(value => (value.report.completeness_receipt.profile_id = 'other')),
            mutate(value => (value.report.completeness_receipt.scope.compiler_source_sha = '0'.repeat(40))),
            mutate(value => (value.source = '')),
            mutate(value => (value.source = ['partial'])),
            mutate(value => (value.report.error = 'partial failure')),
            mutate(value => (value.schema = 'mixed flat report')),
        ];
        it.each(
            malformed.map((stdout, i) => ({stdout, i})),
        )('refuses malformed report$i without source', async item => {
            const c = compiler();
            vi.spyOn(c, 'exec').mockResolvedValue(execution(item.stdout));
            const result = await c.runCompiler(executable, ['--activity', 'Double'], inputFilename, options());
            expect(result).toMatchObject({code: 1, asm: [], stdout: [], okToCache: false});
            expect(fs.writeFile).not.toHaveBeenCalled();
            expect(fs.rename).not.toHaveBeenCalled();
            expect(fs.rm).toHaveBeenCalledWith(c.getOutputFilename(path.dirname(inputFilename), c.outputFilebase), {
                force: true,
            });
        });

        it('renders the actual flat FAIL_CLOSED diagnostic and never a partial source', async () => {
            const c = compiler();
            vi.spyOn(c, 'exec').mockResolvedValue({...execution(failClosed, 1), stderr: 'original stderr'});
            const result = await c.runCompiler(executable, ['--activity', 'Double'], inputFilename, options());
            expect(result).toMatchObject({code: 1, stdout: [], asm: []});
            const diagnostic = result.stderr.map(line => line.text).join('\n');
            expect(diagnostic).toContain('original stderr');
            expect(diagnostic).toContain('undefined: input0');
            expect(fs.writeFile).not.toHaveBeenCalled();
        });

        it.each([
            {...execution(valid, 7)},
            {...execution(failClosed)},
            {...execution(), timedOut: true},
            {...execution(), truncated: true},
        ])('rejects nonzero, refusal, timeout or truncation even with JSON output', async response => {
            const c = compiler();
            vi.spyOn(c, 'exec').mockResolvedValue(response);
            const result = await c.runCompiler(executable, ['--activity', 'Double'], inputFilename, options());
            expect(result.code).not.toBe(0);
            expect(result.asm).toEqual([]);
            expect(result.stdout).toEqual([]);
            expect(fs.writeFile).not.toHaveBeenCalled();
        });

        it('returns subprocess and write failures without publishing partial output', async () => {
            const c = compiler();
            vi.spyOn(c, 'exec')
                .mockRejectedValueOnce(new Error('subprocess unavailable'))
                .mockResolvedValueOnce(execution());
            expect((await c.runCompiler(executable, ['--activity', 'Double'], inputFilename, options())).code).toBe(1);
            expect(fs.writeFile).not.toHaveBeenCalled();
            vi.mocked(fs.writeFile).mockRejectedValueOnce(new Error('disk unavailable'));
            const result = await c.runCompiler(executable, ['--activity', 'Double'], inputFilename, options());
            expect(result).toMatchObject({code: 1, asm: [], stdout: []});
            expect(fs.rename).not.toHaveBeenCalled();
            expect(result.stderr[0].text).toContain('disk unavailable');
        });
    });

    describe('pinned build identity', () => {
        const identity = buildIdentity;
        it('uses a bounded JSON identity probe instead of an explicit version override', async () => {
            const c = compiler();
            const exec = vi.spyOn(c, 'exec').mockResolvedValue(execution(JSON.stringify(identity)));
            expect((await c.getVersion())?.code).toBe(0);
            expect(exec.mock.calls[0][1]).toEqual(['version', '--build', '--json']);
            expect(exec.mock.calls[0][2]?.timeoutMs).toBeGreaterThan(0);
            expect(exec.mock.calls[0][2]?.env).toMatchObject({GOOO_LAYA_URL: '', GOOO_LAYA_API_KEY: ''});
        });
        it.each([
            {...identity, schema: 'other'},
            {...identity, version: '0.6.36-dev'},
            {...identity, compiler_source_sha: '0'.repeat(40)},
            {...identity, vcs_revision: '0'.repeat(40)},
            {...identity, build_info_observed: false},
            {...identity, build_info_observed: 'true'},
            {...identity, go_version: 'go1.27.1'},
            {...identity, native_go_required: 'go1.27.1'},
            {...identity, source_status: 'MODIFIED_VCS'},
            {...identity, vcs_modified: 'true'},
            {...identity, vcs_modified: false},
            {...identity, build_info_observed: undefined},
            null,
        ])('refuses an unpinned or malformed build identity %j', async value => {
            const c = compiler();
            vi.spyOn(c, 'exec').mockResolvedValue(execution(JSON.stringify(value)));
            expect(await c.getVersion()).toBeNull();
        });

        it.each([
            {...execution(JSON.stringify(identity)), code: 1},
            {...execution(JSON.stringify(identity)), timedOut: true},
            {...execution(JSON.stringify(identity)), truncated: true},
        ])('refuses a failed or incomplete identity probe', async response => {
            const c = compiler();
            vi.spyOn(c, 'exec').mockResolvedValue(response);
            expect(await c.getVersion()).toBeNull();
        });
    });
});
