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

/**
 * End-to-end coverage for issue #9015.
 *
 * Drives a real BaseCompiler.compile() in compilation-worker mode against an in-memory S3 that
 * records the exact object keys written, then replays what ce-router does with the s3Key it was
 * handed: GetObject on COMPILATION_RESULTS_PREFIX + s3Key.
 */

import fs from 'node:fs';
import path from 'node:path';

import {GetObjectCommand, NoSuchKey, PutObjectCommand, S3Client} from '@aws-sdk/client-s3';
import {mockClient} from 'aws-sdk-client-mock';
import {beforeEach, describe, expect, it} from 'vitest';

import {BaseCompiler} from '../lib/base-compiler.js';
import {CompilationEnvironment} from '../lib/compilation-env.js';
import {CompilationQueue} from '../lib/compilation-queue.js';
import {FormattingService} from '../lib/formatting-service.js';
import {CompilerProps, fakeProps} from '../lib/properties.js';
import {BypassCache, CompilationResult} from '../types/compilation/compilation.interfaces.js';
import {CompilerInfo} from '../types/compiler.interfaces.js';
import {makeFakeCompilerInfo, makeFakeParseFiltersAndOutputOptions, newTempDir} from './utils.js';

const BUCKET = 'storage.godbolt.org';
// ce-router's COMPILATION_RESULTS_PREFIX default (src/services/result-waiter.ts).
const ROUTER_PREFIX = 'cache/';

const s3Objects = new Map<string, Buffer>();

// S3 extends S3Client, so mocking the base covers both CE's S3Bucket and the router-side client.
const mockS3 = mockClient(S3Client);

function installFakeS3() {
    s3Objects.clear();
    mockS3.reset();
    mockS3.on(PutObjectCommand).callsFake(params => {
        s3Objects.set(`${params.Bucket}/${params.Key}`, Buffer.from(params.Body));
        return {};
    });
    mockS3.on(GetObjectCommand).callsFake(params => {
        const data = s3Objects.get(`${params.Bucket}/${params.Key}`);
        if (data === undefined) throw new NoSuchKey({message: 'No such key', $metadata: {}});
        return {
            Body: {
                transformToByteArray: async () => data,
                transformToString: async () => data.toString('utf8'),
            },
        };
    });
}

// What ce-router's fetchResultFromS3 does with the key the worker reported.
async function routerFetch(s3Key: string): Promise<any> {
    const response: any = await new S3Client({region: 'us-east-1'}).send(
        new GetObjectCommand({Bucket: BUCKET, Key: `${ROUTER_PREFIX}${s3Key}`}),
    );
    return JSON.parse(await response.Body.transformToString());
}

function makeWorkerEnvironment(cacheConfig: string): CompilationEnvironment {
    const compilerProps = new CompilerProps(
        {'c++': {id: 'c++'}} as any,
        fakeProps({
            cacheConfig,
            'compilequeue.is_worker': true,
            compileTimeoutMs: 500,
            optionsAllowedRe: '.*',
            optionsForbiddenRe: '(?!)',
        }),
    );
    return new CompilationEnvironment(
        compilerProps,
        fakeProps({}),
        new CompilationQueue(1, 5000, 100_000),
        new FormattingService(),
    );
}

// A "compiler" that spews past the 31KiB websocket threshold and then hangs, so CE's real exec()
// times it out and the result is marked okToCache: false: the shape of result the worker is
// supposed to hand off out of band.
function makeSlowNoisyCompiler(): string {
    const exe = path.join(newTempDir(), 'noisy-compiler.sh');
    fs.writeFileSync(
        exe,
        [
            '#!/bin/bash',
            'for i in $(seq 1 2000); do',
            '  echo "error: something went wrong on line $i" >&2',
            'done',
            'sleep 30',
            '',
        ].join('\n'),
    );
    fs.chmodSync(exe, 0o755);
    return exe;
}

function makeWorkerCompiler(ce: CompilationEnvironment): BaseCompiler {
    const info: Partial<CompilerInfo> = {
        id: 'noisy',
        name: 'Noisy',
        lang: 'c++',
        exe: makeSlowNoisyCompiler(),
        options: '',
        ldPath: [],
        libPath: [],
        version: '1.0',
        supportsBinary: false,
        supportsExecute: false,
    };
    return new BaseCompiler(makeFakeCompilerInfo(info), ce);
}

function doCompile(compiler: BaseCompiler): Promise<CompilationResult> {
    return compiler.compile(
        'int main() {}',
        [],
        {},
        makeFakeParseFiltersAndOutputOptions({}),
        BypassCache.None,
        [],
        {},
        [],
        [],
    );
}

// Both suites drive a real compilation, and the stand-in compiler below is a shell script, which
// Windows cannot spawn. What they cover - which key an oversized result is reported under, and
// where it is stored - has no platform-specific part; the script is only a way to produce a result
// that is both too large for the websocket and marked okToCache: false.
describe.skipIf(process.platform === 'win32')(
    'Issue 9015: a worker reports where it actually put a large result',
    () => {
        beforeEach(() => installFakeS3());

        it.each([
            // Exactly what etc/config/compiler-explorer.amazon.properties has.
            ['layered, as deployed', `InMemory(25);S3(${BUCKET},cache,us-east-1)`],
            ['a bare S3 cache', `S3(${BUCKET},cache,us-east-1)`],
        ])('is fetchable by ce-router with %s', async (_name, cacheConfig) => {
            const compiler = makeWorkerCompiler(makeWorkerEnvironment(cacheConfig));
            const result = await doCompile(compiler);

            // Preconditions: this is the shape of result the issue is about.
            expect(result.okToCache).toBe(false);
            expect(JSON.stringify(result).length).toBeGreaterThan(31 * 1024);
            expect(result.s3Key).toBeDefined();

            await expect(routerFetch(result.s3Key!)).resolves.toHaveProperty('asm');

            // Hand the same bucket state to ce-router's own test, which runs the real ResultWaiter.
            if (process.env.ISSUE_9015_DUMP) {
                fs.writeFileSync(
                    process.env.ISSUE_9015_DUMP,
                    JSON.stringify({
                        bucket: BUCKET,
                        s3Key: result.s3Key,
                        objects: Object.fromEntries([...s3Objects].map(([k, v]) => [k, v.toString('utf8')])),
                    }),
                );
            }
        });
    },
);

describe.skipIf(process.platform === 'win32')('Issue 9015, second half: where that result is stored', () => {
    beforeEach(() => installFakeS3());

    // The result goes under temp/, which is not a key cacheGet reads, so a result we were told not
    // to cache can never come back as a cache hit.
    it('does not serve an okToCache: false result as a cache hit', async () => {
        const compiler = makeWorkerCompiler(makeWorkerEnvironment(`InMemory(25);S3(${BUCKET},cache,us-east-1)`));

        const first = await doCompile(compiler);
        expect(first.okToCache).toBe(false);

        const second = await doCompile(compiler);
        expect(second.retreivedFromCache).toBeUndefined();
    });
});
