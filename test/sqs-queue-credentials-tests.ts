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

import {describe, expect, it} from 'vitest';

import {SqsCompilationWorkerMode} from '../lib/compilation/sqs-compilation-queue.js';
import {SqsExecuteRequester, SqsWorkerMode} from '../lib/execution/sqs-execution-queue.js';

function makeProps(values: Record<string, string>) {
    return ((key: string, defaultValue?: any) => values[key] ?? defaultValue) as any;
}

const awsProps = makeProps({region: 'us-east-1'});
const compileProps = makeProps({'compilequeue.queue_url': 'https://sqs/prod-compilation-queue.fifo'});
const execProps = makeProps({'execqueue.queue_url': 'https://sqs/prod-execqueue'});

async function resolvedAccessKey(client: any) {
    return (await client.config.credentials()).accessKeyId;
}

// The test setup installs fake credentials as the shared provider, so a client that resolves anything else is
// using the SDK's own default chain: 1s instance-metadata timeout, no retries, and a cold fetch per client.
describe('SQS queue AWS clients use the shared credential provider', () => {
    it('for the compilation worker queue and its S3 overflow client', async () => {
        const queue = new SqsCompilationWorkerMode(compileProps, awsProps) as any;
        expect(await resolvedAccessKey(queue.sqs)).toEqual('not-a-real-key');
        expect(await resolvedAccessKey(queue.s3)).toEqual('not-a-real-key');
    });

    it('for the execution worker queue', async () => {
        const queue = new SqsWorkerMode(execProps, awsProps) as any;
        expect(await resolvedAccessKey(queue.sqs)).toEqual('not-a-real-key');
    });

    it('for the execution requester', async () => {
        const requester = new SqsExecuteRequester(execProps, awsProps) as any;
        expect(await resolvedAccessKey(requester.sqs)).toEqual('not-a-real-key');
    });
});
