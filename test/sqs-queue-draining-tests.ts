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
import {BaseExecutionTriple} from '../lib/execution/base-execution-triple.js';
import {SqsExecuteRequester, SqsWorkerMode} from '../lib/execution/sqs-execution-queue.js';

function makeProps(values: Record<string, string>) {
    return ((key: string, defaultValue?: any) => values[key] ?? defaultValue) as any;
}

function makeFakeSqs(message?: {Body: string; Attributes?: Record<string, string>}) {
    const calls: {op: string; params: any}[] = [];
    return {
        calls,
        sqs: {
            receiveMessage: async (params: any) => {
                calls.push({op: 'receive', params});
                return {Messages: message ? [{ReceiptHandle: 'handle-1', ...message}] : []};
            },
            deleteMessage: async (params: any) => calls.push({op: 'delete', params}),
            changeMessageVisibility: async (params: any) => calls.push({op: 'changeVisibility', params}),
            sendMessage: async (params: any) => calls.push({op: 'send', params}),
        },
    };
}

const awsProps = makeProps({region: 'us-east-1'});

function makeCompilationQueue(fake: ReturnType<typeof makeFakeSqs>) {
    const queue = new SqsCompilationWorkerMode(
        makeProps({'compilequeue.queue_url': 'https://sqs/prod-compilation-queue.fifo'}),
        awsProps,
    );
    (queue as any).sqs = fake.sqs;
    return queue;
}

function makeExecutionWorker(fake: ReturnType<typeof makeFakeSqs>) {
    const queue = new SqsWorkerMode(makeProps({'execqueue.queue_url': 'https://sqs/prod-execqueue'}), awsProps);
    (queue as any).sqs = fake.sqs;
    return queue;
}

describe('A compilation worker receiving a message', () => {
    it('deletes it and hands it on while running normally', async () => {
        const fake = makeFakeSqs({Body: JSON.stringify({guid: 'g1'})});
        const msg = await makeCompilationQueue(fake).pop();

        expect(msg?.guid).toEqual('g1');
        expect(fake.calls.map(call => call.op)).toEqual(['receive', 'delete']);
    });

    it('puts it straight back on the queue while draining', async () => {
        const fake = makeFakeSqs({Body: JSON.stringify({guid: 'g1'})});
        const queue = makeCompilationQueue(fake);
        queue.startDraining();

        expect(await queue.pop()).toBeUndefined();
        expect(fake.calls.map(call => call.op)).toEqual(['receive', 'changeVisibility']);
        expect(fake.calls[1].params).toMatchObject({ReceiptHandle: 'handle-1', VisibilityTimeout: 0});
    });

    it('records how many times SQS has delivered it', async () => {
        const fake = makeFakeSqs({Body: JSON.stringify({guid: 'g1'}), Attributes: {ApproximateReceiveCount: '2'}});
        const queue = makeCompilationQueue(fake);

        expect((await queue.pop())?.receiveCount).toEqual(2);
        expect(fake.calls[0].params.MessageSystemAttributeNames).toContain('ApproximateReceiveCount');
    });
});

describe('An execution worker receiving a message', () => {
    it('deletes it and hands it on while running normally', async () => {
        const fake = makeFakeSqs({Body: JSON.stringify({guid: 'g1'})});
        const msg = await makeExecutionWorker(fake).pop();

        expect(msg?.guid).toEqual('g1');
        expect(fake.calls.map(call => call.op)).toEqual(['receive', 'delete']);
    });

    it('puts it straight back on the queue while draining', async () => {
        const fake = makeFakeSqs({Body: JSON.stringify({guid: 'g1'})});
        const queue = makeExecutionWorker(fake);
        queue.startDraining();

        expect(await queue.pop()).toBeUndefined();
        expect(fake.calls.map(call => call.op)).toEqual(['receive', 'changeVisibility']);
        expect(fake.calls[1].params).toMatchObject({ReceiptHandle: 'handle-1', VisibilityTimeout: 0});
    });
});

describe('Queueing an execution request', () => {
    it('gives each request its own FIFO message group', async () => {
        const fake = makeFakeSqs();
        const requester = new SqsExecuteRequester(
            makeProps({'execqueue.queue_url': 'https://sqs/prod-execqueue'}),
            awsProps,
        );
        (requester as any).sqs = fake.sqs;

        await requester.push(new BaseExecutionTriple(), {guid: 'g1', hash: 'h', params: {}} as any);
        await requester.push(new BaseExecutionTriple(), {guid: 'g2', hash: 'h', params: {}} as any);

        expect(fake.calls.map(call => call.params.MessageGroupId)).toEqual(['g1', 'g2']);
    });
});
