// Copyright (c) 2024, Compiler Explorer Authors
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

import {SQS} from '@aws-sdk/client-sqs';

import {ExecutionParams} from '../../types/compilation/compilation.interfaces.js';
import {BasicExecutionResult} from '../../types/execution/execution.interfaces.js';
import {CompilationEnvironment} from '../compilation-env.js';
import {logger} from '../logger.js';
import {PropertyGetter} from '../properties.interfaces.js';
import {getHash} from '../utils.js';

import {LocalExecutionEnvironment} from './_all.js';
import {BaseExecutionTriple} from './base-execution-triple.js';
import {EventsWsSender} from './events-websocket.js';
import {getExecutionTriplesForCurrentHost} from './execution-triple.js';

export type RemoteExecutionMessage = {
    guid: string;
    hash: string;
    params: ExecutionParams;
};

export class SqsExecuteQueueBase {
    protected sqs: SQS;
    protected queue_url: string;

    constructor(props: PropertyGetter, awsProps: PropertyGetter) {
        const region = awsProps<string>('region', '');
        this.sqs = new SQS({region: region});
        this.queue_url = props<string>('execqueue.queue_url', '');
        if (this.queue_url === '') throw new Error('execqueue.queue_url property required');
    }

    getSqsQueueUrl(triple: BaseExecutionTriple) {
        return this.queue_url + '-' + triple.toString() + '.fifo';
    }
}

export class SqsExecuteRequester extends SqsExecuteQueueBase {
    private async sendMsg(url: string, body: string) {
        try {
            return await this.sqs.sendMessage({
                QueueUrl: url,
                MessageBody: body,
                MessageGroupId: 'default',
                MessageDeduplicationId: getHash(body),
            });
        } catch (e) {
            logger.error(`Error sending message to queue with URL: ${url}`);
            throw e;
        }
    }

    async push(triple: BaseExecutionTriple, message: RemoteExecutionMessage): Promise<any> {
        const body = JSON.stringify(message);
        const url = this.getSqsQueueUrl(triple);

        return this.sendMsg(url, body);
    }
}

export class SqsWorkerMode extends SqsExecuteQueueBase {
    protected triples: BaseExecutionTriple[];

    constructor(props: PropertyGetter, awsProps: PropertyGetter) {
        super(props, awsProps);
        this.triples = getExecutionTriplesForCurrentHost();
    }

    private async receiveMsg(url: string) {
        try {
            return await this.sqs.receiveMessage({
                QueueUrl: url,
                MaxNumberOfMessages: 1,
            });
        } catch (e) {
            logger.error(`Error retreiving message from queue with URL: ${url}`);
            throw e;
        }
    }

    async pop(): Promise<RemoteExecutionMessage | undefined> {
        const url = this.getSqsQueueUrl(this.triples[0]);

        const queued_messages = await this.receiveMsg(url);

        if (queued_messages.Messages && queued_messages.Messages.length === 1) {
            const queued_message = queued_messages.Messages[0];

            try {
                if (queued_message.Body) {
                    const json = queued_message.Body;
                    return JSON.parse(json) as RemoteExecutionMessage;
                } else {
                    return undefined;
                }
            } finally {
                if (queued_message.ReceiptHandle) {
                    await this.sqs.deleteMessage({
                        QueueUrl: url,
                        ReceiptHandle: queued_message.ReceiptHandle,
                    });
                }
            }
        }

        return undefined;
    }
}

async function sendResultViaWebsocket(
    compilationEnvironment: CompilationEnvironment,
    guid: string,
    result: BasicExecutionResult,
) {
    try {
        const sender = new EventsWsSender(compilationEnvironment.ceProps);
        await sender.send(guid, result);
        await sender.close();
    } catch (error) {
        logger.error(error);
    }
}

async function doOneExecution(queue: SqsWorkerMode, compilationEnvironment: CompilationEnvironment) {
    const msg = await queue.pop();
    if (msg && msg.guid) {
        try {
            const executor = new LocalExecutionEnvironment(compilationEnvironment);
            await executor.downloadExecutablePackage(msg.hash);
            const result = await executor.execute(msg.params);

            await sendResultViaWebsocket(compilationEnvironment, msg.guid, result);
        } catch (e) {
            // todo: e is undefined somehow?
            logger.error(e);

            await sendResultViaWebsocket(compilationEnvironment, msg.guid, {
                code: -1,
                stderr: [{text: 'Internal error when remotely executing'}],
                stdout: [],
                okToCache: false,
                timedOut: false,
                filenameTransform: f => f,
                execTime: 0,
            });
        }
    }
}

export function startExecutionWorkerThread(
    ceProps: PropertyGetter,
    awsProps: PropertyGetter,
    compilationEnvironment: CompilationEnvironment,
) {
    const queue = new SqsWorkerMode(ceProps, awsProps);

    // allow 2 executions at the same time

    const doExecutionWork1 = async () => {
        await doOneExecution(queue, compilationEnvironment);
        setTimeout(doExecutionWork1, 100);
    };

    const doExecutionWork2 = async () => {
        await doOneExecution(queue, compilationEnvironment);
        setTimeout(doExecutionWork2, 100);
    };

    setTimeout(doExecutionWork1, 1500);
    setTimeout(doExecutionWork2, 1530);
}
