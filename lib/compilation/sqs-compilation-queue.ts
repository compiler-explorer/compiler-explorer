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

import {CompilationResult} from '../../types/compilation/compilation.interfaces.js';
import {CompilationEnvironment} from '../compilation-env.js';
import {EventsWsSender} from '../execution/events-websocket.js';
import {logger} from '../logger.js';
import {PropertyGetter} from '../properties.interfaces.js';

export type RemoteCompilationRequest = {
    guid: string;
    compilerId: string;
    source: string;
    options: string[];
    backendOptions: any;
    filters: any;
    bypassCache: any;
    tools: any;
    executeParameters: any;
    libraries: any[];
    lang: string;
    files: any[];
    isCMake?: boolean;
};

export class SqsCompilationQueueBase {
    protected sqs: SQS;
    protected queue_url: string;

    constructor(props: PropertyGetter, awsProps: PropertyGetter) {
        const region = awsProps<string>('region', '');
        this.sqs = new SQS({region: region});
        this.queue_url = props<string>('compilequeue.queue_url', '');

        if (this.queue_url === '') {
            throw new Error('compilequeue.queue_url is required for worker mode');
        }
    }
}

export class SqsCompilationWorkerMode extends SqsCompilationQueueBase {
    private async receiveMsg(url: string) {
        try {
            return await this.sqs.receiveMessage({
                QueueUrl: url,
                MaxNumberOfMessages: 1,
            });
        } catch (e) {
            logger.error(`Error retrieving compilation message from queue with URL: ${url}`);
            throw e;
        }
    }

    async pop(): Promise<RemoteCompilationRequest | undefined> {
        const url = this.queue_url;
        const queued_messages = await this.receiveMsg(url);

        if (queued_messages.Messages && queued_messages.Messages.length === 1) {
            const queued_message = queued_messages.Messages[0];

            try {
                if (queued_message.Body) {
                    const json = queued_message.Body;
                    return JSON.parse(json) as RemoteCompilationRequest;
                }
                return undefined;
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

async function sendCompilationResultViaWebsocket(
    compilationEnvironment: CompilationEnvironment,
    guid: string,
    result: CompilationResult,
) {
    try {
        const sender = new EventsWsSender(compilationEnvironment.ceProps);
        // Convert CompilationResult to BasicExecutionResult format expected by EventsWsSender
        const basicResult = {
            ...result, // Include all fields from compilation result first
            okToCache: result.okToCache ?? false,
            filenameTransform: (f: string) => f,
            execTime: 0,
        };
        await sender.send(guid, basicResult);
        await sender.close();
    } catch (error) {
        logger.error(error);
    }
}

async function doOneCompilation(queue: SqsCompilationWorkerMode, compilationEnvironment: CompilationEnvironment) {
    const msg = await queue.pop();
    if (msg?.guid) {
        const startTime = Date.now();
        const compilationType = msg.isCMake ? 'cmake' : 'compile';
        logger.debug(`Processing ${compilationType} request ${msg.guid} for compiler ${msg.compilerId}`);

        try {
            const compiler = compilationEnvironment.findCompiler(msg.lang as any, msg.compilerId);
            if (!compiler) {
                throw new Error(`Compiler with ID ${msg.compilerId} not found for language ${msg.lang}`);
            }

            let result: CompilationResult;
            if (msg.isCMake) {
                const parsedRequest = {
                    source: msg.source,
                    options: msg.options,
                    backendOptions: msg.backendOptions,
                    filters: msg.filters,
                    bypassCache: msg.bypassCache,
                    tools: msg.tools,
                    executeParameters: msg.executeParameters,
                    libraries: msg.libraries,
                };
                result = await compiler.cmake(msg.files, parsedRequest, msg.bypassCache);
            } else {
                result = await compiler.compile(
                    msg.source,
                    msg.options,
                    msg.backendOptions,
                    msg.filters,
                    msg.bypassCache,
                    msg.tools,
                    msg.executeParameters,
                    msg.libraries,
                    msg.files,
                );
            }

            await sendCompilationResultViaWebsocket(compilationEnvironment, msg.guid, result);

            const endTime = Date.now();
            const duration = endTime - startTime;
            logger.info(`Completed ${compilationType} request ${msg.guid} in ${duration}ms`, {
                compilerId: msg.compilerId,
                language: msg.lang,
                success: result.code === 0,
                duration,
            });
        } catch (e) {
            const endTime = Date.now();
            const duration = endTime - startTime;
            logger.error(`Failed ${compilationType} request ${msg.guid} after ${duration}ms:`, e);

            await sendCompilationResultViaWebsocket(compilationEnvironment, msg.guid, {
                code: -1,
                stderr: [{text: 'Internal error when remotely compiling'}],
                stdout: [],
                okToCache: false,
                timedOut: false,
                inputFilename: '',
                asm: [],
                tools: [],
            });
        }
    }
}

export function startCompilationWorkerThread(
    ceProps: PropertyGetter,
    awsProps: PropertyGetter,
    compilationEnvironment: CompilationEnvironment,
) {
    const queue = new SqsCompilationWorkerMode(ceProps, awsProps);
    const numThreads = ceProps<number>('compilequeue.worker_threads', 2);

    logger.info('Starting compilation worker threads', {
        numThreads,
    });

    logger.info(`Starting ${numThreads} compilation worker threads`);
    for (let i = 0; i < numThreads; i++) {
        const doCompilationWork = async () => {
            await doOneCompilation(queue, compilationEnvironment);
            setTimeout(doCompilationWork, 100);
        };
        setTimeout(doCompilationWork, 1500 + i * 30);
    }
}
