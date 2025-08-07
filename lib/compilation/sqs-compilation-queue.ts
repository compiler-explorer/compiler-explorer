// Copyright (c) 2025, Compiler Explorer Authors
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
import {PersistentEventsSender} from '../execution/events-websocket.js';
import {logger} from '../logger.js';
import {PropertyGetter} from '../properties.interfaces.js';
import {SentryCapture} from '../sentry.js';
import {parseCompilationRequest, sqsMessageToCompilationRequestData} from './compilation-request-parser.js';

export type RemoteCompilationRequest = {
    guid: string;
    compilerId: string;
    source: string;
    options: any;
    backendOptions: any;
    filters: any;
    bypassCache: any;
    tools: any;
    executeParameters: any;
    libraries: any[];
    lang: string;
    files: any[];
    isCMake?: boolean;
    queueTimeMs?: number;
};

export class SqsCompilationQueueBase {
    protected sqs: SQS;
    protected readonly queue_url: string;

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
                WaitTimeSeconds: 20, // Long polling - wait up to 20 seconds for a message
                MessageSystemAttributeNames: ['SentTimestamp'],
            });
        } catch (e) {
            logger.error(`Error retrieving compilation message from queue with URL: ${url}`);
            throw e;
        }
    }

    async pop(): Promise<RemoteCompilationRequest | undefined> {
        const url = this.queue_url;

        let queued_messages;
        try {
            queued_messages = await this.receiveMsg(url);
        } catch (receiveError) {
            logger.error(
                `SQS receiveMsg failed: ${receiveError instanceof Error ? receiveError.message : String(receiveError)}`,
            );
            throw receiveError;
        }

        if (queued_messages.Messages && queued_messages.Messages.length === 1) {
            const queued_message = queued_messages.Messages[0];

            try {
                if (queued_message.Body) {
                    const json = queued_message.Body;
                    let parsed;
                    try {
                        parsed = JSON.parse(json);
                    } catch (parseError) {
                        logger.error(
                            `JSON.parse failed: ${parseError instanceof Error ? parseError.message : String(parseError)}`,
                        );
                        throw parseError;
                    }

                    if (parsed.options && typeof parsed.options === 'object' && !Array.isArray(parsed.options)) {
                        if (parsed.options.userArguments && typeof parsed.options.userArguments === 'string') {
                            parsed.options = parsed.options.userArguments.split(' ').filter(Boolean);
                        } else {
                            parsed.options = [];
                        }
                    }

                    // Calculate queue time from SentTimestamp
                    const sentTimestamp = queued_message.Attributes?.SentTimestamp;
                    if (sentTimestamp) {
                        const queueTimeMs = Date.now() - Number.parseInt(sentTimestamp, 10);
                        parsed.queueTimeMs = queueTimeMs;
                    }

                    return parsed as RemoteCompilationRequest;
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
    persistentSender: PersistentEventsSender,
    guid: string,
    result: CompilationResult,
) {
    try {
        const basicResult = {
            ...result,
            okToCache: result.okToCache ?? false,
            filenameTransform: (f: string) => f,
            execTime: 0,
        };

        await persistentSender.send(guid, basicResult);
    } catch (error) {
        logger.error('WebSocket send error:', error);
    }
}

async function doOneCompilation(
    queue: SqsCompilationWorkerMode,
    compilationEnvironment: CompilationEnvironment,
    persistentSender: PersistentEventsSender,
) {
    const msg = await queue.pop();

    if (msg?.guid) {
        const startTime = Date.now();
        const compilationType = msg.isCMake ? 'cmake' : 'compile';

        try {
            const compiler = compilationEnvironment.findCompiler(msg.lang as any, msg.compilerId);
            if (!compiler) {
                throw new Error(`Compiler with ID ${msg.compilerId} not found for language ${msg.lang}`);
            }

            const requestData = sqsMessageToCompilationRequestData(msg);
            const parsedRequest = parseCompilationRequest(requestData, compiler);

            let result: CompilationResult;

            // Local compilation only
            if (msg.isCMake) {
                result = await compiler.cmake(msg.files || [], parsedRequest, parsedRequest.bypassCache);
            } else {
                result = await compiler.compile(
                    parsedRequest.source,
                    parsedRequest.options,
                    parsedRequest.backendOptions,
                    parsedRequest.filters,
                    parsedRequest.bypassCache,
                    parsedRequest.tools,
                    parsedRequest.executeParameters,
                    parsedRequest.libraries,
                    msg.files || [],
                );
            }

            // Add queue time to result if available
            if (msg.queueTimeMs !== undefined) {
                result.queueTime = msg.queueTimeMs;
            }

            await sendCompilationResultViaWebsocket(persistentSender, msg.guid, result);

            const endTime = Date.now();
            const duration = endTime - startTime;
            logger.info(`Completed ${compilationType} request ${msg.guid} in ${duration}ms`);
        } catch (e: any) {
            const endTime = Date.now();
            const duration = endTime - startTime;
            logger.error(`Failed ${compilationType} request ${msg.guid} after ${duration}ms:`, e);

            // Create a more descriptive error message
            let errorMessage = 'Internal error during compilation';
            if (e.message) {
                errorMessage = e.message;
            } else if (typeof e === 'string') {
                errorMessage = e;
            }

            const errorResult: CompilationResult = {
                code: -1,
                stderr: [{text: errorMessage}],
                stdout: [],
                okToCache: false,
                timedOut: false,
                inputFilename: '',
                asm: [],
                tools: [],
            };

            // Add queue time to error result if available
            if (msg.queueTimeMs !== undefined) {
                errorResult.queueTime = msg.queueTimeMs;
            }

            await sendCompilationResultViaWebsocket(persistentSender, msg.guid, errorResult);
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
    const pollIntervalMs = ceProps<number>('compilequeue.poll_interval_ms', 50);

    // Create persistent WebSocket sender
    const execqueueEventsUrl = compilationEnvironment.ceProps('execqueue.events_url', '');
    const compilequeueEventsUrl = compilationEnvironment.ceProps('compilequeue.events_url', '');
    const eventsUrl = compilequeueEventsUrl || execqueueEventsUrl;

    if (!eventsUrl) {
        throw new Error('No events URL configured - need either compilequeue.events_url or execqueue.events_url');
    }

    const compilationEventsProps = (key: string, defaultValue?: any) => {
        if (key === 'execqueue.events_url') {
            return eventsUrl;
        }
        return compilationEnvironment.ceProps(key, defaultValue);
    };

    const persistentSender = new PersistentEventsSender(compilationEventsProps);

    // Handle graceful shutdown
    const shutdown = async () => {
        logger.info('Shutting down compilation worker - closing persistent WebSocket connection');
        await persistentSender.close();
        process.exit(0);
    };

    process.on('SIGINT', shutdown);
    process.on('SIGTERM', shutdown);

    logger.info(`Starting ${numThreads} compilation worker threads with ${pollIntervalMs}ms poll interval`);

    for (let i = 0; i < numThreads; i++) {
        const doCompilationWork = async () => {
            try {
                await doOneCompilation(queue, compilationEnvironment, persistentSender);
            } catch (error) {
                logger.error('Error in compilation worker thread:', error);
                SentryCapture(error, 'compilation worker thread error');
            }
            setTimeout(doCompilationWork, pollIntervalMs);
        };
        setTimeout(doCompilationWork, 1500 + i * 30);
    }
}
