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
import {BaseCompiler} from '../base-compiler.js';
import {CompilationEnvironment} from '../compilation-env.js';
import {EventsWsSender} from '../execution/events-websocket.js';
import {ParsedRequest} from '../handlers/compile.js';
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
                WaitTimeSeconds: 20, // Long polling - wait up to 20 seconds for a message
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
    compilationEnvironment: CompilationEnvironment,
    guid: string,
    result: CompilationResult,
) {
    try {
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

        const sender = new EventsWsSender(compilationEventsProps);
        const basicResult = {
            ...result,
            okToCache: result.okToCache ?? false,
            filenameTransform: (f: string) => f,
            execTime: 0,
        };

        await sender.send(guid, basicResult);
        await sender.close();
    } catch (error) {
        logger.error('WebSocket send error:', error);
    }
}

async function executeRemoteCompilation(
    compiler: BaseCompiler,
    msg: RemoteCompilationRequest,
    parsedRequest: ParsedRequest,
): Promise<CompilationResult> {
    const remote = compiler.getRemote();
    if (!remote || !remote.target) {
        throw new Error(`Remote configuration missing for compiler ${compiler.getInfo().id}`);
    }

    const compilationType = msg.isCMake ? 'cmake' : 'compile';
    const endpoint = msg.isCMake ? remote.cmakePath || '/api/compiler/cmake' : remote.path || '/api/compiler/compile';
    const url = new URL(endpoint, remote.target);

    logger.info(`Forwarding ${compilationType} request for ${msg.compilerId} to remote: ${url.href}`);

    const requestBody = {
        source: parsedRequest.source,
        options: {
            userArguments: parsedRequest.options.join(' '),
            compilerOptions: parsedRequest.backendOptions,
            filters: parsedRequest.filters,
            tools: parsedRequest.tools,
            libraries: parsedRequest.libraries,
            executeParameters: parsedRequest.executeParameters,
        },
        lang: msg.lang,
        files: msg.files || [],
        bypassCache: parsedRequest.bypassCache,
    };

    try {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 35000); // 35 second timeout

        const response = await fetch(url.href, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                Accept: 'application/json',
            },
            body: JSON.stringify(requestBody),
            signal: controller.signal,
        });

        clearTimeout(timeout);

        if (!response.ok) {
            const errorText = await response.text();
            let errorMessage = `Remote compilation failed with status ${response.status}`;
            try {
                // Try to parse error response as JSON
                const errorJson = JSON.parse(errorText);
                if (errorJson.error) {
                    errorMessage = errorJson.error;
                } else if (errorJson.message) {
                    errorMessage = errorJson.message;
                }
            } catch {
                // If not JSON, use the text as-is
                errorMessage = errorText || errorMessage;
            }

            // Return error as CompilationResult instead of throwing
            return {
                code: response.status,
                stderr: [{text: errorMessage}],
                stdout: [],
                okToCache: false,
                timedOut: false,
                inputFilename: '',
                asm: [],
                tools: [],
            };
        }

        const result = await response.json();
        return result as CompilationResult;
    } catch (error: any) {
        if (error.name === 'AbortError') {
            return {
                code: -1,
                stderr: [{text: `Remote compilation timeout after 35 seconds for ${msg.compilerId}`}],
                stdout: [],
                okToCache: false,
                timedOut: true,
                inputFilename: '',
                asm: [],
                tools: [],
            };
        }

        // Network errors, connection refused, etc.
        const errorMessage = error.message || 'Unknown remote compilation error';
        logger.error(`Remote compilation error for ${msg.compilerId}:`, error);

        return {
            code: -1,
            stderr: [{text: `Remote compilation error: ${errorMessage}`}],
            stdout: [],
            okToCache: false,
            timedOut: false,
            inputFilename: '',
            asm: [],
            tools: [],
        };
    }
}

async function doOneCompilation(queue: SqsCompilationWorkerMode, compilationEnvironment: CompilationEnvironment) {
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

            // Check if this is a remote compiler
            if (compiler.getRemote()) {
                logger.debug(`Using remote compilation for ${msg.compilerId}`);
                try {
                    result = await executeRemoteCompilation(compiler, msg, parsedRequest);
                } catch (remoteError: any) {
                    // executeRemoteCompilation should normally not throw, but handle it just in case
                    logger.error('Unexpected error in executeRemoteCompilation:', remoteError);
                    result = {
                        code: -1,
                        stderr: [{text: `Internal error: ${remoteError.message || remoteError}`}],
                        stdout: [],
                        okToCache: false,
                        timedOut: false,
                        inputFilename: '',
                        asm: [],
                        tools: [],
                    };
                }
            } else {
                // Local compilation
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
            }

            await sendCompilationResultViaWebsocket(compilationEnvironment, msg.guid, result);

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

            await sendCompilationResultViaWebsocket(compilationEnvironment, msg.guid, {
                code: -1,
                stderr: [{text: errorMessage}],
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
    const pollIntervalMs = ceProps<number>('compilequeue.poll_interval_ms', 50);

    logger.info(`Starting ${numThreads} compilation worker threads with ${pollIntervalMs}ms poll interval`);

    for (let i = 0; i < numThreads; i++) {
        const doCompilationWork = async () => {
            try {
                await doOneCompilation(queue, compilationEnvironment);
            } catch (error) {
                logger.error('Error in compilation worker thread:', error);
                SentryCapture(error, 'compilation worker thread error');
            }
            setTimeout(doCompilationWork, pollIntervalMs);
        };
        setTimeout(doCompilationWork, 1500 + i * 30);
    }
}
