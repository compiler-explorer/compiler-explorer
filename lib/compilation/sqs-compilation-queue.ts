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
import {parseCompilationRequest, sqsMessageToCompilationRequestData} from './compilation-request-parser.js';

export type RemoteCompilationRequest = {
    guid: string;
    compilerId: string;
    source: string;
    options: any; // This can be either string[] (after our defensive fix) or the original object structure
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
        logger.info('=== SQS POP DEBUG START ===');
        logger.info(`Queue URL: ${url}`);

        let queued_messages;
        try {
            queued_messages = await this.receiveMsg(url);
            logger.info(`SQS receiveMsg SUCCESS - Response type: ${typeof queued_messages}`);
            logger.info(`Response keys: ${Object.keys(queued_messages)}`);
            logger.info(`Has Messages: ${!!queued_messages.Messages}`);
            if (queued_messages.Messages) {
                logger.info(`Messages length: ${queued_messages.Messages.length}`);
            }
        } catch (receiveError) {
            logger.error(
                `SQS receiveMsg FAILED: ${receiveError instanceof Error ? receiveError.message : String(receiveError)}`,
            );
            logger.info('=== SQS POP DEBUG END (ERROR) ===');
            throw receiveError;
        }

        logger.info('=== SQS POP DEBUG END ===');

        if (queued_messages.Messages && queued_messages.Messages.length === 1) {
            const queued_message = queued_messages.Messages[0];

            logger.info('=== SQS MESSAGE STRUCTURE DEBUG ===');
            logger.info(`Full SQS response keys: ${Object.keys(queued_messages)}`);
            logger.info(`Message count: ${queued_messages.Messages.length}`);
            logger.info(`Message keys: ${Object.keys(queued_message)}`);
            logger.info(`MessageId: ${queued_message.MessageId}`);
            logger.info(`ReceiptHandle exists: ${!!queued_message.ReceiptHandle}`);
            logger.info(`Body exists: ${!!queued_message.Body}`);
            logger.info('=== END SQS MESSAGE STRUCTURE DEBUG ===');

            try {
                if (queued_message.Body) {
                    const json = queued_message.Body;
                    logger.info('=== SQS PARSING DEBUG START ===');
                    logger.info(`Raw queued_message keys: ${Object.keys(queued_message)}`);
                    logger.info(`Body type: ${typeof json}, constructor: ${json.constructor.name}`);
                    logger.info(`Body length: ${json.length}`);
                    logger.info(`Body first 1000 chars: ${json.substring(0, 1000)}`);
                    logger.info(`Body last 500 chars: ${json.substring(Math.max(0, json.length - 500))}`);

                    let parsed;
                    try {
                        parsed = JSON.parse(json);
                        logger.info(`Parse SUCCESS - type: ${typeof parsed}, constructor: ${parsed.constructor.name}`);
                        logger.info(`Parsed keys: ${Object.keys(parsed)}`);
                        logger.info(`Has guid: ${!!parsed.guid}, guid value: ${parsed.guid}`);
                        logger.info(`Has options: ${!!parsed.options}, options type: ${typeof parsed.options}`);
                        if (parsed.options) {
                            logger.info(
                                `Options is array: ${Array.isArray(parsed.options)}, length: ${parsed.options.length}`,
                            );
                        }
                    } catch (parseError) {
                        logger.error(
                            `JSON.parse FAILED: ${parseError instanceof Error ? parseError.message : String(parseError)}`,
                        );
                        logger.info('Attempting to log raw body as object properties:');
                        for (let i = 0; i < Math.min(10, json.length); i++) {
                            logger.info(`  json[${i}] = "${json[i]}"`);
                        }
                        throw parseError;
                    }

                    logger.info('=== SQS PARSING DEBUG END ===');

                    // Defensive fix: Convert options object to array if needed
                    if (parsed.options && typeof parsed.options === 'object' && !Array.isArray(parsed.options)) {
                        logger.warn('Converting options from object to array for compatibility');
                        // If it's an object with userArguments field, extract that
                        if (parsed.options.userArguments && typeof parsed.options.userArguments === 'string') {
                            parsed.options = parsed.options.userArguments.split(' ').filter(Boolean);
                        } else {
                            // Otherwise try to extract meaningful values or default to empty array
                            parsed.options = [];
                        }
                        logger.info(`Converted options: ${JSON.stringify(parsed.options)}`);
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
        } else {
            logger.info('=== NO MESSAGES RECEIVED ===');
            if (!queued_messages.Messages) {
                logger.info('No Messages field in response');
            } else {
                logger.info(`Messages array length: ${queued_messages.Messages.length}`);
            }
            logger.info('=== END NO MESSAGES ===');
        }

        return undefined;
    }
}

// Note: This function is now replaced by shared parsing functions in compilation-request-parser.ts

async function sendCompilationResultViaWebsocket(
    compilationEnvironment: CompilationEnvironment,
    guid: string,
    result: CompilationResult,
) {
    logger.info('=== WEBSOCKET SEND DEBUG START ===');
    logger.info(`Sending result for guid: ${guid}`);
    logger.info(`Result keys: ${Object.keys(result)}`);
    logger.info(`Result success: ${result.code === 0}, code: ${result.code}`);
    logger.info(`Result asm length: ${result.asm?.length || 0}`);
    logger.info(`Result stdout length: ${result.stdout?.length || 0}`);
    logger.info(`Result stderr length: ${result.stderr?.length || 0}`);

    try {
        logger.info('Creating EventsWsSender...');
        logger.info('Checking WebSocket configuration...');

        // Check for the configuration properties
        const execqueueEventsUrl = compilationEnvironment.ceProps('execqueue.events_url', '');
        const compilequeueEventsUrl = compilationEnvironment.ceProps('compilequeue.events_url', '');
        logger.info(`execqueue.events_url: ${execqueueEventsUrl}`);
        logger.info(`compilequeue.events_url: ${compilequeueEventsUrl}`);

        // Use compilequeue.events_url if available, otherwise fall back to execqueue.events_url
        const eventsUrl = compilequeueEventsUrl || execqueueEventsUrl;
        logger.info(`Using events URL: ${eventsUrl}`);

        if (!eventsUrl) {
            throw new Error('No events URL configured - need either compilequeue.events_url or execqueue.events_url');
        }

        // Create a custom property getter that uses the compilation queue events URL
        const compilationEventsProps = (key: string, defaultValue?: any) => {
            if (key === 'execqueue.events_url') {
                return eventsUrl;
            }
            return compilationEnvironment.ceProps(key, defaultValue);
        };

        const sender = new EventsWsSender(compilationEventsProps);
        logger.info('EventsWsSender created successfully');

        // Convert CompilationResult to BasicExecutionResult format expected by EventsWsSender
        const basicResult = {
            ...result, // Include all fields from compilation result first
            okToCache: result.okToCache ?? false,
            filenameTransform: (f: string) => f,
            execTime: 0,
        };
        logger.info('Converted result for WebSocket format');
        logger.info(`Basic result keys: ${Object.keys(basicResult)}`);

        logger.info('Calling sender.send()...');
        await sender.send(guid, basicResult);
        logger.info('sender.send() completed successfully');

        logger.info('Closing sender...');
        await sender.close();
        logger.info('Sender closed successfully');
    } catch (error) {
        logger.error('WebSocket send error:', error);
        logger.error('Error details:', {
            message: error instanceof Error ? error.message : String(error),
            stack: error instanceof Error ? error.stack : undefined,
        });
    }
    logger.info('=== WEBSOCKET SEND DEBUG END ===');
}

async function doOneCompilation(queue: SqsCompilationWorkerMode, compilationEnvironment: CompilationEnvironment) {
    const msg = await queue.pop();

    logger.info('=== DOONECOMPILATION DEBUG START ===');
    logger.info(`msg after pop(): type=${typeof msg}, constructor=${msg?.constructor?.name}`);
    if (msg) {
        logger.info(`msg keys: ${Object.keys(msg)}`);
        logger.info(`msg.guid: type=${typeof msg.guid}, value=${msg.guid}`);
        logger.info(`msg.options: type=${typeof msg.options}, isArray=${Array.isArray(msg.options)}`);
        if (msg.options) {
            logger.info(`msg.options keys: ${Object.keys(msg.options)}`);
            logger.info(`msg.options content: ${JSON.stringify(msg.options)}`);
        }
        logger.info(`msg.compilerId: type=${typeof msg.compilerId}, value=${msg.compilerId}`);

        // Test accessing properties to see if they work
        try {
            const testGuid = msg.guid;
            const testOptions = msg.options;
            const testCompilerId = msg.compilerId;
            logger.info(
                `Property access test: guid=${testGuid}, options length=${testOptions?.length}, compilerId=${testCompilerId}`,
            );
        } catch (propError) {
            logger.error(
                `Property access failed: ${propError instanceof Error ? propError.message : String(propError)}`,
            );
        }

        // Test JSON.stringify to see what happens
        try {
            const stringified = JSON.stringify(msg, null, 2);
            logger.info(`JSON.stringify test: first 200 chars: ${stringified.substring(0, 200)}`);
        } catch (stringifyError) {
            logger.error(
                `JSON.stringify failed: ${stringifyError instanceof Error ? stringifyError.message : String(stringifyError)}`,
            );
        }
    }
    logger.info('=== DOONECOMPILATION DEBUG END ===');

    if (msg?.guid) {
        const startTime = Date.now();
        const compilationType = msg.isCMake ? 'cmake' : 'compile';
        logger.info(`Processing ${compilationType} request ${msg.guid}`);
        logger.debug(`Processing ${compilationType} request ${msg.guid} for compiler ${msg.compilerId}`);

        try {
            const compiler = compilationEnvironment.findCompiler(msg.lang as any, msg.compilerId);
            if (!compiler) {
                throw new Error(`Compiler with ID ${msg.compilerId} not found for language ${msg.lang}`);
            }

            // Convert SQS message to the proper structure using shared parsing functions
            const requestData = sqsMessageToCompilationRequestData(msg);
            const parsedRequest = parseCompilationRequest(requestData, compiler);

            let result: CompilationResult;
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
    const pollIntervalMs = ceProps<number>('compilequeue.poll_interval_ms', 50);

    logger.info('Starting compilation worker threads', {
        numThreads,
        pollIntervalMs,
    });

    logger.info(`Starting ${numThreads} compilation worker threads with ${pollIntervalMs}ms poll interval`);

    // Note: With WaitTimeSeconds=20, the receiveMessage call will wait up to 20 seconds
    // for a message to arrive. This means the actual response time is:
    // - Immediate when messages are available (< 100ms)
    // - Up to 20 seconds when queue is empty (cost-effective)
    // The pollIntervalMs only applies between successful message processing or errors.

    for (let i = 0; i < numThreads; i++) {
        const doCompilationWork = async () => {
            try {
                await doOneCompilation(queue, compilationEnvironment);
            } catch (error) {
                logger.error('Error in compilation worker thread:', error);
            }
            // Always reschedule, even after errors
            setTimeout(doCompilationWork, pollIntervalMs);
        };
        setTimeout(doCompilationWork, 1500 + i * 30);
    }
}
