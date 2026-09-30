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

import {S3} from '@aws-sdk/client-s3';
import {SQS} from '@aws-sdk/client-sqs';
import {Counter, Histogram} from 'prom-client';

import {
    CompilationResult,
    FiledataPair,
    TEMP_STORAGE_TTL_DAYS,
    WEBSOCKET_SIZE_THRESHOLD,
} from '../../types/compilation/compilation.interfaces.js';
import type {BuildSystemDriver} from '../build-systems/index.js';
import {cmakeBuildSystem, getBuildSystem} from '../build-systems/index.js';
import {CompilationEnvironment} from '../compilation-env.js';
import {PersistentEventsSender} from '../execution/events-websocket.js';
import {CompileHandler} from '../handlers/compile.js';
import {logger} from '../logger.js';
import {PropertyGetter} from '../properties.interfaces.js';
import {SentryCapture} from '../sentry.js';
import {KnownBuildMethod} from '../stats.js';

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
    /** Which build system to build the project with, if any. Supersedes isCMake. */
    buildSystem?: string;
    /** The original, CMake-only spelling of buildSystem. Still sent by producers we don't deploy in lockstep with. */
    isCMake?: boolean;
    queueTimeMs?: number;
    /** SQS SentTimestamp, so the result sender can tell when the caller stops waiting. */
    sentTimestampMs?: number;
    headers: Record<string, string | string[]>;
    queryStringParameters: Record<string, string>;
};

export type S3OverflowMessage = {
    type: 's3-overflow';
    guid: string;
    compilerId: string;
    s3Bucket: string;
    s3Key: string;
    originalSize: number;
    timestamp: string;
};

const queueWaitHistogram = new Histogram({
    name: 'ce_sqs_compilation_queue_wait_seconds',
    help: 'Time a compilation request spent in SQS before a worker collected it',
    buckets: [0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 40, 60],
});

const sqsCompileCounter = new Counter({
    name: 'ce_sqs_compilations_total',
    help: 'Number of SQS compilations',
    labelNames: ['language'],
});

const sqsExecuteCounter = new Counter({
    name: 'ce_sqs_executions_total',
    help: 'Number of SQS executions',
    labelNames: ['language'],
});

const sqsCmakeCounter = new Counter({
    name: 'ce_sqs_cmake_compilations_total',
    help: 'Number of SQS CMake compilations',
    labelNames: ['language'],
});

const sqsCmakeExecuteCounter = new Counter({
    name: 'ce_sqs_cmake_executions_total',
    help: 'Number of SQS executions after CMake',
    labelNames: ['language'],
});

const sqsProjectBuildCounter = new Counter({
    name: 'ce_sqs_project_build_compilations_total',
    help: 'Number of SQS build system compilations',
    labelNames: ['language', 'build_system'],
});

const sqsProjectBuildExecuteCounter = new Counter({
    name: 'ce_sqs_project_build_executions_total',
    help: 'Number of SQS executions after a build system compilation',
    labelNames: ['language', 'build_system'],
});

/**
 * Which build system a queued request wants, or undefined for a plain single-file compilation. Producers live outside
 * this repo, so both the new `buildSystem` field and the original `isCMake` boolean have to work.
 *
 * A name this worker does not know is not the same as no name at all, and must not be read as one: a producer can be
 * ahead of a worker across a deploy, and taking `cargo` for a plain compilation would compile the project's manifest
 * as source and answer with syntax errors about it. Throws, so it reaches the user as a failed compilation naming the
 * build system, which is what the HTTP route says too.
 */
export function getRequestedBuildSystem(msg: RemoteCompilationRequest): BuildSystemDriver | undefined {
    if (msg.buildSystem) {
        const buildSystem = getBuildSystem(msg.buildSystem);
        if (!buildSystem) throw new Error(`Unknown build system '${msg.buildSystem}'`);
        return buildSystem;
    }
    return msg.isCMake ? cmakeBuildSystem : undefined;
}

/**
 * Whether a queued request's recorded content-type names JSON. Producers record the caller's header verbatim, so it
 * can carry parameters (`application/json; charset=utf-8`) or arrive repeated. This matches what `req.is('json')`
 * decides for the same header on the HTTP route, so a request compiles the same way whichever path it arrived by.
 */
export function isJsonContentType(contentType: string | string[] | undefined): boolean {
    const value = Array.isArray(contentType) ? contentType[0] : contentType;
    return value?.split(';')[0].trim().toLowerCase() === 'application/json';
}

export class SqsCompilationQueueBase {
    protected sqs: SQS;
    protected s3: S3;
    protected readonly queue_url: string;

    constructor(props: PropertyGetter, awsProps: PropertyGetter, appArgs?: {instanceColor?: string}) {
        let queue_url = props<string>('compilequeue.queue_url', '');

        // If instance color is provided, modify the queue URL to include the color
        if (appArgs?.instanceColor && queue_url) {
            // Replace the queue name with color suffix
            // e.g., "staging-compilation-queue.fifo" becomes "staging-compilation-queue-blue.fifo"
            queue_url = queue_url.replace(
                '-compilation-queue.fifo',
                `-compilation-queue-${appArgs.instanceColor}.fifo`,
            );
        }

        this.queue_url = queue_url;

        if (!this.queue_url) {
            throw new Error(
                'Configuration error: compilequeue.queue_url is required when compilequeue.is_worker=true. ' +
                    'Please set the SQS queue URL in your configuration.',
            );
        }

        const region = awsProps<string>('region', '');
        if (!region) {
            throw new Error(
                'Configuration error: AWS region is required when compilequeue.is_worker=true. ' +
                    'Please set the AWS region in your configuration.',
            );
        }

        this.sqs = new SQS({region: region});
        this.s3 = new S3({region: region});
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

    private isS3OverflowMessage(msg: any): msg is S3OverflowMessage {
        return msg && msg.type === 's3-overflow' && msg.s3Bucket && msg.s3Key;
    }

    private async fetchFromS3(bucket: string, key: string): Promise<RemoteCompilationRequest | undefined> {
        try {
            logger.info(`Fetching overflow message from S3: ${bucket}/${key}`);
            const response = await this.s3.getObject({
                Bucket: bucket,
                Key: key,
            });

            if (!response.Body) {
                logger.error(`S3 object ${bucket}/${key} has no body`);
                return undefined;
            }

            const bodyString = await response.Body.transformToString();
            const parsed = JSON.parse(bodyString) as RemoteCompilationRequest;
            logger.info(`Successfully fetched overflow message for ${parsed.guid} from S3`);
            return parsed;
        } catch (error) {
            logger.error(
                `Failed to fetch overflow message from S3: ${error instanceof Error ? error.message : String(error)}`,
            );
            throw error;
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

                    if (this.isS3OverflowMessage(parsed)) {
                        logger.info(
                            `Received S3 overflow message for ${parsed.guid}, original size: ${parsed.originalSize} bytes`,
                        );

                        try {
                            const compilationRequest = await this.fetchFromS3(parsed.s3Bucket, parsed.s3Key);

                            if (compilationRequest) {
                                const sentTimestamp = queued_message.Attributes?.SentTimestamp;
                                if (sentTimestamp) {
                                    const queueTimeMs = Date.now() - Number.parseInt(sentTimestamp, 10);
                                    compilationRequest.queueTimeMs = queueTimeMs;
                                    compilationRequest.sentTimestampMs = Number.parseInt(sentTimestamp, 10);
                                }
                                return compilationRequest;
                            }
                        } catch (s3Error) {
                            logger.error(
                                `Failed to fetch S3 overflow message for ${parsed.guid}: ${s3Error instanceof Error ? s3Error.message : String(s3Error)}`,
                            );
                            throw new Error(
                                `S3 overflow fetch failed for ${parsed.guid}: ${s3Error instanceof Error ? s3Error.message : String(s3Error)}`,
                            );
                        }

                        return undefined;
                    }

                    const sentTimestamp = queued_message.Attributes?.SentTimestamp;
                    if (sentTimestamp) {
                        const queueTimeMs = Date.now() - Number.parseInt(sentTimestamp, 10);
                        parsed.queueTimeMs = queueTimeMs;
                        parsed.sentTimestampMs = Number.parseInt(sentTimestamp, 10);
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

/**
 * Puts a value where the router, or a person investigating, can fetch it later. The key is derived
 * from keyFor, so callers naming it differently get a different object. Returns the key it went
 * under, or undefined if it could not be stored.
 */
async function storeInTempCache(
    compilationEnvironment: CompilationEnvironment,
    keyFor: string,
    value: unknown,
): Promise<string | undefined> {
    try {
        return await compilationEnvironment.tempCachePutWithTTL(
            keyFor,
            JSON.stringify(value),
            TEMP_STORAGE_TTL_DAYS,
            undefined,
        );
    } catch (error) {
        logger.error(`Failed to store ${keyFor} in the temp cache:`, error);
        return undefined;
    }
}

export async function sendCompilationResultViaWebsocket(
    persistentSender: PersistentEventsSender,
    compilationEnvironment: CompilationEnvironment,
    guid: string,
    result: CompilationResult,
    totalTimeMs: number,
    sentTimestampMs?: number,
    request?: unknown,
) {
    try {
        const basicResult = {
            ...result,
            okToCache: result.okToCache ?? false,
            execTime: result.execTime !== undefined ? result.execTime : totalTimeMs,
        };

        const resultSize = JSON.stringify(basicResult).length;

        let webResult;
        let sentAs: string;
        if (resultSize > WEBSOCKET_SIZE_THRESHOLD) {
            // Over this size API Gateway closes the connection rather than refusing the frame, and
            // that connection is shared, so one oversized result costs every other result this
            // worker has in flight. Send the key and let the router fetch the rest.
            if (!result.s3Key) {
                // Whatever produced a result this size was meant to have stored it already, so
                // storing it here is a repair, not the design: worth saying out loud, or the path
                // that skipped it stays invisible. The request goes alongside it, because knowing
                // which one did this is the only way to find the path that skipped it.
                const requestKey = await storeInTempCache(
                    compilationEnvironment,
                    `${guid}_faultyrequest`,
                    request ?? null,
                );
                logger.warn(
                    `Sending ${guid} at ${resultSize} bytes with no s3Key, over the ` +
                        `${WEBSOCKET_SIZE_THRESHOLD} byte threshold: storing it now` +
                        (requestKey ? `, request saved at ${requestKey}` : ''),
                );
            }
            const s3Key = result.s3Key ?? (await storeInTempCache(compilationEnvironment, guid, basicResult));
            if (s3Key) {
                webResult = {
                    s3Key: s3Key,
                    okToCache: basicResult.okToCache,
                    execTime: basicResult.execTime,
                };
                sentAs = 's3Key reference';
            } else {
                logger.error(
                    `Could not store ${guid} at ${resultSize} bytes, which is too large to send: ` +
                        'returning an error to the user instead',
                );
                webResult = {
                    code: -1,
                    stderr: [{text: 'The compilation result was too large to return'}],
                    stdout: [],
                    okToCache: false,
                    timedOut: false,
                    inputFilename: '',
                    asm: [],
                    tools: [],
                    execTime: basicResult.execTime,
                };
                sentAs = 'too-large error';
            }
        } else {
            webResult = basicResult;
            sentAs = 'inline';
        }

        await persistentSender.send(guid, webResult, sentTimestampMs);
        logger.info(
            `Successfully sent compilation result for ${guid} via WebSocket ` +
                `(${resultSize} bytes, ${sentAs}, total time: ${totalTimeMs}ms)`,
        );
    } catch (error) {
        logger.error(`WebSocket send error for ${guid}:`, error);
    }
}

async function doOneCompilation(
    queue: SqsCompilationWorkerMode,
    compilationEnvironment: CompilationEnvironment,
    persistentSender: PersistentEventsSender,
) {
    if (!persistentSender.isReadyForNewMessages()) {
        logger.debug(
            `Skipping message pull - WebSocket not ready or has ${persistentSender.getPendingAckCount()} pending acknowledgments`,
        );
        return;
    }

    const msg = await queue.pop();

    if (msg?.guid) {
        const startTime = Date.now();
        // Named from the request rather than from a resolved driver, because resolving is itself something that can
        // fail, and the logs for that failure want to say which build system was asked for.
        const compilationType = msg.buildSystem ?? (msg.isCMake ? 'cmake' : 'compile');

        // How long this sat in the queue, which is the half of a slow request the worker's own
        // timings cannot show: "Completed in 60s" reads the same whether the compile was slow or
        // nobody collected the message until its requester had already given up.
        const queuedMs = msg.sentTimestampMs === undefined ? undefined : startTime - msg.sentTimestampMs;
        if (queuedMs !== undefined) queueWaitHistogram.observe(queuedMs / 1000);
        logger.info(
            `Picked up ${compilationType} request ${msg.guid}` +
                (queuedMs === undefined ? '' : ` after ${queuedMs}ms queued`),
        );

        try {
            // Inside the try: an unknown build system is reported to the user like any other failed compilation.
            const buildSystem = getRequestedBuildSystem(msg);
            const compiler = compilationEnvironment.findCompiler(msg.lang as any, msg.compilerId);
            if (!compiler) {
                throw new Error(`Compiler with ID ${msg.compilerId} not found for language ${msg.lang}`);
            }

            const isJson = isJsonContentType(msg.headers['content-type']);
            const query = msg.queryStringParameters;

            const parsedRequest = CompileHandler.parseRequestReusable(
                isJson,
                query,
                isJson ? msg : msg.source,
                compiler,
            );

            let result: CompilationResult;
            const files = (msg.files || []) as FiledataPair[];

            if (buildSystem) {
                if (buildSystem.id === 'cmake') sqsCmakeCounter.inc({language: compiler.lang.id});
                sqsProjectBuildCounter.inc({language: compiler.lang.id, build_system: buildSystem.id});
                compilationEnvironment.statsNoter.noteCompilation(
                    compiler.getInfo().id,
                    parsedRequest,
                    files,
                    buildSystem.id,
                );

                result = await compiler.buildProject(buildSystem, files, parsedRequest, parsedRequest.bypassCache);

                if (result.didExecute || result.execResult?.didExecute) {
                    if (buildSystem.id === 'cmake') sqsCmakeExecuteCounter.inc({language: compiler.lang.id});
                    sqsProjectBuildExecuteCounter.inc({language: compiler.lang.id, build_system: buildSystem.id});
                }
            } else {
                sqsCompileCounter.inc({language: compiler.lang.id});
                compilationEnvironment.statsNoter.noteCompilation(
                    compiler.getInfo().id,
                    parsedRequest,
                    files,
                    KnownBuildMethod.Compile,
                );

                result = await compiler.compile(
                    parsedRequest.source,
                    parsedRequest.options,
                    parsedRequest.backendOptions,
                    parsedRequest.filters,
                    parsedRequest.bypassCache,
                    parsedRequest.tools,
                    parsedRequest.executeParameters,
                    parsedRequest.libraries,
                    files,
                );

                if (result.didExecute || result.execResult?.didExecute) {
                    sqsExecuteCounter.inc({language: compiler.lang.id});
                }
            }

            if (msg.queueTimeMs !== undefined) {
                result.queueTime = msg.queueTimeMs;
            }

            const endTime = Date.now();
            const duration = endTime - startTime;

            await sendCompilationResultViaWebsocket(
                persistentSender,
                compilationEnvironment,
                msg.guid,
                result,
                duration,
                msg.sentTimestampMs,
                msg,
            );

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

            if (msg.queueTimeMs !== undefined) {
                errorResult.queueTime = msg.queueTimeMs;
            }

            await sendCompilationResultViaWebsocket(
                persistentSender,
                compilationEnvironment,
                msg.guid,
                errorResult,
                duration,
                msg.sentTimestampMs,
                msg,
            );
        }
    }
}

export function startCompilationWorkerThread(
    ceProps: PropertyGetter,
    awsProps: PropertyGetter,
    compilationEnvironment: CompilationEnvironment,
    appArgs?: {instanceColor?: string},
): () => boolean {
    const queue = new SqsCompilationWorkerMode(ceProps, awsProps, appArgs);
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

    return () => !persistentSender.hasFailedPermanently();
}
