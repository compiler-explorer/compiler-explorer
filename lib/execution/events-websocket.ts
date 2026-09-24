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

import {WebSocket} from 'ws';

import {CompilationResult} from '../../types/compilation/compilation.interfaces.js';
import {BasicExecutionResult} from '../../types/execution/execution.interfaces.js';
import {logger} from '../logger.js';
import {PropertyGetter} from '../properties.interfaces.js';

export class EventsWsBase {
    protected expectClose = false;
    protected events_url: string;
    protected ws: WebSocket | undefined = undefined;
    protected got_error = false;

    constructor(props: PropertyGetter) {
        this.events_url = props<string>('execqueue.events_url', '');
        if (this.events_url === '') throw new Error('execqueue.events_url property required');
    }

    protected connect() {
        if (!this.ws) {
            this.ws = new WebSocket(this.events_url);
            this.ws.on('error', (e: any) => {
                this.got_error = true;
                logger.error(`Error while trying to communicate with websocket at URL ${this.events_url}`);
                logger.error(e);
            });
        }
    }

    async close(): Promise<void> {
        this.expectClose = true;
        if (this.ws) {
            this.ws.close();
        }
    }
}

export class EventsWsSender extends EventsWsBase {
    async send(guid: string, result: CompilationResult): Promise<void> {
        this.connect();
        return new Promise(resolve => {
            this.ws!.on('open', async () => {
                this.ws!.send(
                    JSON.stringify({
                        guid: guid,
                        ...result,
                    }),
                );
                resolve();
            });
        });
    }
}

export class PersistentEventsSender extends EventsWsBase {
    private messageQueue: Array<{
        guid: string;
        result: CompilationResult;
        resolve: () => void;
        reject: (error: any) => void;
    }> = [];
    private isConnected = false;
    private isConnecting = false;
    private reconnectAttempts = 0;
    private maxReconnectAttempts = 5;
    private reconnectDelay = 1000; // Start with 1 second
    private hasPermanentlyFailed = false;
    private heartbeatInterval: NodeJS.Timeout | undefined;
    private heartbeatIntervalMs = 30000; // 30 seconds
    private pendingAcks = new Map<
        string,
        {
            timeout: NodeJS.Timeout;
            retryCount: number;
            resolve: () => void;
            reject: (error: any) => void;
            messageData: any;
            expiresAtMs?: number;
        }
    >();
    private maxRetries = 3;
    private ackTimeoutMs = 3000;
    private requireAcknowledgments = true;
    // How long the caller waits, measured from when its request was queued. One value for
    // compilations and executions alike, since one sender carries both. Must track ce-router's
    // own request deadline: a result produced after it reaches nobody.
    private requestDeadlineMs: number;

    constructor(props: PropertyGetter, requireAcknowledgments = true) {
        super(props);
        this.requireAcknowledgments = requireAcknowledgments;
        this.requestDeadlineMs = props<number>('execqueue.request_deadline_ms', 60000);
        this.connect();
    }

    isReadyForNewMessages(): boolean {
        if (!this.requireAcknowledgments) {
            return this.isConnected && this.ws?.readyState === WebSocket.OPEN;
        }
        return this.isConnected && this.ws?.readyState === WebSocket.OPEN && this.pendingAcks.size === 0;
    }

    getPendingAckCount(): number {
        return this.pendingAcks.size;
    }

    hasFailedPermanently(): boolean {
        return this.hasPermanentlyFailed;
    }

    protected override connect(): void {
        if (this.isConnecting || this.isConnected) {
            return;
        }

        this.isConnecting = true;
        this.ws = new WebSocket(this.events_url);

        this.ws.on('open', () => {
            this.isConnected = true;
            this.isConnecting = false;
            this.reconnectAttempts = 0;
            this.reconnectDelay = 1000;
            logger.info(`Persistent WebSocket connection established to ${this.events_url}`);

            this.startHeartbeat();
            this.processQueuedMessages();
            this.retryPendingAcknowledgments();
        });

        this.ws.on('error', (error: any) => {
            this.got_error = true;
            this.isConnected = false;
            this.isConnecting = false;
            logger.error(`Persistent WebSocket error for URL ${this.events_url}:`, error);
            this.scheduleReconnect();
        });

        this.ws.on('close', () => {
            this.isConnected = false;
            this.isConnecting = false;
            this.stopHeartbeat();

            if (!this.expectClose) {
                logger.warn(`Persistent WebSocket connection closed unexpectedly for ${this.events_url}`);
                this.pauseAckTimeouts();
                this.scheduleReconnect();
            }
        });

        this.ws.on('message', (data: any) => {
            try {
                const message = JSON.parse(data.toString());
                if (message.type === 'ack' && message.guid) {
                    this.handleAcknowledgment(message.guid);
                }
            } catch (error) {
                logger.warn('Failed to parse WebSocket message:', error);
            }
        });

        this.ws.on('pong', () => {});
    }

    private startHeartbeat(): void {
        this.heartbeatInterval = setInterval(() => {
            if (this.ws?.readyState === WebSocket.OPEN) {
                this.ws.ping();
            }
        }, this.heartbeatIntervalMs);
    }

    private stopHeartbeat(): void {
        if (this.heartbeatInterval) {
            clearInterval(this.heartbeatInterval);
            this.heartbeatInterval = undefined;
        }
    }

    private scheduleReconnect(): void {
        if (this.expectClose || this.reconnectAttempts >= this.maxReconnectAttempts) {
            logger.error(
                `Max websocket reconnection attempts (${this.maxReconnectAttempts}) reached for ${this.events_url}`,
            );
            this.hasPermanentlyFailed = true;
            const error = new Error('WebSocket connection failed permanently');
            this.rejectQueuedMessages(error);
            // Their timeouts were cleared when the socket closed and nothing will reschedule them, so without
            // this they stay in the map forever and isReadyForNewMessages() never returns true again.
            this.rejectPendingAcks(error);
            return;
        }

        const delay = this.reconnectDelay * 2 ** this.reconnectAttempts; // Exponential backoff
        this.reconnectAttempts++;

        logger.info(
            `Scheduling websocket reconnect attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts} in ${delay}ms`,
        );

        setTimeout(() => {
            if (!this.expectClose) {
                this.connect();
            }
        }, delay);
    }

    private processQueuedMessages(): void {
        while (this.messageQueue.length > 0 && this.isConnected) {
            const message = this.messageQueue.shift();
            if (message && this.ws?.readyState === WebSocket.OPEN) {
                try {
                    this.ws.send(
                        JSON.stringify({
                            guid: message.guid,
                            ...message.result,
                        }),
                    );
                    message.resolve();
                } catch (error) {
                    message.reject(error);
                }
            }
        }
    }

    private rejectQueuedMessages(error: Error): void {
        while (this.messageQueue.length > 0) {
            const message = this.messageQueue.shift();
            if (message) {
                message.reject(error);
            }
        }
    }

    private rejectPendingAcks(error: Error): void {
        for (const [, pending] of this.pendingAcks.entries()) {
            clearTimeout(pending.timeout);
            pending.reject(error);
        }
        this.pendingAcks.clear();
    }

    private handleAcknowledgment(guid: string): void {
        const pending = this.pendingAcks.get(guid);
        if (pending) {
            clearTimeout(pending.timeout);
            this.pendingAcks.delete(guid);
            pending.resolve();
            logger.debug(`Received acknowledgment for ${guid}`);
        }
    }

    /**
     * Whether a result can still reach anyone.
     *
     * An unacknowledged result is usually worth resending - the subscriber may be a router
     * that is reconnecting, and its subscriptions come back with it. Once the requester's own
     * deadline has passed there is nobody left to reconnect for, so further retries cannot
     * succeed, and this worker pulls no new work until the result stops being outstanding.
     */
    private hasExpired(expiresAtMs: number | undefined): boolean {
        return expiresAtMs !== undefined && Date.now() >= expiresAtMs;
    }

    private setupAckTimeout(
        guid: string,
        messageData: any,
        resolve: () => void,
        reject: (error: any) => void,
        expiresAtMs?: number,
    ): void {
        const timeout = setTimeout(() => {
            const pending = this.pendingAcks.get(guid);
            if (pending) {
                pending.retryCount++;
                if (this.hasExpired(expiresAtMs)) {
                    logger.warn(`No acknowledgment for ${guid} and its deadline has passed, giving up`);
                    this.pendingAcks.delete(guid);
                    reject(new Error(`Gave up on ${guid}: the requester's deadline passed`));
                } else if (pending.retryCount < this.maxRetries) {
                    logger.warn(`No acknowledgment for ${guid}, retry ${pending.retryCount}/${this.maxRetries}`);
                    this.sendWithRetry(guid, messageData, pending.retryCount, resolve, reject, expiresAtMs);
                } else {
                    logger.error(`Max retries (${this.maxRetries}) reached for ${guid}, giving up`);
                    this.pendingAcks.delete(guid);
                    reject(new Error(`Failed to receive acknowledgment after ${this.maxRetries} retries`));
                }
            }
        }, this.ackTimeoutMs);

        this.pendingAcks.set(guid, {
            timeout,
            retryCount: 0,
            resolve,
            reject,
            messageData,
            expiresAtMs,
        });
    }

    private sendWithRetry(
        guid: string,
        messageData: any,
        retryCount: number,
        resolve: () => void,
        reject: (error: any) => void,
        expiresAtMs?: number,
    ): void {
        if (!this.isConnected || this.ws?.readyState !== WebSocket.OPEN) {
            reject(new Error('WebSocket not connected'));
            return;
        }

        try {
            this.ws.send(JSON.stringify(messageData));

            const timeout = setTimeout(() => {
                const pending = this.pendingAcks.get(guid);
                if (pending) {
                    if (this.hasExpired(expiresAtMs)) {
                        logger.warn(`No acknowledgment for ${guid} and its deadline has passed, giving up`);
                        this.pendingAcks.delete(guid);
                        reject(new Error(`Gave up on ${guid}: the requester's deadline passed`));
                    } else if (retryCount < this.maxRetries) {
                        logger.warn(`No acknowledgment for ${guid}, retry ${retryCount + 1}/${this.maxRetries}`);
                        this.sendWithRetry(guid, messageData, retryCount + 1, resolve, reject, expiresAtMs);
                    } else {
                        logger.error(`Max retries (${this.maxRetries}) reached for ${guid}, giving up`);
                        this.pendingAcks.delete(guid);
                        reject(new Error(`Failed to receive acknowledgment after ${this.maxRetries} retries`));
                    }
                }
            }, this.ackTimeoutMs);

            this.pendingAcks.set(guid, {
                timeout,
                retryCount,
                resolve,
                reject,
                messageData,
                expiresAtMs,
            });
        } catch (error) {
            reject(error);
        }
    }

    private pauseAckTimeouts(): void {
        for (const [, pending] of this.pendingAcks.entries()) {
            clearTimeout(pending.timeout);
        }
    }

    private retryPendingAcknowledgments(): void {
        for (const [guid, pending] of this.pendingAcks.entries()) {
            logger.info(`Retrying pending acknowledgment for ${guid} after reconnection`);
            try {
                if (this.ws?.readyState === WebSocket.OPEN) {
                    this.ws.send(JSON.stringify(pending.messageData));

                    // Reset timeout for this message
                    const timeout = setTimeout(() => {
                        const stillPending = this.pendingAcks.get(guid);
                        if (stillPending) {
                            if (stillPending.retryCount < this.maxRetries) {
                                logger.warn(
                                    `No acknowledgment for ${guid} after reconnection, retry ${stillPending.retryCount + 1}/${this.maxRetries}`,
                                );
                                this.sendWithRetry(
                                    guid,
                                    pending.messageData,
                                    stillPending.retryCount + 1,
                                    pending.resolve,
                                    pending.reject,
                                );
                            } else {
                                logger.error(
                                    `Max retries (${this.maxRetries}) reached for ${guid} after reconnection, giving up`,
                                );
                                this.pendingAcks.delete(guid);
                                pending.reject(
                                    new Error(`Failed to receive acknowledgment after ${this.maxRetries} retries`),
                                );
                            }
                        }
                    }, this.ackTimeoutMs);

                    pending.timeout = timeout;
                }
            } catch (error) {
                logger.error(`Failed to retry pending acknowledgment for ${guid}:`, error);
                this.pendingAcks.delete(guid);
                pending.reject(error);
            }
        }
    }

    /**
     * @param sentTimestampMs When the request was queued, from the SQS SentTimestamp. Retries
     *     stop once the deadline measured from it has passed, since a result can no longer
     *     reach anyone; the acknowledgement itself is still asked for as usual.
     */
    async send(guid: string, result: CompilationResult, sentTimestampMs?: number): Promise<void> {
        const expiresAtMs = sentTimestampMs === undefined ? undefined : sentTimestampMs + this.requestDeadlineMs;
        return new Promise((resolve, reject) => {
            if (this.isConnected && this.ws?.readyState === WebSocket.OPEN) {
                const messageData = {
                    guid: guid,
                    ...result,
                };
                if (this.requireAcknowledgments) {
                    this.setupAckTimeout(guid, messageData, resolve, reject, expiresAtMs);
                }

                try {
                    this.ws.send(JSON.stringify(messageData));

                    if (!this.requireAcknowledgments) {
                        resolve();
                    }
                } catch (error) {
                    if (this.requireAcknowledgments) {
                        // Don't immediately fail - let the acknowledgment timeout handle retries
                        // A failed send means we won't get an acknowledgment, so the timeout will trigger retries
                        logger.warn(
                            `Initial send failed for ${guid}, letting acknowledgment timeout handle retries:`,
                            error,
                        );
                    } else {
                        logger.warn(`Send failed for ${guid} (no acks required):`, error);
                        reject(error);
                    }
                }
            } else {
                // Queue the message for when connection is available
                this.messageQueue.push({guid, result, resolve, reject});

                // Ensure we're trying to connect
                if (!this.isConnecting && !this.isConnected) {
                    this.connect();
                }
            }
        });
    }

    override async close(): Promise<void> {
        this.expectClose = true;
        this.stopHeartbeat();

        // Only clear pending acknowledgments if this is an intentional close
        this.rejectPendingAcks(new Error('WebSocket connection closing'));

        // Reject any queued messages
        this.rejectQueuedMessages(new Error('WebSocket connection closing'));

        if (this.ws) {
            this.ws.close();
            this.ws = undefined;
        }
    }
}

export class EventsWsWaiter extends EventsWsBase {
    private timeout: number;

    constructor(props: PropertyGetter) {
        super(props);

        // binaryExecTimeoutMs + 2500 to allow for some generous network latency between completion and receiving the result
        this.timeout = props<number>('binaryExecTimeoutMs', 10000) + 2500;
    }

    async subscribe(guid: string): Promise<void> {
        this.connect();
        return new Promise((resolve, reject) => {
            const errorCheck = setInterval(() => {
                if (this.got_error) {
                    reject();
                }
            }, 500);

            this.ws!.on('open', async () => {
                this.ws!.send(`subscribe: ${guid}`);
                clearInterval(errorCheck);
                resolve();
            });
        });
    }

    async data(): Promise<BasicExecutionResult> {
        let runningTime = 0;
        return new Promise((resolve, reject) => {
            const t = setInterval(() => {
                runningTime = runningTime + 1000;
                if (runningTime > this.timeout) {
                    clearInterval(t);
                    reject('Remote execution timed out without returning a result');
                }
            }, 1000);

            this.ws!.on('message', async (message: any) => {
                clearInterval(t);
                try {
                    const data = JSON.parse(message.toString());
                    resolve(data);
                } catch (e) {
                    reject(e);
                }
            });

            this.ws!.on('close', () => {
                clearInterval(t);
                if (!this.expectClose) {
                    reject('Unable to complete remote execution due to unexpected situation');
                }
            });
        });
    }
}
