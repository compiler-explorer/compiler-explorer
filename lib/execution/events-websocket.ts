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

import {BasicExecutionResult} from '../../types/execution/execution.interfaces.js';
import {logger} from '../logger.js';
import {PropertyGetter} from '../properties.interfaces.js';

export class EventsWsBase {
    protected expectClose: boolean = false;
    protected events_url: string;
    protected ws: WebSocket;

    constructor(props: PropertyGetter) {
        this.events_url = props<string>('execqueue.events_url', '');
        if (this.events_url === '') throw new Error('execqueue.events_url property required');

        this.ws = new WebSocket(this.events_url);
        this.ws.on('error', e => {
            logger.error(e);
        });
    }

    async close(): Promise<void> {
        this.expectClose = true;
        this.ws.close();
    }
}

export class EventsWsSender extends EventsWsBase {
    async send(guid: string, result: BasicExecutionResult): Promise<void> {
        return new Promise(resolve => {
            this.ws.on('open', async () => {
                this.ws.send(
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

export class EventsWsWaiter extends EventsWsBase {
    private timeout: number;

    constructor(props: PropertyGetter) {
        super(props);

        // binaryExecTimeoutMs + 2500 to allow for some generous network latency between completion and receiving the result
        this.timeout = props<number>('binaryExecTimeoutMs', 10000) + 2500;
    }

    async subscribe(guid: string): Promise<void> {
        return new Promise(resolve => {
            this.ws.on('open', async () => {
                this.ws.send(`subscribe: ${guid}`);
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

            this.ws.on('message', async message => {
                clearInterval(t);
                try {
                    const data = JSON.parse(message.toString());
                    resolve(data);
                } catch (e) {
                    reject(e);
                }
            });

            this.ws.on('close', () => {
                clearInterval(t);
                if (!this.expectClose) {
                    reject('Unable to complete remote execution due to unexpected situation');
                }
            });
        });
    }
}
