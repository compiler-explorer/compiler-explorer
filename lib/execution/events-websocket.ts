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
        this.timeout = props<number>('binaryExecTimeoutMs', 10000);
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
