import crypto from 'crypto';

import {SQS} from '@aws-sdk/client-sqs';
import {WebSocket} from 'ws';

import {ExecutionParams} from '../../types/compilation/compilation.interfaces.js';
import {BasicExecutionResult, ExecutableExecutionOptions} from '../../types/execution/execution.interfaces.js';
import {CompilationEnvironment} from '../compilation-env.js';
import {logger} from '../logger.js';
import {PropertyGetter} from '../properties.interfaces.js';
import {getHash} from '../utils.js';

import {IExecutionEnvironment} from './execution-env.interfaces.js';
import {ExecutionTriple} from './execution-triple.js';

type RemoteExecutionMessage = {
    guid: string;
    hash: string;
    params: ExecutionParams;
};

class SqsExecuteQueue {
    private sqs: SQS;
    private queue_url: string;

    constructor(props: PropertyGetter) {
        this.sqs = new SQS();
        this.queue_url = props<string>('execqueue.url', '');
        if (this.queue_url === '') throw new Error('execqueue.url property required');
    }

    async push(triple: ExecutionTriple, message: RemoteExecutionMessage): Promise<any> {
        const body = JSON.stringify(message);
        return this.sqs.sendMessage({
            QueueUrl: this.queue_url + triple.toString(),
            MessageBody: body,
            MessageGroupId: 'default',
            MessageDeduplicationId: getHash(body),
        });
    }
}

class ExecutionResultWaiter {
    private events_url: string;
    private ws: WebSocket;
    private expectClose: boolean = false;

    constructor(props: PropertyGetter) {
        this.events_url = props<string>('execqueue.events_url', '');
        if (this.events_url === '') throw new Error('execqueue.events_url property required');

        this.ws = new WebSocket(this.events_url);
        this.ws.on('error', logger.error);
    }

    async close(): Promise<void> {
        this.expectClose = true;
        this.ws.close();
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
        return new Promise((resolve, reject) => {
            this.ws.on('message', async message => {
                try {
                    const data = JSON.parse(message.toString());
                    resolve(data);
                } catch (e) {
                    reject(e);
                }
            });
            this.ws.on('close', () => {
                if (!this.expectClose) {
                    reject('closed unexpectedly');
                }
            });
        });
    }
}

export class RemoteExecutionEnvironment implements IExecutionEnvironment {
    private packageHash: string;
    private triple: ExecutionTriple;
    private execQueue: SqsExecuteQueue;
    private guid: string;
    private environment: CompilationEnvironment;

    constructor(environment: CompilationEnvironment, triple: ExecutionTriple, executablePackageHash: string) {
        this.environment = environment;
        this.triple = triple;
        this.guid = crypto.randomUUID();
        this.packageHash = executablePackageHash;
        this.execQueue = new SqsExecuteQueue(environment.ceProps);
    }

    async downloadExecutablePackage(hash: string): Promise<void> {
        throw new Error('Method not implemented.');
    }

    async execute(params: ExecutionParams): Promise<BasicExecutionResult> {
        const waiter = new ExecutionResultWaiter(this.environment.ceProps);
        try {
            await waiter.subscribe(this.guid);

            await this.execQueue.push(this.triple, {
                guid: this.guid,
                hash: this.packageHash,
                params: params,
            });

            const result = await waiter.data();
            await waiter.close();

            return result;
        } catch (e) {
            waiter.close();
            logger.error(e);
            throw e;
        }
    }

    async execBinary(
        executable: string,
        executeParameters: ExecutableExecutionOptions,
        homeDir: string,
        extraConfiguration?: any,
    ): Promise<BasicExecutionResult> {
        throw new Error('Method not implemented.');
    }
}
