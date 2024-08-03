import net from 'net';

import {SQS} from '@aws-sdk/client-sqs';

import {PropertyGetter} from '../properties.interfaces.js';
import {S3Bucket} from '../s3-handler.js';
import {getHash} from '../utils.js';

import {CompileMessage, CompileQueueResult, ICompileQueue} from './compilequeue.interfaces.js';

export class SqsCompileQueue implements ICompileQueue {
    private sqs: SQS;
    private queue_url: string;
    private result_bucket: string;
    private result_path: string;
    private s3: S3Bucket;
    private event_service_host: string;
    private event_service_port: number;

    constructor(props: PropertyGetter) {
        this.sqs = new SQS();
        this.queue_url = props<string>('compilequeue.url', '');
        if (this.queue_url === '') throw new Error('compilequeue.url property required');

        this.result_bucket = props<string>('compilequeue.result_bucket', 'storage.godbolt.org');
        this.result_path = props<string>('compilequeue.result_path', 'compilation-results');
        if (this.result_bucket === '') throw new Error('compilequeue.result_bucket property required');
        if (this.result_path === '') throw new Error('compilequeue.result_path property required');

        this.event_service_host = props<string>('compilequeue.event_service_host', '127.0.0.1');
        this.event_service_port = props<number>('compilequeue.event_service_port', 1337);

        this.s3 = new S3Bucket(this.result_bucket, 'us-east-1');
    }

    async pop(): Promise<CompileMessage | undefined> {
        const queued_messages = await this.sqs.receiveMessage({
            QueueUrl: this.queue_url,
            MaxNumberOfMessages: 1,
        });

        if (queued_messages.Messages && queued_messages.Messages.length === 1) {
            const queued_message = queued_messages.Messages[0];

            try {
                if (queued_message.Body) {
                    const json = queued_message.Body;
                    const compileMessage = JSON.parse(json) as CompileMessage;
                    compileMessage.requestId = queued_message.MessageId;
                    return compileMessage;
                } else {
                    return undefined;
                }
            } finally {
                if (queued_message.ReceiptHandle) {
                    await this.sqs.deleteMessage({
                        QueueUrl: this.queue_url,
                        ReceiptHandle: queued_message.ReceiptHandle,
                    });
                }
            }
        }

        return undefined;
    }

    async push(message: CompileMessage): Promise<string | undefined> {
        const body = JSON.stringify(message);
        const result = await this.sqs.sendMessage({
            QueueUrl: this.queue_url,
            MessageBody: body,
            MessageGroupId: 'default',
            MessageDeduplicationId: getHash(body),
        });
        return result.MessageId;
    }

    async pushResult(result: CompileQueueResult): Promise<void> {
        const body = JSON.stringify(result);
        await this.s3.put(result.requestId, Buffer.from(body), this.result_path, {
            redundancy: 'REDUCED_REDUNDANCY',
        });

        return new Promise(resolve => {
            const client = new net.Socket();
            client.connect(this.event_service_port, this.event_service_host, () => {
                client.write('alertNoParam: ' + result.requestId);
                client.destroy();

                // no need to wait for a response
                resolve();
            });
        });
    }

    async popResult(requestId: string): Promise<CompileQueueResult | undefined> {
        return new Promise(resolve => {
            const client = new net.Socket();
            client.connect(this.event_service_port, this.event_service_host, () => {
                client.write('subscribe: ' + requestId);
            });

            client.on('data', async data => {
                const lines = data.toString().split('\n');
                for (const line of lines) {
                    if (line.startsWith('alert: ' + requestId)) {
                        client.destroy();

                        const result = await this.s3.get(requestId, this.result_path);
                        await this.s3.delete(requestId, this.result_path);
                        if (result.data) {
                            resolve(JSON.parse(result.data.toString('utf8')));
                        } else {
                            resolve({
                                status: 404,
                                body: '',
                                dt: new Date(),
                                headers: {},
                                requestId: requestId,
                            });
                        }
                    }
                }
            });
        });
    }
}
