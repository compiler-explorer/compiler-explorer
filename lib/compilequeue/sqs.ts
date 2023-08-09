import {SQS} from '@aws-sdk/client-sqs';

import {CompileMessage, CompileQueueResult, ICompileQueue} from './compilequeue.interfaces.js';
import {PropertyGetter} from '../properties.interfaces.js';
import {getHash} from '../utils.js';

export class SqsCompileQueue implements ICompileQueue {
    private sqs: SQS;
    private queue_url: string;
    private result_queue_url: string;
    private results: CompileQueueResult[] = [];

    constructor(props: PropertyGetter) {
        this.sqs = new SQS();
        this.queue_url = props<string>('compilequeue.url', '');
        if (this.queue_url === '') throw new Error('compilequeue.url property required');

        this.result_queue_url = props<string>('compilequeue.result_url', '');
        if (this.result_queue_url === '') throw new Error('compilequeue.url property required');
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

    async pushResult(result: CompileQueueResult) {
        const body = JSON.stringify(result);

        await this.sqs.sendMessage({
            QueueUrl: this.result_queue_url,
            MessageBody: body,
        });
    }

    private async fetchAllResults() {
        const res = await this.sqs.receiveMessage({
            QueueUrl: this.result_queue_url,
            MaxNumberOfMessages: 10,
            VisibilityTimeout: 1,
            WaitTimeSeconds: 1,
        });

        if (res.Messages) {
            for (const queued_message of res.Messages) {
                if (queued_message.ReceiptHandle) {
                    this.sqs.deleteMessage({
                        QueueUrl: this.result_queue_url,
                        ReceiptHandle: queued_message.ReceiptHandle,
                    });
                }

                if (queued_message.Body) {
                    const parsed = JSON.parse(queued_message.Body) as CompileQueueResult;
                    this.results[parsed.requestId] = parsed;
                }
            }
        }
    }

    async popResult(requestId: string): Promise<CompileQueueResult | undefined> {
        if (this.results[requestId]) {
            const res = this.results[requestId];
            delete this.results[requestId];
            return res;
        }

        await this.fetchAllResults();

        if (this.results[requestId]) {
            const res = this.results[requestId];
            delete this.results[requestId];
            return res;
        } else {
            return undefined;
        }
    }
}
