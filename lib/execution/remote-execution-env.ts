import crypto from 'crypto';

import {ExecutionParams} from '../../types/compilation/compilation.interfaces.js';
import {BasicExecutionResult, ExecutableExecutionOptions} from '../../types/execution/execution.interfaces.js';
import {CompilationEnvironment} from '../compilation-env.js';
import {logger} from '../logger.js';

import {EventsWsWaiter} from './events-websocket.js';
import {IExecutionEnvironment} from './execution-env.interfaces.js';
import {ExecutionTriple} from './execution-triple.js';
import {SqsExecuteRequester} from './sqs-execution-queue.js';

export class RemoteExecutionEnvironment implements IExecutionEnvironment {
    private packageHash: string;
    private triple: ExecutionTriple;
    private execQueue: SqsExecuteRequester;
    private guid: string;
    private environment: CompilationEnvironment;

    constructor(environment: CompilationEnvironment, triple: ExecutionTriple, executablePackageHash: string) {
        this.environment = environment;
        this.triple = triple;
        this.guid = crypto.randomUUID();
        this.packageHash = executablePackageHash;
        this.execQueue = new SqsExecuteRequester(environment.ceProps);

        logger.info(
            `RemoteExecutionEnvironment with ${triple.toString()} and ${executablePackageHash} - guid ${this.guid}`,
        );
    }

    async downloadExecutablePackage(hash: string): Promise<void> {
        throw new Error('Method not implemented.');
    }

    private async queueRemoteExecution(params: ExecutionParams) {
        await this.execQueue.push(this.triple, {
            guid: this.guid,
            hash: this.packageHash,
            params: params,
        });
    }

    async execute(params: ExecutionParams): Promise<BasicExecutionResult> {
        const waiter = new EventsWsWaiter(this.environment.ceProps);
        try {
            await waiter.subscribe(this.guid);

            await this.queueRemoteExecution(params);

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
