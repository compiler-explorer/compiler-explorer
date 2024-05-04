import {ExecutionParams} from '../../types/compilation/compilation.interfaces.js';
import {
    BasicExecutionResult,
    ExecutableExecutionOptions,
    UnprocessedExecResult,
} from '../../types/execution/execution.interfaces.js';

export interface IExecutionEnvironment {
    downloadExecutablePackage(hash: string): Promise<void>;
    execute(params: ExecutionParams): Promise<UnprocessedExecResult>;
    execBinary(
        executable: string,
        executeParameters: ExecutableExecutionOptions,
        homeDir: string,
        extraConfiguration?: any,
    ): Promise<BasicExecutionResult>;
}

export class ExecutablePackageCacheMiss extends Error {}
