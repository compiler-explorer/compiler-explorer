import {ResultLine} from '../resultline/resultline.interfaces';

export type FilenameTransformFunc = (filename: string) => string;

export type UnprocessedExecResult = {
    code: number;
    okToCache: boolean;
    filenameTransform: FilenameTransformFunc;
    stdout: string;
    stderr: string;
    execTime: string;
    timedOut: boolean;
    languageId?: string;
};

export type TypicalExecutionFunc = (
    executable: string,
    args: string[],
    execOptions: object,
) => Promise<UnprocessedExecResult>;

export type BasicExecutionResult = {
    code: number;
    okToCache: boolean;
    filenameTransform: FilenameTransformFunc;
    stdout: ResultLine[];
    stderr: ResultLine[];
    execTime: string;
    timedOut: boolean;
};

export type ExecutableExecutionOptions = {
    args: string[];
    stdin: string;
    ldPath: string[];
    env: any;
};
