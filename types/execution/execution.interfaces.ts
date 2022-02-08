export type FilenameTransformFunc = (filename: string) => string;

export class UnprocessedExecResult {
    code: number;
    okToCache: boolean;
    filenameTransform: FilenameTransformFunc;
    stdout: string;
    stderr: string;
    execTime: string;
}

export type TypicalExecutionFunc = (executable: string, args: string[], execOptions: object) => UnprocessedExecResult;
