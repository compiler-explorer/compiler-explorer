import {ResultLine} from '../resultline/resultline.interfaces.js';

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
    truncated: boolean;
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
    processExecutionResultTime?: number;
    timedOut: boolean;
};

export enum RuntimeToolType {
    env = 'env',
    heaptrack = 'heaptrack',
}

export type RuntimeToolOption = {
    name: string;
    value: string;
};

export type PossibleRuntimeToolOption = {
    name: string;
    possibleValues: string[];
};

export type PossibleRuntimeTool = {
    name: RuntimeToolType;
    description: string;
    possibleOptions: PossibleRuntimeToolOption[];
};
export type PossibleRuntimeTools = PossibleRuntimeTool[];

export type RuntimeToolOptions = RuntimeToolOption[];

export type ConfiguredRuntimeTool = {
    name: RuntimeToolType;
    options: RuntimeToolOptions;
};

export type ConfiguredRuntimeTools = ConfiguredRuntimeTool[];

export type ExecutableExecutionOptions = {
    args: string[];
    stdin: string;
    ldPath: string[];
    env: Record<string, string>;
    runtimeTools?: ConfiguredRuntimeTools;
};
