// Copyright (c) 2022, Compiler Explorer Authors
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

import {ExecutionOptions} from '../compilation/compilation.interfaces.js';
import {ResultLine} from '../resultline/resultline.interfaces.js';

export type FilenameTransformFunc = (filename: string) => string;

export type UnprocessedExecResult = {
    code: number;
    okToCache: boolean;
    filenameTransform: FilenameTransformFunc;
    stdout: string;
    stderr: string;
    execTime: number;
    timedOut: boolean;
    languageId?: string;
    truncated: boolean;
};

export type TypicalExecutionFunc = (
    executable: string,
    args: string[],
    execOptions: ExecutionOptions,
) => Promise<UnprocessedExecResult>;

export type BasicExecutionResult = {
    code: number;
    okToCache: boolean;
    filenameTransform: FilenameTransformFunc;
    stdout: ResultLine[];
    stderr: ResultLine[];
    execTime: number;
    processExecutionResultTime?: number;
    timedOut: boolean;
    languageId?: string;
    truncated?: boolean;
};

export enum RuntimeToolType {
    env = 'env',
    heaptrack = 'heaptrack',
    libsegfault = 'libsegfault',
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
