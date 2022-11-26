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

import {BuildEnvDownloadInfo} from '../../lib/buildenvsetup/buildenv.interfaces';
import {IAsmParser} from '../../lib/parsers/asm-parser.interfaces';
import {CompilerInfo} from '../compiler.interfaces';
import {BasicExecutionResult} from '../execution/execution.interfaces';
import {ResultLine} from '../resultline/resultline.interfaces';

import {LLVMOptPipelineOutput} from './llvm-opt-pipeline-output.interfaces';

export type CompilationResult = {
    code: number;
    timedOut: boolean;
    buildResult?: BuildResult;
    buildsteps?: BuildStep[];
    inputFilename?: string;
    asm?: ResultLine[];
    devices?: Record<string, CompilationResult>;
    stdout: ResultLine[];
    stderr: ResultLine[];
    didExecute?: boolean;
    execResult?: {
        stdout?: ResultLine[];
        stderr?: ResultLine[];
        code: number;
        didExecute: boolean;
        buildResult?: BuildResult;
        execTime?: number;
    };
    hasGnatDebugOutput?: boolean;
    gnatDebugOutput?: ResultLine[];
    hasGnatDebugTreeOutput?: boolean;
    gnatDebugTreeOutput?: ResultLine[];
    tools?: any;
    dirPath?: string;
    compilationOptions?: string[];
    downloads?: BuildEnvDownloadInfo[];
    gccDumpOutput?: any;
    languageId?: string;

    hasPpOutput?: boolean;
    ppOutput?: any;

    hasOptOutput?: boolean;
    optOutput?: any;
    optPath?: string;

    hasAstOutput?: boolean;
    astOutput?: any;

    hasIrOutput?: boolean;
    irOutput?: any;

    hasLLVMOptPipelineOutput?: boolean;
    llvmOptPipelineOutput?: LLVMOptPipelineOutput | string;

    hasRustMirOutput?: boolean;
    rustMirOutput?: any;

    hasRustMacroExpOutput?: boolean;
    rustMacroExpOutput?: any;

    hasRustHirOutput?: boolean;
    rustHirOutput?: any;

    hasHaskellCoreOutput?: boolean;
    haskellCoreOutput?: any;

    hasHaskellStgOutput?: boolean;
    haskellStgOutput?: any;

    hasHaskellCmmOutput?: boolean;
    haskellCmmOutput?: any;

    forceBinaryView?: boolean;

    bbcdiskimage?: string;
    speccytape?: string;
    miraclesms?: string;
    jsnesrom?: string;

    hints?: string[];

    retreivedFromCache?: boolean;
    retreivedFromCacheTime?: number;
    packageDownloadAndUnzipTime?: number;
    execTime?: number | string;
    processExecutionResultTime?: number;
    objdumpTime?: number;
    parsingTime?: number;
};

export type ExecutionOptions = {
    timeoutMs?: number;
    maxErrorOutput?: number;
    env?: any;
    wrapper?: any;
    maxOutput?: number;
    ldPath?: string[];
    appHome?: string;
    customCwd?: string;
    // Stdin
    input?: any;
    killChild?: () => void;
};

export type BuildResult = CompilationResult & {
    downloads: BuildEnvDownloadInfo[];
    executableFilename: string;
    compilationOptions: string[];
};

export type BuildStep = BasicExecutionResult & {
    compilationOptions: string[];
    step: string;
};

export type CompilationInfo = {
    mtime: Date | null;
    compiler: CompilerInfo & Record<string, unknown>;
    args: string[];
    options: ExecutionOptions;
    outputFilename: string;
    executableFilename: string;
    asmParser: IAsmParser;
    inputFilename?: string;
    dirPath?: string;
};

export type CompilationCacheKey = {
    mtime: any;
    compiler: any;
    args: string[];
    options: ExecutionOptions;
};

export type CustomInputForTool = {
    inputFilename: string;
    dirPath: string;
    outputFilename: string;
};
