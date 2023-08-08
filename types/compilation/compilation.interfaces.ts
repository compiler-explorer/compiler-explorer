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

import {BuildEnvDownloadInfo} from '../../lib/buildenvsetup/buildenv.interfaces.js';
import {IAsmParser} from '../../lib/parsers/asm-parser.interfaces.js';
import type {GccDumpViewSelectedPass} from '../../static/panes/gccdump-view.interfaces.js';
import type {PPOptions} from '../../static/panes/pp-view.interfaces.js';
import {suCodeEntry} from '../../static/panes/stack-usage-view.interfaces.js';
import {ParsedAsmResultLine} from '../asmresult/asmresult.interfaces.js';
import {CompilerInfo} from '../compiler.interfaces.js';
import {BasicExecutionResult} from '../execution/execution.interfaces.js';
import {ParseFiltersAndOutputOptions} from '../features/filters.interfaces.js';
import {ResultLine} from '../resultline/resultline.interfaces.js';
import {Artifact, ToolResult} from '../tool.interfaces.js';

import {CFGResult} from './cfg.interfaces.js';
import {ConfiguredOverrides} from './compiler-overrides.interfaces.js';
import {LLVMIrBackendOptions} from './ir.interfaces.js';
import {LLVMOptPipelineBackendOptions, LLVMOptPipelineOutput} from './llvm-opt-pipeline-output.interfaces.js';

export type ActiveTools = {
    id: number;
    args: string[];
    stdin: string;
};

export type ExecutionParams = {
    args: string[] | string;
    stdin: string;
};

export type CompileChildLibraries = {
    id: string;
    version: string;
};

export type CompilationRequestOptions = {
    userArguments: string;
    compilerOptions: {
        executorRequest?: boolean;
        skipAsm?: boolean;
        producePp?: PPOptions | null;
        produceAst?: boolean;
        produceGccDump?: {
            opened: boolean;
            pass?: GccDumpViewSelectedPass;
            treeDump?: boolean;
            rtlDump?: boolean;
            ipaDump?: boolean;
            dumpFlags: any;
        };
        produceStackUsageInfo?: boolean;
        produceOptInfo?: boolean;
        produceCfg?: {asm: boolean; ir: boolean} | false;
        produceGnatDebugTree?: boolean;
        produceGnatDebug?: boolean;
        produceIr?: LLVMIrBackendOptions | null;
        produceLLVMOptPipeline?: LLVMOptPipelineBackendOptions | null;
        produceDevice?: boolean;
        produceRustMir?: boolean;
        produceRustMacroExp?: boolean;
        produceRustHir?: boolean;
        produceHaskellCore?: boolean;
        produceHaskellStg?: boolean;
        produceHaskellCmm?: boolean;
        cmakeArgs?: string;
        customOutputFilename?: string;
        overrides?: ConfiguredOverrides;
    };
    executeParameters: ExecutionParams;
    filters: ParseFiltersAndOutputOptions;
    tools: ActiveTools[];
    libraries: CompileChildLibraries[];
};

// Carefully chosen for backwards compatibility
// Compilation will imply exec (this is important for backward compatibility, though there is a world in which it could
// be desirable to only bypass a compilation cache and not the execution pass)
export enum BypassCache {
    None = 0,
    Compilation = 1,
    Execution = 2,
}

export function bypassCompilationCache(value: BypassCache) {
    return value === BypassCache.Compilation;
}

export function bypassExecutionCache(value: BypassCache) {
    return value === BypassCache.Compilation || value === BypassCache.Execution;
}

export type CompilationRequest = {
    source: string;
    compiler: string;
    options: CompilationRequestOptions;
    lang: string | null;
    files: FiledataPair[];
    bypassCache?: BypassCache;
};

export type CompilationResult = {
    code: number;
    timedOut: boolean;
    okToCache?: boolean;
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
    tools?: ToolResult[];
    dirPath?: string;
    compilationOptions?: string[];
    downloads?: BuildEnvDownloadInfo[];
    gccDumpOutput?: any;
    languageId?: string;
    result?: CompilationResult; // cmake inner result

    hasPpOutput?: boolean;
    ppOutput?: any;

    hasOptOutput?: boolean;
    optOutput?: any;
    optPath?: string;

    hasStackUsageOutput?: boolean;
    stackUsageOutput?: suCodeEntry[];
    stackUsagePath?: string;

    hasAstOutput?: boolean;
    astOutput?: any;

    hasIrOutput?: boolean;
    irOutput?: {
        asm: ParsedAsmResultLine[];
        cfg?: CFGResult;
    };

    hasLLVMOptPipelineOutput?: boolean;
    llvmOptPipelineOutput?: LLVMOptPipelineOutput;

    cfg?: CFGResult;

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

    artifacts?: Artifact[];

    hints?: string[];

    retreivedFromCache?: boolean;
    retreivedFromCacheTime?: number;
    packageDownloadAndUnzipTime?: number;
    execTime?: number | string;
    processExecutionResultTime?: number;
    objdumpTime?: number;
    parsingTime?: number;

    source?: string; // todo: this is a crazy hack, we should get rid of it
};

export type ExecutionOptions = {
    timeoutMs?: number;
    maxErrorOutput?: number;
    env?: Record<string, string>;
    wrapper?: any;
    maxOutput?: number;
    ldPath?: string[];
    appHome?: string;
    customCwd?: string;
    createAndUseTempDir?: boolean;
    // Stdin
    input?: any;
    killChild?: () => void;
};

export type BuildResult = CompilationResult & {
    downloads: BuildEnvDownloadInfo[];
    executableFilename: string;
    compilationOptions: string[];
    stdout: ResultLine[];
    stderr: ResultLine[];
    code: number;
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

export type FiledataPair = {
    filename: string;
    contents: string;
};

export type BufferOkFunc = (buffer: Buffer) => boolean;
