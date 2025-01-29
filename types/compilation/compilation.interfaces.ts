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
import {OptRemark} from '../../static/panes/opt-view.interfaces.js';
import type {PPOptions} from '../../static/panes/pp-view.interfaces.js';
import {suCodeEntry} from '../../static/panes/stack-usage-view.interfaces.js';
import {ParsedAsmResultLine} from '../asmresult/asmresult.interfaces.js';
import {CompilerInfo} from '../compiler.interfaces.js';
import {BasicExecutionResult, ConfiguredRuntimeTools} from '../execution/execution.interfaces.js';
import {ParseFiltersAndOutputOptions} from '../features/filters.interfaces.js';
import {InstructionSet} from '../instructionsets.js';
import {SelectedLibraryVersion} from '../libraries/libraries.interfaces.js';
import {ResultLine} from '../resultline/resultline.interfaces.js';
import {Artifact, ToolResult} from '../tool.interfaces.js';

import {CFGResult} from './cfg.interfaces.js';
import {ClangirBackendOptions} from './clangir.interfaces.js';
import {ConfiguredOverrides} from './compiler-overrides.interfaces.js';
import {LLVMIrBackendOptions} from './ir.interfaces.js';
import {OptPipelineBackendOptions, OptPipelineOutput} from './opt-pipeline-output.interfaces.js';

export type ActiveTool = {
    id: string;
    args: string[];
    stdin: string;
};

// This is a legacy type that allows a single string to be passed as args but is otherwise identical to ActiveTool:
export type LegacyCompatibleActiveTool = Exclude<ActiveTool, 'args'> & {args: string | string[]};

export type UnparsedExecutionParams = {
    args?: string | string[];
    stdin?: string;
    runtimeTools?: ConfiguredRuntimeTools;
};

export type ExecutionParams = {
    args?: string[];
    stdin?: string;
    runtimeTools?: ConfiguredRuntimeTools;
};

export type LibsAndOptions = {
    libraries: SelectedLibraryVersion[];
    options: string[];
};

export type GccDumpFlags = {
    gimpleFe: boolean;
    address: boolean;
    slim: boolean;
    raw: boolean;
    details: boolean;
    stats: boolean;
    blocks: boolean;
    vops: boolean;
    lineno: boolean;
    uid: boolean;
    all: boolean;
};

export type GccDumpOptions = {
    opened: boolean;
    pass?: GccDumpViewSelectedPass;
    treeDump?: boolean;
    rtlDump?: boolean;
    ipaDump?: boolean;
    dumpFlags?: GccDumpFlags;
};

export type CompilationRequestOptions = {
    userArguments: string;
    compilerOptions: {
        executorRequest?: boolean;
        skipAsm?: boolean;
        producePp?: PPOptions | null;
        produceAst?: boolean;
        produceGccDump?: GccDumpOptions;
        produceStackUsageInfo?: boolean;
        produceOptInfo?: boolean;
        produceCfg?: {asm: boolean; ir: boolean} | false;
        produceGnatDebugTree?: boolean;
        produceGnatDebug?: boolean;
        produceIr?: LLVMIrBackendOptions | null;
        produceClangir?: ClangirBackendOptions | null;
        produceOptPipeline?: OptPipelineBackendOptions | null;
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
    executeParameters: UnparsedExecutionParams;
    filters: ParseFiltersAndOutputOptions;
    tools: ActiveTool[];
    libraries: SelectedLibraryVersion[];
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

export type PPOutput = {
    numberOfLinesFiltered: number;
    output: string;
};

export type CompilationResult = {
    code: number;
    timedOut: boolean;
    okToCache?: boolean;
    buildResult?: BuildResult;
    buildsteps?: BuildStep[];
    inputFilename?: string;
    // Temp hack until we get all code to agree on type of asm
    asm?: ResultLine[] | string;
    asmSize?: number;
    devices?: Record<string, CompilationResult>;
    stdout: ResultLine[];
    stderr: ResultLine[];
    truncated?: boolean;
    didExecute?: boolean;
    validatorTool?: boolean;
    executableFilename?: string;
    execResult?: CompilationResult;
    gnatDebugOutput?: ResultLine[];
    gnatDebugTreeOutput?: ResultLine[];
    tools?: ToolResult[];
    dirPath?: string;
    compilationOptions?: string[];
    downloads?: BuildEnvDownloadInfo[];
    gccDumpOutput?;
    languageId?: string;
    result?: CompilationResult; // cmake inner result

    ppOutput?: PPOutput;

    optOutput?: OptRemark[];
    optPath?: string;

    stackUsageOutput?: suCodeEntry[];
    stackUsagePath?: string;

    astOutput?: ResultLine[];

    irOutput?: {
        asm: ParsedAsmResultLine[];
        cfg?: CFGResult;
    };
    clangirOutput?: ResultLine[];

    optPipelineOutput?: OptPipelineOutput;

    cfg?: CFGResult;

    rustMirOutput?: ResultLine[];
    rustMacroExpOutput?: ResultLine[];
    rustHirOutput?: ResultLine[];

    haskellCoreOutput?: ResultLine[];
    haskellStgOutput?: ResultLine[];
    haskellCmmOutput?: ResultLine[];

    forceBinaryView?: boolean;

    artifacts?: Artifact[];

    hints?: string[];

    retreivedFromCache?: boolean;
    retreivedFromCacheTime?: number;
    packageDownloadAndUnzipTime?: number;
    execTime?: number;
    processExecutionResultTime?: number;
    objdumpTime?: number;
    parsingTime?: number;

    source?: string; // todo: this is a crazy hack, we should get rid of it

    instructionSet?: InstructionSet;
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

export type ExecutionOptionsWithEnv = ExecutionOptions & {env: Record<string, string>};

export type BuildResult = CompilationResult & {
    downloads: BuildEnvDownloadInfo[];
    executableFilename: string;
    compilationOptions: string[];
    preparedLdPaths?: string[];
    defaultExecOptions?: ExecutionOptions;
    stdout: ResultLine[];
    stderr: ResultLine[];
    code: number;
};

export type Arch = 'x86' | 'x86_64' | null;

export type BuildStep = BasicExecutionResult & {
    compilationOptions: string[];
    step: string;
};

export type CompilationInfo = CacheKey &
    CompilationResult & {
        mtime: Date | null;
        compiler: CompilerInfo & Record<string, unknown>;
        args: string[];
        options: string[];
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

export type SingleFileCacheKey = {
    compiler: any;
    source: string;
    options: string[];
    backendOptions: any;
    filters?: any;
    tools: any[];
    libraries: SelectedLibraryVersion[];
    files: any[];
};

export type CmakeCacheKey = Omit<SingleFileCacheKey, 'tools'> & {
    compiler: CompilerInfo;
    files: FiledataPair[];
    api: string;
};

export type CacheKey = SingleFileCacheKey | CmakeCacheKey;

export type FiledataPair = {
    filename: string;
    contents: string;
};

export type BufferOkFunc = (buffer: Buffer) => boolean;
