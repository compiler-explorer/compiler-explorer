// Copyright (c) 2023, Compiler Explorer Authors
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

import type {LLVMOptPipelineBackendOptions} from '../../types/compilation/llvm-opt-pipeline-output.interfaces.js';
import type {PPOptions} from './pp-view.interfaces.js';
import type {GccDumpViewSelectedPass} from './gccdump-view.interfaces.js';
import type {FiledataPair} from '../../types/compilation/compilation.interfaces.js';
import type {ConfiguredOverrides} from '../compilation/compiler-overrides.interfaces.js';

export type ActiveTools = {
    id: number;
    args: string[];
    stdin: string;
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
        produceOptInfo?: boolean;
        produceCfg?: boolean;
        produceGnatDebugTree?: boolean;
        produceGnatDebug?: boolean;
        produceIr?: boolean;
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
    executeParameters: {
        args: string;
        stdin: string;
    };
    filters: Record<string, boolean>;
    tools: ActiveTools[];
    libraries: CompileChildLibraries[];
};

export type CompilationRequest = {
    source: string;
    compiler: string;
    options: CompilationRequestOptions;
    lang: string | null;
    files: FiledataPair[];
    bypassCache?: boolean;
};

export type LangInfo = {
    compiler: string;
    options: string;
    execArgs: string;
    execStdin: string;
};

export type CompileChildLibraries = {
    id: string;
    version: string;
};
