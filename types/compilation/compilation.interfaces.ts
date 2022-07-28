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
import {ResultLine} from '../resultline/resultline.interfaces';

export type CompilationResult = {
    code: number;
    buildResult?: unknown;
    inputFilename?: string;
    asm?: ResultLine[];
    stdout: ResultLine[];
    stderr: ResultLine[];
    didExecute?: boolean;
    execResult?: {
        stdout?: ResultLine[];
        stderr?: ResultLine[];
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

    hasPpOutput?: boolean;
    ppOutput?: any;

    hasOptOutput?: boolean;
    optPath?: string;

    hasAstOutput?: boolean;
    astOutput?: any;

    hasIrOutput?: boolean;
    irOutput?: any;

    hasLLVMOptPipelineOutput?: boolean;
    llvmOptPipelineOutput?: any;

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
    input?: any;
};

export type BuildResult = {
    downloads: BuildEnvDownloadInfo[];
    executableFilename: string;
    compilationOptions: any[];
};

export type Artifact = {
    content: string;
    type: string;
    name: string;
    title: string;
};

export type ToolResult = {
    id: string;
    name: string;
    code: number;
    languageId: string;
    stderr: ResultLine[];
    stdout: ResultLine[];
    artifact?: Artifact;
};
