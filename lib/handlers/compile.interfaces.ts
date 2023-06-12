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

import {BypassCache} from '../../types/compilation/compilation.interfaces.js';

// IF YOU MODIFY ANYTHING HERE PLEASE UPDATE THE DOCUMENTATION!

// This type models a request so all fields must be optional strings.
export type CompileRequestQueryArgs = {
    options?: string;
    filters?: string;
    addFilters?: string;
    removeFilters?: string;
    skipAsm?: string;
    skipPopArgs?: string;
};

export type ExecutionRequestParams = {
    args?: string | string[];
    stdin?: string;
};

// TODO find more types for these.
export type CompilationRequestArgs = {
    userArguments: string;
    compilerOptions: Record<string, any>;
    executeParameters: ExecutionRequestParams;
    filters: Record<string, boolean>;
    tools: any;
    libraries: any[];
};

export type CompileRequestJsonBody = {
    options: CompilationRequestArgs;
    source: string;
    bypassCache: BypassCache;
};

export type CompileRequestTextBody = {
    source: string;
    bypassCache: BypassCache;
    options: any;
    userArguments: string;
    executeParametersArgs: any;
    executeParametersStdin: any;
    skipAsm: string;
};
