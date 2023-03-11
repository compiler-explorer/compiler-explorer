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

import {LanguageKey} from './languages.interfaces.js';
import {ResultLine} from './resultline/resultline.interfaces.js';

export type ToolTypeKey = 'independent' | 'postcompilation';

export type ToolInfo = {
    id: string;
    name?: string;
    type?: ToolTypeKey;
    exe: string;
    exclude: string[];
    includeKey?: string;
    options: string[];
    args?: string;
    languageId?: LanguageKey;
    stdinHint?: string;
    monacoStdin?: string;
    icon?: string;
    darkIcon?: string;
    compilerLanguage: LanguageKey;
};

export type Tool = {
    readonly tool: ToolInfo;
    readonly id: string;
    readonly type: string;
};

export enum ArtifactType {
    download = 'application/octet-stream',
    nesrom = 'nesrom',
    bbcdiskimage = 'bbcdiskimage',
    zxtape = 'zxtape',
    smsrom = 'smsrom',
}

export type Artifact = {
    content: string;
    type: string;
    name: string;
    title: string;
};

export type ToolResult = {
    id: string;
    name?: string;
    code: number;
    languageId?: LanguageKey | 'stderr';
    stderr: ResultLine[];
    stdout: ResultLine[];
    artifact?: Artifact;
    sourcechanged?: boolean;
    newsource?: string;
};
