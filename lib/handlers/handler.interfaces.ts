// Copyright (c) 2025, Compiler Explorer Authors
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

import express from 'express';

import {AppArguments} from '../app.interfaces.js';
import {GoldenLayoutRootStruct} from '../clientstate-normalizer.js';
import {CompilationEnvironment} from '../compilation-env.js';
import {ClientOptionsHandler} from '../options-handler.js';
import {PropertyGetter} from '../properties.interfaces.js';
import {StorageBase} from '../storage/index.js';
import {CompileHandler} from './compile.js';

export type RenderConfig = (extra: Record<string, any>, urlOptions?: Record<string, any>) => Record<string, any>;

export type RenderGoldenLayout = (
    config: GoldenLayoutRootStruct,
    metadata: ShortLinkMetaData,
    req: express.Request,
    res: express.Response,
) => void;

export type HandlerConfig = {
    compileHandler: CompileHandler;
    clientOptionsHandler: ClientOptionsHandler;
    storageHandler: StorageBase;
    ceProps: PropertyGetter;
    defArgs: AppArguments;
    renderConfig: RenderConfig;
    renderGoldenLayout: RenderGoldenLayout;
    compilationEnvironment: CompilationEnvironment;
};

export type ShortLinkMetaData = {
    ogDescription?: string;
    ogAuthor?: string;
    ogTitle?: string;
    ogCreated?: Date;
};
