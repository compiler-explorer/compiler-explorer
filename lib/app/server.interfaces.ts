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

import type {Express, Request, Response, Router} from 'express';
import type {GoldenLayoutRootStruct} from '../clientstate-normalizer.js';
import type {HttpController} from '../handlers/api/controller.interfaces.js';
import type {ShortLinkMetaData} from '../handlers/handler.interfaces.js';
import type {ClientOptionsSource} from '../options-handler.interfaces.js';
import type {PropertyGetter} from '../properties.interfaces.js';
import type {Sponsors} from '../sponsors.interfaces.js';

export interface ServerOptions {
    staticPath: string;
    staticMaxAgeSecs: number;
    staticUrl?: string;
    staticRoot: string;
    httpRoot: string;
    sentrySlowRequestMs: number;
    distPath: string;
    extraBodyClass: string;
    maxUploadSize: string;
}

export interface PugOptions {
    extraBodyClass: string;
    httpRoot: string;
    staticRoot: string;
    storageSolution: string;
    optionsHash: string;
    compilerExplorerOptions: string;
}

export interface RenderConfig extends PugOptions {
    embedded: boolean;
    mobileViewer: boolean;
    readOnly?: boolean;
    config?: GoldenLayoutRootStruct;
    metadata?: ShortLinkMetaData;
    storedStateId?: string | false;
    require?: PugRequireHandler;
    sponsors?: Sponsors;
    slides?: any[];
}

export type RenderConfigFunction = (extra: Record<string, any>, urlOptions?: Record<string, any>) => RenderConfig;

export type RenderGoldenLayoutHandler = (
    config: GoldenLayoutRootStruct,
    metadata: ShortLinkMetaData,
    req: Request,
    res: Response,
) => void;

export type PugRequireHandler = (path: string) => string;

export interface WebServerResult {
    webServer: Express;
    router: Router;
    pugRequireHandler: PugRequireHandler;
    renderConfig: RenderConfigFunction;
    renderGoldenLayout: RenderGoldenLayoutHandler;
}

export interface ServerDependencies {
    ceProps: PropertyGetter;
    sponsorConfig: Sponsors;
    clientOptionsHandler: ClientOptionsSource;
    storageSolution: string;
    healthcheckController: HttpController;
}
