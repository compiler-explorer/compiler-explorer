// Copyright (c) 2024, Compiler Explorer Authors
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

import {v4 as uuidv4} from 'uuid';

import {CompilationResult} from '../../types/compilation/compilation.interfaces.js';
import {EventsWsWaiter} from '../execution/events-websocket.js';
import {ParsedRequest} from '../handlers/compile.js';
import {logger} from '../logger.js';
import {PropertyGetter} from '../properties.interfaces.js';

import {RemoteCompilationRequest, SqsCompileRequester} from './sqs-compilation-queue.js';

export class RemoteCompilationEnvironment {
    private guid: string;
    private requester: SqsCompileRequester;
    private waiter: EventsWsWaiter;

    constructor(ceProps: PropertyGetter, awsProps: PropertyGetter) {
        this.guid = uuidv4();
        this.requester = new SqsCompileRequester(ceProps, awsProps);
        this.waiter = new EventsWsWaiter(ceProps);
    }

    async compile(
        compilerId: string,
        request: ParsedRequest,
        languageId: string,
        files: any[] = [],
        isCMake = false,
    ): Promise<CompilationResult> {
        const startTime = Date.now();

        try {
            await this.waiter.subscribe(this.guid);

            const remoteRequest: RemoteCompilationRequest = {
                guid: this.guid,
                compilerId: compilerId,
                source: request.source,
                options: request.options || [],
                backendOptions: request.backendOptions || {},
                filters: request.filters || {},
                bypassCache: request.bypassCache,
                tools: request.tools || [],
                executeParameters: request.executeParameters || {},
                libraries: request.libraries || [],
                lang: languageId,
                files: files,
                isCMake: isCMake,
            };

            await this.requester.push(remoteRequest);
            logger.debug(`Sent remote compilation request ${this.guid} to queue`);

            const result = await this.waiter.data();

            const endTime = Date.now();
            const compilationTime = endTime - startTime;

            logger.debug(`Remote compilation ${this.guid} completed in ${compilationTime}ms`);

            return result as CompilationResult;
        } catch (error) {
            logger.error(`Remote compilation ${this.guid} failed:`, error);
            return this.createErrorResult(`Remote compilation failed: ${error}`);
        } finally {
            try {
                await this.waiter.close();
            } catch (e) {
                logger.warn(`Failed to close WebSocket for compilation ${this.guid}:`, e);
            }
        }
    }

    private createErrorResult(message: string): CompilationResult {
        return {
            code: -1,
            stderr: [{text: message}],
            stdout: [],
            okToCache: false,
            timedOut: false,
            inputFilename: '',
            asm: [],
            tools: [],
        };
    }

    getGuid(): string {
        return this.guid;
    }
}

export interface IRemoteCompilationEnvironment {
    compile(
        compilerId: string,
        request: ParsedRequest,
        languageId: string,
        files?: any[],
        isCMake?: boolean,
    ): Promise<CompilationResult>;
}
