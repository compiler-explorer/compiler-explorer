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
import {CompilationQueue} from '../compilation-queue.js';
import {FormattingService} from '../formatting-service.js';
import {AssemblyDocumentationController} from '../handlers/api/assembly-documentation-controller.js';
import {FormattingController} from '../handlers/api/formatting-controller.js';
import {HealthcheckController} from '../handlers/api/healthcheck-controller.js';
import {NoScriptController} from '../handlers/api/noscript-controller.js';
import {SiteTemplateController} from '../handlers/api/site-template-controller.js';
import {SourceController} from '../handlers/api/source-controller.js';
import {CompileHandler} from '../handlers/compile.js';
import {sources} from '../sources/index.js';

export interface ApiControllers {
    siteTemplateController: SiteTemplateController;
    sourceController: SourceController;
    assemblyDocumentationController: AssemblyDocumentationController;
    formattingController: FormattingController;
    noScriptController: NoScriptController;
    healthcheckController: HealthcheckController;
}

/**
 * Initialize all API controllers used by the application
 * @param compileHandler - The compile handler instance
 * @param formattingService - The formatting service instance
 * @param compilationQueue - The compilation queue instance
 * @param healthCheckFilePath - Optional path to health check file
 * @param isExecutionWorker - Whether the server is running as an execution worker
 * @param isCompilationWorker - Whether the server is running as a compilation worker
 * @param formDataHandler - Handler for form data
 * @returns Object containing all initialized controllers
 */
export function setupControllersAndHandlers(
    compileHandler: CompileHandler,
    formattingService: FormattingService,
    compilationQueue: CompilationQueue,
    healthCheckFilePath: string | null,
    isExecutionWorker: boolean,
    isCompilationWorker: boolean,
    formDataHandler: express.Handler,
): ApiControllers {
    // Initialize API controllers
    const siteTemplateController = new SiteTemplateController();
    const sourceController = new SourceController(sources);
    const assemblyDocumentationController = new AssemblyDocumentationController();
    const formattingController = new FormattingController(formattingService);
    const noScriptController = new NoScriptController(compileHandler, formDataHandler);

    // Initialize healthcheck controller (handled separately in web server setup)
    const healthcheckController = new HealthcheckController(
        compilationQueue,
        healthCheckFilePath,
        compileHandler,
        isExecutionWorker,
    );

    return {
        siteTemplateController,
        sourceController,
        assemblyDocumentationController,
        formattingController,
        noScriptController,
        healthcheckController,
    };
}
