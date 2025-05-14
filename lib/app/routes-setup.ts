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

import type {Router} from 'express';
import type {AppArguments} from '../app.interfaces.js';
import {NoScriptHandler} from '../handlers/noscript.js';
import {RouteAPI} from '../handlers/route-api.js';
import {logger} from '../logger.js';
import type {ApiControllers} from './controllers.js';

/**
 * Set up routes and API endpoints for the application
 * @param router - Express router
 * @param controllers - Controller instances
 * @param clientOptionsHandler - Client options handler
 * @param renderConfig - Config rendering function
 * @param renderGoldenLayout - Golden layout rendering function
 * @param storageHandler - Storage handler
 * @param appArgs - Application arguments
 * @param compileHandler - Compile handler
 * @param compilationEnvironment - Compilation environment
 * @param ceProps - Compiler Explorer properties
 * @returns RouteAPI instance
 */
export function setupRoutesAndApi(
    router: Router,
    controllers: ApiControllers,
    clientOptionsHandler: any,
    renderConfig: any,
    renderGoldenLayout: any,
    storageHandler: any,
    appArgs: AppArguments,
    compileHandler: any,
    compilationEnvironment: any,
    ceProps: any,
): RouteAPI {
    const {
        siteTemplateController,
        sourceController,
        assemblyDocumentationController,
        formattingController,
        noScriptController,
    } = controllers;

    // Set up NoScript handler and RouteAPI
    const noscriptHandler = new NoScriptHandler(
        router,
        clientOptionsHandler,
        renderConfig,
        storageHandler,
        appArgs.wantedLanguages?.[0],
    );

    const routeApi = new RouteAPI(router, {
        compileHandler,
        clientOptionsHandler,
        storageHandler,
        compilationEnvironment,
        ceProps,
        defArgs: appArgs,
        renderConfig,
        renderGoldenLayout,
    });

    // Set up controllers
    try {
        router.use(siteTemplateController.createRouter());
        router.use(sourceController.createRouter());
        router.use(assemblyDocumentationController.createRouter());
        router.use(formattingController.createRouter());
        router.use(noScriptController.createRouter());
    } catch (err: unknown) {
        // In case of errors (e.g. during testing), log but continue
        logger.debug('Error setting up controllers, possibly in test environment:', err);
    }

    noscriptHandler.initializeRoutes();
    routeApi.initializeRoutes();

    return routeApi;
}
