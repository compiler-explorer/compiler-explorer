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

import type {AppArguments} from '../app.interfaces.js';
import {logger} from '../logger.js';
import {createRenderHandlers} from './rendering.js';
import {ServerDependencies, ServerOptions, WebServerResult} from './server.interfaces.js';
import {setupBaseServerConfig, setupBasicRoutes, setupLoggingMiddleware} from './server-config.js';
import {setupStaticMiddleware, setupWebPackDevMiddleware} from './static-assets.js';

export {startListening} from './server-listening.js';
export {isMobileViewer} from './url-handlers.js';

/**
 * Configure a web server and its routes
 * @param appArgs - Application arguments
 * @param options - Server options
 * @param dependencies - Server dependencies
 * @returns Web server configuration
 */
export async function setupWebServer(
    appArgs: AppArguments,
    options: ServerOptions,
    dependencies: ServerDependencies,
): Promise<WebServerResult> {
    const webServer = express();
    const router = express.Router();

    let pugRequireHandler;

    try {
        pugRequireHandler = await (appArgs.devMode
            ? setupWebPackDevMiddleware(options, router)
            : setupStaticMiddleware(options, router));
    } catch (err: unknown) {
        const error = err as Error;
        logger.warn(`Error setting up static middleware: ${error.message}`);
        pugRequireHandler = path => `${options.staticRoot}/${path}`;
    }

    const {renderConfig, renderGoldenLayout, embeddedHandler} = createRenderHandlers(
        pugRequireHandler,
        options,
        dependencies,
        appArgs,
    );

    // Add healthcheck before logging middleware to prevent excessive log entries
    webServer.use(dependencies.healthcheckController.createRouter());

    setupBaseServerConfig(options, renderConfig, webServer, router);
    setupLoggingMiddleware(appArgs.devMode, router);

    setupBasicRoutes(
        router,
        renderConfig,
        embeddedHandler,
        dependencies.ceProps,
        dependencies.awsProps,
        options,
        dependencies.clientOptionsHandler,
    );

    return {
        webServer,
        router,
        pugRequireHandler,
        renderConfig,
        renderGoldenLayout,
    };
}
