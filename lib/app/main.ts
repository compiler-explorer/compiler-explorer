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

import fs from 'node:fs/promises';
import path from 'node:path';

import type {AppArguments} from '../app.interfaces.js';
import * as aws from '../aws.js';
import {CompilerFinder} from '../compiler-finder.js';
import {startWineInit} from '../exec.js';
import {RemoteExecutionQuery} from '../execution/execution-query.js';
import {initHostSpecialties} from '../execution/execution-triple.js';
import {startExecutionWorkerThread} from '../execution/sqs-execution-queue.js';
import {createFormDataHandler} from '../handlers/middleware.js';
import {logger} from '../logger.js';
import {setupMetricsServer} from '../metrics-server.js';
import {ClientOptionsHandler} from '../options-handler.js';
import {SetupSentry} from '../sentry.js';
import {sources} from '../sources/index.js';
import {loadSponsorsFromString} from '../sponsors.js';
import {getStorageTypeByKey} from '../storage/index.js';
import {initializeCompilationEnvironment} from './compilation-env.js';
import {setupCompilerChangeHandling} from './compiler-changes.js';
import {discoverCompilers} from './compiler-discovery.js';
import {setupControllersAndHandlers} from './controllers.js';
import {ApplicationOptions, ApplicationResult} from './main.interfaces.js';
import {setupRoutesAndApi} from './routes-setup.js';
import {setupWebServer, startListening} from './server.js';
import {setupTempDir} from './temp-dir.js';

/**
 * Initialize the Compiler Explorer application
 */
export async function initialiseApplication(options: ApplicationOptions): Promise<ApplicationResult> {
    const {appArgs, config, distPath, awsProps} = options;
    const {ceProps, compilerProps, languages, storageSolution} = config;

    setupTempDir(appArgs.tmpDir, appArgs.isWsl);

    await aws.initConfig(awsProps);
    SetupSentry(aws.getConfig('sentryDsn'), ceProps, appArgs.releaseBuildNumber, appArgs.gitReleaseName, appArgs);

    startWineInit();

    RemoteExecutionQuery.initRemoteExecutionArchs(ceProps, appArgs.env);

    const {compilationEnvironment, compileHandler} = await initializeCompilationEnvironment(
        appArgs,
        compilerProps,
        ceProps,
        awsProps,
    );

    const clientOptionsHandler = new ClientOptionsHandler(sources, compilerProps, appArgs);
    const storageType = getStorageTypeByKey(storageSolution);
    const storageHandler = new storageType(config.httpRoot, compilerProps, awsProps);

    const compilerFinder = new CompilerFinder(compileHandler, compilerProps, appArgs, clientOptionsHandler);

    const isExecutionWorker = ceProps<boolean>('execqueue.is_worker', false);
    const healthCheckFilePath = ceProps('healthCheckFilePath', null) as string | null;

    const formDataHandler = createFormDataHandler();

    const controllers = setupControllersAndHandlers(
        compileHandler,
        compilationEnvironment.formattingService,
        compilationEnvironment.compilationQueue,
        healthCheckFilePath,
        isExecutionWorker,
        formDataHandler,
    );

    logVersionInfo(appArgs);

    const initialCompilers = await discoverCompilers(appArgs, compilerFinder, isExecutionWorker);

    const serverOptions = {
        staticPath: appArgs.staticPath || path.join(distPath, 'static'),
        staticMaxAgeSecs: config.staticMaxAgeSecs,
        staticUrl: config.staticUrl,
        staticRoot: config.staticRoot,
        httpRoot: config.httpRoot,
        sentrySlowRequestMs: ceProps('sentrySlowRequestMs', 0),
        manifestPath: distPath,
        extraBodyClass: config.extraBodyClass,
        maxUploadSize: config.maxUploadSize,
    };

    const serverDependencies = {
        ceProps: ceProps,
        awsProps: awsProps,
        sponsorConfig: loadSponsorsFromString(
            await fs.readFile(path.join(appArgs.rootDir, 'config', 'sponsors.yaml'), 'utf8'),
        ),
        clientOptionsHandler: clientOptionsHandler,
        storageSolution: storageSolution,
        healthcheckController: controllers.healthcheckController,
    };

    const {webServer, router, renderConfig, renderGoldenLayout} = await setupWebServer(
        appArgs,
        serverOptions,
        serverDependencies,
    );

    const routeApi = setupRoutesAndApi(
        router,
        controllers,
        clientOptionsHandler,
        renderConfig,
        renderGoldenLayout,
        storageHandler,
        appArgs,
        compileHandler,
        compilationEnvironment,
        ceProps,
    );

    await setupCompilerChangeHandling(
        initialCompilers,
        clientOptionsHandler,
        routeApi,
        languages,
        ceProps,
        compilerFinder,
        appArgs,
    );

    if (appArgs.metricsPort) {
        logger.info(`Running metrics server on port ${appArgs.metricsPort}`);
        setupMetricsServer(appArgs.metricsPort, appArgs.hostname);
    }

    if (!appArgs.doCache) {
        logger.info('  with disabled caching');
    }

    if (isExecutionWorker) {
        await initHostSpecialties();
        startExecutionWorkerThread(ceProps, awsProps, compilationEnvironment);
    }

    startListening(webServer, appArgs);

    return {webServer};
}

/**
 * Log version information
 */
function logVersionInfo(appArgs: AppArguments) {
    logger.info('=======================================');
    if (appArgs.gitReleaseName) logger.info(`  git release ${appArgs.gitReleaseName}`);
    if (appArgs.releaseBuildNumber) logger.info(`  release build ${appArgs.releaseBuildNumber}`);
}
