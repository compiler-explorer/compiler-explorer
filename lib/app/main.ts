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

import child_process from 'node:child_process';
import fs from 'node:fs/promises';
import path from 'node:path';

import _ from 'underscore';

import type {Router} from 'express';
import {AppArguments} from '../app.interfaces.js';
import {unwrap} from '../assert.js';
import * as aws from '../aws.js';
import {CompilationEnvironment} from '../compilation-env.js';
import {CompilationQueue} from '../compilation-queue.js';
import {CompilerFinder} from '../compiler-finder.js';
import {startWineInit} from '../exec.js';
import {RemoteExecutionQuery} from '../execution/execution-query.js';
import {initHostSpecialties} from '../execution/execution-triple.js';
import {startExecutionWorkerThread} from '../execution/sqs-execution-queue.js';
import {FormattingService} from '../formatting-service.js';
import {AssemblyDocumentationController} from '../handlers/api/assembly-documentation-controller.js';
import {FormattingController} from '../handlers/api/formatting-controller.js';
import {HealthcheckController} from '../handlers/api/healthcheck-controller.js';
import {NoScriptController} from '../handlers/api/noscript-controller.js';
import {SiteTemplateController} from '../handlers/api/site-template-controller.js';
import {SourceController} from '../handlers/api/source-controller.js';
import {CompileHandler} from '../handlers/compile.js';
import {createFormDataHandler} from '../handlers/middleware.js';
import {NoScriptHandler} from '../handlers/noscript.js';
import {RouteAPI} from '../handlers/route-api.js';
import {logger} from '../logger.js';
import {setupMetricsServer} from '../metrics-server.js';
import {ClientOptionsHandler} from '../options-handler.js';
import {PropertyGetter} from '../properties.interfaces.js';
import {CompilerProps} from '../properties.js';
import {SetupSentry} from '../sentry.js';
import {sources} from '../sources/index.js';
import {loadSponsorsFromString} from '../sponsors.js';
import {getStorageTypeByKey} from '../storage/index.js';
import type {AppConfiguration} from './config.interfaces.js';
import {ApplicationOptions, ApplicationResult} from './main.interfaces.js';
import {setupWebServer, startListening} from './server.js';

/**
 * Set up temporary directory, especially for WSL environments
 */
export function setupTempDir(tmpDir: string | undefined, isWsl: boolean): void {
    // If a tempDir is supplied, use it
    if (tmpDir) {
        if (isWsl) {
            process.env.TEMP = tmpDir; // for Windows
        } else {
            process.env.TMP = tmpDir; // for Linux
        }
    }
    // If running under WSL without explicit tmpDir, try to use Windows %TEMP%
    else if (isWsl) {
        try {
            const windowsTemp = child_process.execSync('cmd.exe /c echo %TEMP%').toString().replaceAll('\\', '/');
            const driveLetter = windowsTemp.substring(0, 1).toLowerCase();
            const directoryPath = windowsTemp.substring(2).trim();
            process.env.TEMP = path.join('/mnt', driveLetter, directoryPath);
        } catch (e) {
            logger.warn('Unable to invoke cmd.exe to get windows %TEMP% path.');
        }
    }
    logger.info(`Using temporary dir: ${process.env.TEMP || process.env.TMP}`);
}

/**
 * Initialize the Compiler Explorer application
 */
export async function initialiseApplication(options: ApplicationOptions): Promise<ApplicationResult> {
    const {appArgs, config, distPath, awsProps} = options;
    const {ceProps, compilerProps, languages, storageSolution} = config;

    setupTempDir(appArgs.tmpDir, appArgs.isWsl);

    await aws.initConfig(awsProps);

    SetupSentry(aws.getConfig('sentryDsn'), ceProps, appArgs.releaseBuildNumber, appArgs.gitReleaseName, appArgs);

    // Start Wine initialization (for Windows compilers)
    startWineInit();

    RemoteExecutionQuery.initRemoteExecutionArchs(ceProps, appArgs.env);

    const {compilationQueue, compilationEnvironment, compileHandler, formattingService} =
        await initializeCompilationEnvironment(appArgs, compilerProps, ceProps, awsProps);

    const clientOptionsHandler = new ClientOptionsHandler(sources, compilerProps, appArgs);
    const storageType = getStorageTypeByKey(storageSolution);
    const storageHandler = new storageType(config.httpRoot, compilerProps, awsProps);

    const compilerFinder = new CompilerFinder(compileHandler, compilerProps, appArgs, clientOptionsHandler);

    // Get execution configuration
    const isExecutionWorker = ceProps<boolean>('execqueue.is_worker', false);
    const healthCheckFilePath = ceProps('healthCheckFilePath', null) as string | null;

    const formDataHandler = createFormDataHandler();

    const controllers = await setupControllersAndHandlers(
        compileHandler,
        formattingService,
        compilationQueue,
        healthCheckFilePath,
        isExecutionWorker,
        formDataHandler,
    );

    logVersionInfo(appArgs);

    const initialCompilers = await discoverCompilers(appArgs, compilerFinder, isExecutionWorker);

    const {webServer, router, renderConfig, renderGoldenLayout} = await setupWebServerWithOptions(
        appArgs,
        config,
        distPath,
        ceProps,
        clientOptionsHandler,
        storageSolution,
        storageHandler,
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

async function initializeCompilationEnvironment(
    appArgs: AppArguments,
    compilerProps: CompilerProps,
    ceProps: PropertyGetter,
    awsProps: PropertyGetter,
) {
    const formattingService = new FormattingService();
    await formattingService.initialize(ceProps);

    const compilationQueue = CompilationQueue.fromProps(compilerProps.ceProps);
    const compilationEnvironment = new CompilationEnvironment(
        compilerProps,
        awsProps,
        compilationQueue,
        formattingService,
        appArgs.doCache,
    );

    const compileHandler = new CompileHandler(compilationEnvironment, awsProps);
    compilationEnvironment.setCompilerFinder(compileHandler.findCompiler.bind(compileHandler));

    return {
        compilationQueue,
        compilationEnvironment,
        compileHandler,
        formattingService,
    };
}

async function setupControllersAndHandlers(
    compileHandler: CompileHandler,
    formattingService: FormattingService,
    compilationQueue: CompilationQueue,
    healthCheckFilePath: string | null,
    isExecutionWorker: boolean,
    formDataHandler: any,
) {
    // Initialize API controllers
    const siteTemplateController = new SiteTemplateController();
    const sourceController = new SourceController(sources);
    const assemblyDocumentationController = new AssemblyDocumentationController();
    const formattingController = new FormattingController(formattingService);
    const noScriptController = new NoScriptController(compileHandler, formDataHandler);

    // Initialize healthcheck controller (handled separately in web server setup)
    new HealthcheckController(compilationQueue, healthCheckFilePath, compileHandler, isExecutionWorker);

    return {
        siteTemplateController,
        sourceController,
        assemblyDocumentationController,
        formattingController,
        noScriptController,
    };
}

function logVersionInfo(appArgs: AppArguments) {
    logger.info('=======================================');
    if (appArgs.gitReleaseName) logger.info(`  git release ${appArgs.gitReleaseName}`);
    if (appArgs.releaseBuildNumber) logger.info(`  release build ${appArgs.releaseBuildNumber}`);
}

async function discoverCompilers(appArgs: AppArguments, compilerFinder: CompilerFinder, isExecutionWorker: boolean) {
    let compilers;
    if (appArgs.prediscovered) {
        compilers = await loadPrediscoveredCompilers(appArgs.prediscovered, compilerFinder);
    } else {
        const result = await findAndValidateCompilers(appArgs, compilerFinder, isExecutionWorker);
        compilers = result.compilers;
    }

    if (appArgs.discoveryOnly) {
        await handleDiscoveryOnlyMode(appArgs.discoveryOnly, compilers, compilerFinder);
    }

    return compilers;
}

async function loadPrediscoveredCompilers(filename: string, compilerFinder: CompilerFinder) {
    const prediscoveredCompilersJson = await fs.readFile(filename, 'utf8');
    const initialCompilers = JSON.parse(prediscoveredCompilersJson);
    const prediscResult = await compilerFinder.loadPrediscovered(initialCompilers);
    if (prediscResult.length === 0) {
        throw new Error('Unexpected failure, no compilers found!');
    }
    return initialCompilers;
}

async function findAndValidateCompilers(
    appArgs: AppArguments,
    compilerFinder: CompilerFinder,
    isExecutionWorker: boolean,
) {
    const initialFindResults = await compilerFinder.find();
    const initialCompilers = initialFindResults.compilers;
    if (!isExecutionWorker && initialCompilers.length === 0) {
        throw new Error('Unexpected failure, no compilers found!');
    }
    if (appArgs.ensureNoCompilerClash) {
        logger.warn('Ensuring no compiler ids clash');
        if (initialFindResults.foundClash) {
            // If we are forced to have no clashes, throw an error with some explanation
            throw new Error('Clashing compilers in the current environment found!');
        }
        logger.info('No clashing ids found, continuing normally...');
    }
    return initialFindResults;
}

async function handleDiscoveryOnlyMode(savePath: string, initialCompilers: any[], compilerFinder: CompilerFinder) {
    for (const compiler of initialCompilers) {
        if (compiler.buildenvsetup && compiler.buildenvsetup.id === '') delete compiler.buildenvsetup;

        if (compiler.externalparser && compiler.externalparser.id === '') delete compiler.externalparser;

        const compilerInstance = compilerFinder.compileHandler.findCompiler(compiler.lang, compiler.id);
        if (compilerInstance) {
            compiler.cachedPossibleArguments = compilerInstance.possibleArguments.possibleArguments;
        }
    }
    await fs.writeFile(savePath, JSON.stringify(initialCompilers));
    logger.info(`Discovered compilers saved to ${savePath}`);
    process.exit(0);
}

async function setupWebServerWithOptions(
    appArgs: AppArguments,
    config: AppConfiguration,
    distPath: string,
    ceProps: PropertyGetter,
    clientOptionsHandler: any,
    storageSolution: string,
    storageHandler: any,
) {
    const serverOptions = {
        staticPath: appArgs.staticPath || path.join(distPath, 'static'),
        staticMaxAgeSecs: config.staticMaxAgeSecs,
        staticUrl: config.staticUrl,
        staticRoot: config.staticRoot,
        httpRoot: config.httpRoot,
        sentrySlowRequestMs: ceProps('sentrySlowRequestMs', 0),
        distPath: distPath,
        extraBodyClass: config.extraBodyClass,
        maxUploadSize: config.maxUploadSize,
    };

    const serverDependencies = {
        ceProps: ceProps,
        sponsorConfig: loadSponsorsFromString(
            await fs.readFile(path.join(appArgs.rootDir, 'config', 'sponsors.yaml'), 'utf8'),
        ),
        clientOptionsHandler: clientOptionsHandler,
        storageSolution: storageSolution,
    };

    // Initialize web server
    return await setupWebServer(appArgs, serverOptions, serverDependencies);
}

function setupRoutesAndApi(
    router: Router,
    controllers: any,
    clientOptionsHandler: any,
    renderConfig: any,
    renderGoldenLayout: any,
    storageHandler: any,
    appArgs: AppArguments,
    compileHandler: any,
    compilationEnvironment: any,
    ceProps: any,
) {
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

async function setupCompilerChangeHandling(
    initialCompilers: any,
    clientOptionsHandler: any,
    routeApi: RouteAPI,
    languages: any,
    ceProps: PropertyGetter,
    compilerFinder: CompilerFinder,
    appArgs: AppArguments,
) {
    let prevCompilers = '';
    async function onCompilerChange(compilers) {
        const compilersAsJson = JSON.stringify(compilers);
        if (prevCompilers === compilersAsJson) {
            return;
        }
        logger.info(`Compiler scan count: ${compilers.length}`);
        logger.debug('Compilers:', compilers);
        prevCompilers = compilersAsJson;
        await clientOptionsHandler.setCompilers(compilers);
        const apiHandler = unwrap(routeApi.apiHandler);
        apiHandler.setCompilers(compilers);
        apiHandler.setLanguages(languages);
        apiHandler.setOptions(clientOptionsHandler);
    }

    // Set initial compilers
    await onCompilerChange(initialCompilers);

    // Set up compiler rescanning if configured
    const rescanCompilerSecs = ceProps('rescanCompilerSecs', 0);
    if (rescanCompilerSecs && !appArgs.prediscovered) {
        logger.info(`Rescanning compilers every ${rescanCompilerSecs} secs`);
        setInterval(
            () => compilerFinder.find().then(result => onCompilerChange(result.compilers)),
            rescanCompilerSecs * 1000,
        );
    }
}
