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

import _ from 'underscore';

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
import {SetupSentry} from '../sentry.js';
import {sources} from '../sources/index.js';
import {loadSponsorsFromString} from '../sponsors.js';
import {getStorageTypeByKey} from '../storage/index.js';
import {ApplicationOptions, ApplicationResult} from './main.interfaces.js';
import {setupWebServer, startListening} from './server.js';

/**
 * Initialize the Compiler Explorer application
 */
export async function initializeApplication(options: ApplicationOptions): Promise<ApplicationResult> {
    const {appArgs, config, distPath, awsProps} = options;
    const {ceProps, compilerProps, languages, storageSolution} = config;

    // Initialize AWS configuration
    await aws.initConfig(awsProps);

    // Setup Sentry for error tracking
    SetupSentry(aws.getConfig('sentryDsn'), ceProps, appArgs.releaseBuildNumber, appArgs.gitReleaseName, appArgs);

    // Start Wine initialization (for Windows compilers)
    startWineInit();

    // Initialize remote execution architectures
    RemoteExecutionQuery.initRemoteExecutionArchs(ceProps, appArgs.env);

    // Initialize the formatting service
    const formattingService = new FormattingService();
    await formattingService.initialize(ceProps);

    // Initialize client options handler
    const clientOptionsHandler = new ClientOptionsHandler(sources, compilerProps, appArgs);

    // Set up compilation queue and environment
    const compilationQueue = CompilationQueue.fromProps(compilerProps.ceProps);
    const compilationEnvironment = new CompilationEnvironment(
        compilerProps,
        awsProps,
        compilationQueue,
        formattingService,
        appArgs.doCache,
    );

    // Set up compilation handler
    const compileHandler = new CompileHandler(compilationEnvironment, awsProps);
    compilationEnvironment.setCompilerFinder(compileHandler.findCompiler.bind(compileHandler));

    // Initialize storage handler
    const storageType = getStorageTypeByKey(storageSolution);
    const storageHandler = new storageType(config.httpRoot, compilerProps, awsProps);

    // Set up compiler finder
    const compilerFinder = new CompilerFinder(compileHandler, compilerProps, appArgs, clientOptionsHandler);

    // Get execution configuration
    const isExecutionWorker = ceProps<boolean>('execqueue.is_worker', false);
    const healthCheckFilePath = ceProps('healthCheckFilePath', null) as string | null;

    // Create form data handler
    const formDataHandler = createFormDataHandler();

    // Initialize API controllers
    const siteTemplateController = new SiteTemplateController();
    const sourceController = new SourceController(sources);
    const assemblyDocumentationController = new AssemblyDocumentationController();
    const formattingController = new FormattingController(formattingService);
    const noScriptController = new NoScriptController(compileHandler, formDataHandler);

    // Initialize healthcheck controller (handled separately in web server setup)
    new HealthcheckController(compilationQueue, healthCheckFilePath, compileHandler, isExecutionWorker);

    // Log version information
    logger.info('=======================================');
    if (appArgs.gitReleaseName) logger.info(`  git release ${appArgs.gitReleaseName}`);
    if (appArgs.releaseBuildNumber) logger.info(`  release build ${appArgs.releaseBuildNumber}`);

    // Discover compilers
    let initialCompilers;
    let prevCompilers;

    if (options.options.prediscovered) {
        const prediscoveredCompilersJson = await fs.readFile(options.options.prediscovered, 'utf8');
        initialCompilers = JSON.parse(prediscoveredCompilersJson);
        const prediscResult = await compilerFinder.loadPrediscovered(initialCompilers);
        if (prediscResult.length === 0) {
            throw new Error('Unexpected failure, no compilers found!');
        }
    } else {
        const initialFindResults = await compilerFinder.find();
        initialCompilers = initialFindResults.compilers;
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
    }

    // Handle discovery-only mode if requested
    if (options.options.discoveryOnly) {
        for (const compiler of initialCompilers) {
            if (compiler.buildenvsetup && compiler.buildenvsetup.id === '') delete compiler.buildenvsetup;

            if (compiler.externalparser && compiler.externalparser.id === '') delete compiler.externalparser;

            const compilerInstance = compilerFinder.compileHandler.findCompiler(compiler.lang, compiler.id);
            if (compilerInstance) {
                compiler.cachedPossibleArguments = compilerInstance.possibleArguments.possibleArguments;
            }
        }
        await fs.writeFile(options.options.discoveryOnly, JSON.stringify(initialCompilers));
        logger.info(`Discovered compilers saved to ${options.options.discoveryOnly}`);
        process.exit(0);
    }

    // Create web server with needed configurations
    const serverOptions = {
        staticPath: options.options.static || path.join(distPath, 'static'),
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
        sponsorConfig: await loadSponsorsFromString(
            await fs.readFile(path.join(appArgs.rootDir, 'config', 'sponsors.yaml'), 'utf8'),
        ),
        clientOptionsHandler: clientOptionsHandler,
        storageSolution: storageSolution,
    };

    // Initialize web server
    const {webServer, router, renderConfig, renderGoldenLayout} = await setupWebServer(
        appArgs,
        serverOptions,
        serverDependencies,
    );

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

    // Define compiler change handler
    async function onCompilerChange(compilers) {
        if (JSON.stringify(prevCompilers) === JSON.stringify(compilers)) {
            return;
        }
        logger.info(`Compiler scan count: ${compilers.length}`);
        logger.debug('Compilers:', compilers);
        prevCompilers = compilers;
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
    if (rescanCompilerSecs && !options.options.prediscovered) {
        logger.info(`Rescanning compilers every ${rescanCompilerSecs} secs`);
        setInterval(
            () => compilerFinder.find().then(result => onCompilerChange(result.compilers)),
            rescanCompilerSecs * 1000,
        );
    }

    // Set up metrics server if configured
    if (options.options.metricsPort) {
        logger.info(`Running metrics server on port ${options.options.metricsPort}`);
        setupMetricsServer(options.options.metricsPort, appArgs.hostname);
    }

    // Set up controllers
    try {
        router.use(siteTemplateController.createRouter());
        router.use(sourceController.createRouter());
        router.use(assemblyDocumentationController.createRouter());
        router.use(formattingController.createRouter());
        router.use(noScriptController.createRouter());
    } catch (error) {
        // In case of errors (e.g. during testing), log but continue
        logger.debug('Error setting up controllers, possibly in test environment:', error);
    }

    // Initialize routes
    noscriptHandler.initializeRoutes();
    routeApi.initializeRoutes();

    if (!appArgs.doCache) {
        logger.info('  with disabled caching');
    }

    // Start execution worker if configured
    if (isExecutionWorker) {
        await initHostSpecialties();
        startExecutionWorkerThread(ceProps, awsProps, compilationEnvironment);
    }

    // Start listening for connections
    startListening(webServer, appArgs);

    return {webServer};
}
