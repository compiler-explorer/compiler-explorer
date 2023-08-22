// Copyright (c) 2012, Compiler Explorer Authors
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

import child_process from 'child_process';
import os from 'os';
import path from 'path';
import process from 'process';
import url from 'url';

import * as Sentry from '@sentry/node';
import bodyParser from 'body-parser';
import compression from 'compression';
import express from 'express';
import fs from 'fs-extra';
import morgan from 'morgan';
import nopt from 'nopt';
import PromClient from 'prom-client';
import responseTime from 'response-time';
import sanitize from 'sanitize-filename';
import sFavicon from 'serve-favicon';
import systemdSocket from 'systemd-socket';
import _ from 'underscore';
import urljoin from 'url-join';

import * as aws from './lib/aws.js';
import * as normalizer from './lib/clientstate-normalizer.js';
import {CompilationEnvironment} from './lib/compilation-env.js';
import {CompilationQueue} from './lib/compilation-queue.js';
import {CompilerFinder} from './lib/compiler-finder.js';
// import { policy as csp } from './lib/csp.js';
import {startWineInit} from './lib/exec.js';
import {CompileHandler} from './lib/handlers/compile.js';
import * as healthCheck from './lib/handlers/health-check.js';
import {NoScriptHandler} from './lib/handlers/noscript.js';
import {RouteAPI} from './lib/handlers/route-api.js';
import {loadSiteTemplates} from './lib/handlers/site-templates.js';
import {SourceHandler} from './lib/handlers/source.js';
import {languages as allLanguages} from './lib/languages.js';
import {logger, logToLoki, logToPapertrail, makeLogStream, suppressConsoleLog} from './lib/logger.js';
import {setupMetricsServer} from './lib/metrics-server.js';
import {ClientOptionsHandler} from './lib/options-handler.js';
import * as props from './lib/properties.js';
import {SetupSentry} from './lib/sentry.js';
import {ShortLinkResolver} from './lib/shortener/google.js';
import {sources} from './lib/sources/index.js';
import {loadSponsorsFromString} from './lib/sponsors.js';
import {getStorageTypeByKey} from './lib/storage/index.js';
import * as utils from './lib/utils.js';
import {ElementType} from './shared/common-utils.js';
import type {Language, LanguageKey} from './types/languages.interfaces.js';

// Used by assert.ts
global.ce_base_directory = new URL('.', import.meta.url);

(nopt as any).invalidHandler = (key, val, types) => {
    logger.error(
        `Command line argument type error for "--${key}=${val}", expected ${types.map(t => typeof t).join(' | ')}`,
    );
};

// Parse arguments from command line 'node ./app.js args...'
const opts = nopt({
    env: [String, Array],
    rootDir: [String],
    host: [String],
    port: [Number],
    propDebug: [Boolean],
    debug: [Boolean],
    dist: [Boolean],
    archivedVersions: [String],
    // Ignore fetch marks and assume every compiler is found locally
    noRemoteFetch: [Boolean],
    tmpDir: [String],
    wsl: [Boolean],
    // If specified, only loads the specified languages, resulting in faster loadup/iteration times
    language: [String],
    // Do not use caching for compilation results (Requests might still be cached by the client's browser)
    noCache: [Boolean],
    // Don't cleanly run if two or more compilers have clashing ids
    ensureNoIdClash: [Boolean],
    logHost: [String],
    logPort: [Number],
    hostnameForLogging: [String],
    suppressConsoleLog: [Boolean],
    metricsPort: [Number],
    loki: [String],
    discoveryonly: [String],
    prediscovered: [String],
    version: [Boolean],
    webpackContent: [String],
    noLocal: [Boolean],
}) as Partial<{
    env: string[];
    rootDir: string;
    host: string;
    port: number;
    propDebug: boolean;
    debug: boolean;
    dist: boolean;
    archivedVersions: string;
    noRemoteFetch: boolean;
    tmpDir: string;
    wsl: boolean;
    language: string;
    noCache: boolean;
    ensureNoIdClash: boolean;
    logHost: string;
    logPort: number;
    hostnameForLogging: string;
    suppressConsoleLog: boolean;
    metricsPort: number;
    loki: string;
    discoveryonly: string;
    prediscovered: string;
    version: boolean;
    webpackContent: string;
    noLocal: boolean;
}>;

if (opts.debug) logger.level = 'debug';

// AP: Detect if we're running under Windows Subsystem for Linux. Temporary modification
// of process.env is allowed: https://nodejs.org/api/process.html#process_process_env
if (process.platform === 'linux' && child_process.execSync('uname -a').toString().toLowerCase().includes('microsoft')) {
    // Node wants process.env is essentially a Record<key, string | undefined>. Any non-empty string should be fine.
    process.env.wsl = 'true';
}

// AP: Allow setting of tmpDir (used in lib/base-compiler.js & lib/exec.js) through opts.
// WSL requires a directory on a Windows volume. Set that to Windows %TEMP% if no tmpDir supplied.
// If a tempDir is supplied then assume that it will work for WSL processes as well.
if (opts.tmpDir) {
    process.env.tmpDir = opts.tmpDir;
    process.env.winTmp = opts.tmpDir;
} else if (process.env.wsl) {
    // Dec 2017 preview builds of WSL include /bin/wslpath; do the parsing work for now.
    // Parsing example %TEMP% is C:\Users\apardoe\AppData\Local\Temp
    const windowsTemp = child_process.execSync('cmd.exe /c echo %TEMP%').toString().replaceAll('\\', '/');
    const driveLetter = windowsTemp.substring(0, 1).toLowerCase();
    const directoryPath = windowsTemp.substring(2).trim();
    process.env.winTmp = path.join('/mnt', driveLetter, directoryPath);
}

const distPath = utils.resolvePathFromAppRoot('.');
logger.debug(`Distpath=${distPath}`);

const gitReleaseName = (() => {
    // Use the canned git_hash if provided
    const gitHashFilePath = path.join(distPath, 'git_hash');
    if (opts.dist && fs.existsSync(gitHashFilePath)) {
        return fs.readFileSync(gitHashFilePath).toString().trim();
    }

    // Just if we have been cloned and not downloaded (Thanks David!)
    if (fs.existsSync('.git/')) {
        return child_process.execSync('git rev-parse HEAD').toString().trim();
    }

    // unknown case
    return '';
})();

const releaseBuildNumber = (() => {
    // Use the canned build only if provided
    const releaseBuildPath = path.join(distPath, 'release_build');
    if (opts.dist && fs.existsSync(releaseBuildPath)) {
        return fs.readFileSync(releaseBuildPath).toString().trim();
    }
    return '';
})();

export type AppDefaultArguments = {
    rootDir: string;
    env: string[];
    hostname?: string;
    port: number;
    gitReleaseName: string;
    releaseBuildNumber: string;
    wantedLanguages: string | null;
    doCache: boolean;
    fetchCompilersFromRemote: boolean;
    ensureNoCompilerClash: boolean | undefined;
    suppressConsoleLog: boolean;
};

// Set default values for omitted arguments
const defArgs: AppDefaultArguments = {
    rootDir: opts.rootDir || './etc',
    env: opts.env || ['dev'],
    hostname: opts.host,
    port: opts.port || 10240,
    gitReleaseName: gitReleaseName,
    releaseBuildNumber: releaseBuildNumber,
    wantedLanguages: opts.language || null,
    doCache: !opts.noCache,
    fetchCompilersFromRemote: !opts.noRemoteFetch,
    ensureNoCompilerClash: opts.ensureNoIdClash,
    suppressConsoleLog: opts.suppressConsoleLog || false,
};

if (opts.logHost && opts.logPort) {
    logToPapertrail(opts.logHost, opts.logPort, defArgs.env.join('.'), opts.hostnameForLogging);
}

if (opts.loki) {
    logToLoki(opts.loki);
}

if (defArgs.suppressConsoleLog) {
    logger.info('Disabling further console logging');
    suppressConsoleLog();
}

const isDevMode = () => process.env.NODE_ENV !== 'production';

function getFaviconFilename() {
    if (isDevMode()) {
        return 'favicon-dev.ico';
    } else if (opts.env && opts.env.includes('beta')) {
        return 'favicon-beta.ico';
    } else if (opts.env && opts.env.includes('staging')) {
        return 'favicon-staging.ico';
    } else {
        return 'favicon.ico';
    }
}

const propHierarchy = [
    'defaults',
    defArgs.env,
    _.map(defArgs.env, e => `${e}.${process.platform}`),
    process.platform,
    os.hostname(),
].flat();
if (!opts.noLocal) {
    propHierarchy.push('local');
}
logger.info(`properties hierarchy: ${propHierarchy.join(', ')}`);

// Propagate debug mode if need be
if (opts.propDebug) props.setDebug(true);

// *All* files in config dir are parsed
const configDir = path.join(defArgs.rootDir, 'config');
props.initialize(configDir, propHierarchy);
// Instantiate a function to access records concerning "compiler-explorer"
// in hidden object props.properties
const ceProps = props.propsFor('compiler-explorer');
defArgs.wantedLanguages = ceProps<string>('restrictToLanguages', defArgs.wantedLanguages);

const languages = (() => {
    if (defArgs.wantedLanguages) {
        const filteredLangs: Partial<Record<LanguageKey, Language>> = {};
        const passedLangs = defArgs.wantedLanguages.split(',');
        for (const wantedLang of passedLangs) {
            for (const lang of Object.values(allLanguages)) {
                if (lang.id === wantedLang || lang.name === wantedLang || lang.alias.includes(wantedLang)) {
                    filteredLangs[lang.id] = lang;
                }
            }
        }
        // Always keep cmake for IDE mode, just in case
        filteredLangs[allLanguages.cmake.id] = allLanguages.cmake;
        return filteredLangs;
    } else {
        return allLanguages;
    }
})();

if (Object.keys(languages).length === 0) {
    logger.error('Trying to start Compiler Explorer without a language');
}

const compilerProps = new props.CompilerProps(languages, ceProps);

const staticPath = opts.webpackContent || path.join(distPath, 'static');
const staticMaxAgeSecs = ceProps('staticMaxAgeSecs', 0);
const maxUploadSize = ceProps('maxUploadSize', '1mb');
const extraBodyClass = ceProps('extraBodyClass', isDevMode() ? 'dev' : '');
const storageSolution = compilerProps.ceProps('storageSolution', 'local');
const httpRoot = urljoin(ceProps('httpRoot', '/'), '/');

const staticUrl = ceProps<string | undefined>('staticUrl');
const staticRoot = urljoin(staticUrl || urljoin(httpRoot, 'static'), '/');

function staticHeaders(res) {
    if (staticMaxAgeSecs) {
        res.setHeader('Cache-Control', 'public, max-age=' + staticMaxAgeSecs + ', must-revalidate');
    }
}

function contentPolicyHeader(res: express.Response) {
    // TODO: re-enable CSP
    // if (csp) {
    //     res.setHeader('Content-Security-Policy', csp);
    // }
}

function measureEventLoopLag(delayMs: number) {
    return new Promise<number>(resolve => {
        const start = process.hrtime.bigint();
        setTimeout(() => {
            const elapsed = process.hrtime.bigint() - start;
            const delta = elapsed - BigInt(delayMs * 1000000);
            return resolve(Number(delta) / 1000000);
        }, delayMs);
    });
}

function setupEventLoopLagLogging() {
    const lagIntervalMs = ceProps('eventLoopMeasureIntervalMs', 0);
    const thresWarn = ceProps('eventLoopLagThresholdWarn', 0);
    const thresErr = ceProps('eventLoopLagThresholdErr', 0);

    let totalLag = 0;
    const ceLagSecondsTotalGauge = new PromClient.Gauge({
        name: 'ce_lag_seconds_total',
        help: 'Total event loop lag since application startup',
    });

    async function eventLoopLagHandler() {
        const lagMs = await measureEventLoopLag(lagIntervalMs);
        totalLag += Math.max(lagMs / 1000, 0);
        ceLagSecondsTotalGauge.set(totalLag);

        if (thresErr && lagMs >= thresErr) {
            logger.error(`Event Loop Lag: ${lagMs} ms`);
        } else if (thresWarn && lagMs >= thresWarn) {
            logger.warn(`Event Loop Lag: ${lagMs} ms`);
        }

        setImmediate(eventLoopLagHandler);
    }

    if (lagIntervalMs > 0) {
        setImmediate(eventLoopLagHandler);
    }
}

let pugRequireHandler: (path: string) => any = () => {
    logger.error('pug require handler not configured');
};

async function setupWebPackDevMiddleware(router: express.Router) {
    logger.info('  using webpack dev middleware');

    /* eslint-disable node/no-unpublished-import,import/extensions, */
    const {default: webpackDevMiddleware} = await import('webpack-dev-middleware');
    const {default: webpackConfig} = await import('./webpack.config.esm.js');
    const {default: webpack} = await import('webpack');
    /* eslint-enable */
    type WebpackConfiguration = ElementType<Parameters<typeof webpack>[0]>;

    const webpackCompiler = webpack([webpackConfig as WebpackConfiguration]);
    router.use(
        webpackDevMiddleware(webpackCompiler, {
            publicPath: '/static',
            stats: {
                preset: 'errors-only',
                timings: true,
            },
        }),
    );

    pugRequireHandler = path => urljoin(httpRoot, 'static', path);
}

async function setupStaticMiddleware(router: express.Router) {
    const staticManifest = await fs.readJson(path.join(distPath, 'manifest.json'));

    if (staticUrl) {
        logger.info(`  using static files from '${staticUrl}'`);
    } else {
        logger.info(`  serving static files from '${staticPath}'`);
        router.use(
            '/static',
            express.static(staticPath, {
                maxAge: staticMaxAgeSecs * 1000,
            }),
        );
    }

    pugRequireHandler = path => {
        if (Object.prototype.hasOwnProperty.call(staticManifest, path)) {
            return urljoin(staticRoot, staticManifest[path]);
        } else {
            logger.error(`failed to locate static asset '${path}' in manifest`);
            return '';
        }
    };
}

const googleShortUrlResolver = new ShortLinkResolver();

function oldGoogleUrlHandler(req: express.Request, res: express.Response, next: express.NextFunction) {
    const id = req.params.id;
    const googleUrl = `https://goo.gl/${encodeURIComponent(id)}`;
    googleShortUrlResolver
        .resolve(googleUrl)
        .then(resultObj => {
            const parsed = new url.URL(resultObj.longUrl);
            const allowedRe = new RegExp(ceProps<string>('allowedShortUrlHostRe'));
            if (parsed.host.match(allowedRe) === null) {
                logger.warn(`Denied access to short URL ${id} - linked to ${resultObj.longUrl}`);
                return next({
                    statusCode: 404,
                    message: `ID "${id}" could not be found`,
                });
            }
            res.writeHead(301, {
                Location: resultObj.longUrl,
                'Cache-Control': 'public',
            });
            res.end();
        })
        .catch(e => {
            logger.error(`Failed to expand ${googleUrl} - ${e}`);
            next({
                statusCode: 404,
                message: `ID "${id}" could not be found`,
            });
        });
}

function startListening(server: express.Express) {
    const ss = systemdSocket();
    let _port;
    if (ss) {
        // ms (5 min default)
        const idleTimeout = process.env.IDLE_TIMEOUT;
        const timeout = (idleTimeout === undefined ? 300 : parseInt(idleTimeout)) * 1000;
        if (idleTimeout) {
            const exit = () => {
                logger.info('Inactivity timeout reached, exiting.');
                process.exit(0);
            };
            let idleTimer = setTimeout(exit, timeout);
            const reset = () => {
                clearTimeout(idleTimer);
                idleTimer = setTimeout(exit, timeout);
            };
            server.all('*', reset);
            logger.info(`  IDLE_TIMEOUT: ${idleTimeout}`);
        }
        _port = ss;
    } else {
        _port = defArgs.port;
    }

    const startupGauge = new PromClient.Gauge({
        name: 'ce_startup_seconds',
        help: 'Time taken from process start to serving requests',
    });
    startupGauge.set(process.uptime());
    const startupDurationMs = Math.floor(process.uptime() * 1000);
    if (isNaN(parseInt(_port))) {
        // unix socket, not a port number...
        logger.info(`  Listening on socket: //${_port}/`);
        logger.info(`  Startup duration: ${startupDurationMs}ms`);
        logger.info('=======================================');
        server.listen(_port);
    } else {
        // normal port number
        logger.info(`  Listening on http://${defArgs.hostname || 'localhost'}:${_port}/`);
        logger.info(`  Startup duration: ${startupDurationMs}ms`);
        logger.info('=======================================');
        // silly express typing, passing undefined is fine but
        if (defArgs.hostname) {
            server.listen(_port, defArgs.hostname);
        } else {
            server.listen(_port);
        }
    }
}

const awsProps = props.propsFor('aws');

// eslint-disable-next-line max-statements
async function main() {
    await aws.initConfig(awsProps);
    // Initialise express and then sentry. Sentry as early as possible to catch errors during startup.
    const webServer = express(),
        router = express.Router();
    SetupSentry(aws.getConfig('sentryDsn'), ceProps, releaseBuildNumber, gitReleaseName, defArgs);

    startWineInit();

    const clientOptionsHandler = new ClientOptionsHandler(sources, compilerProps, defArgs);
    const compilationQueue = CompilationQueue.fromProps(compilerProps.ceProps);
    const compilationEnvironment = new CompilationEnvironment(compilerProps, compilationQueue, defArgs.doCache);
    const compileHandler = new CompileHandler(compilationEnvironment, awsProps);
    const storageType = getStorageTypeByKey(storageSolution);
    const storageHandler = new storageType(httpRoot, compilerProps, awsProps);
    const sourceHandler = new SourceHandler(sources, staticHeaders);
    const compilerFinder = new CompilerFinder(compileHandler, compilerProps, awsProps, defArgs, clientOptionsHandler);

    logger.info('=======================================');
    if (gitReleaseName) logger.info(`  git release ${gitReleaseName}`);
    if (releaseBuildNumber) logger.info(`  release build ${releaseBuildNumber}`);

    let initialCompilers;
    let prevCompilers;

    if (opts.prediscovered) {
        const prediscoveredCompilersJson = await fs.readFile(opts.prediscovered, 'utf8');
        initialCompilers = JSON.parse(prediscoveredCompilersJson);
        await compilerFinder.loadPrediscovered(initialCompilers);
    } else {
        const initialFindResults = await compilerFinder.find();
        initialCompilers = initialFindResults.compilers;
        if (defArgs.ensureNoCompilerClash) {
            logger.warn('Ensuring no compiler ids clash');
            if (initialFindResults.foundClash) {
                // If we are forced to have no clashes, throw an error with some explanation
                throw new Error('Clashing compilers in the current environment found!');
            } else {
                logger.info('No clashing ids found, continuing normally...');
            }
        }
    }

    if (opts.discoveryonly) {
        for (const compiler of initialCompilers) {
            if (compiler.buildenvsetup && compiler.buildenvsetup.id === '') delete compiler.buildenvsetup;

            if (compiler.externalparser && compiler.externalparser.id === '') delete compiler.externalparser;

            const compilerInstance = compilerFinder.compileHandler.findCompiler(compiler.lang, compiler.id);
            if (compilerInstance) {
                compiler.cachedPossibleArguments = compilerInstance.possibleArguments.possibleArguments;
            }
        }
        await fs.writeFile(opts.discoveryonly, JSON.stringify(initialCompilers));
        logger.info(`Discovered compilers saved to ${opts.discoveryonly}`);
        process.exit(0);
    }

    const healthCheckFilePath = ceProps('healthCheckFilePath', false);

    const handlerConfig = {
        compileHandler,
        clientOptionsHandler,
        storageHandler,
        ceProps,
        opts,
        defArgs,
        renderConfig,
        renderGoldenLayout,
        staticHeaders,
        contentPolicyHeader,
    };

    const noscriptHandler = new NoScriptHandler(router, handlerConfig);
    const routeApi = new RouteAPI(router, handlerConfig);

    async function onCompilerChange(compilers) {
        if (JSON.stringify(prevCompilers) === JSON.stringify(compilers)) {
            return;
        }
        logger.info(`Compiler scan count: ${_.size(compilers)}`);
        logger.debug('Compilers:', compilers);
        prevCompilers = compilers;
        await clientOptionsHandler.setCompilers(compilers);
        routeApi.apiHandler.setCompilers(compilers);
        routeApi.apiHandler.setLanguages(languages);
        routeApi.apiHandler.setOptions(clientOptionsHandler);
    }

    await onCompilerChange(initialCompilers);

    const rescanCompilerSecs = ceProps('rescanCompilerSecs', 0);
    if (rescanCompilerSecs && !opts.prediscovered) {
        logger.info(`Rescanning compilers every ${rescanCompilerSecs} secs`);
        setInterval(
            () => compilerFinder.find().then(result => onCompilerChange(result.compilers)),
            rescanCompilerSecs * 1000,
        );
    }

    const sentrySlowRequestMs = ceProps('sentrySlowRequestMs', 0);

    if (opts.metricsPort) {
        logger.info(`Running metrics server on port ${opts.metricsPort}`);
        setupMetricsServer(opts.metricsPort, defArgs.hostname);
    }

    webServer
        .set('trust proxy', true)
        .set('view engine', 'pug')
        .on('error', err => logger.error('Caught error in web handler; continuing:', err))
        // sentry request handler must be the first middleware on the app
        .use(
            Sentry.Handlers.requestHandler({
                ip: true,
            }),
        )
        // eslint-disable-next-line no-unused-vars
        .use(
            responseTime((req, res, time) => {
                if (sentrySlowRequestMs > 0 && time >= sentrySlowRequestMs) {
                    Sentry.withScope(scope => {
                        scope.setExtra('duration_ms', time);
                        Sentry.captureMessage('SlowRequest', 'warning');
                    });
                }
            }),
        )
        // Handle healthchecks at the root, as they're not expected from the outside world
        .use('/healthcheck', new healthCheck.HealthCheckHandler(compilationQueue, healthCheckFilePath).handle)
        .use(httpRoot, router)
        .use((req, res, next) => {
            next({status: 404, message: `page "${req.path}" could not be found`});
        })
        // sentry error handler must be the first error handling middleware
        .use(Sentry.Handlers.errorHandler)
        // eslint-disable-next-line no-unused-vars
        .use((err, req, res, next) => {
            const status =
                err.status || err.statusCode || err.status_code || (err.output && err.output.statusCode) || 500;
            const message = err.message || 'Internal Server Error';
            res.status(status);
            res.render('error', renderConfig({error: {code: status, message: message}}));
            if (status >= 500) {
                logger.error('Internal server error:', err);
            }
        });

    const sponsorConfig = loadSponsorsFromString(fs.readFileSync(configDir + '/sponsors.yaml', 'utf8'));

    loadSiteTemplates(configDir);

    function renderConfig(extra, urlOptions?) {
        const urlOptionsAllowed = ['readOnly', 'hideEditorToolbars', 'language'];
        const filteredUrlOptions = _.mapObject(_.pick(urlOptions, urlOptionsAllowed), val => utils.toProperty(val));
        const allExtraOptions = _.extend({}, filteredUrlOptions, extra);

        if (allExtraOptions.mobileViewer && allExtraOptions.config) {
            const clnormalizer = new normalizer.ClientStateNormalizer();
            clnormalizer.fromGoldenLayout(allExtraOptions.config);
            const clientstate = clnormalizer.normalized;

            const glnormalizer = new normalizer.ClientStateGoldenifier();
            allExtraOptions.slides = glnormalizer.generatePresentationModeMobileViewerSlides(clientstate);
        }

        const options = _.extend({}, allExtraOptions, clientOptionsHandler.get());
        options.optionsHash = clientOptionsHandler.getHash();
        options.compilerExplorerOptions = JSON.stringify(allExtraOptions);
        options.extraBodyClass = options.embedded ? 'embedded' : extraBodyClass;
        options.httpRoot = httpRoot;
        options.staticRoot = staticRoot;
        options.storageSolution = storageSolution;
        options.require = pugRequireHandler;
        options.sponsors = sponsorConfig;
        return options;
    }

    function isMobileViewer(req) {
        return req.header('CloudFront-Is-Mobile-Viewer') === 'true';
    }

    function renderGoldenLayout(config, metadata, req, res) {
        staticHeaders(res);
        contentPolicyHeader(res);

        const embedded = req.query.embedded === 'true' ? true : false;

        res.render(
            embedded ? 'embed' : 'index',
            renderConfig(
                {
                    embedded: embedded,
                    mobileViewer: isMobileViewer(req),
                    config: config,
                    metadata: metadata,
                    storedStateId: req.params.id || false,
                },
                req.query,
            ),
        );
    }

    const embeddedHandler = function (req: express.Request, res: express.Response) {
        staticHeaders(res);
        contentPolicyHeader(res);
        res.render(
            'embed',
            renderConfig(
                {
                    embedded: true,
                    mobileViewer: isMobileViewer(req),
                },
                req.query,
            ),
        );
    };

    await (isDevMode() ? setupWebPackDevMiddleware(router) : setupStaticMiddleware(router));

    morgan.token('gdpr_ip', (req: any) => (req.ip ? utils.anonymizeIp(req.ip) : ''));

    // Based on combined format, but: GDPR compliant IP, no timestamp & no unused fields for our usecase
    const morganFormat = isDevMode() ? 'dev' : ':gdpr_ip ":method :url" :status';

    /*
     * This is a workaround to make cross origin monaco web workers function
     * in spite of the monaco webpack plugin hijacking the MonacoEnvironment global.
     *
     * see https://github.com/microsoft/monaco-editor-webpack-plugin/issues/42
     *
     * This workaround wouldn't be so bad, if it didn't _also_ rely on *another* bug to
     * actually work.
     *
     * The webpack plugin incorrectly uses
     *     window.__webpack_public_path__
     * when it should use
     *     __webpack_public_path__
     *
     * see https://github.com/microsoft/monaco-editor-webpack-plugin/pull/63
     *
     * We can leave __webpack_public_path__ with the correct value, which lets runtime chunk
     * loading continue to function correctly.
     *
     * We can then set window.__webpack_public_path__ to the below handler, which lets us
     * fabricate a worker on the fly.
     *
     * This is bad and I feel bad.
     *
     * This should no longer be needed, but is left here for safety because people with
     * workers already installed from this url may still try to hit this page for some time
     *
     * TODO: remove this route in the future now that it is not needed
     */
    router.get('/workers/:worker', (req, res) => {
        staticHeaders(res);
        res.set('Content-Type', 'application/javascript');
        res.end(`importScripts('${urljoin(staticRoot, req.params.worker)}');`);
    });

    router
        .use(
            morgan(morganFormat, {
                stream: makeLogStream('info'),
                // Skip for non errors (2xx, 3xx)
                skip: (req, res) => res.statusCode >= 400,
            }),
        )
        .use(
            morgan(morganFormat, {
                stream: makeLogStream('warn'),
                // Skip for non user errors (4xx)
                skip: (req, res) => res.statusCode < 400 || res.statusCode >= 500,
            }),
        )
        .use(
            morgan(morganFormat, {
                stream: makeLogStream('error'),
                // Skip for non server errors (5xx)
                skip: (req, res) => res.statusCode < 500,
            }),
        )
        .use(compression())
        .get('/', (req, res) => {
            staticHeaders(res);
            contentPolicyHeader(res);
            res.render(
                'index',
                renderConfig(
                    {
                        embedded: false,
                        mobileViewer: isMobileViewer(req),
                    },
                    req.query,
                ),
            );
        })
        .get('/e', embeddedHandler)
        // legacy. not a 301 to prevent any redirect loops between old e links and embed.html
        .get('/embed.html', embeddedHandler)
        .get('/embed-ro', (req, res) => {
            staticHeaders(res);
            contentPolicyHeader(res);
            res.render(
                'embed',
                renderConfig(
                    {
                        embedded: true,
                        readOnly: true,
                        mobileViewer: isMobileViewer(req),
                    },
                    req.query,
                ),
            );
        })
        .get('/robots.txt', (req, res) => {
            staticHeaders(res);
            res.end('User-agent: *\nSitemap: https://godbolt.org/sitemap.xml\nDisallow:');
        })
        .get('/sitemap.xml', (req, res) => {
            staticHeaders(res);
            res.set('Content-Type', 'application/xml');
            res.render('sitemap');
        })
        .use(sFavicon(utils.resolvePathFromAppRoot('static/favicons', getFaviconFilename())))
        .get('/client-options.js', (req, res) => {
            staticHeaders(res);
            res.set('Content-Type', 'application/javascript');
            res.end(`window.compilerExplorerOptions = ${clientOptionsHandler.getJSON()};`);
        })
        .use('/bits/:bits(\\w+).html', (req, res) => {
            staticHeaders(res);
            contentPolicyHeader(res);
            res.render(
                `bits/${sanitize(req.params.bits)}`,
                renderConfig(
                    {
                        embedded: false,
                        mobileViewer: isMobileViewer(req),
                    },
                    req.query,
                ),
            );
        })
        .use(bodyParser.json({limit: ceProps('bodyParserLimit', maxUploadSize)}))
        .use('/source', sourceHandler.handle.bind(sourceHandler))
        .get('/g/:id', oldGoogleUrlHandler)
        // Deprecated old route for this -- TODO remove in late 2021
        .post('/shortener', routeApi.apiHandler.shortener.handle.bind(routeApi.apiHandler.shortener));

    noscriptHandler.InitializeRoutes({limit: ceProps('bodyParserLimit', maxUploadSize)});
    routeApi.InitializeRoutes();

    if (!defArgs.doCache) {
        logger.info('  with disabled caching');
    }
    setupEventLoopLagLogging();
    startListening(webServer);
}

if (opts.version) {
    logger.info('Compiler Explorer version info:');
    logger.info(`  git release ${gitReleaseName}`);
    logger.info(`  release build ${releaseBuildNumber}`);
    logger.info('Exiting');
    process.exit(0);
}

process.on('uncaughtException', uncaughtHandler);
process.on('SIGINT', signalHandler('SIGINT'));
process.on('SIGTERM', signalHandler('SIGTERM'));
process.on('SIGQUIT', signalHandler('SIGQUIT'));

function signalHandler(name: string) {
    return () => {
        logger.info(`stopping process: ${name}`);
        process.exit(0);
    };
}

function uncaughtHandler(err: Error, origin: NodeJS.UncaughtExceptionOrigin) {
    logger.info(`stopping process: Uncaught exception: ${err}\nException origin: ${origin}`);
    // The app will exit naturally from here, but if we call `process.exit()` we may lose log lines.
    // see https://github.com/winstonjs/winston/issues/1504#issuecomment-1033087411
    process.exitCode = 1;
}

// Once we move to modules, we can remove this and use a top level await.
// eslint-disable-next-line unicorn/prefer-top-level-await
main().catch(err => {
    logger.error('Top-level error (shutting down):', err);
    // Shut down after a second to hopefully let logs flush.
    setTimeout(() => process.exit(1), 1000);
});
