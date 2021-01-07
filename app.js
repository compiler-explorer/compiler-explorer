#!/usr/bin/env node
// shebang interferes with license header plugin
/* eslint-disable header/header */

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
import * as Tracing from '@sentry/tracing';
import bodyParser from 'body-parser';
import compression from 'compression';
import express from 'express';
import fs from 'fs-extra';
import morgan from 'morgan';
import nopt from 'nopt';
import PromClient from 'prom-client';
import responseTime from 'response-time';
import sFavicon from 'serve-favicon';
import systemdSocket from 'systemd-socket';
import _ from 'underscore';
import urljoin from 'url-join';

import * as aws from './lib/aws';
import * as normalizer from './lib/clientstate-normalizer';
import { CompilationEnvironment } from './lib/compilation-env';
import { CompilationQueue } from './lib/compilation-queue';
import { CompilerFinder } from './lib/compiler-finder';
import { policy as csp } from './lib/csp';
import { initialiseWine } from './lib/exec';
import { ShortLinkResolver } from './lib/google';
import { CompileHandler } from './lib/handlers/compile';
import * as healthCheck from './lib/handlers/health-check';
import { NoScriptHandler } from './lib/handlers/noscript';
import { RouteAPI } from './lib/handlers/route-api';
import { SourceHandler } from './lib/handlers/source';
import { languages as allLanguages } from './lib/languages';
import { logger, logToPapertrail, suppressConsoleLog } from './lib/logger';
import { ClientOptionsHandler } from './lib/options-handler';
import * as props from './lib/properties';
import { getShortenerTypeByKey } from './lib/shortener';
import { sources } from './lib/sources';
import { loadSponsorsFromString } from './lib/sponsors';
import { getStorageTypeByKey } from './lib/storage';
import * as utils from './lib/utils';

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
    // If specified, only loads the specified language, resulting in faster loadup/iteration times
    language: [String],
    // Do not use caching for compilation results (Requests might still be cached by the client's browser)
    noCache: [Boolean],
    // Don't cleanly run if two or more compilers have clashing ids
    ensureNoIdClash: [Boolean],
    logHost: [String],
    logPort: [Number],
    suppressConsoleLog: [Boolean],
    metricsPort: [Number],
});

if (opts.debug) logger.level = 'debug';

// AP: Detect if we're running under Windows Subsystem for Linux. Temporary modification
// of process.env is allowed: https://nodejs.org/api/process.html#process_process_env
if (process.platform === 'linux' && child_process.execSync('uname -a').toString().includes('Microsoft')) {
    process.env.wsl = true;
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
    const windowsTemp = child_process.execSync('cmd.exe /c echo %TEMP%').toString().replace(/\\/g, '/');
    const driveLetter = windowsTemp.substring(0, 1).toLowerCase();
    const directoryPath = windowsTemp.substring(2).trim();
    process.env.winTmp = path.join('/mnt', driveLetter, directoryPath);
}

const distPath = utils.resolvePathFromAppRoot('out', 'dist');

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

const travisBuildNumber = (() => {
    // Use the canned travis_build only if provided
    const travisBuildPath = path.join(distPath, 'travis_build');
    if (opts.dist && fs.existsSync(travisBuildPath)) {
        return fs.readFileSync(travisBuildPath).toString().trim();
    }

    // non-travis build
    return '';
})();

// Set default values for omitted arguments
const defArgs = {
    rootDir: opts.rootDir || './etc',
    env: opts.env || ['dev'],
    hostname: opts.host,
    port: opts.port || 10240,
    gitReleaseName: gitReleaseName,
    travisBuildNumber: travisBuildNumber,
    wantedLanguage: opts.language || null,
    doCache: !opts.noCache,
    fetchCompilersFromRemote: !opts.noRemoteFetch,
    ensureNoCompilerClash: opts.ensureNoIdClash,
    suppressConsoleLog: opts.suppressConsoleLog || false,
};

if (opts.logHost && opts.logPort) {
    logToPapertrail(opts.logHost, opts.logPort, defArgs.env.join('.'));
}

if (defArgs.suppressConsoleLog) {
    logger.info('Disabling further console logging');
    suppressConsoleLog();
}

const isDevMode = () => process.env.NODE_ENV !== 'production';

const propHierarchy = _.flatten([
    'defaults',
    defArgs.env,
    _.map(defArgs.env, e => `${e}.${process.platform}`),
    process.platform,
    os.hostname(),
    'local']);
logger.info(`properties hierarchy: ${propHierarchy.join(', ')}`);

// Propagate debug mode if need be
if (opts.propDebug) props.setDebug(true);

// *All* files in config dir are parsed
const configDir = path.join(defArgs.rootDir, 'config');
props.initialize(configDir, propHierarchy);
// Instantiate a function to access records concerning "compiler-explorer"
// in hidden object props.properties
const ceProps = props.propsFor('compiler-explorer');

let languages = allLanguages;
if (defArgs.wantedLanguage) {
    const filteredLangs = {};
    _.each(languages, lang => {
        if (lang.id === defArgs.wantedLanguage ||
            lang.name === defArgs.wantedLanguage ||
            (lang.alias && lang.alias.includes(defArgs.wantedLanguage))) {
            filteredLangs[lang.id] = lang;
        }
    });
    languages = filteredLangs;
}

if (languages.length === 0) {
    logger.error('Trying to start Compiler Explorer without a language');
}

const compilerProps = new props.CompilerProps(languages, ceProps);

const staticMaxAgeSecs = ceProps('staticMaxAgeSecs', 0);
const maxUploadSize = ceProps('maxUploadSize', '1mb');
const extraBodyClass = ceProps('extraBodyClass', isDevMode() ? 'dev' : '');
const storageSolution = compilerProps.ceProps('storageSolution', 'local');
const httpRoot = urljoin(ceProps('httpRoot', '/'), '/');

const staticUrl = ceProps('staticUrl');
const staticRoot = urljoin(staticUrl || urljoin(httpRoot, 'static'), '/');

function staticHeaders(res) {
    if (staticMaxAgeSecs) {
        res.setHeader('Cache-Control', 'public, max-age=' + staticMaxAgeSecs + ', must-revalidate');
    }
}

function contentPolicyHeader(res) {
    // TODO: re-enable CSP
    if (csp && 0) {
        res.setHeader('Content-Security-Policy', csp);
    }
}

function measureEventLoopLag(delayMs) {
    return new Promise((resolve) => {
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

let pugRequireHandler = () => {
    logger.error('pug require handler not configured');
};

async function setupWebPackDevMiddleware(router) {
    logger.info('  using webpack dev middleware');

    /* eslint-disable node/no-unpublished-import */
    const webpackDevMiddleware = (await import('webpack-dev-middleware')).default;
    const webpackConfig = (await import('./webpack.config.esm.js')).default;
    const webpack = (await import('webpack')).default;
    /* eslint-enable */

    const webpackCompiler = webpack(webpackConfig);
    router.use(webpackDevMiddleware(webpackCompiler, {
        publicPath: '/static',
        logger: logger,
        stats: 'errors-only',
    }));

    pugRequireHandler = (path) => urljoin(httpRoot, 'static', path);
}

async function setupStaticMiddleware(router) {
    const staticManifest = await fs.readJson(path.join(distPath, 'manifest.json'));

    if (staticUrl) {
        logger.info(`  using static files from '${staticUrl}'`);
    } else {
        const staticPath = path.join(distPath, 'static');
        logger.info(`  serving static files from '${staticPath}'`);
        router.use('/static', express.static(staticPath, {
            maxAge: staticMaxAgeSecs * 1000,
        }));
    }

    pugRequireHandler = (path) => {
        if (Object.prototype.hasOwnProperty.call(staticManifest, path)) {
            return urljoin(staticRoot, staticManifest[path]);
        } else {
            logger.error(`failed to locate static asset '${path}' in manifest`);
            return '';
        }
    };
}

function shouldRedactRequestData(data) {
    try {
        const parsed = JSON.parse(data);
        return !parsed['allowStoreCodeDebug'];
    } catch (e) {
        return true;
    }
}

const googleShortUrlResolver = new ShortLinkResolver();

function oldGoogleUrlHandler(req, res, next) {
    const bits = req.url.split('/');
    if (bits.length !== 2 || req.method !== 'GET') return next();
    const googleUrl = `https://goo.gl/${encodeURIComponent(bits[1])}`;
    googleShortUrlResolver.resolve(googleUrl)
        .then(resultObj => {
            const parsed = new url.URL(resultObj.longUrl);
            const allowedRe = new RegExp(ceProps('allowedShortUrlHostRe'));
            if (parsed.host.match(allowedRe) === null) {
                logger.warn(`Denied access to short URL ${bits[1]} - linked to ${resultObj.longUrl}`);
                return next();
            }
            res.writeHead(301, {
                Location: resultObj.longUrl,
                'Cache-Control': 'public',
            });
            res.end();
        })
        .catch(e => {
            logger.error(`Failed to expand ${googleUrl} - ${e}`);
            next();
        });
}

function startListening(server) {
    const ss = systemdSocket();
    let _port;
    if (ss) {
        // ms (5 min default)
        const idleTimeout = process.env.IDLE_TIMEOUT;
        const timeout = (typeof idleTimeout !== 'undefined' ? idleTimeout : 300) * 1000;
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

    const startupDurationMs = Math.floor(process.uptime() * 1000);
    logger.info(`  Listening on http://${defArgs.hostname || 'localhost'}:${_port}/`);
    logger.info(`  Startup duration: ${startupDurationMs}ms`);
    logger.info('=======================================');
    server.listen(_port, defArgs.hostname);
}

function setupSentry(sentryDsn, expressApp) {
    if (!sentryDsn) {
        logger.info('Not configuring sentry');
        return;
    }
    const sentryEnv = ceProps('sentryEnvironment');
    Sentry.init({
        dsn: sentryDsn,
        release: travisBuildNumber || gitReleaseName,
        environment: sentryEnv || defArgs.env[0],
        beforeSend(event) {
            if (event.request
                && event.request.data
                && shouldRedactRequestData(event.request.data)) {
                event.request.data = JSON.stringify({redacted: true});
            }
            return event;
        },
        integrations: [
            // enable HTTP calls tracing
            new Sentry.Integrations.Http({tracing: true}),
            // enable Express.js middleware tracing
            new Tracing.Integrations.Express({expressApp}),
        ],
        tracesSampleRate: 0.1,
    });
    logger.info(`Configured with Sentry endpoint ${sentryDsn}`);
}

const awsProps = props.propsFor('aws');

// eslint-disable-next-line max-statements
async function main() {
    await aws.initConfig(awsProps);
    await initialiseWine();

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
    if (travisBuildNumber) logger.info(`  travis build ${travisBuildNumber}`);

    const initialFindResults = await compilerFinder.find();
    const initialCompilers = initialFindResults.compilers;
    let prevCompilers;
    if (defArgs.ensureNoCompilerClash) {
        logger.warn('Ensuring no compiler ids clash');
        if (initialFindResults.foundClash) {
            // If we are forced to have no clashes, throw an error with some explanation
            throw new Error('Clashing compilers in the current environment found!');
        } else {
            logger.info('No clashing ids found, continuing normally...');
        }
    }

    const webServer = express(), router = express.Router();
    setupSentry(aws.getConfig('sentryDsn'), webServer);
    const healthCheckFilePath = ceProps('healthCheckFilePath', false);

    const handlerConfig = {
        compileHandler,
        clientOptionsHandler,
        storageHandler,
        ceProps,
        opts,
        renderConfig,
        renderGoldenLayout,
        staticHeaders,
        contentPolicyHeader,
    };

    const noscriptHandler = new NoScriptHandler(router, handlerConfig);
    const routeApi = new RouteAPI(router, handlerConfig, noscriptHandler.renderNoScriptLayout);

    function onCompilerChange(compilers) {
        if (JSON.stringify(prevCompilers) === JSON.stringify(compilers)) {
            return;
        }
        logger.debug('Compilers:', compilers);
        if (compilers.length === 0) {
            logger.error('#### No compilers found: no compilation will be done!');
        }
        prevCompilers = compilers;
        clientOptionsHandler.setCompilers(compilers);
        routeApi.apiHandler.setCompilers(compilers);
        routeApi.apiHandler.setLanguages(languages);
        routeApi.apiHandler.setOptions(clientOptionsHandler);
    }

    onCompilerChange(initialCompilers);

    const rescanCompilerSecs = ceProps('rescanCompilerSecs', 0);
    if (rescanCompilerSecs) {
        logger.info(`Rescanning compilers every ${rescanCompilerSecs} secs`);
        setInterval(() => compilerFinder.find().then(result => onCompilerChange(result.compilers)),
            rescanCompilerSecs * 1000);
    }

    const sentrySlowRequestMs = ceProps('sentrySlowRequestMs', 0);

    if (opts.metricsPort) {
        logger.info(`Running metrics server on port ${opts.metricsPort}`);
        PromClient.collectDefaultMetrics();
        const metricsServer = express();

        metricsServer.get('/metrics', async (req, res) => {
            try {
                res.set('Content-Type', PromClient.register.contentType);
                res.end(await PromClient.register.metrics());
            } catch (ex) {
                res.status(500).end(ex);
            }
        });

        metricsServer.listen(opts.metricsPort, defArgs.hostname);
    }

    webServer
        .set('trust proxy', true)
        .set('view engine', 'pug')
        .on('error', err => logger.error('Caught error in web handler; continuing:', err))
        // sentry request handler must be the first middleware on the app
        .use(Sentry.Handlers.requestHandler({
            ip: true,
        }))
        .use(Sentry.Handlers.tracingHandler())
        // eslint-disable-next-line no-unused-vars
        .use(responseTime((req, res, time) => {
            if (sentrySlowRequestMs > 0 && time >= sentrySlowRequestMs) {
                Sentry.withScope(scope => {
                    scope.setExtra('duration_ms', time);
                    Sentry.captureMessage('SlowRequest', 'warning');
                });
            }
        }))
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
                err.status ||
                err.statusCode ||
                err.status_code ||
                (err.output && err.output.statusCode) ||
                500;
            const message = err.message || 'Internal Server Error';
            res.status(status);
            res.render('error', renderConfig({error: {code: status, message: message}}));
        });

    const sponsorConfig = loadSponsorsFromString(fs.readFileSync(configDir + '/sponsors.yaml', 'utf-8'));

    function renderConfig(extra, urlOptions) {
        const urlOptionsAllowed = [
            'readOnly', 'hideEditorToolbars', 'language',
        ];
        const filteredUrlOptions = _.mapObject(
            _.pick(urlOptions, urlOptionsAllowed),
            val => utils.toProperty(val));
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

        res.render('index', renderConfig({
            embedded: false,
            mobileViewer: isMobileViewer(req),
            config: config,
            metadata: metadata,
            storedStateId: req.params.id ? req.params.id : false,
        }, req.query));
    }

    const embeddedHandler = function (req, res) {
        staticHeaders(res);
        contentPolicyHeader(res);
        res.render('embed', renderConfig({
            embedded: true,
            mobileViewer: isMobileViewer(req),
        }, req.query));
    };
    if (isDevMode()) {
        await setupWebPackDevMiddleware(router);
    } else {
        await setupStaticMiddleware(router);
    }

    morgan.token('gdpr_ip', req => utils.anonymizeIp(req.ip));

    // Based on combined format, but: GDPR compliant IP, no timestamp & no unused fields for our usecase
    const morganFormat = isDevMode() ? 'dev' : ':gdpr_ip ":method :url" :status';

    const shortenerType = getShortenerTypeByKey(clientOptionsHandler.options.urlShortenService);
    const shortener = new shortenerType(storageHandler);

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
        .use(morgan(morganFormat, {
            stream: logger.stream,
            // Skip for non errors (2xx, 3xx)
            skip: (req, res) => res.statusCode >= 400,
        }))
        .use(morgan(morganFormat, {
            stream: logger.warnStream,
            // Skip for non user errors (4xx)
            skip: (req, res) => res.statusCode < 400 || res.statusCode >= 500,
        }))
        .use(morgan(morganFormat, {
            stream: logger.errStream,
            // Skip for non server errors (5xx)
            skip: (req, res) => res.statusCode < 500,
        }))
        .use(compression())
        .get('/', (req, res) => {
            staticHeaders(res);
            contentPolicyHeader(res);
            res.render('index', renderConfig({
                embedded: false,
                mobileViewer: isMobileViewer(req),
            }, req.query));
        })
        .get('/e', embeddedHandler)
        // legacy. not a 301 to prevent any redirect loops between old e links and embed.html
        .get('/embed.html', embeddedHandler)
        .get('/embed-ro', (req, res) => {
            staticHeaders(res);
            contentPolicyHeader(res);
            res.render('embed', renderConfig({
                embedded: true,
                readOnly: true,
                mobileViewer: isMobileViewer(req),
            }, req.query));
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
        .use(sFavicon(utils.resolvePathFromAppRoot('static', 'favicon.ico')))
        .get('/client-options.js', (req, res) => {
            staticHeaders(res);
            res.set('Content-Type', 'application/javascript');
            res.end(`window.compilerExplorerOptions = ${clientOptionsHandler.getJSON()};`);
        })
        .use('/bits/:bits(\\w+).html', (req, res) => {
            staticHeaders(res);
            contentPolicyHeader(res);
            res.render('bits/' + req.params.bits, renderConfig({
                embedded: false,
                mobileViewer: isMobileViewer(req),
            }, req.query));
        })
        .use(bodyParser.json({limit: ceProps('bodyParserLimit', maxUploadSize)}))
        .use('/source', sourceHandler.handle.bind(sourceHandler))
        .use('/g', oldGoogleUrlHandler)
        .post('/shortener', shortener.handle.bind(shortener));

    noscriptHandler.InitializeRoutes({limit: ceProps('bodyParserLimit', maxUploadSize)});
    routeApi.InitializeRoutes();

    if (!defArgs.doCache) {
        logger.info('  with disabled caching');
    }
    setupEventLoopLagLogging();
    startListening(webServer);
}

main()
    .then(() => {
    })
    .catch(err => {
        logger.error('Top-level error (shutting down):', err);
        process.exit(1);
    });
