#!/usr/bin/env node

// Copyright (c) 2012, Matt Godbolt
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

const startTime = new Date();

// Initialise options and properties. Don't load any handlers here; they
// may need an initialised properties library.
const nopt = require('nopt'),
    os = require('os'),
    props = require('./lib/properties'),
    child_process = require('child_process'),
    path = require('path'),
    process = require('process'),
    fs = require('fs-extra'),
    systemdSocket = require('systemd-socket'),
    url = require('url'),
    urljoin = require('url-join'),
    _ = require('underscore'),
    express = require('express'),
    Sentry = require('@sentry/node'),
    {logger, logToPapertrail, suppressConsoleLog} = require('./lib/logger'),
    utils = require('./lib/utils'),
    initialiseWine = require('./lib/exec').initialiseWine,
    RouteAPI = require('./lib/handlers/route-api');


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
    suppressConsoleLog: [Boolean]
});

if (opts.debug) logger.level = 'debug';

// AP: Detect if we're running under Windows Subsystem for Linux. Temporary modification
// of process.env is allowed: https://nodejs.org/api/process.html#process_process_env
if (process.platform === "linux" && child_process.execSync('uname -a').toString().indexOf('Microsoft') > -1) {
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
    const windowsTemp = child_process.execSync('cmd.exe /c echo %TEMP%').toString().replace(/\\/g, "/");
    const driveLetter = windowsTemp.substring(0, 1).toLowerCase();
    const directoryPath = windowsTemp.substring(2).trim();
    process.env.winTmp = path.join("/mnt", driveLetter, directoryPath);
}

const distPath = path.resolve(__dirname, 'out', 'dist');

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
    suppressConsoleLog: opts.suppressConsoleLog || false
};

if (opts.logHost && opts.logPort) {
    logToPapertrail(opts.logHost, opts.logPort, defArgs.env.join("."));
}

if (defArgs.suppressConsoleLog) {
    logger.info("Disabling further console logging");
    suppressConsoleLog();
}

const isDevMode = () => process.env.NODE_ENV !== "production";

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
props.initialize(defArgs.rootDir + '/config', propHierarchy);

// Now load up our libraries.
const aws = require('./lib/aws'),
    google = require('./lib/google');

// Instantiate a function to access records concerning "compiler-explorer"
// in hidden object props.properties
const ceProps = props.propsFor("compiler-explorer");

let languages = require('./lib/languages').list;

if (defArgs.wantedLanguage) {
    const filteredLangs = {};
    _.each(languages, lang => {
        if (lang.id === defArgs.wantedLanguage ||
            lang.name === defArgs.wantedLanguage ||
            (lang.alias && lang.alias.indexOf(defArgs.wantedLanguage) >= 0)) {
            filteredLangs[lang.id] = lang;
        }
    });
    languages = filteredLangs;
}

if (languages.length === 0) {
    logger.error("Trying to start Compiler Explorer without a language");
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

const csp = require('./lib/csp').policy;

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
            return resolve(Number(delta) / 1000000.0);
        }, delayMs);
    });
}

function setupEventLoopLagLogging() {
    const lagIntervalMs = ceProps('eventLoopMeasureIntervalMs', 0);
    const thresWarn = ceProps('eventLoopLagThresholdWarn', 0);
    const thresErr = ceProps('eventLoopLagThresholdErr', 0);

    async function eventLoopLagHandler() {
        const lagMs = await measureEventLoopLag(lagIntervalMs);

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
    logger.error("pug require handler not configured");
};

function setupWebPackDevMiddleware(router) {
    logger.info("  using webpack dev middleware");

    const webpackDevMiddleware = require("webpack-dev-middleware"),
        webpackConfig = require('./webpack.config.js'),
        webpackCompiler = require('webpack')(webpackConfig);

    router.use(webpackDevMiddleware(webpackCompiler, {
        publicPath: '/static',
        logger: logger
    }));

    pugRequireHandler = (path) => urljoin(httpRoot, 'static', path);
}

function setupStaticMiddleware(router) {
    const staticManifest = require(path.join(distPath, 'manifest.json'));

    if (staticUrl) {
        logger.info(`  using static files from '${staticUrl}'`);
    } else {
        const staticPath = path.join(distPath, 'static');
        logger.info(`  serving static files from '${staticPath}'`);
        router.use('/static', express.static(staticPath, {
            maxAge: staticMaxAgeSecs * 1000
        }));
    }

    pugRequireHandler = (path) => {
        if (staticManifest.hasOwnProperty(path)) {
            return urljoin(staticRoot, staticManifest[path]);
        } else {
            logger.error(`failed to locate static asset '${path}' in manifest`);
            return '';
        }
    };
}

const awsProps = props.propsFor("aws");

aws.initConfig(awsProps)
    .then(initialiseWine)
    .then(() => {
        // function to load internal binaries (i.e. lib/source/*.js)
        function loadSources() {
            const sourcesDir = "lib/sources";
            return fs.readdirSync(sourcesDir)
                .filter(file => file.match(/.*\.js$/))
                .map(file => require("./" + path.join(sourcesDir, file)));
        }

        const fileSources = loadSources();
        const ClientOptionsHandler = require('./lib/options-handler');
        const clientOptionsHandler = new ClientOptionsHandler(fileSources, compilerProps, defArgs);
        const CompilationEnvironment = require('./lib/compilation-env');
        const compilationEnvironment = new CompilationEnvironment(compilerProps, defArgs.doCache);
        const CompileHandler = require('./lib/handlers/compile').Handler;
        const compileHandler = new CompileHandler(compilationEnvironment, awsProps);
        const StorageHandler = require('./lib/storage/storage');
        const storageHandler = StorageHandler.storageFactory(storageSolution, compilerProps, awsProps, httpRoot);
        const SourceHandler = require('./lib/handlers/source').Handler;
        const sourceHandler = new SourceHandler(fileSources, staticHeaders);
        const CompilerFinder = require('./lib/compiler-finder');
        const compilerFinder = new CompilerFinder(compileHandler, compilerProps, awsProps, defArgs,
            clientOptionsHandler);
        const googleShortUrlResolver = new google.ShortLinkResolver();

        function oldGoogleUrlHandler(req, res, next) {
            const bits = req.url.split("/");
            if (bits.length !== 2 || req.method !== "GET") return next();
            const googleUrl = `https://goo.gl/${encodeURIComponent(bits[1])}`;
            googleShortUrlResolver.resolve(googleUrl)
                .then(resultObj => {
                    const parsed = url.parse(resultObj.longUrl);
                    const allowedRe = new RegExp(ceProps('allowedShortUrlHostRe'));
                    if (parsed.host.match(allowedRe) === null) {
                        logger.warn(`Denied access to short URL ${bits[1]} - linked to ${resultObj.longUrl}`);
                        return next();
                    }
                    res.writeHead(301, {
                        Location: resultObj.longUrl,
                        'Cache-Control': 'public'
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
                    let exit = () => {
                        logger.info("Inactivity timeout reached, exiting.");
                        process.exit(0);
                    };
                    let idleTimer = setTimeout(exit, timeout);
                    let reset = () => {
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
            logger.info(`  Listening on http://${defArgs.hostname || 'localhost'}:${_port}/`);
            logger.info(`  Startup duration: ${new Date() - startTime}ms`);
            logger.info("=======================================");
            server.listen(_port, defArgs.hostname);
        }

        function shouldRedactRequestData(data) {
            try {
                const parsed = JSON.parse(data);
                return !parsed['allowStoreCodeDebug'];
            } catch (e) {
                return true;
            }
        }

        compilerFinder.find()
            .then(result => {
                let compilers = result.compilers;
                let prevCompilers;
                if (defArgs.ensureNoCompilerClash) {
                    logger.warn('Ensuring no compiler ids clash');
                    if (result.foundClash) {
                        // If we are forced to have no clashes, throw an error with some explanation
                        throw new Error('Clashing compilers in the current environment found!');
                    } else {
                        logger.info('No clashing ids found, continuing normally...');
                    }
                }

                const sentryDsn = aws.getConfig('sentryDsn');
                if (sentryDsn) {
                    const sentryEnv = ceProps("sentryEnvironment");
                    Sentry.init({
                        dsn: sentryDsn,
                        release: travisBuildNumber,
                        environment: sentryEnv,
                        beforeSend(event) {
                            if (event.request
                                && event.request.data
                                && shouldRedactRequestData(event.request.data)) {
                                event.request.data = JSON.stringify({redacted: true});
                            }
                            return event;
                        }
                    });
                    logger.info(`Configured with Sentry endpoint ${sentryDsn}`);
                } else {
                    logger.info("Not configuring sentry");
                }

                const webServer = express(),
                    sFavicon = require('serve-favicon'),
                    bodyParser = require('body-parser'),
                    morgan = require('morgan'),
                    compression = require('compression'),
                    router = express.Router(),
                    healthCheck = require('./lib/handlers/health-check');

                const healthCheckFilePath = ceProps("healthCheckFilePath", false);

                const routeApi = new RouteAPI(router, compileHandler, ceProps,
                    storageHandler, renderGoldenLayout);

                function onCompilerChange(compilers) {
                    if (JSON.stringify(prevCompilers) === JSON.stringify(compilers)) {
                        return;
                    }
                    logger.debug("Compilers:", compilers);
                    if (compilers.length === 0) {
                        logger.error("#### No compilers found: no compilation will be done!");
                    }
                    prevCompilers = compilers;
                    clientOptionsHandler.setCompilers(compilers);
                    routeApi.apiHandler.setCompilers(compilers);
                    routeApi.apiHandler.setLanguages(languages);
                    routeApi.apiHandler.setOptions(clientOptionsHandler);
                }

                onCompilerChange(compilers);

                const rescanCompilerSecs = ceProps('rescanCompilerSecs', 0);
                if (rescanCompilerSecs) {
                    logger.info(`Rescanning compilers every ${rescanCompilerSecs} secs`);
                    setInterval(() => compilerFinder.find().then(result => onCompilerChange(result.compilers)),
                        rescanCompilerSecs * 1000);
                }

                webServer
                    .set('trust proxy', true)
                    .set('view engine', 'pug')
                    .on('error', err => logger.error('Caught error in web handler; continuing:', err))
                    // Handle healthchecks at the root, as they're not expected from the outside world
                    .use('/healthcheck', new healthCheck.HealthCheckHandler(healthCheckFilePath).handle)
                    .use(httpRoot, router)
                    .use((req, res, next) => {
                        next({status: 404, message: `page "${req.path}" could not be found`});
                    })
                    .use(Sentry.Handlers.errorHandler)
                    // eslint-disable-next-line no-unused-vars
                    .use((err, req, res) => {
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

                logger.info("=======================================");
                if (gitReleaseName) logger.info(`  git release ${gitReleaseName}`);
                if (travisBuildNumber) logger.info(`  travis build ${travisBuildNumber}`);

                function renderConfig(extra, urlOptions) {
                    const urlOptionsWhitelist = [
                        'readOnly', 'hideEditorToolbars'
                    ];
                    const filteredUrlOptions = _.mapObject(
                        _.pick(urlOptions, urlOptionsWhitelist),
                        val => utils.toProperty(val));
                    const allExtraOptions = _.extend({}, filteredUrlOptions, extra);
                    const options = _.extend({}, allExtraOptions, clientOptionsHandler.get());
                    options.optionsHash = clientOptionsHandler.getHash();
                    options.compilerExplorerOptions = JSON.stringify(allExtraOptions);
                    options.extraBodyClass = options.embedded ? 'embedded' : extraBodyClass;
                    options.httpRoot = httpRoot;
                    options.staticRoot = staticRoot;
                    options.storageSolution = storageSolution;
                    options.require = pugRequireHandler;
                    return options;
                }

                function renderGoldenLayout(config, metadata, req, res) {
                    staticHeaders(res);
                    contentPolicyHeader(res);
                    res.render('index', renderConfig({
                        embedded: false,
                        config: config,
                        metadata: metadata
                    }, req.query));
                }

                const embeddedHandler = function (req, res) {
                    staticHeaders(res);
                    contentPolicyHeader(res);
                    res.render('embed', renderConfig({embedded: true}, req.query));
                };
                if (isDevMode()) {
                    setupWebPackDevMiddleware(router);
                } else {
                    setupStaticMiddleware(router);
                }

                morgan.token('gdpr_ip', req => utils.anonymizeIp(req.ip));

                // Based on combined format, but: GDPR compliant IP, no timestamp & no unused fields for our usecase
                const morganFormat = isDevMode() ? 'dev' : ':gdpr_ip ":method :url" :status';

                const shortenerLib = require(`./lib/shortener-${clientOptionsHandler.options.urlShortenService}`);
                const shortener = shortenerLib({storageHandler});

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
                 */
                router.get('/workers/:worker', (req, res) => {
                    staticHeaders(res);
                    res.set('Content-Type', 'application/javascript');
                    res.end(`importScripts('${urljoin(staticRoot, req.params.worker)}');`);
                });

                router
                    .use(Sentry.Handlers.requestHandler())
                    .use(morgan(morganFormat, {
                        stream: logger.stream,
                        // Skip for non errors (2xx, 3xx)
                        skip: (req, res) => res.statusCode >= 400
                    }))
                    .use(morgan(morganFormat, {
                        stream: logger.warnStream,
                        // Skip for non user errors (4xx)
                        skip: (req, res) => res.statusCode < 400 || res.statusCode >= 500
                    }))
                    .use(morgan(morganFormat, {
                        stream: logger.errStream,
                        // Skip for non server errors (5xx)
                        skip: (req, res) => res.statusCode < 500
                    }))
                    .use(compression())
                    .get('/', (req, res) => {
                        staticHeaders(res);
                        contentPolicyHeader(res);
                        res.render('index', renderConfig({embedded: false}, req.query));
                    })
                    .get('/e', embeddedHandler)
                    // legacy. not a 301 to prevent any redirect loops between old e links and embed.html
                    .get('/embed.html', embeddedHandler)
                    .get('/embed-ro', (req, res) => {
                        staticHeaders(res);
                        contentPolicyHeader(res);
                        res.render('embed', renderConfig({embedded: true, readOnly: true}, req.query));
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
                    .use(sFavicon(path.resolve(__dirname, 'static', 'favicon.ico')))
                    .get('/client-options.js', (req, res) => {
                        staticHeaders(res);
                        res.set('Content-Type', 'application/javascript');
                        const options = JSON.stringify(clientOptionsHandler.get());
                        res.end(`window.compilerExplorerOptions = ${options};`);
                    })
                    .use(bodyParser.json({limit: ceProps('bodyParserLimit', maxUploadSize)}))
                    .use(bodyParser.text({limit: ceProps('bodyParserLimit', maxUploadSize), type: () => true}))
                    .use('/source', sourceHandler.handle.bind(sourceHandler))
                    .use('/g', oldGoogleUrlHandler)
                    .post('/shortener', shortener);

                routeApi.InitializeRoutes();

                if (!defArgs.doCache) {
                    logger.info("  with disabled caching");
                }
                setupEventLoopLagLogging();
                startListening(webServer);
            })
            .catch(err => {
                logger.error("Promise error (shutting down):", err);
                process.exit(1);
            });
    })
    .catch(err => {
        logger.error("AWS Init Promise error (shutting down)", err);
        process.exit(1);
    });
